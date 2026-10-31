"""Spawn-safe bounded CPU work scheduling; exact orbit groups stay together."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import replace
import ctypes
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import time

import numpy as np


def cpu_topology():
    """Count physical cores permitted by affinity, without a new dependency."""
    logical = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    try:
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            mask, system = ctypes.c_size_t(), ctypes.c_size_t()
            if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system)):
                raise OSError("process affinity unavailable")
            logical = mask.value.bit_count()
            length = ctypes.c_ulong()
            kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
            buffer = ctypes.create_string_buffer(length.value)
            if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(length)):
                raise OSError("CPU topology unavailable")
            raw, offset, physical = buffer.raw, 0, 0
            while offset < length.value:
                size = int.from_bytes(raw[offset+4:offset+8], "little")
                if size < 32:
                    raise OSError("invalid CPU topology record")
                groups = int.from_bytes(raw[offset+30:offset+32], "little")
                if groups != 1 or int.from_bytes(raw[offset+40:offset+42], "little") != 0:
                    raise OSError("multiple processor groups require fallback")
                core_mask = int.from_bytes(raw[offset+32:offset+40], "little")
                physical += bool(core_mask & mask.value)
                offset += size
            if physical:
                return dict(logical=logical, physical=physical, source="Windows core topology intersected with process affinity")
        elif hasattr(os, "sched_getaffinity"):
            allowed = os.sched_getaffinity(0)
            logical = len(allowed)
            cores = set()
            for cpu in allowed:
                base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                cores.add(((base/"physical_package_id").read_text().strip(),
                           (base/"core_id").read_text().strip()))
            if cores:
                return dict(logical=logical, physical=len(cores), source="Linux sysfs cores intersected with process affinity")
    except (OSError, ValueError, AttributeError):
        pass
    return dict(logical=max(1, logical), physical=None, source="fallback: half available logical CPUs, rounded up")


def resolve_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be an integer >= 0")
    if workers == 0:
        topology = cpu_topology()
        workers = topology["physical"] or max(1, (topology["logical"]+1)//2)
    if os.name == "nt" and workers > 61:
        raise ValueError("Windows multiprocessing supports at most 61 workers")
    return workers


def peak_memory_bytes():
    try:
        if os.name == "nt":
            class Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)]+[
                    (name, ctypes.c_size_t) for name in ("peak", "working", "poolpeak", "pool", "nonpeak", "non", "page", "pagepeak", "private")]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32")
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.peak
        else:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if os.sys.platform == "darwin" else 1024)
    except (OSError, AttributeError):
        pass
    return None


@contextmanager
def _worker_environment():
    # Spawned NumPy/SciPy imports must not create N independent BLAS thread pools.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _terminate(executor):
    # Python 3.11--3.13 have no public terminate_workers(). Snapshot before
    # shutdown clears _processes. All processes belong to this executor only.
    processes = list((getattr(executor, "_processes", None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def _initialize(scene, ready):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Load integration modules before declaring the worker ready.
    from . import geodesic, surface_render, symmetry, schwarzschild_orbit
    global _SCENE
    _SCENE = scene
    ready.put((os.getpid(), time.perf_counter(), peak_memory_bytes()))


def _run(task):
    """One task contains complete exact orbit groups, or a small pixel tile."""
    from .geodesic import summarize_solution
    from .surface_render import SurfacePathCache
    from .symmetry import CenteredRayCache
    from .schwarzschild_orbit import orbit
    start, cpu = time.perf_counter(), time.process_time()
    if _SCENE[0] == "quadrature":
        radius = _SCENE[1]
        values = [(index, orbit(angle, radius).direction) for index, angle in task]
        work = (len(task), 0, 0)
    elif _SCENE[0] == "disk_layers":
        from .layer_tracing import layer_task
        values, work = layer_task(_SCENE[1], task)
    else:
        _, metric, camera, integration, domain, symmetry, surface = _SCENE
        cache = (SurfacePathCache(metric, camera, integration, domain, symmetry) if surface is not None
                 else CenteredRayCache(metric, camera, integration, symmetry, domain))
        forward, right, up = camera.axes()
        half_width = np.tan(np.deg2rad(camera.fov_degrees)/2)
        values = []
        for index in task:
            row, col = divmod(int(index), camera.width)
            # Same arithmetic and operation order as Camera.rays.
            y = (1-2*(row+.5)/camera.height)*half_width*camera.height/camera.width
            x = (2*(col+.5)/camera.width-1)*half_width
            momentum = camera.momentum(metric, forward+x*right+y*up)
            if surface is None:
                result = cache.ray(row, col, momentum)
            else:
                sol = cache.solution(row, col, momentum)
                result = summarize_solution(metric, camera.event, momentum, integration, sol,
                                            domain=domain, builtin_domain=domain is None, surface=surface)
            values.append((row, col, momentum, result))
        work = (cache.integrated_rays, cache.steps, cache.evaluations)
    return values, work, dict(pid=os.getpid(), cpu=time.process_time()-cpu,
                              wall=time.perf_counter()-start, peak_memory=peak_memory_bytes())


def _tasks(camera, symmetry):
    size = camera.width*camera.height
    if symmetry:
        yy, xx = np.mgrid[:camera.height, :camera.width]
        keys = ((2*xx+1-camera.width)**2+(camera.height-2*yy-1)**2).ravel()
        order = np.argsort(keys, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(keys[order]))+1, size]
        # Four distinct orbits per task; each orbit occurs in exactly one task.
        return [order[boundaries[i]:boundaries[min(i+4, len(boundaries)-1)]]
                for i in range(0, len(boundaries)-1, 4)]
    return [np.arange(start, min(start+64, size)) for start in range(0, size, 64)]


def scheduled(scene, tasks, workers, statistics):
    """Bound in-flight messages to 2N. Exceptions/interrupts terminate the pool."""
    try:
        pickle.dumps(scene)
    except Exception as exc:
        raise ValueError("parallel scene must be pickleable; use module-level metric/domain/surface definitions or workers=1") from exc
    requested_workers = workers
    workers = min(workers, max(1, len(tasks)))
    context = mp.get_context("spawn")
    ready = context.Queue()
    executor = None
    start = time.perf_counter()
    statistics.update(workers=workers, requested_workers=requested_workers, start_method="spawn", task_count=len(tasks), worker_cpu_seconds=0.,
                      worker_task_seconds=0., worker_peak_memory_bytes={})
    try:
        with _worker_environment():
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                           initializer=_initialize, initargs=(scene, ready))
            iterator = iter(tasks)
            pending = set()
            def fill():
                while len(pending) < 2*workers:
                    task = next(iterator, None)
                    if task is None:
                        break
                    pending.add(executor.submit(_run, task))
            fill()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    values, work, measured = future.result()
                    statistics["worker_cpu_seconds"] += measured["cpu"]
                    statistics["worker_task_seconds"] += measured["wall"]
                    statistics["worker_peak_memory_bytes"][str(measured["pid"])] = measured["peak_memory"]
                    yield values, work
                fill()
            # Executor may reuse an idle process instead of launching its full
            # capacity for short tasks. Count actual children, not max_workers.
            spawned = len(executor._processes)
            executor.shutdown(wait=True)
            initialized = [ready.get(timeout=5) for _ in range(spawned)]
            statistics["workers"] = spawned
            statistics["startup_span_seconds"] = max(item[1] for item in initialized)-start
            for pid, _, memory in initialized:
                key = str(pid)
                statistics["worker_peak_memory_bytes"][key] = max(memory or 0,
                    statistics["worker_peak_memory_bytes"].get(key) or 0) or None
    except BaseException as exc:
        if executor is not None:
            _terminate(executor)
        if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
            raise
        raise RuntimeError(f"parallel rendering failed ({type(exc).__name__}): {exc}") from exc
    finally:
        ready.close()
        ready.join_thread()
        statistics["pool_wall_seconds"] = time.perf_counter()-start


class ParallelRays:
    def __init__(self, metric, camera, integration, workers, symmetry=True, domain=None, surface=None):
        from .surface_render import SurfacePathCache
        from .symmetry import CenteredRayCache
        cache = (SurfacePathCache(metric, camera, integration, domain, symmetry) if surface is not None
                 else CenteredRayCache(metric, camera, integration, symmetry, domain))
        self.enabled = cache.enabled
        self.scene = ("ode", metric, camera, integration, domain, symmetry, surface)
        self.tasks = _tasks(camera, self.enabled)
        self.workers = workers
        self.integrated_rays = self.steps = self.evaluations = 0
        self.statistics = {}

    def results(self, progress=None):
        completed = pixels = 0
        for values, (rays, steps, evaluations) in scheduled(self.scene, self.tasks, self.workers, self.statistics):
            self.integrated_rays += rays
            self.steps += steps
            self.evaluations += evaluations
            completed += 1
            pixels += len(values)
            if progress is not None and hasattr(progress, "work"):
                progress.work(completed, len(self.tasks), pixels, self.scene[2].width*self.scene[2].height)
            yield from values


def parallel_orbits(angles, radius, workers, progress=None):
    tasks = [list(chunk) for chunk in np.array_split(np.arange(len(angles)), max(1, (len(angles)+15)//16))]
    tasks = [[(int(i), float(angles[i])) for i in chunk] for chunk in tasks]
    statistics, results = {}, {}
    completed = items = 0
    for values, _ in scheduled(("quadrature", radius), tasks, workers, statistics):
        results.update(values)
        completed += 1
        items += len(values)
        if progress is not None and hasattr(progress, "work"):
            progress.work(completed, len(tasks), items, len(angles))
    return results, statistics
