"""Controlled spawn benchmarks. Each case runs in a fresh Python process."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib

import numpy as np

from numerical_observatory.background import make_background
from numerical_observatory.cli import load_configuration
from numerical_observatory.disk import make_disk
from numerical_observatory.parallel import cpu_topology, peak_memory_bytes, scheduled
from numerical_observatory.render import render


SCENES = {
    "small_background": dict(width=64, height=48, samples=1, disk=False, method="quadrature", background="grid"),
    "disk_moderate": dict(width=128, height=96, samples=1, disk=True, method="ode", background="starfield"),
    "disk_supersampled": dict(width=24, height=18, samples=2, disk=True, method="ode", background="grid"),
    "rotational_reuse": dict(width=48, height=36, samples=1, disk=False, method="ode", background="grid"),
}


def verify_outputs(output):
    checks = {}
    for name in SCENES:
        folder = output/name
        ray_name = "disk_rays.npz" if SCENES[name]["disk"] else "rays.npz"
        serial_report = json.loads((folder/"1/diagnostics.json").read_text(encoding="utf-8"))
        ignore = {"elapsed_seconds", "execution"}
        for count in (5, 10, 20):
            assert (folder/"1/shadow.png").read_bytes() == (folder/str(count)/"shadow.png").read_bytes()
            with np.load(folder/"1"/ray_name) as a, np.load(folder/str(count)/ray_name) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f"{name}/{count}/{key}")
                    assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
            parallel_report = json.loads((folder/str(count)/"diagnostics.json").read_text(encoding="utf-8"))
            assert {k: v for k, v in serial_report.items() if k not in ignore} == {
                k: v for k, v in parallel_report.items() if k not in ignore}
        checks[name] = "PNG bytes and every saved array byte exactly equal; all non-execution diagnostics exactly equal"
    (output/"equality.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")


def one_case(name, workers, output):
    output.mkdir(parents=True, exist_ok=True)
    if name == "overhead":
        stats = {}
        start = time.perf_counter()
        for _ in scheduled(("quadrature", 60.), [[] for _ in range(128)], workers, stats):
            pass
        result = dict(wall_seconds=time.perf_counter()-start, execution=stats,
                      description="128 empty messages; spawn/import/shutdown and minimal communication baseline")
    else:
        spec = SCENES[name]
        metric, camera, integration = load_configuration("examples/disk_edge_on.toml")
        camera = replace(camera, width=spec["width"], height=spec["height"])
        config = tomllib.loads(Path("examples/disk_edge_on.toml").read_text(encoding="utf-8"))
        background = make_background(dict(config["background"], mode=spec["background"]))
        disk = make_disk(metric, config["disk"]) if spec["disk"] else None
        start, cpu = time.perf_counter(), time.process_time()
        report = render(metric, camera, integration, output, background=background, disk=disk,
                        samples=spec["samples"], method=spec["method"], workers=workers)
        wall, parent_cpu = time.perf_counter()-start, time.process_time()-cpu
        execution = report.get("execution", {})
        cpu_seconds = parent_cpu+execution.get("worker_cpu_seconds", 0.)
        worker_memory = sum(v or 0 for v in execution.get("worker_peak_memory_bytes", {}).values())
        result = dict(wall_seconds=wall, parent_cpu_seconds=parent_cpu,
                      rendering_cpu_seconds=cpu_seconds,
                      cpu_core_equivalents=cpu_seconds/wall,
                      parent_peak_memory_bytes=peak_memory_bytes(), worker_sum_peak_memory_bytes=worker_memory,
                      scene=spec, counts=report["counts"], integrated_rays=report.get("integrated_rays"),
                      distinct_orbits=report.get("distinct_orbits"), execution=execution)
    result.update(workers=workers, case=name)
    (output/"measurement.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/parallel_benchmark"))
    parser.add_argument("--case", choices=[*SCENES, "overhead"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true", help="keep existing completed measurements")
    parser.add_argument("--repeat-selection", action="store_true", help="repeat the two fastest disk worker counts twice")
    args = parser.parse_args()
    if args.case:
        one_case(args.case, args.workers, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in [*SCENES, "overhead"]:
        for workers in (1, 5, 10, 20):
            destination = args.output/name/str(workers)
            measurement = destination/"measurement.json"
            if not (args.resume and measurement.exists()):
                started = time.perf_counter()
                subprocess.run([sys.executable, __file__, "--case", name, "--workers", str(workers),
                                "--output", str(destination)], check=True)
                print(f"Completed {name}, workers={workers}: command wall {time.perf_counter()-started:.2f}s", flush=True)
            row = json.loads(measurement.read_text(encoding="utf-8"))
            baseline = next((r["wall_seconds"] for r in results if r["case"] == name and r["workers"] == 1), row["wall_seconds"])
            row.update(speedup=baseline/row["wall_seconds"], efficiency=baseline/row["wall_seconds"]/workers)
            results.append(row)
            report = dict(platform=platform.platform(), python=platform.python_version(), cpu=cpu_topology(),
                          results=results, notes=[
                              "Separate fresh process per case; identical settings across worker counts.",
                              "Wall includes render staging, computation, assembly, compression and writing; scene loading excluded.",
                              "CPU core equivalents=sum parent+worker task CPU/wall; worker import/startup CPU excluded (lower bound).",
                              "Memory=sum per-process peak working sets is an upper bound, not simultaneous peak memory.",
                              "Empty-message baseline includes spawn/import/shutdown; it does not isolate real payload transfer costs."])
            (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    verify_outputs(args.output)
    disk_rows = [row for row in results if row["case"] == "disk_moderate" and row["workers"] > 1]
    candidates = sorted(disk_rows, key=lambda row: row["wall_seconds"])[:2]
    selection = {str(row["workers"]): [row["wall_seconds"]] for row in candidates}
    if args.repeat_selection:
        for repeat in (2, 3):
            for row in (candidates if repeat == 2 else list(reversed(candidates))):
                workers = row["workers"]
                destination = args.output/"selection_repeats"/str(repeat)/str(workers)
                measurement = destination/"measurement.json"
                if not (args.resume and measurement.exists()):
                    subprocess.run([sys.executable, __file__, "--case", "disk_moderate", "--workers", str(workers),
                                    "--output", str(destination)], check=True)
                seconds = json.loads(measurement.read_text(encoding="utf-8"))["wall_seconds"]
                selection[str(workers)].append(seconds)
                print(f"Selection repeat {repeat}, workers={workers}: {seconds:.2f}s", flush=True)
    report["disk_selection_seconds"] = selection
    report["recommended_workers"] = int(min(selection, key=lambda key: np.median(selection[key])))
    (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
