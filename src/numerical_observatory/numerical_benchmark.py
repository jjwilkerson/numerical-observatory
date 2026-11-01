"""Deterministic numerical-metric benchmarks; explicitly separate cold/warm work."""
import argparse,cProfile,hashlib,importlib.metadata,json,os,pickle,platform,pstats,statistics,subprocess,sys,time,tracemalloc
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scipy
from .numerical_metric import NumericalMetric,MODES,reconstruct
from .numerical_data import validate_fields
from .numerical_fixtures import write_dataset,nested_boxes,manufactured_fields
from .geodesic import hamilton_rhs,Integration,trace
from .camera import Camera


def timed(action,repeats):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();action();samples.append(time.perf_counter()-start)
    return dict(seconds=samples,median_seconds=statistics.median(samples),minimum_seconds=min(samples),
                maximum_seconds=max(samples),stdev_seconds=statistics.stdev(samples) if repeats>1 else 0.)


def machine():
    from .parallel import cpu_topology
    cpu=platform.processor()
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
            cpu=winreg.QueryValueEx(k,'ProcessorNameString')[0].strip()
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    try:numba_version=importlib.metadata.version('numba')
    except importlib.metadata.PackageNotFoundError:numba_version=None
    sources={name:hashlib.sha256(Path(__file__).with_name(name+'.py').read_bytes()).hexdigest() for name in
             ('numerical_metric','numerical_numpy','numerical_kernels','numerical_acceleration','numerical_data','amr_geometry','numerical_trace','numerical_benchmark')}
    return dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numba=numba_version,
                cpu=cpu,logical_cpus=os.cpu_count(),affinity=affinity,topology=cpu_topology(),source_sha256=sources)


def memory():
    """Process RSS/peak RSS, distinct from mapped bytes and tracemalloc."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        value=Counters();value.cb=ctypes.sizeof(value)
        handle=ctypes.windll.kernel32.GetCurrentProcess;handle.restype=wintypes.HANDLE
        read=ctypes.windll.psapi.GetProcessMemoryInfo;read.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];read.restype=wintypes.BOOL
        if not read(handle(),ctypes.byref(value),value.cb):return dict(unavailable=True)
        return dict(rss_bytes=value.WorkingSetSize,lifetime_peak_rss_bytes=value.PeakWorkingSetSize)
    import resource
    scale=1 if sys.platform=='darwin' else 1024
    return dict(lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale)


def queries(count,real=False):
    rng=np.random.default_rng(20261101);rows=[]
    # Deterministic mixed interiors, both shells and arbitrary off-grid events.
    ranges=[(2,2.8),(3.05,3.45),(4.2,6.2),(6.1,6.9),(8.2,11.8)] if real else [(4,5.5),(7.05,7.95),(8.55,8.95),(9.3,10.7),(12.05,12.95),(14.1,15.8)]
    for i in range(count):
        a,b=ranges[i%len(ranges)];xyz=[rng.uniform(a,b),rng.uniform(-.35,.35),rng.uniform(-.35,.35)]
        t=4+rng.uniform(0,.49) if real else rng.uniform(.01,.99)
        if i%8==0:
            # Actual geometry endpoints and adjacent float64 values.
            boundaries=[2.5,3.,3.5,5.,6.,7.] if real else [7.,8.,8.5,9.,11.,11.5,12.,13.]
            edge=boundaries[(i//8)%len(boundaries)];side=(i//(8*len(boundaries)))%3
            xyz[0]=edge if side==1 else np.nextafter(edge,-np.inf if side==0 else np.inf)
            t=4. if real else (0. if i%16==0 else 1.)
        rows.append([t,*xyz])
    return np.array(rows)


def make_case(root,mode,backend='python',real=None):
    options={}
    if real is None:
        levels=2 if mode in ('cubic_amr_c1','cubic_amr_c1_box') else 3
        dataset=root/f'manufactured-{levels}'
        if not dataset.exists():write_dataset(dataset,[0,1],nested_boxes(levels),manufactured_fields)
        if mode in ('cubic_amr_c1','cubic_amr_c1_box'):
            auto=NumericalMetric(dataset,'cubic_amr_c1_hierarchy');b=auto.bands[0]
            options={'box_transition':asdict(b)} if mode.endswith('_box') else {'transition':dict(axis=0,coarse_level=0,fine_level=1,lower=b.inner_upper[0],upper=b.outer_upper[0],fine_on_lower=True)}
    else:dataset=Path(real)
    if backend!='python':options['backend']=backend
    start=time.perf_counter();m=NumericalMetric(dataset,mode,**options);constructor=time.perf_counter()-start
    return m,constructor


def bench_case(root,mode,backend,count,repeats,real=None):
    memory_before=memory()
    m,constructor=make_case(root,mode,backend,real)
    source=queries(count,real is not None);valid=[];invalid=[];rejects={};start=time.perf_counter()
    # First lookup includes mapping, global sample validation and optional JIT.
    first=source[0].copy();first[1:]=[10,.17,.23] if real is None else [6.5,.23,.17]
    m.evaluate(first);first_seconds=time.perf_counter()-start
    memory_after_first_lookup=memory()
    for q in source:
        try:m.evaluate(q);valid.append(q)
        except ValueError as exc:rejects[exc.code]=rejects.get(exc.code,0)+1;invalid.append(q)
    events=np.array(valid)
    def evaluate():
        for q in events:m.evaluate(q)
    metric=timed(evaluate,repeats)
    def reject():
        for q in invalid:
            try:m.evaluate(q)
            except ValueError:pass
            else:raise AssertionError('rejected benchmark query changed validity')
    rejected=timed(reject,repeats) if invalid else None
    # Each microbenchmark reports total time for the SAME number of operations.
    adm=[m.evaluate_adm(q) for q in events]
    snapshot=0 if real is None else 8
    spatial=timed(lambda:[m.spatial(snapshot,q[1:]) for q in events],repeats)
    validation=timed(lambda:[validate_fields(f) for f,df in adm],repeats)
    reconstruction=timed(lambda:[reconstruct(f,df) for f,df in adm],repeats)
    # Measure actual selected reconstruction kernel as well as the reference.
    backend_reconstruction=None;compiled_reconstruction_prototype=None
    if backend=='numpy':
        from .numerical_numpy import reconstruct as selected
        backend_reconstruction=timed(lambda:[selected(f,df) for f,df in adm],repeats)
    elif backend=='numba':
        selected=m._compiled().kernels.reconstruct
        selected(*adm[0])
        compiled_reconstruction_prototype=timed(lambda:[selected(f,df) for f,df in adm],repeats)
        from .numerical_numpy import reconstruct as production_reconstruction
        backend_reconstruction=timed(lambda:[production_reconstruction(f,df) for f,df in adm],repeats)
    # Gather actual node coordinates, not repeated inverse-cache hits.
    anchor=m.anchors[m.levels[0]]
    def construct_stencils():
        nodes=[]
        for q in events:
            h=np.asarray(anchor.spacing);lo=np.floor((q[1:]-anchor.origin)/h)
            from .numerical_metric import OFFSETS
            nodes.append(np.asarray(anchor.origin)+(lo+OFFSETS[4]-1)*h)
        return nodes
    stencil_construction=timed(construct_stencils,repeats);nodes=construct_stencils()
    m.data.activate(0,1) if real is None else m.data.activate(8,9)
    snapshot=0 if real is None else 8
    gather=timed(lambda:[m._nodes(snapshot,anchor.level,p) for p in nodes],repeats)
    backend_gather=None
    if backend=='numba':
        k=m._compiled();arrays=k.buffers(m,snapshot);levels,origins,spacing,shapes=k.args[:4]
        def gather_selected():
            for points in nodes:k.kernels.gather(arrays,levels,origins,spacing,shapes,anchor.level,points)
        gather_selected();backend_gather=timed(gather_selected,repeats)
    # Spatial queries use active snapshots, including real data's correct index.
    if real is not None:spatial=timed(lambda:[m.spatial(snapshot,q[1:]) for q in events],repeats)
    states=[np.r_[q,[1.,.2,.3,.9]] for q in events]
    rhs=timed(lambda:[hamilton_rhs(m,z) for z in states],repeats)
    tracemalloc.start();evaluate();current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    profile=cProfile.Profile();profile.runcall(evaluate)
    top=[]
    for (file,line,func),(_,calls,own,cumulative,_) in sorted(pstats.Stats(profile).stats.items(),key=lambda x:x[1][3],reverse=True)[:16]:
        top.append(dict(file=Path(file).name,line=line,function=func,calls=calls,own_seconds=own,cumulative_seconds=cumulative))
    return dict(mode=mode,backend=backend,real=real is not None,requested_events=count,valid_events=len(events),rejections=rejects,
        query_sha256=hashlib.sha256(source.tobytes()).hexdigest(),constructor_seconds=constructor,first_lookup_seconds=first_seconds,
        metric=metric,rejected=rejected,evaluations_per_second=len(events)/metric['median_seconds'],spatial=spatial,validation=validation,
        reconstruction=reconstruction,gather=gather,rhs=rhs,python_allocations=dict(current_bytes=current,peak_bytes=peak),
        backend_reconstruction=backend_reconstruction,backend_gather=backend_gather,
        reference_coordinate_stencil_construction=stencil_construction,
        compiled_reconstruction_prototype=compiled_reconstruction_prototype,
        process_memory_before=memory_before,process_memory_after=memory(),backend_fallbacks=m.backend_fallbacks,
        memory_after_first_lookup=memory_after_first_lookup,
        mapped_bytes=m.data.loaded_bytes,pickled_provider_bytes=len(pickle.dumps(m)),profile=top,dataset=m.data.identity())


def seams(root,backend,count,repeats):
    p=root/'seam'
    if not p.exists():write_dataset(p,[0,1],[dict(level=0,shape=[6,9,9],origin=[-3,-4,-4],spacing=[.5,1,1],centering='cell'),
        dict(level=0,shape=[7,9,9],origin=[0,-4,-4],spacing=[.5,1,1],centering='cell')],manufactured_fields)
    rng=np.random.default_rng(310);events=np.column_stack((rng.uniform(0,1,count),rng.uniform(-.25,.25,count),rng.uniform(-1,1,count),rng.uniform(-1,1,count)))
    rows=[]
    for mode in ('linear','cubic'):
        m=NumericalMetric(p,mode,backend=backend);m.evaluate(events[0])
        row=timed(lambda:[m.evaluate(q) for q in events],repeats)
        rows.append(dict(mode=mode,events=count,timing=row,evaluations_per_second=count/row['median_seconds']))
    return rows


def trajectories(backend,repeats,real):
    from .numerical_validation import ROOT
    from .kerr_reference import pixel_momentum
    rows=[]
    cases=[]
    if real is not None:
        c=Camera(position=(6.5,.23,.17),target=(7.5,.23,.17),up=(0,0,1),time=4)
        cases.append(('real_escape',Path(real),'cubic_amr_c1_hierarchy',c,[1,.1,.05],dict(background='finite_direction_sphere',background_radius=8.),5.))
    for name in ('schwarzschild','kerr05'):
        p=ROOT/'examples/numerical_spacetime/validation/synthetic'/f'{name}-shadow-0.5'
        if not p.exists():continue
        c=Camera(position=(12,0,0),target=(0,0,0),up=(0,0,1),time=30,width=96,height=72)
        metadata=json.loads((p.parent.parent/'shadow_boundary_convergence.json').read_text())
        radius=next(r['reference_radius_pixels'] for r in metadata if r['model']==name)
        for offset,kind in ((-.5,'capture'),(.5,'escape')):
            cases.append((name+'_'+kind,p,'cubic',c,radius+offset,dict(background='finite_direction_sphere',background_radius=14.,capture_spheres=[dict(center=[0,0,0],radius=1.99 if name=='schwarzschild' else 1.7)]),150.))
    for name,p,mode,c,direction,domain,limit in cases:
        m=NumericalMetric(p,mode,backend=backend,domain=domain)
        photon=c.momentum(m,direction) if isinstance(direction,list) else pixel_momentum(m,c,47.5+direction,35.5)
        settings=Integration(rtol=1e-10,atol=1e-12,max_step=.3 if limit==150 else .04,max_affine=limit)
        results=[]
        def action():results.append(trace(m,c.event,photon,settings))
        timing=timed(action,repeats)
        rows.append(dict(case=name,backend=backend,timing=timing,statuses=[r.status for r in results],
            reasons=[r.reason for r in results],rhs_evaluations=[r.evaluations for r in results],
            null_max=max(r.max_null_error for r in results),fallbacks=sum(r.numerical_diagnostics['backend_fallbacks'] for r in results)))
    return rows


def renders(root,backend,repeats,real):
    from .render import render
    from .background import CelestialGrid
    rows=[]
    for workers in (1,2):
        # Keep worker startup, remapping and JIT cache loading in wall time.
        c=Camera(position=(6.5,.23,.17),target=(7.5,.23,.17),up=(0,0,1),time=4,width=4,height=4,fov_degrees=40)
        m=NumericalMetric(real,'cubic_amr_c1_hierarchy',backend=backend,domain=dict(background='finite_direction_sphere',background_radius=8.))
        m.evaluate(c.event)  # Parent initialization separately measured in microbenchmark.
        reports=[]
        def action():
            reports.append(render(m,c,Integration(rtol=1e-10,atol=1e-12,max_step=.06,max_affine=6),
                                  root/f'w{workers}-run{len(reports)}',background=CelestialGrid(),workers=workers))
        timing=timed(action,repeats)
        rows.append(dict(workers=workers,backend=backend,rays=16,timing=timing,counts=[r['counts'] for r in reports],
                         rhs_evaluations=[r['rhs_evaluations'] for r in reports],parent_memory=memory(),
                         execution=[r.get('execution') for r in reports]))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('python','numpy','numba'),default='python');p.add_argument('--count',type=int,default=512);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--real',type=Path);p.add_argument('--modes',nargs='+',default=list(MODES))
    p.add_argument('--trajectories',action='store_true');p.add_argument('--renders',action='store_true');p.add_argument('--seams',action='store_true')
    p.add_argument('--suite',action='store_true',help='sequential isolated subprocesses for all backends; requires --real')
    a=p.parse_args()
    if a.count<1 or a.repeats<1:p.error('count and repeats must be positive')
    a.output.mkdir(parents=True,exist_ok=True);result=dict(machine=machine(),parameters=vars(a).copy(),cases=[])
    if a.suite:
        if a.real is None:p.error('--suite requires --real')
        for backend in ('python','numpy','numba'):
            common=[sys.executable,'-m','numerical_observatory.numerical_benchmark','--backend',backend,'--count',str(a.count),'--repeats',str(a.repeats)]
            subprocess.run(common+['--output',str(a.output/backend/'synthetic'),'--seams'],check=True)
            subprocess.run(common+['--output',str(a.output/backend/'real'),'--real',str(a.real.resolve()),'--modes','cubic_amr_c1_hierarchy','--trajectories','--renders'],check=True)
        return
    result['parameters']={k:str(v) if isinstance(v,Path) else v for k,v in result['parameters'].items()}
    for mode in a.modes:
        row=bench_case(a.output/'fixtures',mode,a.backend,a.count,a.repeats,a.real);result['cases'].append(row)
        (a.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
        print(mode,a.backend,round(row['evaluations_per_second']),'eval/s',flush=True)
    if a.seams:result['seams']=seams(a.output/'fixtures',a.backend,a.count,a.repeats)
    if a.trajectories:result['trajectories']=trajectories(a.backend,a.repeats,a.real)
    if a.renders:
        if a.real is None:p.error('--renders requires --real')
        result['renders']=renders(a.output/'renders',a.backend,a.repeats,a.real)
    (a.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
