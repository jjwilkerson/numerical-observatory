"""Strict equivalence tests; optional compiled path never required by base install."""
from dataclasses import asdict
import json,pickle
from pathlib import Path
import numpy as np
import pytest
from numerical_observatory.numerical_fixtures import write_dataset,nested_boxes,manufactured_fields,flat_fields
from numerical_observatory.numerical_metric import NumericalMetric,MODES
from numerical_observatory.numerical_data import NumericalLookupError
from numerical_observatory.camera import Camera
from numerical_observatory.geodesic import trace,Integration
from numerical_observatory.background import CelestialGrid
from numerical_observatory.render import render


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    root=tmp_path_factory.mktemp('acceleration')
    for levels in (2,3):write_dataset(root/str(levels),[-20,0,20],nested_boxes(levels),manufactured_fields)
    return root


def enabled(backend):
    if backend=='numba':pytest.importorskip('numba')


def provider(data,mode,backend):
    enabled(backend);levels=2 if mode in ('cubic_amr_c1','cubic_amr_c1_box') else 3;options={}
    if levels==2:
        b=NumericalMetric(data/'2','cubic_amr_c1_hierarchy').bands[0]
        options={'box_transition':asdict(b)} if mode.endswith('_box') else dict(transition=dict(axis=0,coarse_level=0,fine_level=1,lower=b.inner_upper[0],upper=b.outer_upper[0],fine_on_lower=True))
    return NumericalMetric(data/str(levels),mode,backend=backend,**options)


def compare(ref,fast,events,tolerance=2e-12):
    maximum=0.
    for event in events:
        try:a=ref.evaluate(event)
        except NumericalLookupError as exc:
            with pytest.raises(NumericalLookupError) as other:fast.evaluate(event)
            assert other.value.code==exc.code
            assert ref.last_levels==fast.last_levels
        else:
            b=fast.evaluate(event)
            assert ref.last_levels==fast.last_levels
            for x,y in zip(a,b):
                maximum=max(maximum,float(np.max(abs(x-y))))
                np.testing.assert_allclose(x,y,atol=tolerance,rtol=0)
        assert ref.stencils==fast.stencils and ref.failures==fast.failures
    return maximum


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('mode',MODES)
def test_all_modes_golden_interfaces_and_temporal_events(data,backend,mode):
    ref=provider(data,mode,'python');fast=provider(data,mode,backend)
    golden=Path(__file__).resolve().parents[1]/'examples/numerical_spacetime/validation/synthetic_cpp_golden.npz'
    with np.load(golden) as z:
        events=z[mode+'_events'];compare(ref,fast,events)
        for event,value in zip(events,z[mode+'_values']):
            if not np.isnan(value).all():
                g,dg=fast.evaluate(event);np.testing.assert_allclose(np.concatenate((g[None],dg)),value,atol=2e-12,rtol=0)
    rng=np.random.default_rng(813)
    events=[np.r_[rng.uniform(-19,19),rng.uniform([3,-3,-3],[16,3,3])] for _ in range(150)]
    for t in (-20.,0.,20.):
        for side in (-np.inf,None,np.inf):events.append(np.array([t if side is None else np.nextafter(t,side),10.,.173,.219]))
    # New objects: saved-golden loop above changed counters on fast only.
    compare(provider(data,mode,'python'),provider(data,mode,backend),events)


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_seams_overlaps_and_missing_stencils(tmp_path,backend):
    enabled(backend)
    boxes=[dict(level=0,shape=[6,9,9],origin=[-3,-4,-4],spacing=[.5,1,1],centering='cell'),
           dict(level=0,shape=[7,9,9],origin=[0,-4,-4],spacing=[.5,1,1],centering='cell')]
    p=write_dataset(tmp_path/'seam',[0,1],boxes,manufactured_fields)
    for mode in ('linear','cubic'):
        compare(NumericalMetric(p,mode),NumericalMetric(p,mode,backend=backend),[[.3,x,.137,.219] for x in (-3,-.5,-.25,-.01,0,.01,.5,3)])
    p=write_dataset(tmp_path/'overlap',[0,1],[boxes[0],boxes[0]],flat_fields)
    values=np.memmap(p/'patch-1.bin',dtype='<f8',mode='r+');values[::10]=2;values.flush();del values
    compare(NumericalMetric(p,'cubic'),NumericalMetric(p,'cubic',backend=backend),[[.3,-2,0,0]])


def test_linear_missing_corner_does_not_hide_inconsistent_overlap(tmp_path):
    enabled('numba')
    coarse=dict(level=0,shape=[9]*3,origin=[-4]*3,spacing=[1]*3)
    fine=dict(level=1,shape=[3]*3,origin=[0]*3,spacing=[.5]*3)
    # Corner 1 is missing at x=1.5, while corner 2 at (1,.5,0) overlaps
    # inconsistently. The original evaluator rejects before coarser fallback.
    p=write_dataset(tmp_path/'data',[0,1],[coarse,fine,fine],flat_fields)
    value=np.memmap(p/'patch-2.bin',dtype='<f8',mode='r+');value[50]=2;value.flush();del value
    # Candidate geometry extends half a cell past the final stored fine sample.
    compare(NumericalMetric(p,'linear'),NumericalMetric(p,'linear',backend='numba'),[[.3,1.25,.25,.25]])


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('bad',['lapse','gamma','native_nan','ill_conditioned','underflow_lapse'])
def test_per_query_physical_validity_and_native_validation(tmp_path,backend,bad):
    enabled(backend)
    def sample(t,x,l):
        f=flat_fields(t,x,l)
        if bad=='ill_conditioned':f[9]=1e-16
        elif bad=='underflow_lapse':f[0]=1e-200
        else:f[0 if bad=='lapse' else 4]=.01 if x[0] in (0,1) else 1.
        return f
    p=write_dataset(tmp_path/'data',[0,1],[dict(level=0,shape=[7]*3,origin=[-3]*3,spacing=[1]*3)],sample)
    if bad=='native_nan':
        with (p/'patch-0.bin').open('r+b') as file:file.write(np.array([np.nan],dtype='<f8').tobytes())
    ref=NumericalMetric(p,'cubic');fast=NumericalMetric(p,'cubic',backend=backend)
    with np.errstate(all='ignore'):compare(ref,fast,[[.3,.5,0,0]])
    if backend=='numba' and bad not in ('native_nan','underflow_lapse'):assert fast.backend_fallbacks==1


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_real_data_queries_and_matched_affine_geodesic(backend):
    enabled(backend)
    from numerical_observatory.numerical_validation import REAL
    if not REAL.exists():pytest.skip('optional real reference data unavailable')
    ref=NumericalMetric(REAL,'cubic_amr_c1_hierarchy');fast=NumericalMetric(REAL,'cubic_amr_c1_hierarchy',backend=backend)
    events=[np.array([4.,r,.23,.17]) for r in (2.,3.,3.25,3.5,4.,5.,6.,6.5,7.,8.,10.,12.)]
    compare(ref,fast,events)
    c=Camera(position=(6.5,.23,.17),target=(7.5,.23,.17),up=(0,0,1),time=4)
    a=[]
    for m in (ref,fast):
        m.domain_options=dict(background='finite_direction_sphere',background_radius=8.)
        a.append(trace(m,c.event,c.momentum(m,[1,.1,.05]),Integration(rtol=1e-11,atol=1e-13,max_step=.04,max_affine=5),return_solution=True))
    (r,s),(q,t)=a;assert r.status==q.status=='escaped' and r.reason==q.reason
    times=np.linspace(0,min(s.t[-1],t.t[-1]),101)
    np.testing.assert_allclose(s.sol(times),t.sol(times),atol=2e-8,rtol=0)
    np.testing.assert_allclose(r.direction,q.direction,atol=2e-9,rtol=0)
    assert abs(r.max_null_error-q.max_null_error)<1e-8


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_time_knots_and_exact_lapse_constraint(tmp_path,backend):
    enabled(backend)
    def sample(t,x,l):
        f=flat_fields(t,x,l);f[0]=[1,2,1.2,2.4][round(t)];return f
    p=write_dataset(tmp_path/'lapse',[0,1,2,3],[dict(level=0,shape=[21,7,7],origin=[-10,-3,-3],spacing=[1]*3)],sample)
    m=NumericalMetric(p,'cubic',backend=backend)
    r,s=trace(m,np.array([3.,0,0,0]),np.array([2.4,1,0,0]),Integration(rtol=1e-12,atol=1e-14,max_step=2,max_affine=6),return_solution=True)
    assert r.reason=='numerical_temporal_boundary' and r.max_null_error<1e-14
    assert s.t[-1]==pytest.approx(4.9,abs=2e-12)
    for knot in (1,2):assert np.count_nonzero(s.y[0]==knot)==1


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_spawn_pickle_and_supersampling(tmp_path,backend):
    enabled(backend)
    p=write_dataset(tmp_path/'data',[-2,2],[dict(level=0,shape=[9]*3,origin=[-4]*3,spacing=[1]*3)],manufactured_fields)
    m=NumericalMetric(p,'cubic',backend=backend,domain=dict(background='finite_direction_sphere',background_radius=.2))
    m.evaluate([0,0,0,0]);blob=pickle.dumps(m);restored=pickle.loads(blob)
    assert len(blob)<20000 and not restored.data._cache and restored._accelerator is None
    c=Camera(position=(0,0,0),target=(1,0,0),up=(0,0,1),width=2,height=2)
    for workers in (1,2):render(m,c,Integration(max_affine=1,max_step=.1),tmp_path/str(workers),background=CelestialGrid(),workers=workers,samples=2)
    with np.load(tmp_path/'1/rays.npz') as a,np.load(tmp_path/'2/rays.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    assert m.description()['evaluation_backend']==backend


def test_invalid_backend(data):
    with pytest.raises(NumericalLookupError,match='backend'):NumericalMetric(data/'3',backend='gpu')


def test_actual_two_worker_compiled_batches(tmp_path):
    enabled('numba')
    from numerical_observatory.numerical_parallel_benchmark import run,summarize
    path=write_dataset(tmp_path/'data',[-2,2],[dict(level=0,shape=[9]*3,origin=[-4]*3,spacing=[1]*3)],manufactured_fields)
    m=NumericalMetric(path,'cubic',backend='numba',domain=dict(background='finite_direction_sphere',background_radius=.2))
    c=Camera(position=(0,0,0),target=(1,0,0),up=(0,0,1),width=4,height=4)
    settings=Integration(max_affine=1,max_step=.1)
    a,_=run(m,c,settings,1);b,stats=run(m,c,settings,2)
    assert stats['workers']==2 and stats['task_count']==4
    assert summarize(a)==summarize(b)


def test_configuration_and_missing_optional_dependency(data,tmp_path,monkeypatch):
    import sys
    from numerical_observatory.cli import load_configuration
    path=tmp_path/'scene.toml'
    path.write_text('[spacetime]\ntype="numerical"\ndataset='+json.dumps(str(data/'3'))+'\nbackend="numpy"\n')
    metric,_,_=load_configuration(path);assert metric.backend=='numpy'
    monkeypatch.setitem(sys.modules,'numba.typed',None)
    with pytest.raises(ValueError,match='optional'):
        NumericalMetric(data/'3',backend='numba').evaluate([0,10,.1,.2])
    assert np.isfinite(NumericalMetric(data/'3').inverse([0,10,.1,.2])).all()


@pytest.mark.parametrize('backend',['numpy','numba'])
def test_interval_and_map_identity_cache_invalidation(data,backend):
    ref=provider(data,'cubic_amr_c1_hierarchy','python');fast=provider(data,'cubic_amr_c1_hierarchy',backend)
    for knot in (-20.,0.,20.):
        for time in (np.nextafter(knot,-np.inf),knot,np.nextafter(knot,np.inf)):
            for direction in (-1,1):
                if -20<=time<=20:assert ref.interval(time,direction)==fast.interval(time,direction)
                else:
                    with pytest.raises(NumericalLookupError):fast.interval(time,direction)
    compare(ref,fast,[[1.5,7.5,.17,.23]])
    for m in (ref,fast):m.data.activate(0,1)
    compare(ref,fast,[[1.6,7.5,.17,.23],[-1.5,8.75,.17,.23],[1.7,10,.17,.23]])
    assert set(fast.data._cache)=={1,2}


@pytest.mark.parametrize('backend',['numpy','numba'])
@pytest.mark.parametrize('model',['schwarzschild','kerr05'])
def test_strong_field_capture_escape_and_radial_turning(backend,model,strong_reference_solutions):
    enabled(backend)
    root=Path(__file__).resolve().parents[1]/'examples/numerical_spacetime/validation'
    data=root/'synthetic'/f'{model}-shadow-0.5'
    if not data.exists():pytest.skip('optional generated analytical shadow cube unavailable')
    from numerical_observatory.kerr_reference import pixel_momentum
    reference=next(r['reference_radius_pixels'] for r in json.loads((root/'shadow_boundary_convergence.json').read_text()) if r['model']==model)
    domain=dict(background='finite_direction_sphere',background_radius=14.,capture_spheres=[dict(center=[0,0,0],radius=1.99 if model=='schwarzschild' else 1.7)])
    c=Camera(position=(12,0,0),target=(0,0,0),up=(0,0,1),time=30,width=96,height=72)
    ref=NumericalMetric(data,'cubic',domain=domain);fast=NumericalMetric(data,'cubic',domain=domain,backend=backend)
    for offset,status in ((-.5,'captured'),(.5,'escaped')):
        # Exactly identical canonical initial state; no different camera frames.
        p=pixel_momentum(ref,c,47.5+reference+offset,35.5)
        settings=Integration(rtol=1e-11,atol=1e-13,max_step=.3,max_affine=150)
        key=(model,offset)
        if key not in strong_reference_solutions:
            strong_reference_solutions[key]=trace(ref,c.event,p,settings,return_solution=True)
        a,s=strong_reference_solutions[key]
        b,t=trace(fast,c.event,p,settings,return_solution=True)
        assert a.status==b.status==status and a.reason==b.reason
        samples=np.linspace(0,min(s.t[-1],t.t[-1]),151)
        np.testing.assert_allclose(s.sol(samples),t.sol(samples),atol=2e-8,rtol=0)
        assert a.max_null_error<settings.null_tolerance and b.max_null_error<settings.null_tolerance
        if status=='escaped':np.testing.assert_allclose(a.direction,b.direction,atol=2e-9,rtol=0)


@pytest.fixture(scope='module')
def strong_reference_solutions():
    # The immutable reference rays are identical for both optional backends.
    # Reuse them without reducing any state/classification/null comparisons.
    return {}
