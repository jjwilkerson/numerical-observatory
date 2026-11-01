from dataclasses import replace
from pathlib import Path
import json,pickle
import numpy as np
import pytest
from numerical_observatory.numerical_fixtures import write_dataset,flat_fields,manufactured_fields,nested_boxes,adm_from_metric
from numerical_observatory.numerical_metric import NumericalMetric,cubic_weights,reconstruct
from numerical_observatory.numerical_data import NumericalLookupError
from numerical_observatory.amr_geometry import hierarchy_weights
from numerical_observatory.geodesic import Integration,trace
from numerical_observatory.camera import Camera
from numerical_observatory.metric import ETA,Schwarzschild
from numerical_observatory.background import CelestialGrid
from numerical_observatory.render import render
from numerical_observatory.cli import load_configuration


@pytest.fixture(scope='module')
def grids(tmp_path_factory):
    root=tmp_path_factory.mktemp('gmds')
    box=dict(level=0,shape=[13]*3,origin=[-6]*3,spacing=[1]*3)
    write_dataset(root/'flat',[0,1,2], [box],flat_fields,stationary=True)
    write_dataset(root/'fields',[0,1,2],[box],manufactured_fields)
    write_dataset(root/'nested',[0,1,2],nested_boxes(),manufactured_fields)
    return root


def test_layout_raw_fields_and_reconstruction(grids):
    m=NumericalMetric(grids/'fields');q=np.array([.3,.25,.37,-.43])
    f,df=m.evaluate_adm(q)
    np.testing.assert_allclose(f,manufactured_fields(q[0],q[1:],0),atol=5e-16)
    g,dg=m.evaluate(q)
    np.testing.assert_allclose(m.covariant(q)@g,np.eye(4),atol=5e-16)
    assert np.count_nonzero(np.linalg.eigvalsh(m.covariant(q))<0)==1
    for a in range(4):
        dx=np.eye(4)[a]*1e-5
        np.testing.assert_allclose(dg[a],(m.inverse(q+dx)-m.inverse(q-dx))/(2e-5),atol=3e-10,rtol=0)
    native=np.memmap(grids/'fields/patch-0.bin',dtype='<f8',mode='r',shape=(13,13,13,10))
    np.testing.assert_array_equal(native[3,2,1],manufactured_fields(0,[-5,-4,-3],0))


def test_camera_frame_initial_past_nullness(grids):
    m=NumericalMetric(grids/'fields','cubic');c=Camera(position=(0,0,0),target=(1,0,0),up=(0,0,1),time=1,width=5,height=3)
    f=c.tetrad(m);np.testing.assert_allclose(f@m.covariant(c.event)@f.T,ETA,atol=2e-14)
    for _,_,p in c.rays(m):assert abs(p@m.inverse(c.event)@p)<2e-14 and (m.inverse(c.event)@p)[0]<0


def test_cubic_is_centered_hermite_not_lagrange(tmp_path):
    def sample(t,x,l):
        f=flat_fields(t,x,l);f[0]=2+.02*x[0]**3;return f
    errors=[]
    for i,h in enumerate((.5,.25,.125)):
        n=int(4/h)+1;p=write_dataset(tmp_path/str(i),[0,1],[dict(level=0,shape=[n]*3,origin=[-2]*3,spacing=[h]*3)],sample,stationary=True)
        m=NumericalMetric(p,'cubic');q=[.3,.37*h,.23*h,.41*h];f,df=m.evaluate_adm(q)
        errors.append(abs(f[0]-sample(q[0],q[1:],0)[0]))
        assert abs(df[1,0]-.06*q[1]**2)<.02*h*h
    assert errors[1]<errors[0]/7 and errors[2]<errors[1]/7
    w=cubic_weights(.37);assert w[0].sum()==pytest.approx(1) and w[1].sum()==pytest.approx(0,abs=3e-16)


def test_same_level_cell_seam_overlap_and_no_cubic_fallback(tmp_path):
    boxes=[dict(level=0,shape=[6,9,9],origin=[-3,-4,-4],spacing=[.5,1,1],centering='cell'),
           dict(level=0,shape=[7,9,9],origin=[0,-4,-4],spacing=[.5,1,1],centering='cell')]
    p=write_dataset(tmp_path/'seam',[0,1],boxes,manufactured_fields)
    for mode in ('linear','cubic'):
        m=NumericalMetric(p,mode)
        for x in (-.25,-.01,0,.01):np.testing.assert_allclose(m.evaluate_adm([.3,x,.137,.219])[0],manufactured_fields(.3,[x,.137,.219],0),atol=2e-15)
    m=NumericalMetric(p,'cubic')
    with pytest.raises(NumericalLookupError):m.inverse([.3,-3,0,0])
    nested=write_dataset(tmp_path/'nested',[0,1],nested_boxes(2),flat_fields)
    m=NumericalMetric(nested,'cubic')
    with pytest.raises(NumericalLookupError,match='refinement_intersection'):m.inverse([.3,5.9,0,0])
    NumericalMetric(nested,'linear').inverse([.3,5.9,0,0])


def test_hierarchy_weights_gradients_boundaries_and_required_pruning(grids):
    m=NumericalMetric(grids/'nested','cubic_amr_c1_hierarchy')
    for b in m.bands:
        for axis in range(3):
            for boundary in (b.outer_lower[axis],b.inner_lower[axis],b.inner_upper[axis],b.outer_upper[axis]):
                q=np.array([10.,0,0]);q[axis]=boundary
                weights=hierarchy_weights(m.bands,q)
                assert sum(w['weight'] for w in weights)==pytest.approx(1,abs=5e-16)
                np.testing.assert_allclose(sum((w['gradient'] for w in weights)),0,atol=2e-15)
                event=np.r_[.3,q];g,dg=m.evaluate(event)
                for a in range(1,4):
                    dx=np.eye(4)[a]*1e-6
                    np.testing.assert_allclose(dg[a],(m.inverse(event+dx)-m.inverse(event-dx))/(2e-6),atol=3e-9,rtol=0)
    m.evaluate([.3,10,0,0]);assert m.last_levels=={2}
    m.evaluate([.3,7.5,0,0]);assert m.last_levels=={0,1}
    m.evaluate([.3,4,0,0]);assert m.last_levels=={0}


def test_constructor_certifies_holes_provenance_ratio_and_static_layout(tmp_path):
    boxes=nested_boxes()
    p=write_dataset(tmp_path/'data',[0,1],boxes,flat_fields)
    manifest=p/'dataset.toml';text=manifest.read_text()
    manifest.write_text(text.replace('provenance = "analytic_sampling"','provenance = "unknown"'))
    with pytest.raises(NumericalLookupError,match='provenance'):NumericalMetric(p,'cubic_amr_c1_hierarchy')
    manifest.write_text(text.replace('spacing = [0.25, 0.25, 0.25]','spacing = [0.2, 0.25, 0.25]'))
    with pytest.raises(NumericalLookupError):NumericalMetric(p,'cubic_amr_c1_hierarchy')
    where=text.rfind('origin = [8.0, -2.0, -2.0]')
    assert where>=0
    manifest.write_text(text[:where]+text[where:].replace('origin = [8.0','origin = [8.25',1))
    with pytest.raises(NumericalLookupError,match='fixed ordered'):NumericalMetric(p)
    manifest.write_text(text)
    from numerical_observatory.amr_geometry import uncovered
    assert uncovered((np.array([0,0,0]),np.array([4,4,4])),[(np.array([0,0,0]),np.array([2,4,4])),(np.array([2,0,0]),np.array([4,4,4]))])==[]


def test_invalid_native_overshoot_overlap_and_cache_budget(tmp_path):
    box=dict(level=0,shape=[7]*3,origin=[-3]*3,spacing=[1]*3)
    def overshoot(t,x,l):
        f=flat_fields(t,x,l);f[0]=.01 if x[0] in (0,1) else 1;return f
    p=write_dataset(tmp_path/'overshoot',[0,1],[box],overshoot)
    m=NumericalMetric(p,'cubic')
    with pytest.raises(NumericalLookupError,match='invalid_lapse'):m.inverse([.3,.5,0,0])
    with pytest.raises(NumericalLookupError,match='budget'):NumericalMetric(p,cache_bytes=1)
    bad=write_dataset(tmp_path/'bad',[0,1],[box],flat_fields)
    with (bad/'patch-0.bin').open('r+b') as f:f.write(np.array([np.nan],dtype='<f8').tobytes())
    with pytest.raises(NumericalLookupError,match='nonfinite'):NumericalMetric(bad).inverse([.3,0,0,0])
    dup=write_dataset(tmp_path/'dup',[0,1],[box,box],flat_fields)
    f=np.memmap(dup/'patch-1.bin',dtype='<f8',mode='r+');f[::10]=2;f.flush();del f
    with pytest.raises(NumericalLookupError,match='overlap'):NumericalMetric(dup).inverse([.3,0,0,0])


def lapse_dataset(path,scale=1.):
    def sample(t,x,l):
        f=flat_fields(t,x,l);f[0]=[1,2,1.2,2.4][round(t/scale)];return f
    return write_dataset(path,np.array([0,1,2,3])*scale,[dict(level=0,shape=[41,9,9],origin=[-20,-4,-4],spacing=[1]*3)],sample)


@pytest.mark.parametrize('scale',[1.,1e-9])
def test_one_sided_temporal_knots_exact_lapse_and_null_constraint(tmp_path,scale):
    p=lapse_dataset(tmp_path/'lapse',scale);m=NumericalMetric(p,'cubic')
    q=np.array([scale,0,0,0]);assert m.interval(scale,-1)==(0,1) and m.interval(scale,1)==(1,2)
    left=m.evaluate(q,(0,1));right=m.evaluate(q,(1,2))
    np.testing.assert_array_equal(left[0],right[0]);assert left[1][0,0,0]>0 and right[1][0,0,0]<0
    for forward in (False,True):
        t=0. if forward else 3*scale;alpha=1. if forward else 2.4
        p0=np.array([-alpha if forward else alpha,1,0,0]);event=np.array([t,0,0,0])
        r,sol=trace(m,event,p0,Integration(rtol=1e-12,atol=1e-14,max_step=2*scale,max_affine=6*scale),return_solution=True)
        assert r.status=='unresolved' and r.reason=='numerical_temporal_boundary'
        assert r.max_null_error<1e-14  # unchanged reference exact-lapse criterion
        assert sol.t[-1]==pytest.approx(4.9*scale,abs=2e-12*scale)
        for knot in (1*scale,2*scale):assert np.count_nonzero(sol.y[0]==knot)==1
        assert abs(sol.y[4,-1])==pytest.approx(2.4 if forward else 1.,abs=2e-12)


def test_explicit_background_and_dataset_exits_are_not_infinity(grids):
    c=Camera(position=(0,0,0),target=(1,0,0),up=(0,0,1),time=1)
    m=NumericalMetric(grids/'flat','cubic');p=c.momentum(m,[1,0,0])
    r=trace(m,c.event,p,Integration(max_affine=5,max_step=2))
    assert r.status=='unresolved' and r.reason=='numerical_temporal_boundary' and r.direction is None
    m=NumericalMetric(grids/'flat','cubic',domain=dict(background='finite_direction_sphere',background_radius=.5))
    r=trace(m,c.event,p,Integration(max_affine=5,max_step=2))
    assert r.status=='escaped' and r.reason=='numerical_finite_background'
    np.testing.assert_allclose(r.direction,[1,0,0],atol=2e-14)
    with pytest.raises(NumericalLookupError):m.inverse([-.1,0,0,0])


def test_pickle_metadata_and_spawn_supersampling(grids,tmp_path):
    m=NumericalMetric(grids/'fields','cubic',domain=dict(background='finite_direction_sphere',background_radius=.25))
    c=Camera(position=(0,0,0),target=(1,0,0),up=(0,0,1),time=1,width=4,height=3)
    m.inverse(c.event);serialized=pickle.dumps(m);restored=pickle.loads(serialized)
    assert not restored.data._cache and len(serialized)<20000
    for workers in (1,2):
        r=render(m,c,Integration(max_affine=2,max_step=.1),tmp_path/str(workers),background=CelestialGrid(),workers=workers,samples=2)
        assert r['counts']['escaped']==48 and not r['rotational_symmetry_reuse']
        assert r['spacetime']['dataset']['manifest_sha256']==m.data.manifest_sha256
    with np.load(tmp_path/'1/rays.npz') as a,np.load(tmp_path/'2/rays.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    cfg=tmp_path/'cfg.toml';cfg.write_text('[spacetime]\ntype="numerical"\ndataset='+json.dumps(str(grids/'fields'))+'\nspatial_interpolation="cubic"\n')
    metric,camera,settings=load_configuration(cfg);assert type(metric) is NumericalMetric
    with pytest.raises(ValueError):render(metric,camera,replace(settings,integrator='carter'),tmp_path/'bad')
    with pytest.raises(ValueError):render(metric,camera,settings,tmp_path/'bad',method='quadrature')


def test_saved_cpp_golden_all_five_modes(tmp_path):
    """Portable reference results: no compiler, WSL or C++ runtime dependency."""
    root=Path(__file__).resolve().parents[1]/'examples/numerical_spacetime/validation'
    with np.load(root/'synthetic_cpp_golden.npz') as golden:
        report=json.loads((root/'cpp_parity.json').read_text())
        for levels in (2,3):write_dataset(tmp_path/str(levels),[-20,0,20],nested_boxes(levels),manufactured_fields)
        for record in report['probes']:
            mode=record['mode'];levels=2 if mode in ('cubic_amr_c1','cubic_amr_c1_box') else 3
            m=NumericalMetric(tmp_path/str(levels),mode,**record['options'])
            for event,expected in zip(golden[mode+'_events'],golden[mode+'_values']):
                if np.isnan(expected).all():
                    with pytest.raises(NumericalLookupError):m.evaluate(event)
                else:
                    g,dg=m.evaluate(event)
                    np.testing.assert_allclose(np.concatenate((g[None],dg)),expected,atol=2e-12,rtol=0)


def test_spatial_exit_and_explicit_capture_are_controlled(grids,tmp_path):
    c=Camera(position=(0,0,0),target=(1,0,0),up=(0,0,1),time=1)
    data=write_dataset(tmp_path/'wide-time',[-100,100],
        [dict(level=0,shape=[9]*3,origin=[-4]*3,spacing=[1]*3)],flat_fields,stationary=True)
    m=NumericalMetric(data,'cubic')
    r=trace(m,c.event,c.momentum(m,[1,0,0]),Integration(max_affine=10,max_step=.5))
    assert r.status=='unresolved' and r.direction is None and r.reason=='numerical_missing_stencil'
    m=NumericalMetric(grids/'flat','cubic',domain=dict(capture_spheres=[dict(center=[.4,0,0],radius=.1,description='test absorber')]))
    r,sol=trace(m,c.event,c.momentum(m,[1,0,0]),Integration(max_affine=2,max_step=.1),return_solution=True)
    assert r.status=='captured' and r.reason=='numerical_supplied_capture' and r.numerical_diagnostics['capture_index']==1
    np.testing.assert_allclose(sol.y[1:4,-1],[.3,0,0],atol=2e-14)


@pytest.mark.parametrize('model',['schwarzschild','kerr'])
def test_exact_metric_samples_derivatives_and_geodesics_converge(tmp_path,model):
    from numerical_observatory.kerr import Kerr
    exact=Schwarzschild() if model=='schwarzschild' else Kerr(spin=.5)
    errors=[]
    for h in (.5,.25):
        path=write_dataset(tmp_path/str(h),[-10,10],
            [dict(level=0,shape=[9]*3,origin=[8,-1,-1],spacing=[h]*3)],
            lambda t,x,l:adm_from_metric(exact,np.r_[t,x]),stationary=True)
        m=NumericalMetric(path,'cubic');q=np.array([0.,9.137,.173,.219]);g,dg=m.evaluate(q)
        errors.append(np.linalg.norm(g-exact.inverse(q)))
        for a in range(1,4):
            dx=np.eye(4)[a]*1e-5
            np.testing.assert_allclose(dg[a],(m.inverse(q+dx)-m.inverse(q-dx))/2e-5,rtol=0,atol=2e-10)
        c=Camera(position=tuple(q[1:]),target=(10,.173,.219),up=(0,0,1))
        r=trace(m,c.event,c.momentum(m,[1,.05,.02]),Integration(rtol=1e-10,atol=1e-12,max_affine=.2,max_step=.05))
        assert r.reason=='affine_limit' and r.max_null_error<1e-8
    assert errors[1]<errors[0]/2


def test_saved_cpp_common_state_trajectory(tmp_path):
    root=Path(__file__).resolve().parents[1]/'examples/numerical_spacetime/validation'
    path=write_dataset(tmp_path/'fields',[-20,0,20],nested_boxes(),manufactured_fields)
    m=NumericalMetric(path,'cubic_amr_c1_hierarchy')
    with np.load(root/'manufactured_trajectories.npz') as data:reference=data['cpp']
    initial=reference[0,1:]
    r,sol=trace(m,initial[:4],initial[4:],Integration(rtol=1e-11,atol=1e-13,max_affine=4,max_step=.08),return_solution=True)
    assert r.reason=='affine_limit' and r.numerical_diagnostics['selected_levels']==[0,1,2]
    np.testing.assert_allclose(sol.sol(reference[:,0]).T,reference[:,1:],rtol=0,atol=2e-8)


def test_native_converter_matches_cpp_saved_fixture_when_available(tmp_path):
    io=pytest.importorskip('openpmd_api')
    if not io.variants.get('adios2'):pytest.skip('optional binding lacks ADIOS2')
    ref=Path(__file__).resolve().parents[1]/'reference/geodesics-cpp/validation'
    if not (ref/'fixture.bp').exists():pytest.skip('copied independent reference fixture unavailable')
    from numerical_observatory.convert_carpetx import convert
    import hashlib,tomllib
    out=tmp_path/'converted';result=convert(ref/'fixture.bp',out,'pytest-reference-roundtrip')
    assert result['snapshots']==3 and result['patches']==6
    a=NumericalMetric(out);b=NumericalMetric(ref/'fixture-dataset')
    assert [p.geometry for p in a.data.snapshots[0]]==[p.geometry for p in b.data.snapshots[0]]
    for aa,bb in zip(a.data.snapshots,b.data.snapshots):
        for pa,pb in zip(aa,bb):
            assert hashlib.sha256((out/pa.file).read_bytes()).digest()==hashlib.sha256((ref/'fixture-dataset'/pb.file).read_bytes()).digest()
    for event in ([.25,2.1,2.2,2.3],[.75,1.3,1.4,1.5]):
        np.testing.assert_array_equal(a.inverse(event),b.inverse(event))
