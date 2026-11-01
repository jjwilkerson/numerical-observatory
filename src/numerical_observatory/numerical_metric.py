"""Faithful static-AMR GMDS evaluator; interpolate ADM, then reconstruct 4D g."""
from dataclasses import asdict
import numpy as np
from .numerical_data import GMDSDataset,NumericalLookupError,COMPONENT,validate_fields,require
from .amr_geometry import BoxTransition,certify,hierarchy_weights

MODES=('linear','cubic','cubic_amr_c1','cubic_amr_c1_box','cubic_amr_c1_hierarchy')
OFFSETS={2:np.array([(i,j,k) for k in range(2) for j in range(2) for i in range(2)]),
         4:np.array([(i,j,k) for k in range(4) for j in range(4) for i in range(4)])}


def cubic_weights(t):
    t2=t*t;t3=t2*t
    return np.array([[-.5*t+t2-.5*t3,1-2.5*t2+1.5*t3,.5*t+2*t2-1.5*t3,-.5*t2+.5*t3],
                     [-.5+2*t-1.5*t2,-5*t+4.5*t2,.5+4*t-4.5*t2,-t+1.5*t2]])


def reconstruct(fields,derivatives):
    validate_fields(fields)
    alpha=fields[0];beta=fields[1:4];gamma=fields[COMPONENT]
    try:inv=np.linalg.inv(gamma)
    except np.linalg.LinAlgError as exc:raise NumericalLookupError('numerical_singular_gamma',str(exc)) from exc
    A=1/(alpha*alpha);g=np.empty((4,4));g[0,0]=-A;g[0,1:]=g[1:,0]=beta*A
    g[1:,1:]=inv-np.outer(beta,beta)*A
    dg=np.empty((4,4,4))
    for d,df in enumerate(derivatives):
        dA=-2*A*df[0]/alpha;db=df[1:4]
        dg[d,0,0]=-dA;dg[d,0,1:]=dg[d,1:,0]=db*A+beta*dA
        dg[d,1:,1:]=-inv@df[COMPONENT]@inv-(np.outer(db,beta)+np.outer(beta,db))*A-np.outer(beta,beta)*dA
    if not np.isfinite(g).all() or not np.isfinite(dg).all():raise NumericalLookupError('numerical_nonfinite_metric','reconstructed inverse/derivatives')
    return g,dg


class NumericalMetric:
    def __init__(self,dataset,spatial_interpolation='linear',cache_bytes=1024**3,
                 transition=None,box_transition=None,hierarchy_override=None,domain=None):
        require(spatial_interpolation in MODES,'unknown interpolation mode')
        self.data=GMDSDataset(dataset,cache_bytes);self.mode=spatial_interpolation
        self.domain_options=domain or {};self.transition=transition;self.bands=[];self.support_strata_count=0
        self.evaluations=0;self.stencils=0;self.selected_levels=set();self.last_levels=set();self.failures={};self._last_key=None;self._last=None
        ps=self.data.snapshots[0];self.levels=sorted(set(p.level for p in ps))
        self.anchors={level:min((p for p in ps if p.level==level),key=lambda p:(p.origin,p.shape)) for level in self.levels}
        require((transition is not None)==(self.mode=='cubic_amr_c1'),'planar transition required exclusively by planar mode')
        require((box_transition is not None)==(self.mode=='cubic_amr_c1_box'),'box transition required exclusively by box mode')
        require(not hierarchy_override or self.mode=='cubic_amr_c1_hierarchy','hierarchy overrides require hierarchy mode')
        if self.mode=='cubic_amr_c1':
            b=transition;require(set(b)=={'axis','coarse_level','fine_level','lower','upper','fine_on_lower'},'invalid planar transition keys')
            require(b['axis'] in (0,1,2) and np.isfinite([b['lower'],b['upper']]).all() and b['lower']<b['upper'],'invalid planar transition')
            require(self.levels==[b['coarse_level'],b['fine_level']],'planar mode requires exactly the declared levels')
            self._ratios()
        if self.mode=='cubic_amr_c1_box':
            self.bands=[BoxTransition(**box_transition)];self._ratios()
            require(self.levels==[self.bands[0].coarse_level,self.bands[0].fine_level],'box mode requires exactly two declared levels')
            self.support_strata_count=certify(self.bands,ps,self.anchors.__getitem__)
        if self.mode=='cubic_amr_c1_hierarchy':self._hierarchy(hierarchy_override)

    def _ratios(self):
        for a,b in zip(self.levels,self.levels[1:]):
            cp,fp=self.anchors[a],self.anchors[b]
            ratio_ok=np.all(abs(np.asarray(cp.spacing)/fp.spacing-2)<1e-12) if self.mode=='cubic_amr_c1' else np.array_equal(cp.spacing,2*np.asarray(fp.spacing))
            require(ratio_ok,'composite requires ratio two')
            u=(np.asarray(cp.origin)-fp.origin)/fp.spacing
            require(np.all(abs(u-np.rint(u))<1e-8),'coarse/fine lattices must align')

    def _hierarchy(self,overrides):
        require(len(self.levels)>=2 and np.all(np.diff(self.levels)==1),'hierarchy needs contiguous levels')
        self._ratios();native=self.data.metadata.get('native_regions',[])
        require(len(native)==len(self.levels),'hierarchy requires native_regions provenance; crops are not native regions')
        previous=None
        for level,r in zip(self.levels,native):
            require(r.get('level')==level and r.get('provenance') in ('analytic_sampling','carpetx_valid_chunks'),'native_regions must have increasing unique levels/provenance')
            lo=np.array(r.get('lower',[]));hi=np.array(r.get('upper',[]));p=self.anchors[level]
            require(lo.shape==hi.shape==(3,) and np.isfinite([lo,hi]).all() and np.all(lo<hi),'invalid native region')
            if previous is not None:require(np.all(lo>previous[0]) and np.all(hi<previous[1]),'native regions must be strictly nested')
            for endpoint in (lo,hi):
                u=(endpoint-p.origin)/p.spacing;require(np.all(abs(u-np.rint(u))<1e-8),'native boundary not lattice aligned')
            for q in self.data.snapshots[0]:
                if q.level==level:
                    require(q.spacing==p.spacing,'hierarchy needs identical same-level spacing')
                    start=np.rint((np.asarray(q.origin)-p.origin)/p.spacing)
                    require(np.all(start>=np.rint((lo-p.origin)/p.spacing)) and np.all(start+q.shape-1<=np.rint((hi-p.origin)/p.spacing)),'stored patch outside native region')
            if previous is not None:
                h=np.asarray(p.spacing);coarse=np.asarray(self.anchors[level-1].spacing)
                self.bands.append(BoxTransition(level-1,level,tuple(lo+2*h),tuple(lo+2*h+coarse),tuple(hi-2*h-coarse),tuple(hi-2*h)))
            previous=lo,hi
        if overrides:self.bands=[BoxTransition(**b) for b in overrides]
        require(len(self.bands)==len(self.levels)-1,'one adjacent transition per pair required')
        for i,b in enumerate(self.bands):
            require((b.coarse_level,b.fine_level)==(self.levels[i],self.levels[i+1]),'transitions must follow adjacent levels')
            r=native[i+1]
            require(np.all(np.asarray(b.outer_lower)>=r['lower']) and np.all(np.asarray(b.outer_upper)<=r['upper']),'transition outside native region')
            if i:require(np.all(np.asarray(b.outer_lower)>=self.bands[i-1].outer_lower) and np.all(np.asarray(b.outer_upper)<=self.bands[i-1].outer_upper),'outer transition boxes must nest')
        self.support_strata_count=certify(self.bands,self.data.snapshots[0],self.anchors.__getitem__)

    def interval(self,time,direction=1):
        times=self.data.times
        if not np.isfinite(time) or time<times[0] or time>times[-1]:raise NumericalLookupError('numerical_temporal_boundary','time outside saved interval')
        hi=int(np.clip(np.searchsorted(times,time,side='left' if direction<0 else 'right'),1,len(times)-1))
        return hi-1,hi

    def _nodes(self,si,level,points):
        values=np.zeros((len(points),10));found=np.zeros(len(points),dtype=bool)
        for pi,p in enumerate(self.data.snapshots[si]):
            if p.level!=level:continue
            u=(points-p.origin)/p.spacing;idx=np.rint(u).astype(np.int64)
            valid=np.all((idx>=0)&(idx<np.array(p.shape))&(abs(u-idx)<1e-8),axis=1)
            if not valid.any():continue
            a=idx[valid];v=self.data._cache[si][pi][a[:,2],a[:,1],a[:,0]]
            overlap=found[valid]
            if overlap.any() and not np.all(abs(v[overlap]-values[valid][overlap])<=1e-11*(1+abs(values[valid][overlap]))):
                raise NumericalLookupError('numerical_inconsistent_overlap',f'snapshot {si}, level {level}')
            values[valid]=v;found[valid]=True
        if not found.all():raise NumericalLookupError('numerical_missing_stencil',f'snapshot {si}, level {level}, missing {points[~found][0].tolist()}; no invented nodes')
        return values

    def _interpolate(self,si,p,x,width,reject_finer=False):
        h=np.asarray(p.spacing);origin=np.asarray(p.origin);u=(x-origin)/h
        if not np.isfinite(u).all() or np.max(abs(u))>2**31-5:raise NumericalLookupError('numerical_spatial_boundary','unrepresentable stencil index')
        lo=np.floor(u).astype(np.int64)
        if self.mode!='cubic_amr_c1_hierarchy':lo=np.where(u<=np.array(p.shape)-1,np.minimum(lo,np.array(p.shape)-2),lo)
        t=u-lo;offset=OFFSETS[width]-(1 if width==4 else 0)
        points=origin+(lo+offset)*h
        if reject_finer:
            lower=origin+(lo-1)*h;upper=origin+(lo+2)*h
            for q in self.data.snapshots[si]:
                if q.level>p.level and np.all(lower<=q.upper) and np.all(upper>=q.origin):
                    raise NumericalLookupError('numerical_refinement_intersection','ordinary cubic stencil intersects finer stored coverage')
        nodes=self._nodes(si,p.level,points)
        if width==4:weights=np.array([cubic_weights(t[d]) for d in range(3)])
        else:weights=np.array([[[1-t[d],t[d]],[-1.,1.]] for d in range(3)])
        weights[:,1,:]/=h[:,None]
        corners=OFFSETS[width];coeff=np.empty((4,len(corners)))
        for derivative in range(4):
            coeff[derivative]=np.prod(np.array([weights[d,int(derivative==d+1),corners[:,d]] for d in range(3)]),axis=0)
        self.stencils+=1;self.selected_levels.add(p.level);self.last_levels.add(p.level)
        return coeff@nodes

    def _candidate(self,si,x,level=None):
        return [p for p in self.data.snapshots[si] if (level is None or p.level==level)
                and np.all(x>=p.origin) and np.all(x<=np.asarray(p.origin)+np.array(p.shape)*p.spacing)]

    def _independent(self,si,x,level):
        candidates=self._candidate(si,x,level)
        if not candidates:raise NumericalLookupError('numerical_spatial_boundary',f'required level {level}, no same-level candidate')
        p=self.anchors[level] if self.mode=='cubic_amr_c1_hierarchy' else candidates[0]
        result=self._interpolate(si,p,x,4);validate_fields(result[0],f'independent interpolant, level {level}')
        return result

    def _blend(self,coarse,fine,w,dw):
        result=(1-w)*coarse+w*fine
        result[1:]+=dw[:,None]*(fine[0]-coarse[0]);validate_fields(result[0],'AMR composite interpolant')
        return result

    def _composite(self,si,x,count):
        if count==0:return self._independent(si,x,self.bands[0].coarse_level)
        b=self.bands[count-1];branch=b.branch(x)
        if branch==0:return self._composite(si,x,count-1)
        if branch==1:return self._independent(si,x,b.fine_level)
        coarse=self._composite(si,x,count-1);fine=self._independent(si,x,b.fine_level)
        return self._blend(coarse,fine,*b.weight(x))

    def spatial(self,si,x):
        x=np.asarray(x)
        if self.bands:return self._composite(si,x,len(self.bands))
        if self.mode=='cubic_amr_c1':
            b=self.transition;z=x[b['axis']]
            if z<b['lower']:return self._independent(si,x,b['fine_level'] if b['fine_on_lower'] else b['coarse_level'])
            if z>b['upper']:return self._independent(si,x,b['coarse_level'] if b['fine_on_lower'] else b['fine_level'])
            u=(z-b['lower'])/(b['upper']-b['lower']);v=1-u
            w=1-u*u*u*(10+u*(-15+6*u)) if u<=.5 else v*v*v*(10+v*(-15+6*v))
            d=-30*u*u*v*v/(b['upper']-b['lower'])
            if not b['fine_on_lower']:w=1-w;d=-d
            dw=np.zeros(3);dw[b['axis']]=d
            return self._blend(self._independent(si,x,b['coarse_level']),self._independent(si,x,b['fine_level']),w,dw)
        candidates=sorted(self._candidate(si,x),key=lambda p:-p.level)
        if not candidates:raise NumericalLookupError('numerical_spatial_boundary','outside stored sample candidate boxes')
        if self.mode=='cubic':return self._interpolate(si,candidates[0],x,4,True)
        for p in candidates:
            try:return self._interpolate(si,p,x,2)
            except NumericalLookupError as exc:
                if exc.code!='numerical_missing_stencil':raise
        raise NumericalLookupError('numerical_missing_stencil','no complete linear AMR stencil; coarser candidates exhausted')

    def evaluate_adm(self,event,slab=None):
        x=np.asarray(event,dtype=float)
        if x.shape!=(4,) or not np.isfinite(x).all():raise NumericalLookupError('numerical_invalid_event','expected finite txyz')
        lo,hi=self.interval(x[0]) if slab is None else slab
        times=self.data.times
        if x[0]<times[lo] or x[0]>times[hi]:raise NumericalLookupError('numerical_temporal_boundary','outside selected closed slab')
        self.data.activate(lo,hi);a=self.spatial(lo,x[1:]);b=self.spatial(hi,x[1:]);dt=times[hi]-times[lo];w=(x[0]-times[lo])/dt
        f=(1-w)*a[0]+w*b[0];df=np.empty((4,10));df[0]=(b[0]-a[0])/dt;df[1:]=(1-w)*a[1:]+w*b[1:]
        validate_fields(f);return f,df

    def evaluate(self,event,slab=None):
        key=(tuple(event),slab)
        if self._last_key==key:return self._last
        self.last_levels=set()
        self.evaluations+=1
        try:result=reconstruct(*self.evaluate_adm(event,slab))
        except NumericalLookupError as exc:
            self.failures[exc.code]=self.failures.get(exc.code,0)+1;raise
        self._last_key=key;self._last=result;return result

    def inverse(self,event):return self.evaluate(event)[0]
    def inverse_derivative(self,event):return self.evaluate(event)[1]
    def covariant(self,event):return np.linalg.inv(self.inverse(event))

    def __getstate__(self):
        state=self.__dict__.copy();state['_last_key']=state['_last']=None;state['selected_levels']=set()
        state['failures']={};state['evaluations']=state['stencils']=0;return state

    def description(self):
        return dict(type='numerical',dataset=self.data.identity(),spatial_interpolation=self.mode,
                    temporal_interpolation='piecewise linear saved ADM; right-sided standalone knots, incoming closed-slab tracing',
                    derivative_method='analytic interpolation-weight and ADM inverse chain rule; AMR weight gradients included',
                    levels=self.levels,transitions=[asdict(b) for b in self.bands],support_strata_certified=self.support_strata_count,
                    background='none unless explicit finite-boundary model configured; data exits are unresolved',domain=self.domain_options)
