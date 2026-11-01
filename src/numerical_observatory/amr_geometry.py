"""Reference box unions, geometric branch selection and quintic AMR blending."""
from dataclasses import dataclass
import itertools
import numpy as np
from .numerical_data import require


def uncovered(query,boxes):
    pending=[(np.array(query[0],dtype=np.int64),np.array(query[1],dtype=np.int64))]
    for nlo,nhi in boxes:
        remaining=[]
        for alo,ahi in pending:
            lo=np.maximum(alo,nlo);hi=np.minimum(ahi,nhi)
            if np.any(lo>=hi):remaining.append((alo,ahi));continue
            a,b=alo.copy(),ahi.copy()
            for d in range(3):
                if a[d]<lo[d]:
                    edge=b.copy();edge[d]=lo[d];remaining.append((a.copy(),edge));a[d]=lo[d]
                if b[d]>hi[d]:
                    edge=a.copy();edge[d]=hi[d];remaining.append((edge,b.copy()));b[d]=hi[d]
        pending=remaining
        if not pending:break
    return pending


@dataclass(frozen=True)
class BoxTransition:
    coarse_level:int
    fine_level:int
    outer_lower:tuple
    inner_lower:tuple
    inner_upper:tuple
    outer_upper:tuple

    def __post_init__(self):
        bounds=np.array([self.outer_lower,self.inner_lower,self.inner_upper,self.outer_upper])
        require(bounds.shape==(4,3) and np.isfinite(bounds).all() and np.all(np.diff(bounds,axis=0)>0),'strictly ordered finite AMR box bounds required')
        require(self.coarse_level>=0 and self.fine_level>self.coarse_level,'invalid transition levels')

    def branch(self,x):
        if np.any(x<=self.outer_lower) or np.any(x>=self.outer_upper):return 0
        return 1 if np.all(x>=self.inner_lower) and np.all(x<=self.inner_upper) else 2

    def weight(self,x):
        value=np.ones(3);gradient=np.zeros(3)
        def decreasing(z,lo,hi):
            u=(z-lo)/(hi-lo);v=1-u
            return (1-u*u*u*(10+u*(-15+6*u)) if u<=.5 else v*v*v*(10+v*(-15+6*v))),-30*u*u*v*v/(hi-lo)
        for d,z in enumerate(x):
            if z<=self.outer_lower[d] or z>=self.outer_upper[d]:value[d]=0
            elif z<self.inner_lower[d]:
                w,dw=decreasing(z,self.outer_lower[d],self.inner_lower[d]);value[d]=1-w;gradient[d]=-dw
            elif z>self.inner_upper[d]:value[d],gradient[d]=decreasing(z,self.inner_upper[d],self.outer_upper[d])
        return float(np.prod(value)),np.array([gradient[0]*value[1]*value[2],value[0]*gradient[1]*value[2],value[0]*value[1]*gradient[2]])


def hierarchy_weights(bands,x):
    rows=[dict(level=bands[0].coarse_level,required=True,weight=1.,gradient=np.zeros(3))]
    for band in bands:
        branch=band.branch(x);w,dw=band.weight(x)
        for row in rows:
            old=row['weight'];row['gradient']=(1-w)*row['gradient']-dw*old;row['weight']=(1-w)*old
            if branch==1:row.update(required=False,weight=0.,gradient=np.zeros(3))
        rows.append(dict(level=band.fine_level,required=branch!=0,weight=w,gradient=dw))
    return rows


def support_strata(bands):
    cuts=[sorted(set(float(v[d]) for b in bands for v in (b.outer_lower,b.inner_lower,b.inner_upper,b.outer_upper))) for d in range(3)]
    axes=[[(a,a) if i%2==0 else (c[i//2],c[i//2+1]) for i,a in enumerate([c[j//2] for j in range(2*len(c)-1)])] for c in cuts]
    # Each open interval and singleton endpoint is a separate stratum.
    for region in itertools.product(*axes):
        lo=np.array([v[0] for v in region]);hi=np.array([v[1] for v in region]);required=[bands[0].coarse_level]
        for b in bands:
            exterior=any((lo[d]<=b.outer_lower[d] or lo[d]>=b.outer_upper[d]) if lo[d]==hi[d]
                         else (hi[d]<=b.outer_lower[d] or lo[d]>=b.outer_upper[d]) for d in range(3))
            core=np.all(lo>=b.inner_lower) and np.all(hi<=b.inner_upper)
            if not exterior:
                if core:required=[]
                required.append(b.fine_level)
        yield lo,hi,required


def certify(bands,patches,anchor_for):
    count=0
    for lo,hi,levels in support_strata(bands):
        count+=1
        for level in levels:
            p=anchor_for(level);origin=np.asarray(p.origin);h=np.asarray(p.spacing)
            a=(lo.astype(np.longdouble)-origin)/h;b=(hi.astype(np.longdouble)-origin)/h
            cells_lo=np.floor(a).astype(np.int64);cells_hi=np.where(lo==hi,cells_lo+1,np.ceil(b)).astype(np.int64)
            first=np.where(lo==hi,lo,np.nextafter(lo,hi));last=np.where(lo==hi,hi,np.nextafter(hi,lo))
            cells_lo=np.minimum(cells_lo,np.floor((first-origin)/h).astype(np.int64))
            cells_hi=np.maximum(cells_hi,np.floor((last-origin)/h).astype(np.int64)+1)
            boxes=[]
            for q in patches:
                if q.level==level:
                    start=np.rint((np.asarray(q.origin)-origin)/h).astype(np.int64);boxes.append((start,start+q.shape))
            missing=uncovered((cells_lo-1,cells_hi+2),boxes)
            require(not missing,f'hierarchy unsupported: level {level}, region {lo}..{hi}, missing stencil coordinates {origin+missing[0][0]*h if missing else ""}')
    return count
