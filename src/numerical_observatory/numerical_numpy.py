"""Allocation reductions only; original interpolation/certification stay in use."""
import numpy as np
from .numerical_data import COMPONENT,validate_fields,NumericalLookupError


def reconstruct(fields,df):
    validate_fields(fields)
    alpha=fields[0];beta=fields[1:4]
    try:inv=np.linalg.inv(fields[COMPONENT])
    except np.linalg.LinAlgError as exc:raise NumericalLookupError('numerical_singular_gamma',str(exc)) from exc
    A=1/(alpha*alpha);g=np.empty((4,4));g[0,0]=-A;g[0,1:]=g[1:,0]=beta*A
    bb=np.outer(beta,beta);g[1:,1:]=inv-bb*A
    dA=-2*A*df[:,0]/alpha;db=df[:,1:4];dg=np.empty((4,4,4))
    dg[:,0,0]=-dA;dg[:,0,1:]=db*A+beta*dA[:,None];dg[:,1:,0]=dg[:,0,1:]
    dg[:,1:,1:]=-inv@df[:,COMPONENT]@inv-(db[:,:,None]*beta[None,None,:]+beta[None,:,None]*db[:,None,:])*A-bb*dA[:,None,None]
    if not np.isfinite(g).all() or not np.isfinite(dg).all():raise NumericalLookupError('numerical_nonfinite_metric','reconstructed inverse/derivatives')
    return g,dg


class Geometry:
    def __init__(self,metric):
        self.patches=metric.data.snapshots[0]
        self.origins=np.array([p.origin for p in self.patches]);self.spacings=np.array([p.spacing for p in self.patches])
        self.shapes=np.array([p.shape for p in self.patches]);self.ends=self.origins+self.shapes*self.spacings
        self.levels=np.array([p.level for p in self.patches]);self.by_level={v:np.flatnonzero(self.levels==v) for v in metric.levels}

    def candidate(self,x,level):
        mask=np.all((x>=self.origins)&(x<=self.ends),axis=1)
        if level is not None:mask&=self.levels==level
        return [self.patches[i] for i in np.flatnonzero(mask)]

    def nodes(self,metric,si,level,points):
        indices=self.by_level[level]
        if len(indices)!=1:return None  # Keep original seam/overlap policy.
        i=indices[0];u=(points-self.origins[i])/self.spacings[i];idx=np.rint(u).astype(np.int64)
        if not np.all((idx>=0)&(idx<self.shapes[i])&(abs(u-idx)<1e-8)):
            return None  # Original code constructs the exact missing-stencil reason.
        return metric.data._cache[si][i][idx[:,2],idx[:,1],idx[:,0]]
