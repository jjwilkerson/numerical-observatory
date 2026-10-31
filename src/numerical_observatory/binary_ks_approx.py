"""Experimental stationary *covariant* KS superposition; NOT a binary vacuum solution.

Signature -+++, outgoing l=(1,-n). No boost, orbital motion, horizon finder,
constraint solve, or claim of satisfying Einstein's equations is included.
"""
from dataclasses import dataclass
import numpy as np
from .metric import ETA


@dataclass(frozen=True)
class BinaryKSApprox:
    masses: tuple = (0.5, 0.5)
    centers: tuple = ((-6., 0., 0.), (6., 0., 0.))

    def __post_init__(self):
        masses=np.asarray(self.masses,dtype=float)
        centers=np.asarray(self.centers,dtype=float)
        if masses.shape!=(2,) or not np.isfinite(masses).all() or np.any(masses<0) or masses.sum()<=0:
            raise ValueError('binary masses must be two finite nonnegative values, with positive sum')
        if centers.shape!=(2,3) or not np.isfinite(centers).all():
            raise ValueError('binary centers must be two finite Cartesian three-vectors')
        if np.all(masses>0) and np.linalg.norm(centers[1]-centers[0])==0:
            raise ValueError('two active binary centers must be distinct')
        object.__setattr__(self,'masses',tuple(masses))
        object.__setattr__(self,'centers',tuple(map(tuple,centers)))

    @property
    def mass(self):
        return sum(self.masses)

    @property
    def barycenter(self):
        return np.asarray(self.masses)@np.asarray(self.centers)/self.mass

    def terms(self,event):
        event=np.asarray(event,dtype=float)
        if event.shape!=(4,) or not np.isfinite(event).all():
            raise ValueError('binary metric requires a finite event (t,x,y,z)')
        for mass,center in zip(self.masses,self.centers):
            if mass==0: continue  # a zero-mass center has no singularity
            offset=event[1:]-center
            radius=np.linalg.norm(offset)
            if radius==0: raise ValueError('binary KS point singularity at an active center')
            yield radius,offset/radius,2*mass/radius

    def covariant(self,event):
        g=ETA.copy()
        for r,n,f in self.terms(event):
            ell=np.r_[1.,-n]
            g+=f*np.outer(ell,ell)
        return g

    def covariant_derivative(self,event):
        dg=np.zeros((4,4,4))
        for r,n,f in self.terms(event):
            ell=np.r_[1.,-n]
            dl=np.zeros((3,4));dl[:,1:]=-(np.eye(3)-np.outer(n,n))/r
            dg[1:]+=(-f/r*n[:,None,None]*np.outer(ell,ell)
                     +f*(dl[:,:,None]*ell[None,None,:]+ell[None,:,None]*dl[:,None,:]))
        return dg

    def inverse(self,event):
        # Inverse of the complete sum: adding individual inverse corrections
        # would omit the cross terms and is incorrect.
        return np.linalg.inv(self.covariant(event))

    def inverse_derivative(self,event):
        inverse=self.inverse(event)
        return -np.einsum('ij,ajk,kl->ail',inverse,self.covariant_derivative(event),inverse)

    def signature_margin(self,event):
        terms=list(self.terms(event))
        if len(terms)<2: return 1.
        (_,n1,f1),(_,n2,f2)=terms
        return float(1-f1*f2*(1-n1@n2)**2)

    def description(self):
        return dict(type='binary_ks_approx',experimental=True,exact_vacuum_solution=False,
                    masses=list(self.masses),centers=[list(c) for c in self.centers],
                    total_mass_parameter=self.mass,coordinates='stationary outgoing Cartesian Kerr-Schild superposition',
                    warning='Provisional absorbing spheres are not computed binary event horizons; no Einstein constraint solve.')
