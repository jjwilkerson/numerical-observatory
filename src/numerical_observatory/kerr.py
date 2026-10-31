"""Exact outgoing Cartesian Kerr–Schild metric; spin is J/M**2 about +Z.

Signature -+++. The outgoing chart is regular for backward rays at the past
horizon. Relative to the usual ingoing KS formula reverse the radial null
congruence, retaining g(t,phi)<0 for positive angular momentum.
"""
from dataclasses import dataclass
import numpy as np
from .metric import ETA, Schwarzschild


@dataclass(frozen=True)
class Kerr:
    mass: float = 1.0
    spin: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError('mass must be finite and positive')
        if isinstance(self.spin, bool) or not np.isfinite(self.spin) or abs(self.spin) > .99:
            raise ValueError('Kerr dimensionless spin must be in [-0.99, 0.99]')

    @property
    def a(self):
        return self.mass*self.spin

    @property
    def horizon_radius(self):
        return self.mass*(1+np.sqrt(1-self.spin**2))

    def radius(self, event):
        x, y, z = np.asarray(event)[1:]
        d = x*x+y*y+z*z-self.a**2
        disc = np.hypot(d, 2*self.a*z)
        q = (d+disc)/2 if d >= 0 else 2*(self.a*z)**2/(disc-d)
        if q <= 0:
            raise ValueError('Kerr ring/interior r=0 coordinate branch is excluded')
        return np.sqrt(q)

    def geometry(self, event):
        x, y, z = np.asarray(event)[1:]
        a = self.a
        r = self.radius(event)
        den = r*r+a*a
        ell = np.array([1., (-r*x+a*y)/den, (-r*y-a*x)/den, -z/r])
        f = 2*self.mass*r**3/(r**4+a*a*z*z)
        return r, ell, f

    def covariant(self, event):
        if self.spin == 0:
            return Schwarzschild(self.mass).covariant(event)
        _, ell, f = self.geometry(event)
        return ETA+f*np.outer(ell, ell)

    def inverse(self, event):
        if self.spin == 0:
            return Schwarzschild(self.mass).inverse(event)
        _, ell, f = self.geometry(event)
        k = ell.copy(); k[0] = -1
        return ETA-f*np.outer(k, k)

    def inverse_derivative(self, event):
        if self.spin == 0:
            return Schwarzschild(self.mass).inverse_derivative(event)
        xyz = np.asarray(event)[1:]
        x, y, z = xyz
        a = self.a
        r, ell, f = self.geometry(event)
        den = r*r+a*a
        disc = r*r+a*a*z*z/(r*r)
        dr = r*xyz/disc
        dr[2] += a*a*z/(r*disc)
        dden = 2*r*dr
        numerator = np.array([-r*x+a*y, -r*y-a*x])
        dn = np.array([-x*dr+np.array([-r,a,0.]), -y*dr+np.array([-a,-r,0.])])
        dk = np.zeros((3,4))
        dk[:,1:3] = (dn/den-numerator[:,None]*dden/den**2).T
        dk[:,3] = z*dr/r**2-np.array([0.,0.,1.])/r
        df = f*(3*dr/r-(4*r**3*dr+np.array([0.,0.,2*a*a*z]))/(r**4+a*a*z*z))
        k = ell.copy(); k[0] = -1
        result = np.zeros((4,4,4))
        result[1:] = -df[:,None,None]*np.outer(k,k)-f*(dk[:,:,None]*k[None,None,:]+k[None,:,None]*dk[:,None,:])
        return result

    def zamo_velocity(self, event):
        r = self.radius(event)
        if r <= self.horizon_radius:
            raise ValueError('ZAMO camera must be outside the Kerr event horizon')
        c = np.clip(event[3]/r,-1,1)
        sigma = r*r+self.a**2*c*c
        delta = r*r-2*self.mass*r+self.a**2
        area = (r*r+self.a**2)**2-self.a**2*delta*(1-c*c)
        omega = 2*self.mass*self.a*r/area
        return np.sqrt(area/(sigma*delta))*np.array([1.,-omega*event[2],omega*event[1],0.])

    def bl_jacobian(self, event):
        """d(T,x,y,z)/d(t_BL,r,theta,phi_BL), exterior away from polar axis.

        T=t_BL-integral(2Mr/Delta dr),
        x+iy=(r-ia) exp(i[phi_BL-integral(a/Delta dr)]) sin(theta).
        No radial BL transformation is used by the production Hamiltonian RHS.
        """
        r = self.radius(event)
        x,y,z = event[1:]
        rho = np.hypot(x,y)
        c = z/r; s = rho/np.sqrt(r*r+self.a*self.a)
        delta = r*r-2*self.mass*r+self.a*self.a
        twist = -2*self.mass*self.a*r/((r*r+self.a*self.a)*delta)
        jac = np.zeros((4,4)); jac[0,0] = 1
        jac[0,1] = -2*self.mass*r/delta
        jac[1:,1] = [r*x/(r*r+self.a*self.a)-twist*y,
                       r*y/(r*r+self.a*self.a)+twist*x,c]
        if s > 1e-14:
            jac[1:,2] = [x*c/s,y*c/s,-r*s]
        else:
            # A meridian convention at the axis; invariants() uses a regular limit.
            jac[1:,2] = [np.sqrt(r*r+self.a*self.a),0.,0.]
        jac[1:,3] = [-y,x,0.]
        return jac

    def invariants(self, state):
        """Signed canonical E,Lz,Q for a past or future null covector.

        Q=p_theta^2+cos(theta)^2*(Lz^2/sin(theta)^2-a^2 E^2).
        The equivalent Cartesian expression avoids the axis's 0/0.
        """
        e,p = state[:4], state[4:]
        r = self.radius(e); a = self.a
        x,y,z = e[1:]; px,py,pz = p[1:]
        E = -p[0]; L = x*py-y*px
        rho2 = x*x+y*y
        c = z/r; s2 = rho2/(r*r+a*a)
        if s2 > 1e-20:
            ptheta = (x*px+y*py)*c/np.sqrt(s2)-r*np.sqrt(s2)*pz
            q = ptheta**2+c*c*(L*L/s2-a*a*E*E)
        else:
            q = (r*r+a*a)*(px*px+py*py)-a*a*E*E
        return np.array([E,L,q])
