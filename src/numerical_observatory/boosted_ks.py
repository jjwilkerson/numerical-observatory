"""Instantaneous Lorentz pullback of an outgoing Schwarzschild KS term.

For acceleration, Lambda is the no-acceleration-Jacobian metric ansatz, not
the exact Jacobian of a global accelerated chart. Derivatives below DO include
the acceleration dependence of this prescribed metric.
"""
import numpy as np


def lorentz_boost(velocity,acceleration=None):
    v=np.asarray(velocity);gamma=1/np.sqrt(1-np.dot(v,v))
    coefficient=gamma*gamma/(gamma+1)  # regular at zero velocity
    spatial=np.eye(3)+coefficient*np.outer(v,v)
    matrix=np.empty((4,4),dtype=np.result_type(v,float))
    matrix[0,0]=gamma;matrix[0,1:]=-gamma*v;matrix[1:,0]=-gamma*v;matrix[1:,1:]=spatial
    if acceleration is None:return matrix
    a=np.asarray(acceleration);gd=gamma**3*np.dot(v,a)
    bd=gamma*(gamma+2)/(gamma+1)**2*gd
    sd=bd*np.outer(v,v)+coefficient*(np.outer(a,v)+np.outer(v,a))
    md=np.empty_like(matrix);md[0,0]=gd
    md[0,1:]=md[1:,0]=-(gd*v+gamma*a);md[1:,1:]=sd
    return matrix,md


def rest_position(event,center,velocity):
    # Stationary rest metric has no dependence on its rest time coordinate.
    return lorentz_boost(velocity)[1:,1:]@(np.asarray(event)[1:]-center)


def contribution(event,mass,center,velocity,acceleration,derivatives=False):
    if derivatives:matrix,md=lorentz_boost(velocity,acceleration)
    else:matrix=lorentz_boost(velocity)
    q=np.asarray(event)[1:]-center;X=matrix[1:,1:]@q
    radius=np.sqrt(np.dot(X,X))  # analytic extension enables independent complex-step audits
    if abs(radius)<1e-30:raise ValueError('boosted KS active point singularity')
    n=X/radius;ell=np.r_[1.,-n];k=matrix.T@ell;f=2*mass/radius
    term=f*np.outer(k,k)
    if not derivatives:return term
    dX=np.empty((4,3),dtype=np.result_type(X,float))
    dX[0]=md[1:,1:]@q-matrix[1:,1:]@velocity
    dX[1:]=matrix[1:,1:].T
    dr=dX@n;dell=np.zeros((4,4),dtype=dX.dtype)
    dell[:,1:]=-dX@(np.eye(3)-np.outer(n,n))/radius
    dk=dell@matrix;dk[0]+=md.T@ell
    df=-f*dr/radius
    dg=df[:,None,None]*np.outer(k,k)+f*(dk[:,:,None]*k[None,None,:]+k[None,:,None]*dk[:,None,:])
    return term,dg
