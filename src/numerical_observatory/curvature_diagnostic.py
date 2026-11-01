"""Small validation-only Ricci stencil; not a physical error estimator.

Christoffels use the provider's analytic first derivatives. Their derivatives
use a centered five-point stencil. Validate stencil convergence against exact
vacuum controls before interpreting approximate-metric residuals.
"""
import numpy as np


def christoffel(metric,event):
    g=metric.covariant(event);dg=metric.covariant_derivative(event)
    lower=dg.transpose(1,0,2)+dg.transpose(1,2,0)-dg
    return .5*np.einsum('rs,smn->rmn',np.linalg.inv(g),lower)


def ricci(metric,event,step):
    x=np.asarray(event);gamma=christoffel(metric,x)
    derivatives=[]
    for axis in np.eye(4):
        dx=step*axis
        derivatives.append((-christoffel(metric,x+2*dx)+8*christoffel(metric,x+dx)
                            -8*christoffel(metric,x-dx)+christoffel(metric,x-2*dx))/(12*step))
    dg=np.asarray(derivatives)
    tensor=(np.einsum('rrmn->mn',dg)-np.einsum('nrmr->mn',dg)
            +np.einsum('rmn,srs->mn',gamma,gamma)-np.einsum('rms,snr->mn',gamma,gamma))
    return dict(scalar=float(np.einsum('mn,mn',metric.inverse(x),tensor)),
                max_coordinate_component=float(np.max(abs(tensor))),tensor=tensor.tolist(),
                antisymmetric_residual=float(np.max(abs(tensor-tensor.T))))
