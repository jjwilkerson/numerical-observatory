# Kerr coordinates, observers and saved data

The metric is exact Kerr, not a frame-dragging perturbation. G=c=1; a*=J/M²,
a=M a*. Spin is about global +Z, restricted to [-0.99,0.99]. Existing mass
scaling applies to all configuration lengths/times.

## Outgoing Cartesian Kerr–Schild chart

Signature (-,+,+,+), g=eta+2H l l, inverse=eta-2H l^raised l^raised:

- H=M r³/(r⁴+a² z²).
- l=(1,(-r x+a y)/(r²+a²),(-r y-a x)/(r²+a²),-z/r).
- (x²+y²)/(r²+a²)+z²/r²=1; the positive radial branch uses a
  cancellation-safe quadratic solution. r is not Euclidean Cartesian radius.
- Analytic derivatives include the implicit r dependence. The zero-spin
  branch evaluates the existing Schwarzschild metric and derivatives exactly.

This is the outgoing chart matching the original Schwarzschild implementation,
regular for past-directed capture at the past horizon. It is not a future-infall
simulation chart. Relative to Boyer–Lindquist (BL), with Delta=r²-2Mr+a²:

T=t_BL-integral(2Mr/Delta dr),
x+iy=(r-ia) exp(i[phi_BL-integral(a/Delta dr)]) sin(theta), z=r cos(theta).

The full Jacobian is tested against the standard BL metric, including
positive-a g(t,phi)<0. Production integration uses no BL horizon transformation.
The metric interface still accepts four-dimensional events, including time.

## Observer and image geometry

`radius` and `inclination_degrees` (alias `inclination_deg`) place the camera at
(sqrt(r²+a²) sin(i),0,r cos(i)). Inclination is measured from +Z, not the disk
plane. Up is (-cos(i),0,sin(i)); right is +Y, including a fixed limiting meridian
at a polar view. The existing pinhole horizontal FOV and symmetric rest-space
orthonormalization are unchanged. The actual metric tetrad can slightly change
how these Cartesian axis labels project into physical local directions.

Alternatively specify ordinary `position`, `target`, `up`; do not combine these
with the inclination camera. A custom future-directed orthonormal
`observer_frame` remains possible. Kerr defaults to `observer="zamo"`:
u proportional to d/dt+omega d/dphi, omega=2Mar/A,
A=(r²+a²)²-a² Delta sin²(theta). ZAMOs have fixed r,theta and zero covariant
axial angular momentum; they are not coordinate-static observers or KS-slice
normals. They require r>r_plus. `observer="static"` is supported where d/dt
is timelike and rejected in the ergosphere. At a=0 ZAMO=static and the old
Schwarzschild projection is recovered. Null covectors have unit local energy
and are past-directed.

## Domain and sky convention

Capture uses r=kerr_capture_fraction*r_plus, r_plus=M(1+sqrt(1-a*²)); default
fraction 0.995. Neither the ergosurface nor Euclidean radius is the capture
surface. The legacy `capture_radius` is Schwarzschild-only; its Kerr replacement
is explicitly dimensionless. `escape_radius` is an outward Kerr-r crossing.
An affine limit, solver failure or excessive sampled null error is unresolved.

The same Hamiltonian equations continue escaping rays to `kerr_source_radius`
(default 100000M), using unrestricted adaptive steps in the far field. The
normalized outgoing Cartesian tangent approximates the asymptotic background
direction with residual bending O(M b/R²). This is not the position-vector
sky direction at the finite escape sphere. Tail constraints, steps and RHS
counts are included. Source-radius convergence is measured independently of
ODE tolerance and does not alter the lensing. Schwarzschild keeps its existing
analytical orbital tail.

## Numerical data

Kerr `rays.npz` retains directions, status (0 capture, 1 escape, 2 unresolved),
and samples where applicable. Added arrays are `null_error`, `energy_drift`,
`axial_momentum_drift`, `carter_drift` and `kerr_constants`=(E,Lz,Q).
For a past covector E=-p_t is generally negative; all impact ratios use the
signed E. Lz=x p_y-y p_x. Q uses proper BL polar momentum:
p_theta²+cos²(theta)[Lz²/sin²(theta)-a² E²], with a regular Cartesian axis limit.
Cartesian p_z is not BL p_theta. Conservation drifts are absolute changes
normalized by |E0|, M|E0| and (M E0)², avoiding division by vanishing Lz/Q.
The null residual is |g^ab p_a p_b|/E0². Accepted endpoints and dense midpoints
are sampled; these are not rigorous global trajectory error bounds.
The legacy angular_momentum_drift aggregate denotes axial Lz for Kerr; the
old Schwarzschild vector diagnostic is unchanged. Metadata includes spin,
dimensional a, coordinates, observer, horizon/capture radii, and tail convention.

## Independent references and limits

The finite-camera analytical shadow uses spherical photon constants satisfying
R=R'=0, then transforms their BL covectors into the actual camera tetrad. It
does not equate asymptotic alpha/beta coordinates with finite camera pixels.
Reference curves currently require r>5M and nonpolar observers (zero spin is
handled separately); the actual renderer supports polar views. Capture/escape
bisection uses production Hamiltonian trajectories and reports its final
brackets. A separate second-order Carter solver resolves radial and polar
turning points without using metric derivatives. It is an exterior validation
tool, not an optimized renderer or horizon solver.

Sources (equations independently implemented, no external runtime dependency):

- [SpECTRE Kerr–Schild formulas](https://spectre-code.org/classgr_1_1Solutions_1_1KerrSchild.html),
  with the explicitly stated outgoing-chart conversion above.
- [Teo, Spherical Photon Orbits Around a Kerr Black Hole](https://phyweb.physics.nus.edu.sg/~phyteoe/kerr/).
- [Grenzebach, Perlick and Lämmerzahl (2014)](https://arxiv.org/abs/1403.5234),
  finite-observer shadow geometry and the importance of observer projection.

Equatorial Kerr disks are now implemented; see [the disk guide](../kerr_disks/README.md).
There is no Kerr planar orbital-angle diagnostic, physical radiative transfer,
GPU backend or Kerr-specific trajectory reuse. Schwarzschild-only optimized
paths are guarded by exact metric type. Near-critical outcomes require
convergence checks; small invariant drift alone cannot certify an arbitrarily
near-separatrix classification.

The analytical reference also rejects nonzero |a*|<1e-7, where its spherical-
orbit parameterization loses precision; the renderer supports these small spins
and tests smooth convergence to a=0. Exactly zero uses the exact Schwarzschild
critical curve. `max_affine` bounds the encounter before escape; the auxiliary
background continuation has its own finite far-field integration interval.

An optional production Carter backend is now selectable; Hamiltonian remains
the default. See [the backend guide](../kerr_carter/README.md) and
[quantitative comparisons](../kerr_carter/REPORT.md). The older independent
reference discussed above is retained for reproducibility; it is distinct from
the compact-variable production backend and the new radial/polar quadrature audit.

