# Architecture proposal and reference review (before implementation)

Reviewed arXiv:2403.13308v3 sections II.2 and Appendix A, 2103.15707v2
sections II.1-II.2, and Zenodo 10841021 public_repo.zip: SuperposedBBH.c,
MetricBuilder/SKS_NoAcc_3D_spinarb.nb, and trajectory_tools.py. Downloaded
reference code is for inspection only, not linked, executed or copied into
production. The notebook explicitly drops acceleration in its boost Jacobian;
the generated C consumes trajectories and velocities. The 2021 model uses
harmonic coordinates; the 2026 model has PN dynamics/merger not implemented here.

Proposed equations (independently implemented):
q=x-s(t), gamma=(1-v^2)^(-1/2), S=I+gamma^2/(gamma+1)*v*v^T,
X=S*q, rest outgoing ell=(1,-X/|X|), global k=Lambda(v)^T ell,
Lambda=[[gamma,-gamma*v^T],[-gamma*v,S]], f=2m/|X|.
g=eta+sum f*k*k^T. Invert FULL g; d inverse=-inverse*(d g)*inverse.
For inertial s=c+v(t-epoch), Lambda is an exact coordinate Jacobian.
For acceleration it is the instantaneous/no-acceleration-Jacobian ansatz,
not the exact derivative of a global Fermi map. Differentiate THIS metric
exactly: Xdot=Sdot*q-S*v, kdot=Lambdadot^T ell+Lambda^T elldot.
Thus acceleration is retained in metric time derivatives. No velocity floors.

Trajectory protocol returns position, velocity, acceleration at EVERY event t.
Circular: omega=sqrt(Mtotal/d^3), mass-weighted radii, +Z angular momentum,
reference phase/epoch/barycenter. Public config restricts d>=20Mtotal (explicit
model policy, not a validity theorem); reference d=30Mtotal, v=0.091287c.
Rate-scaled circular families and inertial pairs are explicit validation controls.

Capture uses Lorentz-contracted rest-radius ellipsoids |X_A|=1.99m_A. Not
binary horizons. Overlap/tied captures are unresolved. Escape stays a fixed
barycenter sphere; all active centers must remain enclosed. The stationary
special determinant formula is not reused: compute -det(full g). The spatial
block is SPD as a sum of positive outer products; -det(g)>0 then establishes
Lorentzian signature. Check the general null-vector determinant lemma separately.

Reuse generic Hamiltonian dx/dlambda and dp/dlambda, static camera and workers;
reject Carter, spherical quadrature/caching, disks and inappropriate observers.
No p_t conservation test; record energy CHANGE and helical error. For rigid
circular motion K=dt+omega*dphi about the barycenter, p.K is conserved. Validate
Lie_K g, nullness and Hamiltonian momentum equations independently.

Validation: direct Lorentz tensor and derivative transforms of Schwarzschild;
transformed capture/escape trajectories at a matched rest-radius boundary;
multiple-time zero-rate limit; all four derivatives with complex-step and
multi-step finite differences; helical identity, symmetry, convergence, capture
sensitivity, finite sky radius, serial/spawn/supersampling, old full suite.
A separate finite-difference-Christoffel Ricci diagnostic must first converge to
zero for exact inertial Schwarzschild before comparing binary model residuals.
Only after focused checks, render two 96x72 fixed-camera times and a matched
stationary/zero-rate control. Repaint grids from saved ray maps.
