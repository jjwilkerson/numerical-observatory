# Orbiting, instantaneously boosted KS approximation

**Experimental prescribed metric; not an exact binary vacuum solution or the
published PN/merger metric.** The old stationary `binary_ks_approx` remains
available unchanged. No disks, orbital decay, eccentricity, spins, merger,
radiative transfer or CarpetX dependency are included.

## Construction and references

The reference review and implementation proposal are in
[ARCHITECTURE.md](ARCHITECTURE.md). Sources reviewed before implementation:

- [Combi & Ressler, arXiv:2403.13308v3](https://arxiv.org/html/2403.13308v3), especially equations 8–11 and Appendix A.
- [Combi et al., PRD 104, 044041 (2021)](https://arxiv.org/abs/2103.15707), a harmonic-coordinate construction.
- [Zenodo 10841021, v1](https://zenodo.org/records/10841021), including `SuperposedBBH.c` and `SKS_NoAcc_3D_spinarb.nb`.

The implementation below is independently written. Reference code is not linked
or executed. Our outgoing chart and Newtonian circular trajectories differ from
the papers' coordinate/dynamics choices; no published error bound is inherited.

Signature is -+++, G=c=1. At each integration event `(t,x,y,z)`, a trajectory
provider returns each body's center s(t), velocity v(t) and acceleration a(t).
Define q=x−s(t), gamma=(1−v²)^−1/2, and

```text
b = gamma²/(gamma+1)                 (regular at v=0)
S = identity + b v v^T
X = S q
Lambda = [[gamma, -gamma v^T], [-gamma v, S]]
ell_rest = (1, -X/|X|)               (outgoing KS covector)
k = Lambda^T ell_rest
g = eta + sum_A (2 m_A/|X_A|) k_A k_A^T
```

For constant velocity this is an **exact Lorentz pullback** of Schwarzschild:
Lambda is the coordinate Jacobian. The rest-frame time can be chosen as
`T=(t−epoch)/gamma−gamma*v.q`; the rest metric is stationary, so only X enters.
Both the metric and canonical photon covector transform as tensors.

For acceleration, Lambda is the **instantaneous/no-acceleration-Jacobian
ansatz**. It is not the exact derivative of a single global accelerated/Fermi
coordinate map. The metric is thereby defined explicitly, with its flat
background retained. Crucially, differentiating this resulting metric DOES
retain acceleration: `Xdot=Sdot*q−S*v` and
`kdot=Lambdadot^T*ell+Lambda^T*elldot`. Gamma, S, Lambda and the source positions
are differentiated analytically. All four inverse derivatives use
`partial(g^-1)=−g^-1*(partial g)*g^-1`, after inversion of the complete g.
No finite-difference derivatives or velocity offsets are used in production.

## Circular trajectories and units

`CircularOrbit.state(t)` supplies position/velocity/acceleration at the actual
photon time. Its default prescription is Omega=sqrt(M_total/d³), radii
R1=d*m2/M_total and R2=d*m1/M_total, opposite phases, orbital angular momentum
along +Z. Phase is specified at a reference epoch. There is no retarded-worldline
solve or evolving separation. The metric evolves during each ray's integration;
it is never frozen to the camera event.

The public configuration restricts **d≥20M_total**, an explicit conservative
model policy, not an accuracy theorem. The reference uses **d=30M_total**:
equal-mass body speeds 0.091287c, Omega=0.006085806/M_total and period
1032.432698M_total. Superluminal velocities and invalid masses/separations are
rejected. Circular configurations require both masses positive. Inertial
single-hole and rate-scaled circular controls are available through Python for
validation; they are not alternative PN prescriptions.

As in the stationary binary configuration, top-level `mass` is M_ref. Individual
mass entries, separation, barycenter coordinates, epoch, camera lengths/time and
integration lengths/time scale with M_ref. The sum of individual masses, which
need not equal M_ref, sets Omega. These are rest-mass parameters, not a measured
ADM mass of the superposition; the coordinate barycenter is Newtonian and
mass-weighted. The existing Schwarzschild/Kerr units are
unchanged. Resolved initial/current centers, velocities, accelerations, period,
epoch, phase and trajectory model are saved in metadata.

```toml
mass = 1.0
[spacetime]
type = "binary_ks_orbit_approx"
masses = [0.5, 0.5]
separation = 30.0
phase_degrees = 0.0
epoch = 0.0
barycenter = [0.0, 0.0, 0.0]
[camera]
time = 0.0
position = [0.0, 0.0, 75.0]
observer = "static"
```

See [circular_30M.toml](circular_30M.toml) for complete camera/integration settings.
[phase45.toml](phase45.toml) changes only observation time to one eighth period.
The label refers to the bodies' phase at the camera event, not the phase of
every photon encounter. The spatial camera, sky and numerical settings match.
The [stationary control configuration](stationary_30M_control.toml) selects the
unchanged old metric at the same epoch positions, separation and camera.

## Domain, observer and diagnostics

The existing metric-based static observer must have a timelike worldline.
Unavailable ZAMO frames are rejected. Carter, Schwarzschild quadrature, spherical
ray caching, custom domains and disks cannot activate. Generic Hamiltonian
equations evolve p_t with the nonzero time derivative. Past-directed rays have
decreasing t while their affine parameter increases.

Provisional absorbers follow `|X_A(t,x)|=2*m_A*binary_capture_fraction` (default
0.995). These are Lorentz-contracted ellipsoids, exact isolated constant-velocity
cutoff transforms but **not computed binary horizons**. An ambiguous overlap
hit is unresolved. Loss of signature or a moving center leaving the finite
escape domain is also unresolved, with an explicit reason.

The stationary direction-vector determinant shortcut is not used. The spatial
block remains positive definite as a sum of positive outer products; together
with this fact, `−det(full g)>0` establishes Lorentzian signature. The existing
positive safety margin, normally 0.05, is applied to that full determinant.
An independent generalized null-vector determinant identity is tested.

Escape uses the fixed barycenter sphere and normalized outgoing Cartesian
tangent, with **no analytic tail**. The example uses 500M. Its background
mapping is finite-boundary, and converges more slowly than the stationary model;
see measured radius dependence in the report. The fixed celestial texture is
sampled geometrically; background radiative-transfer/redshift photometry is
outside this milestone.

The old status codes remain: 0 provisional capture, 1 escaped, 2 unresolved.
`capture_index` identifies center 1/2; non-escaping directions are NaN and
unresolved pixels are magenta. Per-ray null error, determinant margin, reason
code and constraint-completeness flag remain available. Early failures may only
have initial-state constraint measurements and are explicitly incomplete.

New `energy_change` measures max |p_t−p_t0|/|p_t0|, **not conservation error**.
`energy_drift` is NaN and the aggregate conservation value is null. For rigid
circular trajectories, the exact metric has helical symmetry about its barycenter:

```text
K = partial_t + Omega * partial_phi
p.K = p_t + Omega * [(x-bx)*p_y - (y-by)*p_x]
```

`helical_error` is sampled max |p.K−(p.K)0|/|p_t0|. Exceeding the unchanged
constraint threshold marks a ray unresolved. This identity applies to this
rigidly rotating ansatz, not to arbitrary future inspirals or inertial controls.
Sampled constraint checks do not certify every near-critical trajectory.

## Reproduction

PowerShell:

```powershell
.\.venv\Scripts\python -m numerical_observatory.orbit_binary_validation --render --workers 10 --output output/orbit-binary-experiment
```

Ubuntu:

```bash
.venv/bin/python -m numerical_observatory.orbit_binary_validation --render --workers 10 --output output/orbit-binary-experiment
```

This performs reference-scene audits and traces **three 96×72 scenes**: two
camera times and the stationary/zero-rate control. It takes several minutes per
scene on the development CPU. Each grid is repainted from that scene's saved
starfield rays without another integration. No high-resolution, supersampled or
animation batch is launched. Use a new output directory to preserve examples.
Omit `--render` for audits only; `--figures-only` rebuilds the atlas from saved
previews without tracing. The audit's selected rays/boundary slice are designed
for the supplied reference configuration; adapt them for other scenes.

For ordinary configurable rendering:

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/binary_ks_orbit_approx/circular_30M.toml --workers 10 --output output/orbit-phase0
.\.venv\Scripts\python -m numerical_observatory --config examples/binary_ks_orbit_approx/phase45.toml --background grid --workers 10 --output output/orbit-phase45-grid
```

On Ubuntu replace the Python path with `.venv/bin/python`. Width/height overrides
allow smaller tests. Existing workers and supersampling remain supported.
`python -m pytest -q` runs regressions; a restricted environment may need a fresh
writable `--basetemp output/pytest-orbit-check`.

See [validation report](REPORT.md), [numerical audit](previews/validation.json),
[comparison atlas](previews/comparison.png) and per-scene `rays.npz` /
`diagnostics.json`. Raw exact-control trajectories and Ricci stencils are retained.
The curvature diagnostic is separately calibrated on boosted vacuum
Schwarzschild; its nonzero binary residuals are model violations, not a rigorous
physical accuracy estimate. Future PN/imported trajectory providers must supply
consistent derivatives; actual numerical spacetime loading remains separate.
