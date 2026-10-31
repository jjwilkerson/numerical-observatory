# Experimental stationary binary KS approximation

**This is not an exact binary solution of Einstein's equations.** It is a
stationary prescribed metric for experiments with the generic ray tracer. It
does not solve the vacuum equations, initial constraints or binary dynamics.
There is no orbital motion, radiation, disk, or CarpetX dependency.

## Geometry and numerical policy

In signature -+++, outgoing Cartesian Kerr–Schild coordinates, define
`r_A=|x-c_A|`, `n_A=(x-c_A)/r_A`, `l_A=(1,-n_A)`, and

```text
g = eta + (2*m1/r1) l1 l1^T + (2*m2/r2) l2 l2^T .
```

The full covariant matrix is inverted numerically. Each covariant term is
analytically differentiated, then `partial(g^-1)=-g^-1*(partial g)*g^-1`.
Adding the isolated inverse-metric corrections would be incorrect. All time
derivatives are zero. With one zero mass the metric becomes exactly a
translated Schwarzschild metric; the inactive center has no singularity.

The spatial metric is positive definite for nonnegative masses. The full metric
has Lorentzian signature precisely when
`D=1-f1*f2*(1-n1.dot(n2))^2 > 0`, with `f_A=2*m_A/r_A` and `det(g)=-D`.
Superposition can violate this condition. The binary trace domain terminates
as **unresolved** when D reaches the configurable positive safety margin,
normally 0.05, rather than treating such a region as a horizon. Numerical
integration/inversion failures are also unresolved and retain their reason.
Sampled diagnostics are not rigorous bounds between all trajectory points.

Each active center has an absorbing sphere of radius
`2*m_A*binary_capture_fraction` (default fraction 0.995). These are
**provisional capture cutoffs, not calculated event horizons** of the combined
metric. Their union terminates rays and records which center absorbed each ray.
Apparent silhouettes depend on this prescription; this experiment does not
validate physical binary black-hole shadows. Strongly overlapping sources may
leave little usable Lorentzian domain.

The existing metric-based static camera is unchanged. Its worldline must be
timelike (`g_tt<0`); being outside both absorbing spheres alone is insufficient.
An invalid static observer is rejected. ZAMO mode is not defined here. Only the
general Hamiltonian solver is allowed. Carter, Schwarzschild orbital quadrature,
spherical rotational caching, custom domains and emitting disks are excluded.

Background sources use the normalized outgoing Cartesian tangent at a finite,
barycenter-centered escape sphere. No spherical analytic tail or empirical
lensing correction is used. Increase `escape_radius` and `max_affine` to check
remaining bending. The preview uses 1000M. The zero-mass limit agrees with
Schwarzschild at the **same finite endpoint**, and approaches the existing
Schwarzschild infinity-map as the escape radius grows.

## Configuration and units

[equal_mass_12M.toml](equal_mass_12M.toml) specifies two masses 0.5 each,
centers ±6 along X, a static camera at (0,0,45), 40° horizontal FOV, and 96×72.
Here the reference mass equals the total mass, so separation is 12M_total.

```toml
mass = 1.0                         # reference length/time scale M_ref
[spacetime]
type = "binary_ks_approx"
masses = [0.5, 0.5]                 # independent masses in M_ref
centers = [[-6, 0, 0], [6, 0, 0]]  # positions in M_ref
[geodesics]
integrator = "hamiltonian"
[integration]
binary_capture_fraction = 0.995
binary_signature_margin = 0.05
```

Both masses must be finite and nonnegative, with positive sum. They need not
sum to one. The top-level `mass` is the reference scale: camera coordinates,
center coordinates, integration lengths and times are multiplied by it, and
the two mass entries are multiplied independently by it. If the entries do not
sum to one, M_ref differs from M_total. Metadata saves resolved masses, centers
and total mass parameter. Existing Schwarzschild/Kerr units are unchanged.

## Reproduce small previews

PowerShell, from the project directory:

```powershell
.\.venv\Scripts\python -m numerical_observatory.binary_validation --render --workers 10 --output output/binary-ks-experiment
```

Ubuntu:

```bash
.venv/bin/python -m numerical_observatory.binary_validation --render --workers 10 --output output/binary-ks-experiment
```

This runs small numerical audits, traces the 96×72 starfield once, and repaints
the grid from the same saved rays. Outputs are `starfield/shadow.png`,
`starfield/rays.npz`, `starfield/diagnostics.json`, `grid.png`,
`grid_metadata.json`, `comparison.png` and `validation.json`. The comparison
atlas enlarges the rasters by nearest-neighbor; it adds no ray resolution.
Use `--config path/to/config.toml` for another scene. Omit `--render` for audits
only. Choose a new output directory to preserve supplied examples.

The ordinary CLI also works, including spawn workers and supersampling:

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/binary_ks_approx/equal_mass_12M.toml --workers 10 --output output/binary-starfield
.\.venv\Scripts\python -m numerical_observatory --config examples/binary_ks_approx/equal_mass_12M.toml --background grid --workers 10 --output output/binary-grid
```

Replace the Python path with `.venv/bin/python` on Ubuntu. Unlike the shared-ray
preview command, these two ordinary commands each integrate their rays.
No large-resolution render is launched by these reproduction commands.

## Diagnostics and interpretation

Existing status codes remain 0 captured, 1 escaped, 2 unresolved. Captured means
**provisionally absorbed** in this mode. Unresolved rays are magenta, not black.
Binary `rays.npz` adds `capture_index` (0 none, 1/2 center index),
`signature_margin`, `null_error`, `energy_drift`,
`constraint_diagnostics_complete`, and `termination_code`. Sorted termination
codebooks and per-center counts are in `diagnostics.json`. Directions of
non-escaping rays are NaN. Early failed integrations may have only initial-state
constraint values: their completeness flag is false and they are not certified
trajectories. No fictitious full angular-momentum conservation is reported.

See [validation report](REPORT.md), [architecture proposal](ARCHITECTURE.md)
and [matching previews](preview/comparison.png). Numerical convergence verifies
integration in this prescribed geometry, **not** accuracy as a solution of
Einstein's equations. No Einstein-equation residual or physical error at 12M
separation has been established.

Superposed Kerr–Schild binary initial-data work solves additional constraints;
see [Pfeiffer, Cook & Teukolsky (2002)](https://arxiv.org/abs/gr-qc/0203085).
That construction is not implemented here and does not validate this stationary
four-metric superposition.

Run the complete suite with `python -m pytest -q`. In restricted environments,
use a fresh writable path such as `--basetemp output/pytest-binary-check` for
pytest temporary files.
