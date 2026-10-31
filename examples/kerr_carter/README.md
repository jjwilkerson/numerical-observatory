# Selectable Carter Kerr backend

Hamiltonian remains the default and the general metric reference. The optional
Carter backend supports the exact `Kerr` metric, its standard capture/escape
domain, and the existing equatorial Kerr annulus. It uses the same camera rays,
opaque first-hit selection, emitter motion, intensity and display pipeline.
Schwarzschild quadrature and rotational reuse are unchanged and cannot activate
for Kerr. No new dependency is required.

```toml
[geodesics]
integrator = "carter"       # "hamiltonian" is the default
fallback = "none"          # optional "hamiltonian"
```

The CLI overrides these choices with `--integrator` and `--carter-fallback`.
Invalid backend names, non-Kerr Carter scenes and custom domains are rejected.
The disk outer edge must lie inside the configured escape sphere.

## Run a scene

Windows PowerShell, from the project directory:

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/kerr_disks/configs/spin09_prograde_i85.toml --integrator carter --workers 10 --output output/kerr-carter
.\.venv\Scripts\python -m numerical_observatory --config examples/kerr_disks/configs/spin09_prograde_i85.toml --integrator hamiltonian --workers 10 --output output/kerr-hamiltonian
```

Ubuntu:

```bash
.venv/bin/python -m numerical_observatory --config examples/kerr_disks/configs/spin09_prograde_i85.toml --integrator carter --workers 10 --output output/kerr-carter
.venv/bin/python -m numerical_observatory --config examples/kerr_disks/configs/spin09_prograde_i85.toml --integrator hamiltonian --workers 10 --output output/kerr-hamiltonian
```

These commands use the example's configured resolution. Add `--width 96 --height
72` for a small preview before an expensive render. Backend selection propagates
through the existing spawn workers and supersampling. No spherical Kerr ray
cache is used. Separate `carter_example.toml` and `hamiltonian_reference.toml`
also demonstrate file-based selection.

## Formulation and numerical checks

The formulation follows the separated null potentials of Carter, using the
conventions reviewed in [Gralla & Lupsasca, Physical Review D 101, 044032
(2020)](https://doi.org/10.1103/PhysRevD.101.044032). Production coordinates remain
the project's outgoing Cartesian Kerr–Schild coordinates; no metric convention
or observer changes.

Internally, `u=M/r`, `mu=cos(theta)` and normalized Mino time
`tau=abs(E)*M*gamma` give compact quartic potentials. Second-order equations
evolve both positions and their Mino velocities, so radial and polar turning
points require no square-root sign changes or velocity resets. The SciPy
independent variable is still affine parameter divided by M, using the exact
Mino-to-affine Jacobian; the existing affine step limit is retained. E, Lz and Q
come from the original finite-camera covector, including its past-directed sign.

Exact on-null algebraic identities avoid cancellation in the horizon-regular
Kerr–Schild time, azimuth and covector transformations. The radial/polar first
integrals and the reconstructed metric null constraint, E, Lz and Q are measured
independently along accepted steps and midpoints. Metric tangent and Hamiltonian
momentum-equation residuals are checked in the validation suite. Near polar
turning points, the coordinate error norm tightens with the polar Jacobian's
conditioning; it does not loosen them beyond SciPy's machine-precision floor.

Escaping trajectories use an exact separated inverse-radius angular integration
to the **same finite source radius** as Hamiltonian (`kerr_source_radius`,
normally 100,000M). This is not a straight-line tail or a new lensing cutoff.
Only angular variables are needed in this stationary far-field continuation;
source arrival time is not provided. Main encounter events retain physical time.

Both backends feed the same dense surface-intersection interface. Earlier plane
crossings through the hole or outside the annulus count normally. The first
valid opaque hit returns the same shared result representation and radiometry.
Dense output past that hit is inaccessible in the returned opaque solution.

## Failure policy and saved data

Default failure handling is **unresolved**, with a reason. The polar-coordinate
chart cannot natively initialize exactly on the spin axis; near-zero Killing
energy and a trajectory reaching the chart's axis limit are also explicit
failures. `--carter-fallback hamiltonian` can genuinely retrace supported failure
cases. Coplanar disk rays and affine-limit termination remain unresolved.
An uncertainty guard around a radial separatrix remains unresolved **even with
fallback enabled**: a finite Hamiltonian integration cannot certify an exactly
critical orbit. No plausible capture/escape label is substituted.

`diagnostics.json` records selected integrator, final backend counts, accepted
Carter rays, failure reasons and fallback count. Carter-selected `rays.npz`
(background) or `disk_rays.npz` (disk)
adds `backend` (0 Hamiltonian, 1 Carter), `backend_failure_code`,
`carter_radial_error`, `carter_polar_error`, and
`carter_constraints_complete`, with sorted reason codebooks in metadata. An
incomplete flag or NaN is meaningful: early failures may only measure the
initial constraint, and are not certified trajectories. Fallbacks retain the
original Carter failure reason and are excluded from Carter-validation counts.
Existing physical array names are preserved; default Hamiltonian renders do
not acquire these extra arrays.

## Reproduce the measurements

Use a **new output directory** to preserve the supplied measurements:

```powershell
.\.venv\Scripts\python -m numerical_observatory.carter_validation --rays --seventh --shadows --output output/carter-validation
.\.venv\Scripts\python -m numerical_observatory.carter_validation --images --workers 10 --output output/carter-validation
.\.venv\Scripts\python -m numerical_observatory.carter_validation --benchmark --workers 10 --output output/carter-validation
.\.venv\Scripts\python -m numerical_observatory.carter_validation --figures-only --output output/carter-validation
.\.venv\Scripts\python -m pytest -q
```

On Ubuntu replace `.\.venv\Scripts\python` with `.venv/bin/python`.
`--figures-only` reads existing measurements and never traces. Other commands
perform numerical integrations: the shadow/near-critical audits and controlled
benchmark collection take several minutes. Benchmarks include 96×72 pairs and
two repeated 32×24 pairs per scene. Do not run concurrent numerical jobs when
measuring speed.

See [REPORT.md](REPORT.md), [architecture/baseline](ARCHITECTURE.md),
[ray validation](ray_validation.json), [shadow brackets](shadow_validation.json),
[seventh-crossing investigation](seventh_crossing.json),
[image comparisons](image_comparisons.json), and [benchmarks](benchmarks.json).
Saved render subdirectories contain the numerical arrays used for comparisons.
The comparison PNGs magnify small rasters; they do not imply extra ray sampling.

The specific seventh-crossing discrepancy has an independent resolution, but
this does not certify arbitrary higher-order or near-critical trajectories.
Retain the Hamiltonian reference, inspect unresolved/fallback counts, and tighten
tolerances for precision studies. Schwarzschild orbital-angle/alignment-ring
postprocessors remain Schwarzschild-only. This change does not extend Kerr
multilayer analysis or add disk physics.
