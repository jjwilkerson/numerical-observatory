# Instantaneously boosted orbiting binary: validation

This experiment defines a time-dependent prescribed four-metric. It does **not**
solve Einstein's equations, reproduce PN inspiral/merger, or establish physical
binary horizon/shadow accuracy. See [construction and commands](README.md) and
the [proposal written before implementation](ARCHITECTURE.md).

## Reference review and implemented model

The two papers and Zenodo implementation were reviewed before production code
was written. The 2026 construction motivates the instantaneous boosted KS
ansatz. The 2021 work uses a different harmonic construction. The public
`SKS_NoAcc_3D_spinarb.nb` and generated `SuperposedBBH.c` were inspected, not
compiled or copied into production. Our implementation uses **outgoing** KS,
nonspinning sources and leading-order rigid circular trajectories, not their
PN/merger model. No accuracy claim from those works is transferred here.

Responsibilities are separated into `trajectories.py` (s,v,a at event time),
`boosted_ks.py` (rest spatial coordinates and covector Lorentz transformation),
`binary_ks_orbit_approx.py` (full covariant sum, inverse and analytic derivatives),
and `orbit_binary_support.py` (moving domain and audits). The original stationary
provider/domain remain unchanged. Generic Hamiltonian equations, camera,
backgrounds, CPU scheduler and prior analytical spacetime physics are retained.

For uniform motion, Lambda is an exact Jacobian and the single-hole metric is
a coordinate-transformed exact Schwarzschild solution. For acceleration it is
an instantaneous/no-acceleration-Jacobian metric prescription. Acceleration is
nevertheless retained when differentiating the resulting metric. These are
distinct operations. Tests also exercise changing speed, where gamma_dot is
nonzero, rather than only constant-speed circular acceleration.

The reference has m1=m2=0.5M, separation **30M_total**, Omega=0.0060858061945/M,
period **1032.4326977M**, and individual speeds **0.091287093c**. Public circular
configuration requires d≥20M_total. This is a documented model restriction, not
a proven validity boundary. Both masses enter the orbital equation through
M_total; M_ref remains the configuration length/time scaling parameter.

## Metric and exact controls

[Raw audit](previews/validation.json) includes near-source, intermediate and
far-field points, all four derivative axes, multiple finite-difference steps and
complex-step references. Maximum inverse-derivative disagreement with
complex-step differentiation is **4.45e-16** in this reference audit. Five-point
finite-difference checks at steps 1e-2, 1e-3 and 1e-4M independently test numerical
convergence; their raw errors are saved. Inverse identity, signature, initial
nullness and static-camera tetrad orthonormality are tested directly.

Constant-velocity controls use zero, positive and negative velocity components.
Their covariant/inverse tensors and inverse derivatives are independently
compared with Lorentz transformations of the existing exact Schwarzschild
provider, not another evaluation of our superposition formula.

For a single moving hole with v=(0.15,−0.07,0), one captured and two escaping
trajectories are compared at 101 shared affine samples with transformed exact
Schwarzschild trajectories. The observer tetrad and canonical covector are
transformed consistently; the reference escape level set is the same global
finite sphere expressed in rest coordinates. Observed maxima:

| Comparison | Maximum discrepancy |
| --- | ---: |
| Event coordinate component | 6.19e-10M |
| Photon covector component | 8.46e-12 |
| Escaping source direction | 1.08e-12 rad |

Capture/escape classifications agree. Raw states and affine samples are in
[inertial_trajectories.npz](previews/inertial_trajectories.npz). Comparison limits
remain the tested 3e-7 for state components and 1e-8 for direction components;
they were not relaxed to accommodate a discrepancy.

The stationary limit uses a consistent **family of worldlines** whose angular
rate, velocity and acceleration approach zero together. Geometry is compared
at t=−70, 0 and 100M for rate scales 0.1, 0.01, 0.001 and zero, with fixed epoch
positions. It converges to `binary_ks_approx`; at zero rate both the metric and
all derivatives agree at floating-point precision. A zero-rate escaping ray
also matches the old provider. A frozen-center comparison at one nonzero-rate
time is explicitly not used as proof of this limit.

## Time evolution, symmetries and convergence

Every RHS evaluates trajectories at the ray's evolving coordinate time. Tests
check decreasing t along past-directed affine integration and compare the dense
p_t derivative with the Hamiltonian time equation. **p_t is not conserved.**
The rigid circular metric instead obeys the helical identity for
K=partial_t+Omega*partial_phi about its barycenter. Its metric Lie derivative
and the numerical conservation of p.K are independently checked. This identity
does not extend unchanged to arbitrary future inspirals.

Six representative rays include weak and strongly lensed escape, capture at
both centers, the central inter-hole ray and a near-boundary escape. A
documented direction-space bisection selects the latter; it is not an
analytical binary critical curve. All six classifications agree at rtol
1e-7, 1e-9 and 1e-11, with atol=rtol/100. The near-boundary source direction
differs from the tight run by **1.85e-5 rad** at 1e-7, improving to **1.88e-8 rad**
at the preview's 1e-9 setting. Other noncentral escaping rays at 1e-9 differ by
at most **4.0e-10 rad**. Time evolution reaches a sampled fractional p_t change
of approximately **13.7%** on the near-boundary ray while helical error remains
approximately **1.1e-11** at 1e-9. Energy change is physical within the prescribed
metric, not an integration-conservation failure. Sampled extrema can vary with
the integrator's sampling even when trajectories converge.

Equal-mass tests cover phase/rotation covariance, half-period geometry, and
capture-label exchange after half a period. No symmetry of the fixed starfield
texture is required. Serial and spawn-worker tests compare every saved array
exactly with identical supersampling. Old Schwarzschild/Kerr/Carter optimizations
cannot activate for the new metric.

For representative captured and near-boundary escaping rays, changing the
capture fraction among **0.98, 0.995 and 0.999** preserves classification in the
sampled experiment; recorded termination events change as expected. This is
limited sensitivity evidence, not proof that silhouettes are independent of
the provisional cutoff.

## Moving domain and finite background

Absorbers are instantaneous rest-radius ellipsoids, |S(v)(x−s(t))|=2m*fraction.
They are not calculated binary horizons. A tied/overlapping hit is unresolved.
Explicit tests exercise overlap and signature loss using non-orbiting diagnostic
controls outside the public wide-orbit policy. Invalid inversions, failed
integration, affine limits and domain exits retain unresolved classification
and reasons. A center leaving the fixed finite escape domain also invalidates
the domain rather than being silently ignored.

Signature uses **−det(full g)**, not the old stationary n1.n2 expression. The
spatial block is positive definite for positive KS weights, so this determinant
sign establishes Lorentzian signature. A generalized boosted-null-vector
determinant identity and direct eigenvalues are tested independently. The
existing positive guard margin is 0.05. No unphysical static observer is allowed.

The same outgoing-Cartesian-tangent convention is used at a finite background
radius, without an analytic tail or an infinity claim. For three escaping rays,
the maximum successive direction changes in this audit are:

| Radius change | Maximum angular difference |
| --- | ---: |
| 250 → 500M | 1.29e-4 rad |
| 500 → 1000M | 3.63e-5 rad |
| 1000 → 2000M | 1.79e-5 rad |

All sampled classifications remain escaped. This time-dependent prescription
converges more slowly than the stationary example. The preview deliberately
uses **500M**: its discrepancy from the sampled 2000M direction is of order
5.4e-5 rad or less for these rays (roughly 0.008 central pixels at 96×72).
These are measured finite-radius changes, **not rigorous errors at infinity**.
Higher-precision source mapping needs a larger boundary and further tests;
none of the old Schwarzschild/Kerr tails are reused.

## Curvature residual diagnostic

A separate validation-only diagnostic forms Christoffels from analytic first
derivatives and differentiates them with a five-point stencil. For exact
constant-velocity Schwarzschild at the control point, maximum Ricci component
decreases from 3.76e-9 through 4.65e-11 to **4.04e-13/M²** as the step changes
0.03 → 0.01 → 0.003M. It therefore distinguishes numerical stencil error from
the binary residuals in this sample.

| Orbiting-model point | Ricci scalar at finest step (1/M²) | Largest coordinate Ricci component (1/M²) |
| --- | ---: | ---: |
| Near: (t,x,y,z)=(0,15,2,1) | −1.78562e-3 | 4.78090e-3 |
| Intermediate: (0,4,3,7) | 1.02449e-5 | 3.49723e-5 |
| Far: (0,2,3,80) | 3.55891e-8 | 3.27542e-8 |

These values persist as the stencil is refined; full tensors and step results
are saved. The Ricci scalar is invariant, while the component maximum depends
on coordinates. This does not bound the metric's physical error, give a
percentage agreement with numerical relativity, or replace constraint-solving.
It demonstrates that the prescribed binary model is **not vacuum**, with model
violations well above this diagnostic's numerical floor at these points.

## Artifacts, regressions and remaining limits

The two 96×72 camera times share position, orientation, FOV, background and
settings. The second time is T/8=129.0540872M after the first, so the bodies'
phase at the camera advances by 45°. Photons encounter the binary at their own
past times, not that single phase. Each celestial grid is repainted from its
scene's saved starfield rays without another integration. A matching old
stationary-provider render also serves as the validated zero-rate control;
it is not a snapshot of the nonzero-rate metric.

Per-scene counts, run times and constraint maxima are recorded in
`previews/preview_records.json` and each `diagnostics.json`. These executions
overlap the regression suite and are **not controlled performance benchmarks**.
The atlas enlarges 96×72 rasters, not their physical ray sampling. No animation,
large-resolution or supersampled example batch is launched.

The exact old stationary metric/domain and physical source hashes are preserved.
Full test results and final per-scene measurements accompany this report.
The main limitations remain prescribed circular motion, no PN dynamics or
backreaction, instantaneous rather than retarded construction, no acceleration
terms in the metric boost Jacobian, finite background mapping, provisional
absorbers, sampled constraints and sensitivity near complicated capture
boundaries. Future trajectory providers must supply consistent position,
velocity and acceleration; future inspirals lose the rigid helical invariant.
Actual CarpetX/numerical spacetime data loading, disks and radiative transfer
remain separate tasks.

## Final preview measurements and regression results

All scenes used 96×72=6,912 rays, one sample/pixel, ten spawn workers, a
75M static camera, 40° horizontal FOV, escape radius 500M, rtol=1e-9 and
atol=1e-11. The same grid directions/statuses are reused from each scene's
starfield file; there are zero additional grid integrations.

| Scene | Provisional captures (1 / 2) | Escaped | Unresolved | Wall time (s) | Max sampled null error | Max helical error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Camera time 0, phase 0 | 72 / 72 | 6,768 | 0 | 447.54 | 1.49e-8 | 2.50e-11 |
| Camera time T/8, phase 45° | 74 / 74 | 6,764 | 0 | 435.78 | 1.81e-8 | 5.06e-11 |
| Stationary / zero-rate control | 72 / 72 | 6,768 | 0 | 218.63 | 1.59e-8 | not applicable |

Minimum sampled determinant margins are 0.86047, 0.85858 and 0.87313, all above
the 0.05 guard. No trajectory has incomplete diagnostics. The orbiting scenes'
maximum canonical p_t changes are 29.30% and 28.12%; these are changes within
the prescribed metric, not observed-frequency photometry or conservation errors.
Each phase has exact 180° status/capture-label symmetry on its raster, with
escaping direction-component discrepancies no larger than 1.07e-9 under that
rotation. The starfield need not share that symmetry.

At this polar camera, the helical metric makes continuous capture patterns
rotationally equivalent between phases. The 144 versus 148 captured-pixel count
is consistent with coarse sampling of a rotated boundary, not evidence that
the physical shadow area grows. The stationary control places the pair on the
reference X axis. Orbiting silhouettes reflect encounters at past times rather
than simply the holes' camera-time positions. Both actual motion and boosted
metric terms contribute; no image-space rotation or distortion was substituted
for geodesic integration.

An additional unequal-mass control (0.7+0.3M, separation 30M) resolves captures
by both centers and a separate escape. Its maximum sampled null error is
2.72e-10 and helical error 6.76e-13. See
[unequal_mass_control.json](previews/unequal_mass_control.json).

**The consolidated full suite passed: 235 tests in 524.86 s.** This includes all
222 previous tests and 13 new orbit/boost tests. The earlier 233-test regression
run and the two supplemental tests also passed; the consolidated run includes
them together. No established numerical threshold or prior test was weakened.
Coverage includes worldline derivatives, exact Lorentz tensor/trajectory
comparisons, analytic time derivatives, changing-speed acceleration, stationary
limits, helical identity, moving capture/overlap/signature failures, source-radius
convergence, half-period labels, observer/nullness, configuration rejection,
saved metadata and deterministic spawn/supersampling.

- [Full test log](full_suite.txt)
- [Saved measurements](previews/validation.json) and [preview records](previews/preview_records.json)
- [Starfield/grid/control comparison](previews/comparison.png)
- [Phase-zero starfield](previews/phase0/shadow.png), [grid](previews/phase0/grid.png), [rays](previews/phase0/rays.npz)
- [Second-time starfield](previews/phase45/shadow.png), [grid](previews/phase45/grid.png), [rays](previews/phase45/rays.npz)
- [Stationary control](previews/stationary_zero_rate_control/shadow.png)
- [Source preservation checks](source_preservation_check.json) and [reference archive provenance](reference_review.json)

Timing overlaps regression tests and is not a controlled benchmark. No
full-resolution, supersampled or animation example batch was performed.
