# Carter backend: numerical validation and performance

Measurements were made on the Windows development host, Python 3.13, CPU only.
Hamiltonian remains the default. No metric, camera, circular emitter, emission,
ISCO, palette or Hamiltonian RHS formula was changed. Both backends use the
existing opaque surface/rendering pipeline. [Architecture and baseline
profile](ARCHITECTURE.md) identify metric/inverse-derivative evaluations and
allocations in the Hamiltonian RHS as the main acceleration target.

## Independent physical and numerical checks

The [representative-ray audit](ray_validation.json) covers 24 cases across
spins 0, 0.5, 0.9, -0.9 and 0.99 and inclinations including 20°, 60°, 85°
and 90°. Both methods start with exactly the same finite-camera position and
past-directed canonical momentum. In this audit the maximum disk-event
coordinate-component difference is 2.65e-9M and the momentum difference is
3.25e-10. Independent comparisons to the Hamiltonian equations give maximum
position-velocity residual 4.03e-12 and momentum-equation residual 7.38e-12.
The sampled metric null error is at most 1.28e-10 and Carter-constant drift
2.41e-8. These are sampled finite-precision checks, not exact bounds between
all sample points or a certification of all possible near-critical rays.

Turning points are traversed by second-order equations for compact radial and
polar variables. Tests cover inward/outward segments, radial and polar turns,
multiple oscillations, first through fourth and seventh disk crossings, positive
and negative spin, mass scaling, source-radius convergence, initial covector
identity, actual reconstructed metric constraints and both Hamiltonian equations.
No acceptance tolerance was relaxed. The coordinate error norm tightens near
polar turning points to control the ill-conditioned polar Jacobian.

### Finite-camera shadow boundary

[Saved bisection brackets](shadow_validation.json) and
[analytical curves](shadow_curves.npz) use a **30M ZAMO camera**, 35° horizontal
FOV, 96×72 pixel projection, rtol=1e-11, atol=1e-13, 8 sampled directions per
case and 20,001 analytical curve samples. The theory is projected into the
actual finite camera; asymptotic celestial coordinates are not compared directly
to pixels. Bisection stops at approximately 1e-4 pixel bracket width.

| Spin | Inclination | Carter maximum midpoint error (pixel) | RMS (pixel) |
| --- | ---: | ---: | ---: |
| 0 | 90° | 0 at symmetric bracket midpoint | 0 at midpoint |
| 0.5 | 90° | 4.93e-5 | 3.48e-5 |
| 0.9 | 20° | 3.42e-5 | 2.99e-5 |
| 0.9 | 60° | 4.40e-5 | 3.74e-5 |
| -0.9 | 85° | 4.20e-5 | 3.68e-5 |
| 0.99 | 85° | 4.30e-5 | 3.48e-5 |

Maximum backend midpoint disagreement is 3.16e-5 pixel (zero-spin case);
other sampled midpoints agree within the same bisection bins. The zero-spin
Carter bisection encountered **8 deliberately unresolved exactly critical
midpoint shots**. Each interval was instead bounded by separately resolved
captured/escaped side shots. Its zero midpoint error does **not** establish
zero integration error. Bracket width, finite analytical-curve sampling,
integration error and finite-image raster sampling remain distinct limits.
No other shadow-audit shots were unresolved. See [shadow_comparison.png](shadow_comparison.png).

### Seventh-crossing discrepancy

The original failure record in `examples/kerr_disks/nearcritical_reference_limit.json`
is retained unchanged. Its 640×480 camera ray is
**(314.5056064910384, 280.9199781195458)**, camera 60M, FOV 60°, inclination
85°, spin 0.9, retrograde disk ISCO–20M. The first emitting hit is at plane
crossing 7. The new [investigation](seventh_crossing.json) compares both
production methods to an **independent separated radial/polar quadrature**.

The quadrature obtains the radial turning root and divided differences using
stdlib Decimal at 30, 50 and 70 digits; a squared integration substitution
removes the radial endpoint singularity. It matches radial and polar Mino phases
and independently integrates time and azimuth. The 50/70-digit runs agree in
event components to about 2e-13M; this is a convergence measurement, not an
assertion of arbitrary precision in the double-precision camera initial data.
The turning radius is approximately 2.40111445194M and the seventh event is
approximately `(t,x,y,z)=(-159.664700217,8.448431442,-9.039786579,0)`.

Maximum absolute event-component difference from quadrature, in M:

| Requested rtol | Hamiltonian | New Carter |
| ---: | ---: | ---: |
| 1e-9 | 5.986e-3 | 7.838e-6 |
| 1e-11 | 9.887e-5 | 9.325e-8 |
| 1e-12 | 8.876e-6 | 2.820e-8 |
| 1e-13 | 9.367e-7 | 1.451e-8 |

The original **3e-4M criterion is unchanged**. Both production backends meet
it for this trajectory at rtol≤1e-11; default Hamiltonian rtol=1e-9 does not.
The new Carter backend also meets it at 1e-9. This specific disagreement is
therefore independently resolved, without forcing two solvers to agree.

The older r-based reference still differs by 0.0122002M at its tightest tested
tolerance. Its polar constraint and crossing Mino phase converge well, while
its radial first-integral residual near the turn is approximately -1.97e-8
in normalized units, with larger drift elsewhere. Its initial radial residual
is already approximately 9.60e-9. This identifies radial potential/velocity
cancellation, drift and scaling near the unstable orbit as the limiting
mechanism, rather than a lost polar crossing or a coordinate-only offset.
The compact inverse-radius production formulation avoids that large-radial-
potential subtraction. No arbitrary phase correction or changed event criterion
was applied. [Convergence figure](seventh_convergence.png).

This result does not certify every seventh or higher crossing. Constraint
conservation alone cannot bound phase error arbitrarily close to a separatrix;
convergence and an independent reference remain necessary.

## Matched images and arrays

[Small comparison images](comparison.png) use matched 32×24 rays for a
photographic starfield shadow, prograde edge-on disk, retrograde edge-on disk,
and spin-0.99 edge-on disk. [Celestial grids](grid_comparison.png) reuse those
saved shadow directions without tracing. All four pairs have identical
quantized PNG pixels, identical statuses/initial constants/crossing ordinals,
zero unresolved rays and zero fallbacks. Floating-point ray arrays are compared
before display quantization; PNG equality alone is not the validation.

The larger 96×72 benchmark pairs likewise have identical final PNG pixels and
statuses/crossing numbers. Across those three pairs, maximum source-direction
disagreement is 9.41e-8 rad, hit-position component difference 6.91e-8M,
full event-component difference 8.45e-8M, frequency-shift difference 3.97e-8,
and relative observed bolometric-intensity difference 8.42e-8. All are below
the existing comparison limits: 2e-6 rad, 1e-4M hit position, 2e-6 in g and
2e-5 relative intensity. Physical linear intensity, wrapped source azimuth,
and initial constants are checked independently of display color.

## Tests and compatibility

**211 tests passed in 289.84 s**, including the complete pre-existing suite
and 18 new Carter tests. See [full_suite.txt](full_suite.txt). Tests cover
configuration rejection, zero-spin equivalence, both geodesic equations,
first-hit opacity, momentum reconstruction, background directions, near-critical
guarding, error/fallback provenance, serial/spawn worker equality, supersampling,
display repainting, saved arrays and analytical shadow brackets. Existing
Schwarzschild numerical thresholds and tests are unchanged.

The metric, camera, emitter-motion, emission, display, blackbody-color and
metric-interface source hashes were captured before implementation in
[protected_source_hashes.json](protected_source_hashes.json). The Hamiltonian
RHS and integration equations retain their existing implementation; the shared
entry point adds guarded backend dispatch and result provenance.

## Controlled CPU benchmarks

All three main workloads use spin 0.9, camera radius 60M, horizontal FOV 60°,
**96×72 = 6,912 rays**, one sample/pixel, 10 spawn workers, rtol=1e-9,
atol=1e-11, affine max-step 4M, escape radius 110M and source radius 100,000M.
Disk scenes are prograde, ISCO–20M, with unchanged temperature and blackbody
display. Tests, other audits and other renders had finished before timing.
Each large pair was run once; repeated-run variability is measured separately
on smaller workloads. [Raw measurements](benchmarks.json) include CPU seconds,
function evaluations, memory, startup and the numerical comparison for every pair.
[Host information](benchmark_environment.json).

| Scene | Inclination | Hamiltonian wall (s) | Carter wall (s) | Wall speedup | Worker CPU speedup | RHS calls H / C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Starfield shadow | 60° | 119.741 | 31.780 | 3.768× | 4.213× | 7,761,248 / 5,594,282 |
| Nearly edge-on disk | 85° | 120.138 | 32.045 | 3.749× | 4.118× | 7,629,422 / 5,498,249 |
| Moderately inclined disk | 60° | 110.413 | 32.132 | 3.436× | 3.802× | 7,106,162 / 5,565,506 |

All nine benchmark pairs (three large, six small) passed numerical equivalence,
with **zero unresolved rays and zero fallbacks in either backend**. Their final
PNG pixels match exactly. See [the 96×72 comparison atlas](benchmark_comparison.png).
The small repeated 32×24 runs quantify startup effects:

| Scene | H wall range, two runs (s) | C wall range, two runs (s) | Wall speedup range |
| --- | ---: | ---: | ---: |
| Shadow | 22.833–22.984 | 7.820–7.878 | 2.917–2.920× |
| Edge-on disk | 22.817–22.962 | 8.182–8.246 | 2.767–2.806× |
| Inclined disk | 21.174–21.894 | 8.154–8.206 | 2.597–2.668× |

Worker startup spans about 2.8–3.0 s in the main pairs. Excluding that measured
span gives wall ratios 4.069×, 4.020× and 3.674× respectively, but subtraction
is only an estimate: startup and scheduling can overlap useful work. Summed
worker CPU ratios in the main table independently exclude parent/spawn cost.
Do not attribute the smaller wall ratios solely to solver arithmetic. The
measured peak **per-worker** memory is approximately 79–81.5 MB; ten workers
therefore have appreciable process/texture overhead. This is not a measurement
of simultaneous whole-machine peak memory or of communication cost in isolation.

At default benchmark tolerances the Carter maxima over the main pairs are
null error 1.58e-8, normalized E drift 0, Lz drift 1.68e-11 and Q drift 6.39e-6,
below the unchanged 1e-5 constraint threshold. Q is measured from reconstructed
Cartesian momenta, not simply copied from the conserved input constant. Its
worst-case drift is larger than Hamiltonian's 5.13e-7 in these workloads;
the actual ray/event comparisons and independently checked equations are
necessary alongside conservation. This is not evidence for universally superior
accuracy of one solver at the same requested tolerance.

The gain comes mainly from replacing per-step tensor derivatives/allocations
with scalar separated equations, with an additional reduction in RHS evaluations
and an angular-only exact background tail. Work remains in SciPy stepping,
dense-output construction, disk-event localization and independent metric audits.
No worker count, sampling, physical model or acceptance threshold was
changed between paired runs. No full-resolution speedup is extrapolated as a
measurement. This study holds worker count at ten; it does not re-optimize
Hyper-Threading or establish that twenty workers would help.

## Limitations and reproduction

See [README.md](README.md) for commands, supported geometry, error handling,
tail convention and saved provenance fields. Native exact-axis and near-zero-
Killing-energy conditions need an explicitly requested Hamiltonian fallback.
Uncertifiable critical rays remain unresolved even with fallback. Main-path
capture uses the existing horizon criterion, without crossing a BL singularity.
The background tail follows exact separated angular equations to the existing
finite source radius; no new tail approximation is introduced and source-time
events are not supplied. Kerr multilayer/Schwarzschild alignment-ring diagnostics
are not extended by this work.

The supplied previews are deliberately small and their enlargement adds no
angular resolution. No 640×480 batch or supersampled production render was
launched. Performance measurements below apply to these scenes and this host,
not arbitrary extreme trajectories or a claim of universal speedup.
