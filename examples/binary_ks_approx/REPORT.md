# Binary KS approximation: validation and limitations

## Implementation

`binary_ks_approx.BinaryKSApprox` adds two stationary, nonspinning outgoing
Cartesian Kerr–Schild **covariant** perturbations to Minkowski spacetime. It
inverts the complete 4×4 matrix and obtains inverse derivatives from analytic
covariant derivatives and the matrix inverse identity. This includes cross
terms that would be lost by adding isolated inverse metrics.

The generic Hamiltonian equations, metric-based camera and existing CPU worker
scheduler are reused. Small additions select the binary provider/domain and
save its classification/constraint diagnostics. The existing Schwarzschild and
Kerr providers, camera, emission and display physics were preserved. Carter,
Schwarzschild quadrature, spherical rotational caching and disks cannot activate
in binary mode. See [the pre-implementation proposal](ARCHITECTURE.md).

**This prescribed stationary geometry is not an exact binary vacuum solution.**
There is no Einstein constraint solve, orbital dynamics or estimate of the
metric's physical error at the example separation. Numerical tests validate
integration in this geometry, not a physical binary model.

## Geometry and observer checks

The equal-mass example uses m1=m2=0.5, centers (−6,0,0) and (6,0,0), in total
mass units. A sampled exterior grid contained **889 points**, with zero
non-Lorentzian signatures and minimum determinant-based signature margin
**0.851852**. This is a sampled volume, not a proof of global validity.

Measured maximum inverse-identity error was **2.23e-16**. Inverse derivatives
agreed with centered finite differences (step 1e-5M) to **4.37e-11**. Camera
tetrad orthonormality error was **2.23e-16**. Automated tests check initial
past-directed null momenta, the determinant identity, zero time derivatives,
independent covariant superposition, unequal masses, mass scaling, source
translation and invalid parameter rejection.

For two nonnegative KS terms the spatial block is positive definite and
`det(g)=-D`, where `D=1-f1*f2*(1-n1.dot(n2))^2`. Lorentzian signature requires
D>0. A deliberately close pair at centers ±1 has a positive-definite four-metric
at its midpoint; the signature is therefore genuinely invalid there. A test
ray aimed toward that region terminates **unresolved at the D=0.05 guard**,
not as a captured ray. An injected inversion failure is likewise unresolved
with an explicit reason and incomplete-diagnostics flag. A static observer
whose coordinate-time worldline is not timelike is rejected.

## Single-mass-zero limit

The metric, inverse and derivatives reduce to the existing Schwarzschild
provider at the active center, including translated-center tests. Inactive
zero-mass centers contribute no singularity. The generic Hamiltonian solver
remains in use; no spherical geodesic optimization is introduced.

Independent existing-Schwarzschild comparisons at rtol=1e-11, atol=1e-13 use
matched 1.99M capture radius and 1000M escape sphere. Three representative rays
(one captured, two escaped) agree in classification. The maximum sampled full
state-component difference is **6.41e-10**; position/time units are M and photon
momenta have the same local normalization. Escaping background directions at
the same finite endpoint agree to **4.46e-13 rad** or better.

The existing Schwarzschild default uses an analytic continuation to infinity,
whereas the binary mode deliberately uses a finite-sphere tangent. At 1000M,
the two escaping single-hole rays differ from that infinity-map by **3.60e-10**
and **1.33e-9 rad**. Tests verify that this difference decreases as the escape
radius grows. These endpoint conventions must be matched before interpreting
any discrepancy as a failure of the single-hole limit. Exact mathematical
metric equivalence does not imply byte-identical adaptive trajectories or
certification of arbitrarily near-critical rays.

## Binary trajectory and background convergence

Seven representative camera directions include two provisional captures,
weak/strongly deflected escaping rays and the symmetric central trajectory.
Their classifications agree for rtol=1e-7, 1e-9 and 1e-11, with atol=rtol/100.
Compared with the tight run, maximum escaping source-direction error decreases
from **2.42e-7 rad** at 1e-7 to **2.68e-9 rad** at the preview's 1e-9 setting.
The largest endpoint coordinate-component difference decreases from
**1.90e-4M** to **2.10e-6M**; long outgoing trajectories amplify small directional
differences in endpoint position. Tight-run sampled null errors are at most
**2.02e-10** for these rays. Stationarity preserves p_t; measured energy drift
is zero at stored precision.

Four escaping directions were checked independently at radii 250, 500, 1000
and 2000M with rtol=1e-11 and atol=1e-13. Maximum successive direction changes:

| Escape-radius change | Maximum angular change |
| --- | ---: |
| 250 → 500M | 6.04e-7 rad |
| 500 → 1000M | 3.78e-8 rad |
| 1000 → 2000M | 2.36e-9 rad |

This establishes finite-radius convergence for the sampled rays. The last
column is a finite-radius **difference**, not a rigorous bound on error at
infinity or a guarantee for every chaotic/near-critical trajectory. No
Schwarzschild asymptotic correction or straight-line modification is applied.

## 96×72 preview and exchange symmetry

The [preview pair](preview/comparison.png) uses a static camera at (0,0,45)M,
40° horizontal FOV, one ray/pixel, escape radius 1000M, rtol=1e-9,
atol=1e-11 and ten spawn workers. There are **6,912 independently integrated
rays**, no rotational reuse, and no emitting disk:

| Classification | Count |
| --- | ---: |
| Provisional capture at center 1 | 216 |
| Provisional capture at center 2 | 216 |
| Escaped | 6,480 |
| Unresolved | 0 |

Maximum sampled null error is **2.19e-8** against the unchanged 1e-5 threshold;
minimum sampled signature margin is **0.642701**, comfortably above the guard.
Energy drift is zero at stored precision and no ray has incomplete diagnostics.
Capture spheres each have radius **0.995M_total** (1.99 times their individual
mass), chosen provisionally from isolated-hole scales, not measured horizons.

Exchanging equal-mass labels leaves the metric unchanged. The saved preview
has zero reflected status mismatches and zero reflected capture-label
mismatches after swapping labels. Reflected escaping Cartesian directions
agree in components within **1.91e-10**, consistent with adaptive floating-point
integration. The background texture itself need not be reflection-symmetric.

The grid is repainted from **the same saved directions and statuses** as the
starfield, with zero additional integrations. It reveals distorted and repeated
sky-grid structures around the two absorption silhouettes. This illustrates
geodesic lensing by the prescribed metric; it does not establish a physically
accurate binary shadow, event horizon, image order or dynamical behavior.
The atlas magnifies the 96×72 rasters; no additional angular sampling is implied.

The render took approximately 359 s on this Windows CPU host with ten workers;
tests were running concurrently, so this is an execution record, **not a
controlled performance benchmark**. No full-resolution render was launched.

## Tests, data and limitations

The full suite passed: **222 tests in 412.40 s**, including all previous
Schwarzschild/Kerr tests and 11 focused binary tests. Tests cover inversion,
analytic derivatives, signature, tetrads/nullness, zero-mass trajectories,
equal-mass exchange/reflection, provisional capture identities, integration
failure, affine-limit/invalid-domain unresolved outcomes, tolerance/escape-radius
convergence, unsupported modes, deterministic serial/spawn arrays and
supersampling. The first test invocation encountered an unwritable pytest temp
directory; rerunning with a writable workspace `--basetemp` resolved that setup
issue without changing tests or tolerances.

- [Complete test log](full_suite.txt)
- [Focused test log](focused_tests.txt)
- [Raw numerical audit](preview/validation.json)
- [Render diagnostics and codebooks](preview/starfield/diagnostics.json)
- [Numerical ray data](preview/starfield/rays.npz)
- [Unchanged physics-source checks](preserved_physics_sources.json)
- [Starfield](preview/starfield/shadow.png) and [celestial grid](preview/grid.png)
- [Configuration, conventions and PowerShell/Ubuntu commands](README.md)

The central limitations are unsolved Einstein equations, stationary fixed
centers, coordinate-dependent superposition, provisional absorbing boundaries,
possible loss of Lorentzian signature, finite background termination, sampled
constraint monitoring and coarse image resolution. No event-horizon finder,
Einstein residual analysis, physical error bound, binary disk model, orbital
motion or CarpetX loader is provided. Future investigations should quantify
metric residuals and cutoff sensitivity before drawing astrophysical
conclusions; actual dynamical binary observations require a more physical
spacetime provider.
