# Numerical Observatory

A small, independent CPU ray tracer for the Schwarzschild black-hole shadow.
It traces null geodesics toward the past from a local camera, lenses an astronomical
star map at infinity, and saves a PNG plus numerical diagnostics. No existing C++
project, CarpetX installation, GPU, CUDA, or compiler is required.

![Updated composition](examples/composition/shadow.png)

The updated composition above uses a **32° horizontal field of view**, a static
camera at **400M**, and a **640×480** image. This shows a narrower sky patch than
the original 60° view from 20M, with a smaller centered shadow and a substantial
border around the strong lensing region. It uses the 8k version of the same NASA
star map. Weak gravitational deflection decreases continuously toward the edges;
there is no artificial cutoff or blend to an unlensed background.

After setup, reproduce it with `python examples/render_composition.py` using your
virtual environment's Python. This specialized example uses exact Schwarzschild
orbital quadrature and checks eight representative rays against the independent
Hamiltonian ODE tracer. Its diagnostics distinguish those sampled null checks
from the quadrature-rendered image. Outputs are in `examples/composition/`,
including `unlensed.png` showing the same sky patch without deflection.
Settings are in `examples/composition.toml`. The normal ODE CLI also accepts
that configuration: `python -m numerical_observatory --config examples/composition.toml`.
The original, closer-camera example remains in `examples/rendered/`.

## Setup and execution

Requires Python 3.11 or newer. Run commands from this project directory.
NumPy and SciPy use their standard platform wheels; this project contains no
compiled extensions. Pillow writes PNGs; pytest is used only for testing.

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m numerical_observatory --config examples/default.toml --output output
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m numerical_observatory.validation --output output/validation.json
```

Ubuntu (install `python3-venv` with the system package manager if unavailable):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m numerical_observatory --config examples/default.toml --output output
.venv/bin/python -m pytest -q
.venv/bin/python -m numerical_observatory.validation --output output/validation.json
```

After activating the environment, `numerical-observatory` is also an installed
command. No configuration is required: defaults match `examples/default.toml`.
For a quick preview, append `--width 96 --height 72`. The default is 320×240 with
one ray per pixel. Centered Schwarzschild views reuse exactly equivalent orbits
by rotational symmetry, without interpolating between rays. Every distinct
image-plane radius still uses the full ODE solver. Other views trace every ray.
Rendering can take several minutes; the star map is bundled for offline use.
The supplied 320×240 example took 375 seconds: 15,584 pixels were captured,
61,216 escaped, and none were unresolved. The maximum sampled normalized null
residual was 2.91×10⁻⁷. Its captured-pixel mask agrees with the analytic
finite-distance shadow at every pixel center.

To reproduce the checked-in artifacts, use `--output examples/rendered` for the
renderer and `--output examples/rendered/validation.json` for the validation module.
The renderer writes `shadow.png`, `diagnostics.json`, and `rays.npz`, replacing those names in
the chosen output directory. The JSON records actual configuration, dependency
versions, counts, integration work, elapsed time, and worst sampled residuals.
Work counts and constraint maxima refer to the explicitly integrated representative
orbits; capture/escape counts refer to all image pixels. `rays.npz` contains
`directions` (height×width×3 source unit vectors) and `status` (0 captured,
1 escaped, 2 unresolved). Non-escaping directions are NaN. This allows another
background or display exposure to be applied without integrating again.

## Configuration and units

We use signature (−+++) and geometrized units G=c=1. The mass parameter is the
length GM_physical/c². In TOML, camera positions, target, time, capture/escape
radii, affine budget, and maximum step are expressed in multiples of M; the
loader scales them by `mass`. The Python API instead takes coordinates and
integration lengths directly in geometrized units. Absolute tolerance is a
single numeric tolerance on all eight state components; it is not mass-scaled.

The camera is static, outside r=2M. `target` specifies the forward orientation,
and `up` sets its roll. These Cartesian directions label axes in the local
orthonormal frame, not coordinate velocities. FOV is horizontal; aspect ratio
sets the vertical extent of the pinhole image plane. Pixel centers are sampled.
`rtol`, `atol`, `max_step`, and `max_affine` control integration. The default
escape radius is 100M; it must exceed both the camera radius and 3M.
Capture occurs at 1.99M, just inside the horizon.

The `[background]` section controls `longitude_degrees`, `latitude_degrees`,
`roll_degrees`, `exposure`, and `resolution` (4 or 8 kilopixels). The first three rotate the astronomical sky
independently of the camera. For a camera looking along −z, its unlensed center
points to the configured galactic longitude/latitude; roll rotates the surrounding
sky. Exposure multiplies decoded linear RGB before display clipping and sRGB
encoding. It is a visual exposure, not a physical luminosity measurement.

## Physics and numerical method

`metric.py` evaluates outgoing Cartesian Kerr–Schild coordinates:

```text
r = |x|, n = x/r, f = 2M/r
g_ab = eta_ab + f l_a l_b,      l_a = (1, -n)
g^ab = eta^ab - f k^a k^b,     k^a = (-1, -n)
```

This chart is regular through the **past** horizon, appropriate for inward
past-directed integration. It is the time reverse of the more usual ingoing
Kerr–Schild chart. Capture labels the shadow of an ideal black hole with no
light emerging from the horizon; the eternal vacuum solution's white-hole
region is not an emission source in this model.

For the static observer, u=(1/sqrt(1−f),0,0,0). A local spatial vector d maps to
the spacetime vector (−f(n·d)/sqrt(1−f), d+(sqrt(1−f)−1)(n·d)n).
The camera launches k=−u+e(d) with |d|=1, then lowers its index to obtain p.
Thus each photon starts past-directed and null, with unit local energy.

SciPy `solve_ivp` with DOP853 evolves Hamilton's equations with increasing
past-directed affine parameter:

```text
H = 1/2 g^ab p_a p_b = 0
dx^a/dlambda = g^ab p_b
dp_a/dlambda = -1/2 (partial_a g^bc) p_b p_c
```

Analytic metric derivatives avoid finite-difference noise. The integration
does not project momentum back onto the null cone or renormalize it. Null
residual |g^ab p_a p_b|/p_t(initial)² is monitored at accepted steps and dense
output midpoints. This is a sampled diagnostic, not a bound everywhere along
the curve. Energy p_t and the Cartesian angular momentum vector are monitored
as independent stationary/spherical-symmetry diagnostics.

An inward crossing of the capture sphere classifies a ray as captured. An
outward crossing of the escape sphere classifies it as escaped. Because the
escape sphere lies outside the photon sphere, an exact outward null orbit
there reaches infinity. Solver failure, exhausted affine budget, or a null
residual exceeding `null_tolerance` instead produce an unresolved ray, shown
in magenta. Captured rays are black.

For escaped rays we integrate the remaining orbital angle analytically by
one-dimensional quadrature, using b=|L|/|p_t| and u=1/r:

```text
delta_phi = integral_0^(1/R) b / sqrt(1-b²u²+2Mb²u³) du.
```

Rotating the escape position's radial unit vector in its conserved orbital
plane by this angle gives the source direction at infinity. This is the
direction *toward* the source, opposite to arriving light propagation.
No finite-radius tangent approximation is used.

The background is NASA's catalog-based
[Deep Star Maps (2012)](https://svs.gsfc.nasa.gov/3895/), containing measured
stellar positions, brightnesses, and colors. Credit: **NASA/Goddard Space Flight
Center Scientific Visualization Studio**. The unmodified 4096×2048 galactic
JPEG is bundled; see [asset attribution](src/numerical_observatory/data/ATTRIBUTION.md).
It is an astronomical catalog visualization, not a telescope photograph.

Escaping source directions sample the spherical map with bilinear interpolation
in linear RGB and periodic longitude wrapping. Distorted and repeated stellar
images, together with the bent Milky Way background, show gravitational lensing.
The shadow is black. There are no grid lines, orange equator, artificial glow,
or accretion disk. Any bright structures are lensed background light. This is
illustrative RGB rendering without spectral/redshift transfer, scattering, or
time variability. One ray per pixel and finite texture resolution can alias
small stars and fine photon rings; this is not calibrated stellar photometry.

## Validation

For a static observer at r>3M looking toward the center,

```text
b_crit = 3 sqrt(3) M
alpha_shadow = asin(b_crit sqrt(1−2M/r) / r).
```

The factor sqrt(1−2M/r) accounts for finite observer distance. A centered
pinhole camera has shadow radius in pixels
`width/2 * tan(alpha_shadow)/tan(horizontal_fov/2)`.
The impact parameter itself is not an angular or pixel radius. See the
[Perlick–Tsupko review](https://arxiv.org/abs/2105.07101) for analytical shadow
geometry and finite-distance observers.

Tests check metric inversion and derivatives across the horizon, tetrad
orthonormality and initial nullness, radial rays, termination and rejection,
mass scaling, and rendering output. They locate the capture/escape boundary
by angular bisection at r=8M, 20M, and 50M; the recovered critical impact
parameter must agree to relative error below 3×10⁻⁷. This boundary test is
independent of the coarse image pixel grid.

Convergence tests tighten (rtol,atol) from (10⁻⁶,10⁻⁸) through
(10⁻⁹,10⁻¹¹) to (10⁻¹²,10⁻¹⁴), including a ray near the critical boundary.
An independent turning-point quadrature of the Schwarzschild orbital equation
checks the escaping directions, including bending orientation. Another test
changes the escape sphere to verify the continuation to infinity.
`validation.json` records boundary brackets and convergence against independent
orbit quadrature. Exactly critical rays have arbitrarily long dwell times near
r=3M and may be unresolved under a finite affine budget.

The supplied validation run recovered b_crit to within 4.8×10⁻⁸ relative error
at all three camera radii. Its near-critical ray gave:

| Relative tolerance | Direction error vs. orbit quadrature (unit-vector norm) | Maximum sampled null residual |
| --- | --- | --- |
| 10⁻⁶ | 1.18×10⁻⁵ | 4.06×10⁻⁶ |
| 10⁻⁹ | 8.05×10⁻⁹ | 5.08×10⁻⁹ |
| 10⁻¹² | 1.27×10⁻¹¹ | 7.26×10⁻¹² |

All 19 automated tests passed on Windows, including sky seam/pole handling,
saved ray maps, and rotational reuse compared with direct ODE integration
for both axis-aligned and arbitrarily positioned centered cameras. Run
`render(..., use_symmetry=False)` through the Python API to force direct tracing.
The installed CLI smoke render also passed on Windows
with Python 3.13. The Ubuntu commands use the same portable Python source;
an Ubuntu runtime was not available for an execution check. In restricted
environments where pytest cannot write to the system temporary directory,
use `python -m pytest -q --basetemp .tmp/pytest-run` (pytest clears that chosen
temporary test directory).

## Architecture and extension

| Module | Responsibility |
| --- | --- |
| `metric.py` | Four-dimensional event-based metric and all four inverse-metric derivatives |
| `geodesic.py` | Generic Hamiltonian RHS, SciPy integration, Schwarzschild boundaries and asymptotic continuation |
| `camera.py` | Camera geometry and Schwarzschild static-observer tetrad |
| `background.py` | Direction-to-RGB emission model |
| `symmetry.py` | Exact rotational reuse for centered Schwarzschild views |
| `render.py` | Pixel loop, statistics and file output |
| `cli.py` | Minimal TOML configuration and CLI |
| `validation.py` | Reproducible boundary and convergence diagnostics |

The metric contract always receives (t,x,y,z), even though this metric is
stationary. The RHS includes the time derivative and can evolve p_t for a
time-dependent metric. The observer frame, capture policy, symmetry diagnostics,
and continuation to infinity are explicitly Schwarzschild-specific; a future
numerical metric must supply suitable replacements. Future CarpetX support
should read saved spacetime fields, interpolate their values/derivatives at
events, and enforce data-domain boundaries. It should not import or require
the CarpetX simulation software. No such loader is implemented here.

Ray generation, integration, and output are separate so optional future batch
CPU/GPU backends can replace the serial loop without changing the observing
model. No acceleration framework or complex configuration system is included.
