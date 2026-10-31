# Numerical Observatory

A small, independent CPU ray tracer for the Schwarzschild black-hole shadow.
It traces null geodesics toward the past from a local camera, lenses an astronomical
star map at infinity, and saves a PNG plus numerical diagnostics. No existing C++
project, CarpetX installation, GPU, CUDA, or compiler is required.

An optional opaque luminous disk now adds circular orbital motion, gravitational
redshift, Doppler beaming, and higher-order surface images. It is **disabled by
default**; existing starfield/grid configurations retain their original results.
See the [disk validation report](examples/disks/REPORT.md).

## Optional luminous disk

![Inclined disk](examples/disks/inclined_starfield/shadow.png)
![Nearly edge-on disk](examples/disks/edge_on_starfield/shadow.png)

Use `examples/disk_inclined.toml` (60°) or `examples/disk_edge_on.toml` (85°).
After the platform-specific setup below, these commands work with either
PowerShell or Ubuntu when the virtual environment is activated; alternatively
replace `python` with `.venv/Scripts/python` or `.venv/bin/python` respectively.

```text
python -m numerical_observatory --config examples/disk_inclined.toml --output output/disk
python -m numerical_observatory --config examples/disk_edge_on.toml --background grid --output output/disk-grid
python -m numerical_observatory --config examples/disk_inclined.toml --no-disk --method quadrature --output output/no-disk
python examples/render_disks.py
python -m numerical_observatory.disk_validation
python examples/validate_disk_sampling.py
python -m pytest -q
```

`render_disks.py` produces all six matched starfield/grid examples in
`examples/disks/`, including diskless controls. Defaults are 192×144 with one
ray per pixel; `--width 384 --height 288 --samples 2` increases resolution and
uses 2×2 subpixel rays. CPU rendering takes minutes at the default size.
The sampling audit renders additional low-resolution scenes and asserts
convergence of physical observables; `--measure-only` rechecks saved results.

Minimal optional configuration (the example files include all settings):

```toml
[disk]
enabled = true
inner_radius = 6.0
outer_radius = 20.0
inclination_degrees = 60.0
position_angle_degrees = 0.0
rotation = 1
temperature_inner = 10000.0
temperature_index = 0.75
emissivity_scale = 1.0

[disk.display]
exposure = 2.0
reference_temperature = 10000.0
tone_mapping = "reinhard"
palette = "thermal"
```

Radii are in M; temperature is in kelvin. The normal is
`(sin(i) sin(PA), sin(i) cos(PA), cos(i))`: inclination is relative to global
+z and equals the observer inclination for these cameras on +z. Moving or
reorienting a camera requires interpreting the angle relative to its actual
view. Rotation is +1 or -1 around that normal. Both disk faces emit identically.
Timelike circular orbits require r>3M; those below 6M are unstable. The factory
rejects an inner radius at or below 3M. This is a toy emitting surface, not a
realistic accretion-flow solution.

The physical model is `T(r)=T_inner*(r/r_inner)**(-temperature_index)` and
`I_bol,emit=emissivity_scale*sigma*T**4/pi`. Covector/four-velocity contractions
give g; observed bolometric intensity is **g⁴ I_bol,emit**. The separate spectral
function implements **g³ I_nu,emit(nu_obs/g)**, preserving I_nu/nu³. Saved
`disk_rays.npz` includes physical intensity, g, hit locations, crossing index,
and escaped directions. Status codes are 0 captured, 1 escaped, 2 unresolved,
3 emitted. RGB uses an illustrative thermal palette, exposure and tone mapping;
it is not a calibrated color observation. `palette="gray"` and
`tone_mapping="linear"` are alternatives. Starfield exposure is independent.

For CIE-based thermal color, set `palette = "blackbody"` in `[disk.display]`.
The default remains `"thermal"`; `"gray"` is also available. The blackbody palette
integrates Planck **spectral radiance per wavelength** against the standard
CIE 1931 2° color-matching functions over 360–830 nm. It uses the **observed**
temperature `g*T_emitted`, already stored by the renderer. Only chromaticity is
retained: the existing observed bolometric radiance (including g⁴), exposure,
and tone mapping supply brightness. There is no extra Doppler multiplier or
visible-flux weighting of the saved intensity.

The bundled, unmodified 1 nm CIE table comes from
[CIE 018:2019, Table 6](https://doi.org/10.25039/CIE.DS.xvudnb9b), licensed
CC BY-SA 4.0; original metadata, checksums, attribution and license links are in
`src/numerical_observatory/data/ATTRIBUTION.md`. No new dependency is required.
A lazy 4,097-entry log-temperature lookup (1,000–10,000,000 K) avoids spectral
integration per pixel. Positive temperatures outside that range use endpoint
chromaticities, without clamping physical temperature or intensity. Invalid
nonpositive/nonfinite display temperatures raise an error.

XYZ is converted to linear sRGB using a matrix derived from the
[standard sRGB primaries and D65 white point](https://registry.color.org/rgb-registry/srgb).
These are self-luminous source colors: no chromatic adaptation or per-temperature
white balancing is applied, and there is no D50 ICC profile-connection step.
In particular, a 6,504 K Planck spectrum is not assumed identical to daylight D65.
Negative out-of-gamut RGB channels are clipped to zero, then the largest channel
is normalized to one. The existing bolometric display level multiplies that
color, and final channels are clipped to [0,1] before standard sRGB encoding.
Thus the brightness control retains its existing maximum-channel convention;
it does not promise constant photopic luminance or a calibrated visible-light
flux. Gamut clipping changes chromaticity for very cool sources. Actual
appearance also depends on exposure, tone mapping, display gamut and viewing
conditions. The palette adds no artistic orange bias.

The tests compare the lookup against independent SI Planck-frequency integration
with the wavelength Jacobian and 0.25 nm Simpson quadrature: normalized XYZ agrees
within 3e-6 across the supported range, and within 2e-6 at 3,000, 5,800, 10,000
and 15,000 K. Lookup interpolation alone agrees with direct 1 nm integration
within 3e-7. These are numerical colorimetry checks, not guarantees about human
appearance.

Recolor an existing completed render without retracing or changing its files:

```text
python examples/compare_disk_palettes.py
```

This reads `examples/disks/edge_on_starfield/disk_rays.npz` and writes a labelled
comparison plus individual images to the separate `examples/blackbody_palette/`
directory. It changes only the palette; background, physical intensities,
frequency shifts, exposure and tone mapping are identical. `--source` and
`--output` select other completed ray data and a separate destination. The
script does not consult or modify the high-resolution render configuration.

![Thermal and blackbody palettes](examples/blackbody_palette/comparison.png)

Disk tracing uses the general ODE path even when `--method quadrature` is
requested; diagnostics record this fallback. Plane crossings through the inner
hole or outside the outer radius are skipped, and the first valid opaque hit
occludes everything behind it. No screen-space glow is added. Exact rotational
reuse, where allowed, reuses trajectories only: surface hits and Doppler shifts
are calculated separately for each pixel. The independent orbital-quadrature
audit checks intersections and photon momenta but is not a production disk
backend.

The added modules separate `surfaces.py` (geometry), `emitter_motion.py`
(Schwarzschild orbits), `emission.py` (radiance), `display.py` (visualization),
`disk.py` (scene assembly), and `surface_render.py` (opaque rendering).
The camera and metric-provider interfaces remain general. New spacetime models
need their own emitter motion and trace domain; they cannot silently use the
Schwarzschild-only optimizations.

![Validated starfield](examples/validated/starfield.png)

![Matching celestial grid](examples/validated/grid.png)

The [lensing validation report](examples/validated/REPORT.md) gives measured
shadow size, ODE/quadrature convergence, winding counts, and sampling limits.
The outer arcs are physical; the finest near-shadow structure is under-resolved.
Both images above use exactly the same saved 4×4 subpixel ray directions.

The updated composition above uses a **32° horizontal field of view**, a static
camera at **400M**, and a **640×480** image. This shows a narrower sky patch than
the original 60° view from 20M, with a smaller centered shadow and a substantial
border around the strong lensing region. It uses the 8k version of the same NASA
star map. Weak gravitational deflection decreases continuously toward the edges;
there is no artificial cutoff or blend to an unlensed background.

Settings are in `examples/composition.toml`. After setup, the commands below
regenerate both backgrounds, unlensed references, a labelled grid atlas, a freshly
traced near-shadow close-up, and quantitative diagnostics in `examples/validated/`.
Historical renders remain in `examples/composition/` and `examples/rendered/`.

## Parallel CPU rendering

`--workers 1` retains serial rendering and is the default. `--workers N` enables
up to N spawned processes for N>1. `--workers 0` selects available physical CPU
cores when Windows topology or Linux sysfs can identify them, intersecting the
result with process CPU affinity. If physical topology is unavailable, it uses
half the available logical CPUs, rounded up (at least one). Here it detects
10 physical cores and 20 logical CPUs, so automatic selection is 10. Containers
with CPU quotas that do not restrict affinity may require an explicit count.
For very small workloads the pool is capped to the number of work batches;
diagnostics record both requested and selected pool sizes. Windows allows at
most 61 workers. Negative counts are errors.

Windows PowerShell (after setup):

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/disk_edge_on.toml --workers 0 --output output/disk-edge-on-auto
.\.venv\Scripts\python -m numerical_observatory --config examples/disk_edge_on.toml --workers 5 --samples 2 --background grid --output output/disk-grid-parallel
```

Ubuntu:

```bash
.venv/bin/python -m numerical_observatory --config examples/disk_edge_on.toml --workers 0 --output output/disk-edge-on-auto
.venv/bin/python -m numerical_observatory --config examples/disk_edge_on.toml --workers 5 --samples 2 --background grid --output output/disk-grid-parallel
```

Spawn is used on both platforms. Python callers must put rendering inside
`if __name__ == "__main__":`; metric, surface and custom domain objects must be
pickleable, with functions/classes defined at module scope. The normal CLI
already handles this. An explicitly supplied dense `path_cache` remains a
serial-only API; parallel rendering creates its own exact orbit tasks.

Tasks are dynamically scheduled, with at most twice the worker count in flight.
General ODE views use 64-pixel linear tiles. Centered Schwarzschild views group
all pixels with the same image-plane radius, batching four distinct radii per
task. Each representative trajectory is integrated exactly once across the
whole pool; every disk ray still evaluates its own rotated trajectory, surface
crossings and emitter state. Quadrature groups sixteen escaping orbits per
task. There is no trajectory interpolation or change to integration tolerances.
The parent assembles results by pixel coordinates, shades the sky/disk with
the original display code and averages subpixels in the original order.

Worker scenes initialize once; large star textures and final image arrays live
only in the parent. Dense trajectories stay within their worker and are released
between batches. Small ray results cross process boundaries. NumPy/BLAS thread
counts are set to one in spawned workers to avoid multiplying process parallelism
by library thread parallelism; the parent's environment is restored afterward.
Each process still incurs Python/NumPy/SciPy memory and startup costs. Small
quadrature renders can be much faster with one worker. Parent-side display,
assembly, compression, communication and serial work limit total speedup.
Hyper-Threading shares physical core resources: 20 workers need benchmarking
against 5 and 10, rather than an assumed 2× gain. On this 10-core/20-thread
Windows computer, **10 workers is the measured recommendation**. The 128×96
disk benchmark took 145.95 s serially, 33.33 s with 5 workers, 23.04 s with 10,
and 24.41 s with 20. Three-run medians confirmed 10 (22.37 s) ahead of 20
(24.04 s): approximately 6.5× the serial speed on this workload. Ten workers also
won the supersampled and background ODE comparisons. The small quadrature
background was fastest serially. See the [benchmark report](examples/parallel_benchmark/REPORT.md)
for efficiency, memory, startup measurements and exact output comparisons.

Recommended 640×480 nearly edge-on command (the full-size runtime has not been
remeasured, so the small-scene speedup is not a promised completion time):

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/disk_edge_on.toml --width 640 --height 480 --workers 10 --output output/disk-edge-on-parallel
```

Ubuntu equivalent:

```bash
.venv/bin/python -m numerical_observatory --config examples/disk_edge_on.toml --width 640 --height 480 --workers 10 --output output/disk-edge-on-parallel
```

Progress reports completed rows, elapsed wall time and estimated remaining time.
When exact orbit groups finish out of row order, batch/work-unit updates keep
the indicator active. Estimates use observed throughput and can shift near
expensive critical trajectories. Updates are throttled to roughly five seconds.
Disk rays, supersampling, both backgrounds and all palettes work with workers.

Worker errors and unexpected exits propagate to the caller; Ctrl+C terminates
and joins this render's workers. Completed images/ray files/diagnostics are staged
in a temporary sibling directory and published after successful rendering.
Interrupted or failed computations leave previous output files intact. Temporary
staging needs additional disk space comparable to the output. Numerical counts,
constraint diagnostics, integration work and saved arrays remain available;
an additional `execution` object records process, startup, CPU-time and memory
measurements. Execution-specific fields and wall-clock timing can differ.

Run the controlled benchmark and inspect its report:

```text
python examples/benchmark_parallel.py --repeat-selection
```

It compares 1, 5, 10 and 20 workers with fresh processes, fixed scene parameters
and tolerances, plus an empty-message startup/communication baseline. Outputs
are isolated in `examples/parallel_benchmark/`; `--resume` retains completed
measurements. The regular tests use exact array/image comparisons, check
integration counts to catch duplicated symmetry work, and exercise worker
failures, abrupt exits, spawn startup and interruption cleanup.

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
.\.venv\Scripts\python -m numerical_observatory.lensing_validation
```

Ubuntu (install `python3-venv` with the system package manager if unavailable):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m numerical_observatory --config examples/default.toml --output output
.venv/bin/python -m pytest -q
.venv/bin/python -m numerical_observatory.validation --output output/validation.json
.venv/bin/python -m numerical_observatory.lensing_validation
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

## Select a background or integration method

Set `[background] mode = "starfield"` or `mode = "grid"` in TOML, or use the
CLI override. For example, after activating the virtual environment:

```bash
python -m numerical_observatory --config examples/composition.toml --method quadrature --samples 4 --background starfield --output output/starfield
python -m numerical_observatory --config examples/composition.toml --method quadrature --samples 4 --background grid --output output/grid
```

`--method ode` is the default and uses SciPy's full Hamiltonian integrator.
`--method quadrature` is optional and restricted to compatible Schwarzschild
views; other metrics/cameras fall back to ODE. `--samples N` traces N×N subpixel
rays per output pixel with either method and averages in linear RGB. Saved ray
maps then have N times the output width and height; counts are subpixel rays.
No interpolation between photon directions is used. Supersampling reduces
aliasing but cannot resolve arbitrarily thin higher-order images.

The grid's cool cyan/blue meridians and warm red/gold parallels are spaced by
`spacing_degrees` (default 15), with minor lines at one-third that interval.
`line_width_degrees` (default 0.25) sets a fixed angular width on the sky.
Sky orientation is shared with the starfield. Starfield-only settings
`exposure` and `resolution` are ignored in grid mode; grid-only spacing/width
are ignored in starfield mode. Grid colors are diagnostic, not disk emission.

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
These conservation diagnostics apply only to the Schwarzschild implementation;
other metrics report them as unavailable rather than assuming symmetries.

An inward crossing of the capture sphere classifies a ray as captured. An
outward crossing of the escape sphere classifies it as escaped. Because the
escape sphere lies outside the photon sphere, an exact outward null orbit
there reaches infinity. Solver failure, exhausted affine budget, or a null
residual exceeding `null_tolerance` instead produce an unresolved ray, shown
in magenta. Captured rays are black.
For inward Schwarzschild rays near the critical impact parameter, sampled
impact drift and the null residual also set an empirical reliability margin.
An uncertain outcome becomes unresolved even when the ordinary null threshold
passes. The audit retains raw outcomes to show when a coarse solver would have
misclassified a ray. Tighten tolerances and check convergence in this regime.

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
In starfield mode the shadow is black. There are no grid lines, orange equator, artificial glow,
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

The expanded automated tests cover sky seam/pole handling,
saved ray maps, and rotational reuse compared with direct ODE integration
for both axis-aligned and arbitrarily positioned centered cameras. Run
`render(..., use_symmetry=False)` through the Python API to force direct tracing.
They also cover general-metric frames, ODE fallback, time-dependent energy,
numerical-domain exit, near-critical rejection, and multi-winding convergence.
Tests and CLI smoke renders run on Windows with Python 3.13.
The Ubuntu commands use the same portable Python source;
an Ubuntu runtime was not available for an execution check. In restricted
environments where pytest cannot write to the system temporary directory,
use `python -m pytest -q --basetemp .tmp/pytest-run` (pytest clears that chosen
temporary test directory).

## Architecture and extension

| Module | Responsibility |
| --- | --- |
| `metric.py` | Four-dimensional event-based metric and all four inverse-metric derivatives |
| `geodesic.py` | Generic Hamiltonian RHS, SciPy integration, Schwarzschild boundaries and asymptotic continuation |
| `camera.py` | Camera geometry, metric-based rest-space tetrad or supplied observer frame |
| `background.py` | Direction-to-RGB emission model |
| `symmetry.py` | Exact rotational reuse for centered Schwarzschild views |
| `schwarzschild_orbit.py` | Guarded optional orbit quadrature and subpixel ray maps |
| `diagnostic_render.py` | Shared-ray background comparisons and sampling |
| `render.py` | Pixel loop, statistics and file output |
| `cli.py` | Minimal TOML configuration and CLI |
| `validation.py` | Reproducible boundary and convergence diagnostics |
| `lensing_validation.py` | Systematic trajectory/winding checks, shadow sweep, matched renders |

The metric contract always receives (t,x,y,z), even though this metric is
stationary. The RHS includes the time derivative and can evolve p_t for a
time-dependent metric. The camera's default observer follows coordinate time
where it is timelike. It uses the positive-definite rest-space metric
`h_ij = g_ij - g_0i g_0j/g_00` and its symmetric inverse square root; this
reproduces the old Schwarzschild frame. A supplied `observer_frame` handles
moving observers and regions where a coordinate-static observer is impossible.
The supplied frame must be orthonormal; the caller chooses its future direction.

Non-Schwarzschild metrics must supply a `TraceDomain` with capture/escape level
sets and a source-direction mapping in the background's sky frame. An optional
valid-data level set terminates unavailable numerical data as unresolved. No
generic metric is presumed spherical or asymptotically flat. The exact
Schwarzschild implementation alone supplies the default horizon, escape sphere,
conservation diagnostics, and analytic continuation to infinity. Quadrature and
rotational reuse require this exact type, a centered coordinate-static camera,
and no custom domain; subclasses cannot silently inherit these optimizations.
Low-level quadrature rejects incompatible inputs; the renderer selects ODE.
Future CarpetX support
should read saved spacetime fields, interpolate their values/derivatives at
events, and enforce data-domain boundaries. It should not import or require
the CarpetX simulation software. No such loader is implemented here.

Ray generation, integration, and output are separate so optional future batch
CPU/GPU backends can replace the serial loop without changing the observing
model. No acceleration framework or complex configuration system is included.
