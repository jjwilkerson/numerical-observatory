# Kerr accretion disks

This guide covers the optional stationary, equatorial, geometrically thin,
opaque disk of circular geodesic test emitters. It shares the existing renderer,
frequency transfer and display code with Schwarzschild. It is not GRMHD,
Novikov–Thorne, a finite-thickness disk, or a radiative-transfer calculation.

## Geometry and rotation

Kerr spin is a*=J/M² about global +Z, with a=M a*; -0.99<=a*<=0.99.
The disk plane is always z=0. Changing spin sign does not tilt or flip its
normal. Set inclination in `[camera]`, measured from +Z (90° equatorial).
Independent disk tilt/position-angle/legacy `rotation` settings are rejected
for an enabled Kerr disk: choose `orbit="prograde"` or `"retrograde"` instead.

Prograde aligns emitter angular momentum with J; retrograde opposes it. Thus
prograde has positive coordinate angular velocity for positive spin and negative
velocity for negative spin. At exactly zero spin, the documented convention is
prograde=+phi and retrograde=-phi. Both then have ISCO=6M.

Kerr uses its **oblate radial coordinate** r. At the equator the Cartesian
cylindrical radius is sqrt(r²+a²), not r. The annulus boundaries and temperature
use r, including in saved emission-radius maps. The default inner edge is the
appropriate exact Bardeen–Press–Teukolsky ISCO:

| Spin magnitude | Prograde r_ISCO/M | Retrograde r_ISCO/M |
|---|---:|---:|
| 0 | 6.000000 | 6.000000 |
| 0.5 | 4.233003 | 7.554585 |
| 0.9 | 2.320883 | 8.717352 |
| 0.99 | 1.454498 | 8.971861 |

`inner_radius="isco"` is the default. A numerical radius in M must be at or
above that orbit's ISCO; `outer_radius` (default 20M) must be larger. The resolved
physical inner radius and ISCO are both saved. Small-spin evaluation uses the
exact identity 3-Z1=a*²(3Z1-1)/Z1² to avoid cancellation of the linear spin
shift near zero. This is algebraic stabilization, not a spin expansion.
The legacy Schwarzschild inner-radius behavior remains unchanged.

## Emitters and frequency transfer

Production motion is u=ut(k+Omega m), k=(1,0,0,0), m=(0,-y,x,0) in outgoing
Cartesian KS coordinates. Omega=s sqrt(M)/(r^(3/2)+s a sqrt(M)), with coordinate
sign s selected by relative orbit direction. ut is normalized using the actual
metric. At fixed r,theta the BL-to-KS time and azimuth shifts are constant, so
these are the correct transformed Killing vectors; BL array components are
never inserted directly into Cartesian arrays.

Null rays follow the existing Hamiltonian equations. Dense-output equatorial
roots count **every** plane crossing, including holes and beyond-edge misses.
The first annulus hit is opaque, even for a zero-emissivity surface. No later
hit substitutes for it. Conservation diagnostics stop at that physical hit;
internally, solver geometry may extend farther to the ordinary domain endpoint.
Only escaping rays with no earlier hit use the existing Kerr background tail.
Exactly coplanar rays are ambiguous for an infinitesimally thin surface and
remain unresolved. Near-tangent/intersection-edge cases require convergence.

Backward covectors have the opposite sign to physical future photon momentum.
The implementation therefore uses positive nu=p.u for both the camera and
emitter, equivalently the ratio of physical -p.u frequencies. It uses the actual
normalized camera ZAMO/static/custom-frame velocity, not an observer at infinity.
Nonpositive/nonfinite frequencies are rejected, not absolute-valued or clipped.
No additional Doppler, lensing or frame-dragging factor is applied.

The rest-frame law is the existing T=T_in*(r/r_in)^(-temperature_index), default
T_in=10000K and index=0.75, with I_emit=emissivity_scale*sigma*T^4/pi. Each face
has isotropic radiance. I_obs=g^4 I_emit is bolometric; the specific-intensity
law is g^3 I_nu_emit(nu/g). Lensing changes the solid-angle mapping. It does not
add a separate intensity multiplier. Blackbody chromaticity uses g*T_emit with
the unchanged CIE palette; exposure/tone mapping remain display choices.

Holding T_in fixed across spins changes temperature at every fixed outer
radius when ISCO changes. It does not represent equal accretion rate or equal
total luminosity. This explains much of the brightness/color difference between
these deliberately matched-display previews.

## Render and reproduce

All six configs are in `configs/`, with 96×72 native resolution, r_camera=60M,
FOV=60°, one ray per pixel, blackbody palette and exposure 5. Main views use
85°; a 60° view is also provided. Negative-spin prograde and positive-spin
retrograde configs make the two rotation conventions explicit.

PowerShell:

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/kerr_disks/configs/spin09_prograde_i85.toml --workers 10 --output output/kerr-disk
.\.venv\Scripts\python -m numerical_observatory --config examples/kerr_disks/configs/spin09_retrograde_i85.toml --background grid --workers 10 --output output/kerr-disk-retro-grid
.\.venv\Scripts\python -m numerical_observatory.kerr_disk_validation --render --workers 10
```

Ubuntu:

```bash
.venv/bin/python -m numerical_observatory --config examples/kerr_disks/configs/spin09_prograde_i85.toml --workers 10 --output output/kerr-disk
.venv/bin/python -m numerical_observatory --config examples/kerr_disks/configs/spin09_retrograde_i85.toml --background grid --workers 10 --output output/kerr-disk-retro-grid
.venv/bin/python -m numerical_observatory.kerr_disk_validation --render --workers 10
```

Without `--render`, the validation module audits selected emitters/trajectories,
including the spatially rotated legacy Schwarzschild setup. `--figures-only`
only repaints saved previews and diagnostics. `--nearcritical-audit` separately
reproduces the seventh-crossing reference-precision experiment; its failed
independent coordinate criterion is explicitly saved, never converted to a pass.
Use `--output` to select a separate directory. None of these commands launches
640×480 or supersampled images automatically.

Deliberate higher resolution (PowerShell; Ubuntu changes the interpreter only):

```powershell
.\.venv\Scripts\python -m numerical_observatory --config examples/kerr_disks/configs/spin09_prograde_i85.toml --width 640 --height 480 --workers 10 --output output/kerr-disk-09-hires
```

This is much more costly than a preview. Kerr has no spherical rotational
trajectory reuse. Ten workers is the measured example setting, not a new
worker-count optimum claim; the existing automatic/serial options remain valid.

## Saved data, repainting and scientific maps

`disk_rays.npz` retains the legacy status (0 captured,1 escaped,2 unresolved,
3 emitted), g, observed bolometric intensity/temperature, hit position/radius,
plane-crossing ordinal, source directions and samples. Kerr adds hit event,
canonical covector, affine value, emitted temperature/intensity, outgoing-KS
emission azimuth, and per-ray null/E/Lz/Q diagnostics and initial constants.
Metadata contains spin, orbit, coordinate rotation, resolved radii, camera and
observer. Arrays retain every subray when supersampling is used.

Azimuth is atan2(y,x) in the outgoing Cartesian KS equatorial plane: 0° is +X,
positive toward +Y about +Z, modulo 360°. It is not BL phi, whose radial chart
shift differs. The profile is axisymmetric, so this coordinate choice changes
neither radiance nor material velocity.

```text
python -m numerical_observatory.surface_render --input output/kerr-disk --output output/kerr-disk-thermal --palette thermal --exposure 3
python -m numerical_observatory.surface_render --input output/kerr-disk --output output/kerr-disk-gray-grid --palette gray --background grid
python -m numerical_observatory.disk_diagnostics --input output/kerr-disk --output output/kerr-disk-maps
python -m numerical_observatory.diagnostic_zoom --input output/kerr-disk --output output/kerr-disk-window --crop 30 22 66 50 --density 2
```

Repainting only reads physical arrays; its sidecar points to the original NPZ,
so keep that source available. Radius/crossing/azimuth maps support Kerr and use
saved actual intersections. If radius must be recovered from positions, the
postprocessor uses Kerr r rather than a Euclidean norm. Categories and cyclic
angles are never averaged. Diagnostic zoom is an **explicit new ray trace** of
a camera-image window with the original projection; ordinary raster crops and
comparison enlargements add no angular information.

Kerr multilayer disk diagnostics remain unsupported. Schwarzschild planar
orbital angles/alignment rings also do not apply: Kerr paths need not lie in
a fixed plane. These commands reject Kerr inputs explicitly.

## Validation and limitations

See [REPORT.md](REPORT.md), [comparison](comparison.png),
[central comparison crop](central_comparison.png), [all Kerr previews](all_kerr_previews.png),
[physical measurements](physical_validation.json), [zero-spin comparison](zero_spin_comparison.json)
and [spin reversal](spin_reversal_comparison.json). Scientific maps and their
arrays are under `diagnostics/<case>/`; grid companions under `grid/<case>/`.

Validation combines BL radial geodesic equilibrium, normalization, full
Hamiltonian circular-worldline residuals, conserved emitter E/Lz, independently
located marginal stability, Carter-separated photon intersections, and invariant
frequency checks. It does not rely on attractive images. The unchanged null
threshold remains enforced; selected very high-order reference calculations
have a documented precision limit. Finite sampling, numerical tolerances and
the existing finite background tail are separate limitations. No claim is made
that a plane-crossing count equals winding number or uniquely establishes image
order. Preview maps may contain single-pixel higher-crossing features.

Sources: [Bardeen, Press & Teukolsky (1972)](https://doi.org/10.1086/151796),
[SageManifolds' circular Kerr reference](https://sagemanifolds.obspm.fr/kerrgeodesic_gw/reference/kerr_spacetime.html)
and the project's independently validated Kerr metric/Carter reference.
No new dependency, GPU backend or approximate production trajectory solver is used.

## Optional Carter backend

The existing disk pipeline also accepts --integrator carter for exact Kerr.
Hamiltonian remains the default. Both paths use the same emitter and display
physics. See [backend selection and failure policy](../kerr_carter/README.md)
and [accuracy, seventh-crossing investigation and benchmarks](../kerr_carter/REPORT.md).

