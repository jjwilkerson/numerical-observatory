# Luminous Schwarzschild disk: validation and reproduction

The optional disk is disabled by default. Existing validated starfield and grid
outputs are preserved. These examples use a static camera at (0,0,60M), horizontal
FOV 60°, 192×144 pixels, and a disk spanning 6–20M. Inclinations are 60° and 85°.
The larger FOV and closer camera are separate disk examples, not a replacement
for the earlier narrow-field starfield composition.

| View | Starfield | Celestial grid |
|---|---|---|
| 60° disk | [Image](inclined_starfield/shadow.png) | [Image](inclined_grid/shadow.png) |
| 85° disk | [Image](edge_on_starfield/shadow.png) | [Image](edge_on_grid/shadow.png) |
| No disk | [Image](no_disk_starfield/shadow.png) | [Image](no_disk_grid/shadow.png) |

Each pair uses identical camera settings. Disk/grid pairs reuse exactly the same
saved surface intersections, frequency shifts and escaping sky directions.
The bright upper far-side arc and narrow lower image are geodesic images of the
emitting surface. Brightness asymmetry follows the orbiting material's motion;
reversing rotation reverses the asymmetry. No glow, artificial deflection cutoff,
or luminous screen-space ring is used. An opaque disk can cover part of the
diskless capture silhouette; the remaining dark region is not itself an
independent measurement of the critical impact parameter.

## Physics and conventions

Units for trajectories are G=c=1. The disk is an infinitesimally thin two-sided
opaque annulus in a plane through the center. Material follows circular timelike
Schwarzschild geodesics: Omega=±sqrt(M/r³), u^t=(1-3M/r)^(-1/2), and
u^i=u^t Omega (normal cross position)^i. These expressions also hold in our
Cartesian Kerr–Schild coordinates because dr=0 along a circular orbit. Circular
orbits at 3M<r<6M are unstable; r<=3M is rejected for this timelike emitter model.

The integrator carries past-directed photon covectors. Their positive measured
frequency is p_mu u^mu; a future-directed physical photon has the opposite
covector and frequency -p_mu u^mu. Taking the ratio at observer and emitter
therefore gives the requested positive g without an absolute-value sign fix.
The observer frequency is evaluated with the camera's tetrad time leg.

Rest-frame emission is isotropic Planck radiance with T=10000 K (r/6M)^(-3/4)
and a configurable nonnegative emissivity multiplier. The stored bolometric
radiance is g⁴ times sigma T⁴/pi in W m^-2 sr^-1. The spectral helper separately
uses g³ I_nu(nu_obs/g). This is the usual invariant-intensity transfer law; see
[the covariant radiative-transfer formulation](https://doi.org/10.1051/0004-6361/202140682).
There is no additional Doppler multiplier on top of this transformation.

Thermal RGB samples Planck radiance at 650, 550 and 450 nm at observed temperature
gT, normalizes the palette, and applies a separately configurable exposure and
Reinhard tone map to bolometric brightness. This is illustrative RGB, not
instrument-calibrated photometry. Display processing occurs per subray before
display-linear averaging; use stored raw intensities for quantitative flux
integration. The photographic sky has independent exposure and is not calibrated
to the disk's absolute radiance.

## Intersections and numerical checks

Production disk rays use DOP853 Hamiltonian ODE integration with rtol=1e-9,
atol=1e-11 and max_step=4M. Dense-output roots of the plane equation are sorted
in affine order. Crossings through the radial hole or beyond 20M do not absorb
the ray. The first crossing in the emitting annulus terminates its radiative
history. Higher images come from different rays, not accumulated emission
through an opaque disk. Returned surface trajectories and null diagnostics stop
at that hit. Internally cached geometry can extend farther for reuse, but no
later light contributes.

Rotational reuse is restricted to the exact Schwarzschild metric and a centered
coordinate-static camera with no custom trace domain. It rotates positions and
spatial covectors; each ray gets its own disk intersection and emitter velocity.
Other cameras use independent ODE trajectories. Production orbital quadrature
does not provide the required dense states and is explicitly bypassed for disks.

An independent audit integrates the Schwarzschild orbital quadratures, solves
plane intersections on both radial branches and reconstructs time, affine
parameter and photon momentum. Its scope is inward, escaping rays with the
observer outside the annulus. Thirteen cases include misses, direct hits,
secondary hits, turning points and near-critical rays whose first valid hit is
the fourth plane crossing. Captured-trajectory disk hits are tested separately
with ODE integration. See [full ray audit](validation.json).

At rtol=1e-11, the largest ODE/quadrature differences across the audit are:

| Quantity | Maximum absolute difference |
|---|---:|
| Hit position | 2.07e-7 M |
| Coordinate time | 3.11e-7 M |
| Affine parameter (camera-normalized momentum) | 2.22e-7 M |
| Covector Euclidean norm | 2.64e-8 |
| Frequency ratio g | 9.11e-9 |

Near-critical multiple-winding rays dominate these maxima. Ordinary intersections
agree much more closely. The regression suite tests circular normalization and
geodesic acceleration, stationary and moving-emitter shifts, photon sign
conventions, spectral/bolometric transformations, repeated crossings, opacity,
turning rays, rotation reversal, and bitwise disk-disabled image compatibility.

Neither full example has unresolved rays; maximum sampled relative null error
through the first hit or escape/capture is 4.56e-8. The inclined view has 6,054
emitting rays (6,016 first-crossing, 34 second-crossing, 4 third-crossing) with
g=0.510–1.254. The edge-on view has 2,034 (1,666 first, 368 second), g=0.489–1.422.
Crossing index counts plane crossings, including non-emitting holes; it is not
the number of complete orbits.

The [sampling audit](sampling_validation.json) compares 32×24, 64×48 and 192×144
using solid-angle-weighted raw bolometric radiance. At 64 pixels, total flux
differs from the 192-pixel reference by 0.43% (60°) and 2.08% (85°), versus 10.27%
and 9.22% at 32 pixels. Tightening rtol from 1e-9 to 1e-11 and halving max_step
on the 32×24 scenes changes no classifications or crossing orders; maximum g
change is 1.24e-9 and relative intensity change is 2.14e-8. The script asserts
these checks and a 5% flux/coverage acceptance bound for 64 versus 192 pixels.
This is evidence of integrated-observable stability, not proof that narrow
higher-order features are spatially resolved. Their coverage changes appreciably
at coarse sampling; 192 pixels remains a compact demonstration.

## Limits and reproduction

There is no disk thickness, atmosphere, limb darkening, absorption spectrum,
scattering, self-gravity, stress prescription, returning-radiation heating or
GRMHD. The simple temperature law does not impose a zero-torque ISCO boundary.
Root detection brackets accepted steps and midpoints; very rapid crossings
require tighter tolerances and smaller max_step. Exactly coplanar rays are
reported unresolved because an infinitesimally thin surface has no isolated
intersection there. The examples deliberately use 85°, not exactly 90°.

After the README's Windows or Ubuntu installation, run from the project root
with that environment's Python:

```text
python examples/render_disks.py
python -m numerical_observatory.disk_validation
python examples/validate_disk_sampling.py
python -m pytest -q
```

Configs: [inclined](../disk_inclined.toml), [edge-on](../disk_edge_on.toml).
For a single grid render add `--background grid` to the normal CLI; `--no-disk`
disables emission. The comparison script creates the six image directories
above; each has numerical diagnostics, and starfield disk directories also
contain compressed ray data shared by their grid companion. No new dependencies
or OS-specific implementation paths were introduced.
