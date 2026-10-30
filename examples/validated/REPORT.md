# Schwarzschild lensing validation

**The broad outer arcs are consistent with the computed geodesics.** No
deflection rescaling, lensing cutoff, screen-space distortion, or blending to
an unlensed image was introduced. Composition remains r=400M, horizontal
FOV=32°, 640×480, centered and coordinate-static.

[Starfield](starfield.png) · [Celestial grid](grid.png) ·
[Inner grid close-up](inner_grid.png) · [Grid coordinate atlas](grid_atlas.png) ·
[Machine-readable measurements](validation.json)

Both full-frame images use **the same saved photon directions**, with 4×4
subpixel samples averaged in linear RGB. The grid is an analytic celestial
background: cool meridians, warm parallels every 15°, gray minor lines every
5°. Line colors and shaded cells identify corresponding sky regions. It is
not an overlay. The close-up traces fresh rays through the central 48×48-pixel
camera window at 16 times the angular sampling; its camera geometry is unchanged.

## Shadow

With G=c=1, b_crit=3√3 M=5.1961524227M. For a static observer outside 3M,
sin(alpha)=b_crit sqrt(1−2M/r)/r. Pinhole image radius is
(width/2) tan(alpha)/tan(FOV/2).

- Expected angular radius: **0.7424517201°**.
- Expected image radius: **14.46183609 pixels**.
- Last captured / first escaping subpixel radii: **14.45791306–14.46655626 pixels**.
- Area-equivalent sampled radius: **14.45305721 pixels** (0.061% low from rasterization).
- Independent ODE boundary bracket: **0.7424517201–0.7424517484°**.

The image's classification uses the analytic Schwarzschild separatrix, so its
agreement alone is not an independent physics test. ODE angular bisection supplies
that independent check. Bisection deliberately disables the new reliability
guard to measure the raw solver separatrix at rtol=1e−11; ordinary rendering keeps
the guard enabled. The pixel bracket reflects finite sampling, not solver error.

| Camera r/M | Expected angular radius | Radius at 32° FOV | Radius at 60° FOV |
| --- | --- | --- | --- |
| 20 | 14.26902733° | 283.81548 px | 140.95911 px |
| 100 | 2.94855520° | 57.48093 px | 28.54834 px |
| 400 | 0.74245172° | 14.46184 px | 7.18258 px |

All six ODE-derived pixel brackets contain or meet these analytic values within
the reported bisection resolution. Distance changes angular size; FOV changes
its pixel projection, not the physical critical impact parameter.

## Trajectories and convergence

The audit covers capture, near capture, five escaping impact parameters near
b_crit, rays inside/on/outside the main outer arc, a frame corner, and a weak
deflection control outside the frame. All escaping paths have radial turning
points. It compares unit directions with atan2(|a×b|, a·b), not subtracted
longitude values. It also compares **unwrapped accumulated azimuth**, including
the escape-to-infinity tail. Maximum phase step in the tight ODE trajectories
is 0.073 radians, safely below the unwrap ambiguity. Turning radii agree within
2.9×10⁻⁹ M at the tightest tolerance.

Absolute tolerance is rtol/100. Representative raw ODE direction errors in radians:

| b/b_crit−1 | Accumulated azimuth / 2π | rtol 1e−8 | rtol 1e−10 | rtol 1e−12 |
| --- | --- | --- | --- | --- |
| 1e−2 | 1.1711 | 3.09e−8 | 3.99e−10 | 4.19e−12 |
| 1e−4 | 1.9002 | 1.38e−6 | 1.72e−8 | 2.26e−10 |
| 1e−6 | 2.6330 | 1.31e−4 | 1.58e−6 | 3.70e−8 |
| 1e−8 | 3.3660 | 1.31e−2, rejected | 1.75e−4 | 2.42e−6 |
| 1e−10 | 4.0989 | false capture, rejected | 1.67e−2, rejected | 2.34e−4 |

The outer alignment-ring direction agrees to 5.1×10⁻¹⁴ rad at rtol=1e−12;
the frame-corner direction agrees to 2.0×10⁻¹³ rad. The unwrapped phase errors
match the directional errors: no hidden extra/missing 2π winding was detected.

**A demonstrated error was corrected:** the coarse ODE could falsely report
capture for b/b_crit−1=1e−10 despite satisfying the old null threshold. An
empirical near-critical guard now compares the initial distance from the
separatrix with sampled impact-parameter drift, normalized null residual, and
a roundoff floor. Uncertain rays become unresolved. Raw outcomes remain in the
audit JSON for transparency. This guard is not a mathematical error bound and
does not certify accurate deflection for every accepted near-critical ray.

The quadrature uses a stable, positive-term turning-point factorization. Tightening
its integration tolerance from 1e−8 to 1e−12 changes sampled phases by less than
8×10⁻¹⁴ rad, but that is **not** total accuracy: at b/b_crit−1=1e−10 a one-ULP
perturbation of the initial angle changes phase by about 1.1×10⁻⁶ rad. Double
precision and the ODE's accumulated error limit extreme-winding comparisons.
No convergence claim is made arbitrarily close to the critical ray.

## What the rings mean

For this camera the primary alignment condition (accumulated azimuth π) lies
at **5.938829°**, or **116.08898 pixels** from the center: about eight shadow
radii. Light from near the direction directly behind the hole is strongly
stretched tangentially there. Extended stellar texture produces conspicuous
arcs. It is an Einstein-ring scale, not the shadow boundary or an accretion disk.
The grid shows corresponding meridians wrapping around this same region.

Inside that ring, the background contains inverted and increasingly compressed
images of the sky. A quiet-looking patch is not weakly lensed: brightness depends
on which source directions land there, not on deflection magnitude alone.
Near the shadow, rays make additional circuits and successive images accumulate.

| Alignment phase | Image radius | Distance outside shadow |
| --- | --- | --- |
| π | 116.08898 px | 101.62714 px |
| 2π | 14.90316 px | 0.44132 px |
| 3π | 14.47977 px | 0.01794 px |
| 4π | 14.46261 px | 0.00077 px |

Even and odd alignments refer to opposite directions on the sky. These are
geometric alignment locations, not guaranteed bright, filled rings for every
background. The higher orders cannot be resolved as separate rings at 640×480;
even 4×4 subpixel sampling cannot recover such tiny radial structure. The fresh
close-up resolves more secondary-image structure, but its finest dotted/moiré
features also remain sampling-limited. The lack of a conspicuous ring beside
the shadow is therefore expected.

This interpretation agrees qualitatively with the inverted-sky description in
[NASA's isolated-black-hole visualization](https://svs.gsfc.nasa.gov/14620/).
That visualization explicitly uses simplified optics; it was not used as a
numerical reference. The analytic shadow relation is also discussed in the
[Perlick–Tsupko review](https://arxiv.org/abs/2105.07101).

## Projection and sampling limits

The star map is sampled on the celestial sphere with a fixed orthogonal sky
rotation, periodic longitude, clamped latitude, and bilinear interpolation in
linear RGB. There is no fisheye projection in the camera, radial image mask,
screen-space ring, or lensing cutoff. Both the photographic map and analytic
grid show the same mapping structure. Periodic-seam and pole-finiteness tests
pass. A rasterized equirectangular texture has finite resolution, especially
problematic near its poles and under large magnification; it does not represent
true point sources or calibrated stellar flux. Display clipping and the finite
angular footprints of atlas stars affect arc appearance and brightness.

Against 4×4 sampling, full-image mean absolute display-RGB error falls from
0.00639 to 0.00154 for the starfield and from 0.00188 to 0.00065 for the grid
when going from one to 2×2 samples. In the near-shadow annulus (shadow edge to
25 px), residual differences remain much larger: 0.0257 for stars and 0.0725
for the grid at 2×2 samples. Four samples per axis are a useful improvement,
not a claim of full image convergence. The broad outer arcs survive these
sampling changes; some fine glitter, breaks, and moiré do not.

## Scope and reproducibility

The camera now constructs a rest-space tetrad from the supplied metric rather
than a Schwarzschild radial formula; a user-supplied moving tetrad is also
supported. The ODE takes explicit `TraceDomain` boundaries/source-direction
mapping for other metrics, including numerical-data exit as unresolved. It
does not infer a spherical horizon or infinity map. Time-dependent-metric tests
verify changing p_t without incorrectly treating it as a conserved quantity.

Quadrature and rotational reuse require the exact Schwarzschild implementation,
a centered coordinate-static camera, and compatible boundaries. Subclasses,
off-center/moving observers and custom domains do not activate them. The main
renderer falls back to ODE; the low-level quadrature routine refuses incompatible
inputs. No Kerr or CarpetX metric was added or claimed tested.

Run `python -m numerical_observatory.lensing_validation` after installation to
regenerate this folder's images and JSON. Run `python -m pytest -q` for regression
tests. **28 tests passed** on Windows with Python 3.13; both ODE and quadrature
grid CLI smoke renders also passed. The full 4×4 matched render contains zero
unresolved subpixel rays. See the root README for Windows/Ubuntu commands and
background selection. Ubuntu execution was not available in this environment.
