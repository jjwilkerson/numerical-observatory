# CarpetX import, Phase 1

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the reference-code audit and exact
algorithm inventory, and [VALIDATION.md](VALIDATION.md) for measured results.
This implementation loads saved data; **CarpetX, Einstein Toolkit and the C++
tracer are not runtime requirements**.

## Supported subset

The portable input is the reference tracer's **GMDS v1** directory: a
`dataset.toml` manifest and interleaved little-endian float64 raw arrays. The
optional converter reads the reference's native CarpetX **ADMBaseX Cartesian
openPMD ADIOS2 BP/BP5** exports. It is deliberately not a general openPMD reader.
All ten physical ADM fields must be present:

```text
alp betax betay betaz gxx gxy gxz gyy gyz gzz
```

The shift is contravariant; the six spatial-metric entries are covariant.
The signature is `-+++`, with `dx^i + beta^i dt`. Cartesian code coordinates,
time, original shifts and units are retained. No implicit reference-mass/SI
rescaling, chart change, reflection, ghost fill, or exterior extrapolation occurs.
Native z,y,x storage is transposed in metadata into x,y,z geometry; payloads
remain z,y,x,field with x fastest. A GMDS origin is the first retained sample;
cell centering must not be added a second time.

There must be at least **two strictly increasing saved times**. Each snapshot
must have the same ordered patch layout, levels, centering, spacing and origins.
Time interpolation is piecewise linear **in ADM fields**, followed by metric
reconstruction. A single frozen slice is rejected. Synthetic stationary examples
explicitly identify their duplicated analytical samples as stationary controls.
Dynamic regridding, moving patches, and different component time offsets are
unsupported.

Spatial modes reproduce the C++ implementations:

| `spatial_interpolation` | Behavior |
|---|---|
| `linear` | Eight-node tensor product; union of same-level patches; reference coarser fallback |
| `cubic` | 64-node centered Hermite/Catmull–Rom tensor product; no silent fallback across refinement |
| `cubic_amr_c1` | Prescribed planar two-level quintic blending |
| `cubic_amr_c1_box` | Prescribed nested-box two-level quintic blending |
| `cubic_amr_c1_hierarchy` | Certified multilevel nested rectangular ratio-2 blending |

The cubic is **not fourth-order Lagrange interpolation**. Generic value and
spatial-derivative convergence orders are three and two. Composite derivatives
include derivatives of blend weights. Hierarchy mode requires uncropped native
region provenance, exact factor-two spacing, aligned lattices, strict nesting,
and complete independent cubic stencils on every required support stratum.
No stencil is filled from a different level. See the audit for endpoint policies,
overlap tolerances and explicit planar/box/override configuration dictionaries.

The inverse 4-metric and all four derivatives follow the 3+1 inverse and its
analytic chain rule. Positive lapse and positive-definite spatial metric are
checked both on native samples and on required interpolants. Cubic overshoot is
reported rather than clipped. Active snapshot pairs are physically validated
globally, including levels not selected by a particular ray.

Optional `spacetime.backend = "numpy"` or `"numba"` accelerates this same
evaluation pipeline; `"python"` remains the default reference. See the
[optimization guide](../numerical_optimization/README.md) for installation,
backend semantics, measured performance and additional parity tests.

## Setup and conversion

Core evaluation uses existing NumPy/SciPy dependencies. Native BP conversion
alone needs the small optional `openpmd-api` extra with ADIOS2 enabled.

Windows PowerShell, from the project root:

```powershell
.\.venv\Scripts\python -m pip install -e '.[test,carpetx]'
.\.venv\Scripts\python -m numerical_observatory.convert_carpetx 'reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-source/j-native.it%08T.bp5' output/my-carpetx-gmds my-carpetx
.\.venv\Scripts\python -m numerical_observatory --config examples/numerical_spacetime/real_preview.toml --workers 2 --output output/numerical-grid
```

Ubuntu:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,carpetx]'
.venv/bin/python -m numerical_observatory.convert_carpetx 'reference/geodesics-cpp/results/amr-composite-multilevel/carpetx-native-source/j-native.it%08T.bp5' output/my-carpetx-gmds my-carpetx
.venv/bin/python -m numerical_observatory --config examples/numerical_spacetime/real_preview.toml --workers 2 --output output/numerical-grid
```

Change `spacetime.dataset` to use a different converted directory. Relative dataset
paths resolve against the configuration's directory. The converter refuses an
existing destination; `--crop XMIN XMAX YMIN YMAX ZMIN ZMAX` retains physical
samples without inventing ghost cells. Cropping can invalidate AMR support and
is checked when constructing the provider. Incomplete conversions have no
`dataset.toml` and cannot be loaded as successful datasets.

Use `--background starfield` to render the photographic sky, or reuse directions
without retracing:

```python
import numpy as np
from numerical_observatory.background import Starfield
from numerical_observatory.diagnostic_render import shade, save_image
with np.load('output/numerical-grid/rays.npz') as rays:
    save_image('output/numerical-starfield.png',
               shade(rays['directions'], rays['status'], Starfield(), int(rays['samples'])))
```

The bundled preview is only **24×18 actual rays**, magnified with nearest-neighbor
sampling for presentation. It uses real evolved three-level data and an outward
local view. It is a finite-boundary lensing demonstration, **not a calibrated
black-hole shadow or distant astronomical observation**. No physical horizon or
vacuum exterior has been supplied for this dataset.

## Domains, integration and diagnostics

The same generic Hamiltonian RHS is stepped using SciPy DOP853. A numerical-only
adapter guards every RK and dense-output stage inside one saved temporal slab,
lands on knots without momentum projection, then uses the next one-sided
derivative. The C++ reference uses DP5(4); converged common-state trajectories
are compared rather than requiring identical adaptive steps.

The default numerical domain has **no capture or escape surfaces**. Missing
stencils, spatial or temporal data exits, invalid fields, failed integration and
constraint failures are unresolved, with separate reason codes. They are not
interpreted as horizons or escape to infinity. Optional `capture_spheres` are
explicit user-supplied absorbers with `center`, `radius`, and `description`; only
an independent physical justification would make one a horizon approximation.

The opt-in `finite_direction_sphere` returns a normalized Cartesian ray tangent
at the configured finite boundary. Changing its radius changes the observation;
there is no analytical tail or asymptotic-flatness assumption. Establish boundary
convergence or a physical matching prescription for your own dataset. Carter,
Schwarzschild quadrature, rotational reuse and disks are rejected.

`rays.npz` preserves status/directions and adds null residual, selected level
counts, finest selected level, supplied capture index, temporal slabs, stage
rejections, diagnostic-completeness flag and termination codes. `diagnostics.json`
contains the codebook, hierarchy, interpolation, source identity, manifest SHA256,
time coverage, lookup failures, tolerances and timing. `numerical-dataset.toml`
is a provenance copy; payloads stay in the input dataset. Supersampled arrays
retain subpixel ordering. No stationary energy invariant is assumed.

Immutable read-only memory maps reopen in Windows `spawn` workers; payloads are
not serialized into each worker. The OS may share file-backed pages. Physical
validation and interpolation scratch arrays remain per-process. The default
1 GiB budget applies to the **sum of two logical snapshot payloads**, not peak
RSS; it is intentionally conservative even when stationary snapshots share files.

## Reproducing validation

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m numerical_observatory.numerical_validation --output output/numerical-validation --analytic --demo --workers 2
.\.venv\Scripts\python -m numerical_observatory.numerical_experiments --output output/numerical-validation --temporal --amr --performance
# Real-data boundary/tolerance check and tiny serial/spawn comparison:
.\.venv\Scripts\python -m numerical_observatory.numerical_experiments --output output/numerical-validation --domain --parallel
# Optional, slower subpixel shadow brackets on generated analytical grids:
.\.venv\Scripts\python -m numerical_observatory.numerical_experiments --output output/numerical-validation --shadow
```

On Ubuntu replace `.\.venv\Scripts\python` with `.venv/bin/python`. Synthetic
datasets are generated locally with explicit analytical or manufactured provenance;
existing directories are reused, never overwritten by the writer.

Portable regression tests use saved C++ metric/derivative golden values. To rerun
the live independent comparison, compile `tools/numerical_cpp_parity.cpp` against
the copied reference headers/library and configure an argument-list command:

```bash
g++ -std=c++20 -O2 -Ireference/geodesics-cpp/include tools/numerical_cpp_parity.cpp reference/geodesics-cpp/build-numerical-release/libgeodesic.a -o output/numerical_cpp_parity
```

Ubuntu oracle JSON: `{"command":["/absolute/path/output/numerical_cpp_parity"]}`.
The included `oracle_windows.json` uses the local WSL build and explicit repository
path mapping; edit both absolute roots for another machine. Then:

```powershell
.\.venv\Scripts\python -m numerical_observatory.numerical_validation --output output/numerical-parity --oracle examples/numerical_spacetime/oracle_windows.json
```

The reference archive is optional for normal use, but required for live C++ and
real-data checks. Its copied sources remain unmodified. The native converter
test skips explicitly if the optional binding or independent fixture is absent.

Future dynamic-AMR work needs per-snapshot changing geometry, revised common
support/temporal policies, new certification and cross-language tests. Current
metadata retains the hierarchy separately for each snapshot without claiming
support for that evolution.
