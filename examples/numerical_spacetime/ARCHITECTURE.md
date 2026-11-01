# CarpetX Phase 1: architecture and parity audit (before implementation)

## Audited reference inventory

Reference is reference/geodesics-cpp (read-only; no Git commands).
- include/geodesic/metrics/numerical.hpp and src/metrics/numerical.cpp: GMDS
  validation, two-snapshot cache, five spatial modes, ADM/time interpolation.
- detail/cubic_weights.hpp: centered-slope Hermite value/gradient weights.
- detail/box_amr_weight.hpp: geometric branches, symmetric quintic weights.
- detail/coverage_boxes.hpp: exact integer half-open box-union subtraction.
- integrator.cpp, metric.hpp, docs/temporal-boundaries.md: guarded temporal slabs.
- tools/convert_openpmd.cpp: native ADMBaseX openPMD ADIOS2 -> GMDS v1.
- tests/numerical,cubic,composite,box_composite,hierarchy,temporal_boundaries,
  temporal_force.cpp and their docs: numerical acceptance definitions.
- tools/probe_numerical, sample_adm, inspect_amr_hierarchy: independent probes.
- tools/synthetic_dataset.hpp, hierarchy_dataset.hpp and saved results: analytic
  and manufactured grids, native-region provenance, genuine evolved CarpetX.

Copied binaries are Linux ELF, not Windows executables. WSL Ubuntu and GCC 11.4
are available with host execution permission. Cross-language audit can compile a
small standalone probe against the COPIED core library, without modifying or
linking it into Python. No CarpetX simulation is needed or will be launched.

## File and coordinate contract

Production evaluator reads GMDS v1: dataset.toml + little-endian binary64 patch
files, no headers. Fields EXACTLY `alp betax betay betaz gxx gxy gxz gyy gyz gzz`.
Each node is ten interleaved values; x fastest, then y, then z. Manifest shape,
origin and spacing are x,y,z. Origin is FIRST STORED SAMPLE, not a cell face.
Centering vertex/cell is explicit; no extra half-cell correction is applied to
GMDS coordinates. Native CarpetX storage axes are z,y,x; converter reverses them
and incorporates record position 0 or 0.5. Fields are physical ADM lapse, upper
shift, symmetric lower spatial metric, not Z4c conformal/evolved deviations.
Signature -+++, Cartesian t,x,y,z, ADM dx+beta*dt. Code coordinates/fields are
retained; unitSI is provenance only. No implicit mass/SI rescaling or symmetry
extension. Positive lapse/SPD gamma and all native samples are validated.

Source backend is openPMD ADIOS2 BP5, not HDF5. Optional Python openpmd_api
conversion will mirror the reference's ADM groups/chunks/centering checks; core
GMDS evaluation needs only existing NumPy/SciPy. Already converted complete
CarpetX datasets are available in the copied tree. Missing shift is NOT zero.

At least two ordered snapshots are required. Phase 1 rejects ANY changing patch
partition/geometry, even where historical C++ linear/cubic allowed it. This is
the deliberate fixed-hierarchy subset, not a new dynamic-AMR implementation.
Provenance native_regions is distinct from stored/cropped patch bounding boxes.

## Exact spatial algorithms and boundary rules

Linear: candidates include [origin,origin+n*h], stable finest-first. floor(u),
with upper-cell clamp to n-2 only when u<=n-1. Gather eight actual nodes across
same-level patches. Missing full stencil tries the next candidate/coarser level.
Overlap nodes must satisfy abs(new-old)<=1e-11*(1+abs(old)); last value wins.
No exterior extrapolation, ghost generation, reflection or implicit masks.

Cubic: complete 4^3 nodes at floor(u)+[-1,0,1,2], centered Hermite slopes
(f[i+1]-f[i-1])/2, (f[i+2]-f[i])/2. Per-axis weights:
[-t/2+t^2-t^3/2, 1-2.5t^2+1.5t^3, t/2+2t^2-1.5t^3, -t^2/2+t^3/2].
Spatial derivatives are analytic derivatives of THESE weights divided by h;
not interpolated FD derivatives. Highest candidate alone is selected; incomplete
stencil or coarse stencil intersecting finer stored coverage fails, NO fallback.
Ordinary cubic is not generally C1 across refinement boundaries. Complete
single-rectangle support is [origin+h,origin+(n-2)h), upper excluded.

Planar/box C1: independent complete level cubics, no refinement-intersection
rejection. Blend ADM fields BEFORE 3+1 inversion; derivative includes
(dw)*(fine-coarse). Box weight is product of three axis smooth steps with
symmetric quintic 1-u^3(10-15u+6u^2), evaluated using v=1-u on upper half.
Geometric outside/core/shell determines required levels, NOT rounded weight==0.
Planar closed endpoints require both levels; box outer endpoints require coarse
only and plateau endpoints fine only. Reference restricts composites to ratio 2.

N-level hierarchy: contiguous level IDs, exact ratio-2 spacing, aligned lattices,
fixed ordered patches, strictly nested rectangular native_regions with explicit
analytic_sampling/carpetx_valid_chunks provenance. Canonical anchor is lexicographic
minimum origin then shape, independent of patch input order. Unlike legacy
modes, hierarchy uses un-clamped floor arithmetic. Automatic fine bands:
outer_lo=native_lo+2*h_f, outer_hi=native_hi-2*h_f;
inner_lo=outer_lo+h_coarse, inner_hi=outer_hi-h_coarse.
Recursive blend prunes suppressed parent levels on plateaux; overlapping shells
retain every required ancestor and all gradient product terms. Exact endpoint/
open-interval strata and integer-box union subtraction certify required stencil
support, including holes/seams. No midpoint sampling substitutes for this proof.

Port all five spatial modes; rectangular multilevel hierarchy is the principal
parity target. Floating accumulation order may differ; weights/index choices and
failure policies must not. Geometry metadata proof must remain conservative
across Windows (longdouble=binary64) and Linux extended precision; runtime
stencils are always checked exactly. Record any certification differences.

## Temporal/metric reconstruction and integration mapping

Linearly interpolate the independently spatial-interpolated ten ADM fields in
time; dt field derivatives are endpoint differences/dt, spatial derivatives
are temporally blended. Reconstruct inverse using alpha,beta,gamma inverse:
g00=-1/alpha^2, g0i=beta^i/alpha^2,
gij=gammaInv_ij-beta^i*beta^j/alpha^2. Differentiate chain rule analytically.
Standalone knot evaluation is right-sided; a requested closed slab evaluates
its own one-sided derivatives. First/last endpoints use adjacent available slab.
No stationary p_t assumption. Cubic overshoot can invalidate lapse/SPD even
when all stored nodes are valid; reject, never clip or choose lower order.

C++ DP5(4) guards every trial and dense/event reintegration against crossing
snapshot knots, reduces attempted steps by actual stage time displacement,
lands within 8eps*max(|slab times|) (capped at width/8), snaps time only, and
switches derivative side on the next step. Observatory retains SciPy DOP853:
add a numerical-only guarded SciPy stepping adapter using the SAME Hamiltonian
RHS, closed-slab provider and guarded dense stages. Do not change analytical
solver paths. This is necessary infrastructure, not an alternative geodesic
model. Compare ODE methods by tightening tolerances at matched affine values;
DP5 and DOP853 accepted-step counts must not be equated.

Python modules: separate GMDS metadata/mmap loader, AMR weights/coverage,
numerical metric, numerical domain/slab adapter and validation/conversion tools.
Map C++ InvalidMetric/DomainExit to typed numerical lookup failures and explicit
unresolved reasons. Last valid accepted state remains available. Dataset exits
are NOT infinity escape; no horizon is inferred from lapse/puncture failure.
Only explicitly configured, tested finite direction-sphere backgrounds and
optional independently justified capture spheres can give physical-model labels.
Default numerical exits/affine limits are unresolved. Reject disks/Carter/
Schwarzschild quadrature/spherical caching. Camera is existing metric-based frame.

Important incompatibilities: C++ analytic KS is INGOING, Observatory analytic KS
is OUTGOING. Imported ADM shift is NEVER flipped. Generate synthetic arrays in
specified chart, compare the C++ numerical evaluator in that same chart.
C++ default photons future-directed/Eulerian/Gram-Schmidt, Observatory default
past-directed/static/symmetric frame. Compare explicit common canonical states
and supplied observer frames, not different default pixels. Absolute 2H in C++
and normalized null residual in Python have different definitions.

## Validation, scope of parity, implementation order

1. Immutable GMDS load, corruption/physical field checks, read-only mmap and
   fixed geometry validation; manifest/source identity and coverage diagnostics.
2. Exact box/weight/interpolation/chain-rule port with manufactured nodal values,
   endpoints, 1-ULP cases, seams, holes, overlaps, invalid cubic overshoot and
   multiple temporal slabs. C++ tolerances: polynomial values ~3e-14, ordinary
   derivatives ~3e-13; manufactured hierarchy inverse FD ~3e-9. These bound
   specific tests, not a universal data/trajectory tolerance.
3. Batch C++ field/metric/derivative/selection probes for synthetic and real data,
   direct raw binary comparisons. Add validation bridge only outside reference.
4. Snapshot-safe generic Hamiltonian integration and explicit numerical domain;
   exact time-varying lapse forward/backward tests and failed-stage classification.
5. Schwarzschild then Kerr grid-resolution/refinement/trajectory studies; exact
   boosted Schwarzschild temporal study. Separate field, ODE, boundary and raster
   error. Small real evolved three-level finite-boundary demonstration if native
   sample validation and C++ comparison succeed. No NR simulation.
6. Full existing suite, serial/spawn equality, timing/load/memory measurements,
   optional BP conversion roundtrip and reproducible docs/report.

Feature parity means GMDS v1 field reconstruction, all five documented static
interpolation modes (particularly certified nested rectangular ratio-2 AMR),
closed slab linear temporal support, failure policy and converged common-state
trajectories within stated tolerances. It does NOT mean DP5/DOP853 bit equality,
dynamic AMR, arbitrary openPMD backends, physical puncture horizon discovery,
Eulerian/static camera equivalence, or asymptotically exact skies.

Risks: 80 bytes/node/snapshot and two active snapshots, cubic stencil lookup over
many patches, eager native physical validation, repeated metric calls and knot
approach rejections. Mmap files read-only and reopen mappings in spawn workers
rather than pickle payloads; OS pages can be shared. Match the reference budget
by active-pair byte accounting. Profile first; no compiled Python acceleration.
Large studies/native BP imports remain opt-in; compact synthetic fixtures and
existing converted native chunks provide normal regression coverage.
