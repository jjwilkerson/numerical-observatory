# Carter backend architecture (before implementation)

Baseline: six fixed Kerr rays, CPU cProfile, baseline_profile.txt. Of 1.49 s,
1.24 s is in Hamiltonian RHS evaluation; 0.67 s is inverse-metric derivatives
and 0.26 s inverse metrics (overlapping cumulative times). 9,569 RHS calls
include 617 large-radius tail calls. Tensor allocation/geometry evaluation is
the principal target. Camera and emission physics are retained verbatim.

Planned specialized path:

- Dispatch only exact Kerr through Integration.integrator, Hamiltonian default.
  Existing shared RayResult and SurfaceHit feed existing renderers and workers.
- Same initial camera covector; derive E,Lz,Q and exact local initial derivatives.
- Normalize lengths by M and energies by |E|. Use u=M/r and mu=cos(theta),
  evolving their second-order Mino velocities with the exact affine/Mino
  Jacobian. Affine stepping retains the existing max_step meaning. The quartic
  potentials are well-scaled near the camera, unlike r and r² radial velocities.
- Integrate unwrapped Cartesian-KS azimuth and KS time, with algebraically
  horizon-regular separated expressions for past capture. Reconstruct full
  Cartesian events and covectors. Monitor potential constraints AND actual
  metric-null and reconstructed E/Lz/Q, not constants by assertion.
- Reuse the existing dense-output plane-crossing and opaque-first-hit logic.
  Do not assume a fixed orbital plane or spherical rotational symmetry.
- Beyond the outward escape surface, integrate the exact separated angular
  tail in monotone inverse radius to the SAME finite source radius. No empirical
  bending correction; finite-radius truncation remains separately measurable.
- Unsupported/ill-conditioned axes, zero Killing energy, failed constraints or
  unreliable near-critical outcomes are explicit unresolved rays. An optional
  configured Hamiltonian fallback genuinely retraces, records the reason and
  final backend, and is excluded from Carter validation successes.
- Validate transformations and intermediate velocity/Hamiltonian consistency,
  first through fourth and seventh crossings, finite-camera shadows, shared
  images/data, serial/spawn/supersampling. Compare timing only after accuracy.
- Preserve the old second-order reference and its failed seventh-crossing
  record. Investigate its radial constraint drift separately from polar phase,
  roots and coordinates. Independent separated quadrature may be needed for
  the single extreme probe; it is validation, not another production backend.

No metric, camera, disk dynamics, emission, palette or tolerance relaxation is
part of this change. No Git commands, new compiled code or acceleration library.
