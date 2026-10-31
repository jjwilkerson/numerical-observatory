# Binary KS approximation: implementation plan

Use g=eta+sum_A (2 m_A/r_A) l_A l_A, l_A=(1,-n_A),
with r_A=|x-c_A|. Sum the covariant terms only. Invert the complete 4x4
matrix, analytically differentiate each covariant term and use
partial(g^-1)=-g^-1 (partial g) g^-1. Time derivatives vanish.

Only the generic Hamiltonian path is permitted. Reuse Camera, backgrounds,
spawn workers and existing status codes. Add a binary policy with union of two
provisional capture spheres, a barycenter-centered escape sphere and a metric
signature validity boundary. Record capture identity and sampled validity/null/
stationary-energy diagnostics separately; unresolved never means captured.

For two nonnegative masses, det(g)=-D, where
D=1-f1*f2*(1-n1.dot(n2))^2 and fA=2mA/rA. The spatial block is positive
definite, so D>0 is equivalent to Lorentzian signature. D can vanish in strong
overlap regions. A conservative positive margin will terminate before this
boundary; numerical stage failures must be unresolved, never provisional hits.
An isolated radius 2mA is NOT a binary event horizon. Capture spheres are
explicit model cutoffs. Static observers require g_tt<0 at the camera.
No orbital motion or vacuum/constraint-solved binary interpretation is implied.

Background directions use the normalized outgoing Cartesian tangent at the
finite escape sphere. Increase the sphere radius to assess remaining bending;
no spherical analytic continuation, Carter solver or rotational reuse is used.
Single-hole trajectory checks use matched finite termination conventions, plus
comparison of source directions with the existing infinity-map as radius grows.

Configuration mass is the reference length/time scale M_ref; spacetime.masses
and centers, camera lengths, and integration lengths are in that scale. The
96x72 example has masses 0.5+0.5 and separation 12 in total-mass units.
Do not change existing Schwarzschild/Kerr equations, conventions or defaults.
