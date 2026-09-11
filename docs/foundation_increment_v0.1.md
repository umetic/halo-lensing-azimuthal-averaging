# Foundation increment v0.1.0

## Scope and status

This increment implements Public Numerical Specification v0.1, Sections 3--4:
physical cosmology/normalization, robust spherical NFW functions, and
volume-preserving triaxial projection. It is a development increment of the
paper-specific reproduction package, **not a validated replacement of the full
publication calculation**.

There is no ring-field solver, population sampler, Colossus concentration
backend, angular mask, alignment calculation, or plotting layer in this
increment. The manuscript and frozen publication outputs are unchanged.
The new implementation still requires local validation in the author's Python
3.11 environment before adoption there. No public license is assigned here.

## Dependencies and execution

Use Python >=3.11 and the already prepared project environment. This increment
imports NumPy and SciPy; tests use pytest. No additional package installation,
editable install, Astropy import, or Colossus import is needed for this step.
`pytest.ini` puts this checkout's `src/` on the test import path.

```bash
python scripts/check_foundation.py
```

The runner uses the same interpreter for pytest, verifies module import paths,
and writes its log and environment/source-fingerprint record under
`outputs/foundation_validation/`. It does not modify Conda, Git, or user aliases.
The local interpreter and path metadata in those reports should not be copied
into the public repository; the generated `outputs/` directory is ignored.

## Public computational interfaces

### cosmology.py

`PublicationCosmology` / `PUBLICATION_COSMOLOGY` provide scalar distances and
source efficiencies, and broadcasting density/r200c functions. Finite-source
operations require z_s > z_l >= 0. Critical surface density requires z_l > 0.
The far-background distance is evaluated by integrating from a=0, rather than
using a large finite redshift.

Preserve the two audited conventions: matter+Lambda for halo critical density,
and radiation+matter+Lambda for distances. This is not an Astropy/Colossus
cosmology substitution. Source weight is beta/beta_infinity. Critical surface
density is in (Msun/h)/(Mpc/h)^2; physical radii are in Mpc/h.

### nfw.py

`nfw_normalization(M, c, z_l)` takes an explicitly supplied concentration and
returns r200c, r_s, rho_s, and kappa_s,infinity. It accepts broadcast mass and
concentration arrays at one scalar lens redshift.

`nfw_f(x)`, `nfw_g(x)`, and `nfw_shear_shape(x)` return the projected shapes f,
G, and h=4G/x^2-2f. `nfw_lensing(x,kappa_s)` returns kappa, gamma_t, and interior
mean kappa. `kappa_s=rho_s*r_s/Sigma_crit`; no factor of two is hidden in that
normalization. Profile outputs are float64 arrays, including 0D scalar outputs.

The profile is untruncated. Radius arguments must be finite and strictly
positive. No implicit radius floor or silent clipping is used. Source-plane
normalization is supplied before evaluating the fields. Reduced shear,
magnification, and nonlinear ring averages are deferred to the ring/observable
increment.

**Explicit numerical changes from legacy helpers:**

1. Branch masks are complementary; no threshold value is left unassigned.
2. Near x=1, f uses the audited sixth-order series. G uses the consistent
   seventh-order antiderivative obtained from G'=x*f and G(1)=1-log(2), rather
   than replacing G by a constant over a finite interval.
3. For x<1e-3, stable series in x^2 through fourth order evaluate f, G/x^2,
   and the shear shape h; h is evaluated directly to avoid cancellation.
4. Below unity, G is algebraically rearranged with log1p to avoid subtracting
   two large logarithmic terms. f retains the standard arctanh expression
   outside its series regions, with (1-x)*(1+x) for the denominator factor.
5. A small-concentration series stabilizes log(1+c)-c/(1+c).

These are deliberate evaluations of the same analytic NFW profile, not changes
to the physical halo model. Agreement with high-precision and integral tests
validates this local numerical foundation. It does not yet establish
end-to-end equivalence of the future population/ring calculations. Keep the
historical output targets and assess any propagated changes explicitly.

### projection.py

`project_triaxial(p,s,los,spin_angle=0)` returns the scalar Schur-complement
projection. The LOS is in the intrinsic major/intermediate/minor axis frame.
The stable sky basis uses the least-aligned coordinate axis as its reference.
The projected major axis corresponds to the smaller eigenvalue of P; its
position angle is defined modulo pi and is physically arbitrary for a circle.

`project_triaxial_batch(p,s,los)` takes supplied p/s arrays and an (N,3) LOS
array, returning q_perp, b_los, and projected scale factors. It consumes no
random numbers. The scalar eigensolver and the historical batch
trace/discriminant construction are distinguished and regression-tested.

`result.circular_reference_parameters(r_s,kappa_s)` applies
r_s,perp=b_los^-1/2*r_s and kappa_s,perp=b_los*kappa_s. An eventual offset-only
control will remove projected ellipticity without resetting these two values.
`projected_radius` uses arbitrary sky-basis coordinates; the
`area_preserving_radius` function instead requires projected principal-axis
coordinates. Do not mix those coordinate conventions.

## Focused acceptance tests

- Reference cosmological distances and source efficiencies; independent
  redshift-variable distance integrals; the mass/r200c density relation.
- High-precision NFW values, all branch edges and adjacent floating-point
  values, analytic limits, LOS integration, enclosed-profile integration,
  the derivative identity G'=x*f, and mass normalization.
- Volume preservation, spherical/principal-axis limits, determinant and
  eigenvalue identities, scalar/batch/historical agreement, sky-basis rotation,
  and direct integration of the intrinsic 3D ellipsoidal density.
- Preservation of projected circular-reference normalization.
- The centered spherical Appendix D safety-radius anchor, evaluated without
  implementing the displaced-ring solver.

Warnings are errors in the focused pytest suite. Test tolerances are stated in
the test files and reference README. None requires byte identity for figures,
NPZ archives, or floating-point results across different numerical libraries.

## Source provenance

Scientific equations: canonical manuscript_v0.648.tex, Appendix C.1 and the
normalization conventions, plus PUBLIC_NUMERICAL_SPECIFICATION.md v0.1
Sections 3--4. The audit's late-time/distance distinction and NFW edge-case
findings are preserved explicitly.

Historical reference archive:
`cluster_lensing_1d_publication_v0.1.0.zip`, SHA-256
`0da278eea14038228bbb1b534b6814b7d32fc6e429a17ca7e3ee54c9bb167ba6`.
Relevant source-member fingerprints are in
`reference/foundation/historical_targets.json`.

The original publication_v0.1.0 finite-source products and the added
far-background alignment authority in publication_v0.1.1 remain separate
future integration-test targets. Neither archive is modified or repackaged by
this increment.
