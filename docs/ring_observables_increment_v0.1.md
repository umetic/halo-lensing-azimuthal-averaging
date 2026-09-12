# Ring-field and nonlinear-observable increment v0.1

This increment adds deterministic adopted-center ring fields and nonlinear
ring-observable summaries. It sits above the validated Fourier--Green multipole
layer and below the future controlled-benchmark and population layers.

## Included

- Midpoint ring angles `phi_j=(j+1/2)2*pi/N_phi`.
- Displaced-ring geometry about an adopted center.
- True-center to adopted-center spin-2 rotation of shear components.
- Circular NFW true-center evaluator.
- Fixed-q multipole true-center evaluator.
- Complete-ring means and `ddof=0` angular covariances.
- Finite-source scaling of linear fields before nonlinear maps.
- Reduced shear, inverse magnification, magnification, and power-law
  magnification-bias residuals.
- Reduced-shear and magnification-family second-order diagnostics used in the
  paper-specific workflow.

## Excluded

No population Monte Carlo, Colossus-backed population sampler, alignment
reweighting, plotting, table generation, or figure-specific orchestration is
introduced in this increment.

## Numerical policies

- The ring-evaluation grid is separate from the Fourier-solver grid.
- Out-of-domain mode radii raise an error by default; this public path does not
  silently clip to the mode-library boundary.
- Mean-field inverse magnification retains the measured complete-ring
  `gamma_cross` mean instead of forcing it to zero.
- Reduced-shear fractional residuals use the mean-field reduced-shear
  denominator and are set to NaN when that denominator is nonfinite or below
  the numerical guard.

## Validation

The validation suite checks geometry, spin-2 rotation, observable identities,
finite-source scaling order, denominator guards, the inverse-magnification
covariance identity, and one Appendix-D displaced-spherical anchor. It extends
the deterministic-core suite through the multipole layer.
