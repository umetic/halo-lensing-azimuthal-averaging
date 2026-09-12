# Increment 5C: interaction diagnostics

This increment adds the unweighted reduced-shear interaction diagnostics used in
Sections 6.1 and 6.2 of the paper. It consumes key-radius population products
from the field-evaluation layer and does not evaluate new lensing fields,
perform alignment reweighting, or render figures.

Implemented scope:

- Section 6.1 hybrid predictor with the additive shape-only plus offset-only
  controls fixed to unit coefficient.
- Per-configuration unweighted least-squares fits for `a0` and `a_int`.
- All-outer-safe and restricted `u < 0.3` domains.
- Section 6.2 no-fit R^2 comparison for fractional residuals and observable
  differences.
- CSV writers and compact report metrics.

Not implemented in this increment:

- Far-background alignment weights, weighted quantiles, weighted R^2, Table 3,
  figure rendering, or any new population field evaluation.
