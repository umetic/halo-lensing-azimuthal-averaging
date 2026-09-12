# Increment 6A: far-background alignment reweighting

This increment adds the weighted offset--major-axis alignment layer for Section
6.4 and Table 3. It consumes far-background key-radius population products and
implements only deterministic reweighting/statistical analysis.

Implemented scope:

- Linear and nematic von Mises angular families for target parent `Qoff`.
- Importance weights relative to a uniform parent offset-angle distribution.
- Weighted represented fractions using the original-population denominator.
- Midpoint-CDF weighted quantiles, weighted means, effective sample size, and
  weighted R^2.
- Table 3 selection logic and compact Section 6.4 diagnostic summaries.
- Baseline predictor-transfer logic: fit unweighted Q=0 coefficients and hold
  them fixed during alignment reweighting.

Not implemented here:

- New lensing-field evaluation, new Monte Carlo sampling, plotting, Table 2 or
  Figure 3 logic, or final full-production validation.
