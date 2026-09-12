# Publication summary increment v0.1

This increment adds only the unweighted publication-summary layer. It consumes
radial-summary rows produced by the population-field engine, identifies the
reference common radial domain for `sigma_d=0.05`, and selects the Table 2 rows
by maximizing the absolute value of the signed median residual within that
domain for each observable and source plane.

It does not evaluate lensing fields, generate Figure 3, fit Section 6
diagnostics, perform alignment reweighting, or render publication figures.

The statistical utilities expose selected and finite-value counts explicitly so
that quantity-specific finite-value filtering is not confused with the
outer-safe represented fraction.
