# Population-realization increment v0.1

This increment adds the stochastic halo/offset realization layer for the
paper-specific reproduction package.  It deliberately does not evaluate ring
lensing fields or population residuals.

Implemented modules:

- `src/azlens/population.py`: redshift-major six-grid layout, grid-dependent
  population seeds, Colossus DJ19/peak-height state, Bonamigo intrinsic axis
  ratios, isotropic viewing directions, triaxial projection, and halo catalog
  containers.
- `src/azlens/miscentering.py`: independent `SeedSequence([20260816,g])`
  offset streams, unit-Rayleigh amplitudes, offset angles, and sigma-scaled
  amplitude matrices.

The central rule is that the publication realization is stochastic in its
scientific model but deterministic under the recorded pseudo-random streams and
draw order.  A smaller run with the same seed is not generally a prefix of the
full `N=10000` catalog after the first vector draw.

Validation is against compact targets extracted from the frozen publication
catalogs.  Full ring-field evaluation, source-plane residuals, safety masks,
common-domain selection, Table 2, and alignment reweighting are left to later
increments.
