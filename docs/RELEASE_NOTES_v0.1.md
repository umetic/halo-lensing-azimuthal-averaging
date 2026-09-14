# Release notes — v0.1.0

This repository state is a paper-specific reproduction package for the numerical calculations in Umetsu (2026), arXiv:2609.08825.

## Scientific implementation status

Implemented and validated modules cover:

```text
cosmology / NFW / triaxial projection
Fourier--Green multipoles
ring fields and nonlinear observables
controlled single-halo benchmarks
seeded population and miscentering realization
population field evaluation and selection masks
publication summary and Table 2 selection
centered-shape leading-response analysis
reduced-shear interaction diagnostics
far-background alignment reweighting
production-validation workflow and target-aware comparison
```

## Validation milestone

The tag `validation-pass-20260913` marks the first production-validation PASS state. The full run regenerated the publication-size products and the target-aware oracle classified the result as PASS.

## Included in this release

- Paper-specific numerical implementation and validation targets.
- Fast validation scripts and a tiny production-workflow smoke test.
- Staged production-validation workflow for publication-size reproduction.
- Public installation, output-schema, validation, citation, and license documentation.

## Not included

- Final publication-figure rendering scripts.
- LaTeX table rendering.
- Large generated products committed to Git.
- Historical development trees and recovery scripts.
- Alternate-seed robustness calculations.
- Survey data, observational masks, noise models, or mass-fitting pipelines.

## Workflow notes

The production field stage is computationally heavier than the fast tests. The validation run used for this milestone required roughly two hours per grid on the validation hardware. Users should prevent system sleep during long runs and may choose an output root outside synchronized folders.
