# Implementation history

This document summarizes the public implementation history at the level useful for users of the reproduction package. The detailed change history is preserved by Git.

## Scientific validation anchor

- `validation-pass-20260913` marks the commit at which the target-aware production comparison reported `Production validation: PASS`.
- Product-generation commit: `83b011d126592bffe6dd74cb5f9241e5787dfabf`.
- Comparison-oracle commit: `161b8b2600631b08a07aa1b5128e45486afba43a`.

## Layered implementation sequence

The implementation was built and validated in layers:

1. Cosmology, NFW normalization, and volume-preserving triaxial projection.
2. Fourier--Green multipole solver with explicit population and controlled-halo numerical settings.
3. Adopted-center ring fields, spin-2 rotation, and nonlinear observables.
4. Controlled single-halo benchmarks for the illustrative paper calculations.
5. Seeded six-grid population and miscentering realization.
6. Population field evaluation, source scaling, and local/outer selection masks.
7. Publication summary statistics and Table 2 selection logic.
8. Centered projected-shape leading-response analysis underlying Figure 3.
9. Reduced-shear interaction and conditioning diagnostics for Sections 6.1 and 6.2.
10. Far-background alignment reweighting for Section 6.4 and Table 3.
11. Production-validation workflow and target-aware comparison oracle.
12. Release documentation, environment metadata, and clean-clone validation.

## Design principles

- The repository implements the final paper's numerical experiment, not the historical development workspace.
- Figures and tables are downstream products; the code is organized by scientific calculation and validation layer.
- Large generated arrays are reproducible outputs, not tracked Git content.
- Compact reference files are validation targets; they are not copied into regenerated science products.
- Documentation-only changes after the validation tag do not require repeating the heavy production run unless numerical code or validation targets change.
