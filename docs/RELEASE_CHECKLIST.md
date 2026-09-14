# Release and maintenance checklist

This checklist records the status of the paper-specific reproduction package and the items to revisit for future release updates.

## Completed for the current release-ready state

- [x] Clean repository initialized.
- [x] Deterministic cosmology/NFW/projection foundation implemented.
- [x] Fourier--Green multipole solver implemented.
- [x] Ring-field and nonlinear-observable layer implemented.
- [x] Controlled single-halo benchmark layer implemented.
- [x] Seeded population and miscentering realization implemented.
- [x] Population field evaluation and selection masks implemented.
- [x] Publication summary and Table 2 selection implemented.
- [x] Centered-shape leading-response analysis implemented.
- [x] Reduced-shear interaction diagnostics implemented.
- [x] Far-background alignment reweighting implemented.
- [x] Production-validation workflow implemented.
- [x] Target-aware comparison oracle implemented.
- [x] Publication-size production validation passed.
- [x] Validation tag created: `validation-pass-20260913`.
- [x] Release documentation and environment specification added.
- [x] Clean-clone fast validation passed.
- [x] Large generated outputs excluded from Git.
- [x] GitHub remote configured and tags pushed.

## Before making the repository public

- [ ] Confirm that `README.md`, documentation, license, citation metadata, and tags render correctly on GitHub.
- [ ] Decide whether to create a formal version tag such as `v0.1.0`.
- [ ] Decide whether to archive a release on Zenodo and obtain a DOI.
- [ ] Decide whether to add final figure-rendering scripts or keep the repository focused on numerical reproduction.
- [ ] Update the manuscript/code-availability statement after the repository URL and release status are finalized.

## Triggers for repeating the full production run

Repeat the full production workflow if any numerical/scientific module changes after `validation-pass-20260913`, including NFW functions, projection, multipoles, ring fields, observables, population generation, mode-library settings, selection masks, statistics, or alignment logic.

Documentation-only changes do not require repeating the multi-hour field stage, provided validation targets and scientific code remain unchanged.
