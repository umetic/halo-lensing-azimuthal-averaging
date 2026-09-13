# Release checklist

## Completed

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

## Pending before public GitHub release

- [ ] Review public README for tone, scope, and completeness.
- [ ] Confirm license choice and third-party attribution statements.
- [ ] Create public GitHub repository and remote.
- [ ] Test a fresh clone and `conda env create -f environment.yml` on a clean machine or clean environment.
- [ ] Run `python scripts/check_full_production_validation.py` from the fresh clone.
- [ ] Decide whether to provide any compact generated example outputs beyond committed reference targets.
- [ ] Add or defer final figure-generation scripts.
- [ ] Decide whether to archive a release on Zenodo and obtain a DOI.
- [ ] Update the manuscript/code-availability statement for arXiv v2 or journal submission.

## Triggers for repeating the full production run

Repeat the full production workflow if any numerical/scientific module changes after `validation-pass-20260913`, including NFW functions, projection, multipoles, ring fields, observables, population generation, mode-library settings, selection masks, statistics, or alignment logic.

Documentation-only changes do not require repeating the multi-hour field stage, provided validation targets and scientific code remain unchanged.
