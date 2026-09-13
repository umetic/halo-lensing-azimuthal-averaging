# Validation summary

**Status:** production-validation PASS for the paper-specific reproduction package.

**Validation tag:** `validation-pass-20260913`

**Product-generation commit:** `83b011d126592bffe6dd74cb5f9241e5787dfabf`

**Comparison-oracle commit:** `161b8b2600631b08a07aa1b5128e45486afba43a`

The product-generation commit produced the publication-size numerical products. The comparison-oracle commit added target-aware comparison logic and was used to classify the generated products against compact frozen targets. The distinction is recorded because the comparison oracle was added after the heavy field products had already been generated; the oracle did not change the scientific calculation.

## Validation environment

The local validation environment used for the PASS event reported:

```text
Python:   3.11.16
NumPy:    1.26.4
SciPy:    1.17.1
pytest:   9.0.3
Colossus: 1.4.0
PyYAML:   6.0.3
```

The public environment specification is `environment.yml`. It is intentionally portable and does not contain author-local paths or shell aliases.

## Production stages

The staged production workflow completed:

```text
preflight       complete
mode-library    complete
realization     complete
fields          complete for grids 0--5
analysis        complete
compare         complete
```

The production field stage used six grids with `N=10000` halos per grid, four offset models, 36 radii, and the population Fourier--Green mode-library profile. The generated products were written under a local output root and are not committed to Git.

## Target-aware comparison result

```text
Production validation: PASS

run_completeness: PASS
common_domain: PASS
table2: PASS
shape_response: PASS
interaction_diagnostics: PASS
alignment_diagnostics: PASS
table3: PASS
```

The PASS verdict means that the regenerated publication-size products satisfied the committed target checks within their category-specific tolerances. It does not mean that large generated arrays are stored in Git; users can regenerate them with the staged workflow.

## Key recovered publication targets

The common domain for the reference offset model, `sigma_d=0.05`, was recovered with

```text
R_min/r200c = 0.3656975063010877
limiting grid = 5
limiting represented fraction = 0.9518
number of retained radii = 10
```

The Table 2 selected rows all correspond to grid 5, `R/r200c=0.3656975063010877`, and `sigma_d=0.05`. The centered-shape analysis recovered retained counts

```text
(10000, 10000, 10000, 10000, 9993, 9258)
```

at the diagnostic radius `R/r200c=0.2924408238028535`. The interaction and alignment diagnostics recovered the reported Section 6 target ranges and Table 3 rows.

## When validation should be rerun

Run the fast validation suite after any documentation or interface change:

```bash
python scripts/check_full_production_validation.py
```

Rerun the full production workflow if a change can affect numerical outputs, including changes to the NFW functions, projection, multipoles, ring fields, observables, population sampling, mode-library settings, selection masks, statistics, or alignment logic.

Documentation-only changes after `validation-pass-20260913` do not by themselves require repeating the multi-hour field stage, provided no numerical code or validation target changes are made.
