# Output schema

The production-validation workflow writes generated products under

```text
outputs/production_validation/<run_id>/
```

or under the user-specified `--output-root` directory. Large products are intentionally excluded from Git.

## Directory layout

```text
<output-root>/<run-id>/
    stages/
    mode_library/
    realization/
    population_fields/
    analysis/
    comparison/
```

## `stages/`

Each completed stage writes a JSON record with status, duration, update time, and output descriptors. These files are useful for resumable runs and provenance checks.

## `mode_library/`

Typical contents:

```text
population_mode_library.npz
mode_library_manifest.json
```

The mode library stores the population Fourier--Green coefficient library on the publication q--x grid. It uses the explicit population numerical profile and is generated locally.

## `realization/`

Typical contents:

```text
grid_00_halo_realization.npz
...
grid_05_halo_realization.npz
grid_00_offset_realization.npz
...
grid_05_offset_realization.npz
realization_manifest.json
```

The halo realization files contain the seeded concentration, intrinsic shape, viewing-direction, and projected-geometry quantities. The offset files contain the independent unit-Rayleigh amplitudes and offset angles used across nonzero offset-scale models.

## `population_fields/`

Each grid has its own directory:

```text
population_fields/grid_00/
    product_arrays.npz
    product_metrics.json
    radial_summary_rows.json
```

The array products contain per-halo/key-radius quantities needed by the downstream summary, centered-shape, interaction, and alignment analyses. The radial summary rows contain signed fractional residual summaries over the sampled radial grid.

## `analysis/`

Typical contents:

```text
radial_summary_all_grids.csv
common_domain.json
table2_selection.csv
table2_selection.json
centered_shape_response.csv
centered_shape_response_metrics.json
hybrid_interaction_diagnostics.csv
conditioning_diagnostics.csv
interaction_diagnostics_metrics.json
alignment_reweighted_summary.csv
table3_selection.csv
alignment_diagnostics.json
```

These are generated analysis products. They are compared against compact committed targets by the `compare` stage.

## `comparison/`

Typical contents:

```text
validation_summary.txt
validation_summary.json
```

The text report is intended for human inspection. The JSON report is machine-readable and records PASS/FAIL/INCOMPLETE/SKIPPED status by target category.

## Quantity-name conventions

Important distinctions are encoded in the public names:

- `Delta_g_plus` is an observable difference.
- `delta_g_plus_fractional` is a fractional residual.
- `safe_local` is the local sampled far-background subcriticality condition.
- `safe_outer` is the outer-contiguous cumulative safety mask.
- finite-source labels such as `z1` and `z2` refer to fields scaled before nonlinear observable construction.
- far-background or `infinity` products use source weight `w=1` and are used for safety and alignment.

Ordinary unweighted percentiles and midpoint-CDF weighted quantiles are different conventions. Table 2 uses ordinary unweighted percentiles. Alignment summaries use weighted statistics.
