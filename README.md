# Halo-Lensing Azimuthal Averaging

Paper-specific reference implementation and reproduction package for the numerical calculations in

> Keiichi Umetsu, *Why Azimuthal Averaging Works in Halo Lensing: Symmetry and Power Counting for Nonlinear Shear and Magnification*, arXiv:2609.08825.

This repository reproduces the controlled halo-lensing benchmarks, seeded triaxial-halo population experiment, finite-source and far-background evaluations, selections, reweightings, and summary statistics used in the paper. It is **not** a general-purpose weak-lensing analysis package, survey estimator, halo mass-fitting pipeline, or observational data-processing system.

## Validation status

A publication-size run regenerated the six-grid, `N=10000` population calculation and the target-aware comparison reported

```text
Production validation: PASS
```

The comparison passed the run-completeness, common-domain, Table 2, centered-shape response, Section 6.1/6.2 interaction-diagnostic, Section 6.4 alignment-diagnostic, and Table 3 checks.

## Installation

The supported environment is a Conda environment named `azlens`:

```bash
conda env create -f environment.yml
conda activate azlens
```

The reference environment uses Python 3.11 with NumPy, SciPy, pandas, Matplotlib, PyYAML, pytest, and Colossus. Colossus is installed through the `pip` subsection of `environment.yml`.

## Quick validation

Run the complete fast validation suite and a tiny production-workflow smoke test:

```bash
python scripts/check_full_production_validation.py
```

This command should finish in seconds on a typical workstation. It does not run the full `N=10000` six-grid production calculation.

## Publication-size reproduction

The full reproduction workflow is staged and resumable. A typical run is:

```bash
OUT=$PWD/outputs/production_validation
RUN_ID=production_v0p1

python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage preflight
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage mode-library
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage realization

for g in 0 1 2 3 4 5; do
    python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage fields --grid "$g"
done

python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage analysis
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage compare
```

The field stage is the expensive part. Runtime is hardware dependent; in the validation run used for this repository milestone, each production grid took roughly two hours of active single-core computation.

Large generated products are written under `outputs/` or a user-specified output root and are not committed to Git.

## Documentation

- [`docs/REPRODUCTION_WORKFLOW.md`](docs/REPRODUCTION_WORKFLOW.md) — detailed staged commands for reproducing the numerical calculations.
- [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md) — generated-output structure and quantity definitions.

## Citation

If you use this reproduction package, please cite the associated paper. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the MIT License; see [`LICENSE`](LICENSE). The package depends on third-party scientific Python packages distributed under their own licenses.
