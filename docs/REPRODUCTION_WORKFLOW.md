# Reproduction workflow

This document gives portable commands for reproducing the numerical workflow. Users may choose any output root with sufficient disk space.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate azlens
```

The supported reference interpreter is Python 3.11. Colossus is installed through the `pip` subsection of `environment.yml`.

## 2. Fast validation

Run the fast validation suite and a tiny production-workflow smoke test:

```bash
python scripts/check_full_production_validation.py
```

This command verifies the committed modules and the production-validation harness. It does not run the full publication-size field calculation.

## 3. Choose an output root

The default example stores generated files under the repository's ignored `outputs/` directory:

```bash
OUT=$PWD/outputs/production_validation
RUN_ID=production_v0p1
```

For long runs in a synchronized directory, a user may prefer an external output root:

```bash
OUT=$HOME/azlens_production_validation_runs
RUN_ID=production_v0p1
```

## 4. Run the staged production workflow

The workflow is staged and resumable.

```bash
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage preflight
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage mode-library
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage realization
```

Run the field stage grid by grid:

```bash
for g in 0 1 2 3 4 5; do
    python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage fields --grid "$g"
done
```

Then run the downstream analysis and comparison stages:

```bash
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage analysis
python scripts/run_full_production_validation.py --run-id "$RUN_ID" --output-root "$OUT" --stage compare
```

A successful target-aware comparison reports:

```text
Production validation: PASS
```

## 5. Runtime expectations

The mode-library and realization stages are lightweight compared with the field stage. Runtime is hardware dependent. In the validation run used for the repository milestone, the field stage required roughly two hours of active single-core computation per production grid.

Prevent system sleep during long runs using the operating system's power-management tools. On systems with `systemd-inhibit`, the field-stage loop can be run inside an inhibited shell, provided the Conda environment and `OUT`/`RUN_ID` variables are defined in that shell.

## 6. Inspect the result

The main validation products are

```text
$OUT/$RUN_ID/comparison/validation_summary.txt
$OUT/$RUN_ID/comparison/validation_summary.json
```

The generated outputs are local artifacts. They are not committed to Git.
