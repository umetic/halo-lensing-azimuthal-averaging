# Production-validation workflow increment v0.1

This increment adds orchestration and manifest machinery for a resumable
publication-scale validation run.  It does not add new scientific definitions,
plotting, table rendering, or manuscript-facing results.

The workflow records the Git/source/environment/configuration state, builds or
loads the population mode library, regenerates configured halo/offset
realizations, evaluates population-field products, runs the already implemented
analysis layers, and fingerprints the generated outputs.  The ordinary checker
runs fast tests and a tiny smoke workflow; the full N=10000 six-grid run must be
launched deliberately with `scripts/run_full_production_validation.py`.
