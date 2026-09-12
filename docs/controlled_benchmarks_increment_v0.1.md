# Controlled benchmark increment v0.1

This increment adds a deterministic benchmark layer for the paper's
single-halo calculations.  It defines the fixed illustrative configuration
`M200c=1e15 Msun/h`, `z_l=0.3`, `z_s=1`, internal `c200c=3.843`, and
`q_perp=0.67`, then evaluates centered elliptical, displaced spherical, and
combined displaced elliptical rings through the already validated core modules.

The module is intentionally an orchestration layer.  It does not define new NFW
profile formulae, Fourier--Green equations, spin-2 rotations, nonlinear
observables, Monte Carlo sampling, alignment weights, or plotting routines.

The validation suite checks compact anchors for Figure 1, Figure 2, and
Appendix D.  The script `scripts/check_controlled_benchmarks.py` writes numeric
CSV/JSON benchmark products under `outputs/controlled_benchmarks_validation/`.
Those generated products are for local validation and should not be committed.

Known numerical policy choices:

- Figure 1/Appendix-D anchors are validated at the sub-micro-percent level, not
  by requiring byte identity with every historical helper.
- Figure 2 panel-(a) miscentered-sphere mode fractions differ from the frozen
  E2 table by at most about `1.2e-7` because the public path uses the corrected
  robust NFW branch handling.  This tolerance is intentionally narrow and guards
  against convention errors without reintroducing legacy branch fragility.
- The old auxiliary far-background helper normalization is not used as a public
  benchmark.
