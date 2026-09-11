# Multipole solver increment v0.1

This increment adds the deterministic Fourier--Green layer that follows the
validated cosmology, NFW, and triaxial-projection foundation.

## Added modules

- `src/azlens/solver_settings.py` defines named numerical profiles for the
  population mode library and the controlled single-halo benchmarks.
- `src/azlens/multipoles.py` implements endpoint-excluding angular Fourier
  coefficients, Green-function reconstruction of potential modes, tangential
  and cross-shear mode coefficients, coefficient interpolation in either `x` or
  `logx`, and area-preserving elliptical NFW mode-grid construction.

The implementation is deterministic. It does not include miscentering,
nonlinear observables, population sampling, reweighting, plotting, or any
manuscript/table updates.

## Design decisions

The controlled and population calculations share the same physical equations
but keep separate numerical profiles. In particular, the population profile uses
linear interpolation in `x`, while the controlled profiles use linear
interpolation in `logx`. The code treats those choices as explicit settings,
not incidental constants.

The Fourier interface rejects the Nyquist mode to avoid the normalization
ambiguity of real-FFT Nyquist coefficients. All publication and controlled
settings have `m_max` safely below Nyquist.

## Validation scope

The local check script

```bash
python scripts/check_multipoles.py
```

runs the full deterministic-core test suite through this increment. The suite
includes the 124 foundation tests plus tests of solver settings, Fourier
coefficients, mode powers, spherical limits, compact historical mode references,
coefficient interpolation, and a small stored-grid builder.

Passing these tests validates the deterministic multipole layer. It is not yet a
validation of displaced-ring fields, nonlinear observables, Monte Carlo
population generation, alignment reweighting, or publication figure/table
reproduction.
