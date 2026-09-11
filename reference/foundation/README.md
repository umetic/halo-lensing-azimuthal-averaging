# Foundation reference data

These files are **test targets**, never inputs to the production physics.
The first increment needs NumPy, SciPy, and pytest only.

## historical_targets.json

Fixed-input calculations newly evaluated through the historical publication
source archive identified in `provenance.source_archive_sha256`. This is not a
new Monte Carlo production and does not require Colossus. It contains:

- physical distances, source weights, and critical surface densities;
- 27 spherical normalizations at specified masses, concentrations, and redshifts;
- 10 scalar and vectorized triaxial projection cases;
- ordinary-radius projected NFW surface-density values.

Source member paths and SHA-256 hashes identify the exact reference functions.
The first-increment tests compare outputs, not source layouts or file bytes.

## nfw_high_precision.json

Independent 80-decimal-digit evaluation using mpmath 1.3.0 at 521 binary64
inputs spanning 1e-12 <= x <= 1e3, including x=1, the new branch thresholds,
the historical thresholds, and their immediately adjacent floating-point values.
`x_hex` specifies the exact input. The outputs were rounded once to binary64.
mpmath is **not required to run the tests**.

For each exact input x, the generation recipe was:

```python
mp.mp.dps = 80
x = mp.mpf(float.fromhex(x_hex))
if x == 1:
    f = mp.mpf(1) / 3
    G = 1 - mp.log(2)
else:
    if x < 1:
        T = mp.acosh(1/x) / mp.sqrt(1-x*x)
    else:
        T = mp.acos(1/x) / mp.sqrt(x*x-1)
    f = (1-T) / (x*x-1)
    G = mp.log(x/2) + T
h = 4*G/x**2 - 2*f
```

This reference did not import `azlens.nfw` or use its Taylor series. Additional
tests independently integrate the 3D density along the line of sight, integrate
the enclosed projected profile, and check G'(x)=x*f(x).

The sampled reference accuracy test is rtol=2e-12 with zero absolute tolerance
for f, G, and h. It is not a proof of a uniform error bound over all positive x.
