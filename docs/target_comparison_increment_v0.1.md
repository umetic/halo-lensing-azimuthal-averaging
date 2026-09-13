# Target-aware production comparison increment v0.1

This increment upgrades the production-validation `compare` stage from a
fingerprint-only report to a target-aware PASS/FAIL oracle.  It compares an
already generated production-validation run directory against committed compact
publication targets for the common domain, Table 2, the centered-shape response,
Section 6.1/6.2 interaction diagnostics, Section 6.4 alignment diagnostics, and
Table 3.

The increment does not run lensing fields, rebuild the mode library, regenerate
Monte Carlo samples, render figures, or change scientific definitions.  It is a
validation-infrastructure layer only.

The comparison stage classifies checks as PASS, FAIL, INCOMPLETE, or SKIPPED.
A production-size run must contain all six grid products and the required
analysis files before scientific target checks are interpreted as a full
production-validation result.
