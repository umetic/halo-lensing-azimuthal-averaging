# Increment 4B: population field evaluation and retained-domain machinery

This increment adds the deterministic/vectorized field-evaluation layer that
consumes the validated seeded halo and offset realization.  It builds or loads a
population Fourier--Green mode library, evaluates far-background ring fields,
scales those fields to finite source planes before nonlinear observable maps,
computes sampled local/outer safety masks, and emits radial summaries and
key-radius halo products.

It deliberately stops before final Table 2 selection, Figure 3 construction,
Section 6 regression diagnostics, and alignment reweighting.  Those are analysis
layers built on the products introduced here.

The public scientific path rejects out-of-domain mode-library evaluations rather
than silently clipping.  Any legacy clipping comparison must be explicit.
