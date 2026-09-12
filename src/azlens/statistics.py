"""Statistical utilities for publication-level summaries.

The functions in this module are deliberately independent of the halo,
field-evaluation, and paper-specific selection layers.  They summarize signed
finite values under explicit masks and expose counts so that finite-value
filters are never confused with the outer-safe represented fraction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_PERCENTILES = (16.0, 50.0, 84.0, 97.5)


@dataclass(frozen=True)
class SummaryStats:
    """Finite-value summary for an explicitly selected one-dimensional sample."""

    n_total: int
    n_selected: int
    n_finite: int
    mean: float
    p16: float
    p50: float
    p84: float
    p97p5: float

    @property
    def selected_fraction_total(self) -> float:
        return np.nan if self.n_total == 0 else self.n_selected / float(self.n_total)

    @property
    def finite_fraction_selected(self) -> float:
        return np.nan if self.n_selected == 0 else self.n_finite / float(self.n_selected)

    def as_dict(self, *, suffix: str = "") -> dict[str, float | int]:
        """Return a flat dictionary using an optional suffix for value columns."""
        s = str(suffix)
        return {
            "n_total": int(self.n_total),
            "n_selected": int(self.n_selected),
            "n_finite": int(self.n_finite),
            "selected_fraction_total": float(self.selected_fraction_total),
            "finite_fraction_selected": float(self.finite_fraction_selected),
            f"mean{s}": float(self.mean),
            f"p16{s}": float(self.p16),
            f"p50{s}": float(self.p50),
            f"p84{s}": float(self.p84),
            f"p97p5{s}": float(self.p97p5),
        }


@dataclass(frozen=True)
class LeastSquaresResult:
    """Unweighted least-squares result with the fitted prediction and residuals."""

    coefficients: FloatArray
    prediction: FloatArray
    residual: FloatArray
    rank: int
    singular_values: FloatArray


def _flat_float_array(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def finite_values(values: ArrayLike, selected_mask: ArrayLike | None = None) -> FloatArray:
    """Return finite values after applying an optional explicit selection mask."""
    vals = _flat_float_array(values, "values")
    if selected_mask is None:
        selected = vals
    else:
        mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if mask.shape != vals.shape:
            raise ValueError("selected_mask must have the same flattened shape as values")
        selected = vals[mask]
    return np.asarray(selected[np.isfinite(selected)], dtype=np.float64)


def ordinary_percentiles(values: ArrayLike, percentiles: ArrayLike = DEFAULT_PERCENTILES) -> FloatArray:
    """Compute NumPy ordinary linear percentiles of finite supplied values."""
    vals = _flat_float_array(values, "values")
    if np.any(~np.isfinite(vals)):
        raise ValueError("ordinary_percentiles requires finite input values")
    probs = np.asarray(percentiles, dtype=np.float64).reshape(-1)
    if probs.size == 0 or np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 100.0)):
        raise ValueError("percentiles must lie between 0 and 100")
    return np.asarray(np.percentile(vals, probs), dtype=np.float64)


def summarize_signed_values(values: ArrayLike, selected_mask: ArrayLike | None = None) -> SummaryStats:
    """Summarize signed finite values under an explicit selection mask.

    The input values are not absolutized.  Nonfinite values are excluded only
    from the value statistics and are reported through ``n_finite``.
    """
    vals = _flat_float_array(values, "values")
    n_total = int(vals.size)
    if selected_mask is None:
        selected = vals
    else:
        mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if mask.shape != vals.shape:
            raise ValueError("selected_mask must have the same flattened shape as values")
        selected = vals[mask]
    finite = np.asarray(selected[np.isfinite(selected)], dtype=np.float64)
    if finite.size == 0:
        return SummaryStats(n_total, int(selected.size), 0, np.nan, np.nan, np.nan, np.nan, np.nan)
    q = ordinary_percentiles(finite, DEFAULT_PERCENTILES)
    return SummaryStats(
        n_total=n_total,
        n_selected=int(selected.size),
        n_finite=int(finite.size),
        mean=float(np.mean(finite)),
        p16=float(q[0]),
        p50=float(q[1]),
        p84=float(q[2]),
        p97p5=float(q[3]),
    )


def r2_score(y_true: ArrayLike, y_pred: ArrayLike, selected_mask: ArrayLike | None = None) -> float:
    """Coefficient of determination with explicit finite-value filtering."""
    y = _flat_float_array(y_true, "y_true")
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("y_true and y_pred must have the same flattened shape")
    if selected_mask is not None:
        mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if mask.shape != y.shape:
            raise ValueError("selected_mask must have the same flattened shape as y_true")
    else:
        mask = np.ones_like(y, dtype=bool)
    mask &= np.isfinite(y) & np.isfinite(p)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    yy = y[mask]
    pp = p[mask]
    denom = float(np.sum((yy - np.mean(yy))**2))
    if denom == 0.0:
        return float("nan")
    numer = float(np.sum((yy - pp)**2))
    return float(1.0 - numer/denom)


def least_squares_fit(
    design_matrix: ArrayLike,
    target: ArrayLike,
    selected_mask: ArrayLike | None = None,
    *,
    rcond: float | None = None,
) -> LeastSquaresResult:
    """Fit an unweighted linear model after explicit finite-value filtering."""
    X = np.asarray(design_matrix, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("design_matrix must have shape (n_sample,n_parameter)")
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if X.shape[0] != y.size:
        raise ValueError("target length must match the design_matrix row count")
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if selected_mask is not None:
        sel = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if sel.shape != y.shape:
            raise ValueError("selected_mask must match target length")
        mask &= sel
    if np.count_nonzero(mask) < X.shape[1]:
        raise ValueError("not enough finite selected rows for the requested fit")
    coeff, _, rank, singular = np.linalg.lstsq(X[mask], y[mask], rcond=rcond)
    prediction = X @ coeff
    return LeastSquaresResult(
        coefficients=np.asarray(coeff, dtype=np.float64),
        prediction=np.asarray(prediction, dtype=np.float64),
        residual=np.asarray(y - prediction, dtype=np.float64),
        rank=int(rank),
        singular_values=np.asarray(singular, dtype=np.float64),
    )
