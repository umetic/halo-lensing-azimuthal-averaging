"""Weighted statistical utilities for alignment reweighting.

The functions here implement the midpoint-CDF weighted quantile and weighted
R^2 conventions used by the offset--major-axis alignment diagnostic.  They are
kept separate from :mod:`azlens.statistics`, whose ordinary NumPy percentile
convention is used for the unweighted publication summaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class WeightedSummary:
    """Weighted finite-value summary under an explicit selected mask."""

    n_total: int
    n_selected: int
    n_used: int
    weight_sum_selected: float
    weight_sum_used: float
    mean: float
    p16: float
    p50: float
    p84: float
    effective_sample_size: float

    def as_dict(self, *, suffix: str = "") -> dict[str, float | int]:
        s = str(suffix)
        return {
            "n_total": int(self.n_total),
            "n_selected": int(self.n_selected),
            "n_used": int(self.n_used),
            "weight_sum_selected": float(self.weight_sum_selected),
            "weight_sum_used": float(self.weight_sum_used),
            f"mean{s}": float(self.mean),
            f"p16{s}": float(self.p16),
            f"p50{s}": float(self.p50),
            f"p84{s}": float(self.p84),
            "effective_sample_size": float(self.effective_sample_size),
        }


def _flat_float(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def _flat_bool(values: ArrayLike, name: str, shape: tuple[int, ...]) -> BoolArray:
    arr = np.asarray(values, dtype=bool).reshape(-1)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape} after flattening")
    return arr


def _values_weights_mask(
    values: ArrayLike,
    weights: ArrayLike,
    selected_mask: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, BoolArray, int, int, float]:
    vals = _flat_float(values, "values")
    w = _flat_float(weights, "weights")
    if w.shape != vals.shape:
        raise ValueError("values and weights must have the same flattened shape")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    selected = np.ones(vals.shape, dtype=bool)
    if selected_mask is not None:
        selected &= _flat_bool(selected_mask, "selected_mask", vals.shape)
    # The selected-weight sum includes all finite selected non-negative weights,
    # independent of whether the paired value is finite.  The used sample also
    # requires a finite value and positive weight.
    finite_weight = selected & np.isfinite(w)
    n_selected = int(np.count_nonzero(selected))
    selected_weight_sum = float(np.sum(w[finite_weight]))
    used_mask = finite_weight & np.isfinite(vals) & (w > 0.0)
    n_used = int(np.count_nonzero(used_mask))
    return vals, w, used_mask, n_selected, n_used, selected_weight_sum


def effective_sample_size(weights: ArrayLike, selected_mask: ArrayLike | None = None) -> float:
    """Return ``(sum w)^2/sum(w^2)`` for finite positive selected weights."""
    w = _flat_float(weights, "weights")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    selected = np.ones(w.shape, dtype=bool)
    if selected_mask is not None:
        selected &= _flat_bool(selected_mask, "selected_mask", w.shape)
    use = selected & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(use) == 0:
        return float("nan")
    ww = w[use]
    denom = float(np.sum(ww*ww))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(ww)**2/denom)


def weighted_mean(values: ArrayLike, weights: ArrayLike, selected_mask: ArrayLike | None = None) -> float:
    """Weighted mean after finite-value and positive-weight filtering."""
    vals, w, use, *_ = _values_weights_mask(values, weights, selected_mask)
    if np.count_nonzero(use) == 0:
        return float("nan")
    return float(np.sum(vals[use]*w[use])/np.sum(w[use]))


def weighted_midpoint_quantile(
    values: ArrayLike,
    weights: ArrayLike,
    quantiles: ArrayLike,
    selected_mask: ArrayLike | None = None,
) -> FloatArray:
    """Weighted quantiles using the midpoint cumulative-coordinate convention.

    Values are sorted, cumulative coordinates are
    ``(cumsum(w)-0.5*w)/sum(w)``, and requested probabilities outside the
    coordinate range are assigned the endpoint values through ``np.interp``.
    ``quantiles`` are probabilities in [0,1], not percentages.
    """
    vals, w, use, *_ = _values_weights_mask(values, weights, selected_mask)
    probs = np.asarray(quantiles, dtype=np.float64).reshape(-1)
    if probs.size == 0 or np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("quantiles must be finite probabilities in [0,1]")
    if np.count_nonzero(use) == 0:
        return np.full(probs.shape, np.nan, dtype=np.float64)
    vv = vals[use]
    ww = w[use]
    order = np.argsort(vv, kind="mergesort")
    vv = vv[order]
    ww = ww[order]
    total = float(np.sum(ww))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(probs.shape, np.nan, dtype=np.float64)
    coords = (np.cumsum(ww) - 0.5*ww)/total
    return np.asarray(np.interp(probs, coords, vv, left=vv[0], right=vv[-1]), dtype=np.float64)


def summarize_weighted_values(values: ArrayLike, weights: ArrayLike, selected_mask: ArrayLike | None = None) -> WeightedSummary:
    """Return weighted mean and 16/50/84 percentiles with explicit counts."""
    vals, w, use, n_selected, n_used, selected_weight_sum = _values_weights_mask(values, weights, selected_mask)
    if n_used == 0:
        return WeightedSummary(int(vals.size), n_selected, 0, selected_weight_sum, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan)
    q16, q50, q84 = weighted_midpoint_quantile(vals, w, [0.16, 0.50, 0.84], use)
    return WeightedSummary(
        n_total=int(vals.size),
        n_selected=n_selected,
        n_used=n_used,
        weight_sum_selected=selected_weight_sum,
        weight_sum_used=float(np.sum(w[use])),
        mean=weighted_mean(vals, w, use),
        p16=float(q16),
        p50=float(q50),
        p84=float(q84),
        effective_sample_size=effective_sample_size(w, use),
    )


def weighted_r2_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    weights: ArrayLike,
    selected_mask: ArrayLike | None = None,
) -> float:
    """Weighted coefficient of determination with explicit finite filtering."""
    y = _flat_float(y_true, "y_true")
    p = _flat_float(y_pred, "y_pred")
    w = _flat_float(weights, "weights")
    if p.shape != y.shape or w.shape != y.shape:
        raise ValueError("y_true, y_pred, and weights must share flattened shape")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    selected = np.ones(y.shape, dtype=bool)
    if selected_mask is not None:
        selected &= _flat_bool(selected_mask, "selected_mask", y.shape)
    use = selected & np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(use) == 0:
        return float("nan")
    yy = y[use]
    pp = p[use]
    ww = w[use]
    mean = float(np.sum(ww*yy)/np.sum(ww))
    denom = float(np.sum(ww*(yy - mean)**2))
    if denom == 0.0:
        return float("nan")
    numer = float(np.sum(ww*(yy - pp)**2))
    return float(1.0 - numer/denom)
