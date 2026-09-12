"""Far-background offset--major-axis alignment reweighting.

This module implements the Section 6.4/Table 3 alignment analysis.  It consumes
far-background key-radius products from :mod:`azlens.population_fields`; it does
not evaluate new lensing fields, draw new Monte-Carlo samples, or render plots.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import i0, i1

from .interaction_diagnostics import (
    DEFAULT_NONZERO_SIGMA_D,
    DEFAULT_U_MAX,
    fit_hybrid_interaction,
    source_index,
)
from .population_fields import PopulationFieldProducts
from .weighted_statistics import (
    effective_sample_size,
    summarize_weighted_values,
    weighted_mean,
    weighted_midpoint_quantile,
    weighted_r2_score,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_QOFF_VALUES = (-0.30, -0.15, 0.0, 0.15, 0.30)
DEFAULT_ALIGNMENT_FAMILIES = ("linear", "von_mises")
DEFAULT_ALIGNMENT_DOMAINS = (("all_outer_safe", None), ("u_lt_0p3", DEFAULT_U_MAX))
DEFAULT_TABLE3_QOFF = (-0.30, 0.0, 0.30)
DEFAULT_TABLE3_SIGMA_D = 0.05
DEFAULT_TABLE3_RADIUS = 0.2924408238028535
DEFAULT_REPRESENTED_THRESHOLD = 0.95
_RADIUS_ATOL = 5.0e-15
_SIGMA_ATOL = 5.0e-15
_Q_ATOL = 5.0e-15

ALIGNMENT_COLUMNS = (
    "grid_index", "mass_msun_h", "z_l", "source_label", "sigma_d", "R_over_r200c",
    "family", "Qoff", "domain", "u_max", "n_total", "n_outer_safe", "n_selected",
    "n_used", "weighted_represented_fraction_outer", "weight_sum_original",
    "weight_sum_selected", "effective_sample_size", "effective_sample_size_fraction",
    "empirical_Qoff_original", "empirical_Qoff_selected", "p16_delta_g_percent",
    "p50_delta_g_percent", "p84_delta_g_percent", "a0_baseline", "a_int_baseline", "weighted_r2",
)

TABLE3_COLUMNS = (
    "grid_index", "mass_msun_h", "z_l", "R_over_r200c", "sigma_d",
    "p50_Qoff_minus_0p30_percent", "p50_Qoff_0p00_percent", "p50_Qoff_plus_0p30_percent",
)


@dataclass(frozen=True)
class AlignmentRow:
    """One weighted alignment summary and predictor-transfer row."""

    grid_index: int
    mass_msun_h: float
    z_l: float
    source_label: str
    sigma_d: float
    R_over_r200c: float
    family: str
    Qoff: float
    domain: str
    u_max: float | None
    n_total: int
    n_outer_safe: int
    n_selected: int
    n_used: int
    weighted_represented_fraction_outer: float
    weight_sum_original: float
    weight_sum_selected: float
    effective_sample_size: float
    empirical_Qoff_original: float
    empirical_Qoff_selected: float
    p16_delta_g_percent: float
    p50_delta_g_percent: float
    p84_delta_g_percent: float
    a0_baseline: float
    a_int_baseline: float
    weighted_r2: float

    @property
    def effective_sample_size_fraction(self) -> float:
        return np.nan if self.n_selected == 0 else float(self.effective_sample_size)/float(self.n_selected)

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "grid_index": int(self.grid_index),
            "mass_msun_h": float(self.mass_msun_h),
            "z_l": float(self.z_l),
            "source_label": self.source_label,
            "sigma_d": float(self.sigma_d),
            "R_over_r200c": float(self.R_over_r200c),
            "family": self.family,
            "Qoff": float(self.Qoff),
            "domain": self.domain,
            "u_max": None if self.u_max is None else float(self.u_max),
            "n_total": int(self.n_total),
            "n_outer_safe": int(self.n_outer_safe),
            "n_selected": int(self.n_selected),
            "n_used": int(self.n_used),
            "weighted_represented_fraction_outer": float(self.weighted_represented_fraction_outer),
            "weight_sum_original": float(self.weight_sum_original),
            "weight_sum_selected": float(self.weight_sum_selected),
            "effective_sample_size": float(self.effective_sample_size),
            "effective_sample_size_fraction": float(self.effective_sample_size_fraction),
            "empirical_Qoff_original": float(self.empirical_Qoff_original),
            "empirical_Qoff_selected": float(self.empirical_Qoff_selected),
            "p16_delta_g_percent": float(self.p16_delta_g_percent),
            "p50_delta_g_percent": float(self.p50_delta_g_percent),
            "p84_delta_g_percent": float(self.p84_delta_g_percent),
            "a0_baseline": float(self.a0_baseline),
            "a_int_baseline": float(self.a_int_baseline),
            "weighted_r2": float(self.weighted_r2),
        }


@dataclass(frozen=True)
class Table3Record:
    """One Table-3 row after the representation cut."""

    grid_index: int
    mass_msun_h: float
    z_l: float
    R_over_r200c: float
    sigma_d: float
    p50_Qoff_minus_0p30_percent: float
    p50_Qoff_0p00_percent: float
    p50_Qoff_plus_0p30_percent: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "grid_index": int(self.grid_index),
            "mass_msun_h": float(self.mass_msun_h),
            "z_l": float(self.z_l),
            "R_over_r200c": float(self.R_over_r200c),
            "sigma_d": float(self.sigma_d),
            "p50_Qoff_minus_0p30_percent": float(self.p50_Qoff_minus_0p30_percent),
            "p50_Qoff_0p00_percent": float(self.p50_Qoff_0p00_percent),
            "p50_Qoff_plus_0p30_percent": float(self.p50_Qoff_plus_0p30_percent),
        }


def _flat_float(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def linear_alignment_weights(phi_offset: ArrayLike, Qoff: float) -> FloatArray:
    """Importance weights for ``p=(1+2 Q cos 2phi)/(2pi)``."""
    q = float(Qoff)
    if not np.isfinite(q) or abs(q) > 0.5:
        raise ValueError("linear alignment requires finite |Qoff| <= 0.5")
    phi = _flat_float(phi_offset, "phi_offset")
    weights = 1.0 + 2.0*q*np.cos(2.0*phi)
    # Roundoff may make the boundary very slightly negative at |Q|=0.5.
    if np.min(weights) < -1.0e-14:
        raise ValueError("linear alignment produced negative weights")
    return np.asarray(np.maximum(weights, 0.0), dtype=np.float64)


def von_mises_kappa_for_Qoff(Qoff: float) -> float:
    """Return signed kappa satisfying I1(kappa)/I0(kappa)=Qoff."""
    q = float(Qoff)
    if not np.isfinite(q) or abs(q) >= 1.0:
        raise ValueError("von Mises alignment requires finite |Qoff| < 1")
    if abs(q) < 1.0e-14:
        return 0.0
    target = abs(q)

    def ratio_minus_target(kappa: float) -> float:
        return float(i1(kappa)/i0(kappa) - target)

    kappa_abs = brentq(ratio_minus_target, 0.0, 50.0)
    return float(np.sign(q)*kappa_abs)


def von_mises_alignment_weights(phi_offset: ArrayLike, Qoff: float) -> FloatArray:
    """Nematic von Mises importance weights with target parent moment Qoff."""
    phi = _flat_float(phi_offset, "phi_offset")
    kappa = von_mises_kappa_for_Qoff(Qoff)
    if kappa == 0.0:
        return np.ones_like(phi, dtype=np.float64)
    return np.asarray(np.exp(kappa*np.cos(2.0*phi))/i0(kappa), dtype=np.float64)


def alignment_weights(phi_offset: ArrayLike, Qoff: float, family: str) -> FloatArray:
    """Return alignment weights for a named angular family."""
    if family == "linear":
        return linear_alignment_weights(phi_offset, Qoff)
    if family in {"von_mises", "vonmises"}:
        return von_mises_alignment_weights(phi_offset, Qoff)
    raise ValueError("family must be 'linear' or 'von_mises'")


def weighted_represented_fraction(safe_outer: ArrayLike, weights: ArrayLike) -> float:
    """Weighted outer represented fraction with original-population denominator."""
    safe = np.asarray(safe_outer, dtype=bool).reshape(-1)
    w = _flat_float(weights, "weights")
    if w.shape != safe.shape:
        raise ValueError("safe_outer and weights must have the same flattened shape")
    if np.any(w < 0.0) or np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(w[safe])/denom)


def empirical_Qoff(phi_offset: ArrayLike, weights: ArrayLike, selected_mask: ArrayLike | None = None) -> float:
    """Weighted empirical mean of cos(2 phi_offset)."""
    phi = _flat_float(phi_offset, "phi_offset")
    w = _flat_float(weights, "weights")
    if w.shape != phi.shape:
        raise ValueError("phi_offset and weights must have matching shapes")
    return weighted_mean(np.cos(2.0*phi), w, selected_mask)


def _find_index(values: ArrayLike, target: float, *, name: str, atol: float) -> int:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    matches = np.where(np.isclose(arr, float(target), rtol=0.0, atol=atol))[0]
    if matches.size != 1:
        raise ValueError(f"expected exactly one {name} matching {target}; found {matches.size}")
    return int(matches[0])


def _source_index_far_background(product: PopulationFieldProducts, source_label: str) -> int:
    if source_label != "infinity":
        raise ValueError("alignment Table-3 workflow requires far-background source_label='infinity'")
    return source_index(product, source_label)


def _baseline_fit(product: PopulationFieldProducts, sidx: int, sigidx: int, ridx: int, *, u_max: float | None):
    key = product.key_samples
    radius = float(np.asarray(key["R_over_r200c"], dtype=np.float64)[ridx])
    d = np.asarray(key["d_over_r200c"], dtype=np.float64)[sigidx]
    u = d/radius
    selected = np.asarray(key["safe_outer"], dtype=bool)[sigidx, :, ridx]
    return fit_hybrid_interaction(
        np.asarray(key["delta_g_plus_fractional"], dtype=np.float64)[sidx, sigidx, :, ridx],
        np.asarray(key["delta_g_shape_only"], dtype=np.float64)[sidx, sigidx, :, ridx],
        np.asarray(key["delta_g_offset_only"], dtype=np.float64)[sidx, sigidx, :, ridx],
        np.asarray(key["epsilon_perp"], dtype=np.float64),
        u,
        np.asarray(key["phi_offset"], dtype=np.float64),
        selected,
        u_max=u_max,
    )


def alignment_rows_from_product(
    product: PopulationFieldProducts,
    *,
    source_label: str = "infinity",
    q_values: Sequence[float] = DEFAULT_QOFF_VALUES,
    families: Sequence[str] = DEFAULT_ALIGNMENT_FAMILIES,
    sigma_values: Sequence[float] = DEFAULT_NONZERO_SIGMA_D,
    domain_specs: Sequence[tuple[str, float | None]] = DEFAULT_ALIGNMENT_DOMAINS,
) -> tuple[AlignmentRow, ...]:
    """Evaluate Section-6.4 alignment rows for one population product.

    Baseline predictor coefficients are fitted only from the unweighted Q=0
    sample for each grid/offset/radius/domain, then held fixed for all Q and
    angular-family reweightings.
    """
    sidx = _source_index_far_background(product, source_label)
    key = product.key_samples
    radii = np.asarray(key["R_over_r200c"], dtype=np.float64).reshape(-1)
    sigma_axis = np.asarray(key["sigma_d"], dtype=np.float64).reshape(-1)
    phi = np.asarray(key["phi_offset"], dtype=np.float64).reshape(-1)
    epsilon = np.asarray(key["epsilon_perp"], dtype=np.float64).reshape(-1)
    rows: list[AlignmentRow] = []
    for sigma in sigma_values:
        sigidx = _find_index(sigma_axis, float(sigma), name="sigma_d", atol=_SIGMA_ATOL)
        d = np.asarray(key["d_over_r200c"], dtype=np.float64)[sigidx]
        for ridx, radius in enumerate(radii):
            safe = np.asarray(key["safe_outer"], dtype=bool)[sigidx, :, ridx]
            y = np.asarray(key["delta_g_plus_fractional"], dtype=np.float64)[sidx, sigidx, :, ridx]
            shape = np.asarray(key["delta_g_shape_only"], dtype=np.float64)[sidx, sigidx, :, ridx]
            offset = np.asarray(key["delta_g_offset_only"], dtype=np.float64)[sidx, sigidx, :, ridx]
            u = d/float(radius)
            # One unweighted baseline fit per selected domain; reused for all family/Q rows.
            baseline_fits = {domain: _baseline_fit(product, sidx, sigidx, ridx, u_max=umax) for domain, umax in domain_specs}
            for family in families:
                for q in q_values:
                    weights = alignment_weights(phi, q, family)
                    rep_outer = weighted_represented_fraction(safe, weights)
                    q_original = empirical_Qoff(phi, weights)
                    for domain, umax in domain_specs:
                        selected = safe.copy()
                        if umax is not None:
                            selected &= u < float(umax)
                        fit = baseline_fits[domain]
                        summary = summarize_weighted_values(y*100.0, weights, selected)
                        r2 = weighted_r2_score(y, fit.prediction, weights, selected)
                        q_selected = empirical_Qoff(phi, weights, selected)
                        rows.append(AlignmentRow(
                            grid_index=int(product.grid_index),
                            mass_msun_h=float(product.mass_msun_h),
                            z_l=float(product.z_l),
                            source_label=str(source_label),
                            sigma_d=float(sigma),
                            R_over_r200c=float(radius),
                            family="von_mises" if family == "vonmises" else str(family),
                            Qoff=float(q),
                            domain=str(domain),
                            u_max=None if umax is None else float(umax),
                            n_total=int(y.size),
                            n_outer_safe=int(np.count_nonzero(safe)),
                            n_selected=int(summary.n_selected),
                            n_used=int(summary.n_used),
                            weighted_represented_fraction_outer=float(rep_outer),
                            weight_sum_original=float(np.sum(weights)),
                            weight_sum_selected=float(summary.weight_sum_selected),
                            effective_sample_size=float(summary.effective_sample_size),
                            empirical_Qoff_original=float(q_original),
                            empirical_Qoff_selected=float(q_selected),
                            p16_delta_g_percent=float(summary.p16),
                            p50_delta_g_percent=float(summary.p50),
                            p84_delta_g_percent=float(summary.p84),
                            a0_baseline=float(fit.a0),
                            a_int_baseline=float(fit.a_int),
                            weighted_r2=float(r2),
                        ))
    return tuple(rows)


def alignment_rows_from_products(products: Iterable[PopulationFieldProducts], **kwargs: object) -> tuple[AlignmentRow, ...]:
    rows: list[AlignmentRow] = []
    for product in products:
        rows.extend(alignment_rows_from_product(product, **kwargs))
    return tuple(rows)


def _row_matches(row: AlignmentRow, *, family: str, domain: str, sigma_d: float, radius: float) -> bool:
    return (
        row.family == family and row.domain == domain and
        np.isclose(row.sigma_d, sigma_d, rtol=0.0, atol=_SIGMA_ATOL) and
        np.isclose(row.R_over_r200c, radius, rtol=0.0, atol=_RADIUS_ATOL)
    )


def table3_records_from_alignment_rows(
    rows: Iterable[AlignmentRow],
    *,
    represented_threshold: float = DEFAULT_REPRESENTED_THRESHOLD,
    q_values_for_cut: Sequence[float] = DEFAULT_QOFF_VALUES,
    table_q_values: Sequence[float] = DEFAULT_TABLE3_QOFF,
    family: str = "linear",
    domain: str = "all_outer_safe",
    sigma_d: float = DEFAULT_TABLE3_SIGMA_D,
    radius: float = DEFAULT_TABLE3_RADIUS,
) -> tuple[Table3Record, ...]:
    """Select Table-3 records with the representation cut applied over Q range."""
    selected = [row for row in rows if _row_matches(row, family=family, domain=domain, sigma_d=sigma_d, radius=radius)]
    by_grid: dict[int, dict[float, AlignmentRow]] = {}
    for row in selected:
        by_grid.setdefault(row.grid_index, {})[round(row.Qoff, 12)] = row
    required_cut = {round(float(q), 12) for q in q_values_for_cut}
    required_table = {round(float(q), 12) for q in table_q_values}
    records: list[Table3Record] = []
    for grid in sorted(by_grid):
        entries = by_grid[grid]
        if not required_cut.issubset(entries):
            continue
        if any(entries[q].weighted_represented_fraction_outer < represented_threshold for q in required_cut):
            continue
        if not required_table.issubset(entries):
            continue
        qminus = entries[round(-0.30, 12)]
        qzero = entries[round(0.0, 12)]
        qplus = entries[round(0.30, 12)]
        records.append(Table3Record(
            grid_index=int(grid),
            mass_msun_h=float(qzero.mass_msun_h),
            z_l=float(qzero.z_l),
            R_over_r200c=float(qzero.R_over_r200c),
            sigma_d=float(qzero.sigma_d),
            p50_Qoff_minus_0p30_percent=float(qminus.p50_delta_g_percent),
            p50_Qoff_0p00_percent=float(qzero.p50_delta_g_percent),
            p50_Qoff_plus_0p30_percent=float(qplus.p50_delta_g_percent),
        ))
    return tuple(records)


def alignment_diagnostics(rows: Iterable[AlignmentRow]) -> dict[str, float | int | None]:
    """Return compact Section-6.4 diagnostic ranges from alignment rows."""
    all_rows = tuple(rows)
    table = table3_records_from_alignment_rows(all_rows)

    q_changes = []
    sigma_changes = []
    if table:
        table_grids = {record.grid_index for record in table}
        lookup = {(row.grid_index, row.family, row.domain, round(row.sigma_d, 12), round(row.R_over_r200c, 12), round(row.Qoff, 12)): row for row in all_rows}
        for grid in sorted(table_grids):
            base = lookup.get((grid, "linear", "all_outer_safe", round(0.05, 12), round(DEFAULT_TABLE3_RADIUS, 12), round(0.0, 12)))
            plus = lookup.get((grid, "linear", "all_outer_safe", round(0.05, 12), round(DEFAULT_TABLE3_RADIUS, 12), round(0.30, 12)))
            low = lookup.get((grid, "linear", "all_outer_safe", round(0.02, 12), round(DEFAULT_TABLE3_RADIUS, 12), round(0.0, 12)))
            high = lookup.get((grid, "linear", "all_outer_safe", round(0.08, 12), round(DEFAULT_TABLE3_RADIUS, 12), round(0.0, 12)))
            if base and plus:
                q_changes.append(float(plus.p50_delta_g_percent - base.p50_delta_g_percent))
            if low and high:
                sigma_changes.append(float(high.p50_delta_g_percent - low.p50_delta_g_percent))

    # Compare linear and von-Mises p50 values for rows that both exist and meet 95% representation.
    lin = {}
    vm = {}
    for row in all_rows:
        if row.domain != "all_outer_safe" or row.weighted_represented_fraction_outer < DEFAULT_REPRESENTED_THRESHOLD:
            continue
        key = (row.grid_index, round(row.sigma_d, 12), round(row.R_over_r200c, 12), round(row.Qoff, 12))
        if row.family == "linear":
            lin[key] = row.p50_delta_g_percent
        elif row.family == "von_mises":
            vm[key] = row.p50_delta_g_percent
    family_diffs = [abs(lin[k] - vm[k]) for k in lin.keys() & vm.keys()]

    restricted = [row.weighted_r2 for row in all_rows if row.family == "linear" and row.domain == "u_lt_0p3" and row.n_used >= 100 and np.isfinite(row.weighted_r2)]
    restricted_arr = np.asarray(restricted, dtype=np.float64)
    return {
        "n_rows": len(all_rows),
        "n_table3_records": len(table),
        "q_change_min_percent_points": float(np.min(q_changes)) if q_changes else None,
        "q_change_max_percent_points": float(np.max(q_changes)) if q_changes else None,
        "sigma_change_min_percent_points": float(np.min(sigma_changes)) if sigma_changes else None,
        "sigma_change_max_percent_points": float(np.max(sigma_changes)) if sigma_changes else None,
        "max_family_difference_percent_points": float(np.max(family_diffs)) if family_diffs else None,
        "restricted_linear_r2_count": int(restricted_arr.size),
        "restricted_linear_r2_min": float(np.min(restricted_arr)) if restricted_arr.size else None,
        "restricted_linear_r2_median": float(np.median(restricted_arr)) if restricted_arr.size else None,
    }


def write_alignment_rows_csv(rows: Iterable[AlignmentRow], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_table3_csv(records: Iterable[Table3Record], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE3_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_dict())
