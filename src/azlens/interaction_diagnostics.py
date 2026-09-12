"""Unweighted reduced-shear interaction diagnostics for the paper.

This module implements the Section 6.1 hybrid shape--centering predictor and
Section 6.2 ratio-conditioning diagnostic.  It consumes already generated
key-radius population products; it does not evaluate ring fields, apply
alignment weights, render figures, or choose Table-2 rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .population_fields import PopulationFieldProducts
from .statistics import r2_score

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_INTERACTION_SOURCE_LABEL = "z1"
DEFAULT_INTERACTION_SOURCE_Z = 1.0
DEFAULT_NONZERO_SIGMA_D = (0.02, 0.05, 0.08)
DEFAULT_U_MAX = 0.3
_DOMAIN_ALL = "all_outer_safe"
_DOMAIN_U = "u_lt_0p3"
_RADIUS_ATOL = 5.0e-15
_SIGMA_ATOL = 5.0e-15

HYBRID_COLUMNS = (
    "grid_index", "mass_msun_h", "z_l", "source_label", "source_z",
    "sigma_d", "R_over_r200c", "domain", "u_max", "n_total",
    "n_selected", "n_used", "a0", "a_int", "r2",
)

CONDITIONING_COLUMNS = (
    "grid_index", "mass_msun_h", "z_l", "source_label", "source_z",
    "sigma_d", "R_over_r200c", "n_total", "n_selected", "n_used",
    "r2_fractional", "r2_difference",
)


@dataclass(frozen=True)
class HybridInteractionFit:
    """Result of one Section-6.1 hybrid interaction fit."""

    a0: float
    a_int: float
    r2: float
    n_total: int
    n_selected: int
    n_used: int
    prediction: FloatArray
    selected_finite_mask: BoolArray

    @property
    def coefficients(self) -> FloatArray:
        return np.asarray([self.a0, self.a_int], dtype=np.float64)


@dataclass(frozen=True)
class HybridInteractionRow:
    """Serializable metadata and fit result for one grid/offset/radius/domain."""

    grid_index: int
    mass_msun_h: float
    z_l: float
    source_label: str
    source_z: float
    sigma_d: float
    R_over_r200c: float
    domain: str
    u_max: float | None
    n_total: int
    n_selected: int
    n_used: int
    a0: float
    a_int: float
    r2: float

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "grid_index": int(self.grid_index),
            "mass_msun_h": float(self.mass_msun_h),
            "z_l": float(self.z_l),
            "source_label": self.source_label,
            "source_z": float(self.source_z),
            "sigma_d": float(self.sigma_d),
            "R_over_r200c": float(self.R_over_r200c),
            "domain": self.domain,
            "u_max": None if self.u_max is None else float(self.u_max),
            "n_total": int(self.n_total),
            "n_selected": int(self.n_selected),
            "n_used": int(self.n_used),
            "a0": float(self.a0),
            "a_int": float(self.a_int),
            "r2": float(self.r2),
        }


@dataclass(frozen=True)
class ConditioningDiagnostic:
    """No-fit Section-6.2 comparison for one selected configuration."""

    r2_fractional: float
    r2_difference: float
    n_total: int
    n_selected: int
    n_used: int
    selected_finite_mask: BoolArray


@dataclass(frozen=True)
class ConditioningRow:
    """Serializable metadata and no-fit conditioning diagnostic."""

    grid_index: int
    mass_msun_h: float
    z_l: float
    source_label: str
    source_z: float
    sigma_d: float
    R_over_r200c: float
    n_total: int
    n_selected: int
    n_used: int
    r2_fractional: float
    r2_difference: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "grid_index": int(self.grid_index),
            "mass_msun_h": float(self.mass_msun_h),
            "z_l": float(self.z_l),
            "source_label": self.source_label,
            "source_z": float(self.source_z),
            "sigma_d": float(self.sigma_d),
            "R_over_r200c": float(self.R_over_r200c),
            "n_total": int(self.n_total),
            "n_selected": int(self.n_selected),
            "n_used": int(self.n_used),
            "r2_fractional": float(self.r2_fractional),
            "r2_difference": float(self.r2_difference),
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


def _same_shape(*named_arrays: tuple[str, ArrayLike]) -> dict[str, FloatArray]:
    converted = {name: _flat_float(values, name) for name, values in named_arrays}
    shapes = {arr.shape for arr in converted.values()}
    if len(shapes) != 1:
        raise ValueError("all numerical inputs must have the same flattened shape")
    return converted


def _selected_finite_mask(arrays: Mapping[str, FloatArray], selected_mask: ArrayLike | None) -> tuple[BoolArray, int, int]:
    shape = next(iter(arrays.values())).shape
    if selected_mask is None:
        selected = np.ones(shape, dtype=bool)
    else:
        selected = _flat_bool(selected_mask, "selected_mask", shape)
    finite = selected.copy()
    for values in arrays.values():
        finite &= np.isfinite(values)
    return finite, int(np.count_nonzero(selected)), int(np.count_nonzero(finite))


def interaction_basis(epsilon_perp: ArrayLike, u: ArrayLike, phi_offset: ArrayLike) -> FloatArray:
    """Return the Section-6.1 orientation basis epsilon_perp*u^2*cos(2 phi_offset)."""
    arr = _same_shape(("epsilon_perp", epsilon_perp), ("u", u), ("phi_offset", phi_offset))
    return np.asarray(arr["epsilon_perp"]*arr["u"]**2*np.cos(2.0*arr["phi_offset"]), dtype=np.float64)


def fit_hybrid_interaction(
    delta_g_combined: ArrayLike,
    delta_g_shape: ArrayLike,
    delta_g_offset: ArrayLike,
    epsilon_perp: ArrayLike,
    u: ArrayLike,
    phi_offset: ArrayLike,
    selected_mask: ArrayLike | None = None,
    *,
    u_max: float | None = None,
    rcond: float | None = None,
) -> HybridInteractionFit:
    """Fit the unweighted Section-6.1 hybrid predictor.

    The coefficient multiplying the additive shape-only plus offset-only
    residual is fixed to unity.  Only an intercept and the
    ``epsilon_perp*u**2*cos(2*phi_offset)`` basis coefficient are fitted.
    """
    arr = _same_shape(
        ("delta_g_combined", delta_g_combined),
        ("delta_g_shape", delta_g_shape),
        ("delta_g_offset", delta_g_offset),
        ("epsilon_perp", epsilon_perp),
        ("u", u),
        ("phi_offset", phi_offset),
    )
    selected = np.ones_like(arr["delta_g_combined"], dtype=bool)
    if selected_mask is not None:
        selected &= _flat_bool(selected_mask, "selected_mask", selected.shape)
    if u_max is not None:
        umax = float(u_max)
        if not np.isfinite(umax) or umax <= 0.0:
            raise ValueError("u_max must be finite and positive when supplied")
        selected &= arr["u"] < umax
    basis = interaction_basis(arr["epsilon_perp"], arr["u"], arr["phi_offset"])
    additive = arr["delta_g_shape"] + arr["delta_g_offset"]
    target = arr["delta_g_combined"] - additive
    arrays = {"target": target, "basis": basis, "combined": arr["delta_g_combined"], "additive": additive}
    finite_mask, n_selected, n_used = _selected_finite_mask(arrays, selected)
    if n_used < 2:
        raise ValueError("not enough selected finite halos to fit a0 and a_int")
    design = np.column_stack((np.ones_like(basis[finite_mask]), basis[finite_mask]))
    coeff, _, rank, singular = np.linalg.lstsq(design, target[finite_mask], rcond=rcond)
    if rank < 2 or np.any(~np.isfinite(singular)):
        raise ValueError("hybrid interaction design matrix is rank deficient")
    full_prediction = additive + coeff[0] + coeff[1]*basis
    r2 = r2_score(arr["delta_g_combined"], full_prediction, selected_mask=finite_mask)
    return HybridInteractionFit(
        a0=float(coeff[0]),
        a_int=float(coeff[1]),
        r2=float(r2),
        n_total=int(arr["delta_g_combined"].size),
        n_selected=n_selected,
        n_used=n_used,
        prediction=np.asarray(full_prediction, dtype=np.float64),
        selected_finite_mask=finite_mask,
    )


def conditioning_r2_diagnostic(
    delta_g_fractional: ArrayLike,
    delta_g_fractional_second_order: ArrayLike,
    Delta_g: ArrayLike,
    Delta_g_second_order: ArrayLike,
    selected_mask: ArrayLike | None = None,
) -> ConditioningDiagnostic:
    """Evaluate the no-fit Section-6.2 fractional/difference R^2 diagnostic."""
    arr = _same_shape(
        ("delta_g_fractional", delta_g_fractional),
        ("delta_g_fractional_second_order", delta_g_fractional_second_order),
        ("Delta_g", Delta_g),
        ("Delta_g_second_order", Delta_g_second_order),
    )
    joint, n_selected, n_used = _selected_finite_mask(arr, selected_mask)
    if n_used == 0:
        return ConditioningDiagnostic(np.nan, np.nan, int(arr["delta_g_fractional"].size), n_selected, 0, joint)
    return ConditioningDiagnostic(
        r2_fractional=r2_score(arr["delta_g_fractional"], arr["delta_g_fractional_second_order"], selected_mask=joint),
        r2_difference=r2_score(arr["Delta_g"], arr["Delta_g_second_order"], selected_mask=joint),
        n_total=int(arr["delta_g_fractional"].size),
        n_selected=n_selected,
        n_used=n_used,
        selected_finite_mask=joint,
    )


def _find_float_index(values: ArrayLike, target: float, *, name: str, atol: float) -> int:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    matches = np.where(np.isclose(arr, float(target), rtol=0.0, atol=atol))[0]
    if matches.size != 1:
        raise ValueError(f"expected exactly one {name} matching {target}; found {matches.size}")
    return int(matches[0])


def source_axis_labels(product: PopulationFieldProducts) -> tuple[str, ...]:
    """Infer key-sample source-axis labels from the product metadata."""
    finite = tuple(f"z{float(z):g}" for z in product.source_redshifts)
    weights = np.asarray(product.key_samples.get("source_weights", ()), dtype=np.float64).reshape(-1)
    if weights.size == len(finite) + 1:
        return finite + ("infinity",)
    if weights.size == len(finite):
        return finite
    raise ValueError("cannot infer source-axis labels from product source weights")


def source_index(product: PopulationFieldProducts, source_label: str = DEFAULT_INTERACTION_SOURCE_LABEL) -> int:
    labels = source_axis_labels(product)
    try:
        return labels.index(str(source_label))
    except ValueError as exc:
        raise ValueError(f"source label {source_label!r} not present; available labels are {labels}") from exc


def _sigma_index(product: PopulationFieldProducts, sigma_d: float) -> int:
    return _find_float_index(product.key_samples["sigma_d"], sigma_d, name="sigma_d", atol=_SIGMA_ATOL)


def _radius_index(product: PopulationFieldProducts, radius: float) -> int:
    return _find_float_index(product.key_samples["R_over_r200c"], radius, name="key radius", atol=_RADIUS_ATOL)


def hybrid_fit_rows_from_product(
    product: PopulationFieldProducts,
    *,
    source_label: str = DEFAULT_INTERACTION_SOURCE_LABEL,
    source_z: float = DEFAULT_INTERACTION_SOURCE_Z,
    sigma_values: Sequence[float] = DEFAULT_NONZERO_SIGMA_D,
    domain_specs: Sequence[tuple[str, float | None]] = ((_DOMAIN_ALL, None), (_DOMAIN_U, DEFAULT_U_MAX)),
) -> tuple[HybridInteractionRow, ...]:
    """Run Section-6.1 hybrid fits for one product's key-radius samples."""
    sidx = source_index(product, source_label)
    key = product.key_samples
    radii = np.asarray(key["R_over_r200c"], dtype=np.float64).reshape(-1)
    epsilon = np.asarray(key["epsilon_perp"], dtype=np.float64).reshape(-1)
    phi = np.asarray(key["phi_offset"], dtype=np.float64).reshape(-1)
    rows: list[HybridInteractionRow] = []
    for sigma in sigma_values:
        sigidx = _sigma_index(product, float(sigma))
        for ridx, radius in enumerate(radii):
            d = np.asarray(key["d_over_r200c"], dtype=np.float64)[sigidx]
            u = d/float(radius)
            selected = np.asarray(key["safe_outer"], dtype=bool)[sigidx, :, ridx]
            combined = np.asarray(key["delta_g_plus_fractional"], dtype=np.float64)[sidx, sigidx, :, ridx]
            shape = np.asarray(key["delta_g_shape_only"], dtype=np.float64)[sidx, sigidx, :, ridx]
            offset = np.asarray(key["delta_g_offset_only"], dtype=np.float64)[sidx, sigidx, :, ridx]
            for domain, umax in domain_specs:
                fit = fit_hybrid_interaction(combined, shape, offset, epsilon, u, phi, selected, u_max=umax)
                rows.append(HybridInteractionRow(
                    grid_index=int(product.grid_index),
                    mass_msun_h=float(product.mass_msun_h),
                    z_l=float(product.z_l),
                    source_label=str(source_label),
                    source_z=float(source_z),
                    sigma_d=float(sigma),
                    R_over_r200c=float(radius),
                    domain=str(domain),
                    u_max=None if umax is None else float(umax),
                    n_total=fit.n_total,
                    n_selected=fit.n_selected,
                    n_used=fit.n_used,
                    a0=fit.a0,
                    a_int=fit.a_int,
                    r2=fit.r2,
                ))
    return tuple(rows)


def hybrid_fit_rows_from_products(
    products: Iterable[PopulationFieldProducts],
    **kwargs: object,
) -> tuple[HybridInteractionRow, ...]:
    rows: list[HybridInteractionRow] = []
    for product in products:
        rows.extend(hybrid_fit_rows_from_product(product, **kwargs))
    return tuple(rows)


def restricted_hybrid_r2_summary(rows: Iterable[HybridInteractionRow], domain: str = _DOMAIN_U) -> dict[str, float | int | str]:
    """Summarize R^2 values for a named hybrid-fit domain."""
    selected = [row.r2 for row in rows if row.domain == domain and np.isfinite(row.r2)]
    if not selected:
        return {"domain": domain, "n_fits": 0, "r2_min": np.nan, "r2_median": np.nan}
    arr = np.asarray(selected, dtype=np.float64)
    return {"domain": domain, "n_fits": int(arr.size), "r2_min": float(np.min(arr)), "r2_median": float(np.median(arr))}


def conditioning_rows_from_product(
    product: PopulationFieldProducts,
    *,
    source_label: str = DEFAULT_INTERACTION_SOURCE_LABEL,
    source_z: float = DEFAULT_INTERACTION_SOURCE_Z,
    sigma_d: float = 0.08,
    radii: Sequence[float] | None = None,
) -> tuple[ConditioningRow, ...]:
    """Evaluate Section-6.2 no-fit diagnostics for selected product radii."""
    sidx = source_index(product, source_label)
    sigidx = _sigma_index(product, sigma_d)
    key = product.key_samples
    available = np.asarray(key["R_over_r200c"], dtype=np.float64).reshape(-1)
    if radii is None:
        radius_indices = tuple(range(available.size))
    else:
        radius_indices = tuple(_radius_index(product, float(r)) for r in radii)
    rows: list[ConditioningRow] = []
    for ridx in radius_indices:
        selected = np.asarray(key["safe_outer"], dtype=bool)[sigidx, :, ridx]
        diag = conditioning_r2_diagnostic(
            np.asarray(key["delta_g_plus_fractional"], dtype=np.float64)[sidx, sigidx, :, ridx],
            np.asarray(key["delta_g_plus_second_order"], dtype=np.float64)[sidx, sigidx, :, ridx],
            np.asarray(key["Delta_g_plus"], dtype=np.float64)[sidx, sigidx, :, ridx],
            np.asarray(key["Delta_g_plus_second_order"], dtype=np.float64)[sidx, sigidx, :, ridx],
            selected,
        )
        rows.append(ConditioningRow(
            grid_index=int(product.grid_index),
            mass_msun_h=float(product.mass_msun_h),
            z_l=float(product.z_l),
            source_label=str(source_label),
            source_z=float(source_z),
            sigma_d=float(sigma_d),
            R_over_r200c=float(available[ridx]),
            n_total=diag.n_total,
            n_selected=diag.n_selected,
            n_used=diag.n_used,
            r2_fractional=diag.r2_fractional,
            r2_difference=diag.r2_difference,
        ))
    return tuple(rows)


def conditioning_rows_from_products(
    products: Iterable[PopulationFieldProducts],
    **kwargs: object,
) -> tuple[ConditioningRow, ...]:
    rows: list[ConditioningRow] = []
    for product in products:
        rows.extend(conditioning_rows_from_product(product, **kwargs))
    return tuple(rows)


def write_hybrid_rows_csv(rows: Iterable[HybridInteractionRow], path: str | Path) -> None:
    """Write hybrid-fit rows to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HYBRID_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_conditioning_rows_csv(rows: Iterable[ConditioningRow], path: str | Path) -> None:
    """Write conditioning-diagnostic rows to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONDITIONING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def compact_interaction_metrics(
    hybrid_rows: Iterable[HybridInteractionRow],
    conditioning_rows: Iterable[ConditioningRow] = (),
) -> dict[str, object]:
    """Return concise report-friendly metrics for validation scripts."""
    hrows = tuple(hybrid_rows)
    crows = tuple(conditioning_rows)
    return {
        "n_hybrid_rows": len(hrows),
        "restricted_hybrid": restricted_hybrid_r2_summary(hrows),
        "conditioning_rows": [row.as_dict() for row in crows],
    }
