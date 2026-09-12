"""Publication radial-summary and Table 2 selection logic.

This module is downstream of population field evaluation.  It consumes rows or
``PopulationFieldProducts`` objects, identifies the reference common radial
domain, and selects the Table 2 rows.  It does not evaluate lensing fields,
construct Figure 3, run Section 6 diagnostics, or perform alignment reweighting.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .population_fields import PopulationFieldProducts

FloatArray = NDArray[np.float64]

TABLE2_OBSERVABLES: tuple[str, ...] = (
    "reduced_tangential_shear",
    "inverse_magnification",
    "magnification",
    "magnification_bias_alpha_0p3",
    "magnification_bias_alpha_1p4",
)
TABLE2_SOURCE_LABELS: tuple[str, ...] = ("z1", "z2")
DEFAULT_TABLE2_SIGMA_D = 0.05
DEFAULT_COMMON_DOMAIN_THRESHOLD = 0.95
_RADIUS_ATOL = 1.0e-14


@dataclass(frozen=True)
class CommonDomainResult:
    """Result of applying the reference common-domain represented-fraction cut."""

    sigma_d: float
    threshold: float
    grid_indices: tuple[int, ...]
    radii: FloatArray
    pass_indices: NDArray[np.int64]
    represented_fraction_matrix: FloatArray
    limiting_grid_index: int | None
    limiting_radius: float
    limiting_fraction: float

    @property
    def n_radii(self) -> int:
        return int(self.radii.size)

    @property
    def r_min(self) -> float:
        return float("nan") if self.radii.size == 0 else float(self.radii[0])

    @property
    def r_max(self) -> float:
        return float("nan") if self.radii.size == 0 else float(self.radii[-1])

    def contains_radius(self, radius: float, *, atol: float = _RADIUS_ATOL) -> bool:
        if self.radii.size == 0:
            return False
        return bool(np.any(np.isclose(self.radii, float(radius), rtol=0.0, atol=atol)))

    def as_dict(self) -> dict[str, object]:
        return {
            "sigma_d": float(self.sigma_d),
            "threshold": float(self.threshold),
            "grid_indices": [int(g) for g in self.grid_indices],
            "radii": [float(r) for r in self.radii],
            "pass_indices": [int(i) for i in self.pass_indices],
            "n_radii": self.n_radii,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "limiting_grid_index": None if self.limiting_grid_index is None else int(self.limiting_grid_index),
            "limiting_radius": float(self.limiting_radius),
            "limiting_fraction": float(self.limiting_fraction),
        }


@dataclass(frozen=True)
class Table2Record:
    """Selected row for one observable/source-plane entry of Table 2."""

    observable: str
    source_label: str
    source_z: float
    grid_index: int
    mass_msun_h: float
    z_l: float
    sigma_d: float
    R_over_r200c: float
    n_total: int
    n_outer_safe: int
    n_finite: int
    represented_fraction_outer: float
    p16_fractional: float
    p50_fractional: float
    p84_fractional: float

    @property
    def p16_percent(self) -> float:
        return 100.0*self.p16_fractional

    @property
    def p50_percent(self) -> float:
        return 100.0*self.p50_fractional

    @property
    def p84_percent(self) -> float:
        return 100.0*self.p84_fractional

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "observable": self.observable,
            "source_label": self.source_label,
            "source_z": float(self.source_z),
            "grid_index": int(self.grid_index),
            "mass_msun_h": float(self.mass_msun_h),
            "z_l": float(self.z_l),
            "sigma_d": float(self.sigma_d),
            "R_over_r200c": float(self.R_over_r200c),
            "n_total": int(self.n_total),
            "n_outer_safe": int(self.n_outer_safe),
            "n_finite": int(self.n_finite),
            "represented_fraction_outer": float(self.represented_fraction_outer),
            "p16_fractional": float(self.p16_fractional),
            "p50_fractional": float(self.p50_fractional),
            "p84_fractional": float(self.p84_fractional),
            "p16_percent": self.p16_percent,
            "p50_percent": self.p50_percent,
            "p84_percent": self.p84_percent,
        }


@dataclass(frozen=True)
class PublicationSummaryResult:
    """Compact result for the unweighted publication-summary/Table 2 layer."""

    common_domain: CommonDomainResult
    table2_records: tuple[Table2Record, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "common_domain": self.common_domain.as_dict(),
            "table2_records": [record.as_dict() for record in self.table2_records],
        }


def _row_float(row: Mapping[str, object], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise KeyError(f"missing required row field {key!r}") from exc


def _row_int(row: Mapping[str, object], key: str) -> int:
    return int(round(_row_float(row, key)))


def _row_str(row: Mapping[str, object], key: str) -> str:
    try:
        return str(row[key])
    except KeyError as exc:
        raise KeyError(f"missing required row field {key!r}") from exc


def radial_summary_rows_from_products(products: Iterable[PopulationFieldProducts]) -> tuple[dict[str, float | int | str], ...]:
    """Flatten radial-summary rows from one or more population-field products."""
    rows: list[dict[str, float | int | str]] = []
    for product in products:
        rows.extend(dict(row) for row in product.radial_summary_rows)
    return tuple(rows)


def _sigma_matches(value: float, sigma_d: float, atol: float = 1.0e-14) -> bool:
    return bool(np.isclose(float(value), float(sigma_d), rtol=0.0, atol=atol))


def common_domain_from_products(
    products: Sequence[PopulationFieldProducts],
    *,
    sigma_d: float = DEFAULT_TABLE2_SIGMA_D,
    threshold: float = DEFAULT_COMMON_DOMAIN_THRESHOLD,
) -> CommonDomainResult:
    """Identify radii retained at the requested fraction in every supplied grid."""
    if not products:
        raise ValueError("at least one product is required")
    radii = np.asarray(products[0].radial_grid, dtype=np.float64)
    grid_indices: list[int] = []
    represented_rows: list[FloatArray] = []
    sigma = float(sigma_d)
    for product in products:
        if not np.allclose(product.radial_grid, radii, rtol=0.0, atol=_RADIUS_ATOL):
            raise ValueError("all products must share the same radial grid")
        sigma_values = np.asarray(product.sigma_d, dtype=np.float64)
        matches = np.where(np.isclose(sigma_values, sigma, rtol=0.0, atol=1.0e-14))[0]
        if matches.size != 1:
            raise ValueError("each product must contain exactly one requested sigma_d entry")
        grid_indices.append(int(product.grid_index))
        represented_rows.append(np.asarray(product.represented_fraction_outer[int(matches[0])], dtype=np.float64))
    return _common_domain_from_matrix(
        radii,
        tuple(grid_indices),
        np.vstack(represented_rows),
        sigma_d=sigma,
        threshold=threshold,
    )


def common_domain_from_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    sigma_d: float = DEFAULT_TABLE2_SIGMA_D,
    threshold: float = DEFAULT_COMMON_DOMAIN_THRESHOLD,
) -> CommonDomainResult:
    """Identify the common radial domain from radial-summary rows.

    Duplicate observable/source rows are deduplicated by ``(grid_index,radius)``
    and must agree in represented fraction.
    """
    sigma = float(sigma_d)
    rep: dict[tuple[int, float], float] = {}
    radii_set: set[float] = set()
    grids_set: set[int] = set()
    for row in rows:
        if not _sigma_matches(_row_float(row, "sigma_d"), sigma):
            continue
        grid = _row_int(row, "grid_index")
        radius = _row_float(row, "R_over_r200c")
        value = _row_float(row, "represented_fraction_outer")
        key = (grid, radius)
        if key in rep and not np.isclose(rep[key], value, rtol=0.0, atol=1.0e-14):
            raise ValueError("inconsistent represented_fraction_outer for duplicate grid/radius rows")
        rep[key] = value
        grids_set.add(grid)
        radii_set.add(radius)
    if not grids_set or not radii_set:
        raise ValueError("no rows found for the requested sigma_d")
    grids = tuple(sorted(grids_set))
    radii = np.asarray(sorted(radii_set), dtype=np.float64)
    matrix = np.full((len(grids), radii.size), np.nan, dtype=np.float64)
    grid_to_i = {grid: i for i, grid in enumerate(grids)}
    for (grid, radius), value in rep.items():
        j = int(np.where(np.isclose(radii, radius, rtol=0.0, atol=_RADIUS_ATOL))[0][0])
        matrix[grid_to_i[grid], j] = value
    if np.any(~np.isfinite(matrix)):
        raise ValueError("missing represented-fraction entries for one or more grid/radius pairs")
    return _common_domain_from_matrix(radii, grids, matrix, sigma_d=sigma, threshold=threshold)


def _common_domain_from_matrix(
    radii: ArrayLike,
    grid_indices: tuple[int, ...],
    represented_matrix: ArrayLike,
    *,
    sigma_d: float,
    threshold: float,
) -> CommonDomainResult:
    r = np.asarray(radii, dtype=np.float64).reshape(-1)
    matrix = np.asarray(represented_matrix, dtype=np.float64)
    if r.size == 0 or matrix.shape != (len(grid_indices), r.size):
        raise ValueError("represented_matrix must have shape (n_grid,n_radius)")
    if np.any(~np.isfinite(r)) or np.any(np.diff(r) <= 0.0) or np.any(~np.isfinite(matrix)):
        raise ValueError("radii and represented fractions must be finite and ordered")
    cut = float(threshold)
    if not np.isfinite(cut):
        raise ValueError("threshold must be finite")
    pass_mask = np.all(matrix >= cut, axis=0)
    pass_indices = np.where(pass_mask)[0].astype(np.int64)
    selected_radii = r[pass_indices]
    if pass_indices.size == 0:
        limiting_grid = None
        limiting_radius = float("nan")
        limiting_fraction = float("nan")
    else:
        selected_matrix = matrix[:, pass_indices]
        flat_index = int(np.argmin(selected_matrix))
        grid_pos, rad_pos = np.unravel_index(flat_index, selected_matrix.shape)
        limiting_grid = int(grid_indices[grid_pos])
        limiting_radius = float(selected_radii[rad_pos])
        limiting_fraction = float(selected_matrix[grid_pos, rad_pos])
    return CommonDomainResult(
        sigma_d=float(sigma_d),
        threshold=cut,
        grid_indices=tuple(int(g) for g in grid_indices),
        radii=np.asarray(selected_radii, dtype=np.float64),
        pass_indices=pass_indices,
        represented_fraction_matrix=np.asarray(matrix, dtype=np.float64),
        limiting_grid_index=limiting_grid,
        limiting_radius=limiting_radius,
        limiting_fraction=limiting_fraction,
    )


def select_table2_rows(
    rows: Iterable[Mapping[str, object]],
    common_domain: CommonDomainResult,
    *,
    sigma_d: float = DEFAULT_TABLE2_SIGMA_D,
    observables: Sequence[str] = TABLE2_OBSERVABLES,
    source_labels: Sequence[str] = TABLE2_SOURCE_LABELS,
) -> tuple[Table2Record, ...]:
    """Select the Table 2 extremal-median rows.

    For each observable/source pair the row maximizing ``abs(p50_fractional)``
    is selected.  The associated P16/P84 values are taken from that same row.
    """
    all_rows = [dict(row) for row in rows]
    selected: list[Table2Record] = []
    for observable in observables:
        for source_label in source_labels:
            candidates = []
            for row in all_rows:
                if _row_str(row, "observable") != observable:
                    continue
                if _row_str(row, "source_label") != source_label:
                    continue
                if not _sigma_matches(_row_float(row, "sigma_d"), sigma_d):
                    continue
                radius = _row_float(row, "R_over_r200c")
                if not common_domain.contains_radius(radius):
                    continue
                p50 = _row_float(row, "p50_fractional")
                if np.isfinite(p50):
                    candidates.append((abs(p50), len(candidates), row))
            if not candidates:
                raise ValueError(f"no finite candidate row for {observable} at {source_label}")
            # Python's max is stable with the explicit counter as a deterministic tie breaker.
            _, _, row = max(candidates, key=lambda item: (item[0], -item[1]))
            selected.append(_table2_record_from_row(row))
    return tuple(selected)


def _table2_record_from_row(row: Mapping[str, object]) -> Table2Record:
    return Table2Record(
        observable=_row_str(row, "observable"),
        source_label=_row_str(row, "source_label"),
        source_z=_row_float(row, "source_z"),
        grid_index=_row_int(row, "grid_index"),
        mass_msun_h=_row_float(row, "mass_msun_h"),
        z_l=_row_float(row, "z_l"),
        sigma_d=_row_float(row, "sigma_d"),
        R_over_r200c=_row_float(row, "R_over_r200c"),
        n_total=_row_int(row, "n_total"),
        n_outer_safe=_row_int(row, "n_outer_safe"),
        n_finite=_row_int(row, "n_finite"),
        represented_fraction_outer=_row_float(row, "represented_fraction_outer"),
        p16_fractional=_row_float(row, "p16_fractional"),
        p50_fractional=_row_float(row, "p50_fractional"),
        p84_fractional=_row_float(row, "p84_fractional"),
    )


def build_publication_summary_from_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    sigma_d: float = DEFAULT_TABLE2_SIGMA_D,
    threshold: float = DEFAULT_COMMON_DOMAIN_THRESHOLD,
) -> PublicationSummaryResult:
    """Return common-domain and Table 2 selections from supplied summary rows."""
    cached = [dict(row) for row in rows]
    domain = common_domain_from_rows(cached, sigma_d=sigma_d, threshold=threshold)
    records = select_table2_rows(cached, domain, sigma_d=sigma_d)
    return PublicationSummaryResult(domain, records)


def build_publication_summary_from_products(
    products: Sequence[PopulationFieldProducts],
    *,
    sigma_d: float = DEFAULT_TABLE2_SIGMA_D,
    threshold: float = DEFAULT_COMMON_DOMAIN_THRESHOLD,
) -> PublicationSummaryResult:
    """Return common-domain and Table 2 selections from field products."""
    domain = common_domain_from_products(products, sigma_d=sigma_d, threshold=threshold)
    rows = radial_summary_rows_from_products(products)
    records = select_table2_rows(rows, domain, sigma_d=sigma_d)
    return PublicationSummaryResult(domain, records)


def write_radial_summary_csv(rows: Iterable[Mapping[str, object]], path: str | Path) -> None:
    """Write radial-summary rows to CSV with a stable union of columns."""
    rows_list = [dict(row) for row in rows]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        target.write_text("", encoding="utf-8")
        return
    preferred = [
        "grid_index", "mass_msun_h", "z_l", "source_label", "source_z", "sigma_d", "R_over_r200c", "observable",
        "n_total", "n_outer_safe", "n_selected", "n_finite", "represented_fraction_outer", "finite_fraction_selected",
        "mean_fractional", "p16_fractional", "p50_fractional", "p84_fractional", "p97p5_fractional",
    ]
    extra = sorted({key for row in rows_list for key in row} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows_list)] + extra
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def read_radial_summary_csv(path: str | Path) -> tuple[dict[str, float | int | str], ...]:
    """Read radial-summary rows written by :func:`write_radial_summary_csv`."""
    source = Path(path)
    if source.stat().st_size == 0:
        return tuple()
    numeric_int = {"grid_index", "n_total", "n_outer_safe", "n_selected", "n_finite"}
    numeric_float = {
        "mass_msun_h", "z_l", "source_z", "sigma_d", "R_over_r200c", "represented_fraction_outer",
        "finite_fraction_selected", "mean_fractional", "p16_fractional", "p50_fractional", "p84_fractional", "p97p5_fractional",
    }
    rows = []
    with source.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, float | int | str] = {}
            for key, value in row.items():
                if key in numeric_int:
                    converted[key] = int(value)
                elif key in numeric_float:
                    converted[key] = float(value)
                else:
                    converted[key] = value
            rows.append(converted)
    return tuple(rows)


def table2_records_to_matrix(records: Sequence[Table2Record]) -> dict[str, dict[str, dict[str, float]]]:
    """Return a compact nested dictionary of Table 2 percent values."""
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for record in records:
        matrix.setdefault(record.observable, {})[record.source_label] = {
            "p16_percent": record.p16_percent,
            "p50_percent": record.p50_percent,
            "p84_percent": record.p84_percent,
            "grid_index": float(record.grid_index),
            "R_over_r200c": float(record.R_over_r200c),
        }
    return matrix
