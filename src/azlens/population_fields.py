"""Population ring-field evaluation and retained-domain products.

This layer consumes a validated halo/offset realization and a population mode
library.  It evaluates far-background fields, finite-source nonlinear
observables, sampled safety masks, radial summaries, and key-radius halo
catalogs.  It intentionally stops before final Table 2/Figure 3/Section 6
statistics and before alignment reweighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cosmology import PUBLICATION_COSMOLOGY, PublicationCosmology
from .miscentering import DEFAULT_SIGMA_D, OffsetRealization
from .mode_library import ModeLibrary
from .observables import RingObservableSummary, evaluate_ring_observables
from .population import DEFAULT_N_HALO, HaloPopulationRealization, PublicationRealization
from .ring_fields import CircularNFWFieldEvaluator, evaluate_ring_fields
from .selection import (
    DEFAULT_KEY_RADIUS_TARGETS,
    DEFAULT_SAFETY_THRESHOLD,
    key_radius_indices,
    key_radius_values,
    local_safety_mask,
    outer_contiguous_mask,
    publication_radial_grid,
    represented_fraction,
    summarize_selected_values,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_SOURCE_REDSHIFTS = (1.0, 2.0)
DEFAULT_COUNT_SLOPES = (0.3, 1.4)
DEFAULT_CHUNK_SIZE = 64


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _float_tuple(values: Iterable[float], name: str, *, positive: bool = True) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not out:
        raise ValueError(f"{name} must be non-empty")
    arr = np.asarray(out, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or (positive and np.any(arr <= 0.0)) or ((not positive) and np.any(arr < 0.0)):
        condition = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must contain finite {condition} values")
    return out


def _source_label(z_s: float | str) -> str:
    if isinstance(z_s, str):
        if z_s != "infinity":
            raise ValueError("only source label string allowed is 'infinity'")
        return z_s
    return f"z{float(z_s):g}"


@dataclass(frozen=True)
class PopulationFieldConfig:
    """Numerical configuration for population field evaluation.

    The default values are the publication calculation settings, but tests and
    smoke runs may supply smaller radial grids, halo counts, or angular sample
    counts without claiming publication reproduction.
    """

    radial_grid: FloatArray = field(default_factory=publication_radial_grid)
    source_redshifts: tuple[float, ...] = DEFAULT_SOURCE_REDSHIFTS
    sigma_d: tuple[float, ...] = DEFAULT_SIGMA_D
    count_slopes: tuple[float, ...] = DEFAULT_COUNT_SLOPES
    key_radius_targets: tuple[float, ...] = DEFAULT_KEY_RADIUS_TARGETS
    n_phi: int = 256
    safety_threshold: float = DEFAULT_SAFETY_THRESHOLD
    chunk_size: int = DEFAULT_CHUNK_SIZE
    include_far_background_key_samples: bool = True

    def __post_init__(self) -> None:
        radii = np.asarray(self.radial_grid, dtype=np.float64).reshape(-1)
        if radii.size < 1 or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
            raise ValueError("radial_grid must be positive, finite, and strictly increasing")
        object.__setattr__(self, "radial_grid", radii)
        object.__setattr__(self, "source_redshifts", _float_tuple(self.source_redshifts, "source_redshifts"))
        object.__setattr__(self, "sigma_d", _float_tuple(self.sigma_d, "sigma_d", positive=False))
        object.__setattr__(self, "count_slopes", _float_tuple(self.count_slopes, "count_slopes", positive=False))
        object.__setattr__(self, "key_radius_targets", _float_tuple(self.key_radius_targets, "key_radius_targets"))
        object.__setattr__(self, "n_phi", _positive_int(self.n_phi, "n_phi"))
        object.__setattr__(self, "chunk_size", _positive_int(self.chunk_size, "chunk_size"))
        threshold = float(self.safety_threshold)
        if not np.isfinite(threshold):
            raise ValueError("safety_threshold must be finite")
        object.__setattr__(self, "safety_threshold", threshold)

    @property
    def key_indices(self) -> NDArray[np.int64]:
        return key_radius_indices(self.radial_grid, self.key_radius_targets)

    @property
    def key_values(self) -> FloatArray:
        return key_radius_values(self.radial_grid, self.key_radius_targets)

    @property
    def finite_source_labels(self) -> tuple[str, ...]:
        return tuple(_source_label(z) for z in self.source_redshifts)

    @property
    def key_source_labels(self) -> tuple[str, ...]:
        labels = list(self.finite_source_labels)
        if self.include_far_background_key_samples:
            labels.append("infinity")
        return tuple(labels)


@dataclass(frozen=True)
class PopulationFieldProducts:
    """Outputs for one grid after field/observable evaluation."""

    grid_index: int
    mass_msun_h: float
    z_l: float
    radial_grid: FloatArray
    sigma_d: tuple[float, ...]
    source_redshifts: tuple[float, ...]
    source_weights: dict[str, float]
    key_indices: NDArray[np.int64]
    lambda_min_infinity: FloatArray
    safe_local: BoolArray
    safe_outer: BoolArray
    represented_fraction_outer: FloatArray
    radial_summary_rows: tuple[dict[str, float | int | str], ...]
    key_samples: dict[str, NDArray[np.float64] | NDArray[np.bool_]]

    def __post_init__(self) -> None:
        n_sigma = len(self.sigma_d)
        n_radius = len(self.radial_grid)
        if self.lambda_min_infinity.shape != (n_sigma, self.safe_local.shape[1], n_radius):
            raise ValueError("lambda_min_infinity has an inconsistent shape")
        if self.safe_local.shape != self.safe_outer.shape or self.safe_local.shape != self.lambda_min_infinity.shape:
            raise ValueError("safety-mask arrays must share shape (sigma,halo,radius)")
        if self.represented_fraction_outer.shape != (n_sigma, n_radius):
            raise ValueError("represented fractions must have shape (sigma,radius)")

    @property
    def n_halo(self) -> int:
        return int(self.safe_outer.shape[1])


def source_weight_map(
    z_l: float,
    source_redshifts: Iterable[float],
    *,
    cosmology: PublicationCosmology = PUBLICATION_COSMOLOGY,
    include_far_background: bool = True,
) -> dict[str, float]:
    """Return finite-source weights and optionally the far-background label."""
    weights = {_source_label(z): float(cosmology.source_weight(z_l, float(z))) for z in source_redshifts}
    if include_far_background:
        weights["infinity"] = 1.0
    return weights


def _chunk_slices(n_total: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, n_total, chunk_size):
        yield slice(start, min(start + chunk_size, n_total))


def _residual_arrays(n_source: int, n_sigma: int, n_halo: int, n_radius: int) -> dict[str, FloatArray]:
    shape = (n_source, n_sigma, n_halo, n_radius)
    names = [
        "delta_g_plus_fractional", "Delta_g_plus", "delta_g_plus_second_order", "Delta_g_plus_second_order",
        "delta_Y_fractional", "Delta_Y", "delta_mu_fractional", "Delta_mu",
        "delta_magnification_bias_alpha_0p3", "delta_magnification_bias_alpha_1p4",
        "g_cross_mean_reduced", "g_plus_mf",
        "delta_g_shape_only", "delta_g_offset_only", "delta_g_additive", "delta_g_interaction",
    ]
    return {name: np.full(shape, np.nan, dtype=np.float64) for name in names}


def _observable_residuals(summary: RingObservableSummary) -> dict[str, FloatArray]:
    return {
        "delta_g_plus_fractional": summary.reduced_shear.g_plus_fractional_residual,
        "Delta_g_plus": summary.reduced_shear.g_plus_difference,
        "delta_g_plus_second_order": summary.reduced_shear.g_plus_second_order_fractional,
        "Delta_g_plus_second_order": summary.reduced_shear.g_plus_second_order_difference,
        "delta_Y_fractional": summary.magnification.Y_fractional_residual,
        "Delta_Y": summary.magnification.Y_difference,
        "delta_mu_fractional": summary.magnification.mu_fractional_residual,
        "Delta_mu": summary.magnification.mu_difference,
        "delta_magnification_bias_alpha_0p3": summary.magnification.magnification_bias_fractional_residual[0.3],
        "delta_magnification_bias_alpha_1p4": summary.magnification.magnification_bias_fractional_residual[1.4],
        "g_cross_mean_reduced": summary.reduced_shear.g_cross_mean,
        "g_plus_mf": summary.reduced_shear.g_plus_mf,
    }


def _write_values(target: dict[str, FloatArray], source_index: int, sigma_index: int, halo_slice: slice, radius_index: int, values: dict[str, FloatArray]) -> None:
    for name, array in values.items():
        if name in target:
            target[name][source_index, sigma_index, halo_slice, radius_index] = array


def _summary_rows(
    *,
    halo: HaloPopulationRealization,
    config: PopulationFieldConfig,
    source_labels: tuple[str, ...],
    residuals: dict[str, FloatArray],
    safe_outer: BoolArray,
    represented: FloatArray,
) -> list[dict[str, float | int | str]]:
    observable_map = {
        "reduced_tangential_shear": "delta_g_plus_fractional",
        "inverse_magnification": "delta_Y_fractional",
        "magnification": "delta_mu_fractional",
        "magnification_bias_alpha_0p3": "delta_magnification_bias_alpha_0p3",
        "magnification_bias_alpha_1p4": "delta_magnification_bias_alpha_1p4",
    }
    rows: list[dict[str, float | int | str]] = []
    for sidx, label in enumerate(source_labels):
        for j, sigma in enumerate(config.sigma_d):
            for ridx, radius in enumerate(config.radial_grid):
                mask = safe_outer[j, :, ridx]
                for observable, name in observable_map.items():
                    stats = summarize_selected_values(residuals[name][sidx, j, :, ridx], mask)
                    rows.append({
                        "grid_index": int(halo.grid.grid_index),
                        "mass_msun_h": float(halo.grid.mass_msun_h),
                        "z_l": float(halo.grid.z_l),
                        "source_label": label,
                        "source_z": float(config.source_redshifts[sidx]),
                        "sigma_d": float(sigma),
                        "R_over_r200c": float(radius),
                        "observable": observable,
                        "n_total": int(halo.size),
                        "n_outer_safe": int(np.count_nonzero(mask)),
                        "n_selected": int(stats.n_selected),
                        "n_finite": int(stats.n_finite),
                        "represented_fraction_outer": float(represented[j, ridx]),
                        "finite_fraction_selected": float(stats.finite_fraction),
                        "mean_fractional": stats.mean,
                        "p16_fractional": stats.p16,
                        "p50_fractional": stats.p50,
                        "p84_fractional": stats.p84,
                        "p97p5_fractional": stats.p97p5,
                    })
    return rows


def _extract_key_arrays(
    *,
    config: PopulationFieldConfig,
    source_labels: tuple[str, ...],
    finite_residuals: dict[str, FloatArray],
    far_residuals: dict[str, FloatArray],
    lambda_min: FloatArray,
    safe_local: BoolArray,
    safe_outer: BoolArray,
    halo: HaloPopulationRealization,
    offsets: OffsetRealization,
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    key = config.key_indices
    n_sigma = len(config.sigma_d)
    n_source_key = len(source_labels) + (1 if config.include_far_background_key_samples else 0)
    arrays: dict[str, NDArray[np.float64] | NDArray[np.bool_]] = {
        "R_over_r200c": config.radial_grid[key].astype(np.float64),
        "sigma_d": np.asarray(config.sigma_d, dtype=np.float64),
        "source_weights": np.asarray(
            [finite_residuals["source_weights"][i] for i in range(len(source_labels))] + ([1.0] if config.include_far_background_key_samples else []),
            dtype=np.float64,
        ),
        "lambda_min_infinity": lambda_min[:, :, key].copy(),
        "safe_local": safe_local[:, :, key].copy(),
        "safe_outer": safe_outer[:, :, key].copy(),
        "epsilon_perp": halo.epsilon_perp.copy(),
        "q_perp": halo.q_perp.copy(),
        "b_los": halo.b_los.copy(),
        "projected_scale_factor": halo.projected_scale_factor.copy(),
        "c200c": halo.c200c.copy(),
        "kappa_s_perp_infinity": halo.kappa_s_perp_infinity.copy(),
        "rayleigh_unit": offsets.rayleigh_unit.copy(),
        "phi_offset": offsets.phi_offset.copy(),
        "d_over_r200c": offsets.amplitude_matrix(config.sigma_d),
    }
    source_axis_names = ["delta_g_plus_fractional", "Delta_g_plus", "delta_g_plus_second_order", "Delta_g_plus_second_order", "g_cross_mean_reduced", "g_plus_mf"]
    for name in source_axis_names:
        finite_key = finite_residuals[name][:, :, :, key]
        if config.include_far_background_key_samples:
            far_key = far_residuals[name][None, :, :, key]
            arrays[name] = np.concatenate((finite_key, far_key), axis=0)
        else:
            arrays[name] = finite_key.copy()

    # Controls are reduced-shear quantities only.  They share the same source axis.
    for name in ("delta_g_shape_only", "delta_g_offset_only", "delta_g_additive", "delta_g_interaction"):
        finite_key = finite_residuals[name][:, :, :, key]
        if config.include_far_background_key_samples:
            arrays[name] = np.concatenate((finite_key, far_residuals[name][None, :, :, key]), axis=0)
        else:
            arrays[name] = finite_key.copy()
    if arrays["delta_g_plus_fractional"].shape != (n_source_key, n_sigma, halo.size, key.size):
        raise RuntimeError("internal key-sample shape error")
    return arrays


def evaluate_grid_population_fields(
    halo: HaloPopulationRealization,
    offsets: OffsetRealization,
    mode_library: ModeLibrary,
    *,
    config: PopulationFieldConfig | None = None,
    cosmology: PublicationCosmology = PUBLICATION_COSMOLOGY,
) -> PopulationFieldProducts:
    """Evaluate one grid's far-background fields and finite-source residuals.

    This is the computational bridge between the realized halo catalog and the
    later statistical analysis.  It stores full radial summaries and reduced-
    shear key-radius products but does not choose final manuscript statistics.
    """
    cfg = config or PopulationFieldConfig()
    if halo.size != offsets.size or halo.grid.grid_index != offsets.grid_index:
        raise ValueError("halo and offset realizations must describe the same grid and size")
    n_halo = halo.size
    n_sigma = len(cfg.sigma_d)
    n_radius = len(cfg.radial_grid)
    source_labels = cfg.finite_source_labels
    weights = source_weight_map(halo.grid.z_l, cfg.source_redshifts, cosmology=cosmology, include_far_background=True)
    finite_weight_values = [weights[label] for label in source_labels]

    lambda_min = np.full((n_sigma, n_halo, n_radius), np.nan, dtype=np.float64)
    finite_residuals = _residual_arrays(len(source_labels), n_sigma, n_halo, n_radius)
    finite_residuals["source_weights"] = np.asarray(finite_weight_values, dtype=np.float64)  # type: ignore[assignment]
    far_residuals = _residual_arrays(1, n_sigma, n_halo, n_radius)
    # Drop the source dimension for far arrays after filling to simplify key extraction.

    circular = CircularNFWFieldEvaluator()

    for sigma_index, sigma in enumerate(cfg.sigma_d):
        d_all = offsets.amplitudes(sigma)
        for radius_index, radius in enumerate(cfg.radial_grid):
            for chunk in _chunk_slices(n_halo, cfg.chunk_size):
                q_eval = mode_library.evaluator_for_q(halo.q_perp[chunk])
                combined = evaluate_ring_fields(
                    q_eval,
                    concentration=halo.c200c[chunk],
                    projected_scale_factor=halo.projected_scale_factor[chunk],
                    kappa_s_projected=halo.kappa_s_perp_infinity[chunk],
                    radius_over_r200c=float(radius),
                    d_over_r200c=d_all[chunk],
                    phi_offset=offsets.phi_offset[chunk],
                    n_phi=cfg.n_phi,
                )
                far_summary = evaluate_ring_observables(combined, source_weight=1.0, count_slopes=cfg.count_slopes)
                lambda_min[sigma_index, chunk, radius_index] = far_summary.lambda_min
                far_values = _observable_residuals(far_summary)
                _write_values(far_residuals, 0, sigma_index, chunk, radius_index, far_values)

                shape = evaluate_ring_fields(
                    q_eval,
                    concentration=halo.c200c[chunk],
                    projected_scale_factor=halo.projected_scale_factor[chunk],
                    kappa_s_projected=halo.kappa_s_perp_infinity[chunk],
                    radius_over_r200c=float(radius),
                    d_over_r200c=0.0,
                    phi_offset=0.0,
                    n_phi=cfg.n_phi,
                )
                offset_only = evaluate_ring_fields(
                    circular,
                    concentration=halo.c200c[chunk],
                    projected_scale_factor=halo.projected_scale_factor[chunk],
                    kappa_s_projected=halo.kappa_s_perp_infinity[chunk],
                    radius_over_r200c=float(radius),
                    d_over_r200c=d_all[chunk],
                    phi_offset=offsets.phi_offset[chunk],
                    n_phi=cfg.n_phi,
                )

                for source_index, weight in enumerate(finite_weight_values):
                    combined_summary = evaluate_ring_observables(combined, source_weight=weight, count_slopes=cfg.count_slopes)
                    shape_summary = evaluate_ring_observables(shape, source_weight=weight, count_slopes=cfg.count_slopes)
                    offset_summary = evaluate_ring_observables(offset_only, source_weight=weight, count_slopes=cfg.count_slopes)
                    values = _observable_residuals(combined_summary)
                    shape_frac = shape_summary.reduced_shear.g_plus_fractional_residual
                    offset_frac = offset_summary.reduced_shear.g_plus_fractional_residual
                    combined_frac = combined_summary.reduced_shear.g_plus_fractional_residual
                    values.update({
                        "delta_g_shape_only": shape_frac,
                        "delta_g_offset_only": offset_frac,
                        "delta_g_additive": shape_frac + offset_frac,
                        "delta_g_interaction": combined_frac - shape_frac - offset_frac,
                    })
                    _write_values(finite_residuals, source_index, sigma_index, chunk, radius_index, values)

                # Far-background control residuals for later alignment work.
                shape_far = evaluate_ring_observables(shape, source_weight=1.0, count_slopes=cfg.count_slopes)
                off_far = evaluate_ring_observables(offset_only, source_weight=1.0, count_slopes=cfg.count_slopes)
                far_control_values = {
                    "delta_g_shape_only": shape_far.reduced_shear.g_plus_fractional_residual,
                    "delta_g_offset_only": off_far.reduced_shear.g_plus_fractional_residual,
                    "delta_g_additive": shape_far.reduced_shear.g_plus_fractional_residual + off_far.reduced_shear.g_plus_fractional_residual,
                    "delta_g_interaction": far_summary.reduced_shear.g_plus_fractional_residual - shape_far.reduced_shear.g_plus_fractional_residual - off_far.reduced_shear.g_plus_fractional_residual,
                }
                _write_values(far_residuals, 0, sigma_index, chunk, radius_index, far_control_values)

    safe_local = local_safety_mask(lambda_min, threshold=cfg.safety_threshold)
    safe_outer = outer_contiguous_mask(safe_local, radius_axis=-1)
    represented = represented_fraction(safe_outer, halo_axis=1)
    summary_rows = _summary_rows(
        halo=halo,
        config=cfg,
        source_labels=source_labels,
        residuals=finite_residuals,
        safe_outer=safe_outer,
        represented=represented,
    )
    # Convert far residual arrays from shape (1,sigma,halo,radius) to (sigma,halo,radius)
    far_key = {name: values[0] for name, values in far_residuals.items()}
    key_samples = _extract_key_arrays(
        config=cfg,
        source_labels=source_labels,
        finite_residuals=finite_residuals,
        far_residuals=far_key,
        lambda_min=lambda_min,
        safe_local=safe_local,
        safe_outer=safe_outer,
        halo=halo,
        offsets=offsets,
    )
    return PopulationFieldProducts(
        grid_index=int(halo.grid.grid_index),
        mass_msun_h=float(halo.grid.mass_msun_h),
        z_l=float(halo.grid.z_l),
        radial_grid=cfg.radial_grid.copy(),
        sigma_d=cfg.sigma_d,
        source_redshifts=cfg.source_redshifts,
        source_weights=weights,
        key_indices=cfg.key_indices,
        lambda_min_infinity=lambda_min,
        safe_local=safe_local,
        safe_outer=safe_outer,
        represented_fraction_outer=represented,
        radial_summary_rows=tuple(summary_rows),
        key_samples=key_samples,
    )


def compact_field_validation_metrics(product: PopulationFieldProducts) -> dict[str, object]:
    """Return concise diagnostics for validation reports."""
    rows = product.radial_summary_rows
    return {
        "grid_index": product.grid_index,
        "n_halo": product.n_halo,
        "n_radial": int(product.radial_grid.size),
        "n_sigma": len(product.sigma_d),
        "n_summary_rows": len(rows),
        "lambda_min_minimum": float(np.nanmin(product.lambda_min_infinity)),
        "represented_fraction_minimum": float(np.nanmin(product.represented_fraction_outer)),
        "key_radii": np.asarray(product.key_samples["R_over_r200c"]).tolist(),
        "source_weights": dict(product.source_weights),
        "first_reduced_shear_p50": next((float(r["p50_fractional"]) for r in rows if r["observable"] == "reduced_tangential_shear"), np.nan),
    }


def evaluate_publication_realization_fields(
    realization: PublicationRealization,
    mode_library: ModeLibrary,
    *,
    config: PopulationFieldConfig | None = None,
    cosmology: PublicationCosmology = PUBLICATION_COSMOLOGY,
) -> tuple[PopulationFieldProducts, ...]:
    """Evaluate all grids in a supplied publication realization."""
    if len(realization.halo_populations) != len(realization.offsets):
        raise ValueError("realization must contain matching halo and offset grids")
    return tuple(
        evaluate_grid_population_fields(halo, off, mode_library, config=config, cosmology=cosmology)
        for halo, off in zip(realization.halo_populations, realization.offsets)
    )
