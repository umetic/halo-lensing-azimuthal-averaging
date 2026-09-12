"""Deterministic controlled benchmarks for the halo-lensing paper.

This layer is orchestration around the validated core modules.  It defines the
paper's illustrative halo configuration and produces numerical benchmark data
for the centered elliptical, displaced spherical, and displaced elliptical
single-halo calculations.  It is deliberately non-stochastic and contains no
plotting code.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .cosmology import PUBLICATION_COSMOLOGY, PublicationCosmology
from .multipoles import angular_fourier_coefficients, build_elliptical_lensing_modes, nonmonopole_mode_power
from .nfw import NFWNormalization, nfw_lensing, nfw_normalization
from .observables import RingObservableSummary, evaluate_ring_observables
from .ring_fields import CircularNFWFieldEvaluator, ModeFieldEvaluator, evaluate_ring_fields
from .solver_settings import CONTROLLED_ELLIPTICAL_SETTINGS

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControlledBenchmarkConfig:
    """The fixed single-halo configuration used for the paper's benchmarks."""

    mass_msun_h: float = 1.0e15
    z_l: float = 0.3
    z_s: float = 1.0
    c200c: float = 3.843
    q_perp: float = 0.67

    @property
    def epsilon_perp(self) -> float:
        return (1.0 - self.q_perp)/(1.0 + self.q_perp)


DEFAULT_CONTROLLED_CONFIG = ControlledBenchmarkConfig()


@dataclass(frozen=True)
class ControlledContext:
    """Cached deterministic evaluators and normalizations for one config."""

    config: ControlledBenchmarkConfig
    normalization: NFWNormalization
    source_weight: float
    circular_evaluator: CircularNFWFieldEvaluator
    elliptical_evaluator: ModeFieldEvaluator

    @property
    def kappa_s_infinity(self) -> float:
        return float(np.asarray(self.normalization.kappa_s_infinity))

    @property
    def projected_scale_factor(self) -> float:
        # The controlled illustrative halos use the circular reference directly,
        # without a triaxial line-of-sight boost.
        return 1.0


@lru_cache(maxsize=8)
def make_controlled_context(
    mass_msun_h: float = DEFAULT_CONTROLLED_CONFIG.mass_msun_h,
    z_l: float = DEFAULT_CONTROLLED_CONFIG.z_l,
    z_s: float = DEFAULT_CONTROLLED_CONFIG.z_s,
    c200c: float = DEFAULT_CONTROLLED_CONFIG.c200c,
    q_perp: float = DEFAULT_CONTROLLED_CONFIG.q_perp,
) -> ControlledContext:
    """Return cached evaluators for the controlled single-halo configuration."""
    config = ControlledBenchmarkConfig(float(mass_msun_h), float(z_l), float(z_s), float(c200c), float(q_perp))
    norm = nfw_normalization(config.mass_msun_h, config.c200c, config.z_l, PUBLICATION_COSMOLOGY)
    modes = build_elliptical_lensing_modes(config.q_perp, CONTROLLED_ELLIPTICAL_SETTINGS)
    return ControlledContext(
        config=config,
        normalization=norm,
        source_weight=PUBLICATION_COSMOLOGY.source_weight(config.z_l, config.z_s),
        circular_evaluator=CircularNFWFieldEvaluator(),
        elliptical_evaluator=ModeFieldEvaluator(modes, CONTROLLED_ELLIPTICAL_SETTINGS.radial_interpolation),
    )


@dataclass(frozen=True)
class ControlledCaseSummary:
    """Compact observable result for one controlled ring."""

    case: str
    radius_over_r200c: float
    d_over_r200c: float
    phi_offset_deg: float
    n_phi: int
    delta_g_percent: float
    delta_g_second_order_percent: float
    g_plus_mean: float
    g_plus_mf: float
    g_cross_mean: float
    g_cross_mf: float
    lambda_min: float
    observables: RingObservableSummary


def _summarize(
    fields,
    *,
    case: str,
    radius_over_r200c: float,
    d_over_r200c: float,
    phi_offset_deg: float,
    n_phi: int,
    source_weight: float,
) -> ControlledCaseSummary:
    obs = evaluate_ring_observables(fields, source_weight=source_weight)
    return ControlledCaseSummary(
        case=case,
        radius_over_r200c=float(radius_over_r200c),
        d_over_r200c=float(d_over_r200c),
        phi_offset_deg=float(phi_offset_deg),
        n_phi=int(n_phi),
        delta_g_percent=float(obs.reduced_shear.g_plus_fractional_residual[0]*100.0),
        delta_g_second_order_percent=float(obs.reduced_shear.g_plus_second_order_fractional[0]*100.0),
        g_plus_mean=float(obs.reduced_shear.g_plus_mean[0]),
        g_plus_mf=float(obs.reduced_shear.g_plus_mf[0]),
        g_cross_mean=float(obs.reduced_shear.g_cross_mean[0]),
        g_cross_mf=float(obs.reduced_shear.g_cross_mf[0]),
        lambda_min=float(obs.lambda_min[0]),
        observables=obs,
    )


def centered_ellipse_summary(
    radius_over_r200c: float,
    *,
    n_phi: int = 8192,
    context: ControlledContext | None = None,
) -> ControlledCaseSummary:
    """Centered q_perp=0.67 elliptical benchmark summary."""
    ctx = context or make_controlled_context()
    fields = evaluate_ring_fields(
        ctx.elliptical_evaluator,
        concentration=ctx.config.c200c,
        projected_scale_factor=ctx.projected_scale_factor,
        kappa_s_projected=ctx.kappa_s_infinity,
        radius_over_r200c=radius_over_r200c,
        d_over_r200c=0.0,
        phi_offset=0.0,
        n_phi=n_phi,
    )
    return _summarize(
        fields,
        case="centered_ellipse",
        radius_over_r200c=radius_over_r200c,
        d_over_r200c=0.0,
        phi_offset_deg=0.0,
        n_phi=n_phi,
        source_weight=ctx.source_weight,
    )


def displaced_spherical_summary(
    radius_over_r200c: float,
    d_over_r200c: float,
    *,
    phi_offset_deg: float = 0.0,
    n_phi: int = 8192,
    context: ControlledContext | None = None,
) -> ControlledCaseSummary:
    """Displaced spherical benchmark summary using the same circular reference."""
    ctx = context or make_controlled_context()
    fields = evaluate_ring_fields(
        ctx.circular_evaluator,
        concentration=ctx.config.c200c,
        projected_scale_factor=ctx.projected_scale_factor,
        kappa_s_projected=ctx.kappa_s_infinity,
        radius_over_r200c=radius_over_r200c,
        d_over_r200c=d_over_r200c,
        phi_offset=np.deg2rad(phi_offset_deg),
        n_phi=n_phi,
    )
    return _summarize(
        fields,
        case="displaced_spherical",
        radius_over_r200c=radius_over_r200c,
        d_over_r200c=d_over_r200c,
        phi_offset_deg=phi_offset_deg,
        n_phi=n_phi,
        source_weight=ctx.source_weight,
    )


def displaced_ellipse_summary(
    radius_over_r200c: float,
    d_over_r200c: float,
    phi_offset_deg: float,
    *,
    n_phi: int = 8192,
    context: ControlledContext | None = None,
) -> ControlledCaseSummary:
    """Displaced q_perp=0.67 elliptical benchmark summary."""
    ctx = context or make_controlled_context()
    fields = evaluate_ring_fields(
        ctx.elliptical_evaluator,
        concentration=ctx.config.c200c,
        projected_scale_factor=ctx.projected_scale_factor,
        kappa_s_projected=ctx.kappa_s_infinity,
        radius_over_r200c=radius_over_r200c,
        d_over_r200c=d_over_r200c,
        phi_offset=np.deg2rad(phi_offset_deg),
        n_phi=n_phi,
    )
    return _summarize(
        fields,
        case="displaced_ellipse",
        radius_over_r200c=radius_over_r200c,
        d_over_r200c=d_over_r200c,
        phi_offset_deg=phi_offset_deg,
        n_phi=n_phi,
        source_weight=ctx.source_weight,
    )


@dataclass(frozen=True)
class InteractionSummary:
    """Combined reduced-shear response and nonadditive interaction."""

    combined: ControlledCaseSummary
    shape_only: ControlledCaseSummary
    offset_only: ControlledCaseSummary
    interaction_percent: float


def shape_centering_interaction_summary(
    radius_over_r200c: float,
    d_over_r200c: float,
    phi_offset_deg: float,
    *,
    n_phi: int = 8192,
    context: ControlledContext | None = None,
) -> InteractionSummary:
    """Return combined, shape-only, offset-only, and interaction responses."""
    ctx = context or make_controlled_context()
    combined = displaced_ellipse_summary(radius_over_r200c, d_over_r200c, phi_offset_deg, n_phi=n_phi, context=ctx)
    shape = centered_ellipse_summary(radius_over_r200c, n_phi=n_phi, context=ctx)
    offset = displaced_spherical_summary(radius_over_r200c, d_over_r200c, n_phi=n_phi, context=ctx)
    interaction = combined.delta_g_percent - shape.delta_g_percent - offset.delta_g_percent
    return InteractionSummary(combined=combined, shape_only=shape, offset_only=offset, interaction_percent=float(interaction))


def figure1_profile_radii() -> FloatArray:
    """The 58 radii used for the Figure 1 profile benchmark."""
    return np.sort(np.unique(np.concatenate([np.geomspace(0.2, 0.5, 57), np.array([0.3])]))).astype(np.float64)


def figure1_profile_table(*, n_phi: int = 4096, context: ControlledContext | None = None) -> list[dict[str, float]]:
    """Return numerical profile data for the Figure 1 controlled cases."""
    ctx = context or make_controlled_context()
    rows: list[dict[str, float]] = []
    for radius in figure1_profile_radii():
        ell = centered_ellipse_summary(float(radius), n_phi=n_phi, context=ctx)
        off = displaced_spherical_summary(float(radius), 0.10, n_phi=n_phi, context=ctx)
        rows.append({
            "R_over_r200c": float(radius),
            "delta_g_centered_ellipse_percent": ell.delta_g_percent,
            "delta_g_displaced_spherical_percent": off.delta_g_percent,
            "g_plus_mean_centered_ellipse": ell.g_plus_mean,
            "g_plus_mf_centered_ellipse": ell.g_plus_mf,
            "g_plus_mean_displaced_spherical": off.g_plus_mean,
            "g_plus_mf_displaced_spherical": off.g_plus_mf,
        })
    return rows


def append_d_spherical_leading_percent(
    radius_over_r200c: float,
    d_over_r200c: float,
    *,
    context: ControlledContext | None = None,
    derivative_step: float = 1.0e-5,
) -> float:
    """Appendix-D leading d^2 prediction for displaced spherical residual.

    The derivatives are with respect to R/r200c and are evaluated for finite-
    source fields.  A five-point stencil is sufficient for this controlled
    benchmark and avoids adding derivative APIs to the public NFW core.
    """
    ctx = context or make_controlled_context()
    R = float(radius_over_r200c)
    h = float(derivative_step)
    if not np.isfinite(R) or R <= 2.0*h:
        raise ValueError("radius must be positive and larger than twice the derivative step")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("derivative_step must be finite and positive")
    kappa_s = ctx.kappa_s_infinity * ctx.source_weight
    c = ctx.config.c200c

    def kg(r: float) -> tuple[float, float]:
        lens = nfw_lensing(c*r, kappa_s)
        return float(lens.kappa), float(lens.gamma_t)

    k, g = kg(R)
    k_m2, g_m2 = kg(R-2*h)
    k_m1, g_m1 = kg(R-h)
    k_p1, g_p1 = kg(R+h)
    k_p2, g_p2 = kg(R+2*h)
    k_prime = (k_m2 - 8*k_m1 + 8*k_p1 - k_p2)/(12.0*h)
    g_prime = (g_m2 - 8*g_m1 + 8*g_p1 - g_p2)/(12.0*h)
    pred = 0.5*d_over_r200c**2 * (k_prime*g_prime/((1.0-k)*g) + k_prime*k_prime/(1.0-k)**2)
    return float(100.0*pred)


def circular_safety_radius_over_r200c(*, context: ControlledContext | None = None) -> float:
    """Return R_sc(0)/r200c where far-background 1-kappa-gamma_t=0.1."""
    ctx = context or make_controlled_context()
    c = ctx.config.c200c
    ks = ctx.kappa_s_infinity

    def target(radius: float) -> float:
        lens = nfw_lensing(c*radius, ks)
        return float(1.0 - lens.mean_kappa - 0.1)

    return float(brentq(target, 1.0e-4, 0.5, xtol=1.0e-14, rtol=1.0e-14))


def figure2_power_table(*, n_phi: int = 8192, m_max: int = 20, context: ControlledContext | None = None) -> list[dict[str, float]]:
    """Return Figure-2(a) convergence power fractions at R/r200c=0.3."""
    ctx = context or make_controlled_context()
    cases = {
        "centered_ellipse": centered_ellipse_summary(0.3, n_phi=n_phi, context=ctx),
        "miscentered_sphere": displaced_spherical_summary(0.3, 0.10, n_phi=n_phi, context=ctx),
    }
    rows: list[dict[str, float]] = []
    for name, summary in cases.items():
        kappa = summary.observables.source_weight * 0.0  # sentinel to make mypy-less linters quiet
        del kappa
        # Power fractions use the finite-source convergence field. The overall
        # source weight cancels in the fractions, so this equals the far-
        # background fraction up to roundoff.
        field = summary.observables
        # Reconstruct from the already scaled kappa?  Observable summaries only
        # store means/covariances, so reevaluate the ring fields directly.
        if name == "centered_ellipse":
            ring = evaluate_ring_fields(
                ctx.elliptical_evaluator,
                concentration=ctx.config.c200c,
                projected_scale_factor=ctx.projected_scale_factor,
                kappa_s_projected=ctx.kappa_s_infinity*ctx.source_weight,
                radius_over_r200c=0.3,
                d_over_r200c=0.0,
                n_phi=n_phi,
            )
        else:
            ring = evaluate_ring_fields(
                ctx.circular_evaluator,
                concentration=ctx.config.c200c,
                projected_scale_factor=ctx.projected_scale_factor,
                kappa_s_projected=ctx.kappa_s_infinity*ctx.source_weight,
                radius_over_r200c=0.3,
                d_over_r200c=0.10,
                n_phi=n_phi,
            )
        modes = angular_fourier_coefficients(ring.kappa[0], m_max=m_max)
        powers = nonmonopole_mode_power(modes)
        variance = float(np.mean((ring.kappa[0] - np.mean(ring.kappa[0]))**2))
        for m, power in enumerate(powers, start=1):
            rows.append({
                "case": name,
                "m": float(m),
                "kappa_mode_power": float(power),
                "kappa_power_fraction": float(power/variance),
                "total_nonmonopole_variance": variance,
            })
    return rows


def figure2_orientation_table(
    *,
    n_phi: int = 2048,
    context: ControlledContext | None = None,
) -> list[dict[str, float]]:
    """Return Figure-2(b,c) orientation samples and fit-ready quantities."""
    ctx = context or make_controlled_context()
    radius = float(np.geomspace(0.04, 2.0, 81)[np.argmin(np.abs(np.geomspace(0.04, 2.0, 81)-0.3))])
    rows: list[dict[str, float]] = []
    for d in (0.05, 0.10, 0.20):
        shape = centered_ellipse_summary(radius, n_phi=n_phi, context=ctx)
        offset = displaced_spherical_summary(radius, d, n_phi=n_phi, context=ctx)
        for angle in range(0, 91, 15):
            combined = displaced_ellipse_summary(radius, d, float(angle), n_phi=n_phi, context=ctx)
            rows.append({
                "R_over_r200c": radius,
                "d_over_r200c": float(d),
                "d_over_R": float(d/radius),
                "phi_offset_deg": float(angle),
                "delta_g_combined_fractional": combined.delta_g_percent/100.0,
                "delta_g_shape_only_fractional": shape.delta_g_percent/100.0,
                "delta_g_offset_only_fractional": offset.delta_g_percent/100.0,
                "delta_g_interaction_fractional": (combined.delta_g_percent-shape.delta_g_percent-offset.delta_g_percent)/100.0,
                "g_cross_mean_reduced": combined.g_cross_mean,
                "lambda_min_finite_source": combined.lambda_min,
            })
    return rows


def fit_interaction_cosine(rows: list[dict[str, float]], d_over_r200c: float) -> dict[str, float]:
    """Fit A0+A2 cos(2 alpha)+A4 cos(4 alpha) to interaction rows."""
    selected = [r for r in rows if np.isclose(r["d_over_r200c"], d_over_r200c)]
    if len(selected) < 3:
        raise ValueError("not enough rows for the selected offset")
    angle = np.deg2rad([r["phi_offset_deg"] for r in selected])
    y = np.array([r["delta_g_interaction_fractional"] for r in selected], dtype=np.float64)
    X = np.column_stack((np.ones_like(angle), np.cos(2*angle), np.cos(4*angle)))
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    rms = float(np.sqrt(np.mean((X@coeff-y)**2)))
    return {"A0": float(coeff[0]), "A2_cos2alpha": float(coeff[1]), "A4_cos4alpha": float(coeff[2]), "fit_rms": rms}


def fit_cross_sine2(rows: list[dict[str, float]], d_over_r200c: float) -> dict[str, float]:
    """Fit B2 sin(2 alpha) with zero intercept and fixed phase."""
    selected = [r for r in rows if np.isclose(r["d_over_r200c"], d_over_r200c)]
    if len(selected) < 2:
        raise ValueError("not enough rows for the selected offset")
    x = np.sin(2*np.deg2rad([r["phi_offset_deg"] for r in selected]))
    y = np.array([r["g_cross_mean_reduced"] for r in selected], dtype=np.float64)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        raise ValueError("singular sine-2 fit")
    b2 = float(np.dot(x, y)/denom)
    rms = float(np.sqrt(np.mean((b2*x-y)**2)))
    return {"B2_sin2alpha": b2, "fit_rms": rms}


def compact_validation_summary(*, context: ControlledContext | None = None) -> dict[str, object]:
    """Return compact deterministic values used by the validation script."""
    ctx = context or make_controlled_context()
    fig2 = figure2_orientation_table(context=ctx)
    return {
        "configuration": {
            "mass_msun_h": ctx.config.mass_msun_h,
            "z_l": ctx.config.z_l,
            "z_s": ctx.config.z_s,
            "c200c": ctx.config.c200c,
            "q_perp": ctx.config.q_perp,
            "epsilon_perp": ctx.config.epsilon_perp,
            "source_weight": ctx.source_weight,
            "kappa_s_infinity": ctx.kappa_s_infinity,
        },
        "figure1_anchor": {
            "centered_ellipse_percent": centered_ellipse_summary(0.3, context=ctx).delta_g_percent,
            "displaced_spherical_d0p10_percent": displaced_spherical_summary(0.3, 0.10, context=ctx).delta_g_percent,
        },
        "appendix_d": {
            str(d): {
                "direct_percent": displaced_spherical_summary(0.3, d, context=ctx).delta_g_percent,
                "leading_percent": append_d_spherical_leading_percent(0.3, d, context=ctx),
            } for d in (0.05, 0.10, 0.20)
        },
        "Rsc0_over_r200c": circular_safety_radius_over_r200c(context=ctx),
        "figure2_orientation_fits": {str(d): fit_interaction_cosine(fig2, d) for d in (0.05, 0.10, 0.20)},
        "figure2_cross_fits": {str(d): fit_cross_sine2(fig2, d) for d in (0.05, 0.10, 0.20)},
    }
