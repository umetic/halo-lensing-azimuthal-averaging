"""Nonlinear ring observables and residual definitions.

This module is array based. It does not know how a ring field was generated;
its responsibility is to scale far-background fields to a source plane before
nonlinear mapping, then compute complete-ring means, mean-field values,
angular covariances, residuals, and second-order diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ring_fields import RingFields

FloatArray = NDArray[np.float64]


def _field_arrays(
    kappa_or_fields: RingFields | ArrayLike,
    gamma_plus: ArrayLike | None = None,
    gamma_cross: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if isinstance(kappa_or_fields, RingFields):
        if gamma_plus is not None or gamma_cross is not None:
            raise ValueError("do not supply separate shear arrays when passing RingFields")
        k = np.asarray(kappa_or_fields.kappa, dtype=np.float64)
        gp = np.asarray(kappa_or_fields.gamma_plus, dtype=np.float64)
        gx = np.asarray(kappa_or_fields.gamma_cross, dtype=np.float64)
    else:
        if gamma_plus is None or gamma_cross is None:
            raise ValueError("gamma_plus and gamma_cross are required with array inputs")
        k = np.asarray(kappa_or_fields, dtype=np.float64)
        gp = np.asarray(gamma_plus, dtype=np.float64)
        gx = np.asarray(gamma_cross, dtype=np.float64)
    if k.shape != gp.shape or k.shape != gx.shape or k.ndim != 2:
        raise ValueError("field arrays must have equal shape (n_halo, n_phi)")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(gp)) or np.any(~np.isfinite(gx)):
        raise ValueError("field arrays must be finite")
    return k, gp, gx


def _source_weight(value: float) -> float:
    weight = float(np.asarray(value, dtype=np.float64))
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("source_weight must be finite and positive")
    return weight


def scale_lensing_fields(
    kappa_or_fields: RingFields | ArrayLike,
    gamma_plus: ArrayLike | None = None,
    gamma_cross: ArrayLike | None = None,
    *,
    source_weight: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Scale linear fields by w before any nonlinear observable map."""
    k, gp, gx = _field_arrays(kappa_or_fields, gamma_plus, gamma_cross)
    w = _source_weight(source_weight)
    return np.asarray(w*k), np.asarray(w*gp), np.asarray(w*gx)


def angular_covariance(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Complete-ring covariance with population normalization, ``ddof=0``."""
    x, y = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    if x.ndim != 2 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("covariance inputs must be finite arrays with shape (n_halo, n_phi)")
    dx = x - np.mean(x, axis=1)[:, None]
    dy = y - np.mean(y, axis=1)[:, None]
    return np.mean(dx*dy, axis=1)


@dataclass(frozen=True)
class AngularCovariances:
    C_kk: FloatArray
    C_k_plus: FloatArray
    C_k_cross: FloatArray
    C_plus_plus: FloatArray
    C_plus_cross: FloatArray
    C_cross_cross: FloatArray


@dataclass(frozen=True)
class ReducedShearSummary:
    g_plus_mean: FloatArray
    g_cross_mean: FloatArray
    g_plus_mf: FloatArray
    g_cross_mf: FloatArray
    g_plus_difference: FloatArray
    g_cross_difference: FloatArray
    g_plus_fractional_residual: FloatArray
    g_plus_second_order_difference: FloatArray
    g_plus_second_order_fractional: FloatArray


@dataclass(frozen=True)
class MagnificationSummary:
    Y_mean: FloatArray
    Y_mf: FloatArray
    Y_difference: FloatArray
    Y_fractional_residual: FloatArray
    Y_second_order_fractional: FloatArray
    mu_mean: FloatArray
    mu_mf: FloatArray
    mu_difference: FloatArray
    mu_fractional_residual: FloatArray
    mu_second_order_fractional: FloatArray
    magnification_bias_mean: dict[float, FloatArray]
    magnification_bias_mf: dict[float, FloatArray]
    magnification_bias_difference: dict[float, FloatArray]
    magnification_bias_fractional_residual: dict[float, FloatArray]
    magnification_bias_second_order_fractional: dict[float, FloatArray]
    var_lnY: FloatArray


@dataclass(frozen=True)
class RingObservableSummary:
    source_weight: float
    kappa_bar: FloatArray
    gamma_plus_bar: FloatArray
    gamma_cross_bar: FloatArray
    covariances: AngularCovariances
    reduced_shear: ReducedShearSummary
    magnification: MagnificationSummary
    lambda_min: FloatArray


def evaluate_ring_observables(
    kappa_or_fields: RingFields | ArrayLike,
    gamma_plus: ArrayLike | None = None,
    gamma_cross: ArrayLike | None = None,
    *,
    source_weight: float = 1.0,
    count_slopes: tuple[float, ...] = (0.3, 1.4),
    denominator_tol: float = 1.0e-14,
) -> RingObservableSummary:
    """Evaluate nonlinear complete-ring observables from far-background fields.

    All fields are first multiplied by ``source_weight``. Fractional residuals
    are defined relative to the mean-field observable at the same adopted
    center, radius, and source plane.
    """
    tol = float(denominator_tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("denominator_tol must be finite and non-negative")
    kap, gp, gx = scale_lensing_fields(kappa_or_fields, gamma_plus, gamma_cross, source_weight=source_weight)
    kbar = np.mean(kap, axis=1)
    gpbar = np.mean(gp, axis=1)
    gxbar = np.mean(gx, axis=1)
    D = 1.0 - kbar

    cov = AngularCovariances(
        C_kk=angular_covariance(kap, kap),
        C_k_plus=angular_covariance(kap, gp),
        C_k_cross=angular_covariance(kap, gx),
        C_plus_plus=angular_covariance(gp, gp),
        C_plus_cross=angular_covariance(gp, gx),
        C_cross_cross=angular_covariance(gx, gx),
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gplus_local = gp/(1.0-kap)
        gcross_local = gx/(1.0-kap)
        gplus_mean = np.mean(gplus_local, axis=1)
        gcross_mean = np.mean(gcross_local, axis=1)
        gplus_mf = gpbar/D
        gcross_mf = gxbar/D
        delta_g2 = cov.C_k_plus/D**2 + gpbar*cov.C_kk/D**3
        frac_g = gplus_mean/gplus_mf - 1.0
        frac_g2 = delta_g2/gplus_mf

    invalid_g = (~np.isfinite(gplus_mf)) | (np.abs(gplus_mf) < tol)
    frac_g = np.where(invalid_g, np.nan, frac_g)
    frac_g2 = np.where(invalid_g, np.nan, frac_g2)
    shear = ReducedShearSummary(
        g_plus_mean=gplus_mean,
        g_cross_mean=gcross_mean,
        g_plus_mf=gplus_mf,
        g_cross_mf=gcross_mf,
        g_plus_difference=gplus_mean-gplus_mf,
        g_cross_difference=gcross_mean-gcross_mf,
        g_plus_fractional_residual=frac_g,
        g_plus_second_order_difference=delta_g2,
        g_plus_second_order_fractional=frac_g2,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Y = (1.0-kap)**2 - gp**2 - gx**2
        Y_mf = D**2 - gpbar**2 - gxbar**2
        Y_mean = np.mean(Y, axis=1)
        S_Y = cov.C_kk - cov.C_plus_plus - cov.C_cross_cross
        V_Y = D**2*cov.C_kk + 2.0*D*gpbar*cov.C_k_plus + gpbar**2*cov.C_plus_plus
        Y_frac = Y_mean/Y_mf - 1.0
        Y_eft = S_Y/Y_mf
        mu_local = 1.0/Y
        mu_mean = np.mean(mu_local, axis=1)
        mu_mf = 1.0/Y_mf
        mu_frac = mu_mean/mu_mf - 1.0
        q_mu = -1.0
        mu_eft = q_mu*S_Y/Y_mf + 2.0*q_mu*(q_mu-1.0)*V_Y/Y_mf**2
        var_lnY = np.var(np.log(Y), axis=1)

    mb_mean: dict[float, FloatArray] = {}
    mb_mf: dict[float, FloatArray] = {}
    mb_diff: dict[float, FloatArray] = {}
    mb_frac: dict[float, FloatArray] = {}
    mb_eft: dict[float, FloatArray] = {}
    for alpha in count_slopes:
        a = float(alpha)
        if not np.isfinite(a):
            raise ValueError("count_slopes must be finite")
        q = 1.0 - a
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            local = Y**q
            mf = Y_mf**q
            mean = np.mean(local, axis=1)
            frac = mean/mf - 1.0
            eft = q*S_Y/Y_mf + 2.0*q*(q-1.0)*V_Y/Y_mf**2
        mb_mean[a] = mean
        mb_mf[a] = mf
        mb_diff[a] = mean - mf
        mb_frac[a] = frac
        mb_eft[a] = eft

    mag = MagnificationSummary(
        Y_mean=Y_mean,
        Y_mf=Y_mf,
        Y_difference=Y_mean-Y_mf,
        Y_fractional_residual=Y_frac,
        Y_second_order_fractional=Y_eft,
        mu_mean=mu_mean,
        mu_mf=mu_mf,
        mu_difference=mu_mean-mu_mf,
        mu_fractional_residual=mu_frac,
        mu_second_order_fractional=mu_eft,
        magnification_bias_mean=mb_mean,
        magnification_bias_mf=mb_mf,
        magnification_bias_difference=mb_diff,
        magnification_bias_fractional_residual=mb_frac,
        magnification_bias_second_order_fractional=mb_eft,
        var_lnY=var_lnY,
    )

    lambda_min = np.min(1.0 - kap - np.hypot(gp, gx), axis=1)
    return RingObservableSummary(
        source_weight=float(source_weight),
        kappa_bar=kbar,
        gamma_plus_bar=gpbar,
        gamma_cross_bar=gxbar,
        covariances=cov,
        reduced_shear=shear,
        magnification=mag,
        lambda_min=lambda_min,
    )
