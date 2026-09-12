"""Seeded halo-population realization for the publication experiment.

This layer generates halo and centering catalogs only.  It deliberately stops
before evaluating ring lensing fields, source-plane residuals, safety masks, or
population summaries.  The stochastic model is pseudo-random but deterministic
under the publication seed schedule and draw order.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .cosmology import PUBLICATION_COSMOLOGY, PublicationCosmology
from .miscentering import DEFAULT_OFFSET_BASE_SEED, OffsetRealization, draw_offset_realization
from .nfw import nfw_normalization
from .projection import project_triaxial_batch

DEFAULT_MASSES_MSUN_H = (3.0e14, 1.0e15, 2.0e15)
DEFAULT_REDSHIFTS = (0.2, 0.5)
DEFAULT_N_HALO = 10_000
DEFAULT_POPULATION_BASE_SEED = 20260815
POPULATION_SEED_STEP = 1009
DEFAULT_SIGMA_LOG10_C = 0.16
BONAMIGO_LOG_S_TILDE_MEAN = -0.49
BONAMIGO_LOG_S_TILDE_SIGMA = 0.20
BONAMIGO_NU_EXPONENT = 0.255


class ColossusUnavailableError(ImportError):
    """Raised when the required Colossus production backend is unavailable."""


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_int(value: int, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _size(value: int) -> int:
    result = int(value)
    if result < 1:
        raise ValueError("size must be at least one")
    return result


def population_seed_for_grid(
    grid_index: int,
    base_seed: int = DEFAULT_POPULATION_BASE_SEED,
    step: int = POPULATION_SEED_STEP,
) -> int:
    """Actual population seed: base_seed + 1009*grid_index."""
    return int(base_seed) + int(step) * _nonnegative_int(grid_index, "grid_index")


@dataclass(frozen=True)
class PopulationGrid:
    """One mass-redshift grid point in redshift-major, mass-minor order."""

    grid_index: int
    mass_msun_h: float
    z_l: float
    population_seed: int


def publication_grids(
    masses: Iterable[float] = DEFAULT_MASSES_MSUN_H,
    redshifts: Iterable[float] = DEFAULT_REDSHIFTS,
    base_seed: int = DEFAULT_POPULATION_BASE_SEED,
) -> tuple[PopulationGrid, ...]:
    grids: list[PopulationGrid] = []
    index = 0
    for z_l in tuple(redshifts):
        for mass in tuple(masses):
            grids.append(
                PopulationGrid(
                    grid_index=index,
                    mass_msun_h=_positive_scalar(mass, "mass"),
                    z_l=_positive_scalar(z_l, "z_l"),
                    population_seed=population_seed_for_grid(index, base_seed=base_seed),
                )
            )
            index += 1
    return tuple(grids)


def canonical_grid_index(
    mass_msun_h: float,
    z_l: float,
    masses: tuple[float, ...] = DEFAULT_MASSES_MSUN_H,
    redshifts: tuple[float, ...] = DEFAULT_REDSHIFTS,
) -> int:
    mass = _positive_scalar(mass_msun_h, "mass_msun_h")
    redshift = _positive_scalar(z_l, "z_l")
    mass_matches = [i for i, value in enumerate(masses) if np.isclose(mass, value, rtol=0.0, atol=1.0e-8*value)]
    redshift_matches = [i for i, value in enumerate(redshifts) if np.isclose(redshift, value, rtol=0.0, atol=1.0e-12)]
    if len(mass_matches) != 1 or len(redshift_matches) != 1:
        raise ValueError("mass/redshift is not a unique publication grid point")
    return redshift_matches[0]*len(masses) + mass_matches[0]


@dataclass(frozen=True)
class ShapeState:
    """Deterministic Colossus state used before stochastic shape draws."""

    c200c_median: float
    mvir_msun_h: float
    peak_height: float
    colossus_version: str


def colossus_version() -> str:
    try:
        return version("colossus")
    except PackageNotFoundError as exc:
        raise ColossusUnavailableError("Colossus is required for publication population generation") from exc


def _set_colossus_planck18() -> None:
    try:
        from colossus.cosmology import cosmology
    except Exception as exc:  # pragma: no cover - exercised on systems without Colossus
        raise ColossusUnavailableError("Cannot import colossus.cosmology") from exc
    cosmology.setCosmology("planck18")


def median_concentration_dj19(mass_msun_h: float, z_l: float) -> float:
    """Colossus DJ19 median concentration for M200c, not a fallback relation."""
    _set_colossus_planck18()
    try:
        from colossus.halo import concentration
    except Exception as exc:  # pragma: no cover
        raise ColossusUnavailableError("Cannot import colossus.halo.concentration") from exc
    value = concentration.concentration(
        _positive_scalar(mass_msun_h, "mass_msun_h"),
        "200c",
        _positive_scalar(z_l, "z_l"),
        model="diemer19",
        statistic="median",
    )
    return float(np.asarray(value, dtype=np.float64))


def colossus_shape_state(mass_msun_h: float, z_l: float) -> ShapeState:
    """Return c_med and Bonamigo peak height using median-concentration Mvir."""
    _set_colossus_planck18()
    try:
        from colossus.halo import mass_defs
        from colossus.lss import peaks
    except Exception as exc:  # pragma: no cover
        raise ColossusUnavailableError("Cannot import Colossus mass-definition or peaks modules") from exc
    mass = _positive_scalar(mass_msun_h, "mass_msun_h")
    redshift = _positive_scalar(z_l, "z_l")
    c_median = median_concentration_dj19(mass, redshift)
    mvir, _, _ = mass_defs.changeMassDefinition(mass, c_median, redshift, "200c", "vir")
    peak_height = peaks.peakHeight(mvir, redshift)
    return ShapeState(float(c_median), float(np.asarray(mvir)), float(np.asarray(peak_height)), colossus_version())


def qtilde_mean(s: NDArray[np.float64] | float) -> NDArray[np.float64]:
    value = np.asarray(s, dtype=np.float64)
    if np.any((value <= 0.0) | (value >= 1.0)):
        raise ValueError("s must lie in (0,1)")
    return np.asarray(0.633*value - 0.007)


def qtilde_beta_parameters(s: NDArray[np.float64] | float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    value = np.asarray(s, dtype=np.float64)
    mu = qtilde_mean(value)
    if np.any((mu <= 0.0) | (mu >= 1.0)):
        raise ValueError("conditional beta mean lies outside (0,1)")
    beta = np.asarray(1.389 * value**(-1.685))
    alpha = np.asarray(beta * mu/(1.0-mu))
    return alpha, beta


def triaxiality_from_axis_ratios(p: NDArray[np.float64], s: NDArray[np.float64]) -> NDArray[np.float64]:
    pp, ss = np.broadcast_arrays(np.asarray(p, dtype=np.float64), np.asarray(s, dtype=np.float64))
    if np.any((ss <= 0.0) | (pp < ss) | (pp > 1.0)):
        raise ValueError("axis ratios must satisfy 0 < s <= p <= 1")
    denom = 1.0 - ss**2
    out = np.zeros_like(denom)
    mask = denom > 1.0e-14
    out[mask] = (1.0 - pp[mask]**2)/denom[mask]
    return np.asarray(out)


@dataclass(frozen=True)
class MinorAxisDraw:
    s_axis_ratio: NDArray[np.float64]
    proposal_count: int
    rejection_rounds: int

    @property
    def rejection_fraction(self) -> float:
        return 1.0 - self.s_axis_ratio.size/float(self.proposal_count)


def sample_minor_axis_ratios(
    rng: np.random.Generator,
    size: int,
    peak_height: float,
    *,
    max_rejection_rounds: int = 100,
) -> MinorAxisDraw:
    """Sample Bonamigo minor-to-major axis ratio with publication draw order."""
    n = _size(size)
    nu = _positive_scalar(peak_height, "peak_height")
    if max_rejection_rounds < 1:
        raise ValueError("max_rejection_rounds must be positive")
    nu_factor = nu**(-BONAMIGO_NU_EXPONENT)
    result = np.empty(n, dtype=np.float64)
    n_filled = 0
    n_drawn = 0
    rounds = 0
    while n_filled < n:
        rounds += 1
        if rounds > max_rejection_rounds:
            raise RuntimeError("minor-axis rejection sampler failed to converge")
        remaining = n - n_filled
        ln_scaled = rng.normal(BONAMIGO_LOG_S_TILDE_MEAN, BONAMIGO_LOG_S_TILDE_SIGMA, size=remaining)
        proposals = np.exp(ln_scaled) * nu_factor
        accepted = proposals[(proposals > 0.0) & (proposals < 1.0)]
        take = min(remaining, accepted.size)
        result[n_filled:n_filled+take] = accepted[:take]
        n_filled += take
        n_drawn += remaining
    return MinorAxisDraw(result, n_drawn, rounds)


def sample_axis_ratios(
    rng: np.random.Generator,
    size: int,
    peak_height: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], MinorAxisDraw]:
    minor = sample_minor_axis_ratios(rng, size, peak_height)
    alpha, beta = qtilde_beta_parameters(minor.s_axis_ratio)
    q_tilde = rng.beta(alpha, beta)
    p_axis_ratio = minor.s_axis_ratio + (1.0-minor.s_axis_ratio)*q_tilde
    triaxiality = triaxiality_from_axis_ratios(p_axis_ratio, minor.s_axis_ratio)
    return np.asarray(p_axis_ratio), minor.s_axis_ratio, np.asarray(triaxiality), minor


def draw_isotropic_los(
    rng: np.random.Generator,
    size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Draw isotropic LOS vectors, returning cos(theta), azimuth, and vectors."""
    n = _size(size)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0*np.pi, size=n)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0-mu**2))
    vectors = np.column_stack((sin_theta*np.cos(phi), sin_theta*np.sin(phi), mu))
    return np.asarray(mu), np.asarray(phi), np.asarray(vectors)


@dataclass(frozen=True)
class HaloPopulationRealization:
    """One regenerated publication halo catalog before ring-field evaluation."""

    grid: PopulationGrid
    c200c_median: float
    shape_peak_height: float
    mvir_shape_msun_h: float
    colossus_version: str
    c200c: NDArray[np.float64]
    p_axis_ratio: NDArray[np.float64]
    s_axis_ratio: NDArray[np.float64]
    triaxiality: NDArray[np.float64]
    los_cos_theta: NDArray[np.float64]
    los_phi: NDArray[np.float64]
    los_vectors: NDArray[np.float64]
    q_perp: NDArray[np.float64]
    b_los: NDArray[np.float64]
    projected_scale_factor: NDArray[np.float64]
    r200c_mpc_h: float
    rs_mpc_h: NDArray[np.float64]
    rs_perp_mpc_h: NDArray[np.float64]
    kappa_s_infinity: NDArray[np.float64]
    kappa_s_perp_infinity: NDArray[np.float64]
    epsilon_perp: NDArray[np.float64]
    minor_axis_proposal_count: int
    minor_axis_rejection_rounds: int

    @property
    def size(self) -> int:
        return int(self.c200c.size)

    @property
    def minor_axis_rejection_fraction(self) -> float:
        return 1.0 - self.size/float(self.minor_axis_proposal_count)

    @property
    def bit_generator_name(self) -> str:
        return "PCG64"

    def selected_catalog(self, indices: NDArray[np.integer] | list[int]) -> dict[str, list[float]]:
        idx = np.asarray(indices, dtype=int)
        return {
            "concentration": self.c200c[idx].tolist(),
            "p_axis_ratio": self.p_axis_ratio[idx].tolist(),
            "s_axis_ratio": self.s_axis_ratio[idx].tolist(),
            "triaxiality": self.triaxiality[idx].tolist(),
            "q_projected": self.q_perp[idx].tolist(),
            "epsilon_perp": self.epsilon_perp[idx].tolist(),
            "los_boost": self.b_los[idx].tolist(),
            "projected_scale_factor": self.projected_scale_factor[idx].tolist(),
        }


def sample_halo_population_from_state(
    grid: PopulationGrid,
    *,
    size: int = DEFAULT_N_HALO,
    c200c_median: float,
    shape_peak_height: float,
    mvir_shape_msun_h: float = np.nan,
    colossus_version: str = "provided-state",
    sigma_log10_c: float = DEFAULT_SIGMA_LOG10_C,
    cosmology: PublicationCosmology = PUBLICATION_COSMOLOGY,
) -> HaloPopulationRealization:
    """Generate one halo catalog from an explicit c_med and shape peak height.

    This entry point is useful for validating the RNG/draw-order contract against
    frozen publication catalogs even on systems where Colossus is unavailable.
    It is not a replacement for the production Colossus state calculation.
    """
    n = _size(size)
    c_med = _positive_scalar(c200c_median, "c200c_median")
    nu = _positive_scalar(shape_peak_height, "shape_peak_height")
    sigma = _positive_scalar(sigma_log10_c, "sigma_log10_c")
    rng = np.random.default_rng(int(grid.population_seed))
    concentration = np.exp(np.log(c_med) + np.log(10.0)*sigma*rng.normal(size=n))
    p_axis, s_axis, triaxiality, minor = sample_axis_ratios(rng, n, nu)
    los_mu, los_phi, los_vectors = draw_isotropic_los(rng, n)
    projection = project_triaxial_batch(p_axis, s_axis, los_vectors)
    norm = nfw_normalization(grid.mass_msun_h, concentration, grid.z_l, cosmology=cosmology)
    r200 = float(np.ravel(norm.r200c_mpc_h)[0])
    rs_perp = projection.scale_factor * norm.r_s_mpc_h
    kappa_s_perp = projection.b_los * norm.kappa_s_infinity
    epsilon = (1.0 - projection.q_perp)/(1.0 + projection.q_perp)
    return HaloPopulationRealization(
        grid=grid,
        c200c_median=c_med,
        shape_peak_height=nu,
        mvir_shape_msun_h=float(mvir_shape_msun_h),
        colossus_version=str(colossus_version),
        c200c=np.asarray(concentration, dtype=np.float64),
        p_axis_ratio=np.asarray(p_axis, dtype=np.float64),
        s_axis_ratio=np.asarray(s_axis, dtype=np.float64),
        triaxiality=np.asarray(triaxiality, dtype=np.float64),
        los_cos_theta=np.asarray(los_mu, dtype=np.float64),
        los_phi=np.asarray(los_phi, dtype=np.float64),
        los_vectors=np.asarray(los_vectors, dtype=np.float64),
        q_perp=np.asarray(projection.q_perp, dtype=np.float64),
        b_los=np.asarray(projection.b_los, dtype=np.float64),
        projected_scale_factor=np.asarray(projection.scale_factor, dtype=np.float64),
        r200c_mpc_h=r200,
        rs_mpc_h=np.asarray(norm.r_s_mpc_h, dtype=np.float64),
        rs_perp_mpc_h=np.asarray(rs_perp, dtype=np.float64),
        kappa_s_infinity=np.asarray(norm.kappa_s_infinity, dtype=np.float64),
        kappa_s_perp_infinity=np.asarray(kappa_s_perp, dtype=np.float64),
        epsilon_perp=np.asarray(epsilon, dtype=np.float64),
        minor_axis_proposal_count=int(minor.proposal_count),
        minor_axis_rejection_rounds=int(minor.rejection_rounds),
    )


def generate_halo_population(
    grid: PopulationGrid,
    *,
    size: int = DEFAULT_N_HALO,
    sigma_log10_c: float = DEFAULT_SIGMA_LOG10_C,
    cosmology: PublicationCosmology = PUBLICATION_COSMOLOGY,
) -> HaloPopulationRealization:
    """Generate one publication halo catalog using the required Colossus state."""
    state = colossus_shape_state(grid.mass_msun_h, grid.z_l)
    return sample_halo_population_from_state(
        grid,
        size=size,
        c200c_median=state.c200c_median,
        shape_peak_height=state.peak_height,
        mvir_shape_msun_h=state.mvir_msun_h,
        colossus_version=state.colossus_version,
        sigma_log10_c=sigma_log10_c,
        cosmology=cosmology,
    )


@dataclass(frozen=True)
class PublicationRealization:
    halo_populations: tuple[HaloPopulationRealization, ...]
    offsets: tuple[OffsetRealization, ...]

    @property
    def n_grid(self) -> int:
        return len(self.halo_populations)


def generate_publication_realization(
    *,
    size: int = DEFAULT_N_HALO,
    include_offsets: bool = True,
) -> PublicationRealization:
    grids = publication_grids()
    halos = tuple(generate_halo_population(grid, size=size) for grid in grids)
    offsets = tuple(
        draw_offset_realization(size=size, grid_index=grid.grid_index, base_seed=DEFAULT_OFFSET_BASE_SEED)
        for grid in grids
    ) if include_offsets else tuple()
    return PublicationRealization(halos, offsets)
