from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.population import (
    BONAMIGO_LOG_S_TILDE_MEAN,
    BONAMIGO_LOG_S_TILDE_SIGMA,
    DEFAULT_MASSES_MSUN_H,
    DEFAULT_N_HALO,
    DEFAULT_POPULATION_BASE_SEED,
    DEFAULT_REDSHIFTS,
    DEFAULT_SIGMA_LOG10_C,
    POPULATION_SEED_STEP,
    ColossusUnavailableError,
    canonical_grid_index,
    colossus_shape_state,
    generate_halo_population,
    population_seed_for_grid,
    publication_grids,
    qtilde_beta_parameters,
    qtilde_mean,
    sample_halo_population_from_state,
    sample_minor_axis_ratios,
    triaxiality_from_axis_ratios,
)

TARGETS = json.loads(Path("reference/population_realization/population_targets.json").read_text())
SELECTED = np.asarray(TARGETS["selected_indices"], dtype=int)


def _stats(array):
    arr = np.asarray(array, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q16": float(np.quantile(arr, 0.16)),
        "q50": float(np.quantile(arr, 0.50)),
        "q84": float(np.quantile(arr, 0.84)),
    }


def _colossus_available() -> bool:
    try:
        colossus_shape_state(3.0e14, 0.2)
    except ColossusUnavailableError:
        return False
    return True


def test_publication_grid_order_and_seed_schedule():
    grids = publication_grids()
    assert len(grids) == 6
    for target, grid in zip(TARGETS["grids"], grids):
        assert grid.grid_index == target["grid_index"]
        assert grid.mass_msun_h == target["mass_msun_h"]
        assert grid.z_l == target["z_l"]
        assert grid.population_seed == target["population_seed"]
        assert grid.population_seed == TARGETS["population_base_seed"] + TARGETS["population_seed_step"]*grid.grid_index
        assert canonical_grid_index(grid.mass_msun_h, grid.z_l) == grid.grid_index


def test_population_seed_helper_and_invalid_grid():
    assert population_seed_for_grid(5) == DEFAULT_POPULATION_BASE_SEED + POPULATION_SEED_STEP*5
    with pytest.raises(ValueError):
        population_seed_for_grid(-1)
    with pytest.raises(ValueError):
        canonical_grid_index(4.0e14, 0.2)


def test_default_constants_match_publication_targets():
    assert DEFAULT_MASSES_MSUN_H == tuple(g["mass_msun_h"] for g in TARGETS["grids"][:3])
    assert DEFAULT_REDSHIFTS == (0.2, 0.5)
    assert DEFAULT_N_HALO == TARGETS["n_halo"]
    assert DEFAULT_POPULATION_BASE_SEED == TARGETS["population_base_seed"]
    assert POPULATION_SEED_STEP == TARGETS["population_seed_step"]
    assert DEFAULT_SIGMA_LOG10_C == TARGETS["sigma_log10_c"]


def test_bonamigo_beta_parameters_and_triaxiality():
    s = np.array([0.3, 0.5, 0.8])
    mu = qtilde_mean(s)
    assert_allclose(mu, 0.633*s - 0.007, rtol=0, atol=0)
    alpha, beta = qtilde_beta_parameters(s)
    assert np.all(alpha > 0)
    assert np.all(beta > 0)
    assert_allclose(alpha/(alpha+beta), mu, rtol=1e-15, atol=1e-15)
    p = np.array([0.4, 0.7, 0.9])
    t = triaxiality_from_axis_ratios(p, s)
    assert_allclose(t, (1.0-p**2)/(1.0-s**2), rtol=0, atol=0)
    with pytest.raises(ValueError):
        qtilde_mean([0.0])
    with pytest.raises(ValueError):
        triaxiality_from_axis_ratios(np.array([0.2]), np.array([0.3]))


def test_minor_axis_rejection_sampler_draws_only_remaining_count():
    # Grid 0 has exactly one historical rejection at N=10000.  The target
    # proposal count verifies that a rejection round proposes the remaining
    # count, not a fixed oversampling batch.
    target = TARGETS["grids"][0]
    rng = np.random.default_rng(target["population_seed"])
    rng.normal(size=TARGETS["n_halo"])
    draw = sample_minor_axis_ratios(rng, TARGETS["n_halo"], target["shape_peak_height_inferred"])
    assert draw.proposal_count == target["minor_axis_proposal_count"]
    assert draw.rejection_rounds == target["minor_axis_rejection_rounds"]
    assert_allclose(draw.rejection_fraction, target["minor_axis_rejection_fraction"], rtol=0, atol=0)


@pytest.mark.parametrize("grid_target", TARGETS["grids"], ids=lambda g: f"grid{g['grid_index']}")
def test_realization_from_explicit_state_matches_frozen_catalog(grid_target):
    grid = publication_grids()[grid_target["grid_index"]]
    realization = sample_halo_population_from_state(
        grid,
        size=TARGETS["n_halo"],
        c200c_median=grid_target["c200c_median_inferred"],
        shape_peak_height=grid_target["shape_peak_height_inferred"],
    )
    assert realization.size == TARGETS["n_halo"]
    assert realization.grid == grid
    assert_allclose(realization.c200c_median, grid_target["c200c_median_inferred"], rtol=0, atol=0)
    assert_allclose(realization.shape_peak_height, grid_target["shape_peak_height_inferred"], rtol=0, atol=0)
    assert realization.minor_axis_proposal_count == grid_target["minor_axis_proposal_count"]
    assert realization.minor_axis_rejection_rounds == grid_target["minor_axis_rejection_rounds"]
    assert_allclose(realization.minor_axis_rejection_fraction, grid_target["minor_axis_rejection_fraction"], rtol=0, atol=0)
    assert realization.los_vectors.shape == (TARGETS["n_halo"], 3)
    assert_allclose(np.linalg.norm(realization.los_vectors, axis=1), 1.0, rtol=0, atol=3e-16)
    assert np.all((realization.s_axis_ratio > 0.0) & (realization.s_axis_ratio <= realization.p_axis_ratio) & (realization.p_axis_ratio <= 1.0))
    assert np.all((realization.q_perp > 0.0) & (realization.q_perp <= 1.0))
    mapping = {
        "concentration": realization.c200c,
        "p_axis_ratio": realization.p_axis_ratio,
        "s_axis_ratio": realization.s_axis_ratio,
        "triaxiality": realization.triaxiality,
        "q_projected": realization.q_perp,
        "epsilon_perp": realization.epsilon_perp,
        "los_boost": realization.b_los,
        "projected_scale_factor": realization.projected_scale_factor,
    }
    for name, array in mapping.items():
        target = grid_target["arrays"][name]
        assert_allclose(array[SELECTED], target["selected_values"], rtol=0, atol=3e-13)
        for stat_name, stat_value in target["statistics"].items():
            assert_allclose(_stats(array)[stat_name], stat_value, rtol=0, atol=5e-13)


def test_small_n_is_not_publication_prefix_for_later_draws():
    target = TARGETS["grids"][1]
    grid = publication_grids()[target["grid_index"]]
    small = sample_halo_population_from_state(
        grid,
        size=16,
        c200c_median=target["c200c_median_inferred"],
        shape_peak_height=target["shape_peak_height_inferred"],
    )
    full = sample_halo_population_from_state(
        grid,
        size=TARGETS["n_halo"],
        c200c_median=target["c200c_median_inferred"],
        shape_peak_height=target["shape_peak_height_inferred"],
    )
    # The first vector draw is concentration, so its prefix agrees.  Subsequent
    # arrays do not, because changing N changes how many concentration normals
    # are consumed before the shape and viewing draws.
    assert_allclose(small.c200c, full.c200c[:16], rtol=0, atol=0)
    assert not np.allclose(small.s_axis_ratio, full.s_axis_ratio[:16], rtol=0, atol=1e-15)
    assert not np.allclose(small.los_phi, full.los_phi[:16], rtol=0, atol=1e-15)


@pytest.mark.skipif(not _colossus_available(), reason="Colossus production backend is not installed")
@pytest.mark.parametrize("grid_target", TARGETS["grids"], ids=lambda g: f"grid{g['grid_index']}")
def test_colossus_shape_state_matches_inferred_publication_state(grid_target):
    state = colossus_shape_state(grid_target["mass_msun_h"], grid_target["z_l"])
    assert state.colossus_version == "1.4.0"
    assert_allclose(state.c200c_median, grid_target["c200c_median_inferred"], rtol=0, atol=2e-12)
    assert_allclose(state.peak_height, grid_target["shape_peak_height_inferred"], rtol=0, atol=2e-12)
    assert state.mvir_msun_h > 0.0


@pytest.mark.skipif(not _colossus_available(), reason="Colossus production backend is not installed")
def test_colossus_generated_grid_reproduces_frozen_catalog_values():
    # One grid is enough for a compact Colossus end-to-end smoke test; the
    # explicit-state tests above exercise all six stochastic streams.
    target = TARGETS["grids"][5]
    realization = generate_halo_population(publication_grids()[5], size=TARGETS["n_halo"])
    assert realization.colossus_version == "1.4.0"
    assert_allclose(realization.c200c[SELECTED], target["arrays"]["concentration"]["selected_values"], rtol=0, atol=3e-13)
    assert_allclose(realization.q_perp[SELECTED], target["arrays"]["q_projected"]["selected_values"], rtol=0, atol=3e-13)
