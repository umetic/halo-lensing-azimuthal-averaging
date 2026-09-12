from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from azlens.miscentering import draw_offset_realization
from azlens.mode_library import ModeLibrary
from azlens.population import publication_grids, sample_halo_population_from_state
from azlens.population_fields import (
    PopulationFieldConfig,
    compact_field_validation_metrics,
    evaluate_grid_population_fields,
    source_weight_map,
)
from azlens.selection import publication_radial_grid
from azlens.solver_settings import SolverSettings

TARGETS = json.loads(Path("reference/population_fields/population_field_targets.json").read_text())
POP_TARGETS = json.loads(Path("reference/population_realization/population_targets.json").read_text())


def small_settings() -> SolverSettings:
    return SolverSettings(
        name="test_population_fields_small",
        q_min=0.18,
        q_max=1.0,
        n_q=5,
        x_min=1.0e-4,
        x_max=20.0,
        n_x=24,
        x_solver_min=1.0e-5,
        x_solver_max=80.0,
        n_radial_solver=96,
        n_phi_solver=64,
        m_max=8,
        radial_interpolation="x",
    )


def make_smoke_inputs():
    target = TARGETS["smoke_evaluation"]
    grid = publication_grids()[0]
    state = POP_TARGETS["grids"][0]
    halo = sample_halo_population_from_state(
        grid,
        size=target["n_halo"],
        c200c_median=state["c200c_median_inferred"],
        shape_peak_height=state["shape_peak_height_inferred"],
    )
    offsets = draw_offset_realization(size=target["n_halo"], grid_index=grid.grid_index)
    config = PopulationFieldConfig(
        radial_grid=np.array(target["radial_grid"], dtype=np.float64),
        source_redshifts=tuple(target["source_redshifts"]),
        sigma_d=tuple(target["sigma_d"]),
        n_phi=target["n_phi"],
        chunk_size=3,
    )
    library = ModeLibrary.build(small_settings())
    return halo, offsets, config, library


def test_population_field_config_defaults_and_keys():
    cfg = PopulationFieldConfig()
    target = TARGETS["publication_radial_grid"]
    assert cfg.radial_grid.size == 36
    assert_array_equal(cfg.key_indices, target["key_indices"])
    assert_allclose(cfg.key_values, target["key_values"], rtol=0, atol=1e-15)
    assert cfg.finite_source_labels == ("z1", "z2")
    assert cfg.key_source_labels == ("z1", "z2", "infinity")


def test_source_weight_map_matches_publication_cosmology():
    weights = source_weight_map(0.2, (1.0, 2.0))
    assert_allclose(weights["z1"], 0.7991722748907393, rtol=0, atol=2e-14)
    assert_allclose(weights["z2"], 0.8943515658339105, rtol=0, atol=2e-14)
    assert weights["infinity"] == 1.0


def test_smoke_population_field_products_have_expected_shapes():
    halo, offsets, config, library = make_smoke_inputs()
    product = evaluate_grid_population_fields(halo, offsets, library, config=config)
    target = TARGETS["smoke_evaluation"]
    assert product.n_halo == target["n_halo"]
    assert product.lambda_min_infinity.shape == (2, target["n_halo"], 4)
    assert product.safe_local.shape == product.safe_outer.shape == product.lambda_min_infinity.shape
    assert product.represented_fraction_outer.shape == (2, 4)
    assert len(product.radial_summary_rows) == target["expected_summary_rows"]
    assert_allclose(product.key_samples["R_over_r200c"], target["radial_grid"], rtol=0, atol=0)
    assert product.key_samples["delta_g_plus_fractional"].shape == (3, 2, target["n_halo"], 4)
    assert product.key_samples["delta_g_shape_only"].shape == product.key_samples["delta_g_plus_fractional"].shape
    assert product.key_samples["d_over_r200c"].shape == (2, target["n_halo"])


def test_population_field_source_pairing_and_offset_reuse():
    halo, offsets, config, library = make_smoke_inputs()
    product = evaluate_grid_population_fields(halo, offsets, library, config=config)
    d = product.key_samples["d_over_r200c"]
    assert_allclose(d[0], 0.0, rtol=0, atol=0)
    assert_allclose(d[1], 0.05*offsets.rayleigh_unit, rtol=0, atol=0)
    assert_allclose(product.key_samples["phi_offset"], offsets.phi_offset, rtol=0, atol=0)
    weights = product.key_samples["source_weights"]
    assert_allclose(weights[:2], [product.source_weights["z1"], product.source_weights["z2"]], rtol=0, atol=0)
    assert weights[2] == 1.0


def test_population_field_safety_masks_are_outer_contiguous():
    halo, offsets, config, library = make_smoke_inputs()
    product = evaluate_grid_population_fields(halo, offsets, library, config=config)
    assert_array_equal(product.safe_outer, np.logical_and.accumulate(product.safe_local[..., ::-1], axis=-1)[..., ::-1])
    assert np.all(product.safe_outer <= product.safe_local)
    assert np.all((product.represented_fraction_outer >= 0.0) & (product.represented_fraction_outer <= 1.0))


def test_population_field_chunk_size_invariance_for_compact_products():
    halo, offsets, config, library = make_smoke_inputs()
    p1 = evaluate_grid_population_fields(halo, offsets, library, config=config)
    config2 = PopulationFieldConfig(
        radial_grid=config.radial_grid,
        source_redshifts=config.source_redshifts,
        sigma_d=config.sigma_d,
        n_phi=config.n_phi,
        chunk_size=halo.size,
    )
    p2 = evaluate_grid_population_fields(halo, offsets, library, config=config2)
    assert_allclose(p1.lambda_min_infinity, p2.lambda_min_infinity, rtol=0, atol=0)
    for key in ("delta_g_plus_fractional", "Delta_g_plus", "delta_g_interaction", "g_cross_mean_reduced"):
        assert_allclose(p1.key_samples[key], p2.key_samples[key], rtol=0, atol=0)
    assert p1.radial_summary_rows == p2.radial_summary_rows


def test_shape_offset_control_identity_in_key_samples():
    halo, offsets, config, library = make_smoke_inputs()
    product = evaluate_grid_population_fields(halo, offsets, library, config=config)
    combined = product.key_samples["delta_g_plus_fractional"]
    shape = product.key_samples["delta_g_shape_only"]
    offset = product.key_samples["delta_g_offset_only"]
    additive = product.key_samples["delta_g_additive"]
    interaction = product.key_samples["delta_g_interaction"]
    assert_allclose(additive, shape + offset, rtol=0, atol=0)
    assert_allclose(interaction, combined - shape - offset, rtol=0, atol=0)


def test_compact_metrics_are_report_friendly():
    halo, offsets, config, library = make_smoke_inputs()
    product = evaluate_grid_population_fields(halo, offsets, library, config=config)
    metrics = compact_field_validation_metrics(product)
    assert metrics["grid_index"] == 0
    assert metrics["n_halo"] == TARGETS["smoke_evaluation"]["n_halo"]
    assert metrics["n_summary_rows"] == TARGETS["smoke_evaluation"]["expected_summary_rows"]
    assert len(metrics["key_radii"]) == 4
    assert "z1" in metrics["source_weights"] and "z2" in metrics["source_weights"]
