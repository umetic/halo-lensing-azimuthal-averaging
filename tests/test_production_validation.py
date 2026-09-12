from __future__ import annotations

from pathlib import Path

import numpy as np

from azlens.mode_library import ModeLibrary
from azlens.production_validation import (
    STAGES,
    _field_config_from_workflow,
    _mode_settings_from_config,
    default_run_id,
    load_halo_population_npz,
    load_offset_npz,
    load_population_field_product,
    load_workflow_config,
    make_run_context,
    normalize_workflow_config,
    run_stages,
    save_halo_population_npz,
    save_offset_npz,
    save_population_field_product,
    smoke_config,
)
from azlens.miscentering import draw_offset_realization
from azlens.population import publication_grids, sample_halo_population_from_state
from azlens.population_fields import PopulationFieldConfig, evaluate_grid_population_fields
from azlens.selection import publication_radial_grid
from azlens.solver_settings import SolverSettings


def test_default_run_id_format_is_stable():
    from datetime import datetime, timezone
    assert default_run_id(datetime(2026, 9, 12, 12, 34, 56, tzinfo=timezone.utc)) == "20260912T123456Z"


def test_config_normalization_and_yaml_loading(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "run_id: unit\n"
        "validation:\n  require_clean_git: false\n"
        "population:\n  n_halos: 3\n  grids: [0]\n  state_source: reference\n"
        "mode_library:\n  profile: smoke\n",
        encoding="utf-8",
    )
    cfg = load_workflow_config(config_path)
    assert cfg["run_id"] == "unit"
    assert cfg["population"]["n_halos"] == 3
    assert cfg["population"]["grids"] == [0]
    assert cfg["validation"]["require_clean_git"] is False


def test_invalid_grid_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        normalize_workflow_config({"population": {"grids": [7]}})


def test_mode_settings_from_smoke_config():
    settings = _mode_settings_from_config(smoke_config())
    assert settings.name == "production_validation_smoke"
    assert settings.radial_interpolation == "x"
    assert settings.n_q == 5


def test_field_config_from_workflow_publication_grid():
    cfg = normalize_workflow_config({"fields": {"radial_grid": "publication"}})
    field_cfg = _field_config_from_workflow(cfg)
    np.testing.assert_allclose(field_cfg.radial_grid, publication_radial_grid())


def test_realization_save_load_roundtrip(tmp_path: Path):
    grid = publication_grids()[0]
    halo = sample_halo_population_from_state(grid, size=5, c200c_median=4.0, shape_peak_height=2.0)
    offsets = draw_offset_realization(size=5, grid_index=0)
    halo_path = tmp_path / "halo.npz"
    offset_path = tmp_path / "offset.npz"
    save_halo_population_npz(halo, halo_path)
    save_offset_npz(offsets, offset_path)
    loaded_halo = load_halo_population_npz(halo_path)
    loaded_offset = load_offset_npz(offset_path)
    assert loaded_halo.grid == halo.grid
    np.testing.assert_allclose(loaded_halo.c200c, halo.c200c)
    np.testing.assert_allclose(loaded_halo.q_perp, halo.q_perp)
    np.testing.assert_allclose(loaded_offset.rayleigh_unit, offsets.rayleigh_unit)
    np.testing.assert_allclose(loaded_offset.phi_offset, offsets.phi_offset)


def test_population_field_product_roundtrip(tmp_path: Path):
    grid = publication_grids()[0]
    halo = sample_halo_population_from_state(grid, size=4, c200c_median=4.0, shape_peak_height=2.0)
    offsets = draw_offset_realization(size=4, grid_index=0)
    settings = SolverSettings(
        name="product_roundtrip", q_min=0.18, q_max=1.0, n_q=4,
        x_min=1.0e-4, x_max=80.0, n_x=16, x_solver_min=1.0e-5,
        x_solver_max=1.0e3, n_radial_solver=64, n_phi_solver=32, m_max=6,
        radial_interpolation="x",
    )
    library = ModeLibrary.build(settings)
    config = PopulationFieldConfig(
        radial_grid=[0.2924408238028535, 0.5113850166642591],
        source_redshifts=(1.0,), sigma_d=(0.0, 0.05), n_phi=16, chunk_size=2,
    )
    product = evaluate_grid_population_fields(halo, offsets, library, config=config)
    save_population_field_product(product, tmp_path / "product")
    loaded = load_population_field_product(tmp_path / "product")
    assert loaded.grid_index == product.grid_index
    assert loaded.source_weights == product.source_weights
    np.testing.assert_allclose(loaded.radial_grid, product.radial_grid)
    np.testing.assert_array_equal(loaded.safe_outer, product.safe_outer)
    assert len(loaded.radial_summary_rows) == len(product.radial_summary_rows)
    np.testing.assert_allclose(loaded.key_samples["epsilon_perp"], product.key_samples["epsilon_perp"])


def test_smoke_workflow_runs_all_stages(tmp_path: Path):
    # Use the repository root because the smoke workflow intentionally reads
    # committed compact reference targets.  Route all outputs to a temporary dir.
    root = Path(__file__).resolve().parents[1]
    context = make_run_context(root, config_path=None, run_id="unit_smoke", output_root=tmp_path, allow_dirty=True)
    context = type(context)(context.repository_root, context.run_dir, smoke_config(n_halos=12), None, True)
    records = run_stages(context, STAGES)
    assert [record["stage"] for record in records] == list(STAGES)
    assert (context.run_dir / "manifest_start.json").is_file()
    assert (context.run_dir / "mode_library" / "population_mode_library.npz").is_file()
    assert (context.run_dir / "realization" / "grid_00_halo_realization.npz").is_file()
    assert (context.run_dir / "population_fields" / "grid_00" / "product_arrays.npz").is_file()
    assert (context.run_dir / "analysis" / "centered_shape_response.csv").is_file()
    assert (context.run_dir / "comparison" / "validation_summary.json").is_file()
