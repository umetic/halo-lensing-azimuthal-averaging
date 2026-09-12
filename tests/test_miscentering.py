from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.miscentering import DEFAULT_SIGMA_D, draw_offset_realization, float64_sha256

TARGETS = json.loads(Path("reference/population_realization/population_targets.json").read_text())
SELECTED = TARGETS["selected_indices"]


def test_offset_seed_sequence_and_ranges():
    offsets = draw_offset_realization(size=128, grid_index=3)
    assert offsets.grid_index == 3
    assert offsets.base_seed == TARGETS["offset_base_seed"]
    assert offsets.seed_sequence_entropy == (TARGETS["offset_base_seed"], 3)
    assert offsets.rayleigh_unit.shape == (128,)
    assert offsets.phi_offset.shape == (128,)
    assert np.all(offsets.rayleigh_unit >= 0.0)
    assert np.all((offsets.phi_offset >= 0.0) & (offsets.phi_offset < 2.0*np.pi))


def test_offset_amplitude_matrix_reuses_unit_draws():
    offsets = draw_offset_realization(size=17, grid_index=1)
    matrix = offsets.amplitude_matrix(DEFAULT_SIGMA_D)
    assert matrix.shape == (4, 17)
    assert_allclose(matrix[0], 0.0, rtol=0, atol=0)
    for row, sigma in enumerate(DEFAULT_SIGMA_D):
        assert_allclose(matrix[row], sigma*offsets.rayleigh_unit, rtol=0, atol=0)


def test_invalid_offset_inputs_are_rejected():
    with pytest.raises(ValueError):
        draw_offset_realization(size=0, grid_index=0)
    with pytest.raises(ValueError):
        draw_offset_realization(size=1, grid_index=-1)
    offsets = draw_offset_realization(size=3, grid_index=0)
    with pytest.raises(ValueError):
        offsets.amplitudes(-0.1)
    with pytest.raises(ValueError):
        offsets.amplitude_matrix([0.0, np.nan])


@pytest.mark.parametrize("grid_target", TARGETS["grids"], ids=lambda g: f"grid{g['grid_index']}")
def test_offset_stream_matches_frozen_publication_catalog(grid_target):
    offsets = draw_offset_realization(size=TARGETS["n_halo"], grid_index=grid_target["grid_index"])
    # The RNG angles are bitwise-identical across the checked runtime, while
    # the derived Rayleigh amplitudes can differ at a few ulp because they
    # involve transcendental functions.  Keep those validation concepts
    # separate.
    assert offsets.digest_summary()["phi_offset_sha256_float64_le"] == grid_target["arrays"]["alpha_offset"]["sha256_float64_le"]
    assert_allclose(offsets.rayleigh_unit[SELECTED], grid_target["arrays"]["rayleigh_unit"]["selected_values"], rtol=0, atol=6e-16)
    assert_allclose(offsets.phi_offset[SELECTED], grid_target["arrays"]["alpha_offset"]["selected_values"], rtol=0, atol=0)
    assert_allclose(offsets.amplitude_matrix()[..., SELECTED], grid_target["arrays"]["d_over_r200"]["selected_columns"], rtol=0, atol=6e-16)
