from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.controlled_benchmarks import (
    DEFAULT_CONTROLLED_CONFIG,
    append_d_spherical_leading_percent,
    centered_ellipse_summary,
    circular_safety_radius_over_r200c,
    displaced_ellipse_summary,
    displaced_spherical_summary,
    figure1_profile_radii,
    figure2_orientation_table,
    figure2_power_table,
    fit_cross_sine2,
    fit_interaction_cosine,
    make_controlled_context,
    shape_centering_interaction_summary,
)

TARGETS = json.loads(Path("reference/controlled_benchmarks/controlled_targets.json").read_text())


@pytest.fixture(scope="module")
def context():
    return make_controlled_context()


def test_default_configuration_matches_publication_values(context):
    cfg = context.config
    target = TARGETS["configuration"]
    assert cfg == DEFAULT_CONTROLLED_CONFIG
    assert cfg.mass_msun_h == target["M200c_Msun_h"]
    assert cfg.z_l == target["z_l"]
    assert cfg.z_s == target["z_s"]
    assert cfg.c200c == target["c200c_internal"]
    assert cfg.q_perp == target["q_perp"]
    assert_allclose(cfg.epsilon_perp, target["epsilon_perp"], rtol=0, atol=1e-15)
    assert_allclose(context.source_weight, TARGETS["source_weight"], rtol=0, atol=1e-15)


def test_figure1_profile_radius_list():
    radii = figure1_profile_radii()
    assert radii.size == 58
    assert np.all(np.diff(radii) > 0)
    assert np.any(radii == 0.3)
    assert_allclose(radii[0], 0.2, rtol=0, atol=1e-15)
    assert_allclose(radii[-1], 0.5, rtol=0, atol=1e-15)


def test_figure1_anchors(context):
    anchor = TARGETS["figure1_anchor"]
    ell = centered_ellipse_summary(anchor["R_over_r200c"], n_phi=anchor["n_phi"], context=context)
    off = displaced_spherical_summary(anchor["R_over_r200c"], 0.10, n_phi=anchor["n_phi"], context=context)
    assert_allclose(ell.delta_g_percent, anchor["centered_ellipse_percent"], rtol=0, atol=2e-12)
    # Historical figure/profile and direct Appendix-D paths differ at the
    # 1e-7 percentage-point level; this tolerance records their agreement
    # without forcing byte-level identity between legacy helpers.
    assert_allclose(off.delta_g_percent, anchor["displaced_spherical_d0p10_percent"], rtol=0, atol=2e-7)
    assert abs(ell.g_cross_mean) < 1e-15
    assert abs(off.g_cross_mean) < 1e-15


def test_appendix_d_spherical_direct_and_leading(context):
    block = TARGETS["appendix_d_spherical"]
    for item in block["offsets"]:
        d = item["d_over_r200c"]
        direct = displaced_spherical_summary(block["R_over_r200c"], d, n_phi=block["n_phi"], context=context)
        leading = append_d_spherical_leading_percent(block["R_over_r200c"], d, context=context)
        assert_allclose(direct.delta_g_percent, item["direct_percent"], rtol=0, atol=5e-12)
        assert_allclose(leading, item["leading_percent"], rtol=0, atol=5e-11)
    assert_allclose(
        circular_safety_radius_over_r200c(context=context),
        block["Rsc0_over_r200c_approx"],
        rtol=0,
        atol=2e-7,
    )


def test_coupled_orientation_anchors(context):
    block = TARGETS["coupled_orientation_anchors"]
    for angle_text, target in block["values_percent"].items():
        summary = displaced_ellipse_summary(
            block["R_over_r200c"], block["d_over_r200c"], float(angle_text),
            n_phi=block["n_phi"], context=context,
        )
        assert_allclose(summary.delta_g_percent, target, rtol=0, atol=5e-12)


def test_interaction_definition_uses_combined_minus_controls(context):
    interaction = shape_centering_interaction_summary(0.3, 0.10, 45.0, n_phi=8192, context=context)
    manual = (
        interaction.combined.delta_g_percent
        - interaction.shape_only.delta_g_percent
        - interaction.offset_only.delta_g_percent
    )
    assert_allclose(interaction.interaction_percent, manual, rtol=0, atol=0)
    assert interaction.shape_only.case == "centered_ellipse"
    assert interaction.offset_only.case == "displaced_spherical"
    assert interaction.combined.case == "displaced_ellipse"


def test_figure2_orientation_rows_match_frozen_inputs(context):
    rows = figure2_orientation_table(context=context)
    targets = TARGETS["figure2_panels_bc_orientation_rows"]
    assert len(rows) == len(targets) == 21
    for row, target in zip(rows, targets):
        assert row["d_over_r200c"] == target["d_over_r200c"]
        assert row["phi_offset_deg"] == target["phi_offset_deg"]
        assert_allclose(row["R_over_r200c"], target["R_over_r200c"], rtol=0, atol=1e-15)
        for key in ("delta_g_combined_fractional", "delta_g_shape_only_fractional", "g_cross_mean_reduced"):
            assert_allclose(row[key], target[key], rtol=0, atol=1e-12)
        # The offset-only control and therefore the interaction inherit the
        # public NFW branch correction.  The largest compact difference from
        # the frozen E2 table is about 5.3e-8 in fractional units.
        for key in ("delta_g_offset_only_fractional", "delta_g_interaction_fractional"):
            assert_allclose(row[key], target[key], rtol=0, atol=6e-8)


def test_figure2_interaction_fits_match_frozen_inputs(context):
    rows = figure2_orientation_table(context=context)
    targets = TARGETS["figure2_panel_b_interaction_fits"]
    for d_text, target in targets.items():
        fit = fit_interaction_cosine(rows, float(d_text))
        assert_allclose(fit["A0"], target["A0"], rtol=0, atol=6e-8)
        for key in ("A2_cos2alpha", "A4_cos4alpha", "fit_rms"):
            assert_allclose(fit[key], target[key], rtol=0, atol=1e-12)


def test_figure2_cross_sine2_fits_match_frozen_inputs(context):
    rows = figure2_orientation_table(context=context)
    targets = TARGETS["figure2_panel_c_cross_sine2_fits"]
    for d_text, target in targets.items():
        fit = fit_cross_sine2(rows, float(d_text))
        for key in ("B2_sin2alpha", "fit_rms"):
            assert_allclose(fit[key], target[key], rtol=0, atol=1e-12)


def test_figure2_power_fractions_match_frozen_targets(context):
    table = figure2_power_table(context=context)
    for target in TARGETS["figure2_panel_a_mode_fractions"]:
        row = next(r for r in table if r["case"] == target["case"] and int(r["m"]) == target["m"])
        # Centered ellipse fractions match roundoff. Miscentered-sphere values
        # differ at <=1.2e-7 from the frozen E2 table because this public path
        # uses the corrected robust NFW branch handling. The displayed physical
        # hierarchy is unchanged, and these compact targets guard against
        # larger convention errors.
        tol = 2e-7 if target["case"] == "miscentered_sphere" else 2e-12
        assert_allclose(row["kappa_power_fraction"], target["I1_fraction"], rtol=0, atol=tol)
