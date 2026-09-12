from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from azlens.population_fields import PopulationFieldProducts
from azlens.publication_summary import (
    TABLE2_OBSERVABLES,
    build_publication_summary_from_rows,
    common_domain_from_products,
    common_domain_from_rows,
    radial_summary_rows_from_products,
    read_radial_summary_csv,
    select_table2_rows,
    table2_records_to_matrix,
    write_radial_summary_csv,
)
from azlens.selection import publication_radial_grid

TARGETS = json.loads(Path("reference/publication_summary/publication_summary_targets.json").read_text())


def _synthetic_rows():
    radii = publication_radial_grid()
    common = TARGETS["common_domain"]
    table_rows = TARGETS["table2"]
    rows = []
    previous = common["previous_radius"]
    r_min = common["r_min"]
    all_radii = [previous] + common["radii"]
    source_z = {"z1": 1.0, "z2": 2.0}
    # Baseline low-amplitude rows for every grid/radius/observable/source.
    for grid in range(6):
        for radius in all_radii:
            rep = 1.0
            if np.isclose(radius, previous, rtol=0, atol=1e-14) and grid == common["limiting_grid_index"]:
                rep = common["previous_limiting_fraction"]
            elif np.isclose(radius, r_min, rtol=0, atol=1e-14) and grid == common["limiting_grid_index"]:
                rep = common["limiting_fraction"]
            for source_label in ("z1", "z2"):
                for observable in TABLE2_OBSERVABLES:
                    rows.append({
                        "grid_index": grid,
                        "mass_msun_h": [3e14, 1e15, 2e15, 3e14, 1e15, 2e15][grid],
                        "z_l": [0.2, 0.2, 0.2, 0.5, 0.5, 0.5][grid],
                        "source_label": source_label,
                        "source_z": source_z[source_label],
                        "sigma_d": 0.05,
                        "R_over_r200c": float(radius),
                        "observable": observable,
                        "n_total": 10000,
                        "n_outer_safe": int(round(10000*rep)),
                        "n_selected": int(round(10000*rep)),
                        "n_finite": int(round(10000*rep)),
                        "represented_fraction_outer": float(rep),
                        "finite_fraction_selected": 1.0,
                        "mean_fractional": 0.0,
                        "p16_fractional": -0.001,
                        "p50_fractional": 1.0e-6 if source_label == "z1" else -1.0e-6,
                        "p84_fractional": 0.001,
                        "p97p5_fractional": 0.002,
                    })
    # Add a tempting row outside the common domain; it must not be selected.
    for source_label in ("z1", "z2"):
        rows.append({
            "grid_index": 5, "mass_msun_h": 2e15, "z_l": 0.5, "source_label": source_label, "source_z": source_z[source_label],
            "sigma_d": 0.05, "R_over_r200c": float(previous), "observable": "magnification",
            "n_total": 10000, "n_outer_safe": 8401, "n_selected": 8401, "n_finite": 8401,
            "represented_fraction_outer": common["previous_limiting_fraction"], "finite_fraction_selected": 1.0,
            "mean_fractional": 0.0, "p16_fractional": 99.0, "p50_fractional": 50.0, "p84_fractional": 100.0,
            "p97p5_fractional": 100.0,
        })
    # Overwrite target rows by appending stronger selected candidates at the publication grid/radius.
    # Duplicates with same represented fraction are allowed and mimic observable/source repetitions.
    for target in table_rows:
        rows.append({
            "grid_index": target["grid_index"],
            "mass_msun_h": 2e15,
            "z_l": 0.5,
            "source_label": target["source_label"],
            "source_z": target["source_z"],
            "sigma_d": 0.05,
            "R_over_r200c": target["R_over_r200c"],
            "observable": target["observable"],
            "n_total": 10000,
            "n_outer_safe": 9518,
            "n_selected": 9518,
            "n_finite": 9518,
            "represented_fraction_outer": common["limiting_fraction"],
            "finite_fraction_selected": 1.0,
            "mean_fractional": 0.0,
            "p16_fractional": target["p16_percent"]/100.0,
            "p50_fractional": target["p50_percent"]/100.0,
            "p84_fractional": target["p84_percent"]/100.0,
            "p97p5_fractional": 0.0,
        })
    return rows


def test_common_domain_from_rows_matches_publication_target():
    domain = common_domain_from_rows(_synthetic_rows())
    common = TARGETS["common_domain"]
    assert domain.n_radii == common["n_radii"]
    assert_allclose(domain.radii, common["radii"], rtol=0, atol=1e-15)
    assert_allclose(domain.r_min, common["r_min"], rtol=0, atol=1e-15)
    assert domain.limiting_grid_index == common["limiting_grid_index"]
    assert_allclose(domain.limiting_fraction, common["limiting_fraction"], rtol=0, atol=0)
    assert_allclose(domain.limiting_radius, common["limiting_radius"], rtol=0, atol=1e-15)


def test_common_domain_requires_all_grids_to_pass_threshold():
    rows = _synthetic_rows()
    domain = common_domain_from_rows(rows)
    assert not domain.contains_radius(TARGETS["common_domain"]["previous_radius"])
    assert domain.contains_radius(TARGETS["common_domain"]["r_min"])


def test_common_domain_from_rows_rejects_inconsistent_duplicate_represented_fraction():
    rows = _synthetic_rows()
    rows.append(dict(rows[0], represented_fraction_outer=0.12345))
    with pytest.raises(ValueError):
        common_domain_from_rows(rows)


def test_select_table2_rows_preserves_signed_median_and_selected_quantiles():
    rows = _synthetic_rows()
    domain = common_domain_from_rows(rows)
    records = select_table2_rows(rows, domain)
    assert len(records) == 10
    for record, target in zip(records, TARGETS["table2"]):
        assert record.observable == target["observable"]
        assert record.source_label == target["source_label"]
        assert record.grid_index == target["grid_index"]
        assert_allclose(record.R_over_r200c, target["R_over_r200c"], rtol=0, atol=1e-15)
        assert_allclose(record.p50_percent, target["p50_percent"], rtol=0, atol=2e-13)
        assert_allclose(record.p16_percent, target["p16_percent"], rtol=0, atol=2e-13)
        assert_allclose(record.p84_percent, target["p84_percent"], rtol=0, atol=2e-13)


def test_table2_selection_does_not_maximize_p16_or_p84_separately():
    rows = _synthetic_rows()
    common = TARGETS["common_domain"]
    # This row is in the common domain and has huge P16/P84 but a small P50.
    rows.append(dict(rows[0], grid_index=4, R_over_r200c=common["r_min"], observable="reduced_tangential_shear",
                     source_label="z1", source_z=1.0, represented_fraction_outer=1.0,
                     p16_fractional=10.0, p50_fractional=1e-5, p84_fractional=20.0))
    records = select_table2_rows(rows, common_domain_from_rows(rows))
    first = records[0]
    assert first.observable == "reduced_tangential_shear"
    assert first.source_label == "z1"
    assert first.grid_index == 5
    assert_allclose(first.p16_percent, TARGETS["table2"][0]["p16_percent"], rtol=0, atol=0)


def test_build_publication_summary_from_rows_combines_domain_and_table2():
    result = build_publication_summary_from_rows(_synthetic_rows())
    assert result.common_domain.n_radii == TARGETS["common_domain"]["n_radii"]
    assert len(result.table2_records) == 10
    as_dict = result.as_dict()
    assert "common_domain" in as_dict and "table2_records" in as_dict


def test_table2_records_to_matrix_is_keyed_by_observable_and_source():
    result = build_publication_summary_from_rows(_synthetic_rows())
    matrix = table2_records_to_matrix(result.table2_records)
    assert set(matrix) == set(TABLE2_OBSERVABLES)
    assert_allclose(matrix["magnification"]["z2"]["p50_percent"], 1.8374216282685998, rtol=0, atol=1e-14)


def test_radial_summary_csv_roundtrip(tmp_path):
    rows = _synthetic_rows()[:17]
    path = tmp_path/"summary.csv"
    write_radial_summary_csv(rows, path)
    loaded = read_radial_summary_csv(path)
    assert len(loaded) == len(rows)
    assert loaded[0]["observable"] == rows[0]["observable"]
    assert_allclose(float(loaded[0]["R_over_r200c"]), float(rows[0]["R_over_r200c"]), rtol=0, atol=0)
    assert int(loaded[0]["grid_index"]) == int(rows[0]["grid_index"])


def test_common_domain_from_products_uses_product_represented_matrix():
    radii = np.array([0.2, 0.4, 0.8])
    products = []
    for grid, row in enumerate(([0.2, 0.95, 1.0], [0.3, 0.97, 1.0], [0.4, 0.96, 1.0])):
        safe = np.ones((2, 3, 3), dtype=bool)
        products.append(PopulationFieldProducts(
            grid_index=grid,
            mass_msun_h=1.0,
            z_l=0.2,
            radial_grid=radii,
            sigma_d=(0.0, 0.05),
            source_redshifts=(1.0, 2.0),
            source_weights={"z1": 0.8, "z2": 0.9, "infinity": 1.0},
            key_indices=np.array([0, 1, 2]),
            lambda_min_infinity=np.ones((2, 3, 3)),
            safe_local=safe,
            safe_outer=safe,
            represented_fraction_outer=np.array([[1.0, 1.0, 1.0], row]),
            radial_summary_rows=tuple(),
            key_samples={"R_over_r200c": radii},
        ))
    domain = common_domain_from_products(products)
    assert_allclose(domain.radii, [0.4, 0.8], rtol=0, atol=0)
    assert domain.limiting_grid_index == 0
    assert_allclose(domain.limiting_fraction, 0.95, rtol=0, atol=0)


def test_radial_summary_rows_from_products_flattens_rows():
    rows = _synthetic_rows()[:2]
    radii = np.array([0.2])
    safe = np.ones((1, 1, 1), dtype=bool)
    product = PopulationFieldProducts(
        grid_index=0,
        mass_msun_h=1.0,
        z_l=0.2,
        radial_grid=radii,
        sigma_d=(0.05,),
        source_redshifts=(1.0,),
        source_weights={"z1": 0.8, "infinity": 1.0},
        key_indices=np.array([0]),
        lambda_min_infinity=np.ones((1, 1, 1)),
        safe_local=safe,
        safe_outer=safe,
        represented_fraction_outer=np.ones((1, 1)),
        radial_summary_rows=tuple(rows),
        key_samples={"R_over_r200c": radii},
    )
    assert radial_summary_rows_from_products([product]) == tuple(rows)
