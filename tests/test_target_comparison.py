from __future__ import annotations

import csv
import json
from pathlib import Path

from azlens.manifest import write_json
from azlens.target_comparison import (
    STATUS_FAIL,
    STATUS_INCOMPLETE,
    STATUS_PASS,
    check_run_completeness,
    compare_production_validation_run,
    format_validation_summary,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _make_valid_run(tmp_path: Path, repo: Path) -> Path:
    run = tmp_path / "production"
    stage_dir = run / "stages"
    stage_dir.mkdir(parents=True)
    for stage in ("preflight", "mode-library", "realization", "fields", "analysis"):
        payload = {"stage": stage, "status": "complete"}
        if stage == "preflight":
            payload["git"] = {"commit": "product-generation-test-commit"}
        write_json(stage_dir / f"{stage}.json", payload)
    write_json(run / "manifest_start.json", {"git": {"commit": "product-generation-test-commit"}})

    prod_targets = _load_json(repo / "reference/production_validation/production_validation_targets.json")
    pub_targets = _load_json(repo / "reference/publication_summary/publication_summary_targets.json")
    shape_targets = _load_json(repo / "reference/shape_response/centered_shape_targets.json")
    interaction_targets = _load_json(repo / "reference/interaction_diagnostics/interaction_diagnostics_targets.json")
    alignment_targets = _load_json(repo / "reference/alignment_reweighting/alignment_targets.json")

    # Minimal production-size field product files for completeness checks.
    for grid in range(prod_targets["publication_size"]["n_grids"]):
        grid_dir = run / "population_fields" / f"grid_{grid:02d}"
        grid_dir.mkdir(parents=True)
        (grid_dir / "product_arrays.npz").write_bytes(b"placeholder")
        (grid_dir / "radial_summary_rows.json").write_text("[]\n", encoding="utf-8")
        write_json(grid_dir / "product_metrics.json", {
            "grid_index": grid,
            "n_halo": prod_targets["publication_size"]["n_halos_per_grid"],
            "n_radial": prod_targets["publication_size"]["n_radii"],
            "n_sigma": prod_targets["publication_size"]["n_sigma_d"],
        })

    analysis = run / "analysis"
    analysis.mkdir(parents=True)
    common = dict(pub_targets["common_domain"])
    common["pass_indices"] = list(range(26, 36))
    write_json(analysis / "common_domain.json", common)

    table2_rows = []
    for target in pub_targets["table2"]:
        row = dict(target)
        row.update({
            "mass_msun_h": 2.0e15,
            "z_l": 0.5,
            "sigma_d": 0.05,
            "n_total": 10000,
            "n_outer_safe": 9518,
            "n_finite": 9518,
            "represented_fraction_outer": 0.9518,
            "p16_fractional": float(target["p16_percent"])/100.0,
            "p50_fractional": float(target["p50_percent"])/100.0,
            "p84_fractional": float(target["p84_percent"])/100.0,
        })
        table2_rows.append(row)
    write_json(analysis / "table2_selection.json", table2_rows)
    _write_csv(analysis / "table2_selection.csv", table2_rows)

    shape_rows = list(shape_targets["figure3_reference_rows"])
    _write_csv(analysis / "centered_shape_response.csv", shape_rows)
    write_json(analysis / "centered_shape_response_metrics.json", {
        "n_rows": shape_targets["expected_n_rows"],
        "grid_counts": {str(i): int(v) for i, v in enumerate(shape_targets["retained_counts_by_grid"])},
        "max_abs_direct_percent": max(abs(float(r["delta_g_median_percent"])) for r in shape_rows),
        "max_abs_leading_prediction_percent": max(abs(float(r["delta_g_leading_prediction_median_percent"])) for r in shape_rows),
    })

    restricted = interaction_targets["section_6_1_restricted"]
    conditioning_rows = []
    for target in interaction_targets["section_6_2_conditioning"]:
        row = dict(target)
        row.update({"mass_msun_h": 3.0e14, "z_l": 0.2, "source_label": "z1", "source_z": 1.0, "n_total": 10000, "n_selected": target["n_used"]})
        conditioning_rows.append(row)
    write_json(analysis / "interaction_diagnostics_metrics.json", {
        "n_hybrid_rows": 144,
        "restricted_hybrid": restricted,
        "conditioning_rows": conditioning_rows,
    })
    _write_csv(analysis / "conditioning_diagnostics.csv", conditioning_rows)
    _write_csv(analysis / "hybrid_interaction_diagnostics.csv", [{"placeholder": 0}])

    reported = alignment_targets["section_6_4_reported_diagnostics"]
    write_json(analysis / "alignment_diagnostics.json", {
        "n_rows": alignment_targets["current_paper_unfiltered_row_count"],
        "n_table3_records": len(alignment_targets["table3"]["rounded_percent_values"]),
        "q_change_min_percent_points": reported["Qoff_0_to_plus_0p30_change_percent_points"][0],
        "q_change_max_percent_points": reported["Qoff_0_to_plus_0p30_change_percent_points"][1],
        "sigma_change_min_percent_points": reported["sigma_0p02_to_0p08_change_at_Q0_percent_points"][0],
        "sigma_change_max_percent_points": reported["sigma_0p02_to_0p08_change_at_Q0_percent_points"][1],
        "max_family_difference_percent_points": reported["max_family_difference_percent_points"],
        "restricted_linear_r2_count": 360,
        "restricted_linear_r2_min": reported["restricted_linear_Rw2_min"],
        "restricted_linear_r2_median": reported["restricted_linear_Rw2_median"],
    })
    table3_rows = []
    for target in alignment_targets["table3"]["rounded_percent_values"]:
        values = target["values"]
        table3_rows.append({
            "grid_index": target["grid_index"],
            "mass_msun_h": target["mass_msun_h"],
            "z_l": target["z_l"],
            "R_over_r200c": alignment_targets["table3"]["R_over_r200c"],
            "sigma_d": alignment_targets["table3"]["sigma_d"],
            "p50_Qoff_minus_0p30_percent": values[0],
            "p50_Qoff_0p00_percent": values[1],
            "p50_Qoff_plus_0p30_percent": values[2],
        })
    _write_csv(analysis / "table3_selection.csv", table3_rows)
    _write_csv(analysis / "alignment_reweighted_summary.csv", [{"placeholder": 0}])
    write_json(analysis / "analysis_manifest.json", {"generated": prod_targets["expected_analysis_products"]})
    (analysis / "radial_summary_all_grids.csv").write_text("placeholder\n", encoding="utf-8")
    return run


def test_run_completeness_detects_missing_products(tmp_path: Path):
    repo = _repository_root()
    targets = _load_json(repo / "reference/production_validation/production_validation_targets.json")
    result = check_run_completeness(tmp_path / "missing", targets)
    assert result.status == STATUS_INCOMPLETE


def test_target_comparison_passes_for_valid_synthetic_run(tmp_path: Path):
    repo = _repository_root()
    run = _make_valid_run(tmp_path, repo)
    report = compare_production_validation_run(run, repo)
    assert report["overall_status"] == STATUS_PASS
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["run_completeness"] == STATUS_PASS
    assert statuses["common_domain"] == STATUS_PASS
    assert statuses["table2"] == STATUS_PASS
    assert statuses["shape_response"] == STATUS_PASS
    assert statuses["interaction_diagnostics"] == STATUS_PASS
    assert statuses["alignment_diagnostics"] == STATUS_PASS
    assert statuses["table3"] == STATUS_PASS
    text = format_validation_summary(report)
    assert "Production validation: PASS" in text
    assert "Table 3" not in text or "table3: PASS" in text


def test_target_comparison_fails_on_wrong_signed_table2_value(tmp_path: Path):
    repo = _repository_root()
    run = _make_valid_run(tmp_path, repo)
    table_path = run / "analysis" / "table2_selection.json"
    rows = _load_json(table_path)
    rows[0]["p50_percent"] = -float(rows[0]["p50_percent"])
    write_json(table_path, rows)
    report = compare_production_validation_run(run, repo)
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert report["overall_status"] == STATUS_FAIL
    assert statuses["table2"] == STATUS_FAIL
