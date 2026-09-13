"""Target-aware comparison for production-validation runs.

This module reads generated analysis products from a production-validation run
and compares them with committed compact publication targets.  It does not run
lensing calculations, modify outputs, or define scientific observables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .manifest import collect_git_state, file_manifest, read_json, utc_now_iso, write_json

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_SKIPPED = "SKIPPED"

DEFAULT_TOLERANCES: dict[str, float] = {
    "radius_abs": 1.0e-14,
    "fraction_abs": 1.0e-12,
    "percent_abs_strict": 1.0e-9,
    "percent_abs_display": 6.0e-3,
    "shape_abs": 1.0e-8,
    "r2_abs_strict": 1.0e-10,
    "reported_percent_point_abs": 1.0e-3,
    "reported_r2_abs": 5.0e-4,
    "table3_display_abs": 5.0e-4,
}

DEFAULT_TARGET_FILES: dict[str, str] = {
    "production_validation": "reference/production_validation/production_validation_targets.json",
    "publication_summary": "reference/publication_summary/publication_summary_targets.json",
    "shape_response": "reference/shape_response/centered_shape_targets.json",
    "interaction_diagnostics": "reference/interaction_diagnostics/interaction_diagnostics_targets.json",
    "alignment_reweighting": "reference/alignment_reweighting/alignment_targets.json",
}


@dataclass
class CheckResult:
    """One target-comparison result."""

    name: str
    status: str
    message: str = ""
    n_checked: int = 0
    max_abs_difference: float | None = None
    tolerance: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "n_checked": int(self.n_checked),
            "max_abs_difference": self.max_abs_difference,
            "tolerance": self.tolerance,
            "details": self.details,
        }


def _json(path: Path) -> Any:
    return read_json(path)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return obj


def _float(value: Any) -> float:
    return float(value)


def _int(value: Any) -> int:
    return int(round(float(value)))


def _close(actual: Any, expected: Any, tol: float) -> float:
    diff = abs(float(actual) - float(expected))
    if not np.isfinite(diff) or diff > float(tol):
        raise AssertionError(f"actual={actual!r}, expected={expected!r}, diff={diff:g}, tol={tol:g}")
    return float(diff)


def _equal(actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"actual={actual!r}, expected={expected!r}")


def _max_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return None if not vals else float(max(vals))


def _status_from_checks(checks: Sequence[CheckResult]) -> str:
    statuses = {check.status for check in checks}
    if STATUS_FAIL in statuses:
        return STATUS_FAIL
    if STATUS_INCOMPLETE in statuses:
        return STATUS_INCOMPLETE
    if statuses and statuses <= {STATUS_SKIPPED}:
        return STATUS_SKIPPED
    return STATUS_PASS


def _target_paths(repository_root: Path, production_targets: Mapping[str, Any]) -> dict[str, Path]:
    mapping = dict(DEFAULT_TARGET_FILES)
    for key, value in _as_mapping(production_targets.get("target_files", {}), "target_files").items():
        mapping[str(key)] = str(value)
    return {key: (repository_root / value).resolve() for key, value in mapping.items()}


def _tolerances(production_targets: Mapping[str, Any]) -> dict[str, float]:
    result = dict(DEFAULT_TOLERANCES)
    for key, value in _as_mapping(production_targets.get("tolerances", {}), "tolerances").items():
        result[str(key)] = float(value)
    return result


def check_run_completeness(run_dir: Path, production_targets: Mapping[str, Any]) -> CheckResult:
    """Check that a run directory contains publication-size products."""
    try:
        pub = _as_mapping(production_targets["publication_size"], "publication_size")
        n_grids = int(pub["n_grids"])
        n_halos = int(pub["n_halos_per_grid"])
        n_sigma = int(pub["n_sigma_d"])
        n_radii = int(pub["n_radii"])
        required_stages = [str(s) for s in production_targets.get("required_stages", []) if str(s) != "compare"]
        required_analysis = [str(name) for name in production_targets.get("expected_analysis_products", [])]
        missing: list[str] = []
        bad: list[str] = []
        for stage in required_stages:
            path = run_dir / "stages" / f"{stage}.json"
            if not path.is_file():
                missing.append(str(path.relative_to(run_dir)))
                continue
            record = _as_mapping(_json(path), f"stage {stage}")
            if record.get("status") != "complete":
                bad.append(f"{stage}: status={record.get('status')!r}")
        field_records: list[dict[str, Any]] = []
        for grid_index in range(n_grids):
            grid_dir = run_dir / "population_fields" / f"grid_{grid_index:02d}"
            for filename in ("product_arrays.npz", "product_metrics.json", "radial_summary_rows.json"):
                path = grid_dir / filename
                if not path.is_file():
                    missing.append(str(path.relative_to(run_dir)))
            metrics_path = grid_dir / "product_metrics.json"
            if metrics_path.is_file():
                metrics = _as_mapping(_json(metrics_path), f"metrics grid {grid_index}")
                field_records.append({"grid_index": grid_index, **dict(metrics)})
                if int(metrics.get("n_halo", -1)) != n_halos:
                    bad.append(f"grid {grid_index}: n_halo={metrics.get('n_halo')!r}")
                if int(metrics.get("n_radial", -1)) != n_radii:
                    bad.append(f"grid {grid_index}: n_radial={metrics.get('n_radial')!r}")
                if int(metrics.get("n_sigma", -1)) != n_sigma:
                    bad.append(f"grid {grid_index}: n_sigma={metrics.get('n_sigma')!r}")
        analysis_dir = run_dir / "analysis"
        for name in required_analysis:
            path = analysis_dir / name
            if not path.is_file():
                missing.append(str(path.relative_to(run_dir)))
        if missing or bad:
            return CheckResult(
                "run_completeness", STATUS_INCOMPLETE,
                "missing or non-publication-size products", n_checked=len(field_records),
                details={"missing": missing, "bad": bad},
            )
        return CheckResult(
            "run_completeness", STATUS_PASS,
            "publication-size run products are present", n_checked=n_grids,
            details={"field_records": field_records},
        )
    except Exception as exc:
        return CheckResult("run_completeness", STATUS_INCOMPLETE, str(exc))


def check_common_domain(run_dir: Path, targets: Mapping[str, Any], tol: Mapping[str, float]) -> CheckResult:
    try:
        actual = _as_mapping(_json(run_dir / "analysis" / "common_domain.json"), "common_domain")
        expected = _as_mapping(targets["common_domain"], "target common_domain")
        diffs: list[float] = []
        for key in ("sigma_d", "threshold", "limiting_fraction", "limiting_radius", "r_min", "r_max"):
            diffs.append(_close(actual[key], expected[key], tol["radius_abs"] if "radius" in key or key in {"r_min", "r_max"} else tol["fraction_abs"]))
        for key in ("limiting_grid_index", "n_radii"):
            _equal(_int(actual[key]), _int(expected[key]))
        expected_radii = [float(x) for x in expected["radii"]]
        actual_radii = [float(x) for x in actual["radii"]]
        _equal(len(actual_radii), len(expected_radii))
        for a, e in zip(actual_radii, expected_radii):
            diffs.append(_close(a, e, tol["radius_abs"]))
        if "pass_indices" in actual:
            _equal([int(x) for x in actual["pass_indices"]], list(range(26, 36)))
        return CheckResult("common_domain", STATUS_PASS, "common domain recovered", n_checked=6 + len(expected_radii), max_abs_difference=_max_or_none(diffs), tolerance=max(tol["radius_abs"], tol["fraction_abs"]))
    except Exception as exc:
        return CheckResult("common_domain", STATUS_FAIL, str(exc))


def check_table2(run_dir: Path, targets: Mapping[str, Any], tol: Mapping[str, float]) -> CheckResult:
    try:
        actual_rows = _json(run_dir / "analysis" / "table2_selection.json")
        if not isinstance(actual_rows, list):
            raise ValueError("table2_selection.json must contain a list")
        actual = {(str(r["observable"]), str(r["source_label"])): r for r in actual_rows}
        target_rows = targets["table2"]
        diffs: list[float] = []
        checked = 0
        for target in target_rows:
            key = (str(target["observable"]), str(target["source_label"]))
            if key not in actual:
                raise AssertionError(f"missing Table 2 row {key}")
            row = actual[key]
            for exact_key in ("grid_index",):
                _equal(_int(row[exact_key]), _int(target[exact_key]))
            for float_key in ("R_over_r200c", "source_z"):
                diffs.append(_close(row[float_key], target[float_key], tol["radius_abs"]))
            diffs.append(_close(row.get("sigma_d", 0.05), 0.05, tol["radius_abs"]))
            if "n_total" in row:
                _equal(_int(row["n_total"]), 10000)
            if "n_outer_safe" in row:
                _equal(_int(row["n_outer_safe"]), 9518)
            if "n_finite" in row:
                _equal(_int(row["n_finite"]), 9518)
            diffs.append(_close(row["p50_percent"], target["p50_percent"], tol["percent_abs_strict"]))
            # P16/P84 targets are manuscript-display values in the compact reference file.
            for display_key in ("p16_percent", "p84_percent"):
                diffs.append(_close(row[display_key], target[display_key], tol["percent_abs_display"]))
            checked += 1
        _equal(len(actual_rows), len(target_rows))
        return CheckResult("table2", STATUS_PASS, "Table 2 selected rows recovered", n_checked=checked, max_abs_difference=_max_or_none(diffs), tolerance=max(tol["percent_abs_strict"], tol["percent_abs_display"]))
    except Exception as exc:
        return CheckResult("table2", STATUS_FAIL, str(exc))


def check_shape_response(run_dir: Path, targets: Mapping[str, Any], tol: Mapping[str, float]) -> CheckResult:
    try:
        metrics = _as_mapping(_json(run_dir / "analysis" / "centered_shape_response_metrics.json"), "shape metrics")
        raw_counts = targets["retained_counts_by_grid"]
        if isinstance(raw_counts, Mapping):
            expected_counts = {int(k): int(v) for k, v in raw_counts.items()}
        else:
            expected_counts = {i: int(v) for i, v in enumerate(raw_counts)}
        actual_counts = {int(k): int(v) for k, v in _as_mapping(metrics["grid_counts"], "grid_counts").items()}
        _equal(actual_counts, expected_counts)
        _equal(_int(metrics["n_rows"]), int(targets["expected_n_rows"]))
        rows = _csv_rows(run_dir / "analysis" / "centered_shape_response.csv")
        target_rows = list(targets.get("figure3_reference_rows", []))
        _equal(len(rows), int(targets["expected_n_rows"]))
        diffs: list[float] = []
        if target_rows:
            actual = {(int(float(r["grid_index"])), int(float(r["bin_index"]))): r for r in rows}
            numeric_keys = (
                "M200c_Msun_h", "z_l", "z_s", "sigma_d", "R_over_R200c", "n_bin",
                "epsilon_perp_sq_median", "delta_g_median_percent",
                "delta_g_leading_prediction_median_percent",
                "delta_g_second_order_prediction_median_percent",
                "leading_response_coefficient_median_percent",
            )
            for target in target_rows:
                key = (int(target["grid_index"]), int(target["bin_index"]))
                row = actual.get(key)
                if row is None:
                    raise AssertionError(f"missing shape-response row {key}")
                for name in numeric_keys:
                    tolerance = 0.0 if name == "n_bin" else tol["shape_abs"]
                    if name == "n_bin":
                        _equal(_int(row[name]), _int(target[name]))
                    else:
                        diffs.append(_close(row[name], target[name], tolerance))
        return CheckResult("shape_response", STATUS_PASS, "centered-shape response recovered", n_checked=len(target_rows) if target_rows else len(rows), max_abs_difference=_max_or_none(diffs), tolerance=tol["shape_abs"])
    except Exception as exc:
        return CheckResult("shape_response", STATUS_FAIL, str(exc))


def _find_conditioning_row(rows: Sequence[Mapping[str, Any]], target: Mapping[str, Any], tol_radius: float) -> Mapping[str, Any]:
    for row in rows:
        if _int(row["grid_index"]) == _int(target["grid_index"]) and abs(_float(row["sigma_d"]) - _float(target["sigma_d"])) <= 5.0e-15 and abs(_float(row["R_over_r200c"]) - _float(target["R_over_r200c"])) <= tol_radius:
            return row
    raise AssertionError(f"missing conditioning row for grid={target['grid_index']} sigma={target['sigma_d']} R={target['R_over_r200c']}")


def check_interaction_diagnostics(run_dir: Path, targets: Mapping[str, Any], tol: Mapping[str, float]) -> CheckResult:
    try:
        metrics = _as_mapping(_json(run_dir / "analysis" / "interaction_diagnostics_metrics.json"), "interaction metrics")
        restricted = _as_mapping(metrics["restricted_hybrid"], "restricted_hybrid")
        target = _as_mapping(targets["section_6_1_restricted"], "section_6_1_restricted")
        diffs = [
            _close(restricted["r2_min"], target["r2_min"], tol["r2_abs_strict"]),
            _close(restricted["r2_median"], target["r2_median"], tol["r2_abs_strict"]),
        ]
        _equal(str(restricted["domain"]), str(target["domain"]))
        _equal(_int(restricted["n_fits"]), _int(target["n_fits"]))
        _equal(_int(metrics["n_hybrid_rows"]), 144)
        conditioning_rows = [_as_mapping(row, "conditioning row") for row in metrics.get("conditioning_rows", [])]
        for crow in targets["section_6_2_conditioning"]:
            row = _find_conditioning_row(conditioning_rows, crow, tol["radius_abs"])
            _equal(_int(row["n_used"]), _int(crow["n_used"]))
            diffs.append(_close(row["r2_fractional"], crow["r2_fractional"], tol["r2_abs_strict"]))
            diffs.append(_close(row["r2_difference"], crow["r2_difference"], tol["r2_abs_strict"]))
        return CheckResult("interaction_diagnostics", STATUS_PASS, "Section 6.1/6.2 diagnostics recovered", n_checked=4, max_abs_difference=_max_or_none(diffs), tolerance=tol["r2_abs_strict"])
    except Exception as exc:
        return CheckResult("interaction_diagnostics", STATUS_FAIL, str(exc))


def check_alignment_diagnostics(run_dir: Path, targets: Mapping[str, Any], tol: Mapping[str, float]) -> CheckResult:
    try:
        metrics = _as_mapping(_json(run_dir / "analysis" / "alignment_diagnostics.json"), "alignment diagnostics")
        reported = _as_mapping(targets["section_6_4_reported_diagnostics"], "section_6_4_reported_diagnostics")
        _equal(_int(metrics["n_rows"]), int(targets["current_paper_unfiltered_row_count"]))
        _equal(_int(metrics["n_table3_records"]), len(targets["table3"]["rounded_percent_values"]))
        diffs: list[float] = []
        q_range = reported["Qoff_0_to_plus_0p30_change_percent_points"]
        s_range = reported["sigma_0p02_to_0p08_change_at_Q0_percent_points"]
        diffs.append(_close(metrics["q_change_min_percent_points"], q_range[0], tol["reported_percent_point_abs"]))
        diffs.append(_close(metrics["q_change_max_percent_points"], q_range[1], tol["reported_percent_point_abs"]))
        diffs.append(_close(metrics["sigma_change_min_percent_points"], s_range[0], tol["reported_percent_point_abs"]))
        diffs.append(_close(metrics["sigma_change_max_percent_points"], s_range[1], tol["reported_percent_point_abs"]))
        diffs.append(_close(metrics["max_family_difference_percent_points"], reported["max_family_difference_percent_points"], tol["reported_percent_point_abs"]))
        diffs.append(_close(metrics["restricted_linear_r2_min"], reported["restricted_linear_Rw2_min"], tol["reported_r2_abs"]))
        diffs.append(_close(metrics["restricted_linear_r2_median"], reported["restricted_linear_Rw2_median"], tol["reported_r2_abs"]))
        if "restricted_linear_r2_count" in metrics:
            _equal(_int(metrics["restricted_linear_r2_count"]), 360)
        return CheckResult("alignment_diagnostics", STATUS_PASS, "Section 6.4 diagnostic ranges recovered", n_checked=9, max_abs_difference=_max_or_none(diffs), tolerance=max(tol["reported_percent_point_abs"], tol["reported_r2_abs"]))
    except Exception as exc:
        return CheckResult("alignment_diagnostics", STATUS_FAIL, str(exc))


def check_table3(run_dir: Path, targets: Mapping[str, Any], tol: Mapping[str, float]) -> CheckResult:
    try:
        table3 = _as_mapping(targets["table3"], "table3")
        rows = _csv_rows(run_dir / "analysis" / "table3_selection.csv")
        target_rows = list(table3["rounded_percent_values"])
        _equal(len(rows), len(target_rows))
        actual = {int(float(row["grid_index"])): row for row in rows}
        columns = [
            "p50_Qoff_minus_0p30_percent",
            "p50_Qoff_0p00_percent",
            "p50_Qoff_plus_0p30_percent",
        ]
        diffs: list[float] = []
        for target in target_rows:
            grid = int(target["grid_index"])
            row = actual.get(grid)
            if row is None:
                raise AssertionError(f"missing Table 3 grid {grid}")
            _equal(_int(row["grid_index"]), grid)
            diffs.append(_close(row["mass_msun_h"], target["mass_msun_h"], 1.0))
            diffs.append(_close(row["z_l"], target["z_l"], tol["radius_abs"]))
            diffs.append(_close(row["R_over_r200c"], table3["R_over_r200c"], tol["radius_abs"]))
            diffs.append(_close(row["sigma_d"], table3["sigma_d"], tol["radius_abs"]))
            for column, expected in zip(columns, target["values"]):
                diffs.append(_close(row[column], expected, tol["table3_display_abs"]))
        return CheckResult("table3", STATUS_PASS, "Table 3 rounded values recovered", n_checked=len(target_rows)*3, max_abs_difference=_max_or_none(diffs), tolerance=tol["table3_display_abs"], details={"comparison": "rounded manuscript-display targets"})
    except Exception as exc:
        return CheckResult("table3", STATUS_FAIL, str(exc))


def compare_production_validation_run(run_dir: str | Path, repository_root: str | Path) -> dict[str, Any]:
    """Compare generated production-validation products with compact targets."""
    run_path = Path(run_dir).resolve()
    repo = Path(repository_root).resolve()
    production_targets = _as_mapping(_json(repo / DEFAULT_TARGET_FILES["production_validation"]), "production targets")
    target_paths = _target_paths(repo, production_targets)
    tol = _tolerances(production_targets)
    checks: list[CheckResult] = []
    completeness = check_run_completeness(run_path, production_targets)
    checks.append(completeness)
    if completeness.status == STATUS_PASS:
        checks.append(check_common_domain(run_path, _as_mapping(_json(target_paths["publication_summary"]), "publication targets"), tol))
        checks.append(check_table2(run_path, _as_mapping(_json(target_paths["publication_summary"]), "publication targets"), tol))
        checks.append(check_shape_response(run_path, _as_mapping(_json(target_paths["shape_response"]), "shape targets"), tol))
        checks.append(check_interaction_diagnostics(run_path, _as_mapping(_json(target_paths["interaction_diagnostics"]), "interaction targets"), tol))
        alignment_targets = _as_mapping(_json(target_paths["alignment_reweighting"]), "alignment targets")
        checks.append(check_alignment_diagnostics(run_path, alignment_targets, tol))
        checks.append(check_table3(run_path, alignment_targets, tol))
    else:
        checks.append(CheckResult("target_checks", STATUS_SKIPPED, "run is not complete publication-size validation"))

    analysis_dir = run_path / "analysis"
    analysis_files = file_manifest(sorted(analysis_dir.glob("*")), base=run_path) if analysis_dir.exists() else []
    start_manifest = _json(run_path / "manifest_start.json") if (run_path / "manifest_start.json").is_file() else {}
    start_git = _as_mapping(start_manifest.get("git", {}), "start git") if isinstance(start_manifest, Mapping) else {}
    comparison_git = collect_git_state(repo).as_dict()
    report = {
        "created_utc": utc_now_iso(),
        "run_dir": str(run_path),
        "overall_status": _status_from_checks(checks),
        "checks": [check.as_dict() for check in checks],
        "tolerances": tol,
        "target_files": {key: str(path.relative_to(repo) if path.is_relative_to(repo) else path) for key, path in target_paths.items()},
        "product_generation_git": start_git,
        "comparison_oracle_git": comparison_git,
        "analysis_files": analysis_files,
    }
    return report


def format_validation_summary(report: Mapping[str, Any]) -> str:
    """Format a target-comparison report for human inspection."""
    lines = [f"Production validation: {report.get('overall_status', 'UNKNOWN')}", ""]
    lines.append(f"Run directory: {report.get('run_dir', '')}")
    gen_git = report.get("product_generation_git", {})
    cmp_git = report.get("comparison_oracle_git", {})
    if isinstance(gen_git, Mapping):
        lines.append(f"Product-generation commit: {gen_git.get('commit') or 'unknown'}")
    if isinstance(cmp_git, Mapping):
        lines.append(f"Comparison-oracle commit: {cmp_git.get('commit') or 'unknown'}")
    lines.append("")
    for item in report.get("checks", []):
        if not isinstance(item, Mapping):
            continue
        line = f"{item.get('name')}: {item.get('status')}"
        if item.get("n_checked"):
            line += f"  n={item.get('n_checked')}"
        if item.get("max_abs_difference") is not None:
            line += f"  max|diff|={float(item['max_abs_difference']):.6g}"
        if item.get("tolerance") is not None:
            line += f"  tol={float(item['tolerance']):.6g}"
        message = item.get("message")
        if message:
            line += f"  ({message})"
        lines.append(line)
    lines.append("")
    lines.append("Status meanings: PASS = target checks passed; FAIL = numerical/selection mismatch; INCOMPLETE = required production-size inputs missing; SKIPPED = intentionally not evaluated.")
    return "\n".join(lines) + "\n"


def write_production_validation_report(report: Mapping[str, Any], directory: str | Path) -> None:
    """Write JSON and text comparison reports."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "validation_summary.json", report)
    (out / "validation_summary.txt").write_text(format_validation_summary(report), encoding="utf-8")
