"""Production-scale validation workflow orchestration.

This module wires together already validated scientific modules into a resumable
workflow.  It is intentionally an orchestration layer: it must not redefine the
lensing calculation, selection rules, or paper-level statistics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .alignment import (
    alignment_diagnostics,
    alignment_rows_from_products,
    table3_records_from_alignment_rows,
    write_alignment_rows_csv,
    write_table3_csv,
)
from .interaction_diagnostics import (
    compact_interaction_metrics,
    conditioning_rows_from_products,
    hybrid_fit_rows_from_products,
    write_conditioning_rows_csv,
    write_hybrid_rows_csv,
)
from .manifest import (
    collect_environment_manifest,
    file_manifest,
    read_json,
    require_clean_git,
    sha256_file,
    utc_now_iso,
    write_json,
)
from .miscentering import OffsetRealization, draw_offset_realization
from .mode_library import ModeLibrary
from .population import (
    DEFAULT_N_HALO,
    HaloPopulationRealization,
    PopulationGrid,
    PublicationRealization,
    generate_halo_population,
    publication_grids,
    sample_halo_population_from_state,
)
from .population_fields import (
    PopulationFieldConfig,
    PopulationFieldProducts,
    compact_field_validation_metrics,
    evaluate_grid_population_fields,
)
from .publication_summary import (
    build_publication_summary_from_products,
    radial_summary_rows_from_products,
    table2_records_to_matrix,
    write_radial_summary_csv,
)
from .selection import publication_radial_grid
from .shape_response import (
    centered_shape_response_rows_from_products,
    compact_shape_response_metrics,
    write_shape_response_csv,
)
from .solver_settings import POPULATION_MODE_SETTINGS, SolverSettings

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_PRODUCTION_VALIDATION_CONFIG: dict[str, object] = {
    "run_id": None,
    "output_root": "outputs/production_validation",
    "validation": {"require_clean_git": True, "compare_targets": True},
    "mode_library": {
        "profile": "population",
        "use_cache": True,
        "rebuild": False,
        "smoke_settings": {
            "name": "production_validation_smoke",
            "q_min": 0.18,
            "q_max": 1.0,
            "n_q": 5,
            "x_min": 1.0e-4,
            "x_max": 80.0,
            "n_x": 24,
            "x_solver_min": 1.0e-5,
            "x_solver_max": 1.0e3,
            "n_radial_solver": 96,
            "n_phi_solver": 64,
            "m_max": 8,
            "radial_interpolation": "x",
        },
    },
    "population": {"n_halos": DEFAULT_N_HALO, "grids": [0, 1, 2, 3, 4, 5], "state_source": "colossus"},
    "fields": {"n_phi": 256, "chunk_size": 64, "source_redshifts": [1.0, 2.0], "sigma_d": [0.0, 0.02, 0.05, 0.08], "radial_grid": "publication"},
    "analysis": {"run_table2": True, "run_shape_response": True, "run_interactions": True, "run_alignment": True},
}

STAGES = ("preflight", "mode-library", "realization", "fields", "analysis", "compare")


def deep_update(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    """Return a recursive dictionary update without modifying inputs."""
    result = {str(k): v for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[str(key)] = deep_update(result[str(key)], value)  # type: ignore[arg-type]
        else:
            result[str(key)] = value
    return result


def load_workflow_config(path: str | Path | None = None) -> dict[str, object]:
    """Load workflow configuration, merging it with safe defaults.

    YAML is supported through PyYAML when available; JSON is also accepted.  The
    file is workflow-level configuration and must not redefine validated physics
    silently.
    """
    config = dict(DEFAULT_PRODUCTION_VALIDATION_CONFIG)
    if path is None:
        return normalize_workflow_config(config)
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML workflow configuration") from exc
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text) if text.strip() else {}
    if not isinstance(loaded, Mapping):
        raise ValueError("workflow configuration must be a mapping")
    return normalize_workflow_config(deep_update(DEFAULT_PRODUCTION_VALIDATION_CONFIG, loaded))


def normalize_workflow_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the workflow configuration at a shallow level."""
    cfg = deep_update(DEFAULT_PRODUCTION_VALIDATION_CONFIG, config)
    pop = cfg.get("population")
    fields = cfg.get("fields")
    mode = cfg.get("mode_library")
    validation = cfg.get("validation")
    if not isinstance(pop, Mapping) or not isinstance(fields, Mapping) or not isinstance(mode, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("configuration sections population, fields, mode_library, validation are required")
    if int(pop["n_halos"]) < 1:
        raise ValueError("population.n_halos must be positive")
    grids = [int(g) for g in pop["grids"]]  # type: ignore[index]
    if not grids or any(g < 0 or g > 5 for g in grids):
        raise ValueError("population.grids must contain publication grid indices 0..5")
    pop = dict(pop)
    pop["grids"] = grids
    pop["n_halos"] = int(pop["n_halos"])
    fields = dict(fields)
    fields["n_phi"] = int(fields["n_phi"])
    fields["chunk_size"] = int(fields["chunk_size"])
    if fields["n_phi"] < 1 or fields["chunk_size"] < 1:
        raise ValueError("fields.n_phi and fields.chunk_size must be positive")
    fields["source_redshifts"] = [float(z) for z in fields["source_redshifts"]]  # type: ignore[index]
    fields["sigma_d"] = [float(s) for s in fields["sigma_d"]]  # type: ignore[index]
    cfg["population"] = pop
    cfg["fields"] = fields
    return cfg


def default_run_id(now: datetime | None = None) -> str:
    """UTC timestamp suitable for an output run directory."""
    dt = now or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class ProductionRunContext:
    """Resolved paths and normalized configuration for one validation run."""

    repository_root: Path
    run_dir: Path
    config: dict[str, object]
    config_path: Path | None = None
    allow_dirty: bool = False

    @property
    def stage_dir(self) -> Path:
        return self.run_dir / "stages"


def make_run_context(
    repository_root: str | Path,
    *,
    config_path: str | Path | None = None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    allow_dirty: bool = False,
) -> ProductionRunContext:
    """Resolve configuration and create the run directory."""
    root = Path(repository_root).resolve()
    cfg_path = None if config_path is None else Path(config_path)
    if cfg_path is not None and not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_workflow_config(cfg_path)
    if run_id is not None:
        cfg["run_id"] = str(run_id)
    rid = str(cfg.get("run_id") or default_run_id())
    out_root = Path(output_root) if output_root is not None else Path(str(cfg["output_root"]))
    if not out_root.is_absolute():
        out_root = root / out_root
    run_dir = out_root / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stages").mkdir(parents=True, exist_ok=True)
    return ProductionRunContext(root, run_dir, cfg, cfg_path, bool(allow_dirty))


def _write_stage(context: ProductionRunContext, stage: str, status: str, **payload: object) -> dict[str, object]:
    record = {"stage": stage, "status": status, "updated_utc": utc_now_iso(), **payload}
    write_json(context.stage_dir / f"{stage}.json", record)
    return record


def _mode_settings_from_config(config: Mapping[str, object]) -> SolverSettings:
    mode = config["mode_library"]  # type: ignore[index]
    if not isinstance(mode, Mapping):
        raise ValueError("mode_library configuration must be a mapping")
    profile = str(mode.get("profile", "population"))
    if profile == "population":
        return POPULATION_MODE_SETTINGS
    if profile == "smoke":
        setting = mode.get("smoke_settings")
        if not isinstance(setting, Mapping):
            raise ValueError("mode_library.smoke_settings is required for profile='smoke'")
        return SolverSettings(
            name=str(setting["name"]), q_min=float(setting["q_min"]), q_max=float(setting["q_max"]), n_q=int(setting["n_q"]),
            x_min=float(setting["x_min"]), x_max=float(setting["x_max"]), n_x=int(setting["n_x"]),
            x_solver_min=float(setting["x_solver_min"]), x_solver_max=float(setting["x_solver_max"]),
            n_radial_solver=int(setting["n_radial_solver"]), n_phi_solver=int(setting["n_phi_solver"]),
            m_max=int(setting["m_max"]), radial_interpolation=str(setting["radial_interpolation"]),
        )
    raise ValueError("mode_library.profile must be 'population' or 'smoke'")


def run_preflight(context: ProductionRunContext) -> dict[str, object]:
    """Record starting manifest and enforce clean-Git policy when requested."""
    validation = context.config["validation"]  # type: ignore[index]
    require_clean = bool(validation.get("require_clean_git", True)) if isinstance(validation, Mapping) else True
    started = time.perf_counter()
    git_state = require_clean_git(context.repository_root, allow_dirty=(context.allow_dirty or not require_clean))
    manifest = collect_environment_manifest(context.repository_root, config_path=context.config_path)
    manifest["git"] = git_state.as_dict()
    manifest["run_dir"] = str(context.run_dir)
    write_json(context.run_dir / "manifest_start.json", manifest)
    return _write_stage(context, "preflight", "complete", duration_seconds=time.perf_counter()-started, manifest="manifest_start.json")


def run_mode_library_stage(context: ProductionRunContext) -> dict[str, object]:
    """Build or reuse the configured population mode library."""
    started = time.perf_counter()
    out = context.run_dir / "mode_library"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "population_mode_library.npz"
    mode_cfg = context.config["mode_library"]  # type: ignore[index]
    if not isinstance(mode_cfg, Mapping):
        raise ValueError("mode_library configuration must be a mapping")
    rebuild = bool(mode_cfg.get("rebuild", False))
    use_cache = bool(mode_cfg.get("use_cache", True))
    if path.exists() and use_cache and not rebuild:
        library = ModeLibrary.load_npz(path)
        action = "loaded_existing"
    else:
        library = ModeLibrary.build(_mode_settings_from_config(context.config))
        library.save_npz(path, extra_metadata={"stage": "production_validation_mode_library"})
        action = "built"
    manifest = {
        "action": action,
        "path": str(path.relative_to(context.run_dir)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "q_nodes": int(library.q_grid.size),
        "x_nodes": int(library.x_grid.size),
        "m_max": int(library.m_max),
        "radial_interpolation": library.radial_interpolation,
    }
    write_json(out / "mode_library_manifest.json", manifest)
    return _write_stage(context, "mode-library", "complete", duration_seconds=time.perf_counter()-started, outputs=[manifest])


def _population_targets_state(repository_root: Path, grid_index: int) -> Mapping[str, object]:
    targets = read_json(repository_root / "reference" / "population_realization" / "population_targets.json")
    if not isinstance(targets, Mapping):
        raise ValueError("invalid population target file")
    grids = targets["grids"]  # type: ignore[index]
    return grids[int(grid_index)]  # type: ignore[index]


def _generate_halo_for_workflow(grid: PopulationGrid, *, size: int, state_source: str, repository_root: Path) -> HaloPopulationRealization:
    if state_source == "colossus":
        return generate_halo_population(grid, size=size)
    if state_source == "reference":
        state = _population_targets_state(repository_root, grid.grid_index)
        return sample_halo_population_from_state(
            grid,
            size=size,
            c200c_median=float(state["c200c_median_inferred"]),  # type: ignore[index]
            shape_peak_height=float(state["shape_peak_height_inferred"]),  # type: ignore[index]
            mvir_shape_msun_h=float(state.get("mvir_shape_msun_h_inferred", np.nan)),  # type: ignore[attr-defined]
            colossus_version="reference-target-state",
        )
    raise ValueError("population.state_source must be 'colossus' or 'reference'")


def save_halo_population_npz(halo: HaloPopulationRealization, path: str | Path) -> None:
    """Save a halo realization for resumable validation stages."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "grid": asdict(halo.grid),
        "c200c_median": halo.c200c_median,
        "shape_peak_height": halo.shape_peak_height,
        "mvir_shape_msun_h": halo.mvir_shape_msun_h,
        "colossus_version": halo.colossus_version,
        "r200c_mpc_h": halo.r200c_mpc_h,
        "minor_axis_proposal_count": halo.minor_axis_proposal_count,
        "minor_axis_rejection_rounds": halo.minor_axis_rejection_rounds,
    }
    np.savez_compressed(
        target,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        c200c=halo.c200c,
        p_axis_ratio=halo.p_axis_ratio,
        s_axis_ratio=halo.s_axis_ratio,
        triaxiality=halo.triaxiality,
        los_cos_theta=halo.los_cos_theta,
        los_phi=halo.los_phi,
        los_vectors=halo.los_vectors,
        q_perp=halo.q_perp,
        b_los=halo.b_los,
        projected_scale_factor=halo.projected_scale_factor,
        rs_mpc_h=halo.rs_mpc_h,
        rs_perp_mpc_h=halo.rs_perp_mpc_h,
        kappa_s_infinity=halo.kappa_s_infinity,
        kappa_s_perp_infinity=halo.kappa_s_perp_infinity,
        epsilon_perp=halo.epsilon_perp,
    )


def load_halo_population_npz(path: str | Path) -> HaloPopulationRealization:
    """Load a halo realization written by :func:`save_halo_population_npz`."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        grid_meta = metadata["grid"]
        grid = PopulationGrid(int(grid_meta["grid_index"]), float(grid_meta["mass_msun_h"]), float(grid_meta["z_l"]), int(grid_meta["population_seed"]))
        return HaloPopulationRealization(
            grid=grid,
            c200c_median=float(metadata["c200c_median"]),
            shape_peak_height=float(metadata["shape_peak_height"]),
            mvir_shape_msun_h=float(metadata["mvir_shape_msun_h"]),
            colossus_version=str(metadata["colossus_version"]),
            c200c=np.asarray(data["c200c"], dtype=np.float64),
            p_axis_ratio=np.asarray(data["p_axis_ratio"], dtype=np.float64),
            s_axis_ratio=np.asarray(data["s_axis_ratio"], dtype=np.float64),
            triaxiality=np.asarray(data["triaxiality"], dtype=np.float64),
            los_cos_theta=np.asarray(data["los_cos_theta"], dtype=np.float64),
            los_phi=np.asarray(data["los_phi"], dtype=np.float64),
            los_vectors=np.asarray(data["los_vectors"], dtype=np.float64),
            q_perp=np.asarray(data["q_perp"], dtype=np.float64),
            b_los=np.asarray(data["b_los"], dtype=np.float64),
            projected_scale_factor=np.asarray(data["projected_scale_factor"], dtype=np.float64),
            r200c_mpc_h=float(metadata["r200c_mpc_h"]),
            rs_mpc_h=np.asarray(data["rs_mpc_h"], dtype=np.float64),
            rs_perp_mpc_h=np.asarray(data["rs_perp_mpc_h"], dtype=np.float64),
            kappa_s_infinity=np.asarray(data["kappa_s_infinity"], dtype=np.float64),
            kappa_s_perp_infinity=np.asarray(data["kappa_s_perp_infinity"], dtype=np.float64),
            epsilon_perp=np.asarray(data["epsilon_perp"], dtype=np.float64),
            minor_axis_proposal_count=int(metadata["minor_axis_proposal_count"]),
            minor_axis_rejection_rounds=int(metadata["minor_axis_rejection_rounds"]),
        )


def save_offset_npz(offset: OffsetRealization, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"grid_index": offset.grid_index, "base_seed": offset.base_seed, "bit_generator_name": offset.bit_generator_name}
    np.savez_compressed(target, metadata_json=np.array(json.dumps(metadata, sort_keys=True)), rayleigh_unit=offset.rayleigh_unit, phi_offset=offset.phi_offset)


def load_offset_npz(path: str | Path) -> OffsetRealization:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return OffsetRealization(
            grid_index=int(metadata["grid_index"]), base_seed=int(metadata["base_seed"]),
            rayleigh_unit=np.asarray(data["rayleigh_unit"], dtype=np.float64),
            phi_offset=np.asarray(data["phi_offset"], dtype=np.float64),
        )


def run_realization_stage(context: ProductionRunContext) -> dict[str, object]:
    """Generate and save halo/offset catalogs for configured grids."""
    started = time.perf_counter()
    out = context.run_dir / "realization"
    out.mkdir(parents=True, exist_ok=True)
    pop_cfg = context.config["population"]  # type: ignore[index]
    if not isinstance(pop_cfg, Mapping):
        raise ValueError("population configuration must be a mapping")
    n_halo = int(pop_cfg["n_halos"])
    state_source = str(pop_cfg.get("state_source", "colossus"))
    grids = publication_grids()
    records = []
    for idx in [int(g) for g in pop_cfg["grids"]]:  # type: ignore[index]
        grid = grids[idx]
        halo = _generate_halo_for_workflow(grid, size=n_halo, state_source=state_source, repository_root=context.repository_root)
        offsets = draw_offset_realization(size=n_halo, grid_index=idx)
        halo_path = out / f"grid_{idx:02d}_halo_realization.npz"
        off_path = out / f"grid_{idx:02d}_offset_realization.npz"
        save_halo_population_npz(halo, halo_path)
        save_offset_npz(offsets, off_path)
        records.append({
            "grid_index": idx,
            "halo_path": str(halo_path.relative_to(context.run_dir)),
            "offset_path": str(off_path.relative_to(context.run_dir)),
            "n_halo": n_halo,
            "halo_sha256": sha256_file(halo_path),
            "offset_sha256": sha256_file(off_path),
            "q_perp_min": float(np.min(halo.q_perp)),
            "q_perp_max": float(np.max(halo.q_perp)),
        })
    manifest = {"state_source": state_source, "records": records}
    write_json(out / "realization_manifest.json", manifest)
    return _write_stage(context, "realization", "complete", duration_seconds=time.perf_counter()-started, outputs=records)


def _field_config_from_workflow(config: Mapping[str, object]) -> PopulationFieldConfig:
    fields = config["fields"]  # type: ignore[index]
    if not isinstance(fields, Mapping):
        raise ValueError("fields configuration must be a mapping")
    radial_spec = fields.get("radial_grid", "publication")
    if radial_spec == "publication":
        radial = publication_radial_grid()
    else:
        radial = np.asarray(radial_spec, dtype=np.float64)
    return PopulationFieldConfig(
        radial_grid=radial,
        source_redshifts=tuple(float(z) for z in fields["source_redshifts"]),  # type: ignore[index]
        sigma_d=tuple(float(s) for s in fields["sigma_d"]),  # type: ignore[index]
        n_phi=int(fields["n_phi"]),
        chunk_size=int(fields["chunk_size"]),
    )


def save_population_field_product(product: PopulationFieldProducts, directory: str | Path) -> None:
    """Save one grid's population-field product for downstream stages."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "grid_index": product.grid_index,
        "mass_msun_h": product.mass_msun_h,
        "z_l": product.z_l,
        "source_weights": product.source_weights,
    }
    arrays: dict[str, NDArray[np.generic]] = {
        "radial_grid": np.asarray(product.radial_grid),
        "sigma_d": np.asarray(product.sigma_d, dtype=np.float64),
        "source_redshifts": np.asarray(product.source_redshifts, dtype=np.float64),
        "key_indices": np.asarray(product.key_indices, dtype=np.int64),
        "lambda_min_infinity": np.asarray(product.lambda_min_infinity, dtype=np.float64),
        "safe_local": np.asarray(product.safe_local, dtype=bool),
        "safe_outer": np.asarray(product.safe_outer, dtype=bool),
        "represented_fraction_outer": np.asarray(product.represented_fraction_outer, dtype=np.float64),
    }
    for key, value in product.key_samples.items():
        arrays[f"key__{key}"] = np.asarray(value)  # type: ignore[assignment]
    np.savez_compressed(out / "product_arrays.npz", metadata_json=np.array(json.dumps(metadata, sort_keys=True)), **arrays)
    write_json(out / "radial_summary_rows.json", list(product.radial_summary_rows))
    write_json(out / "product_metrics.json", compact_field_validation_metrics(product))


def load_population_field_product(directory: str | Path) -> PopulationFieldProducts:
    """Load one grid's field product saved by :func:`save_population_field_product`."""
    source = Path(directory)
    rows = read_json(source / "radial_summary_rows.json")
    with np.load(source / "product_arrays.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        key_samples = {name[5:]: np.asarray(data[name]) for name in data.files if name.startswith("key__")}
        return PopulationFieldProducts(
            grid_index=int(metadata["grid_index"]),
            mass_msun_h=float(metadata["mass_msun_h"]),
            z_l=float(metadata["z_l"]),
            radial_grid=np.asarray(data["radial_grid"], dtype=np.float64),
            sigma_d=tuple(float(x) for x in np.asarray(data["sigma_d"], dtype=np.float64)),
            source_redshifts=tuple(float(x) for x in np.asarray(data["source_redshifts"], dtype=np.float64)),
            source_weights={str(k): float(v) for k, v in metadata["source_weights"].items()},
            key_indices=np.asarray(data["key_indices"], dtype=np.int64),
            lambda_min_infinity=np.asarray(data["lambda_min_infinity"], dtype=np.float64),
            safe_local=np.asarray(data["safe_local"], dtype=bool),
            safe_outer=np.asarray(data["safe_outer"], dtype=bool),
            represented_fraction_outer=np.asarray(data["represented_fraction_outer"], dtype=np.float64),
            radial_summary_rows=tuple(rows),  # type: ignore[arg-type]
            key_samples=key_samples,
        )


def run_fields_stage(context: ProductionRunContext, *, grid_indices: Sequence[int] | None = None) -> dict[str, object]:
    """Evaluate population fields for saved realizations."""
    started = time.perf_counter()
    out = context.run_dir / "population_fields"
    out.mkdir(parents=True, exist_ok=True)
    lib_path = context.run_dir / "mode_library" / "population_mode_library.npz"
    if not lib_path.is_file():
        raise FileNotFoundError("mode library stage must complete before fields")
    library = ModeLibrary.load_npz(lib_path)
    pop_cfg = context.config["population"]  # type: ignore[index]
    configured = [int(g) for g in pop_cfg["grids"]] if isinstance(pop_cfg, Mapping) else []  # type: ignore[index]
    selected = list(grid_indices) if grid_indices is not None else configured
    field_config = _field_config_from_workflow(context.config)
    records = []
    for idx in selected:
        halo = load_halo_population_npz(context.run_dir / "realization" / f"grid_{idx:02d}_halo_realization.npz")
        offsets = load_offset_npz(context.run_dir / "realization" / f"grid_{idx:02d}_offset_realization.npz")
        product = evaluate_grid_population_fields(halo, offsets, library, config=field_config)
        grid_dir = out / f"grid_{idx:02d}"
        save_population_field_product(product, grid_dir)
        records.append({"grid_index": idx, "directory": str(grid_dir.relative_to(context.run_dir)), "metrics": compact_field_validation_metrics(product)})
    write_json(out / "population_fields_manifest.json", {"records": records})
    return _write_stage(context, "fields", "complete", duration_seconds=time.perf_counter()-started, outputs=records)


def _load_products_from_run(context: ProductionRunContext) -> tuple[PopulationFieldProducts, ...]:
    pop_cfg = context.config["population"]  # type: ignore[index]
    grid_indices = [int(g) for g in pop_cfg["grids"]] if isinstance(pop_cfg, Mapping) else []  # type: ignore[index]
    products = []
    for idx in grid_indices:
        products.append(load_population_field_product(context.run_dir / "population_fields" / f"grid_{idx:02d}"))
    return tuple(products)


def _write_table2_csv(records: Iterable[object], path: Path) -> None:
    rows = [record.as_dict() for record in records]  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis_stage(context: ProductionRunContext) -> dict[str, object]:
    """Run paper-analysis layers on generated population-field products."""
    started = time.perf_counter()
    out = context.run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    products = _load_products_from_run(context)
    analysis_cfg = context.config.get("analysis", {})
    if not isinstance(analysis_cfg, Mapping):
        analysis_cfg = {}
    generated = []
    metrics: dict[str, object] = {"n_products": len(products)}

    if bool(analysis_cfg.get("run_table2", True)):
        summary = build_publication_summary_from_products(products)
        rows = radial_summary_rows_from_products(products)
        write_radial_summary_csv(rows, out / "radial_summary_all_grids.csv")
        write_json(out / "common_domain.json", summary.common_domain.as_dict())
        _write_table2_csv(summary.table2_records, out / "table2_selection.csv")
        write_json(out / "table2_selection.json", [r.as_dict() for r in summary.table2_records])
        metrics["table2"] = table2_records_to_matrix(summary.table2_records)
        generated.extend(["radial_summary_all_grids.csv", "common_domain.json", "table2_selection.csv", "table2_selection.json"])

    if bool(analysis_cfg.get("run_shape_response", True)):
        shape_rows = centered_shape_response_rows_from_products(products)
        write_shape_response_csv(shape_rows, out / "centered_shape_response.csv")
        write_json(out / "centered_shape_response_metrics.json", compact_shape_response_metrics(shape_rows))
        generated.extend(["centered_shape_response.csv", "centered_shape_response_metrics.json"])

    if bool(analysis_cfg.get("run_interactions", True)):
        hybrid = hybrid_fit_rows_from_products(products)
        conditioning = conditioning_rows_from_products(products)
        write_hybrid_rows_csv(hybrid, out / "hybrid_interaction_diagnostics.csv")
        write_conditioning_rows_csv(conditioning, out / "conditioning_diagnostics.csv")
        write_json(out / "interaction_diagnostics_metrics.json", compact_interaction_metrics(hybrid, conditioning))
        generated.extend(["hybrid_interaction_diagnostics.csv", "conditioning_diagnostics.csv", "interaction_diagnostics_metrics.json"])

    if bool(analysis_cfg.get("run_alignment", True)):
        rows = alignment_rows_from_products(products)
        table3 = table3_records_from_alignment_rows(rows)
        write_alignment_rows_csv(rows, out / "alignment_reweighted_summary.csv")
        write_table3_csv(table3, out / "table3_selection.csv")
        write_json(out / "alignment_diagnostics.json", alignment_diagnostics(rows))
        generated.extend(["alignment_reweighted_summary.csv", "table3_selection.csv", "alignment_diagnostics.json"])

    files = [out / name for name in generated if (out / name).exists()]
    write_json(out / "analysis_manifest.json", {"generated": generated, "metrics": metrics, "files": file_manifest(files, base=context.run_dir)})
    return _write_stage(context, "analysis", "complete", duration_seconds=time.perf_counter()-started, outputs=generated, metrics=metrics)


def run_compare_stage(context: ProductionRunContext) -> dict[str, object]:
    """Create a comparison skeleton and fingerprint generated analysis products."""
    started = time.perf_counter()
    out = context.run_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    analysis_dir = context.run_dir / "analysis"
    files = sorted(path for path in analysis_dir.glob("*") if path.is_file()) if analysis_dir.exists() else []
    report = {
        "status": "fingerprinted",
        "note": "Full numerical target comparisons are meaningful only after a publication-size run is complete.",
        "analysis_files": file_manifest(files, base=context.run_dir),
    }
    write_json(out / "validation_summary.json", report)
    (out / "validation_summary.txt").write_text(
        "Production-validation comparison fingerprinted generated analysis products.\n"
        "Run publication-size stages before interpreting this as full scientific reproduction.\n",
        encoding="utf-8",
    )
    return _write_stage(context, "compare", "complete", duration_seconds=time.perf_counter()-started, outputs=["comparison/validation_summary.json", "comparison/validation_summary.txt"])


def run_stage(context: ProductionRunContext, stage: str, *, grid_indices: Sequence[int] | None = None) -> dict[str, object]:
    """Run one named validation stage."""
    if stage == "preflight":
        return run_preflight(context)
    if stage == "mode-library":
        return run_mode_library_stage(context)
    if stage == "realization":
        return run_realization_stage(context)
    if stage == "fields":
        return run_fields_stage(context, grid_indices=grid_indices)
    if stage == "analysis":
        return run_analysis_stage(context)
    if stage == "compare":
        return run_compare_stage(context)
    raise ValueError(f"unknown stage {stage!r}")


def run_stages(context: ProductionRunContext, stages: Sequence[str], *, grid_indices: Sequence[int] | None = None) -> tuple[dict[str, object], ...]:
    """Run stages sequentially and write a final manifest."""
    records = []
    for stage in stages:
        records.append(run_stage(context, stage, grid_indices=grid_indices if stage == "fields" else None))
    if "compare" in stages or stages == tuple(STAGES) or list(stages) == list(STAGES):
        manifest = collect_environment_manifest(context.repository_root, config_path=context.config_path)
        produced = sorted(path for path in context.run_dir.rglob("*") if path.is_file())
        manifest["run_dir"] = str(context.run_dir)
        manifest["outputs"] = file_manifest(produced, base=context.run_dir)
        write_json(context.run_dir / "manifest_final.json", manifest)
    return tuple(records)


def smoke_config(*, n_halos: int = 12) -> dict[str, object]:
    """Return a small no-Colossus workflow configuration for orchestration tests."""
    return normalize_workflow_config({
        "run_id": "smoke",
        "output_root": "outputs/production_validation",
        "validation": {"require_clean_git": False, "compare_targets": False},
        "mode_library": {"profile": "smoke", "use_cache": False, "rebuild": True},
        "population": {"n_halos": int(n_halos), "grids": [0], "state_source": "reference"},
        "fields": {
            "n_phi": 32,
            "chunk_size": 5,
            "source_redshifts": [1.0, 2.0],
            "sigma_d": [0.0, 0.05],
            "radial_grid": [0.2091279105182546, 0.2924408238028535, 0.5113850166642591, 1.0],
        },
        "analysis": {"run_table2": False, "run_shape_response": True, "run_interactions": False, "run_alignment": False},
    })
