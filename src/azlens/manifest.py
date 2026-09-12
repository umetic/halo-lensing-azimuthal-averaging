"""Run-manifest and fingerprint utilities for production validation.

These helpers keep validation metadata separate from the scientific modules.  A
production-validation run should identify the source tree, environment,
configuration, and generated files without changing any scientific definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping

DEFAULT_PACKAGE_NAMES = (
    "numpy", "scipy", "pandas", "matplotlib", "astropy", "pyyaml", "pytest", "colossus"
)

DEFAULT_AZLENS_MODULES = (
    "cosmology", "nfw", "projection", "solver_settings", "multipoles", "ring_fields",
    "observables", "controlled_benchmarks", "miscentering", "population", "mode_library",
    "selection", "population_fields", "statistics", "publication_summary", "shape_response",
    "interaction_diagnostics", "weighted_statistics", "alignment", "manifest",
)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with timezone information."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest of a file."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dumps_stable(data: object) -> str:
    """Serialize JSON with deterministic key ordering and a trailing newline."""
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_json(path: str | Path, data: object) -> None:
    """Write stable JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json_dumps_stable(data), encoding="utf-8")


def read_json(path: str | Path) -> object:
    """Read JSON from a path."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    """Return an installed package version, or ``'not installed'``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def package_versions(names: Iterable[str] = DEFAULT_PACKAGE_NAMES) -> dict[str, str]:
    """Return versions for requested distribution names."""
    return {str(name): package_version(str(name)) for name in names}


@dataclass(frozen=True)
class GitState:
    """Minimal Git source-state record."""

    available: bool
    repository_root: str | None
    branch: str | None
    commit: str | None
    dirty: bool | None
    status_short: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "repository_root": self.repository_root,
            "branch": self.branch,
            "commit": self.commit,
            "dirty": self.dirty,
            "status_short": list(self.status_short),
        }


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repository, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def collect_git_state(repository: str | Path) -> GitState:
    """Collect Git metadata without requiring that the path be a repository."""
    root = Path(repository).resolve()
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return GitState(False, None, None, None, None, tuple())
    git_root_proc = _git(root, "rev-parse", "--show-toplevel")
    branch_proc = _git(root, "branch", "--show-current")
    commit_proc = _git(root, "rev-parse", "HEAD")
    status_proc = _git(root, "status", "--short")
    status = tuple(line for line in status_proc.stdout.splitlines() if line.strip())
    return GitState(
        available=True,
        repository_root=git_root_proc.stdout.strip() or str(root),
        branch=branch_proc.stdout.strip() or None,
        commit=commit_proc.stdout.strip() or None,
        dirty=bool(status),
        status_short=status,
    )


def require_clean_git(repository: str | Path, *, allow_dirty: bool = False) -> GitState:
    """Return Git state, raising when a dirty tree is disallowed."""
    state = collect_git_state(repository)
    if state.available and state.dirty and not allow_dirty:
        raise RuntimeError("production validation requires a clean Git tree; use allow_dirty only for development")
    return state


def collect_azlens_module_hashes(
    repository: str | Path,
    modules: Iterable[str] = DEFAULT_AZLENS_MODULES,
) -> dict[str, dict[str, str]]:
    """Hash imported ``azlens`` modules and verify they come from this checkout."""
    root = Path(repository).resolve()
    source_root = (root / "src" / "azlens").resolve()
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    records: dict[str, dict[str, str]] = {}
    for module_name in modules:
        module = importlib.import_module("azlens." + module_name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source_root):
            raise RuntimeError(f"Wrong source imported for azlens.{module_name}: {path}")
        records[module_name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
    return records


def collect_environment_manifest(
    repository: str | Path,
    *,
    config_path: str | Path | None = None,
    module_names: Iterable[str] = DEFAULT_AZLENS_MODULES,
) -> dict[str, object]:
    """Collect environment, Git, config, and source fingerprints."""
    root = Path(repository).resolve()
    config_record: dict[str, object] | None = None
    if config_path is not None:
        cfg = Path(config_path)
        if not cfg.is_absolute():
            cfg = root / cfg
        config_record = {
            "path": str(cfg.relative_to(root) if cfg.is_relative_to(root) else cfg),
            "sha256": sha256_file(cfg),
        }
    return {
        "created_utc": utc_now_iso(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(),
        "git": collect_git_state(root).as_dict(),
        "configuration": config_record,
        "azlens_modules": collect_azlens_module_hashes(root, module_names),
    }


def file_manifest(paths: Iterable[str | Path], *, base: str | Path | None = None) -> list[dict[str, object]]:
    """Return path, byte-size, and SHA-256 records for files."""
    base_path = None if base is None else Path(base).resolve()
    records: list[dict[str, object]] = []
    for item in paths:
        path = Path(item).resolve()
        rel = str(path if base_path is None or not path.is_relative_to(base_path) else path.relative_to(base_path))
        records.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records
