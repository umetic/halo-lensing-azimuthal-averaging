#!/usr/bin/env python3
"""Run the fast validation suite including release-documentation checks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable ({exc})"
    return getattr(module, "__version__", "unknown")


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    src = repository / "src" / "azlens"
    print(f"Interpreter: {sys.executable}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"numpy: {_version('numpy')}")
    print(f"scipy: {_version('scipy')}")
    print(f"pytest: {_version('pytest')}")
    print(f"pyyaml: {_version('yaml')}")
    print(f"Source tree: {src}")
    print("Running fast validation including release-documentation checks...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    out_dir = repository / "outputs" / "release_documentation_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pytest.txt").write_text(result.stdout, encoding="utf-8")
    (out_dir / "environment_and_result.json").write_text(
        json.dumps(
            {
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "returncode": result.returncode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Reports: {out_dir}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
