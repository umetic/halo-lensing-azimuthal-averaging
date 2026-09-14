#!/usr/bin/env python3
"""Run deterministic tests through controlled single-halo benchmarks.

From the repository root, after activating the project environment:
    python scripts/check_controlled_benchmarks.py

No packages are installed and no Git operations are performed. Reports go to
outputs/controlled_benchmarks_validation/, which belongs in the repository's
ignore rules.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        "This validation requires Python >= 3.11. Activate the documented environment "
        "or invoke its interpreter explicitly."
    )

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/controlled_benchmarks_validation',
                        help='Report directory, relative to repository root unless absolute.')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root/output
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root/'src'))
    module_names = (
        'cosmology', 'nfw', 'projection', 'solver_settings', 'multipoles',
        'ring_fields', 'observables', 'controlled_benchmarks',
    )
    try:
        dependencies = {name: version(name) for name in ('numpy', 'scipy', 'pytest')}
        origins = {}
        for name in module_names:
            module = importlib.import_module('azlens.'+name)
            path = Path(module.__file__).resolve()
            if not path.is_relative_to((root/'src/azlens').resolve()):
                raise RuntimeError(f'Wrong source imported for azlens.{name}: {path}')
            origins[name] = str(path)
    except (ImportError, RuntimeError) as exc:
        print(f'Preflight failed: {exc}', file=sys.stderr)
        return 2

    metadata = {
        'validation_layer': 'controlled-benchmarks-v0.1.0',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'executable': sys.executable,
        'prefix': sys.prefix,
        'dependencies': dependencies,
        'module_paths': origins,
        'module_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in origins.items()},
    }
    print('Interpreter:', sys.executable)
    print('Python:', sys.version.split()[0])
    for name, value in dependencies.items():
        print(f'{name}: {value}')
    print('Source tree:', root/'src/azlens')
    print('Running deterministic-core tests through controlled benchmarks...')
    sys.stdout.flush()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    command = [
        sys.executable, '-m', 'pytest', '-q', '-ra',
        'tests/test_cosmology.py', 'tests/test_nfw.py', 'tests/test_projection.py',
        'tests/test_solver_settings.py', 'tests/test_multipoles.py',
        'tests/test_ring_fields.py', 'tests/test_observables.py',
        'tests/test_controlled_benchmarks.py',
    ]
    result = subprocess.run(command, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    (output/'pytest.txt').write_text(result.stdout, encoding='utf-8')

    if result.returncode == 0:
        from azlens.controlled_benchmarks import (
            compact_validation_summary, figure1_profile_table,
            figure2_orientation_table, figure2_power_table,
        )
        summary = compact_validation_summary()
        (output/'controlled_benchmark_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        _write_csv(output/'figure1_profile_table.csv', figure1_profile_table())
        _write_csv(output/'figure2_orientation_table.csv', figure2_orientation_table())
        _write_csv(output/'figure2_power_table.csv', figure2_power_table())
        metadata['generated_products'] = [
            'controlled_benchmark_summary.json',
            'figure1_profile_table.csv',
            'figure2_orientation_table.csv',
            'figure2_power_table.csv',
        ]

    metadata.update(returncode=result.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment_and_result.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Reports:', output)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
