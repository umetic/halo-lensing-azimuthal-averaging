#!/usr/bin/env python3
"""Run tests through the production-validation workflow layer.

This checker is intentionally fast.  It runs unit/orchestration tests and a tiny
smoke workflow; it does not launch the publication-size N=10000 production run.
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
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/full_production_validation_check',
                        help='Report directory, relative to repository root unless absolute.')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root/output
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root/'src'))
    module_names = (
        'cosmology', 'nfw', 'projection', 'solver_settings', 'multipoles',
        'ring_fields', 'observables', 'controlled_benchmarks', 'miscentering',
        'population', 'mode_library', 'selection', 'population_fields',
        'statistics', 'publication_summary', 'shape_response',
        'interaction_diagnostics', 'weighted_statistics', 'alignment',
        'manifest', 'target_comparison', 'production_validation',
    )
    try:
        dependencies = {name: _package_version(name) for name in ('numpy', 'scipy', 'pytest', 'colossus', 'pyyaml')}
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
        'validation_layer': 'target-aware-production-comparison-v0.1.0',
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
    print('Running tests through production-validation workflow...')
    sys.stdout.flush()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    command = [
        sys.executable, '-m', 'pytest', '-q', '-ra',
        'tests/test_cosmology.py', 'tests/test_nfw.py', 'tests/test_projection.py',
        'tests/test_solver_settings.py', 'tests/test_multipoles.py',
        'tests/test_ring_fields.py', 'tests/test_observables.py',
        'tests/test_controlled_benchmarks.py', 'tests/test_miscentering.py',
        'tests/test_population.py', 'tests/test_mode_library.py', 'tests/test_selection.py',
        'tests/test_population_fields.py', 'tests/test_statistics.py',
        'tests/test_publication_summary.py', 'tests/test_shape_response.py',
        'tests/test_interaction_diagnostics.py', 'tests/test_weighted_statistics.py',
        'tests/test_alignment.py', 'tests/test_manifest.py', 'tests/test_target_comparison.py', 'tests/test_production_validation.py',
    ]
    result = subprocess.run(command, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    (output/'pytest.txt').write_text(result.stdout, encoding='utf-8')

    if result.returncode == 0:
        smoke_dir = output/'smoke_script_run'
        smoke_cmd = [
            sys.executable, 'scripts/run_full_production_validation.py',
            '--smoke', '--stage', 'preflight', '--stage', 'mode-library',
            '--stage', 'realization', '--stage', 'fields', '--stage', 'analysis',
            '--stage', 'compare', '--output-root', str(smoke_dir), '--run-id', 'smoke',
        ]
        smoke = subprocess.run(smoke_cmd, cwd=root, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(smoke.stdout, end='')
        (output/'smoke_workflow.txt').write_text(smoke.stdout, encoding='utf-8')
        metadata['smoke_returncode'] = smoke.returncode
        if smoke.returncode != 0:
            result_returncode = smoke.returncode
        else:
            result_returncode = 0
    else:
        result_returncode = result.returncode

    metadata.update(returncode=result_returncode, completed_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment_and_result.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Reports:', output)
    return result_returncode


if __name__ == '__main__':
    raise SystemExit(main())
