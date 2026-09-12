#!/usr/bin/env python3
"""Run tests through the unweighted publication-summary layer.

From the repository root, after activating the project environment:
    python scripts/check_publication_summary.py

This validation does not run the full population production. It tests summary
statistics, common-domain identification, and Table 2 row-selection logic.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        "This increment requires Python >= 3.11. Activate the project environment "
        "or invoke its interpreter explicitly; do not use an old python3 alias."
    )

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
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
    parser.add_argument('--output-dir', default='outputs/publication_summary_validation',
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
        'statistics', 'publication_summary',
    )
    try:
        dependencies = {name: _package_version(name) for name in ('numpy', 'scipy', 'pytest', 'colossus')}
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
        'increment': 'publication-summary-v0.1.0',
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
    print('Running tests through publication-summary and Table 2 logic...')
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
        'tests/test_publication_summary.py',
    ]
    result = subprocess.run(command, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    (output/'pytest.txt').write_text(result.stdout, encoding='utf-8')

    if result.returncode == 0:
        from azlens.publication_summary import build_publication_summary_from_rows
        # Reuse the compact synthetic construction from the test module only to
        # create a report artifact; production analysis will consume generated
        # population-field rows instead.
        test_path = root/'tests/test_publication_summary.py'
        spec = importlib.util.spec_from_file_location('_azlens_publication_summary_test_helpers', test_path)
        if spec is not None and spec.loader is not None:
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            summary = build_publication_summary_from_rows(helper._synthetic_rows())
            (output/'publication_summary_smoke.json').write_text(
                json.dumps(summary.as_dict(), indent=2, sort_keys=True)+'\n', encoding='utf-8'
            )
            metadata['generated_products'] = ['publication_summary_smoke.json']

    metadata.update(returncode=result.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment_and_result.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Reports:', output)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
