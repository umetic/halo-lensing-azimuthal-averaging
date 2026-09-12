#!/usr/bin/env python3
"""Run deterministic tests through the seeded population-realization layer.

From the repository root, after activating the project environment:
    python scripts/check_population_realization.py

No packages are installed and no Git operations are performed. Reports go to
outputs/population_realization_validation/.
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
    parser.add_argument('--output-dir', default='outputs/population_realization_validation',
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
        'miscentering', 'population',
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
        'increment': 'population-realization-v0.1.0',
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
    print('Running tests through seeded population and miscentering realization...')
    sys.stdout.flush()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    command = [
        sys.executable, '-m', 'pytest', '-q', '-ra',
        'tests/test_cosmology.py', 'tests/test_nfw.py', 'tests/test_projection.py',
        'tests/test_solver_settings.py', 'tests/test_multipoles.py',
        'tests/test_ring_fields.py', 'tests/test_observables.py',
        'tests/test_controlled_benchmarks.py',
        'tests/test_miscentering.py', 'tests/test_population.py',
    ]
    result = subprocess.run(command, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    (output/'pytest.txt').write_text(result.stdout, encoding='utf-8')

    if result.returncode == 0:
        from azlens.miscentering import draw_offset_realization
        from azlens.population import publication_grids, sample_halo_population_from_state
        targets = json.loads((root/'reference/population_realization/population_targets.json').read_text())
        metrics = {'grids': [], 'colossus_version': dependencies.get('colossus')}
        selected = targets['selected_indices']
        for target, grid in zip(targets['grids'], publication_grids()):
            halo = sample_halo_population_from_state(
                grid,
                size=targets['n_halo'],
                c200c_median=target['c200c_median_inferred'],
                shape_peak_height=target['shape_peak_height_inferred'],
            )
            offsets = draw_offset_realization(size=targets['n_halo'], grid_index=grid.grid_index)
            metrics['grids'].append({
                'grid_index': grid.grid_index,
                'population_seed': grid.population_seed,
                'size': halo.size,
                'c200c_median': halo.c200c_median,
                'shape_peak_height': halo.shape_peak_height,
                'minor_axis_proposal_count': halo.minor_axis_proposal_count,
                'minor_axis_rejection_fraction': halo.minor_axis_rejection_fraction,
                'selected_catalog': halo.selected_catalog(selected),
                'offset_digest_summary': offsets.digest_summary(),
            })
        (output/'population_realization_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        metadata['generated_products'] = ['population_realization_metrics.json']

    metadata.update(returncode=result.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment_and_result.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Reports:', output)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
