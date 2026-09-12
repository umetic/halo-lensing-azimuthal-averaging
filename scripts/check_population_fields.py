#!/usr/bin/env python3
"""Run tests through the population field-evaluation layer.

From the repository root, after activating the project environment:
    python scripts/check_population_fields.py

This validation does not run the full six-grid publication production by
default. It exercises the population field engine on compact deterministic
smoke inputs and writes report artifacts under outputs/population_fields_validation/.
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
    parser.add_argument('--output-dir', default='outputs/population_fields_validation',
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
        'increment': 'population-fields-v0.1.0',
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
    print('Running tests through population field evaluation and selection...')
    sys.stdout.flush()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    command = [
        sys.executable, '-m', 'pytest', '-q', '-ra',
        'tests/test_cosmology.py', 'tests/test_nfw.py', 'tests/test_projection.py',
        'tests/test_solver_settings.py', 'tests/test_multipoles.py',
        'tests/test_ring_fields.py', 'tests/test_observables.py',
        'tests/test_controlled_benchmarks.py', 'tests/test_miscentering.py',
        'tests/test_population.py', 'tests/test_mode_library.py',
        'tests/test_selection.py', 'tests/test_population_fields.py',
    ]
    result = subprocess.run(command, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    (output/'pytest.txt').write_text(result.stdout, encoding='utf-8')

    if result.returncode == 0:
        # Generate a compact smoke product for inspection without running the
        # full publication production.
        from azlens.miscentering import draw_offset_realization
        from azlens.mode_library import ModeLibrary
        from azlens.population import publication_grids, sample_halo_population_from_state
        from azlens.population_fields import PopulationFieldConfig, compact_field_validation_metrics, evaluate_grid_population_fields
        from azlens.solver_settings import SolverSettings
        pop_targets = json.loads((root/'reference/population_realization/population_targets.json').read_text())
        field_targets = json.loads((root/'reference/population_fields/population_field_targets.json').read_text())
        grid = publication_grids()[0]
        state = pop_targets['grids'][0]
        smoke = field_targets['smoke_evaluation']
        halo = sample_halo_population_from_state(
            grid,
            size=smoke['n_halo'],
            c200c_median=state['c200c_median_inferred'],
            shape_peak_height=state['shape_peak_height_inferred'],
        )
        offsets = draw_offset_realization(size=smoke['n_halo'], grid_index=grid.grid_index)
        settings = SolverSettings(
            name='validation_smoke_population_fields', q_min=0.18, q_max=1.0, n_q=5,
            x_min=1.0e-4, x_max=20.0, n_x=24, x_solver_min=1.0e-5,
            x_solver_max=80.0, n_radial_solver=96, n_phi_solver=64, m_max=8,
            radial_interpolation='x',
        )
        library = ModeLibrary.build(settings)
        config = PopulationFieldConfig(
            radial_grid=smoke['radial_grid'], source_redshifts=tuple(smoke['source_redshifts']),
            sigma_d=tuple(smoke['sigma_d']), n_phi=smoke['n_phi'], chunk_size=3,
        )
        product = evaluate_grid_population_fields(halo, offsets, library, config=config)
        metrics = compact_field_validation_metrics(product)
        (output/'population_field_smoke_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        metadata['generated_products'] = ['population_field_smoke_metrics.json']

    metadata.update(returncode=result.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment_and_result.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Reports:', output)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
