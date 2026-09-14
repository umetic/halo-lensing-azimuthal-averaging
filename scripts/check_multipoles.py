#!/usr/bin/env python3
"""Run deterministic-core tests after adding solver settings and multipoles.

From the repository root, after activating the project environment:
    python scripts/check_multipoles.py

No packages are installed and no Git operations are performed. Reports go to
outputs/multipole_validation/, which belongs in the repository's ignore rules.
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
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/multipole_validation',
                        help='Report directory, relative to repository root unless absolute.')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root/output
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root/'src'))
    module_names = ('cosmology', 'nfw', 'projection', 'solver_settings', 'multipoles')
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
        'validation_layer': 'multipoles-v0.1.0',
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
    print('Running deterministic-core tests through Fourier--Green multipoles...')
    sys.stdout.flush()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    command = [sys.executable, '-m', 'pytest', '-q', '-ra',
               'tests/test_cosmology.py', 'tests/test_nfw.py', 'tests/test_projection.py',
               'tests/test_solver_settings.py', 'tests/test_multipoles.py']
    result = subprocess.run(command, cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    (output/'pytest.txt').write_text(result.stdout, encoding='utf-8')
    metadata.update(returncode=result.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
    (output/'environment_and_result.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Reports:', output)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
