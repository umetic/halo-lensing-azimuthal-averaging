#!/usr/bin/env python3
"""Run population field evaluation for one or more grids.

This is a production-oriented entry point, not part of ordinary pytest.  The
full publication-resolution mode library and six-grid field evaluation can be
costly; use small explicit settings for development smoke tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required; activate the project environment first.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from azlens.miscentering import draw_offset_realization
from azlens.mode_library import ModeLibrary
from azlens.population import generate_halo_population, publication_grids
from azlens.population_fields import PopulationFieldConfig, compact_field_validation_metrics, evaluate_grid_population_fields
from azlens.solver_settings import POPULATION_MODE_SETTINGS, SolverSettings


def _small_settings() -> SolverSettings:
    return SolverSettings(
        name='run_population_fields_smoke', q_min=0.18, q_max=1.0, n_q=5,
        x_min=1.0e-4, x_max=20.0, n_x=24, x_solver_min=1.0e-5,
        x_solver_max=80.0, n_radial_solver=96, n_phi_solver=64, m_max=8,
        radial_interpolation='x',
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, action='append', help='Grid index to run; repeatable. Default: 0.')
    parser.add_argument('--production-settings', action='store_true', help='Build the full publication mode library settings.')
    parser.add_argument('--size', type=int, default=16, help='Number of halos per grid for this run. Use 10000 for publication-size runs.')
    parser.add_argument('--n-phi', type=int, default=64, help='Ring azimuthal samples. Publication value is 256.')
    parser.add_argument('--output-dir', default='outputs/population_fields_run')
    args = parser.parse_args()

    grids = publication_grids()
    selected = args.grid if args.grid else [0]
    for idx in selected:
        if idx < 0 or idx >= len(grids):
            parser.error(f'grid index out of range: {idx}')
    settings = POPULATION_MODE_SETTINGS if args.production_settings else _small_settings()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root/out
    out.mkdir(parents=True, exist_ok=True)

    library = ModeLibrary.build(settings)
    library.save_npz(out/'mode_library.npz', extra_metadata={'script': 'run_population_fields.py'})
    config = PopulationFieldConfig(n_phi=args.n_phi, chunk_size=64)
    all_metrics = []
    for idx in selected:
        grid = grids[idx]
        halo = generate_halo_population(grid, size=args.size)
        offsets = draw_offset_realization(size=args.size, grid_index=idx)
        product = evaluate_grid_population_fields(halo, offsets, library, config=config)
        metrics = compact_field_validation_metrics(product)
        all_metrics.append(metrics)
        npz_path = out/f'grid_{idx:02d}_key_samples.npz'
        import numpy as np
        np.savez_compressed(npz_path, **product.key_samples)
        (out/f'grid_{idx:02d}_summary_rows.json').write_text(json.dumps(product.radial_summary_rows, indent=2)+'\n', encoding='utf-8')
    (out/'run_metrics.json').write_text(json.dumps({'settings': settings.name, 'metrics': all_metrics}, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print('Wrote population-field outputs to', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
