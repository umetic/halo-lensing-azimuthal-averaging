#!/usr/bin/env python3
"""Run far-background alignment reweighting from saved key-radius products.

Example after a population-field run:
    python scripts/run_alignment_reweighting.py \
        --key-samples outputs/population_fields_run/grid_00_key_samples.npz \
        --summary-rows outputs/population_fields_run/grid_00_summary_rows.json

The script consumes saved key-radius products. It does not evaluate new lensing
fields, draw new populations, or render figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required; activate the project environment first.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np

from azlens.alignment import (
    alignment_diagnostics,
    alignment_rows_from_products,
    table3_records_from_alignment_rows,
    write_alignment_rows_csv,
    write_table3_csv,
)
from azlens.population_fields import PopulationFieldProducts


def _read_summary_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path} must contain a non-empty list of summary rows')
    return tuple(dict(row) for row in rows)


def _product_from_key_samples(key_path: Path, summary_path: Path) -> PopulationFieldProducts:
    key = {name: value for name, value in np.load(key_path, allow_pickle=False).items()}
    rows = _read_summary_rows(summary_path)
    first = rows[0]
    grid_index = int(first['grid_index'])
    mass = float(first['mass_msun_h'])
    z_l = float(first['z_l'])
    radii = np.asarray(key['R_over_r200c'], dtype=np.float64)
    sigma = tuple(float(x) for x in np.asarray(key['sigma_d'], dtype=np.float64))
    safe_outer = np.asarray(key['safe_outer'], dtype=bool)
    safe_local = np.asarray(key.get('safe_local', safe_outer), dtype=bool)
    lamb = np.asarray(key.get('lambda_min_infinity', np.ones_like(safe_outer, dtype=np.float64)), dtype=np.float64)
    represented = np.mean(safe_outer, axis=1)
    weights = np.asarray(key['source_weights'], dtype=np.float64).reshape(-1)
    source_weights = {'z1': float(weights[0]), 'z2': float(weights[1])}
    if weights.size >= 3:
        source_weights['infinity'] = float(weights[-1])
    return PopulationFieldProducts(
        grid_index=grid_index,
        mass_msun_h=mass,
        z_l=z_l,
        radial_grid=radii,
        sigma_d=sigma,
        source_redshifts=(1.0, 2.0),
        source_weights=source_weights,
        key_indices=np.arange(radii.size),
        lambda_min_infinity=lamb,
        safe_local=safe_local,
        safe_outer=safe_outer,
        represented_fraction_outer=represented,
        radial_summary_rows=rows,
        key_samples=key,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-samples', action='append', type=Path, required=True,
                        help='Saved grid_XX_key_samples.npz file. Repeat for multiple grids.')
    parser.add_argument('--summary-rows', action='append', type=Path, required=True,
                        help='Matching grid_XX_summary_rows.json file. Repeat in the same order.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/alignment_reweighting'))
    args = parser.parse_args()
    if len(args.key_samples) != len(args.summary_rows):
        parser.error('--key-samples and --summary-rows must be supplied the same number of times')
    output = args.output_dir if args.output_dir.is_absolute() else root/args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    products = tuple(_product_from_key_samples(k, s) for k, s in zip(args.key_samples, args.summary_rows))
    rows = alignment_rows_from_products(products)
    table3 = table3_records_from_alignment_rows(rows)
    write_alignment_rows_csv(rows, output/'alignment_reweighted_summary.csv')
    write_table3_csv(table3, output/'alignment_table3.csv')
    (output/'alignment_diagnostics.json').write_text(
        json.dumps(alignment_diagnostics(rows), indent=2, sort_keys=True)+'\n', encoding='utf-8'
    )
    print('Wrote alignment products to', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
