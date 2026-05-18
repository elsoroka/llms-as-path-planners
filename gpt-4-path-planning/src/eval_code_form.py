"""
Evaluate all code-form output files in src/outputs/ and write a CSV summary.

Usage:
    python eval_code_form.py [--outputs-dir DIR] [--out CSV]

For each *_parsed.json file found, reads the matching *.jsonl for config metadata,
computes aggregate metrics, and writes one row to the CSV.

CSV columns:
    file, geometry, split, representation, model, temperature, max_tries, max_samples,
    n_samples, n_success, success_rate, n_optimal, optimal_rate,
    n_success_try_1 ... n_success_try_N  (per-try breakdown for multi-try runs)
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def evaluate_records(records):
    """Compute aggregate metrics from a list of parsed records."""
    n = len(records)

    # Success: error field is null (path reached the goal without hitting obstacles)
    successes = [r for r in records if r['error'] is None]
    n_success = len(successes)

    # Optimal: succeeded AND path length equals the A*-optimal ground truth length
    n_optimal = sum(
        1 for r in successes
        if len(r['generated'][0].split()) == len(r['ground_truth'].split())
    )

    # Per-try success counts (meaningful only for multi-try runs)
    successful_tries = [r['successful_try'] for r in records if r['successful_try'] is not None]
    max_try = max(successful_tries, default=1)
    by_try = {t: successful_tries.count(t) for t in range(1, max_try + 1)}

    return {
        'n_samples':    n,
        'n_success':    n_success,
        'success_rate': n_success / n if n else 0.0,
        'n_optimal':    n_optimal,
        'optimal_rate': n_optimal / n if n else 0.0,
        'by_try':       by_try,
        'max_try':      max_try,
    }


def load_config(jsonl_path):
    """Read the first (config) line from a raw inference JSONL file."""
    with open(jsonl_path) as f:
        first = json.loads(f.readline())
    return first if first.get('_config') else {}


def parse_filename(stem):
    """
    Extract geometry, split, max_tries, representation from output filename stem.
    Expected pattern: code_form_{geometry}_{split}_k{max_tries}_{repr}
    """
    m = re.match(
        r'code_form_(?P<geometry>.+)_(?P<split>iid|ood)_k(?P<max_tries>\d+)'
        r'_(?P<repr>text|code|grid)$',
        stem,
    )
    return m.groupdict() if m else {}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code-form outputs and write a CSV summary."
    )
    parser.add_argument(
        '--outputs-dir',
        default=Path(__file__).parent / 'outputs',
        type=Path,
        help="Directory containing *_parsed.json files (default: src/outputs/).",
    )
    parser.add_argument(
        '--out',
        default=None,
        type=Path,
        help="Output CSV path (default: <outputs-dir>/results.csv).",
    )
    args = parser.parse_args()

    out_dir  = args.outputs_dir
    csv_path = args.out or out_dir / 'results.csv'

    parsed_files = sorted(out_dir.glob('*_parsed.json'))
    if not parsed_files:
        print(f"No *_parsed.json files found in {out_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    max_try_global = 1

    for parsed_path in parsed_files:
        stem      = parsed_path.stem.removesuffix('_parsed')
        raw_path  = out_dir / f'{stem}.jsonl'
        fn_info   = parse_filename(stem)

        records = json.loads(parsed_path.read_text())
        config  = load_config(raw_path) if raw_path.exists() else {}
        argv    = config.get('argv', {})
        metrics = evaluate_records(records)

        max_try_global = max(max_try_global, metrics['max_try'])

        row = {
            'file':           stem,
            'geometry':       fn_info.get('geometry', ''),
            'split':          fn_info.get('split', ''),
            'representation': config.get('representation', fn_info.get('repr', '')),
            'model':          config.get('model', ''),
            'temperature':    config.get('temperature', ''),
            'max_tries':      argv.get('max_tries', fn_info.get('max_tries', '')),
            'max_samples':    argv.get('max_samples', ''),   # blank = full dataset
            'n_samples':      metrics['n_samples'],
            'n_success':      metrics['n_success'],
            'success_rate':   round(metrics['success_rate'], 4),
            'n_optimal':      metrics['n_optimal'],
            'optimal_rate':   round(metrics['optimal_rate'], 4),
        }
        for t, n in metrics['by_try'].items():
            row[f'n_success_try_{t}'] = n

        rows.append(row)
        print(
            f"{stem}: "
            f"{metrics['n_success']}/{metrics['n_samples']} success "
            f"({metrics['success_rate']:.1%}), "
            f"{metrics['n_optimal']}/{metrics['n_samples']} optimal "
            f"({metrics['optimal_rate']:.1%})"
        )

    # Fixed columns first, then per-try columns up to the global maximum
    fixed_fields = [
        'file', 'geometry', 'split', 'representation',
        'model', 'temperature', 'max_tries', 'max_samples',
        'n_samples', 'n_success', 'success_rate', 'n_optimal', 'optimal_rate',
    ]
    try_fields = [f'n_success_try_{t}' for t in range(1, max_try_global + 1)]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=fixed_fields + try_fields, extrasaction='ignore'
        )
        writer.writeheader()
        for row in rows:
            for field in try_fields:
                row.setdefault(field, 0)
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {csv_path}")


if __name__ == '__main__':
    main()
