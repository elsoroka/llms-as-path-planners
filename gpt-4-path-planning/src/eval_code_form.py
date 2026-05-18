"""
Evaluate code-form and baseline inference outputs and write a CSV summary.

Works with outputs from:
  - gpt-4-path-planning/src/outputs/        code_form_{geometry}_{split}_k{K}_{repr}.jsonl
  - ppnl-spatial-temporal-reasoning/ICL/outputs/  code_form_k{K}_{testset}.jsonl
  - ppnl-spatial-temporal-reasoning/ICL/outputs/  baseline_5shot_{testset}.jsonl

Reads directly from the raw *.jsonl files, so *_parsed.json intermediaries
are not required.

Usage:
    python eval_code_form.py [--outputs-dir DIR] [--out CSV]

CSV columns:
    file, geometry, split, grid_size, visibility, representation,
    model, temperature, max_tries, max_samples,
    n_samples, n_success, success_rate, n_optimal, optimal_rate,
    n_success_try_1 ... n_success_try_N
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Inline parse logic (mirrors parse_code_form.py, no import needed)
# ---------------------------------------------------------------------------

def _parse_response(response):
    """Extract move_x/move_y calls and expand to action tokens."""
    actions = []
    skipped = []
    for match in re.finditer(r'move_([xy])\(([^)]+)\)', response):
        axis, arg = match.group(1), match.group(2).strip()
        try:
            dist = int(arg)
        except ValueError:
            skipped.append(f"move_{axis}({arg})")
            continue
        if axis == 'x':
            actions.extend(['right' if dist > 0 else 'left'] * abs(dist))
        else:
            actions.extend(['down' if dist > 0 else 'up'] * abs(dist))
    return ' '.join(actions), skipped


def _extract_direction_tokens(response):
    """Extract plain direction tokens from a baseline response string."""
    return ' '.join(t for t in response.split() if t in ('up', 'down', 'left', 'right'))


def _resolve(record):
    """
    Return (action_str, error, successful_try) for one raw jsonl record.

    Handles three formats:
      - multi-try code_form  (has 'attempts' list with move_x/move_y responses)
      - single-response code_form  (has 'response' with move_x/move_y calls)
      - baseline  (has 'response' with plain direction tokens)
    """
    attempts = record.get('attempts', [])
    if attempts:
        for a in attempts:
            if a['error'] is None:
                actions, _ = _parse_response(a['response'])
                return actions, None, a['try']
        last = attempts[-1]
        actions, _ = _parse_response(last['response'])
        return actions, last['error'], None

    response = record['response']
    error    = record.get('error')

    # Detect format by presence of move_x/move_y calls
    if 'move_' in response:
        actions, skipped = _parse_response(response)
        if not error and skipped:
            error = f"Non-integer arg(s): {skipped}"
    else:
        # Baseline: response is plain direction tokens
        actions = _extract_direction_tokens(response)

    return actions, error, (1 if error is None else None)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _ppnl_testset_meta(testset):
    """Extract grid_size and visibility from a PPNL testset name string."""
    grid_m  = re.search(r'\d+x\d+(?:more_obstacles)?', testset)
    grid    = grid_m.group(0) if grid_m else ''
    vis     = 'unseen' if 'unseen' in testset else 'seen'
    return grid, vis


def _parse_filename(stem):
    """
    Try all known filename patterns and return a metadata dict.

    gpt-4 pattern:     code_form_{geometry}_{split}_k{K}_{repr}
    PPNL code pattern: code_form_k{K}_{testset_name}
    PPNL baseline:     baseline_5shot_{testset_name}
    """
    # gpt-4-path-planning
    m = re.match(
        r'code_form_(?P<geometry>.+)_(?P<split>iid|ood)_k(?P<max_tries>\d+)'
        r'_(?P<repr>text|code|grid)$',
        stem,
    )
    if m:
        d = m.groupdict()
        return {
            'geometry':   d['geometry'],
            'split':      d['split'],
            'grid_size':  '',
            'visibility': '',
            'repr':       d['repr'],
            'max_tries':  int(d['max_tries']),
        }

    # PPNL code_form
    m = re.match(r'code_form_k(?P<max_tries>\d+)_(?P<testset>.+)$', stem)
    if m:
        grid, vis = _ppnl_testset_meta(m.group('testset'))
        return {
            'geometry':   '',
            'split':      '',
            'grid_size':  grid,
            'visibility': vis,
            'repr':       'text',
            'max_tries':  int(m.group('max_tries')),
        }

    # PPNL baseline (5-shot ICL, single attempt)
    m = re.match(r'baseline_5shot_(?P<testset>.+)$', stem)
    if m:
        grid, vis = _ppnl_testset_meta(m.group('testset'))
        return {
            'geometry':   '',
            'split':      '',
            'grid_size':  grid,
            'visibility': vis,
            'repr':       '5-shot',
            'max_tries':  1,
        }

    return {'geometry': '', 'split': '', 'grid_size': '', 'visibility': '',
            'repr': '', 'max_tries': ''}


def _extract_config(config):
    """Normalise config fields across argparse-dict and sys.argv-list formats."""
    argv = config.get('argv', {})
    if isinstance(argv, dict):
        max_tries   = argv.get('max_tries', '')
        max_samples = argv.get('max_samples', '')
        repr_       = config.get('representation', argv.get('representation', ''))
    else:
        # sys.argv list: positional args vary; read max_tries from last positional if present
        max_tries   = argv[-1] if len(argv) >= 5 else ''
        max_samples = argv[3]  if len(argv) >= 4 else ''
        repr_       = config.get('representation', '')
    return {
        'model':       config.get('model', ''),
        'temperature': config.get('temperature', ''),
        'max_tries':   max_tries,
        'max_samples': max_samples,
        'repr':        repr_,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(jsonl_path):
    """
    Read a raw *.jsonl file and return (config, metrics).
    Skips the config line; processes all sample lines.
    """
    with open(jsonl_path) as f:
        lines = f.readlines()

    config  = {}
    records = []
    for line in lines:
        obj = json.loads(line)
        if obj.get('_config'):
            config = obj
        else:
            records.append(obj)

    n           = len(records)
    n_success   = 0
    n_optimal   = 0
    by_try      = {}

    for rec in records:
        actions, error, successful_try = _resolve(rec)
        gt = rec.get('ground_truth', rec.get('path', ''))

        if error is None:
            n_success += 1
            if len(actions.split()) == len(gt.split()):
                n_optimal += 1
            t = successful_try or 1
            by_try[t] = by_try.get(t, 0) + 1

    max_try = max(by_try, default=1)

    metrics = {
        'n_samples':    n,
        'n_success':    n_success,
        'success_rate': n_success / n if n else 0.0,
        'n_optimal':    n_optimal,
        'optimal_rate': n_optimal / n if n else 0.0,
        'by_try':       by_try,
        'max_try':      max_try,
    }
    return config, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code-form outputs and write a CSV summary."
    )
    parser.add_argument(
        '--outputs-dir',
        default=Path(__file__).parent / 'outputs',
        type=Path,
        help="Directory containing *.jsonl output files (default: src/outputs/).",
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

    # Find raw inference files; skip test/smoke files without a known pattern
    jsonl_files = sorted(
        p for p in out_dir.glob('*.jsonl')
        if re.match(r'(code_form|baseline)_', p.stem)
    )
    if not jsonl_files:
        print(f"No code_form_*.jsonl files found in {out_dir}", file=sys.stderr)
        sys.exit(1)

    rows           = []
    max_try_global = 1

    for jsonl_path in jsonl_files:
        stem     = jsonl_path.stem
        fn_info  = _parse_filename(stem)
        config, metrics = _evaluate(jsonl_path)
        cfg      = _extract_config(config)

        max_try_global = max(max_try_global, metrics['max_try'])

        row = {
            'file':           stem,
            'geometry':       fn_info['geometry'],
            'split':          fn_info['split'],
            'grid_size':      fn_info['grid_size'],
            'visibility':     fn_info['visibility'],
            'representation': cfg['repr'] or fn_info['repr'],
            'model':          cfg['model'],
            'temperature':    cfg['temperature'],
            'max_tries':      cfg['max_tries'] or fn_info['max_tries'],
            'max_samples':    cfg['max_samples'],
            'n_samples':      metrics['n_samples'],
            'n_success':      metrics['n_success'],
            'success_rate':   round(metrics['success_rate'], 4),
            'n_optimal':      metrics['n_optimal'],
            'optimal_rate':   round(metrics['optimal_rate'], 4),
        }
        for t, cnt in metrics['by_try'].items():
            row[f'n_success_try_{t}'] = cnt

        rows.append(row)
        print(
            f"{stem}: "
            f"{metrics['n_success']}/{metrics['n_samples']} success "
            f"({metrics['success_rate']:.1%}), "
            f"{metrics['n_optimal']}/{metrics['n_samples']} optimal "
            f"({metrics['optimal_rate']:.1%})"
        )

    fixed_fields = [
        'file', 'geometry', 'split', 'grid_size', 'visibility', 'representation',
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
