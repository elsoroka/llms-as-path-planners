"""
Plot PPNL benchmark results: 5-shot baseline vs code-form (try 1) vs code-form (k=7).

Produces a two-panel figure (success rate | optimal rate) with three grouped bars
per test set, saved as a PDF.

Usage:
    python plot_code_form.py [--outputs-dir DIR] [--out PDF]
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Canonical test-set ordering and display labels
# ---------------------------------------------------------------------------
TESTSETS = [
    ('1_goals_test_unseen_5x5_samples',               '5×5\nunseen'),
    ('1_goals_test_seen_6x6_samples',                 '6×6\nseen'),
    ('1goals_unseen_6x6_samples',                     '6×6\nunseen'),
    ('1_goals_test_unseen_6x6more_obstacles_samples', '6×6+obs\nunseen'),
    ('1_goals_test_unseen_7x7_samples',               '7×7\nunseen'),
]

# Three methods — distinct color + hatch for B&W legibility
METHODS = [
    {'key': '5-shot baseline',    'color': '#AAAAAA', 'hatch': '///'},
    {'key': 'Code-form (try 1)',  'color': '#88BB44', 'hatch': '...'},
    {'key': 'Code-form (k=7)',    'color': '#4878CF', 'hatch': 'xxx'},
]

BAR_WIDTH = 0.25


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_move_response(response):
    """Expand move_x/move_y calls to direction tokens."""
    actions = []
    for m in re.finditer(r'move_([xy])\(([^)]+)\)', response):
        axis, arg = m.group(1), m.group(2).strip()
        try:
            dist = int(arg)
        except ValueError:
            continue
        if axis == 'x':
            actions.extend(['right' if dist > 0 else 'left'] * abs(dist))
        else:
            actions.extend(['down' if dist > 0 else 'up'] * abs(dist))
    return ' '.join(actions)


def _load_code_form_jsonl(path):
    """
    Read a code_form JSONL and return per-sample dicts with:
        successful_try, actions, ground_truth
    """
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('_config'):
                continue
            attempts = obj.get('attempts', [])
            gt = obj.get('ground_truth', '')
            successful_try = None
            actions = ''
            for a in attempts:
                if a['error'] is None:
                    successful_try = a['try']
                    actions = _parse_move_response(a['response'])
                    break
            else:
                if attempts:
                    actions = _parse_move_response(attempts[-1]['response'])
            records.append({
                'successful_try': successful_try,
                'actions':        actions,
                'ground_truth':   gt,
            })
    return records


def load_data(outputs_dir):
    """
    Return a dict keyed by testset suffix with metrics for all three methods.
    {testset_key: {
        'baseline':  {'success': float, 'optimal': float, 'n': int},
        'cf_try1':   {'success': float, 'optimal': float, 'n': int},
        'cf_k7':     {'success': float, 'optimal': float, 'n': int},
    }}
    """
    # Load CSV for baseline and k7 aggregate metrics
    csv_path = outputs_dir / 'results.csv'
    csv_rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            csv_rows[r['file']] = r

    data = {}
    for testset_key, _ in TESTSETS:
        b_stem  = 'baseline_5shot_'  + testset_key
        cf_stem = 'code_form_k7_'    + testset_key
        b_row   = csv_rows.get(b_stem,  {})
        cf_row  = csv_rows.get(cf_stem, {})
        n       = int(cf_row.get('n_samples', 0))

        # Baseline: directly from CSV
        baseline = {
            'success': float(b_row.get('success_rate', 0)),
            'optimal': float(b_row.get('optimal_rate', 0)),
            'n':       int(b_row.get('n_samples', 0)),
        }

        # Code-form k7: directly from CSV
        cf_k7 = {
            'success': float(cf_row.get('success_rate', 0)),
            'optimal': float(cf_row.get('optimal_rate', 0)),
            'n':       n,
        }

        # Code-form try-1: re-read from JSONL for exact per-try counts
        jsonl_path = outputs_dir / f'{cf_stem}.jsonl'
        cf_try1 = {'success': 0.0, 'optimal': 0.0, 'n': n}
        if jsonl_path.exists() and n > 0:
            records = _load_code_form_jsonl(jsonl_path)
            n_success = sum(1 for r in records if r['successful_try'] == 1)
            n_optimal = sum(
                1 for r in records
                if r['successful_try'] == 1
                and len(r['actions'].split()) == len(r['ground_truth'].split())
            )
            cf_try1 = {
                'success': n_success / len(records),
                'optimal': n_optimal / len(records),
                'n':       len(records),
            }

        data[testset_key] = {
            'baseline': baseline,
            'cf_try1':  cf_try1,
            'cf_k7':    cf_k7,
        }

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_panel(ax, data, metric_key, title, ylabel):
    n_groups  = len(TESTSETS)
    n_methods = len(METHODS)
    offsets   = np.linspace(-(n_methods - 1) / 2,
                             (n_methods - 1) / 2, n_methods) * BAR_WIDTH
    x         = np.arange(n_groups)
    method_data_keys = ['baseline', 'cf_try1', 'cf_k7']

    for method, offset, data_key in zip(METHODS, offsets, method_data_keys):
        values = [data[ts][data_key][metric_key] for ts, _ in TESTSETS]
        bars = ax.bar(
            x + offset, values,
            width=BAR_WIDTH,
            label=method['key'],
            color=method['color'],
            hatch=method['hatch'],
            edgecolor='black',
            linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f'{val:.0%}',
                ha='center', va='bottom',
                fontsize=9.5, fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in TESTSETS], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.20)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.tick_params(axis='y', labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outputs-dir',
        default=Path(__file__).parent / 'outputs',
        type=Path,
    )
    parser.add_argument(
        '--out',
        default=Path(__file__).parent / 'outputs' / 'code_form_k7_results.pdf',
        type=Path,
    )
    args = parser.parse_args()

    data = load_data(args.outputs_dir)

    fig, (ax_success, ax_optimal) = plt.subplots(
        1, 2, figsize=(15, 5.5), constrained_layout=True
    )

    make_panel(ax_success, data, 'success', 'Success Rate', 'Success rate')
    make_panel(ax_optimal,  data, 'optimal', 'Optimal Rate', 'Optimal rate')

    fig.suptitle(
        'PPNL benchmark: 5-shot baseline vs code-form  (GPT-4)',
        fontsize=16, fontweight='bold',
    )

    with PdfPages(args.out) as pdf:
        pdf.savefig(fig)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
