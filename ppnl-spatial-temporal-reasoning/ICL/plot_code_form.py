"""
Plot PPNL benchmark results: code-form (k=7) vs 5-shot ICL baseline.

Produces a two-panel figure (success rate | optimal rate) with grouped bars
(one group per test set, two bars per group), saved as a PDF.

Usage:
    python plot_code_form.py [--results CSV] [--out PDF]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Canonical test-set ordering and display labels
# (keyed by the testset suffix shared by both methods)
# ---------------------------------------------------------------------------
TESTSETS = [
    ('1_goals_test_unseen_5x5_samples',              '5×5\nunseen'),
    ('1_goals_test_seen_6x6_samples',                '6×6\nseen'),
    ('1goals_unseen_6x6_samples',                    '6×6\nunseen'),
    ('1_goals_test_unseen_6x6more_obstacles_samples','6×6+obs\nunseen'),
    ('1_goals_test_unseen_7x7_samples',              '7×7\nunseen'),
]

# Method styles — distinct color + hatch for B&W legibility
METHODS = [
    {
        'key':    '5-shot baseline',
        'prefix': 'baseline_5shot_',
        'color':  '#AAAAAA',
        'hatch':  '///',
    },
    {
        'key':    'Code-form (k=7)',
        'prefix': 'code_form_k7_',
        'color':  '#4878CF',
        'hatch':  'xxx',
    },
]

BAR_WIDTH = 0.35


def load_data(csv_path):
    """Return {stem: {success_rate, optimal_rate}} for all rows."""
    with open(csv_path) as f:
        return {r['file']: r for r in csv.DictReader(f)}


def make_panel(ax, rows, testsets, metric_key, title, ylabel):
    n_groups  = len(testsets)
    n_methods = len(METHODS)
    offsets   = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * BAR_WIDTH
    x         = np.arange(n_groups)

    for method, offset in zip(METHODS, offsets):
        values = []
        for testset_key, _ in testsets:
            stem = method['prefix'] + testset_key
            row  = rows.get(stem)
            values.append(float(row[metric_key]) if row else 0.0)

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
                fontsize=10, fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in testsets], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.tick_params(axis='y', labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results',
        default=Path(__file__).parent / 'outputs' / 'results.csv',
        type=Path,
    )
    parser.add_argument(
        '--out',
        default=Path(__file__).parent / 'outputs' / 'code_form_k7_results.pdf',
        type=Path,
    )
    args = parser.parse_args()

    rows = load_data(args.results)

    fig, (ax_success, ax_optimal) = plt.subplots(
        1, 2, figsize=(14, 5.5), constrained_layout=True
    )

    make_panel(ax_success, rows, TESTSETS, 'success_rate', 'Success Rate', 'Success rate')
    make_panel(ax_optimal,  rows, TESTSETS, 'optimal_rate', 'Optimal Rate', 'Optimal rate')

    fig.suptitle(
        'PPNL benchmark: code-form (k=7) vs 5-shot ICL baseline  (GPT-4)',
        fontsize=16, fontweight='bold',
    )

    with PdfPages(args.out) as pdf:
        pdf.savefig(fig)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
