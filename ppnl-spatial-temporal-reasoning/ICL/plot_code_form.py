"""
Plot k7 code-form results from the PPNL benchmark.

Produces a two-panel figure (success rate | optimal rate) saved as a PDF.

Usage:
    python plot_code_form.py [--results CSV] [--out PDF]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Map raw file stems to clean x-axis labels and a canonical sort order
# ---------------------------------------------------------------------------
LABEL_MAP = {
    'code_form_k7_1_goals_test_unseen_5x5_samples':             ('5×5\nunseen',      0),
    'code_form_k7_1_goals_test_seen_6x6_samples':               ('6×6\nseen',        1),
    'code_form_k7_1goals_unseen_6x6_samples':                   ('6×6\nunseen',      2),
    'code_form_k7_1_goals_test_unseen_6x6more_obstacles_samples':('6×6+obs\nunseen', 3),
    'code_form_k7_1_goals_test_unseen_7x7_samples':             ('7×7\nunseen',      4),
}

# Each bar gets a distinct color + hatch so it reads in B&W
STYLES = [
    {'color': '#4878CF', 'hatch': ''},
    {'color': '#6ACC65', 'hatch': '///'},
    {'color': '#D65F5F', 'hatch': '...'},
    {'color': '#B47CC7', 'hatch': 'xxx'},
    {'color': '#C4AD66', 'hatch': '\\\\\\'},
]


def load_k7_rows(csv_path):
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if 'k7' in r['file']]

    entries = []
    for r in rows:
        stem = r['file']
        if stem not in LABEL_MAP:
            continue
        label, order = LABEL_MAP[stem]
        entries.append({
            'label':        label,
            'order':        order,
            'success_rate': float(r['success_rate']),
            'optimal_rate': float(r['optimal_rate']),
            'n_samples':    int(r['n_samples']),
        })
    entries.sort(key=lambda x: x['order'])
    return entries


def make_bar_panel(ax, entries, metric_key, title, ylabel):
    labels  = [e['label']       for e in entries]
    values  = [e[metric_key]    for e in entries]
    n       = len(entries)

    for i, (val, style) in enumerate(zip(values, STYLES)):
        ax.bar(
            i, val,
            color=style['color'],
            hatch=style['hatch'],
            edgecolor='black',
            linewidth=0.8,
            width=0.6,
            label=labels[i],
        )
        # Value label above each bar
        ax.text(i, val + 0.01, f'{val:.0%}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.tick_params(axis='y', labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


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

    entries = load_k7_rows(args.results)
    if not entries:
        raise SystemExit("No k7 rows found in results CSV.")

    fig, (ax_success, ax_optimal) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    make_bar_panel(ax_success, entries,
                   'success_rate', 'Success Rate', 'Success rate')
    make_bar_panel(ax_optimal,  entries,
                   'optimal_rate', 'Optimal Rate', 'Optimal rate')

    fig.suptitle(
        'Code-form planning on PPNL benchmark (GPT-4, k=7)',
        fontsize=17, fontweight='bold',
    )

    with PdfPages(args.out) as pdf:
        pdf.savefig(fig)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
