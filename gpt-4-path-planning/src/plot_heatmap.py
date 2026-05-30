"""
Heatmap of cumulative success rate by try number for gpt-4-path-planning results.

Shape: 7 rows (try 1–7) × 6 columns (benchmark conditions)
  rows    = success after at most n tries  (n = 1 … 7)
  columns = 6 geometry × split conditions

Usage:
  python plot_heatmap.py --model gemini-2.0-flash-001 --strategy code_form
  python plot_heatmap.py --model gemini-2.0-flash-lite-001 --strategy text_feedback

  Or pass a custom key template (must contain {geo} and {split} placeholders):
  python plot_heatmap.py --key-template "code_form_stanford_gemini-2.0-flash-001_{geo}_{split}_k7_code" --title "My Model"

Saved to:
  outputs/heatmap_{model}_{strategy}.pdf   (or --output to override)
"""

import argparse
import csv
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR  = Path(__file__).parent
CSV_PATH = SRC_DIR / "outputs" / "results.csv"
OUT_DIR  = SRC_DIR / "outputs"

MAX_TRIES = 7

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

COLS = [
    ("rectangle", "iid", "Rectangle\nIID"),
    ("rectangle", "ood", "Rectangle\nOOD"),
    ("maze",      "iid", "Maze\nIID"),
    ("maze",      "ood", "Maze\nOOD"),
    ("zig_zag",   "iid", "Zig-zag\nIID"),
    ("zig_zag",   "ood", "Zig-zag\nOOD"),
]

ROW_LABELS = [f"Try {k}" for k in range(1, MAX_TRIES + 1)]

KEY_TEMPLATES = {
    "code_form":     "code_form_{model}_{geo}_{split}_k7_code",
    "text_feedback": "text_feedback_{geo}_{split}_k7_code_{model}",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv():
    rows = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows[r["file"]] = r
    return rows


def cumulative_success_rate(csv_rows, key, n_tries):
    """Cumulative success rate after at most n_tries attempts.
    Matches any CSV row whose file key ends with `key` to ignore provider prefixes."""
    row = next((r for k, r in csv_rows.items() if k.endswith(key)), {})
    n = row.get("n_samples", "")
    if n in ("", None):
        return np.nan
    n = float(n)
    if n == 0:
        return np.nan
    total = 0.0
    for k in range(1, n_tries + 1):
        val = row.get(f"n_success_try_{k}", "")
        if val in ("", None):
            break
        total += float(val)
    return total / n


def build_matrix(csv_rows, key_template, model):
    """Return (MAX_TRIES × 6) matrix of cumulative success rates."""
    mat = np.full((MAX_TRIES, len(COLS)), np.nan)
    for j, (geo, split, _) in enumerate(COLS):
        key = key_template.format(model=model, geo=geo, split=split)
        for i in range(MAX_TRIES):
            mat[i, j] = cumulative_success_rate(csv_rows, key, i + 1)
    return mat

# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def plot_heatmap(ax, matrix, col_labels, title):
    cmap = matplotlib.colormaps["YlGn"].copy()
    cmap.set_bad(color="#CCCCCC")

    masked = np.ma.masked_invalid(matrix * 100)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=11, color="#666666")
            else:
                color = "white" if v > 0.55 else "#666666"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(MAX_TRIES))
    ax.set_yticklabels(ROW_LABELS, fontsize=12)
    ax.set_ylabel("Success after ≤ n tries", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Success rate (%)", fontsize=12)
    cb.ax.tick_params(labelsize=10)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot per-try cumulative success heatmap.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model",    help="Model fragment (e.g. gemini-2.0-flash-001).")
    group.add_argument("--key-template",
                       help="Full CSV key template with {geo} and {split} placeholders.")

    parser.add_argument("--strategy", choices=list(KEY_TEMPLATES), default="code_form",
                        help="Strategy name (used with --model). Default: code_form.")
    parser.add_argument("--title",  default=None,
                        help="Plot title. Defaults to model + strategy.")
    parser.add_argument("--output", default=None,
                        help="Output PDF path. Defaults to outputs/heatmap_{model}_{strategy}.pdf.")
    args = parser.parse_args()

    if args.model:
        key_template_full = KEY_TEMPLATES[args.strategy]
        model = args.model
        title  = args.title or f"{model} — {args.strategy}"
        out    = args.output or OUT_DIR / f"heatmap_{model}_{args.strategy}.pdf"
    else:
        key_template_full = args.key_template
        model = ""
        title  = args.title or args.key_template
        out    = args.output or OUT_DIR / "heatmap_custom.pdf"

    csv_rows = load_csv()
    col_labels = [lbl for _, _, lbl in COLS]
    mat = build_matrix(csv_rows, key_template_full, model)

    fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    plot_heatmap(ax, mat, col_labels, title)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
