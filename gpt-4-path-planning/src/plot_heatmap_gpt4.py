"""
Heatmap for gpt-4-path-planning results — GPT-4 model only.

One heatmap (1 page):
  rows    = text (1 try) | code (1 try) | text-feedback k=7 | code-form k=7
  columns = 6 geometry × split conditions
  cells   = success rate

Saved to:
  outputs/heatmap_gpt-4.pdf
"""

import csv
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR  = Path(__file__).parent
CSV_PATH = SRC_DIR / "outputs" / "results.csv"
OUT_DIR  = SRC_DIR / "outputs"

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

ROW_LABELS = [
    "Text\n(1 try)",
    "Code\n(1 try)",
    "Text-feedback\n(k=7)",
    "Code-form\n(k=7)",
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv():
    rows = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows[r["file"]] = r
    return rows


def _float(csv_rows, key, field):
    val = csv_rows.get(key, {}).get(field, "")
    return float(val) if val not in ("", None) else np.nan


def get_rate(csv_rows, key):
    return _float(csv_rows, key, "success_rate")


def get_try1_rate(csv_rows, key):
    n   = _float(csv_rows, key, "n_samples")
    n1  = _float(csv_rows, key, "n_success_try_1")
    if np.isnan(n) or n == 0:
        return np.nan
    return n1 / n


def build_matrix(csv_rows):
    """4 × 6 success-rate matrix. NaN where data is absent."""
    model_frag = "openai_gpt-4"
    mat = np.full((4, 6), np.nan)
    for j, (geo, split, _) in enumerate(COLS):
        cf_key = f"code_form_{model_frag}_{geo}_{split}_k7_code"
        tf_key = f"text_feedback_{geo}_{split}_k7_code_gpt-4"
        mat[0, j] = get_try1_rate(csv_rows, tf_key)   # text 1-try
        mat[1, j] = get_try1_rate(csv_rows, cf_key)    # code 1-try
        mat[2, j] = get_rate(csv_rows, tf_key)          # text-feedback k=7
        mat[3, j] = get_rate(csv_rows, cf_key)          # code-form k=7
    return mat

# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def plot_heatmap(ax, matrix, col_labels, title):
    cmap = matplotlib.colormaps["YlGn"].copy()
    cmap.set_bad(color="#CCCCCC")   # grey for N/A cells

    masked = np.ma.masked_invalid(matrix * 100)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=15, color="#666666")
            else:
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=16, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=16)
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=16)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Success rate (%)", fontsize=15)
    cb.ax.tick_params(labelsize=14)

    # horizontal divider between 1-try rows and k=7 rows
    ax.axhline(1.5, color="black", linewidth=1.5, linestyle="--")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_rows = load_csv()
    col_labels = [lbl for _, _, lbl in COLS]

    mat = build_matrix(csv_rows)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    plot_heatmap(ax, mat, col_labels, "25×25 gridworld planning — GPT-4")
    out = OUT_DIR / "heatmap_gpt-4.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
