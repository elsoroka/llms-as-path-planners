"""
Heatmap + ablation delta for gpt-4-path-planning results.

One heatmap per model (2 pages):
  rows    = text (1 try) | code (1 try) | text-feedback k=7 | code-form k=7
  columns = 6 geometry × split conditions
  cells   = success rate

Third page: ablation delta bar chart (code-form k=7 minus text-feedback k=7)
            for Gemini 2.0 Flash Lite, the only model with both strategies.

Saved to:
  outputs/heatmap_{model_id}.pdf  (one per model)
  outputs/heatmap_ablation_delta.pdf
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

MODELS = [
    # (csv_model_fragment, display_label, has_text_feedback)
    ("gemini-2.0-flash-001",      "Gemini 2.0 Flash",      False),
    ("gemini-2.0-flash-lite-001", "Gemini 2.0 Flash Lite", True),
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


def build_matrix(csv_rows, model_frag, has_text_feedback):
    """4 × 6 success-rate matrix. NaN where data is absent."""
    mat = np.full((4, 6), np.nan)
    for j, (geo, split, _) in enumerate(COLS):
        cf_key = f"code_form_stanford_{model_frag}_{geo}_{split}_k7_code"
        tf_key = f"text_feedback_{geo}_{split}_k7_code_{model_frag}"
        if has_text_feedback:
            mat[0, j] = get_try1_rate(csv_rows, tf_key)   # text 1-try
            mat[2, j] = get_rate(csv_rows, tf_key)         # text-feedback k=7
        mat[1, j] = get_try1_rate(csv_rows, cf_key)        # code 1-try
        mat[3, j] = get_rate(csv_rows, cf_key)             # code-form k=7
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
                        fontsize=12, color="#666666")
            else:
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=13)
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Success rate (%)", fontsize=12)
    cb.ax.tick_params(labelsize=11)

    # horizontal divider between 1-try rows and k=7 rows
    ax.axhline(1.5, color="black", linewidth=1.5, linestyle="--")

# ---------------------------------------------------------------------------
# Ablation delta figure
# ---------------------------------------------------------------------------

def plot_ablation_delta(ax, csv_rows):
    model_frag = "gemini-2.0-flash-lite-001"
    cf_rates, tf_rates = [], []
    for geo, split, _ in COLS:
        cf_rates.append(get_rate(csv_rows, f"code_form_stanford_{model_frag}_{geo}_{split}_k7_code"))
        tf_rates.append(get_rate(csv_rows, f"text_feedback_{geo}_{split}_k7_code_{model_frag}"))

    cf = np.array(cf_rates) * 100
    tf = np.array(tf_rates) * 100
    delta = cf - tf

    x = np.arange(len(COLS))
    col_labels = [lbl for _, _, lbl in COLS]

    # side-by-side bars: text-feedback and code-form
    w = 0.35
    ax.bar(x - w / 2, tf,    width=w, label="Text-feedback (k=7)",
           color="#AAAAAA", hatch="///", edgecolor="black", linewidth=0.8)
    ax.bar(x + w / 2, cf,    width=w, label="Code-form (k=7)",
           color="#4878CF", hatch="xxx", edgecolor="black", linewidth=0.8)

    for xi, (t, c) in enumerate(zip(tf, cf)):
        d = c - t
        top = max(t, c)
        ax.text(xi, top + 1.5, f"Δ{d:+.0f}pp",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.set_title(
        "Ablation: code representation vs. direction-token output\n"
        "Gemini 2.0 Flash Lite, k=7 feedback rounds",
        fontsize=14, fontweight="bold", pad=10,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_rows = load_csv()
    col_labels = [lbl for _, _, lbl in COLS]

    for model_frag, model_label, has_tf in MODELS:
        mat = build_matrix(csv_rows, model_frag, has_tf)
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        plot_heatmap(ax, mat, col_labels,
                     f"25×25 gridworld planning — {model_label}")
        out = OUT_DIR / f"heatmap_{model_frag}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved → {out}")

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    plot_ablation_delta(ax, csv_rows)
    out = OUT_DIR / "heatmap_ablation_delta.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
