"""
Bar chart of 25x25 gridworld planning results.

One figure per Gemini model (2-page PDF), six x-axis groups
(rectangle/maze/zig-zag × IID/OOD), three bars per group:
  - grey  (///): 5-shot Code baseline  (outputs_baseline/ JSON files)
  - green (...): code-form pass at try 1
  - blue  (xxx): code-form pass at any try (k=7)

Bootstrap 95% CIs throughout.
Saved to outputs/results.pdf.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------

SRC_DIR      = Path(__file__).parent
BASELINE_DIR = SRC_DIR / "outputs_baseline"
CSV_PATH     = SRC_DIR / "outputs" / "results.csv"
OUT_PDF      = SRC_DIR / "outputs" / "results.pdf"

# Add src/ to path so we can import evaluate.success_sg
sys.path.insert(0, str(SRC_DIR))
from evaluate import success_sg

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    ("gemini-2.0-flash-001",      "stanford_gemini-2.0-flash-001",
     "Gemini 2.0 Flash"),
    ("gemini-2.0-flash-lite-001", "stanford_gemini-2.0-flash-lite-001",
     "Gemini 2.0 Flash Lite"),
]

GROUPS = [
    ("rectangle", "iid", "Rectangle\nIID"),
    ("rectangle", "ood", "Rectangle\nOOD"),
    ("maze",      "iid", "Maze\nIID"),
    ("maze",      "ood", "Maze\nOOD"),
    ("zig_zag",   "iid", "Zig-zag\nIID"),
    ("zig_zag",   "ood", "Zig-zag\nOOD"),
]

SERIES = [
    {"label": "5-shot baseline",    "color": "#AAAAAA", "hatch": "///"},
    {"label": "code-form (try 1)",  "color": "#88BB44", "hatch": "..."},
    {"label": "code-form (k=7)",    "color": "#4878CF", "hatch": "xxx"},
]

BAR_WIDTH   = 0.22
N_BOOTSTRAP = 2000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(samples, n=N_BOOTSTRAP, rng=None):
    """Return (lower_err, upper_err) distances from point estimate to CI bounds."""
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(samples, dtype=float)
    sz = len(arr)
    if sz == 0:
        return 0.0, 0.0
    means = np.array([rng.choice(arr, size=sz, replace=True).mean()
                      for _ in range(n)])
    pt = arr.mean()
    return pt - np.percentile(means, 2.5), np.percentile(means, 97.5) - pt


def samples_from_rate(rate, n):
    n_pos = round(float(rate) * int(n))
    return np.array([1.0] * n_pos + [0.0] * (int(n) - n_pos))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline_samples(model_id, geometry, split):
    """Return binary success array from the 5-shot Code baseline JSON."""
    path = BASELINE_DIR / (
        f"env_from_file_out_5_shot_{geometry}_Code_{model_id}_{split}_fewShot_25x25.json"
    )
    if not path.exists():
        print(f"  WARNING: missing baseline {path.name}")
        return np.array([])
    with open(path) as f:
        data = json.load(f)
    arr = []
    for env in data:
        for sample in env:
            arr.append(1.0 if success_sg(sample["World"], sample["Predicted"]) else 0.0)
    return np.array(arr)


def load_csv_rows():
    rows = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows[r["file"]] = r
    return rows


def load_data():
    """
    Returns dict[model_id][group_key] = {
        "baseline": np.ndarray,
        "cf_try1":  np.ndarray,
        "cf_k7":    np.ndarray,
    }
    """
    csv_rows = load_csv_rows()
    data = {}

    for model_id, frag, _ in MODELS:
        data[model_id] = {}
        for geometry, split, _ in GROUPS:
            group_key = f"{geometry}_{split}"

            # Baseline: from JSON
            baseline = load_baseline_samples(model_id, geometry, split)

            # Code-form: from CSV
            csv_key = f"code_form_{frag}_{geometry}_{split}_k7_code"
            row = csv_rows.get(csv_key, {})
            n       = int(row.get("n_samples", 0) or 0)
            sr      = float(row.get("success_rate", 0) or 0)
            n_try1  = int(row.get("n_success_try_1", 0) or 0)

            cf_k7   = samples_from_rate(sr, n)
            cf_try1 = samples_from_rate(n_try1 / n if n else 0.0, n)

            data[model_id][group_key] = {
                "baseline": baseline,
                "cf_try1":  cf_try1,
                "cf_k7":    cf_k7,
            }

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(model_data, model_label, rng):
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)

    n_groups = len(GROUPS)
    n_series = len(SERIES)
    offsets  = np.linspace(-(n_series - 1) / 2,
                            (n_series - 1) / 2, n_series) * BAR_WIDTH
    x        = np.arange(n_groups)
    data_keys = ["baseline", "cf_try1", "cf_k7"]

    for s, offset, dk in zip(SERIES, offsets, data_keys):
        arrays = [model_data[f"{geom}_{split}"][dk]
                  for geom, split, _ in GROUPS]
        values = [arr.mean() if len(arr) > 0 else 0.0 for arr in arrays]
        cis    = [bootstrap_ci(arr, rng=rng) for arr in arrays]
        ci_lo, ci_hi = zip(*cis)

        bars = ax.bar(
            x + offset, values,
            width=BAR_WIDTH,
            label=s["label"],
            color=s["color"],
            hatch=s["hatch"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.errorbar(
            x + offset, values,
            yerr=[list(ci_lo), list(ci_hi)],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.2,
        )
        for bar, val, err_hi in zip(bars, values, ci_hi):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + err_hi + 0.015,
                f"{val:.0%}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in GROUPS], fontsize=13)
    ax.set_ylabel("Success rate", fontsize=14)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="y", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_title(
        f"25×25 gridworld planning: {model_label}",
        fontsize=15, fontweight="bold", pad=10,
    )

    return fig


def main():
    rng  = np.random.default_rng(42)
    data = load_data()

    OUT_PDF.parent.mkdir(exist_ok=True)
    for model_id, frag, model_label in MODELS:
        out_path = OUT_PDF.parent / f"results_{model_id}.pdf"
        fig = make_figure(data[model_id], model_label, rng)
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
