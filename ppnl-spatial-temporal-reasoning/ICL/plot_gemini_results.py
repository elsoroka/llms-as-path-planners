"""
Plot PPNL benchmark success rates for Gemini models via the Stanford API.

One figure per model, three bars per test set:
  1. 5-shot baseline
  2. code-form pass at try 1
  3. code-form total pass (any try)

Bootstrap 95% CIs from the raw JSONL files.
Saved as a two-page PDF to outputs/gemini_results.pdf.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUT_PDF     = OUTPUTS_DIR / "gemini_results.pdf"

TESTSETS = [
    ("1_goals_test_unseen_5x5_samples",               "5×5\nunseen"),
    ("1_goals_test_seen_6x6_samples",                 "6×6\nseen"),
    ("1goals_unseen_6x6_samples",                     "6×6\nunseen"),
    ("1_goals_test_unseen_6x6more_obstacles_samples", "6×6+obs\nunseen"),
    ("1_goals_test_unseen_7x7_samples",               "7×7\nunseen"),
]

MODELS = [
    ("gemini-2.0-flash-001",      "Gemini 2.0 Flash"),
    ("gemini-2.0-flash-lite-001", "Gemini 2.0 Flash Lite"),
]

# Four series per model — distinct color + hatch for B&W legibility
SERIES = [
    {"label": "baseline 5-shot",         "color": "#AAAAAA", "hatch": "///"},
    {"label": "code-form (try 1)",        "color": "#88BB44", "hatch": "..."},
    {"label": "code-form (any try)",      "color": "#4878CF", "hatch": "xxx"},
    {"label": "code-form (no comments)",  "color": "#CC6677", "hatch": "+++"},
]

BAR_WIDTH   = 0.18
N_BOOTSTRAP = 2000


def bootstrap_ci(samples, n=N_BOOTSTRAP, rng=None):
    """Return (lower_err, upper_err) distances from point estimate to CI bounds."""
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(samples, dtype=float)
    sz  = len(arr)
    if sz == 0:
        return 0.0, 0.0
    means = np.array([rng.choice(arr, size=sz, replace=True).mean()
                      for _ in range(n)])
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    pt = arr.mean()
    return pt - lo, hi - pt


def load_baseline(path):
    """Binary success array for baseline JSONL (top-level 'error' field)."""
    arr = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_config"):
                continue
            arr.append(0.0 if obj.get("error") else 1.0)
    return np.array(arr)


def load_code_form(path):
    """
    Returns (try1_array, total_array) for a code_form JSONL.

    try1_array : 1 if attempt #1 succeeded, else 0
    total_array: 1 if any attempt succeeded, else 0
    """
    try1, total = [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_config"):
                continue
            attempts = obj["attempts"]
            try1.append(1.0 if attempts[0]["error"] is None else 0.0)
            total.append(1.0 if attempts[-1]["error"] is None else 0.0)
    return np.array(try1), np.array(total)


def load_model_data(model_id):
    """
    Returns dict: testset_key -> {"baseline", "cf_try1", "cf_total"} numpy arrays.
    """
    data = {}
    for stem, _ in TESTSETS:
        b_path  = OUTPUTS_DIR / f"baseline_5shot_stanford_{model_id}_{stem}.jsonl"
        cf_path = OUTPUTS_DIR / f"code_form_stanford_{model_id}_{stem}.jsonl"

        baseline = load_baseline(b_path) if b_path.exists() else np.array([])
        if not b_path.exists():
            print(f"  WARNING: missing {b_path.name}")

        if cf_path.exists():
            cf_try1, cf_total = load_code_form(cf_path)
        else:
            print(f"  WARNING: missing {cf_path.name}")
            cf_try1 = cf_total = np.array([])

        harder_path = OUTPUTS_DIR / f"code_form_harder_stanford_{model_id}_{stem}.jsonl"
        if harder_path.exists():
            _, cf_harder = load_code_form(harder_path)
        else:
            print(f"  WARNING: missing {harder_path.name}")
            cf_harder = np.array([])

        data[stem] = {"baseline": baseline, "cf_try1": cf_try1, "cf_total": cf_total, "cf_harder": cf_harder}
    return data


def make_figure(data, model_label, rng):
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)

    n_groups = len(TESTSETS)
    n_series = len(SERIES)
    offsets  = np.linspace(-(n_series - 1) / 2,
                            (n_series - 1) / 2, n_series) * BAR_WIDTH
    x        = np.arange(n_groups)
    data_keys = ["baseline", "cf_try1", "cf_total", "cf_harder"]

    for s, offset, dk in zip(SERIES, offsets, data_keys):
        arrays = [data[ts][dk] for ts, _ in TESTSETS]
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
    ax.set_xticklabels([lbl for _, lbl in TESTSETS], fontsize=13)
    ax.set_ylabel("Success rate", fontsize=14)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="y", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_title(
        f"PPNL benchmark: {model_label}  (Stanford API)",
        fontsize=15, fontweight="bold", pad=10,
    )

    return fig


def main():
    rng = np.random.default_rng(42)
    with PdfPages(OUT_PDF) as pdf:
        for model_id, model_label in MODELS:
            data = load_model_data(model_id)
            fig  = make_figure(data, model_label, rng)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {model_label} done")
    print(f"Saved → {OUT_PDF}")


if __name__ == "__main__":
    main()
