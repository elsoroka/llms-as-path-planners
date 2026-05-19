"""
Plot PPNL benchmark success rates for Gemini models via the Stanford API.

Four grouped bars per test set:
  gemini-2.0-flash-001     5-shot baseline
  gemini-2.0-flash-001     code-form (1-shot)
  gemini-2.0-flash-lite-001  5-shot baseline
  gemini-2.0-flash-lite-001  code-form (1-shot)

Bootstrap 95% CIs from the raw JSONL files.
Saved as a PDF to outputs/gemini_results.pdf.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUT_PDF     = OUTPUTS_DIR / "gemini_results.pdf"

# Canonical ordering of test sets (key → x-axis label)
TESTSETS = [
    ("1_goals_test_unseen_5x5_samples",               "5×5\nunseen"),
    ("1_goals_test_seen_6x6_samples",                 "6×6\nseen"),
    ("1goals_unseen_6x6_samples",                     "6×6\nunseen"),
    ("1_goals_test_unseen_6x6more_obstacles_samples", "6×6+obs\nunseen"),
    ("1_goals_test_unseen_7x7_samples",               "7×7\nunseen"),
]

# Four series — distinct color + hatch for B&W legibility
SERIES = [
    {
        "label":   "flash-001\nbaseline 5-shot",
        "file":    "baseline_5shot_stanford_gemini-2.0-flash-001_{stem}.jsonl",
        "color":   "#AAAAAA",
        "hatch":   "///",
        "format":  "baseline",
    },
    {
        "label":   "flash-001\ncode-form 1-shot",
        "file":    "code_form_stanford_gemini-2.0-flash-001_{stem}.jsonl",
        "color":   "#88BB44",
        "hatch":   "...",
        "format":  "code_form",
    },
    {
        "label":   "flash-lite-001\nbaseline 5-shot",
        "file":    "baseline_5shot_stanford_gemini-2.0-flash-lite-001_{stem}.jsonl",
        "color":   "#CCCCCC",
        "hatch":   "\\\\\\",
        "format":  "baseline",
    },
    {
        "label":   "flash-lite-001\ncode-form 1-shot",
        "file":    "code_form_stanford_gemini-2.0-flash-lite-001_{stem}.jsonl",
        "color":   "#4878CF",
        "hatch":   "xxx",
        "format":  "code_form",
    },
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
    lo    = np.percentile(means, 2.5)
    hi    = np.percentile(means, 97.5)
    pt    = arr.mean()
    return pt - lo, hi - pt


def load_success_array(path, fmt):
    """
    Return a binary numpy array (1=success, 0=fail) for each sample.

    baseline format : top-level 'error' field
    code_form format: 'attempts' list, last attempt's 'error'
    """
    arr = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_config"):
                continue
            if fmt == "code_form":
                err = obj["attempts"][-1]["error"]
            else:
                err = obj.get("error")
            arr.append(0.0 if err else 1.0)
    return np.array(arr)


def load_all_data():
    """
    Returns dict: testset_key -> list of numpy arrays (one per series),
    in the same order as SERIES.
    """
    data = {}
    rng  = np.random.default_rng(42)
    for stem, _ in TESTSETS:
        arrays = []
        for s in SERIES:
            fpath = OUTPUTS_DIR / s["file"].format(stem=stem)
            if fpath.exists():
                arrays.append(load_success_array(fpath, s["format"]))
            else:
                print(f"  WARNING: missing {fpath.name}")
                arrays.append(np.array([]))
        data[stem] = arrays
    return data, rng


def make_figure(data, rng):
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)

    n_groups  = len(TESTSETS)
    n_series  = len(SERIES)
    offsets   = np.linspace(-(n_series - 1) / 2,
                             (n_series - 1) / 2, n_series) * BAR_WIDTH
    x         = np.arange(n_groups)

    for s, offset, arrays_by_testset in zip(
        SERIES, offsets,
        [[data[ts][i] for ts, _ in TESTSETS] for i in range(n_series)]
    ):
        values = [arr.mean() if len(arr) > 0 else 0.0
                  for arr in arrays_by_testset]
        cis    = [bootstrap_ci(arr, rng=rng) for arr in arrays_by_testset]
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
            capsize=3.0,
            capthick=1.2,
        )
        for bar, val, err_hi in zip(bars, values, ci_hi):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + err_hi + 0.015,
                f"{val:.0%}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
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
    ax.legend(fontsize=10, framealpha=0.9, ncol=n_series,
              loc="upper left", bbox_to_anchor=(0, 1))
    ax.set_title(
        "PPNL benchmark: 5-shot baseline vs code-form  (Gemini via Stanford API)",
        fontsize=15, fontweight="bold", pad=10,
    )

    return fig


def main():
    data, rng = load_all_data()
    fig = make_figure(data, rng)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig)
    print(f"Saved → {OUT_PDF}")


if __name__ == "__main__":
    main()
