import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv("results_summary.csv")

# Define display order and labels for x-axis
test_set_order = [
    "unseen 5x5",
    "seen 6x6",
    "unseen 6x6",
    "6x6 more obstacles",
    "unseen 7x7",
]
test_set_labels = ["5x5", "6x6 (seen)", "6x6 (unseen)", "6x6 (+obs)", "7x7"]
x = list(range(len(test_set_order)))

# Color per model, line style per method (solid=code_form, dashed=baseline)
model_color = {
    "gemini-2.0-flash-001":      "tab:blue",
    "gemini-2.0-flash-lite-001": "tab:orange",
    "gpt4":                      "tab:green",
    "gpt4_k7":                   "tab:green",
}
combos = [
    ("baseline_5shot", "gemini-2.0-flash-001",      "--", "o", "Baseline / Flash"),
    ("baseline_5shot", "gemini-2.0-flash-lite-001", "--", "s", "Baseline / Flash-Lite"),
    ("baseline_5shot", "gpt4",                      "--", "^", "Baseline / GPT-4"),
    ("code_form",      "gemini-2.0-flash-001",      "-",  "o", "Code-Form / Flash"),
    ("code_form",      "gemini-2.0-flash-lite-001", "-",  "s", "Code-Form / Flash-Lite"),
    ("code_form",      "gpt4_k7",                   "-",  "^", "Code-Form / GPT-4"),
]

fig, ax = plt.subplots(figsize=(8, 5))

for method, model, ls, marker, label in combos:
    subset = df[(df["method"] == method) & (df["model"] == model)]
    rates = []
    for ts in test_set_order:
        row = subset[subset["test_set"] == ts]
        rates.append(row["success_rate"].values[0])
    ax.plot(x, rates, linestyle=ls, marker=marker, label=label,
            color=model_color[model], linewidth=1.8, markersize=7)

ax.set_xticks(x)
ax.set_xticklabels(test_set_labels, fontsize=12)
ax.set_ylabel("Pass Rate (1 attempt)", fontsize=14)
ax.set_xlabel("Benchmark", fontsize=14)
ax.set_title("PPNL Results by Model", fontsize=16)
ax.set_ylim(0, 1.1)
ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle=":", linewidth=0.7)

fig.tight_layout()
fig.savefig("results_by_model.pdf")
print("Saved results_by_model.pdf")


fig, ax = plt.subplots(figsize=(8, 5))

for method, model, ls, marker, label in combos:
    subset = df[(df["method"] == method) & (df["model"] == model)]
    rates = []
    for ts in test_set_order:
        row = subset[subset["test_set"] == ts]
        rates.append(row["optimality_rate"].values[0])
    ax.plot(x, rates, linestyle=ls, marker=marker, label=label,
            color=model_color[model], linewidth=1.8, markersize=7)

ax.set_xticks(x)
ax.set_xticklabels(test_set_labels, fontsize=12)
ax.set_ylabel("Optimal Rate (1 attempt)", fontsize=14)
ax.set_xlabel("Benchmark", fontsize=14)
ax.set_title("PPNL Optimality by Model", fontsize=16)
ax.set_ylim(0, 1.0)
ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle=":", linewidth=0.7)

fig.tight_layout()
fig.savefig("optimality_results_by_model.pdf")
print("Saved optimality_results_by_model.pdf")
