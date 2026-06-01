import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv("stl_results_summary.csv")

# Define display order and labels for x-axis
test_set_order = [
    "unseen 5x5",
    "seen 6x6",
    "6x6 more obstacles",
    "unseen 7x7",
]
test_set_labels = ["5x5", "6x6 (seen)", "6x6 (+obs)", "7x7"]
x = list(range(len(test_set_order)))

# Color per model, line style per method (solid=stl code-form, dashed=stl math)
model_color = {
    "gemini-2.0-flash-001":      "tab:blue",
    "gemini-2.0-flash-lite-001": "tab:orange",
    "gpt-4.1":                   "tab:green",
}
combos = [
    ("stl",      "gemini-2.0-flash-001",      "-",  "o", "STL Code-Form / Flash"),
    ("stl",      "gemini-2.0-flash-lite-001", "-",  "s", "STL Code-Form / Flash-Lite"),
    ("stl",      "gpt-4.1",                   "-",  "^", "STL Code-Form / GPT-4.1"),
    ("stl_math", "gemini-2.0-flash-001",      "--", "o", "STL Math / Flash"),
    ("stl_math", "gemini-2.0-flash-lite-001", "--", "s", "STL Math / Flash-Lite"),
    ("stl_math", "gpt-4.1",                   "--", "^", "STL Math / GPT-4.1"),
    ("stl_math_harder", "gemini-2.0-flash-001",      "--", "o", "STL Math (no example) / Flash"),
    ("stl_math_harder", "gemini-2.0-flash-lite-001", "--", "s", "STL Math (no example) / Flash-Lite"),
]

fig, ax = plt.subplots(figsize=(9, 4))

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
ax.set_ylabel("Pass Rate", fontsize=14)
ax.set_xlabel("Benchmark", fontsize=14)
ax.set_title("PPNL STL Results by Model", fontsize=16)
ax.set_ylim(0, 1.1)
ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle=":", linewidth=0.7)

fig.tight_layout()
fig.savefig("ppnl_stl_results_by_model.pdf")
print("Saved stl_results_by_model.pdf")
