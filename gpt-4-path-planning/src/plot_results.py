import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_csv("outputs/results.csv")

# Keep only the main k7 / 100-sample runs
df = df[(df["max_tries"] == 7) & (df["max_samples"] == 100)].copy()

# Derive method from filename prefix
df["method"] = df["file"].apply(
    lambda f: "text_feedback" if f.startswith("text_feedback") else "code_form"
)

# Define display order for x-axis: geometry x split
test_set_order = [
    ("rectangle", "iid"),
    ("rectangle", "ood"),
    ("maze",      "iid"),
    ("maze",      "ood"),
    ("zig_zag",   "iid"),
    ("zig_zag",   "ood"),
]
test_set_labels = [
    "Rectangle\n(IID)", "Rectangle\n(OOD)",
    "Maze\n(IID)",      "Maze\n(OOD)",
    "Zig-Zag\n(IID)",   "Zig-Zag\n(OOD)",
]
x = list(range(len(test_set_order)))

model_color = {
    "gpt-4":                    "tab:green",
    "gemini-2.0-flash-001":     "tab:blue",
    "gemini-2.0-flash-lite-001":"tab:orange",
}

# (method, model, linestyle, marker, label)
combos = [
    ("code_form",    "gpt-4",                     "-",  "^", "Code-Form / GPT-4"),
    ("code_form",    "gemini-2.0-flash-001",       "-",  "o", "Code-Form / Flash"),
    ("code_form",    "gemini-2.0-flash-lite-001",  "-",  "s", "Code-Form / Flash-Lite"),
    ("text_feedback","gemini-2.0-flash-001",       "--", "o", "Text-Feedback / Flash"),
    ("text_feedback","gemini-2.0-flash-lite-001",  "--", "s", "Text-Feedback / Flash-Lite"),
]


def get_rates(df, method, model, metric):
    subset = df[(df["method"] == method) & (df["model"] == model)]
    rates = []
    for geom, split in test_set_order:
        row = subset[(subset["geometry"] == geom) & (subset["split"] == split)]
        if len(row) == 0:
            rates.append(float("nan"))
        else:
            rates.append(row[metric].values[0])
    return rates


# --- Success rate ---
fig, ax = plt.subplots(figsize=(9, 6))

for method, model, ls, marker, label in combos:
    rates = get_rates(df, method, model, "success_rate")
    ax.plot(x, rates, linestyle=ls, marker=marker, label=label,
            color=model_color[model], linewidth=1.8, markersize=7)

ax.set_xticks(x)
ax.set_xticklabels(test_set_labels, fontsize=13)
ax.set_ylabel("Pass Rate (up to 7 attempts)", fontsize=13)
#ax.set_xlabel("Benchmark", fontsize=13)
ax.set_title("25x25 Path Planning Results by Model", fontsize=15)
ax.set_ylim(0, 1.1)
ax.tick_params(axis="y", labelsize=13)
# put the legend outside the plot
ax.legend(fontsize=13, ncols=2, loc="upper left", bbox_to_anchor=(0, -0.15),
          borderaxespad=0)
ax.grid(axis="y", linestyle=":", linewidth=0.7)

fig.tight_layout()
fig.savefig("outputs/25x25_results_by_model.pdf", bbox_inches="tight")
print("Saved outputs/25x25_results_by_model.pdf")


# --- Optimality rate ---
fig, ax = plt.subplots(figsize=(9, 6))

for method, model, ls, marker, label in combos:
    rates = get_rates(df, method, model, "optimal_rate")
    ax.plot(x, rates, linestyle=ls, marker=marker, label=label,
            color=model_color[model], linewidth=1.8, markersize=7)

ax.set_xticks(x)
ax.set_xticklabels(test_set_labels, fontsize=13)
ax.set_ylabel("Optimal Rate (up to 7 attempts)", fontsize=13)
#ax.set_xlabel("Benchmark", fontsize=13)
ax.set_title("25x25 Path Planning Optimality by Model", fontsize=15)
ax.set_ylim(0, 1.0)
ax.tick_params(axis="y", labelsize=13)
ax.legend(fontsize=13, ncols=2, loc="upper left", bbox_to_anchor=(0, -0.15),
          borderaxespad=0)
ax.grid(axis="y", linestyle=":", linewidth=0.7)

fig.tight_layout()
fig.savefig("outputs/25x25_optimality_results_by_model.pdf", bbox_inches="tight")
print("Saved outputs/25x25_optimality_results_by_model.pdf")
