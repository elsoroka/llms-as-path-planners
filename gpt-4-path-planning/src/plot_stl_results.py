import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

df = pd.read_csv(os.path.join(OUTPUTS_DIR, "stl_results_summary.csv"))
df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")

# Only use max_tries=7 (primary experiment) and code representation
df = df[(df["max_tries"] == 7) & (df["representation"] == "code")]

# x-axis: geometry × split combinations
geo_split_order = [
    ("rectangle", "iid"),
    ("rectangle", "ood"),
    ("zig_zag",   "iid"),
    ("zig_zag",   "ood"),
    ("maze",      "iid"),
    ("maze",      "ood"),
]
x_labels = ["rect IID", "rect OOD", "zig-zag IID", "zig-zag OOD", "maze IID", "maze OOD"]
x = list(range(len(geo_split_order)))

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
]

fig, ax = plt.subplots(figsize=(9, 5))

for method, model, ls, marker, label in combos:
    subset = df[(df["method"] == method) & (df["model"] == model)]
    rates  = []
    xs     = []
    for i, (geo, split) in enumerate(geo_split_order):
        row = subset[(subset["geometry"] == geo) & (subset["split"] == split)]
        if row.empty or row["n_samples"].values[0] == 0:
            continue
        rates.append(float(row["success_rate"].values[0]))
        xs.append(i)
    if not rates:
        continue
    ax.plot(xs, rates, linestyle=ls, marker=marker, label=label,
            color=model_color[model], linewidth=1.8, markersize=7)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel("Pass Rate", fontsize=14)
ax.set_xlabel("Geometry / Split", fontsize=14)
ax.set_title("GPT-4 Path Planning: STL Results by Model", fontsize=15)
ax.set_ylim(0, 1.1)
ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle=":", linewidth=0.7)

fig.tight_layout()
out_path = os.path.join(OUTPUTS_DIR, "stl_results_by_model.pdf")
fig.savefig(out_path)
print(f"Saved {out_path}")
