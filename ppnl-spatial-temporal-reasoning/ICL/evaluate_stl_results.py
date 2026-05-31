"""
Evaluate STL results from the JSONL output format.

Reads all stl_*.jsonl files in the outputs/ directory and writes a summary
CSV to outputs/stl_results_summary.csv.

Success is defined as: the last recorded attempt has error == null.

Filename conventions parsed:
  stl[_math][_stanford]_<model>_<test_stem>.jsonl

Test stems → human-readable labels:
  ICL_test_set                  -> seen 6x6
  ICL_test_set_5x5worlds        -> unseen 5x5
  ICL_test_set_7x7worlds        -> unseen 7x7
  ICL_test_set_moreobsts        -> 6x6 more obstacles
  test_seen_6x6_samples         -> seen 6x6
"""
import json
import os
import re
import csv

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUT_CSV     = os.path.join(OUTPUTS_DIR, "stl_results_summary.csv")

STEM_LABELS = {
    "ICL_test_set":               "seen 6x6",
    "ICL_test_set_5x5worlds":     "unseen 5x5",
    "ICL_test_set_7x7worlds":     "unseen 7x7",
    "ICL_test_set_moreobsts":     "6x6 more obstacles",
    "test_seen_6x6_samples":      "seen 6x6",
}

KNOWN_MODELS = sorted([
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
], key=len, reverse=True)

KNOWN_STEMS = sorted(STEM_LABELS.keys(), key=len, reverse=True)


def parse_filename(name: str):
    """
    Return (method, provider, model, test_set_label) or None if unrecognised.

    method   : 'stl' | 'stl_math'
    provider : 'stanford' | 'openai' | ''
    model    : model name string
    test_set : human-readable label
    """
    if not name.endswith(".jsonl"):
        return None
    stem = name[:-len(".jsonl")]

    if not stem.startswith("stl"):
        return None

    # --- method prefix ---
    if stem.startswith("stl_math_harder"):
        method = "stl_math_harder"
        rest   = stem[len("stl_math_harder"):]
    elif stem.startswith("stl_math"):
        method = "stl_math"
        rest   = stem[len("stl_math"):]
    else:
        method = "stl"
        rest   = stem[len("stl"):]

    # --- optional provider ---
    provider = ""
    for tok in ("_stanford", "_openai", "_vllm"):
        if rest.startswith(tok):
            provider = tok[1:]
            rest = rest[len(tok):]
            break

    # rest is now _<model>_<test_stem>
    rest = rest.lstrip("_")

    # --- model ---
    model = None
    for km in KNOWN_MODELS:
        if rest.startswith(km):
            model = km
            rest  = rest[len(km):].lstrip("_")
            break

    if model is None:
        # Fallback: consume up to the first known test stem
        for ks in KNOWN_STEMS:
            if ks in rest:
                idx = rest.index(ks)
                model = rest[:idx].rstrip("_")
                rest  = ks
                break
        if model is None:
            return None

    test_set = STEM_LABELS.get(rest, rest)
    return method, provider, model, test_set


def success_rate(records):
    n_success = 0
    n_total   = 0
    for r in records:
        if "_config" in r:
            continue
        n_total += 1
        attempts = r.get("attempts", [])
        if not attempts:
            # No attempts recorded — treat as failure
            continue
        if attempts[-1]["error"] is None:
            n_success += 1
    return n_total, n_success


MATH_ERROR_CATEGORIES = [
    "syntax_error",
    "eval_error",
    "rejects_correct_path",
    "accepts_wrong_endpoint",
    "accepts_obstacle",
]

CODE_ERROR_CATEGORIES = [
    "exec_error",
    "type_error",
    "rejects_correct_path",
    "accepts_wrong_endpoint",
    "accepts_obstacle",
]

# Union of both sets for CSV columns (math-only, code-only, shared, other).
ALL_ERROR_CATEGORIES = [
    "syntax_error",       # stl_math: rtamt parse failure
    "eval_error",         # stl_math: formula evaluation exception
    "exec_error",         # stl (code): exec/eval of safe_paths raised an exception
    "type_error",         # stl (code): typecheck failure
    "rejects_correct_path",
    "accepts_wrong_endpoint",
    "accepts_obstacle",
    "other",
]


def classify_math_error(error_str: str) -> str:
    """Map a check_stl_math error string to a category name."""
    if error_str is None:
        return None
    if "Failed to parse STL formula" in error_str:
        return "syntax_error"
    if "Failed to evaluate the STL formula" in error_str:
        return "eval_error"
    if "not satisfied for the ground-truth trajectory" in error_str:
        return "rejects_correct_path"
    if "Doesn't reach the goal" in error_str:
        return "accepts_wrong_endpoint"
    if "Hits an obstacle" in error_str:
        return "accepts_obstacle"
    return "other"


def classify_code_error(error_str: str) -> str:
    """Map a check_stl (code-form) error string to a category name."""
    if error_str is None:
        return None
    if "Failed to evaluate the STL expression" in error_str:
        return "exec_error"
    if "The STL expression is not valid" in error_str:
        return "type_error"
    if "not true for the ground-truth trajectory" in error_str:
        return "rejects_correct_path"
    if "Doesn't reach the goal" in error_str:
        return "accepts_wrong_endpoint"
    if "Hits an obstacle" in error_str:
        return "accepts_obstacle"
    return "other"


def error_counts(records, method: str) -> dict:
    """Count how many failed samples fall into each error category (final attempt)."""
    if method.startswith("stl_math"):
        categories = MATH_ERROR_CATEGORIES
        classify_fn = classify_math_error
    else:
        categories = CODE_ERROR_CATEGORIES
        classify_fn = classify_code_error
    counts = {cat: 0 for cat in categories}
    counts["other"] = 0
    for r in records:
        if "_config" in r:
            continue
        attempts = r.get("attempts", [])
        if not attempts:
            continue
        last_error = attempts[-1]["error"]
        if last_error is None:
            continue  # success — don't count
        cat = classify_fn(last_error)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    rows = []
    for fname in sorted(os.listdir(OUTPUTS_DIR)):
        if not fname.startswith("stl") or not fname.endswith(".jsonl"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            print(f"  skip (unrecognised): {fname}")
            continue
        method, provider, model, test_set = parsed

        records   = load_jsonl(os.path.join(OUTPUTS_DIR, fname))
        n, n_succ = success_rate(records)
        rate      = n_succ / n if n > 0 else float("nan")

        ecounts = error_counts(records, method)

        row = {
            "method":       method,
            "provider":     provider,
            "model":        model,
            "test_set":     test_set,
            "n_samples":    n,
            "n_success":    n_succ,
            "success_rate": f"{rate:.4f}",
        }
        for cat in ALL_ERROR_CATEGORIES:
            row[f"n_{cat}"] = ecounts.get(cat, "")

        rows.append(row)
        print(f"  {method:10s} | {model:30s} | {test_set:25s} | "
              f"{n_succ}/{n} = {rate:.1%}")

    if not rows:
        print("No STL results found.")
        return

    fieldnames = ["method", "provider", "model", "test_set",
                  "n_samples", "n_success", "success_rate"] + \
                 [f"n_{cat}" for cat in ALL_ERROR_CATEGORIES]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
