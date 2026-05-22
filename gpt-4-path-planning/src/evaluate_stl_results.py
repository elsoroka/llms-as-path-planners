"""
Evaluate STL results from the JSONL output format.

Reads all stl_*.jsonl files in the outputs/ directory and writes a summary
CSV to outputs/stl_results_summary.csv.

Success is defined as: the last recorded attempt has error == null.

Filename convention parsed:
  stl[_math][_stanford|_openai|_vllm]_<model>_<geometry>_<split>_k<max_tries>_<representation>.jsonl

  e.g. stl_gpt-4.1_maze_iid_k7_code.jsonl
       stl_math_gemini-2.0-flash-001_rectangle_ood_k3_text.jsonl
"""
import json
import os
import re
import csv

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUT_CSV     = os.path.join(OUTPUTS_DIR, "stl_results_summary.csv")

KNOWN_MODELS = sorted([
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
], key=len, reverse=True)

KNOWN_GEOMETRIES = ["zig_zag", "rectangle", "maze"]  # longest first to avoid prefix clash
KNOWN_SPLITS     = ["iid", "ood"]


def parse_filename(name: str):
    """
    Return (method, provider, model, geometry, split, max_tries, representation)
    or None if unrecognised.
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

    rest = rest.lstrip("_")

    # --- model ---
    model = None
    for km in KNOWN_MODELS:
        if rest.startswith(km):
            model = km
            rest  = rest[len(km):].lstrip("_")
            break

    if model is None:
        # Fallback: consume up to the first known geometry
        for geo in KNOWN_GEOMETRIES:
            if geo in rest:
                idx = rest.index(geo)
                model = rest[:idx].rstrip("_")
                rest  = rest[idx:]
                break
        if model is None:
            return None

    # --- geometry ---
    geometry = None
    for geo in KNOWN_GEOMETRIES:
        if rest.startswith(geo):
            geometry = geo
            rest = rest[len(geo):].lstrip("_")
            break

    if geometry is None:
        return None

    # --- split ---
    split = None
    for sp in KNOWN_SPLITS:
        if rest.startswith(sp):
            split = sp
            rest  = rest[len(sp):].lstrip("_")
            break

    if split is None:
        return None

    # --- max_tries (kN) ---
    m = re.match(r"k(\d+)_(.*)", rest)
    if m is None:
        return None
    max_tries      = int(m.group(1))
    representation = m.group(2)  # e.g. "code" or "text"

    return method, provider, model, geometry, split, max_tries, representation


def success_rate(records):
    n_success = 0
    n_total   = 0
    for r in records:
        if "_config" in r:
            continue
        n_total += 1
        attempts = r.get("attempts", [])
        if not attempts:
            continue
        if attempts[-1]["error"] is None:
            n_success += 1
    return n_total, n_success


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
        method, provider, model, geometry, split, max_tries, representation = parsed

        records   = load_jsonl(os.path.join(OUTPUTS_DIR, fname))
        n, n_succ = success_rate(records)
        rate      = n_succ / n if n > 0 else float("nan")

        rows.append({
            "method":         method,
            "provider":       provider,
            "model":          model,
            "geometry":       geometry,
            "split":          split,
            "max_tries":      max_tries,
            "representation": representation,
            "n_samples":      n,
            "n_success":      n_succ,
            "success_rate":   f"{rate:.4f}",
        })
        print(f"  {method:10s} | {model:30s} | {geometry:10s} | {split:3s} | k{max_tries} | "
              f"{representation:5s} | {n_succ}/{n} = {rate:.1%}")

    if not rows:
        print("No STL results found.")
        return

    fieldnames = ["method", "provider", "model", "geometry", "split",
                  "max_tries", "representation", "n_samples", "n_success", "success_rate"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
