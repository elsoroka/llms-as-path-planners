"""
Evaluate PPNL single-goal results from the new JSONL output format.

Reads all *.jsonl files in the outputs/ directory whose names match
the naming convention produced by run_baseline.py and code_form.py,
then writes a summary CSV to outputs/results_summary.csv.

Success is defined as: the last recorded attempt has error == null.

Filename conventions parsed:
  baseline_5shot[_stanford]_<model>_<stem>.jsonl
  code_form[_stanford]_<model>_<stem>.jsonl
  code_form_k<k>[_stanford]_<model>_<stem>.jsonl  (older format)

Stems → human-readable test-set labels:
  1_goals_test_seen_6x6_samples          -> seen 6x6
  1goals_unseen_6x6_samples              -> unseen 6x6
  1_goals_test_unseen_5x5_samples        -> unseen 5x5
  1_goals_test_unseen_7x7_samples        -> unseen 7x7
  1_goals_test_unseen_6x6more_obstacles_samples -> 6x6 more obstacles
"""
import json
import os
import re
import csv
from parse_code_form import parse_response

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUT_CSV     = os.path.join(OUTPUTS_DIR, "results_summary.csv")

STEM_LABELS = {
    "1_goals_test_seen_6x6_samples":                    "seen 6x6",
    "1goals_unseen_6x6_samples":                        "unseen 6x6",
    "1_goals_test_unseen_5x5_samples":                  "unseen 5x5",
    "1_goals_test_unseen_7x7_samples":                  "unseen 7x7",
    "1_goals_test_unseen_6x6more_obstacles_samples":    "6x6 more obstacles",
}

# Known model names (longest first so greedy prefix matching works correctly).
# Extend this list if new models are added.
KNOWN_MODELS = sorted([
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gpt-4",
    "gpt-4o",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
], key=len, reverse=True)


def parse_filename(name: str):
    """
    Return (method, provider, model, test_set_label) or None if unrecognised.

    method   : 'baseline_5shot' | 'code_form' | 'code_form_k<N>'
    provider : 'stanford' | 'openai' | 'vllm' | ''
    model    : model name string
    test_set : human-readable label
    """
    stem = name[:-len(".jsonl")] if name.endswith(".jsonl") else name

    # --- method prefix ---
    if stem.startswith("baseline_5shot"):
        method  = "baseline_5shot"
        rest    = stem[len("baseline_5shot"):]
    elif re.match(r"code_form_k\d+", stem):
        m       = re.match(r"(code_form_k\d+)(.*)", stem)
        method  = m.group(1)
        rest    = m.group(2)
    elif stem.startswith("code_form_harder"):
        method  = "code_form_harder"
        rest    = stem[len("code_form_harder"):]
    elif stem.startswith("code_form"):
        method  = "code_form"
        rest    = stem[len("code_form"):]
    else:
        return None

    # rest starts with optional _<provider> then _<model>_<test_stem>
    # Try to strip known provider tokens
    provider = ""
    for tok in ("_stanford", "_openai", "_vllm"):
        if rest.startswith(tok):
            provider = tok[1:]
            rest = rest[len(tok):]
            break

    # rest is now _<model>_<test_stem>
    rest = rest.lstrip("_")

    # Match known model names greedily
    model = None
    for km in KNOWN_MODELS:
        if rest.startswith(km):
            model = km
            rest  = rest[len(km):].lstrip("_")
            break

    if model is None:
        # Fall back: first token before the test stem
        # Test stems all start with "1_goals" or "1goals"
        m = re.match(r"(.+?)_(1(?:_goals|goals).*)", rest)
        if m:
            model = m.group(1)
            rest  = m.group(2)
        else:
            return None

    test_set = STEM_LABELS.get(rest, rest)
    return method, provider, model, test_set


def success_rate(records):
    """
    Compute success rate from a list of sample records.

    Baseline format: record has top-level 'error' key.
    Code-form format: record has 'attempts' list; use last attempt's 'error'.
    """
    n_success = 0
    n_total   = 0
    for r in records:
        if "_config" in r:
            continue
        n_total += 1
        if "attempts" in r:
            err = r["attempts"][-1]["error"]
        else:
            err = r.get("error")
        if err is None:
            n_success += 1
    return n_total, n_success


def optimality_stats(records):
    """
    Compute optimality rate for a list of sample records.

    Baseline format: record has top-level 'response' (direction tokens).
    Code-form format: record has 'attempts' list; parse move_x/move_y calls
      from the first successful attempt's response.

    A successful record is optimal when its path length equals the number of
    tokens in 'ground_truth' (the A*-optimal path).

    Returns (n_optimal, n_success).
    """
    n_optimal = 0
    n_success = 0
    for r in records:
        if "_config" in r:
            continue

        gt_len = len(r.get("ground_truth", "").split())

        if "attempts" in r:
            # Code-form: find first successful attempt and parse its moves.
            successful = next((a for a in r["attempts"] if a["error"] is None), None)
            if successful is None:
                continue
            n_success += 1
            actions, _ = parse_response(successful["response"])
            if len(actions.split()) == gt_len:
                n_optimal += 1
        else:
            # Baseline: top-level error + response are direction tokens.
            if r.get("error") is not None:
                continue
            n_success += 1
            if len(r.get("response", "").split()) == gt_len:
                n_optimal += 1

    return n_optimal, n_success


def count_comment_stats(records):
    """
    Return (total_comment_lines, total_code_lines) across all attempts in all records.

    Definitions (so that `something() # note` and `# note\\nsomething()` yield
    the same ratio):
      - code_lines:    non-empty, non-fence lines whose stripped content does NOT
                       start with '#' (lines that contain actual code, with or
                       without an inline comment).
      - comment_lines: non-empty, non-fence lines whose stripped content contains
                       '#' anywhere (pure comment lines AND inline-commented code
                       lines both count).
    """
    def _tally(response):
        comment, code = 0, 0
        for raw in response.splitlines():
            s = raw.strip()
            if not s or s.startswith("```"):
                continue
            if "#" in s:
                comment += 1
            if not s.startswith("#"):
                code += 1
        return comment, code

    total_comment, total_code = 0, 0
    for r in records:
        if "_config" in r:
            continue
        for attempt in r.get("attempts", []):
            c, t = _tally(attempt.get("response", ""))
            total_comment += c
            total_code    += t
        if "response" in r and "attempts" not in r:
            c, t = _tally(r["response"])
            total_comment += c
            total_code    += t
    return total_comment, total_code


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
        if not fname.endswith(".jsonl"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            print(f"  skip (unrecognised): {fname}")
            continue
        method, provider, model, test_set = parsed

        records        = load_jsonl(os.path.join(OUTPUTS_DIR, fname))
        n, n_succ      = success_rate(records)
        rate           = n_succ / n if n > 0 else float("nan")
        n_opt, n_succ2 = optimality_stats(records)
        opt_rate       = (n_opt / n) if n > 0 else float("nan")
        n_comment, n_code = count_comment_stats(records)
        cr             = n_comment / n_code if n_code else None

        rows.append({
            "method":                method,
            "provider":              provider,
            "model":                 model,
            "test_set":              test_set,
            "n_samples":             n,
            "n_success":             n_succ,
            "success_rate":          f"{rate:.4f}",
            "n_optimal":             n_opt,
            "optimality_rate":       f"{opt_rate:.4f}",
            "total_comment_lines":   n_comment,
            "total_code_lines":      n_code,
            "comment_ratio":         f"{cr:.4f}" if cr is not None else "",
        })
        opt_str = f"{n_opt}/{n_succ2} = {opt_rate:.1%}"
        cr_str  = f"{cr:.1%}" if cr is not None else "n/a"
        print(f"  {method:20s} | {model:30s} | {test_set:30s} | "
              f"success {n_succ}/{n} = {rate:.1%} | optimal {opt_str} | "
              f"comment ratio {cr_str}")

    if not rows:
        print("No results found.")
        return

    fieldnames = ["method", "provider", "model", "test_set",
                  "n_samples", "n_success", "success_rate",
                  "n_optimal", "optimality_rate",
                  "total_comment_lines", "total_code_lines", "comment_ratio"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
