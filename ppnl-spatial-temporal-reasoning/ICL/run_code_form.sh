#!/usr/bin/env bash
# Run code-form one-shot inference on all single-goal test sets.
# Usage: bash run_code_form.sh [max_samples [max_tries]]
#   max_samples  (optional) limit number of samples per file, e.g. for a quick smoke test
#   max_tries    (optional, default 3) max LLM retries per sample with error feedback
#
# Outputs per test set under ICL/outputs/:
#   code_form_k<K>_<name>.jsonl        raw inference (id, attempts list, ...)
#   code_form_k<K>_<name>_parsed.json  parsed into executor-point-sg.py format
#
# Prerequisites: venv activated, OPENAI_API_KEY set in .env at repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../single_goal"
OUT_DIR="$SCRIPT_DIR/outputs"
MAX_SAMPLES="${1:-}"
MAX_TRIES="${2:-3}"

mkdir -p "$OUT_DIR"

TEST_SETS=(
    "1_goals_test_seen_6x6_samples.json"
    "1goals_unseen_6x6_samples.json"
    "1_goals_test_unseen_5x5_samples.json"
    "1_goals_test_unseen_7x7_samples.json"
    "1_goals_test_unseen_6x6more_obstacles_samples.json"
)

for test_file in "${TEST_SETS[@]}"; do
    input="$DATA_DIR/$test_file"
    stem="code_form_harder_k${MAX_TRIES}_${test_file%.json}"
    raw="$OUT_DIR/${stem}.jsonl"
    parsed="$OUT_DIR/${stem}_parsed.json"

    echo "=== $test_file (max_tries=${MAX_TRIES}) ==="
    # Pass MAX_SAMPLES as empty string when unset; code_form.py treats "" as None.
    python "$SCRIPT_DIR/code_form.py" "$input" "$raw" "$MAX_SAMPLES" "$MAX_TRIES"

    python "$SCRIPT_DIR/parse_code_form.py" "$raw" "$parsed"

    echo "Evaluating..."
    python "$SCRIPT_DIR/../evaluate/executor-point-sg.py" "$parsed" "$input"
    echo
done

echo "All done."
