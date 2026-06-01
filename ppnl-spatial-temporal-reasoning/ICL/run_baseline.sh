#!/usr/bin/env bash
# Run 5-shot ICL baseline on all single-goal PPNL test sets.
# Usage: bash run_baseline.sh [--max-samples N]
#
# Outputs under ICL/outputs/:
#   baseline_5shot_{testset}.jsonl   raw inference records
#
# Prerequisites: venv activated, OPENAI_API_KEY set in .env at repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../single_goal"
OUT_DIR="$SCRIPT_DIR/outputs"

MAX_SAMPLES_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-samples) MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

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
    stem="baseline_5shot_${test_file%.json}"
    output="$OUT_DIR/${stem}.jsonl"

    echo "=== $test_file ==="
    python "$SCRIPT_DIR/run_baseline.py" "$input" "$output" $MAX_SAMPLES_ARG
    echo
done

echo "All done. Results in $OUT_DIR"
