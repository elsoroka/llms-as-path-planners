#!/usr/bin/env bash
# Run baseline (5-shot) and code-form (1-shot) PPNL experiments via the
# Stanford AI Playground API using Gemini models, 100 samples per test set.
# Requires STANFORD_API_KEY in .env at the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../single_goal"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

MODELS=(
    "gemini-2.0-flash-001"
    "gemini-2.0-flash-lite-001"
)

TEST_SETS=(
    "1_goals_test_seen_6x6_samples.json"
    "1goals_unseen_6x6_samples.json"
    "1_goals_test_unseen_5x5_samples.json"
    "1_goals_test_unseen_7x7_samples.json"
    "1_goals_test_unseen_6x6more_obstacles_samples.json"
)

for model in "${MODELS[@]}"; do
    for test_file in "${TEST_SETS[@]}"; do
        input="$DATA_DIR/$test_file"
        stem="${test_file%.json}"

        #echo "=== baseline | $model | $test_file ==="
        #python "$SCRIPT_DIR/run_baseline.py" "$input" \
        #    "$OUT_DIR/baseline_5shot_stanford_${model}_${stem}.jsonl" \
        #    --provider stanford --model "$model" --max-samples 100

        echo "=== code-form | $model | $test_file ==="
        python "$SCRIPT_DIR/code_form.py" "$input" \
            "$OUT_DIR/code_form_harder_stanford_${model}_${stem}.jsonl" \
            --provider stanford --model "$model" --max-samples 100
    done
done

echo "All done. Results in $OUT_DIR"
