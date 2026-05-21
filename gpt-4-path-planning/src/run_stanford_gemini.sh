#!/usr/bin/env bash
# Run code-form inference on all gpt-4-path-planning test sets via the
# Stanford AI Playground API using Gemini models, 100 samples per test set.
# Requires STANFORD_API_KEY in .env at the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

MODELS=(
    "gemini-2.0-flash-001"
    "gemini-2.0-flash-lite-001"
)

# (geometry, split, filename) triples — mirrors run_code_form.sh
declare -a TEST_SETS=(
    "rectangle iid rectangle_data_sg_iid.json"
    "rectangle ood rectangle_data_sg_ood.json"
    "maze      iid maze_data_sg_iid.json"
    "maze      ood maze_data_sg_ood.json"
    "zig_zag   iid zig_zag_data_sg_indist.json"
    "zig_zag   ood zig_zag_data_sg_ood.json"
)

REPR="code"
MAX_TRIES=7
MAX_SAMPLES=100

for model in "${MODELS[@]}"; do
    for entry in "${TEST_SETS[@]}"; do
        read -r geometry split filename <<< "$entry"
        input="$DATA_DIR/$filename"
        stem="code_form_stanford_${model}_${geometry}_${split}_k${MAX_TRIES}_${REPR}"
        raw="$OUT_DIR/${stem}.jsonl"

        echo "=== code-form | $model | ${geometry} ${split} ==="
        python "$SCRIPT_DIR/code_form.py" "$input" "$raw" \
            --provider stanford \
            --model "$model" \
            --representation "$REPR" \
            --max-tries "$MAX_TRIES" \
            --max-samples "$MAX_SAMPLES"
    done
done

echo "All done. Results in $OUT_DIR"
