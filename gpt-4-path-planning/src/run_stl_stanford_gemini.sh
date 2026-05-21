#!/usr/bin/env bash
# Run STL math-form generation on all gpt-4-path-planning test sets using GPT-4.1.
# Requires OPENAI_API_KEY in .env at the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

MODELS=(
    "gemini-2.0-flash-001"
    "gemini-2.0-flash-lite-001"
)
MAX_TRIES=7
MAX_SAMPLES=50

TEST_SETS=(
    "rectangle iid rectangle_data_sg_iid.json"
    "rectangle ood rectangle_data_sg_ood.json"
    "maze      iid maze_data_sg_iid.json"
    "maze      ood maze_data_sg_ood.json"
    "zig_zag   iid zig_zag_data_sg_indist.json"
    "zig_zag   ood zig_zag_data_sg_ood.json"
)

for model in "${MODELS[@]}"; do
    for entry in "${TEST_SETS[@]}"; do
        read -r geometry split filename <<< "$entry"
        input="$DATA_DIR/$filename"
        stem="stl_${model}_${geometry}_${split}_k${MAX_TRIES}_code"

        echo "=== stl | $model | $geometry $split ==="
        python "$SCRIPT_DIR/stl_generation.py" "$input" \
            "$OUT_DIR/${stem}.jsonl" \
            --provider stanford --model "$model" \
            --max-samples $MAX_SAMPLES --max-tries $MAX_TRIES
    done
done

echo "All done. Results in $OUT_DIR"
