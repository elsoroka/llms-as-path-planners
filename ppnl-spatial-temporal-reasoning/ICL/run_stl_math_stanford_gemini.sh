#!/usr/bin/env bash
# Run STL generation experiments on PPNL ICL test sets via the
# Stanford AI Playground API using Gemini models.
# Requires STANFORD_API_KEY in .env at the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../icl-test-sets"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

MODELS=(
    "gemini-2.0-flash-001"
    "gemini-2.0-flash-lite-001"
)

TEST_SETS=(
    "ICL_test_set.json"
    "ICL_test_set_5x5worlds.json"
    "ICL_test_set_7x7worlds.json"
    "ICL_test_set_moreobsts.json"
)

for model in "${MODELS[@]}"; do
    for test_file in "${TEST_SETS[@]}"; do
        input="$DATA_DIR/$test_file"
        stem="${test_file%.json}"

        echo "=== stl | $model | $test_file ==="
        python "$SCRIPT_DIR/stl_math_generation.py" "$input" \
            "$OUT_DIR/stl_math_harder_stanford_${model}_${stem}.jsonl" \
            --provider stanford --model "$model" \
            --max-samples 100 --max-tries 3
    done
done

echo "All done. Results in $OUT_DIR"
