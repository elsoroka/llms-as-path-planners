#!/usr/bin/env bash
# Run STL generation experiments on PPNL ICL test sets using GPT-4.1 via OpenAI.
# Requires OPENAI_API_KEY in .env at the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../icl-test-sets"
OUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUT_DIR"

MODEL="gpt-4.1"

TEST_SETS=(
    "ICL_test_set.json"
    "ICL_test_set_5x5worlds.json"
    "ICL_test_set_7x7worlds.json"
    "ICL_test_set_moreobsts.json"
)

for test_file in "${TEST_SETS[@]}"; do
    input="$DATA_DIR/$test_file"
    stem="${test_file%.json}"

    echo "=== stl | $MODEL | $test_file ==="
    python "$SCRIPT_DIR/stl_generation.py" "$input" \
        "$OUT_DIR/stl_${MODEL}_${stem}.jsonl" \
        --provider openai --model "$MODEL" \
        --max-samples 100 --max-tries 3
done

echo "All done. Results in $OUT_DIR"
