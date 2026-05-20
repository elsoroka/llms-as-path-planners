#!/usr/bin/env bash
# Run text-output with feedback loop on all single-goal test sets.
# Ablation counterpart to run_code_form.sh: same feedback mechanism, text output.
#
# Usage:
#   bash run_text_feedback.sh [--representation {text,code,grid}]
#                             [--max-samples N]
#                             [--max-tries N]
#                             [--model MODEL]
#                             [--provider {openai,vllm,stanford}]
#
# Options (all optional):
#   --representation  Input format fed to the model: text (default), code, or grid.
#   --max-samples     Limit samples per file, e.g. 5 for a quick smoke test.
#   --max-tries       Max LLM retries per sample with error feedback (default: 7).
#   --model           Model name passed to the provider (default: gpt-4).
#   --provider        API provider: openai (default), vllm, or stanford.
#
# Outputs per test file under src/outputs/:
#   text_feedback_{geometry}_{split}_k{max_tries}_{repr}_{model}.jsonl        raw inference
#   text_feedback_{geometry}_{split}_k{max_tries}_{repr}_{model}_parsed.json  parsed results
#
# Prerequisites: venv activated, OPENAI_API_KEY set in .env at repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
OUT_DIR="$SCRIPT_DIR/outputs"

# --- Defaults ---
REPR="text"
MAX_SAMPLES_ARG=100
MAX_TRIES=7
MODEL="gpt-4"
PROVIDER="openai"

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --representation) REPR="$2";                            shift 2 ;;
        --max-samples)    MAX_SAMPLES_ARG="--max-samples $2";  shift 2 ;;
        --max-tries)      MAX_TRIES="$2";                       shift 2 ;;
        --model)          MODEL="$2";                           shift 2 ;;
        --provider)       PROVIDER="$2";                        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Sanitize model name for use in filenames (replace / and : with _).
MODEL_SLUG="${MODEL//\//_}"
MODEL_SLUG="${MODEL_SLUG//:/_}"

mkdir -p "$OUT_DIR"

# (geometry, split, filename) triples
declare -a TEST_SETS=(
    "rectangle iid rectangle_data_sg_iid.json"
    "rectangle ood rectangle_data_sg_ood.json"
    "maze      iid maze_data_sg_iid.json"
    "maze      ood maze_data_sg_ood.json"
    "zig_zag   iid zig_zag_data_sg_indist.json"
    "zig_zag   ood zig_zag_data_sg_ood.json"
)

for entry in "${TEST_SETS[@]}"; do
    read -r geometry split filename <<< "$entry"
    input="$DATA_DIR/$filename"
    stem="text_feedback_${geometry}_${split}_k${MAX_TRIES}_${REPR}_${MODEL_SLUG}"
    raw="$OUT_DIR/${stem}.jsonl"
    parsed="$OUT_DIR/${stem}_parsed.json"

    echo "=== ${geometry} ${split} (repr=${REPR}, max_tries=${MAX_TRIES}, model=${MODEL}) ==="
    python "$SCRIPT_DIR/text_feedback.py" "$input" "$raw" \
        --representation "$REPR" \
        --max-tries "$MAX_TRIES" \
        --model "$MODEL" \
        --provider "$PROVIDER" \
        $MAX_SAMPLES_ARG

    python "$SCRIPT_DIR/parse_text_feedback.py" "$raw" "$parsed"
    echo
done

echo "All done. Results in $OUT_DIR"
