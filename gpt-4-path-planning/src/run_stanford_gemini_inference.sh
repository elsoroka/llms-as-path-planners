#!/usr/bin/env bash
# Run inference.py with the Code representation on all gpt-4-path-planning
# test sets via the Stanford AI Playground API using Gemini models.
# Requires STANFORD_API_KEY in .env at the repo root.
#
# Outputs saved to src/outputs_fullSet/ following inference.py conventions:
#   env_from_file_out_5_shot_{geometry}_Code_{model}_{iid|ood}_fewShot_25x25.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

MODELS=(
    "gemini-2.0-flash-001"
    "gemini-2.0-flash-lite-001"
)

GEOMETRIES=(
    "rectangle"
    "maze"
    "zig_zag"
)

# IID and OOD filenames per geometry
declare -A IID_FILE
declare -A OOD_FILE
IID_FILE["rectangle"]="rectangle_data_sg_iid.json"
OOD_FILE["rectangle"]="rectangle_data_sg_ood.json"
IID_FILE["maze"]="maze_data_sg_iid.json"
OOD_FILE["maze"]="maze_data_sg_ood.json"
IID_FILE["zig_zag"]="zig_zag_data_sg_indist.json"
OOD_FILE["zig_zag"]="zig_zag_data_sg_ood.json"

mkdir -p "$SCRIPT_DIR/outputs_fullSet"

# inference.py writes to outputs_fullSet/ relative to cwd, so run from SCRIPT_DIR.
cd "$SCRIPT_DIR"

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Model: $model"
    echo "========================================"

    for geometry in "${GEOMETRIES[@]}"; do
        iid="$DATA_DIR/${IID_FILE[$geometry]}"
        ood="$DATA_DIR/${OOD_FILE[$geometry]}"

        echo "=== ${geometry} | Code | ${model} ==="
        python inference.py \
            --provider stanford \
            --model "$model" \
            env_from_file "$geometry" Code "$iid" "$ood"
    done
done

echo ""
echo "All done. Results in $SCRIPT_DIR/outputs_fullSet/"
