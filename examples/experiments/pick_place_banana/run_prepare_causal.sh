#!/usr/bin/env bash
set -euo pipefail

: "${DEMO_PATH:?Set DEMO_PATH=/abs/path/to/pick_place_banana_demos.pkl before preparing CID causal model.}"

EXP_NAME=${EXP_NAME:-pick_place_banana}
OBJ_EXTRACTOR=${OBJ_EXTRACTOR:-hsv_banana}
OUTPUT_DIR=${OUTPUT_DIR:-./cid_causal_ckpt/${EXP_NAME}}

python ../../prepare_causal_model.py \
    --exp_name="$EXP_NAME" \
    --demo_path="$DEMO_PATH" \
    --obj_extractor="$OBJ_EXTRACTOR" \
    --output_dir="$OUTPUT_DIR" "$@"
