#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/configs/example_eval_config.json}"

echo "Starting evaluation..."
echo "Using config: $CONFIG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python "$PROJECT_DIR/src/eval/exec/unlearn_model.py" \
  --config-file "$CONFIG_FILE"

echo "Evaluation completed."
