#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/cloudseg_meanflow.json}"
SAMPLE_STEPS="${2:-10}"

cd "$ROOT_DIR"
python train_cloudseg_meanflow.py --config "$CONFIG_PATH" --sample_steps "$SAMPLE_STEPS"
