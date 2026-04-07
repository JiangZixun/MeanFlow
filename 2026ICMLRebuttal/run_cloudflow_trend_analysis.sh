#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/MeanFlow

PYTHON_BIN="${PYTHON_BIN:-/home/jzx/anaconda3/envs/torchcfm/bin/python}"
CONFIG="${CONFIG:-configs/CloudFlow_JiU.yaml}"
CKPT_PATH="${CKPT_PATH:-logs/CloudFlow_JiU_add_alpha_0.125/checkpoints/step_100000-loss_0.0915.ckpt}"
OUT_DIR="${OUT_DIR:-/mnt/data1/MeanFlow/2026ICMLRebuttal/cloudflow_trend_metrics}"

"${PYTHON_BIN}" /mnt/data1/MeanFlow/2026ICMLRebuttal/cloudflow_trend_analysis.py \
  --config "${CONFIG}" \
  --ckpt_path "${CKPT_PATH}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --sample_steps "${SAMPLE_STEPS:-10}" \
  --gpus "${GPUS:-1}" \
  --out_dir "${OUT_DIR}" \
  "$@"
