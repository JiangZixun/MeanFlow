#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/MeanFlow

PYTHON_BIN="${PYTHON_BIN:-/home/jzx/anaconda3/envs/torchcfm/bin/python}"
CONFIG="${CONFIG:-configs/JiT-B_RFDPIC.yaml}"
RFDPIC_CONFIG="${RFDPIC_CONFIG:-configs/rfdpic_config.yaml}"
RFDPIC_CKPT="${RFDPIC_CKPT:-pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt}"
OUT_DIR="${OUT_DIR:-/mnt/data1/MeanFlow/2026ICMLRebuttal/rfdpic_speed_distribution}"

"${PYTHON_BIN}" /mnt/data1/MeanFlow/2026ICMLRebuttal/rfdpic_speed_distribution.py \
  --config "${CONFIG}" \
  --rfdpic_config "${RFDPIC_CONFIG}" \
  --rfdpic_ckpt "${RFDPIC_CKPT}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --gpus "${GPUS:-1}" \
  --num_bins "${NUM_BINS:-120}" \
  --out_dir "${OUT_DIR}" \
  "$@"
