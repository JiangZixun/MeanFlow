#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/MeanFlow

PYTHON_BIN="${PYTHON_BIN:-/home/jzx/anaconda3/envs/torchcfm/bin/python}"
CONFIG="${CONFIG:-configs/JiT-B_RFDPIC.yaml}"
RFDPIC_CONFIG="${RFDPIC_CONFIG:-configs/rfdpic_config.yaml}"
RFDPIC_CKPT="${RFDPIC_CKPT:-pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt}"
CKPT_PATH="${CKPT_PATH:-logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt}"
OUT_DIR="${OUT_DIR:-/mnt/data1/MeanFlow/2026ICMLRebuttal/final_physical_consistency}"

"${PYTHON_BIN}" /mnt/data1/MeanFlow/2026ICMLRebuttal/final_physical_consistency.py \
  --config "${CONFIG}" \
  --rfdpic_config "${RFDPIC_CONFIG}" \
  --rfdpic_ckpt "${RFDPIC_CKPT}" \
  --ckpt_path "${CKPT_PATH}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --sample_steps "${SAMPLE_STEPS:-10}" \
  --gpus "${GPUS:-1}" \
  --out_dir "${OUT_DIR}" \
  "$@"
