#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/MeanFlow

PYTHON_BIN="${PYTHON_BIN:-/home/jzx/anaconda3/envs/torchcfm/bin/python}"
CONFIG="${CONFIG:-configs/JiT-B_RFDPIC.yaml}"
RFDPIC_CONFIG="${RFDPIC_CONFIG:-configs/rfdpic_config.yaml}"
RFDPIC_CKPT="${RFDPIC_CKPT:-pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt}"
LOG_DIR="${LOG_DIR:-logs/2026ICMLRebuttal/JiT-B_persistence_baseline_test}"

"${PYTHON_BIN}" /mnt/data1/MeanFlow/2026ICMLRebuttal/persistence_baseline_test.py \
  --config "${CONFIG}" \
  --rfdpic_config "${RFDPIC_CONFIG}" \
  --rfdpic_ckpt "${RFDPIC_CKPT}" \
  --log_dir "${LOG_DIR}" \
  --sample_steps "${SAMPLE_STEPS:-10}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --mode test \
  --gpus "${GPUS:-1}" \
  "$@"
