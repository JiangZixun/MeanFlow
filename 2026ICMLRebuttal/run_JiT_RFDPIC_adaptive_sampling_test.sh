#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/MeanFlow

PYTHON_BIN="${PYTHON_BIN:-/home/jzx/anaconda3/envs/torchcfm/bin/python}"
CONFIG="${CONFIG:-configs/JiT-B_RFDPIC.yaml}"
RFDPIC_CONFIG="${RFDPIC_CONFIG:-configs/rfdpic_config.yaml}"
RFDPIC_CKPT="${RFDPIC_CKPT:-pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt}"
CKPT_PATH="${CKPT_PATH:-logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt}"
LOG_DIR="${LOG_DIR:-logs/2026ICMLRebuttal/JiT-B_adaptive_sampling_test}"

"${PYTHON_BIN}" /mnt/data1/MeanFlow/2026ICMLRebuttal/train_JiT_RFDPIC_adaptive_sampling.py \
  --config "${CONFIG}" \
  --rfdpic_config "${RFDPIC_CONFIG}" \
  --rfdpic_ckpt "${RFDPIC_CKPT}" \
  --log_dir "${LOG_DIR}" \
  --sample_steps "${SAMPLE_STEPS:-10}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --mode test \
  --gpus "${GPUS:-1}" \
  --adaptive_method "${ADAPTIVE_METHOD:-dopri5}" \
  --ode_rtol "${ODE_RTOL:-1e-4}" \
  --ode_atol "${ODE_ATOL:-1e-5}" \
  --ckpt_path "${CKPT_PATH}" \
  "$@"
