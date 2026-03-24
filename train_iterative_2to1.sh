#!/usr/bin/env bash
set -e

cd /mnt/data1/MeanFlow/

PYTHON_BIN="/home/jzx/anaconda3/envs/torchcfm/bin/python"
CONFIG_PATH="configs/JiT-B_RFDPIC_iterative_2to1.yaml"
RFDPIC_CONFIG="configs/rfdpic_config.yaml"
RFDPIC_CKPT="pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt"
LOG_DIR="logs/JiT-B_iterative_2to1"

echo "Starting iterative 2->1 MeanFlow training..."

# "${PYTHON_BIN}" train_JiT_RFDPIC_iterative_2to1.py \
#     --config "${CONFIG_PATH}" \
#     --rfdpic_config "${RFDPIC_CONFIG}" \
#     --rfdpic_ckpt "${RFDPIC_CKPT}" \
#     --log_dir "${LOG_DIR}" \
#     --sample_steps 10 \
#     --batch_size 8 \
#     --mode train \
#     --gpus 1

# Test example:
"${PYTHON_BIN}" train_JiT_RFDPIC_iterative_2to1.py \
    --config "${CONFIG_PATH}" \
    --rfdpic_config "${RFDPIC_CONFIG}" \
    --rfdpic_ckpt "${RFDPIC_CKPT}" \
    --log_dir "${LOG_DIR}" \
    --sample_steps 10 \
    --batch_size 8 \
    --mode test \
    --gpus 1 \
    --ckpt_path "${LOG_DIR}/checkpoints/step_100000-loss_0.0908.ckpt"

echo "Finished."
