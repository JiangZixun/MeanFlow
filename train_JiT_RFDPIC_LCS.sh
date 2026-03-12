#!/bin/bash

cd /mnt/data1/MeanFlow/ 

echo "Starting training CloudFlow ..."
    
/home/jzx/anaconda3/envs/torchcfm/bin/python train_JiT_RFDPIC_LCS.py \
    --config configs/JiT-B_RFDPIC_LCS.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
    --log_dir logs/JiT-B_RFDPIC_LCS \
    --batch_size 8 \
    --mode train \
    --gpus 1 \
    --use_wandb \
    # --ckpt_path logs/JiT-B_RFDPIC_LCS/checkpoints/step_100000-loss_0.1257.ckpt

echo "All training finished."