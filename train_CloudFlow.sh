#!/bin/bash

cd /mnt/data1/MeanFlow/ 

echo "Starting training CloudFlow ..."

# 遍历 alpha 值
for alpha in 0.125 0.25 0.5; do
    echo "========================================="
    echo "Running with alpha = ${alpha} ..."
    echo "========================================="
    
    /home/jzx/anaconda3/envs/torchcfm/bin/python train_CloudFlow_JiU_add.py \
        --config configs/CloudFlow_JiU.yaml \
        --log_dir logs/CloudFlow_JiU_add_alpha_${alpha} \
        --sample_steps 10 \
        --batch_size 4 \
        --alpha ${alpha} \
        --mode train \
        --gpus 1 \
        --use_wandb
        # 如果后续需要恢复权重，这里的路径可能也需要改成对应的变量：
        # --ckpt_path logs/CloudFlow_JiU_add_alpha_${alpha}/checkpoints/step_100000-loss_0.1293.ckpt

done

echo "All training finished."