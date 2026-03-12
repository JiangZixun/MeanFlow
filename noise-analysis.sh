#!/bin/bash

cd /mnt/data1/MeanFlow/

echo "Starting noise sensitivity analysis for Meanflow..."

# 定义噪声类型和强度
NOISE_TYPES=("additive" "multiplicative")
NOISE_STDS=("0.01" "0.05" "0.1")

# 嵌套循环处理所有组合
for TYPE in "${NOISE_TYPES[@]}"
do
    for STD in "${NOISE_STDS[@]}"
    do
        echo "========================================"
        echo "Noise Type: $TYPE | Noise Std: $STD"
        echo "========================================"
        
        # 创建噪声实验特定的日志目录，例如: logs/Noise-Analysis/additive/0.01
        LOG_DIR="logs/Noise-Analysis/$TYPE/$STD"
        mkdir -p "$LOG_DIR"
        
        /home/jzx/anaconda3/envs/torchcfm/bin/python JiT_RFDPIC-Noise-Analysis.py \
            --config configs/JiT-B_RFDPIC.yaml \
            --rfdpic_config configs/rfdpic_config.yaml \
            --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
            --log_dir "$LOG_DIR" \
            --sample_steps 10 \
            --batch_size 8 \
            --mode test \
            --gpus 1 \
            --ckpt_path logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt \
            --noise_type "$TYPE" \
            --noise_std "$STD"
        
        # 检查上一个命令的退出状态
        if [ $? -eq 0 ]; then
            echo "Successfully completed: $TYPE with std $STD"
        else
            echo "Error processing: $TYPE with std $STD"
            # exit 1 # 如果出错想停止脚本，取消注释此行
        fi
        
        echo ""
    done
done

echo "========================================"
echo "All noise scenarios processed successfully!"
echo "Results saved in: logs/Noise-Analysis/"
echo "========================================"