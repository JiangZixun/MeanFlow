#!/bin/bash

cd /mnt/data1/MeanFlow/

echo "Starting testing Meanflow for all seasons..."

# 定义四个季节
# SEASONS=("spring" "summer" "autumn" "winter")
SEASONS=("autumn")

# 串行处理所有季节
for SEASON in "${SEASONS[@]}"
do
    echo "========================================"
    echo "Processing season: $SEASON"
    echo "========================================"
    
    # 创建季节特定的日志目录
    SEASON_LOG_DIR="logs/Seasonal-Analysis/$SEASON"
    mkdir -p "$SEASON_LOG_DIR"
    
    /home/jzx/anaconda3/envs/torchcfm/bin/python JiT_RFDPIC-Seasonal-Analysis.py \
        --config configs/JiT-B_RFDPIC.yaml \
        --rfdpic_config configs/rfdpic_config.yaml \
        --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
        --log_dir "$SEASON_LOG_DIR" \
        --sample_steps 10 \
        --batch_size 1 \
        --mode test \
        --gpus 1 \
        --ckpt_path logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt \
        --season "$SEASON"
    
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "Successfully completed season: $SEASON"
    else
        echo "Error processing season: $SEASON"
        # 可以选择在此处退出或继续处理下一个季节
        # exit 1
    fi
    
    echo ""
done

echo "========================================"
echo "All seasons processed successfully!"
echo "Results saved in: logs/Seasonal-Analysis/"
echo "========================================"