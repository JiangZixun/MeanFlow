#!/bin/bash

cd /mnt/data1/MeanFlow/

echo "Starting testing Meanflow for missing frame analysis..."

# 定义模式
# MODES=("normal" "copy" "optical_flow")
MODES=("copy" "optical_flow")

for MODE in "${MODES[@]}"
do
    # 根据模式决定需要测试的 frame 列表
    if [ "$MODE" == "normal" ]; then
        FRAMES=(0 1 2 3 4 5)
    else
        FRAMES=(5)
    fi

    for FRAME in "${FRAMES[@]}"
    do
        echo "========================================"
        echo "Processing: Mode=$MODE, Frame=$FRAME"
        echo "========================================"
        
        # 结果保存路径：logs/Missing-Analysis/normal_frame0 等
        LOG_DIR="logs/Missing-Analysis/${MODE}_frame${FRAME}"
        mkdir -p "$LOG_DIR"
        
        /home/jzx/anaconda3/envs/torchcfm/bin/python JiT_RFDPIC-Missing-Analysis.py \
            --config configs/JiT-B_RFDPIC.yaml \
            --rfdpic_config configs/rfdpic_config.yaml \
            --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
            --log_dir "$LOG_DIR" \
            --sample_steps 10 \
            --batch_size 1 \
            --mode test \
            --gpus 1 \
            --ckpt_path logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt \
            --missing_mode "$MODE" \
            --missing_frame "$FRAME"
        
        if [ $? -eq 0 ]; then
            echo "Success: $MODE frame $FRAME"
        else
            echo "Error: $MODE frame $FRAME failed"
        fi
        echo ""
    done
done

echo "========================================"
echo "All tasks completed!"
echo "Results saved in: logs/Missing-Analysis/"
echo "========================================"