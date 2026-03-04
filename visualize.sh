# # Visualization
# cd /mnt/data1/MeanFlow/ 

# echo "Starting training Meanflow ..."

# /home/jzx/anaconda3/envs/torchcfm/bin/python vis_JiT_RFDPIC.py \
#     --config configs/JiT-B_RFDPIC_vis.yaml \
#     --rfdpic_config configs/rfdpic_config.yaml \
#     --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
#     --log_dir logs/JiT-B_visualize \
#     --sample_steps 10 \
#     --batch_size 4 \
#     --mode test \
#     --gpus 1 \
#     --ckpt_path logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt
    
# echo "Training finished."


# Analysis Inference Steps ~ N
cd /mnt/data1/MeanFlow/

echo "Starting training Meanflow ..."

# 定义要循环的sample_steps值
sample_steps_array=(1 5 10 50 100 500 1000)

# 遍历数组
for steps in "${sample_steps_array[@]}"
do
    echo "=============================================="
    echo "Running with sample_steps = $steps"
    echo "=============================================="
    
    /home/jzx/anaconda3/envs/torchcfm/bin/python vis_JiT_RFDPIC.py \
        --config configs/JiT-B_RFDPIC_vis.yaml \
        --rfdpic_config configs/rfdpic_config.yaml \
        --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
        --log_dir logs/JiT-B_visualize \
        --sample_steps "$steps" \
        --batch_size 4 \
        --mode test \
        --gpus 1 \
        --ckpt_path logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt
    
    echo "sample_steps = $steps completed."
    echo ""
done

echo "All training finished."