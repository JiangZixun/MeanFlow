# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."

/opt/conda/envs/mamba/bin/python train_JiT_RFDPIC_Gaussion_DDP.py \
    --config configs/JiT-H_RFDPIC_Gaussion.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
    --log_dir logs/JiT-H_Gaussion \
    --batch_size 8 \
    --mode train \
    --gpus 4 \
    # --ckpt_path logs/UNet_RFDPIC_1e-4_flowratio_1.0/checkpoints/step_975000-loss_0.0859.ckpt

echo "Training finished."