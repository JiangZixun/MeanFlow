# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."
echo "F_past as x_0"
echo "F_pred as condition"

/opt/conda/envs/mamba/bin/python train_JiT_RFDPIC_reverse.py \
    --config configs/JiT-B_RFDPIC_reverse.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
    --log_dir logs/JiT-B_reverse \
    --sample_steps 10 \
    --batch_size 8 \
    --mode train \
    --gpus 2 \
    # --ckpt_path logs/JiT-B/chechkpoints/step_500000-loss_0.1448.ckpt
    # --use_wandb
    # --ckpt_path logs/UNet_RFDPIC_Residual/checkpoints/step_500000-loss_0.1334.ckpt \
    

echo "Training finished."