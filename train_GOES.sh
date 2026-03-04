# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."

/opt/conda/envs/mamba/bin/python train_JiT_RFDPIC.py \
    --config configs/GOES/JiT-B_RFDPIC.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn_GOES.pt \
    --log_dir logs/GOES/JiT-B \
    --sample_steps 10 \
    --batch_size 8 \
    --mode train \
    --gpus 1
    # --ckpt_path logs/UNet_RFDPIC_Residual/checkpoints/step_500000-loss_0.1334.ckpt \
    

echo "Training finished."