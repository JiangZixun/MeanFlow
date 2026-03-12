# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."

/opt/conda/envs/mamba/bin/python train_JiT_RFDPIC_Gaussion_Pure_past.py \
    --config configs/GOES/JiT-B_RFDPIC_Pure_Gaussion.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn_GOES.pt \
    --log_dir logs/GOES/JiT-B_Gaussion_Past \
    --sample_steps 10 \
    --batch_size 8 \
    --mode train \
    --gpus 1 \
    --ckpt_path logs/GOES/JiT-B_Gaussion_Past/checkpoints/step_100000-loss_0.1722.ckpt

echo "Training finished."