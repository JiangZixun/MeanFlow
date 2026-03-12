# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."

/opt/conda/envs/mamba/bin/python train_JiT_RFDPIC_Gaussion.py \
    --config configs/GOES/JiT_RFDPIC_Gaussion.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn_GOES.pt \
    --log_dir logs/GOES/JiT-B_Gaussion \
    --sample_steps 4 \
    --batch_size 8 \
    --mode test \
    --gpus 1 \
    --ckpt_path logs/GOES/JiT-B_Gaussion/checkpoints/step_500000-loss_0.1201.ckpt

echo "Training finished."