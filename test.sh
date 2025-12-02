cd /mnt/data1/MeanFlow/ 
# cd /opt/data/private/MeanFlow/ 

echo "Starting testing Meanflow ..."

/home/jzx/anaconda3/envs/torchcfm/bin/python train_JiT_RFDPIC.py \
    --config configs/JiT_RFDPIC.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
    --log_dir logs/JiT_RFDPIC_1e-4_flowratio_1.0 \
    --sample_steps 100 \
    --batch_size 8 \
    --mode test \
    --gpus 1 \
    --ckpt_path logs/JiT_RFDPIC_1e-4_flowratio_1.0/checkpoints/step_1000000-loss_0.1542.ckpt
    # --use_wandb