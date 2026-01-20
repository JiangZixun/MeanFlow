# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."

/opt/conda/envs/mamba/bin/python train_JiT_RFDPIC_Gaussion_Pure_pred.py \
    --config configs/JiT-B_RFDPIC_Pure_Gaussion.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
    --log_dir logs/JiT-B_Gaussion_Pred \
    --sample_steps 8 \
    --batch_size 8 \
    --mode test \
    --gpus 1 \
    --ckpt_path logs/JiT-B_Gaussion_Pred/checkpoints/step_500000-loss_0.2339.ckpt

echo "Training finished."