cd /mnt/data1/MeanFlow/ 
# cd /opt/data/private/MeanFlow/ 

echo "Starting testing Meanflow ..."

# /root/anaconda3/envs/prediff/bin/python 
/home/jzx/anaconda3/envs/torchcfm/bin/python train_UNet_RFDPIC.py \
    --config configs/UNet_RFDPIC.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt "pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt" \
    --ckpt_path "logs/UNet_RFDPIC_1e-4_flowratio_1.0/checkpoints/step_565000-loss_0.0916.ckpt" \
    --log_dir logs/test \
    --sample_steps 1000 \
    --batch_size 8 \
    --mode test \
    --gpus 1 \

echo "Test finished."

echo "Starting testing Meanflow ..."

# /root/anaconda3/envs/prediff/bin/python 
/home/jzx/anaconda3/envs/torchcfm/bin/python train_UNet_RFDPIC.py \
    --config configs/UNet_RFDPIC.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt "pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt" \
    --ckpt_path "logs/UNet_RFDPIC_1e-4_flowratio_1.0/checkpoints/step_975000-loss_0.0859.ckpt" \
    --log_dir logs/test \
    --sample_steps 1000 \
    --batch_size 8 \
    --mode test \
    --gpus 1 \

echo "Test finished."