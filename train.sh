cd /mnt/data1/MeanFlow/ 
# cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."

# /home/jzx/anaconda3/envs/torchcfm/bin/python trainVideo.py
# /home/jzx/anaconda3/envs/torchcfm/bin/python
# /root/anaconda3/envs/prediff/bin/python train_UNet.py \
#     --config configs/UNet_server.yaml \
#     --log_dir logs/UNet_1e-3_flowratio_0.75 \
#     --batch_size 8 \
#     --mode train \
#     --gpus 1

# /home/jzx/anaconda3/envs/torchcfm/bin/python train_UNet_Unconditional.py \
#     --config configs/UNet_Unconditional.yaml \
#     --log_dir logs/UNet_Unconditional_1e-4_flowratio_1.0 \
#     --batch_size 8 \
#     --mode train \
#     --ckpt_path logs/UNet_Unconditional_1e-4_flowratio_1.0/checkpoints/step_135000-loss_0.1081.ckpt \
#     --gpus 1 

# /root/anaconda3/envs/prediff/bin/python 
# /home/jzx/anaconda3/envs/torchcfm/bin/python train_JiU_RFDPIC.py \
#     --config configs/JiU_RFDPIC.yaml \
#     --rfdpic_config configs/rfdpic_config.yaml \
#     --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
#     --log_dir logs/JiU_RFDPIC_1e-4_flowratio_1.0 \
#     --sample_steps 10 \
#     --batch_size 8 \
#     --mode train \
#     --gpus 1 \
#     --use_wandb
#     # --ckpt_path logs/JiU_RFDPIC_1e-4_flowratio_1.0/checkpoints/step_050000-loss_0.1065.ckpt \
#     # --use_wandb


/home/jzx/anaconda3/envs/torchcfm/bin/python train_UNet_RFDPIC_Residual.py\
    --config configs/UNet_RFDPIC_Residual.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
    --log_dir logs/UNet_RFDPIC_Residual \
    --sample_steps 10 \
    --batch_size 8 \
    --mode train \
    --gpus 1 \
    --use_wandb
    # --ckpt_path logs/JiU_RFDPIC_1e-4_flowratio_1.0/checkpoints/step_050000-loss_0.1065.ckpt \
    # --use_wandb

echo "Training finished."