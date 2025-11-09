# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

echo "Starting training Meanflow ..."

# /home/jzx/anaconda3/envs/torchcfm/bin/python trainVideo.py
# /home/jzx/anaconda3/envs/torchcfm/bin/python
# /root/anaconda3/envs/prediff/bin/python train_UNet.py \
#     --config configs/UNet_server.yaml \
#     --log_dir logs/UNet_1e-3_flowratio_0.75 \
#     --batch_size 8 \
#     --mode train \
#     --gpus 1

/root/anaconda3/envs/prediff/bin/python train_UNet_Unconditional.py \
    --config configs/UNet_Unconditional_server.yaml \
    --log_dir logs/UNet_Unconditional_1e-3_flowratio_0.75 \
    --batch_size 8 \
    --mode train \
    --gpus 1 \
    --ckpt_path logs/UNet_1e-3_flowratio_1.0/checkpoints/step_255000-loss_0.3062.ckpt

echo "Training finished."