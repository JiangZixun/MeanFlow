cd /mnt/data1/MeanFlow/ 

echo "Starting training Meanflow ..."

# /home/jzx/anaconda3/envs/torchcfm/bin/python trainVideo.py
/home/jzx/anaconda3/envs/torchcfm/bin/python train_UNet.py \
    --config configs/UNet.yaml \
    --log_dir logs/UNet_1e-3_flowratio_1.0 \
    --batch_size 8 \
    --mode train \
    --gpus 1 \
    --ckpt_path logs/UNet_1e-3_flowratio_1.0/checkpoints/step_255000-loss_0.3062.ckpt

echo "Training finished."