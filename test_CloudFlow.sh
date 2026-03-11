cd /mnt/data1/MeanFlow/ 

echo "Starting testing CloudFlow ..."

/home/jzx/anaconda3/envs/torchcfm/bin/python train_CloudFlow_JiU_add.py \
    --config configs/CloudFlow_JiU.yaml \
    --log_dir logs/CloudFlow_JiU_add_alpha_0.125 \
    --sample_steps 10 \
    --batch_size 4 \
    --mode test \
    --gpus 1 \
    --ckpt_path logs/CloudFlow_JiU_add_alpha_0.125/checkpoints/step_100000-loss_0.0915.ckpt
    # --use_wandb

echo "Test finished."