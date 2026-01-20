# cd /mnt/data1/MeanFlow/ 
cd /opt/data/private/MeanFlow/ 

/opt/conda/envs/mamba/bin/python visualize_JiT_RFDPIC.py \
    --config configs/JiT-B_RFDPIC.yaml \
    --rfdpic_config configs/rfdpic_config.yaml \
    --rfdpic_ckpt pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt \
    --log_dir logs/JiT-B \
    --sample_steps 10 \
    --batch_size 4 \
    --mode test \
    --gpus 1 \
    --ckpt_path logs/JiT-B/chechkpoints/step_1000000-loss_0.1542.ckpt