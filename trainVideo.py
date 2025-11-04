from models.videoDit import MFDiT
from models.UNet import UNet
import torch
import torchvision
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from videoMeanflow import MeanFlow
from accelerate import Accelerator
import time
import os
from transformers import get_linear_schedule_with_warmup

# 1. --- 增加新的 Imports ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage
# (确保你的 torchcfm 环境中安装了 matplotlib 和 scipy: pip install matplotlib scipy)

# 2. --- 导入你的 Dataset ---
from dataset_btchw import Xiaoshan_6steps_30min_Dataset 

# 3. --- 为可视化函数增加全局变量 ---
CHANNEL_NAME = [f'Ch{i+1}' for i in range(8)]
CMAP = ['gray'] * 8
label_fontsize = 12


# 4. --- 粘贴你提供的可视化函数 ---
def vis_himawari8_seq_btchw(
        save_dir: str,
        context_seq: list,
        pred_seq: list,
        target_seq: list,
        min_max_dict=None,
    ):
    import numpy as np
    
    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取序列长度
    n_col = max(len(context_seq), len(pred_seq), len(target_seq))
    if n_col == 0:
        print("Warning: No sequences provided to visualization.")
        return
        
    # 行标签配置
    row_labels = ["Context", "Pred", "Target", "Pred - Target", "Edge"]
    
    # 动态计算每个通道的min/max值（只基于context和target序列）
    def calculate_channel_minmax(context_seq, target_seq, channel_idx):
        all_values = []
        # 只使用context和target序列来计算min/max
        for seq in [context_seq, target_seq]:
            if seq:  # 确保序列不为空
                for frame in seq:
                    if frame is not None:
                        # 处理PyTorch张量
                        if hasattr(frame, 'detach'):  # 如果是PyTorch张量
                            frame_np = frame.detach().cpu().numpy()
                        else:
                            frame_np = frame
                        
                        if channel_idx < frame_np.shape[0]:
                             all_values.append(frame_np[channel_idx, ...].flatten())
        
        if all_values:
            combined_values = np.concatenate(all_values)
            return combined_values.min(), combined_values.max()
        else:
            return 0, 1  # 默认值
    
    # 为每个通道创建单独的可视化
    for channel_idx in range(8): # 假设固定为 8 通道
        plt.figure(figsize=(n_col*2, 5 * 1.8))  # 高度改为5行
        
        # 计算当前通道的min/max值（只基于context和target序列）
        if min_max_dict is None:
            vmin, vmax = calculate_channel_minmax(context_seq, target_seq, channel_idx)
        else:
            vmin = min_max_dict['min'][channel_idx]
            vmax = min_max_dict['max'][channel_idx]
        
        # 使用GridSpec实现紧密布局
        gs = GridSpec(5, n_col+1, figure=plt.gcf(),  # 增加为5行，增加1列用于标签
                    width_ratios=[0.5]+[1]*n_col,  # 第一列较窄用于标签
                    wspace=0.05, hspace=0.05,
                    left=0, right=1, bottom=0, top=1)
        
        # 添加行标签（左侧）
        for row in range(5):
            ax_label = plt.subplot(gs[row, 0])
            ax_label.text(1, 0.5, row_labels[row], 
                        ha='right', va='center',
                        fontsize=label_fontsize, rotation=90)
            ax_label.axis('off')
        
        def get_img_np(frame, ch_idx):
            if hasattr(frame, 'detach'):
                frame = frame.detach().cpu().numpy()
            if ch_idx < frame.shape[0]:
                return frame[ch_idx, ...]
            else:
                return np.zeros((256, 256)) # 返回一个空图像

        # 绘制context序列
        for t in range(len(context_seq)):
            ax = plt.subplot(gs[0, t+1])  # 从第1列开始
            img = get_img_np(context_seq[t], channel_idx)
            ax.imshow(img, cmap=CMAP[channel_idx], vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        # 绘制pred序列
        for t in range(len(pred_seq)):
            ax = plt.subplot(gs[1, t+1])
            img = get_img_np(pred_seq[t], channel_idx)
            ax.imshow(img, cmap=CMAP[channel_idx], vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        # 绘制target序列
        for t in range(len(target_seq)):
            ax = plt.subplot(gs[2, t+1])
            img = get_img_np(target_seq[t], channel_idx)
            ax.imshow(img, cmap=CMAP[channel_idx], vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        # 绘制pred-target差异
        diff_values = []
        valid_diff_frames = 0
        if len(pred_seq) > 0 and len(target_seq) > 0:
            for t in range(min(len(pred_seq), len(target_seq))):
                pred_img = get_img_np(pred_seq[t], channel_idx)
                target_img = get_img_np(target_seq[t], channel_idx)
                diff = pred_img - target_img
                diff_values.append(diff.flatten())
                valid_diff_frames += 1

            if diff_values:
                combined_diff = np.concatenate(diff_values)
                diff_vmin, diff_vmax = combined_diff.min(), combined_diff.max()
                diff_abs_max = max(abs(diff_vmin), abs(diff_vmax))
                diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max
            else:
                diff_vmin, diff_vmax = -1, 1
        else:
            diff_vmin, diff_vmax = -1, 1
        
        for t in range(valid_diff_frames):
            ax = plt.subplot(gs[3, t+1])
            pred_img = get_img_np(pred_seq[t], channel_idx)
            target_img = get_img_np(target_seq[t], channel_idx)
            diff = pred_img - target_img
            ax.imshow(diff, cmap='coolwarm', vmin=diff_vmin, vmax=diff_vmax)
            ax.axis('off')
        
        # 绘制边缘检测图
        def apply_edge_detection(img):
            """简单的梯度边缘检测，保留小孔洞"""
            from scipy import ndimage
            
            if hasattr(img, 'detach'):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img
            
            img_smooth = ndimage.gaussian_filter(img_np, sigma=0.1)
            grad_x = ndimage.sobel(img_smooth, axis=1)
            grad_y = ndimage.sobel(img_smooth, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            threshold = np.percentile(gradient_magnitude, 80)
            edge_mask = gradient_magnitude > threshold
            binary_edge = edge_mask.astype(np.float32)
            return binary_edge
        
        edge_vmin, edge_vmax = 0, 1
        for t in range(len(pred_seq)):
            ax = plt.subplot(gs[4, t+1])
            edge_img = apply_edge_detection(get_img_np(pred_seq[t], channel_idx))
            ax.imshow(edge_img, cmap='gray', vmin=edge_vmin, vmax=edge_vmax)
            ax.axis('off')
        
        # 保存图像
        save_path = os.path.join(save_dir, f"{CHANNEL_NAME[channel_idx]}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    # --- 填写你的路径和参数 ---
    DATA_PATH = "/mnt/data1/Dataset/Himawari8/train/xiaoshan_6steps_30min_Data_day.h5"     # <<< 必须修改
    JSON_PATH = "/mnt/data1/Dataset/Himawari8/train/xiaoshan_6steps_30min_Data_day_all.json"  # <<< 必须修改
    DATA_PREFIX = "2022"                    # <<< 确认这个前缀
    TRAIN_RATIO = 0.9
    RANDOM_FLIP = 0.5
    
    # HDF5 数据维度 (T=6, C=8, H=256, W=256)
    T_dim = 6
    C_dim = 8
    H_dim = 256
    W_dim = 256
    
    # 5. --- 修改点 1: Patch Size ---
    # (pt, ph, pw) - (时间, 高度, 宽度)
    # (2, 16, 16) -> 16*16*3 = 768 tokens (VRAM 消耗较低)
    patch_size = (2, 16, 16) # <-- 已修改

    n_steps = 20000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("This training script requires a CUDA-capable GPU.")
    
    batch_size = 4
    
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    # --- 替换为你的 Dataset ---
    train_dataset = Xiaoshan_6steps_30min_Dataset(
        data_path=DATA_PATH,
        json_path=JSON_PATH,
        dataset_prefix=DATA_PREFIX,
        train_ratio=TRAIN_RATIO,
        split='train',
        random_flip=RANDOM_FLIP
    )
    
    val_dataset = Xiaoshan_6steps_30min_Dataset(
        data_path=DATA_PATH,
        json_path=JSON_PATH,
        dataset_prefix=DATA_PREFIX,
        train_ratio=TRAIN_RATIO,
        split='valid',
        random_flip=0.0
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8
    )
    
    train_dataloader = cycle(train_dataloader)
    val_dataloader_cycle = cycle(val_dataloader) # 用于采样

    # --- 实例化 3D MFDiT ---
    # model = MFDiT(
    #     input_size=(T_dim, H_dim, W_dim),
    #     patch_size=patch_size,
    #     in_channels=C_dim * 2,
    #     out_channels=C_dim,
    #     dim=384,
    #     depth=12,
    #     num_heads=6,
    #     num_classes=None, 
    # ).to(accelerator.device)
    model = UNet(
        input_size=(T_dim, H_dim, W_dim),
        in_channels=C_dim * 2,
        out_channels=C_dim,
        dim=64,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    num_warmup_steps = 1000 # (例如，在 1000 步内从 0 升到 xxx)

    # --- 实例化 MeanFlow ---
    meanflow = MeanFlow(
        channels=C_dim,
        time_dim=T_dim,
        height_dim=H_dim,
        width_dim=W_dim,
        num_classes=None,
        # flow_ratio=0.50,
        flow_ratio=1.0,  # 训练时全程使用流模型
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond='v'
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=n_steps, # 你的总训练步数
    )

    # model, optimizer, train_dataloader, val_dataloader_cycle = accelerator.prepare(
    #     model, optimizer, train_dataloader, val_dataloader_cycle
    # )
    model, optimizer, train_dataloader, val_dataloader_cycle, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader_cycle, lr_scheduler
    )

    global_step = 0.0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 500

    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            
            c_past = data[0].to(accelerator.device)
            x_future = data[1].to(accelerator.device)

            loss, mse_val = meanflow.loss(model, x_future, c_past)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'
                    # lr = optimizer.param_groups[0]['lr']
                    lr = lr_scheduler.get_last_lr()[0] # <-- 7. 从 scheduler 获取 LR
                    lr_info = f'Learning Rate: {lr:.6f}'
                    log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'
                    with open('log.txt', mode='a') as n:
                        n.write(log_message)
                    losses = 0.0
                    mse_losses = 0.0

            # 6. --- 修改点 3: 替换为新的可视化逻辑 ---
            if global_step % sample_step == 0:
                if accelerator.is_main_process:
                    model_module = accelerator.unwrap_model(model)
                    model_module.eval()
                    
                    with torch.no_grad():
                        val_data = next(val_dataloader_cycle)
                        c_past_val = val_data[0].to(accelerator.device) # (B, T_past, C, H, W)
                        x_future_val = val_data[1].to(accelerator.device) # (B, T_pred, C, H, W)
                        
                        # 7. --- 修改点 2: Sample Steps ---
                        z = meanflow.sample_prediction(
                            model_module, 
                            c_past_val, 
                            sample_steps=20, # 采样步数 (可调, 越多质量越好, 速度越慢)
                            device=accelerator.device
                        ) # (B, T_pred, C, H, W)

                    # --- 新的可视化调用 ---
                    # B=0 的样本, 并转移到 CPU
                    c_past_sample = c_past_val[0].cpu()   # (T_past, C, H, W)
                    z_sample = z[0].cpu()                 # (T_pred, C, H, W)
                    x_future_sample = x_future_val[0].cpu() # (T_pred, C, H, W)
                    
                    # 转换为 List[Tensor(C,H,W)]
                    context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
                    pred_list = [z_sample[t] for t in range(z_sample.shape[0])]
                    target_list = [x_future_sample[t] for t in range(x_future_sample.shape[0])]

                    save_dir = f"images/step_{global_step}"
                    
                    # 调用你提供的函数
                    vis_himawari8_seq_btchw(
                        save_dir=save_dir,
                        context_seq=context_list,
                        pred_seq=pred_list,
                        target_seq=target_list
                    )
                    
                accelerator.wait_for_everyone()
                model.train()
                
    if accelerator.is_main_process:
        ckpt_path = f"checkpoints/step_{global_step}.pt"
        model_module = accelerator.unwrap_model(model)
        accelerator.save(model_module.state_dict(), ckpt_path)