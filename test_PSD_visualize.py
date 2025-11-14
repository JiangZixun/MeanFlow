import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt

from dataset_btchw import Xiaoshan_6steps_30min_Dataset

# --- 步骤 1: 2D PSD 计算函数 (无需更改) ---
def get_1d_psd(image):
    """
    计算单个 2D 图像的 1D 功率谱密度（通过径向平均）。
    假设 image 是一个 2D NumPy 数组 [H, W]。
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D")
    
    H, W = image.shape
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    power_spectrum_2d = np.abs(f_transform_shifted)**2

    y, x = np.indices((H, W))
    center_y, center_x = H // 2, W // 2
    radial_distance = np.hypot(x - center_x, y - center_y)

    k_bins = np.round(radial_distance).astype(int)
    max_k = np.max(k_bins)
    
    total_power_per_k = np.bincount(k_bins.ravel(), weights=power_spectrum_2d.ravel())
    count_per_k = np.bincount(k_bins.ravel())
    
    count_per_k[count_per_k == 0] = 1 
    mean_power_per_k = total_power_per_k / count_per_k
    
    k_indices = np.arange(0, max_k + 1)
    nyquist_limit = min(H // 2, W // 2)
    
    return k_indices[:nyquist_limit], mean_power_per_k[:nyquist_limit]

# --- 步骤 2: 5D 张量处理函数 (无需更改) ---
# (我们仍用它来计算总平均值)
def compute_psd_stats(tensor_5d):
    """
    为 5D (B, T, C, H, W) 张量计算 PSD 统计数据。
    """
    if not isinstance(tensor_5d, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
        
    tensor_5d_np = tensor_5d.detach().cpu().numpy()
    B, T, C, H, W = tensor_5d_np.shape
    
    k_axis = None 
    
    # [T, C, B, k_len]
    all_psds = np.zeros((T, C, B, min(H//2, W//2)))
    
    for t in range(T):
        for c in range(C):
            for b in range(B):
                image_2d = tensor_5d_np[b, t, c, :, :]
                k, psd_1d = get_1d_psd(image_2d)
                
                if k_axis is None:
                    k_axis = k
                
                if len(k) == all_psds.shape[3]:
                    all_psds[t, c, b, :] = psd_1d
    
    results = {}
    results['k_axis'] = k_axis
    # (T, C, B) 都合并平均
    results['total_avg'] = np.mean(all_psds, axis=(0, 1, 2))
    
    return results

# --- 步骤 3: (新增) 量化PSD差距的函数 ---
def calculate_psd_error_metrics(psd_gt, psd_pred, k_axis, high_k_split=0.25):
    """
    计算两条 PSD 曲线之间的量化误差。
    在 Log 空间中计算 MAE。
    
    Args:
        psd_gt (np.array): 真实的 1D PSD 曲线 P(k)
        psd_pred (np.array): 预测的 1D PSD 曲线 P(k)
        k_axis (np.array): 对应的波数 k
        high_k_split (float): 定义 "高频" 的 k 轴百分比 (例如 0.25 表示后75%为高频)
    
    Returns:
        dict: 包含量化指标的字典
    """
    # 添加一个极小值 (epsilon) 来防止 log(0)
    epsilon = 1e-10
    log_psd_gt = np.log(psd_gt + epsilon)
    log_psd_pred = np.log(psd_pred + epsilon)
    
    # 1. 全局对数 MAE
    log_mae_total = np.mean(np.abs(log_psd_gt - log_psd_pred))
    
    # 2. 高频对数 MAE (我们真正关心的)
    
    # 忽略 k=0 (直流分量)，它通常没有意义
    valid_indices = k_axis > 0
    k_axis_valid = k_axis[valid_indices]
    log_psd_gt_valid = log_psd_gt[valid_indices]
    log_psd_pred_valid = log_psd_pred[valid_indices]
    
    # 确定高频的起始索引
    split_index = int(len(k_axis_valid) * high_k_split)
    
    log_psd_gt_high_k = log_psd_gt_valid[split_index:]
    log_psd_pred_high_k = log_psd_pred_valid[split_index:]
    
    log_mae_high_k = np.mean(np.abs(log_psd_gt_high_k - log_psd_pred_high_k))
    
    return {
        "Total_Log_MAE": log_mae_total,
        "High_Freq_Log_MAE": log_mae_high_k
    }

# --- 步骤 4: 如何使用 (TCB 合并 + 量化) ---

# 1. 模拟您的输入和输出 (B=4, T=6, C=8, H=256, W=256)
B, T, C, H, W = 1, 6, 8, 256, 256
dataset = Xiaoshan_6steps_30min_Dataset(
    data_path="/mnt/data1/Dataset/Himawari8/train/xiaoshan_6steps_30min_Data_day.h5",
    json_path="/mnt/data1/Dataset/Himawari8/train/xiaoshan_6steps_30min_Data_day_all.json",
    dataset_prefix='2022',
    train_ratio=0.9,
    random_flip=0.0,
    split='val',
)
gt_tensor = dataset[100][1].unsqueeze(0)
# 模拟 "模糊" 的预测数据
pred_tensor = dataset[0][0].unsqueeze(0)

# 2. 计算两者的 PSD 统计
print("Calculating PSD for Ground Truth...")
gt_psd_stats = compute_psd_stats(gt_tensor)
print("Calculating PSD for Prediction...")
pred_psd_stats = compute_psd_stats(pred_tensor)

# 3. 提取 "Total Average" 结果
k_axis = gt_psd_stats['k_axis']
psd_gt_total = gt_psd_stats['total_avg']
psd_pred_total = pred_psd_stats['total_avg']

# 4. (回答请求1) 绘制一张总平均图
print("Plotting...")
plt.figure(figsize=(10, 6))
plt.loglog(k_axis, psd_gt_total, label='Ground Truth (Total Avg)', color='blue')
plt.loglog(k_axis, psd_pred_total, label='Prediction (Total Avg)', color='red', linestyle='--')
plt.xlabel('Wavenumber (k) - (Log Scale)')
plt.ylabel('Power P(k) - (Log Scale)')
plt.title('Global Average Power Spectral Density (PSD) Comparison')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

# 5. (回答请求2) 计算量化指标
metrics = calculate_psd_error_metrics(psd_gt_total, psd_pred_total, k_axis)

print("\n--- Quantitative PSD Metrics ---")
print(f"  Total Log-MAE: {metrics['Total_Log_MAE']:.4f}")
print(f"  High-Freq Log-MAE: {metrics['High_Freq_Log_MAE']:.4f}")
print("--------------------------------")
print("\n(目标：两个指标都应尽可能接近 0。高频指标对'模糊度'更敏感。)")