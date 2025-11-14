# evaluation/psd_tools.py
import numpy as np
import matplotlib.pyplot as plt
import os

def _get_1d_psd_numpy(image_2d):
    """(辅助函数) 计算单个 2D 图像的 1D 功率谱 (Numpy)。"""
    if image_2d.ndim != 2:
        image_2d = image_2d.squeeze() # 尝试处理 (H, W, 1)
        if image_2d.ndim != 2:
            raise ValueError("Input image must be 2D")
    
    H, W = image_2d.shape
    
    f_transform = np.fft.fft2(image_2d)
    f_transform_shifted = np.fft.fftshift(f_transform)
    power_spectrum_2d = np.abs(f_transform_shifted)**2

    y, x = np.indices((H, W))
    center_y, center_x = H // 2, W // 2
    radial_distance = np.hypot(x - center_x, y - center_y)

    k_bins = np.round(radial_distance).astype(int)
    max_k = np.max(k_bins)
    
    total_power_per_k = np.bincount(k_bins.ravel(), weights=power_spectrum_2d.ravel(), minlength=max_k + 1)
    count_per_k = np.bincount(k_bins.ravel(), minlength=max_k + 1)
    
    count_per_k[count_per_k == 0] = 1 
    mean_power_per_k = total_power_per_k / count_per_k
    
    k_indices = np.arange(0, max_k + 1)
    nyquist_limit = min(H // 2, W // 2)
    
    return k_indices[:nyquist_limit], mean_power_per_k[:nyquist_limit]

def plot_sample_psd(pred_4d, rfdpic_4d, gt_4d, save_dir):
    """
    (回答您的可视化要求)
    一个函数解决：为单个样本 (T, C, H, W) 计算并绘制PSD对比图。
    
    Args:
        pred_4d (np.array): (T, C, H, W) MeanFlow 预测
        rfdpic_4d (np.array): (T, C, H, W) RFDPIC 预测
        gt_4d (np.array): (T, C, H, W) Ground Truth
        save_dir (str): 保存图像的目录
    """
    samples = {
        "Prediction (MeanFlow)": (pred_4d, "red", "--"),
        "Prediction (RFDPIC)": (rfdpic_4d, "green", ":"),
        "Ground Truth": (gt_4d, "blue", "-")
    }
    
    k_axis = None
    plt.figure(figsize=(10, 6))

    try:
        for name, (data_4d, color, linestyle) in samples.items():
            T, C, H, W = data_4d.shape
            all_psds_for_sample = []
            
            for t in range(T):
                for c in range(C):
                    k, psd_1d = _get_1d_psd_numpy(data_4d[t, c])
                    if k_axis is None: k_axis = k
                    if len(k) == len(k_axis): all_psds_for_sample.append(psd_1d)
            
            if all_psds_for_sample:
                avg_psd = np.mean(all_psds_for_sample, axis=0)
                plt.loglog(k_axis, avg_psd, label=name, color=color, linestyle=linestyle, linewidth=2)
        
        plt.xlabel('Wavenumber (k) - (Log Scale)')
        plt.ylabel('Power P(k) - (Log Scale)')
        plt.title(f'Sample Average PSD Comparison (T={T}, C={C})')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        save_path = os.path.join(save_dir, "sample_psd_comparison.png")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Failed to generate sample PSD plot: {e}")
        plt.close()


def calculate_psd_error_metrics(psd_gt: np.ndarray, psd_pred: np.ndarray, k_axis: np.ndarray, 
                                high_k_splits: list = [0.50, 0.75, 0.90, 0.95]):
    """
    (回答您的指标要求)
    计算两条 PSD 曲线之间的量化误差 (Log-MAE)。
    """
    epsilon = 1e-10
    log_psd_gt = np.log(psd_gt + epsilon)
    log_psd_pred = np.log(psd_pred + epsilon)
    
    valid_indices = k_axis > 0 # 忽略 k=0
    k_axis_valid = k_axis[valid_indices]
    log_psd_gt_valid = log_psd_gt[valid_indices]
    log_psd_pred_valid = log_psd_pred[valid_indices]

    metrics = {}
    metrics["Total_Log_MAE"] = np.mean(np.abs(log_psd_gt_valid - log_psd_pred_valid))
    
    for split_ratio in high_k_splits:
        split_index = int(len(k_axis_valid) * split_ratio)
        log_mae_high_k = np.mean(np.abs(log_psd_gt_valid[split_index:] - log_psd_pred_valid[split_index:]))
        
        # --- 🔴 修复在这里 ---
        # 显式使用 round() 来避免浮点数精度问题
        percent = int(round((1.0 - split_ratio) * 100))
        # --- 修复结束 ---
        
        metrics[f"High_Freq_Log_MAE_top_{percent}%"] = log_mae_high_k

    return metrics