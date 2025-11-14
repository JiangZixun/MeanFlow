# evaluation/psd_metric.py
import torch
import torchmetrics

def _get_1d_psd_torch(image_2d: torch.Tensor, k_axis_len: int, k_axis_ref: torch.Tensor):
    """(辅助函数) 在PyTorch中计算单个2D图像的1D PSD。"""
    if image_2d.device != k_axis_ref.device:
        k_axis_ref = k_axis_ref.to(image_2d.device)
        
    H, W = image_2d.shape
    
    # 确保输入是 float, FFT 需要
    if not image_2d.is_floating_point():
        image_2d = image_2d.to(torch.float32)
        
    f_transform = torch.fft.fft2(image_2d)
    f_transform_shifted = torch.fft.fftshift(f_transform)
    power_spectrum_2d = torch.abs(f_transform_shifted)**2

    # 创建 float 类型的坐标
    y_coords = torch.arange(H, device=image_2d.device, dtype=torch.float32)
    x_coords = torch.arange(W, device=image_2d.device, dtype=torch.float32)
    y, x = torch.meshgrid(y_coords, x_coords, indexing='ij') 

    center_y, center_x = H // 2, W // 2
    radial_distance = torch.hypot(x - center_x, y - center_y)

    # k_bins 是 int64 (Long)
    k_bins = torch.round(radial_distance).long()
    
    # 确保 k_bins 不会超出我们预期的 k_len 范围
    k_bins[k_bins >= k_axis_len] = k_axis_len - 1

    k_bins_flat = k_bins.flatten()
    power_spectrum_flat = power_spectrum_2d.flatten()
    
    # total_power_per_k 是 float32
    total_power_per_k = torch.zeros(k_axis_len, device=image_2d.device, dtype=torch.float32)
    total_power_per_k.scatter_add_(0, k_bins_flat, power_spectrum_flat)
    
    # --- 🔴 修复在这里 ---
    # 1. 'self' (count_per_k) 必须是 float32，以便与 total_power_per_k 兼容
    count_per_k = torch.zeros(k_axis_len, device=image_2d.device, dtype=torch.float32)
    # 2. 'src' (我们加上的值) 也必须是 float32
    ones_to_add = torch.ones_like(k_bins_flat, dtype=torch.float32)
    # 3. 'index' (k_bins_flat) 必须是 long (它已经是了)
    count_per_k.scatter_add_(0, k_bins_flat, ones_to_add)
    # --- 修复结束 ---

    count_per_k[count_per_k == 0] = 1 
    mean_power_per_k = total_power_per_k / count_per_k
    
    return mean_power_per_k


class PSDAverageMetric(torchmetrics.Metric):
    """
    一个 DDP 安全、不爆内存的流式 PSD 指标。
    用法与 FrechetVideoDistance 完全相同。
    """
    full_state_update = False
    
    def __init__(self, H=256, W=256, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.k_len = min(H // 2, W // 2)
        
        self.add_state("total_power", default=torch.zeros(self.k_len), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.register_buffer("k_axis", torch.arange(0, self.k_len, dtype=torch.float32), persistent=False)

    def update(self, tensor_5d: torch.Tensor):
        # (B, T, C, H, W) -> (B*T*C, H, W)
        B, T, C, H, W = tensor_5d.shape
        images_2d_batch = tensor_5d.reshape(B*T*C, H, W)
        
        for i in range(images_2d_batch.shape[0]):
            psd_1d = _get_1d_psd_torch(images_2d_batch[i], self.k_len, self.k_axis)
            self.total_power += psd_1d
            self.total_count += 1.0

    def compute(self):
        if self.total_count == 0:
            return self.k_axis, torch.zeros_like(self.total_power)
        return self.k_axis, self.total_power / self.total_count