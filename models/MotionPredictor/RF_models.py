import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalRefinementNet(nn.Module):
    def __init__(self, channels=8, hidden_dim=32):
        super().__init__()
        # 空间平滑
        self.spatial_smooth = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),  # 较大卷积核平滑
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1)
        )
        
        # 时间平滑 (1D卷积在时间维度)
        self.temporal_smooth = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 3, padding=1)
        )
        
        # 自适应权重
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        original = x
        
        # 空间平滑
        x_spatial = x.view(B*T, C, H, W)
        x_spatial = self.spatial_smooth(x_spatial)
        x_spatial = x_spatial.view(B, T, C, H, W)
        
        # 时间平滑
        x_temporal = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, C, H, W, T]
        x_temporal = x_temporal.view(B*C*H*W, T)
        x_temporal = self.temporal_smooth(x_temporal.unsqueeze(1)).squeeze(1)
        x_temporal = x_temporal.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)
        
        # 自适应融合
        weight = self.weight_net(x.view(B*T, C, H, W)).view(B, T, 1, H, W)
        refined = weight * x_spatial + (1 - weight) * x_temporal
        
        # 残差连接
        output = refined + original
        
        return output

class LightweightRefinementNet(nn.Module):
    def __init__(self, channels=8):
        super(LightweightRefinementNet, self).__init__()
        # 只用一个深度可分离卷积
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 可学习的融合权重
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        shape = x.shape
        original = x
        
        if len(shape) == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
        
        # 深度可分离卷积平滑
        x = self.depthwise(x)
        x = self.pointwise(x)

        if len(shape) == 5:
            x = x.view(B, T, C, H, W)
        
        # 可学习的残差连接
        return original + self.alpha * (x - original)


############################### WarpSharpen Module ###############################
# ESDR-Lite
class SobelGrad(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        gy = gx.t()
        self.register_buffer('kx', gx.view(1,1,3,3))
        self.register_buffer('ky', gy.view(1,1,3,3))
        self.channels = channels

    def forward(self, x):
        # x: (B,C,H,W); 分通道做深度可分卷积
        kx = self.kx.repeat(self.channels,1,1,1)
        ky = self.ky.repeat(self.channels,1,1,1)
        grad_x = F.conv2d(x, kx, padding=1, groups=self.channels)
        grad_y = F.conv2d(x, ky, padding=1, groups=self.channels)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return grad_mag


class ResBlock(nn.Module):
    def __init__(self, ch, expansion=4, res_scale=0.1):
        super().__init__()
        mid = ch * expansion
        self.body = nn.Sequential(
            nn.Conv2d(ch, mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 3, 1, 1),
        )
        self.res_scale = res_scale
    def forward(self, x):
        return x + self.body(x) * self.res_scale


class ResidualSRHead(nn.Module):
    """
    轻量锐化头：输入 x_warp 与 x_t 的梯度，输出要加回去的细节残差 Δ
    - in_ch = C(原图通道) + C(梯度通道)，默认为 concat
    """
    def __init__(self, in_ch, base_ch=64, num_blocks=5, res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(base_ch, expansion=4, res_scale=res_scale) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_ch, in_ch//2, 3, 1, 1)  # 输出残差 Δ 与图像通道 C 对齐
    def forward(self, x_warp, grad_xt):
        x = torch.cat([x_warp, grad_xt], dim=1)
        feat = self.head(x)
        feat = self.body(feat)
        delta = self.tail(feat)
        return delta


class WarpSharpen(nn.Module):
    """
    x_t --(Sobel)--> grad_xt
    x_warp --(SRHead with grad_xt)--> Δ
    x_hat = x_warp + Δ
    """
    def __init__(self, channels, base_ch=64, num_blocks=5, res_scale=0.1):
        super().__init__()
        self.sobel = SobelGrad(channels)
        self.sr = ResidualSRHead(in_ch=channels*2, base_ch=base_ch, num_blocks=num_blocks, res_scale=res_scale)

    def forward(self, x_t, x_warp):
        grad_xt = self.sobel(x_t)
        delta = self.sr(x_warp, grad_xt)
        x_hat = x_warp + delta
        return x_hat #, delta, grad_xt

from typing import Optional, Tuple, Union, Dict
try:
    from ...taming.autoencoder_kl import AutoencoderKL
except Exception:
    print("Warning: Cannot import AutoencoderKL from taming package. Make sure taming-transformers is installed.")

class WarpSharpen_VAE(nn.Module):
    """
    x_t --(Sobel)--> grad_xt
    x_warp --(SRHead with grad_xt)--> Δ
    x_hat = x_warp + Δ
    """
    def __init__(
            self, 
            in_channels: int = 8,
            out_channels: int = 8,
            down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,):
        super().__init__()
        self.sobel = SobelGrad(in_channels)
        self.sr = AutoencoderKL(
            in_channels=in_channels*2,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
        )
    
    def forward(self, x_t, x_warp):
        grad_xt = self.sobel(x_t)
        x = torch.cat([x_warp, grad_xt], dim=1)
        delta = self.sr(x, sample_posterior=True)
        x_hat = x_warp + delta
        return x_hat #, delta, grad_xt