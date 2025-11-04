import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

# --- U-Net 仍然需要 TimestepEmbedder ---
class TimestepEmbedder(nn.Module):
    """
    Standard sinusoidal time embedding module
    """
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t*1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

# --- 2D U-Net 卷积块 (带时间注入) ---
class DoubleConv2D(nn.Module):
    """(Conv2D -> GroupNorm -> SiLU) * 2 + Time Embedding"""
    def __init__(self, in_channels, out_channels, mid_channels=None, time_emb_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim else None

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        if t_emb is not None and self.time_mlp is not None:
            # 投射时间嵌入并增加 (B, C, 1, 1) 维度用于广播
            t = self.time_mlp(t_emb)
            x = x + t.unsqueeze(-1).unsqueeze(-1)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x

# --- 2D U-Net 下采样块 ---
class DownBlock2D(nn.Module):
    """Downscaling with MaxPool then DoubleConv2D"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv2D(in_channels, out_channels, time_emb_dim=time_emb_dim)

    def forward(self, x, t_emb=None):
        x = self.pool(x)
        x = self.conv(x, t_emb)
        return x

# --- 2D U-Net 上采样块 (带 Skip Connection) ---
class UpBlock2D(nn.Module):
    """Upscaling then DoubleConv2D"""
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        # 上采样，输出通道为 out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 卷积的输入 = 上采样通道 + skip 通道
        self.conv = DoubleConv2D(out_channels + skip_channels, out_channels, time_emb_dim=time_emb_dim)

    def forward(self, x1, x2, t_emb=None):
        x1 = self.up(x1)
        
        # 处理可能的维度不匹配 (e.g., 63 vs 64)
        diffH = x2.size()[2] - x1.size()[2]
        diffW = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t_emb)


# --- 核心类：UNet (2D U-Net 实现) ---
class UNet(nn.Module):
    def __init__(
        self,
        input_size=(6, 256, 256), # (T, H, W)
        in_channels=16,         # C_x + C_y
        out_channels=8,         # C_x (噪声)
        dim=64,                # 用作时间嵌入的维度
    ):
        super().__init__()
        self.time_dim = input_size[0] # T=6
        self.in_channels_c = in_channels # C_in = 16
        self.out_channels_c = out_channels # C_out = 8
        time_emb_dim = dim # 重用 'dim' 作为时间嵌入维度

        # 1. 计算 2D U-Net 的输入和输出通道
        # (B, T, C_in, H, W) -> (B, T*C_in, H, W)
        self.in_channels_2d = self.time_dim * self.in_channels_c # 6 * 16 = 96
        
        # (B, T*C_out, H, W) -> (B, T, C_out, H, W)
        self.out_channels_2d = self.time_dim * self.out_channels_c # 6 * 8 = 48
        
        # 2. 时间嵌入
        self.t_embedder = TimestepEmbedder(time_emb_dim)
        self.r_embedder = TimestepEmbedder(time_emb_dim)

        # 3. 2D U-Net 架构 (4 层下采样)
        base_c = 64
        
        # --- 下采样路径 ---
        self.inc = DoubleConv2D(self.in_channels_2d, base_c, time_emb_dim=time_emb_dim)
        # (B, 64, 256, 256)
        self.down1 = DownBlock2D(base_c, base_c*2, time_emb_dim=time_emb_dim)
        # (B, 128, 128, 128)
        self.down2 = DownBlock2D(base_c*2, base_c*4, time_emb_dim=time_emb_dim)
        # (B, 256, 64, 64)
        self.down3 = DownBlock2D(base_c*4, base_c*8, time_emb_dim=time_emb_dim)
        # (B, 512, 32, 32)
        self.down4 = DownBlock2D(base_c*8, base_c*8, time_emb_dim=time_emb_dim)
        # (B, 512, 16, 16)
        
        # --- Bottleneck ---
        self.bot = DoubleConv2D(base_c*8, base_c*16, time_emb_dim=time_emb_dim)
        # (B, 1024, 16, 16)

        # --- 上采样路径 ---
        # (in_channels, skip_channels, out_channels, ...)
        self.up1 = UpBlock2D(1024, 512, 512, time_emb_dim=time_emb_dim)
        # (B, 512, 32, 32)
        self.up2 = UpBlock2D(512, 512, 256, time_emb_dim=time_emb_dim)
        # (B, 256, 64, 64)
        self.up3 = UpBlock2D(256, 256, 128, time_emb_dim=time_emb_dim)
        # (B, 128, 128, 128)
        
        # --- 🔴 修正点 1 ---
        # skip_channels 应该是 64 (来自 x1), 而不是 128 (来自 x2)
        self.up4 = UpBlock2D(128, 64, 64, time_emb_dim=time_emb_dim)
        # (B, 64, 256, 256)
        
        self.outc = nn.Conv2d(base_c, self.out_channels_2d, kernel_size=1)
        # (B, 48, 256, 256)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Kaiming He 初始化
        def _kaiming_init(module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        self.apply(_kaiming_init)
        
        # 初始化时间 MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)

        # 零初始化最后的输出卷积
        nn.init.constant_(self.outc.weight, 0)
        nn.init.constant_(self.outc.bias, 0)

    def forward(self, x, t, r, y=None):
        """
        Forward pass of "Pseudo-3D" 2D U-Net.
        x: (N, T, C_x, H, W) tensor of noised future video
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of diffusion timesteps
        y: (N, T, C_y, H, W) tensor of past condition video
        """
        if y is None:
            raise ValueError("Conditional video 'y' (c_past) cannot be None for this model.")
        
        B, T, C_x, H, W = x.shape
        _, _, C_y, _, _ = y.shape
        
        # 1. 计算时间嵌入
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(r)
        time_emb = t_emb + r_emb

        # 2. 拼接 x 和 y -> (B, T, C_x+C_y, H, W)
        x_inp = torch.cat([x, y], dim=2) # dim 2 是 C 维度

        # 3. --- 关键：执行 T+C 合并 ---
        # (B, T, C_in, H, W) -> (B, T*C_in, H, W)
        x_2d = x_inp.reshape(B, T * (C_x + C_y), H, W)

        # 4. 2D U-Net Forward
        x1 = self.inc(x_2d, time_emb)     # (B, 64, 256, 256)
        x2 = self.down1(x1, time_emb)    # (B, 128, 128, 128)
        x3 = self.down2(x2, time_emb)    # (B, 256, 64, 64)
        x4 = self.down3(x3, time_emb)    # (B, 512, 32, 32)
        x5 = self.down4(x4, time_emb)    # (B, 512, 16, 16)
        
        x_bot = self.bot(x5, time_emb)   # (B, 1024, 16, 16)
        
        x_up = self.up1(x_bot, x5, time_emb) # (1024->512) + skip(512) -> (B, 512, 32, 32)
        x_up = self.up2(x_up, x4, time_emb) # (512->256)  + skip(512) -> (B, 256, 64, 64)
        x_up = self.up3(x_up, x3, time_emb) # (256->128)  + skip(256) -> (B, 128, 128, 128)
        
        # --- 🔴 修正点 2 ---
        # skip connection 应该使用 x1, 而不是 x2
        x_up = self.up4(x_up, x1, time_emb) # (128->64)   + skip(64) -> (B, 64, 256, 256)
        
        logits_2d = self.outc(x_up)      # -> (B, 48, 256, 256)
        
        # 5. --- 关键：执行 T+C 拆分 ---
        # (B, T*C_out, H, W) -> (B, T, C_out, H, W)
        logits_3d = logits_2d.reshape(B, self.time_dim, self.out_channels_c, H, W)
        
        return logits_3d