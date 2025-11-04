import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp, Attention
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast


# --- 新增：3D Patch Embedding ---
class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=(6, 256, 256), patch_size=(2, 16, 16), in_chans=16, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert T == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({T}x{H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]}x{self.img_size[2]})."
        
        # 3D 卷积: (B, C, T, H, W) -> (B, D, T', H', W')
        x = self.proj(x)
        # 展平: (B, D, T', H', W') -> (B, D, N) N=T'*H'*W'
        x = x.flatten(2)
        # 交换维度: (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)
        return x


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
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


# --- LabelEmbedder 已被移除，不再需要 ---


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c): # c 现在是时间嵌入 (t+r)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size_t, patch_size_h, patch_size_w, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        # 线性层输出通道数 * patch体积
        self.linear = nn.Linear(dim, patch_size_t * patch_size_h * patch_size_w * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MFDiT(nn.Module):
    def __init__(
        self,
        input_size=(6, 256, 256),  # (T, H, W)
        patch_size=(2, 16, 16), # (pt, ph, pw)
        in_channels=16,         # C_x + C_y
        out_channels=8,         # C_x (噪声)
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=None, # 已废弃，但保留以兼容
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size_t = input_size[0]
        self.input_size_h = input_size[1]
        self.input_size_w = input_size[2]
        self.patch_size_t = patch_size[0]
        self.patch_size_h = patch_size[1]
        self.patch_size_w = patch_size[2]
        
        # 1. 使用 3D Patch Embedding
        self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)

        # 2. 移除 LabelEmbedder (y_embedder)
        self.use_cond = False # 不再使用类别条件

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        # 3. FinalLayer 需要知道 patch 体积和输出通道
        self.final_layer = FinalLayer(dim, self.patch_size_t, self.patch_size_h, self.patch_size_w, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 4. 初始化 3D 位置嵌入
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                            self.input_size_t // self.patch_size_t,
                                            self.input_size_h // self.patch_size_h, 
                                            self.input_size_w // self.patch_size_w)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 5. 移除 y_embedder 初始化
        # if self.y_embedder is not None: ...

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T*H*W, patch_t * patch_h * patch_w * C_out)
        imgs: (N, C_out, T, H, W)
        """
        c = self.out_channels
        pt, ph, pw = self.patch_size_t, self.patch_size_h, self.patch_size_w
        t, h, w = self.input_size_t // pt, self.input_size_h // ph, self.input_size_w // pw
        assert t * h * w == x.shape[1]

        # (N, T*H*W, pt*ph*pw*C) -> (N, t, h, w, pt, ph, pw, c)
        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        
        # --- THIS IS THE CORRECTED LINE ---
        # (N, t, h, w, pt, ph, pw, c) -> (N, c, t, pt, h, ph, w, pw)
        # Dims: 0, 1, 2, 3,  4,  5,  6, 7
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6) 
        
        # (N, c, T, H, W)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def forward(self, x, t, r, y=None):
        """
        Forward pass of 3D DiT.
        x: (N, T, C_x, H, W) tensor of noised future video
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of diffusion timesteps
        y: (N, T, C_y, H, W) tensor of past condition video
        """
        if y is None:
            raise ValueError("Conditional video 'y' (c_past) cannot be None for this model.")
            
        # 7. 假设输入格式为 (B, T, C, H, W)，但 Conv3D 需要 (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C_x, T, H, W)
        y = y.permute(0, 2, 1, 3, 4)  # (B, C_y, T, H, W)

        # 8. 通道拼接 (Channel Concatenation)
        x_inp = torch.cat([x, y], dim=1) # (B, C_x+C_y, T, H, W)

        x = self.x_embedder(x_inp) + self.pos_embed  # (N, N_patches, D)

        t = self.t_embedder(t)                   # (N, D)
        r = self.r_embedder(r)
        c = t + r                                # (N, D) (c 现在只是时间条件)

        # 9. 移除旧的 y_embedder 逻辑
        # if self.use_cond: ...

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, N_patches, D)

        x = self.final_layer(x, c)               # (N, N_patches, pt*ph*pw*C_out)
        x = self.unpatchify(x)                   # (N, C_out, T, H, W)
        
        # 10. 转换回 (B, T, C, H, W) 格式以匹配 loss 计算
        x = x.permute(0, 2, 1, 3, 4)
        return x


# --- 新增：3D 位置嵌入 ---
def get_3d_sincos_pos_embed(embed_dim, grid_t, grid_h, grid_w):
    """
    grid_t, grid_h, grid_w: time, height, width of the patch grid
    return:
    pos_embed: [grid_t*grid_h*grid_w, embed_dim]
    """
    assert embed_dim % 3 == 0, "Embedding dimension must be divisible by 3 for 3D SinCos"
    
    # 分配维度
    dim_t = embed_dim // 3
    dim_h = embed_dim // 3
    dim_w = embed_dim - dim_t - dim_h # 确保总和正确

    # 1D 位置
    pos_t = np.arange(grid_t, dtype=np.float32)
    pos_h = np.arange(grid_h, dtype=np.float32)
    pos_w = np.arange(grid_w, dtype=np.float32)

    # 1D SinCos 嵌入
    emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, pos_t) # (grid_t, dim_t)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, pos_h) # (grid_h, dim_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, pos_w) # (grid_w, dim_w)

    # 广播以创建 3D 网格
    emb_t = np.repeat(emb_t, grid_h * grid_w, axis=0)
    emb_h = np.tile(np.repeat(emb_h, grid_w, axis=0), (grid_t, 1))
    emb_w = np.tile(emb_w, (grid_t * grid_h, 1))

    # 拼接
    pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1) # (T*H*W, D)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb