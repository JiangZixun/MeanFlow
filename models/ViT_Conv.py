import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange

class ConvRefiner(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        # 使用分组卷积或者简单的残差块来平滑特征
        padding = kernel_size // 2
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, x):
        # x: (B, T*C, H, W)
        # 使用残差连接，只学习“修正量”，保证基础特征不丢失
        return x + self.refine(x)

# --- 1. 复用 TimestepEmbedder (保持不变) ---
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
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

# --- 2. 核心组件: DiT Block (带 AdaLN 的 Transformer Block) ---
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, Heads, N, Head_Dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class DiTBlock(nn.Module):
    """
    Transformer Block with Adaptive Layer Norm (AdaLN) for time conditioning.
    这是 ViT 用于生成的标准 Block 形式。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        
        # AdaLN modulation: 用于回归 (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c 是时间嵌入 (B, hidden_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention Block
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # MLP Block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of JiT / DiT.
    Regresses the output channels from the hidden vector.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        # AdaLN modulation for final layer (shift, scale)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# --- 3. 主模型: ViT-B (JiT Style) ---
class JiT_Conv(nn.Module):
    def __init__(
        self,
        input_size=(6, 256, 256), # (T, H, W)
        in_channels_c=16,         # C_x + C_y
        out_channels_c=8,         # C_x (Target)
        time_emb_dim=None,        # 兼容参数，如果不传则默认等于 hidden_size
        patch_size=16,
        hidden_size=768,          # ViT-B standard
        depth=12,                 # ViT-B standard
        num_heads=12,             # ViT-B standard
        mlp_ratio=4.0,
        bottleneck_dim=None,
        # Conv Refiner 参数
        refine_kernel_size=3,
    ):
        super().__init__()
        self.time_dim = input_size[0]
        self.in_channels_c = in_channels_c
        self.out_channels_c = out_channels_c
        
        # 1. 计算扁平化后的通道数 (逻辑同 UNet)
        # 输入: (B, T, C_in, H, W) -> 视为 (B, T*C_in, H, W)
        self.in_channels_2d = self.time_dim * self.in_channels_c
        # 输出: (B, T*C_out, H, W) -> 视为 (B, T, C_out, H, W)
        self.out_channels_2d = self.time_dim * self.out_channels_c
        
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # 2. Patch Embedding
        # --- 🟢 修改: Patch Embedding (支持 Bottleneck) ---
        # 论文实现: "replacing it with a pair of bottleneck (yet still linear) layers" [cite: 330]
        # 我们用两个连续的 Conv2d 来等效实现 Linear(Raw -> Bottleneck) -> Linear(Bottleneck -> Hidden)
        if bottleneck_dim is not None:
            print(f"Using Bottleneck Patch Embedding: {bottleneck_dim}")
            self.x_embedder = nn.Sequential(
                # 第一层: 降维 (Raw Patch -> Bottleneck)
                # kernel_size=patch_size, stride=patch_size 实现了 Patchify + Linear Projection
                nn.Conv2d(
                    self.in_channels_2d, 
                    bottleneck_dim, 
                    kernel_size=patch_size, 
                    stride=patch_size, 
                    bias=True
                ),
                # 第二层: 升维 (Bottleneck -> Hidden)
                # 1x1 卷积等效于 Linear 层
                nn.Conv2d(
                    bottleneck_dim,
                    hidden_size,
                    kernel_size=1,
                    bias=True
                )
            )
        else:
            # 标准 ViT (无 Bottleneck)
            self.x_embedder = nn.Conv2d(
                self.in_channels_2d, 
                hidden_size, 
                kernel_size=patch_size, 
                stride=patch_size
            )
        
        # 3. Learnable Positional Embedding
        # 假设输入是 256x256, patch=16 -> 16x16 = 256 个 patch
        num_patches = (input_size[1] // patch_size) * (input_size[2] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # 4. Time Embedding
        # 如果没有指定 time_emb_dim，默认使用 hidden_size 以匹配 AdaLN 输入
        t_dim = time_emb_dim if time_emb_dim else hidden_size
        self.t_embedder = TimestepEmbedder(t_dim)
        self.r_embedder = TimestepEmbedder(t_dim)
        
        # 如果外部传入的 time_emb_dim 不等于 hidden_size，需要一个投影层
        # 因为 AdaLN 需要输入维度等于 hidden_size
        self.t_block_proj = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_dim, hidden_size)
        ) if t_dim != hidden_size else nn.Identity()

        # 5. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # 6. Final Layer (Linear Predict + Unpatchify logic inside)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels_2d)

        self.refiner = ConvRefiner(self.out_channels_2d, kernel_size=refine_kernel_size)

        self.initialize_weights()

    def initialize_weights(self):
        # ... (使用上一条回答中修复过的初始化逻辑) ...
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
        
        self.apply(_init_weights)

        # 🟢 修改: 初始化 Patch Embed
        # 如果是 Sequential (Bottleneck)，需要遍历初始化
        if isinstance(self.x_embedder, nn.Sequential):
            for m in self.x_embedder:
                if isinstance(m, nn.Conv2d):
                    w = m.weight.data
                    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            # 标准层
            w = self.x_embedder.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if self.x_embedder.bias is not None:
                nn.init.constant_(self.x_embedder.bias, 0)
        
        torch.nn.init.normal_(self.pos_embed, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (B, N, patch_size**2 * C)
        return: (B, C, H, W)
        """
        c = self.out_channels_2d
        p = self.patch_size
        # h = w = int(x.shape[1] ** 0.5) 
        # 为了更稳健，我们应该基于输入计算 H, W，或者假设正方形
        # 这里假设是正方形 (256/16 = 16)
        h = w = int(math.sqrt(x.shape[1])) 
        
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None):
        """
        Forward pass matching UNet signature.
        x: (B, T, C_x, H, W) Future Noise
        t: (B,) Timestep
        r: (B,) Timestep
        y: (B, T, C_y, H, W) Condition
        """
        if y is None:
            raise ValueError("Conditional video 'y' cannot be None.")

        # 1. 准备输入数据 (与 UNet 逻辑完全一致)
        # (B, T, C_in, H, W) -> (B, T*C_in, H, W)
        B, T, Cx, H, W = x.shape
        x_inp = torch.cat([x, y], dim=2) # 沿通道 C 拼接
        x_2d = x_inp.reshape(B, T * (self.in_channels_c), H, W)

        # 2. Patchify & Embedding
        # (B, C, H, W) -> (B, Hidden, H/p, W/p) -> (B, Hidden, N) -> (B, N, Hidden)
        x = self.x_embedder(x_2d)
        x = rearrange(x, 'b c h w -> b (h w) c') # Flatten spatial dimensions
        
        # Add Positional Embedding
        x = x + self.pos_embed

        # 3. Time Embedding Calculation
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(r)
        time_emb = t_emb + r_emb
        
        # 投影以匹配 hidden_size (如果是 DiT 结构，这很重要)
        c = self.t_block_proj(time_emb)

        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)

        # 5. Final Projection (Linear Predict)
        # (B, N, Hidden) -> (B, N, patch_size^2 * C_out)
        x = self.final_layer(x, c)

        # 6. Unpatchify & Reshape (恢复空间和时间维度)
        # (B, N, P^2*C) -> (B, T*C_out, H, W)
        logits_2d = self.unpatchify(x)
        logits_2d = self.refiner(logits_2d) # 使用卷积 Refiner 进行特征修正
        
        # 7. 拆分时间和通道
        # (B, T*C_out, H, W) -> (B, T, C_out, H, W)
        logits_3d = logits_2d.reshape(B, self.time_dim, self.out_channels_c, H, W)

        return logits_3d