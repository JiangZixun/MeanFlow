import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from einops import rearrange

class AttentionBlock(nn.Module):
    """轻量化注意力模块"""
    def __init__(self, in_channels, gate_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    """带残差连接的轻量化卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，需要1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out

class LightweightUNet(nn.Module):
    """
    轻量化Attention-UNet
    输入: 过去6帧 [B, t_in*C, H, W]
    输出: 6个速度场 [B, t_out, C, H, W, 2]
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32):
        super().__init__()
        
        # 输入是t_in帧拼接
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # 底层
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # 输出层：预测t_out个时间步的速度场
        self.velocity_head = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)  # t_out个时间步 × 通道数量 × 2个速度分量
        
    def forward(self, x):
        """
        Args:
            x: [B, t_in, C, H, W] 拼接的t_in帧
        Returns:
            velocity_fields: [B, t_out, H, W, 2] t_out个速度场
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # 底层
        b = self.bottleneck(self.pool4(e4))
        
        # 解码器 + 注意力
        d4 = self.up4(b)
        e4_att = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))
        
        # 输出速度场
        velocity_fields = self.velocity_head(d1)  # [B, 12, H, W]
        
        # 重塑为 [B, 6, C, H, W, 2]
        B, _, H, W = velocity_fields.shape
        velocity_fields = velocity_fields.view(B, self.out_frames, self.frame_channels, 2, H, W).permute(0, 1, 2, 4, 5, 3)
        
        return velocity_fields

class EarthformerUNet(nn.Module):
    
    def __init__(self, model_cfg):
        super().__init__()

        # Model Configs
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.model = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )
    
    def forward(self, x):
        """
        Parameters
        ----------
        x
            Shape (B, T, C, H, W)

        Returns
        -------
        out
            The output Shape (B, T_out, H, W, C_out)
        """
        x = self.model(x)
        x = rearrange(x, 'b t c h w -> b t h w c')
        return x
    
class InceptionUNet(nn.Module):
    """
    基于Inception模块的UNet
    输入: 过去6帧 [B, t_in*C, H, W]
    输出: 6个速度场 [B, t_out, H, W, 2]
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32, incep_ker=[3,5,7,11], groups=8):
        super().__init__()
        
        # 输入是t_in帧拼接
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        
        # 编码器 - 使用Inception模块
        self.enc1 = Inception(in_channels, base_channels//2, base_channels, incep_ker, groups)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = Inception(base_channels, base_channels, base_channels * 2, incep_ker, groups)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = Inception(base_channels * 2, base_channels * 2, base_channels * 4, incep_ker, groups)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = Inception(base_channels * 4, base_channels * 4, base_channels * 8, incep_ker, groups)
        self.pool4 = nn.MaxPool2d(2)
        
        # 底层 - 最深层的Inception模块
        self.bottleneck = Inception(base_channels * 8, base_channels * 8, base_channels * 16, incep_ker, groups)
        
        # 解码器 - 上采样 + Inception模块
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4 = Inception(base_channels * 16, base_channels * 8, base_channels * 8, incep_ker, groups)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = Inception(base_channels * 8, base_channels * 4, base_channels * 4, incep_ker, groups)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = Inception(base_channels * 4, base_channels * 2, base_channels * 2, incep_ker, groups)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = Inception(base_channels * 2, base_channels, base_channels, incep_ker, groups)
        
        # 输出头：使用1x1卷积生成最终的速度场
        self.velocity_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, out_frames * 2, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, t_in, C, H, W] 输入的t_in帧
        Returns:
            velocity_fields: [B, t_out, H, W, 2] t_out个速度场
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        
        # 编码器路径
        e1 = self.enc1(x)           # [B, base_channels, H, W]
        e2 = self.enc2(self.pool1(e1))     # [B, base_channels*2, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))     # [B, base_channels*4, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))     # [B, base_channels*8, H/8, W/8]
        
        # 底部瓶颈层
        b = self.bottleneck(self.pool4(e4))  # [B, base_channels*16, H/16, W/16]
        
        # 解码器路径 + 跳跃连接 + 注意力机制
        d4 = self.up4(b)                    # [B, base_channels*8, H/8, W/8]
        e4_att = self.att4(e4, d4)          # 注意力增强的编码器特征
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))  # [B, base_channels*8, H/8, W/8]
        
        d3 = self.up3(d4)                   # [B, base_channels*4, H/4, W/4]
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))  # [B, base_channels*4, H/4, W/4]
        
        d2 = self.up2(d3)                   # [B, base_channels*2, H/2, W/2]
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))  # [B, base_channels*2, H/2, W/2]
        
        d1 = self.up1(d2)                   # [B, base_channels, H, W]
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))  # [B, base_channels, H, W]
        
        # 生成速度场
        velocity_fields = self.velocity_head(d1)  # [B, out_frames*2, H, W]
        
        # 重塑为 [B, out_frames, H, W, 2]
        B, _, H, W = velocity_fields.shape
        velocity_fields = velocity_fields.view(B, self.out_frames, 2, H, W).permute(0, 1, 3, 4, 2)
        
        return velocity_fields

class LightweightInceptionUNet(nn.Module):
    """
    轻量化版本的Inception-UNet
    减少了层数和通道数，适合计算资源有限的场景
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=16, incep_ker=[3,5,7], groups=4):
        super().__init__()
        
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        
        # 编码器 - 3层
        self.enc1 = Inception(in_channels, base_channels//2, base_channels, incep_ker, groups)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = Inception(base_channels, base_channels, base_channels * 2, incep_ker, groups)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = Inception(base_channels * 2, base_channels * 2, base_channels * 4, incep_ker, groups)
        self.pool3 = nn.MaxPool2d(2)
        
        # 底层
        self.bottleneck = Inception(base_channels * 4, base_channels * 4, base_channels * 8, incep_ker, groups)
        
        # 解码器 - 3层
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = Inception(base_channels * 8, base_channels * 4, base_channels * 4, incep_ker, groups)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = Inception(base_channels * 4, base_channels * 2, base_channels * 2, incep_ker, groups)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = Inception(base_channels * 2, base_channels, base_channels, incep_ker, groups)
        
        # 输出头
        self.velocity_head = nn.Conv2d(base_channels, out_frames * 2, 1)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # 底层
        b = self.bottleneck(self.pool3(e3))
        
        # 解码器
        d3 = self.up3(b)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))
        
        # 输出
        velocity_fields = self.velocity_head(d1)
        B, _, H, W = velocity_fields.shape
        velocity_fields = velocity_fields.view(B, self.out_frames, 2, H, W).permute(0, 1, 3, 4, 2)
        
        return velocity_fields


class LightweightUNet_v2(nn.Module):
    """
    轻量化Attention-UNet
    输入: 过去6帧 [B, t_in*C, H, W]
    输出: 6个速度场 [B, t_out, H, W, 2]
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32, vel_proj=True):
        super().__init__()
        
        # 是否需要速度映射头
        self.vel_proj=vel_proj

        # 输入是t_in帧拼接
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # 底层
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # 输出层：预测t_out个时间步的速度场
        if self.vel_proj:
            self.head = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)  # t_out个时间步 × 通道数量 × 2个速度分量
        else:
            self.head = nn.Conv2d(base_channels, out_frames*frame_channels, 1) # t_out个时间步 × 通道数量

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 拼接的t_in帧
        Returns:
            velocity_fields: [B, T, H, W, 2] t_out个速度场
            Or
            fileds:
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # 底层
        b = self.bottleneck(self.pool4(e4))
        
        # 解码器 + 注意力
        d4 = self.up4(b)
        e4_att = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))
        
        fields = self.head(d1)  # [B, 12, C, H, W]
        # 输出速度场
        if self.vel_proj:
            # 重塑为 [B, 6, C, H, W, 2]
            B, _, H, W = fields.shape
            fields = fields.view(B, self.out_frames, self.frame_channels, 2, H, W).permute(0, 1, 2, 4, 5, 3)
        else:
            fields = fields.reshape(B,T,C,H,W)

        return fields
    

class SwinTransformerUNet(nn.Module):
    def __init__(
            self, 
            img_size=256,
            patch_size=4,
            in_chans=48,
            num_classes=96,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=True,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first",
            vel_mode=False,
            frame_channels=8):
        
        super().__init__()
        self.model = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            final_upsample=final_upsample
        )
        self.vel_mode = vel_mode
        self.frame_ch = frame_channels

    def forward(self, x):
        # x.sahpe: BTCHW
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.model(x)
        if self.vel_mode:
            # B, C, H, W = x.shape
            x = rearrange(x, 'b (t c v) h w -> b t c h w v', v=2, c=self.frame_ch)
        return x
    


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


class GradLightweightUNet(nn.Module):
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32, vel_proj=True):
        super().__init__()
        
        # Grad
        self.sobel_grad = SobelGrad(channels=frame_channels*in_frames)

        # 是否需要速度映射头
        self.vel_proj=vel_proj

        # 输入是t_in帧拼接
        in_channels = frame_channels * in_frames * 2
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # 底层
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # 输出层：预测t_out个时间步的速度场
        if self.vel_proj:
            self.head = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)  # t_out个时间步 × 通道数量 × 2个速度分量
        else:
            self.head = nn.Conv2d(base_channels, out_frames*frame_channels, 1) # t_out个时间步 × 通道数量

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 拼接的t_in帧
        Returns:
            velocity_fields: [B, T, H, W, 2] t_out个速度场
            Or
            fileds:
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        grad_x =self.sobel_grad(x)
        x = torch.cat([x, grad_x], dim=1)  # 拼接
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # 底层
        b = self.bottleneck(self.pool4(e4))
        
        # 解码器 + 注意力
        d4 = self.up4(b)
        e4_att = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))
        
        fields = self.head(d1)  # [B, 12, C, H, W]
        # 输出速度场
        if self.vel_proj:
            # 重塑为 [B, 6, C, H, W, 2]
            B, _, H, W = fields.shape
            fields = fields.view(B, self.out_frames, self.frame_channels, 2, H, W).permute(0, 1, 2, 4, 5, 3)
        else:
            fields = fields.reshape(B,T,C,H,W)

        return fields
    

class LightweightUNetDual(nn.Module):
    """
    与 LightweightUNet 相同的编码器；复制一套完全相同的解码器。
    - decoder_v：输出未来位移场 [B, T_out, C, H, W, 2]
    - decoder_r：输出强度残差   [B, T_out, C, H, W]
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32):
        super().__init__()
        # ------- 公共参数 -------
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels

        # ------- 编码器（与原版一致） -------
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # ===================== 解码器 A：位移（velocity） =====================
        self.up4_v  = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4_v = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4_v = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3_v  = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3_v = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3_v = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2_v  = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2_v = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2_v = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1_v  = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1_v = AttentionBlock(base_channels, base_channels)
        self.dec1_v = ConvBlock(base_channels * 2, base_channels)

        # 头：T_out * C * 2
        self.head_v = nn.Conv2d(base_channels, out_frames * frame_channels * 2, 1)

        # ===================== 解码器 B：残差（residual/intensity） =====================
        # 完全同构，权重独立
        self.up4_r  = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4_r = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4_r = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3_r  = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3_r = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3_r = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2_r  = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2_r = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2_r = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1_r  = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1_r = AttentionBlock(base_channels, base_channels)
        self.dec1_r = ConvBlock(base_channels * 2, base_channels)

        # 头：T_out * C（每个通道的强度残差）
        self.head_r = nn.Conv2d(base_channels, out_frames * frame_channels, 1)

    def forward(self, x):
        """
        x: [B, T_in, C, H, W]
        returns:
          vel: [B, T_out, C, H, W, 2]
          res: [B, T_out, C, H, W]
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)

        # ---- encoder ----
        e1 = self.enc1(x)                # [B, Bc,   H,   W]
        e2 = self.enc2(self.pool1(e1))   # [B, 2Bc,  H/2, W/2]
        e3 = self.enc3(self.pool2(e2))   # [B, 4Bc,  H/4, W/4]
        e4 = self.enc4(self.pool3(e3))   # [B, 8Bc,  H/8, W/8]
        b  = self.bottleneck(self.pool4(e4))  # [B,16Bc, H/16,W/16]

        # ---- decoder A: velocity ----
        d4v = self.up4_v(b)
        e4v = self.att4_v(e4, d4v)
        d4v = self.dec4_v(torch.cat([e4v, d4v], dim=1))

        d3v = self.up3_v(d4v)
        e3v = self.att3_v(e3, d3v)
        d3v = self.dec3_v(torch.cat([e3v, d3v], dim=1))

        d2v = self.up2_v(d3v)
        e2v = self.att2_v(e2, d2v)
        d2v = self.dec2_v(torch.cat([e2v, d2v], dim=1))

        d1v = self.up1_v(d2v)
        e1v = self.att1_v(e1, d1v)
        d1v = self.dec1_v(torch.cat([e1v, d1v], dim=1))

        vel = self.head_v(d1v)  # [B, T_out*C*2, H, W]
        vel = vel.view(B, self.out_frames, self.frame_channels, 2, H, W).permute(0,1,2,4,5,3)

        # ---- decoder B: residual ----
        d4r = self.up4_r(b)
        e4r = self.att4_r(e4, d4r)
        d4r = self.dec4_r(torch.cat([e4r, d4r], dim=1))

        d3r = self.up3_r(d4r)
        e3r = self.att3_r(e3, d3r)
        d3r = self.dec3_r(torch.cat([e3r, d3r], dim=1))

        d2r = self.up2_r(d3r)
        e2r = self.att2_r(e2, d2r)
        d2r = self.dec2_r(torch.cat([e2r, d2r], dim=1))

        d1r = self.up1_r(d2r)
        e1r = self.att1_r(e1, d1r)
        d1r = self.dec1_r(torch.cat([e1r, d1r], dim=1))

        res = self.head_r(d1r)  # [B, T_out*C, H, W]
        res = res.view(B, self.out_frames, self.frame_channels, H, W)

        return vel, res


def make_xy_grid(B, H, W, device):
    """返回 [B, 1, H, W] 的 x、y 像素坐标网格"""
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    xs = xs.view(1, 1, H, W).expand(B, -1, -1, -1)
    ys = ys.view(1, 1, H, W).expand(B, -1, -1, -1)
    return xs, ys

def compose_global_flow(params, H, W):
    """
    params: dict with tensors (每步一组)
        omega: [B, T, 1]    角速度 (pixel/frame)
        cx,cy: [B, T, 1]    旋转中心 (像素)
        tx,ty: [B, T, 1]    平移 (像素/frame)
    return:
        v_global: [B, T, 1, H, W, 2]  （通道共享；若要 per-channel 可再扩展到 C）
    """
    B, T, _ = params['omega'].shape
    device = params['omega'].device
    xs, ys = make_xy_grid(B, H, W, device)          # [B,1,H,W]

    # 展开到 T： [B,T,1,H,W]
    xs = xs.unsqueeze(1).expand(-1, T, -1, -1, -1)
    ys = ys.unsqueeze(1).expand(-1, T, -1, -1, -1)

    cx = params['cx'].view(B, T, 1, 1, 1)
    cy = params['cy'].view(B, T, 1, 1, 1)
    tx = params['tx'].view(B, T, 1, 1, 1)
    ty = params['ty'].view(B, T, 1, 1, 1)
    om = params['omega'].view(B, T, 1, 1, 1)

    # r - c
    rx = xs - cx
    ry = ys - cy

    # J @ (r-c) = [ -ry, rx ]
    rot_x = -ry
    rot_y =  rx

    vx = om * rot_x + tx
    vy = om * rot_y + ty

    v = torch.stack([vx, vy], dim=-1)              # [B,T,1,H,W,2]
    return v

class GlobalRigidFlowHead(nn.Module):
    """
    从帧间上下文特征预测每个时间步的全局环流参数 (omega, cx, cy, tx, ty)。
    - 预测范围做了合理约束（sigmoid/tanh），避免训练初期数值暴走。
    """
    def __init__(self, feat_ch, T_out, H, W):
        super().__init__()
        self.T = T_out
        self.H, self.W = H, W

        hidden = max(128, feat_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)     # [B,C,1,1]
        self.fc   = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 5*T_out, 1)       # 每个时间步 5 个参数
        )
        # 可学习的尺度：把无量纲输出映射到合理物理量级
        self.register_parameter('omega_scale', nn.Parameter(torch.tensor(1e-3)))  # rad/pixel ~ 1e-3 量级
        self.register_parameter('shift_scale', nn.Parameter(torch.tensor(1.0)))   # 像素/frame
        self.register_parameter('center_bias_x', nn.Parameter(torch.tensor(0.5))) # 初始中心在图中心
        self.register_parameter('center_bias_y', nn.Parameter(torch.tensor(0.5)))

    def forward(self, feat):
        """
        feat: [B, C, Hf, Wf]  — 例如 bottleneck 或 decoder 大特征图
        return dict of params, each [B, T, 1]
        """
        B = feat.size(0)
        x = self.pool(feat)          # [B,C,1,1]
        x = self.fc(x).view(B, self.T, 5)  # [B,T,5]

        # 拆分并约束
        raw_omega = torch.tanh(x[..., 0:1])                 # [-1,1]
        raw_cx    = torch.sigmoid(x[..., 1:2])              # [0,1] * W
        raw_cy    = torch.sigmoid(x[..., 2:3])              # [0,1] * H
        raw_tx    = torch.tanh(x[..., 3:4])                 # [-1,1]
        raw_ty    = torch.tanh(x[..., 4:5])                 # [-1,1]

        omega = raw_omega * self.omega_scale                # 小角速度
        cx = (self.center_bias_x + (raw_cx-0.5)) * self.W   # 以中心为基准微调
        cy = (self.center_bias_y + (raw_cy-0.5)) * self.H
        tx = raw_tx * self.shift_scale
        ty = raw_ty * self.shift_scale

        return {'omega': omega, 'cx': cx, 'cy': cy, 'tx': tx, 'ty': ty}


class LightweightUNetDual_Rotation(nn.Module):
    """
    编码器共享；解码器 A 输出 残差流 delta_v；解码器 B 输出 强度残差 s。
    额外的 GlobalRigidFlowHead 输出 (omega,c,t) 并生成 v_global；最终 v = v_global + delta_v。
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32, full_H=256, full_W=256):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels
        # === 编码器（同你现有） ===
        self.enc1 = ConvBlock(frame_channels*in_frames, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16)

        # === 解码器 A：残差流 delta_v ===
        self.up4_v  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_v = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_v = ConvBlock(base_channels*16, base_channels*8)
        self.up3_v  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_v = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_v = ConvBlock(base_channels*8, base_channels*4)
        self.up2_v  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_v = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_v = ConvBlock(base_channels*4, base_channels*2)
        self.up1_v  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_v = AttentionBlock(base_channels, base_channels)
        self.dec1_v = ConvBlock(base_channels*2, base_channels)
        self.head_delta_v = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)

        # === 解码器 B：强度残差 s（保持你的原样） ===
        self.up4_r  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_r = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_r = ConvBlock(base_channels*16, base_channels*8)
        self.up3_r  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_r = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_r = ConvBlock(base_channels*8, base_channels*4)
        self.up2_r  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_r = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_r = ConvBlock(base_channels*4, base_channels*2)
        self.up1_r  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_r = AttentionBlock(base_channels, base_channels)
        self.dec1_r = ConvBlock(base_channels*2, base_channels)
        self.head_residual = nn.Conv2d(base_channels, out_frames*frame_channels, 1)

        # === 全局流头 ===
        self.global_head = GlobalRigidFlowHead(
            feat_ch=base_channels*16, T_out=out_frames, H=full_H, W=full_W
        )

    def forward(self, x):
        """
        x: [B, T_in, C, H, W]
        return:
          v:   [B, T_out, C, H, W, 2] = v_global + delta_v
          s:   [B, T_out, C, H, W]
          pars: dict of global params (便于可视化/诊断)
        """
        B, Tin, C, H, W = x.shape
        x_cat = x.view(B, Tin*C, H, W)

        # 编码
        e1 = self.enc1(x_cat)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))   # [B, 16Bc, H/16, W/16]

        # —— 解码 A：delta_v —— #
        d4v = self.dec4_v(torch.cat([self.att4_v(e4, self.up4_v(b)), self.up4_v(b)], 1))
        d3v = self.dec3_v(torch.cat([self.att3_v(e3, self.up3_v(d4v)), self.up3_v(d4v)], 1))
        d2v = self.dec2_v(torch.cat([self.att2_v(e2, self.up2_v(d3v)), self.up2_v(d3v)], 1))
        d1v = self.dec1_v(torch.cat([self.att1_v(e1, self.up1_v(d2v)), self.up1_v(d2v)], 1))
        delta_v = self.head_delta_v(d1v).view(B, self.out_frames, self.frame_channels, 2, H, W)
        delta_v = delta_v.permute(0,1,2,4,5,3)  # [B,T,C,H,W,2]

        # —— 解码 B：s —— #
        d4r = self.dec4_r(torch.cat([self.att4_r(e4, self.up4_r(b)), self.up4_r(b)], 1))
        d3r = self.dec3_r(torch.cat([self.att3_r(e3, self.up3_r(d4r)), self.up3_r(d4r)], 1))
        d2r = self.dec2_r(torch.cat([self.att2_r(e2, self.up2_r(d3r)), self.up2_r(d3r)], 1))
        d1r = self.dec1_r(torch.cat([self.att1_r(e1, self.up1_r(d2r)), self.up1_r(d2r)], 1))
        s = self.head_residual(d1r).view(B, self.out_frames, self.frame_channels, H, W)

        # —— 全局参数与 v_global —— #
        pars = self.global_head(b)                          # dict: each [B,T,1]
        v_global = compose_global_flow(pars, H, W)          # [B,T,1,H,W,2]
        v_global = v_global.expand(-1, -1, self.frame_channels, -1, -1, -1)  # 通道共享 → [B,T,C,H,W,2]

        v = v_global + delta_v
        return v, s, pars


class LightweightUNetDual_Rotation_Grad(nn.Module):
    """
    在 LightweightUNetDual_Rotation 基础上引入梯度特征：
    - 先对输入帧堆叠图做 Sobel 梯度幅值（逐通道、逐帧）
    - 在通道维与原图拼接，作为编码器的输入
    其余结构保持不变：
      解码器A输出 delta_v（残差流），解码器B输出强度残差 s，
      GlobalRigidFlowHead 输出 (omega, cx, cy, tx, ty) 并生成 v_global，
      最终 v = v_global + delta_v
    """
    def __init__(self, frame_channels, in_frames, out_frames,
                 base_channels=32, full_H=256, full_W=256,
                 use_grad=True):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels
        self.use_grad = use_grad

        # --- 输入通道数：是否拼接梯度 ---
        raw_in_ch = frame_channels * in_frames
        enc_in_ch = raw_in_ch * 2 if use_grad else raw_in_ch

        # === 编码器 ===
        self.enc1 = ConvBlock(enc_in_ch, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16)

        # 若使用梯度，建立 Sobel 模块（按输入“原图通道数”逐通道做深度可分卷积）
        self.sobel_grad = SobelGrad(channels=raw_in_ch) if use_grad else None

        # === 解码器 A：残差流 delta_v ===
        self.up4_v  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_v = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_v = ConvBlock(base_channels*16, base_channels*8)
        self.up3_v  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_v = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_v = ConvBlock(base_channels*8, base_channels*4)
        self.up2_v  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_v = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_v = ConvBlock(base_channels*4, base_channels*2)
        self.up1_v  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_v = AttentionBlock(base_channels, base_channels)
        self.dec1_v = ConvBlock(base_channels*2, base_channels)
        self.head_delta_v = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)

        # === 解码器 B：强度残差 s ===
        self.up4_r  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_r = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_r = ConvBlock(base_channels*16, base_channels*8)
        self.up3_r  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_r = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_r = ConvBlock(base_channels*8, base_channels*4)
        self.up2_r  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_r = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_r = ConvBlock(base_channels*4, base_channels*2)
        self.up1_r  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_r = AttentionBlock(base_channels, base_channels)
        self.dec1_r = ConvBlock(base_channels*2, base_channels)
        self.head_residual = nn.Conv2d(base_channels, out_frames*frame_channels, 1)

        # === 全局刚体流头 ===
        self.global_head = GlobalRigidFlowHead(
            feat_ch=base_channels*16, T_out=out_frames, H=full_H, W=full_W
        )

    def forward(self, x):
        """
        x: [B, T_in, C, H, W]
        return:
          v:    [B, T_out, C, H, W, 2]  = v_global + delta_v
          s:    [B, T_out, C, H, W]
          pars: dict of global params
        """
        B, Tin, C, H, W = x.shape
        x_cat = x.view(B, Tin*C, H, W)       # [B, raw_in_ch, H, W]

        if self.use_grad:
            # 逐通道 Sobel 幅值 (深度可分)
            g_mag = self.sobel_grad(x_cat)   # [B, raw_in_ch, H, W]
            x_in  = torch.cat([x_cat, g_mag], dim=1)  # [B, 2*raw_in_ch, H, W]
        else:
            x_in = x_cat

        # ------ 编码 ------
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))   # [B,16Bc, H/16, W/16]

        # ------ 解码 A：delta_v ------
        d4v = self.up4_v(b)
        e4v = self.att4_v(e4, d4v)
        d4v = self.dec4_v(torch.cat([e4v, d4v], 1))

        d3v = self.up3_v(d4v)
        e3v = self.att3_v(e3, d3v)
        d3v = self.dec3_v(torch.cat([e3v, d3v], 1))

        d2v = self.up2_v(d3v)
        e2v = self.att2_v(e2, d2v)
        d2v = self.dec2_v(torch.cat([e2v, d2v], 1))

        d1v = self.up1_v(d2v)
        e1v = self.att1_v(e1, d1v)
        d1v = self.dec1_v(torch.cat([e1v, d1v], 1))

        delta_v = self.head_delta_v(d1v).view(B, self.out_frames, self.frame_channels, 2, H, W)
        delta_v = delta_v.permute(0,1,2,4,5,3)  # [B,T,C,H,W,2]

        # ------ 解码 B：s ------
        d4r = self.up4_r(b)
        e4r = self.att4_r(e4, d4r)
        d4r = self.dec4_r(torch.cat([e4r, d4r], 1))

        d3r = self.up3_r(d4r)
        e3r = self.att3_r(e3, d3r)
        d3r = self.dec3_r(torch.cat([e3r, d3r], 1))

        d2r = self.up2_r(d3r)
        e2r = self.att2_r(e2, d2r)
        d2r = self.dec2_r(torch.cat([e2r, d2r], 1))

        d1r = self.up1_r(d2r)
        e1r = self.att1_r(e1, d1r)
        d1r = self.dec1_r(torch.cat([e1r, d1r], 1))

        s = self.head_residual(d1r).view(B, self.out_frames, self.frame_channels, H, W)

        # ------ 全局参数 & 刚体流 ------
        pars = self.global_head(b)                  # dict: each [B,T,1]
        v_global = compose_global_flow(pars, H, W)  # [B,T,1,H,W,2]
        v_global = v_global.expand(-1, -1, self.frame_channels, -1, -1, -1)

        v = v_global + delta_v
        return v, s, pars


class LightweightUNetTri_Rotation_Grad(nn.Module):
    """
    Encoder + 3 Decoders:
      A: delta_v (残差流)            → [B,T_out,C,H,W,2]
      B: λ_mix  (源汇率, 单通道)      → [B,T_out,1,H,W]
      C: mask   (门控logits→prob)     → [B,T_out,1,H,W]
    仍含全局刚体流头: v = v_global + delta_v
    """
    def __init__(self, frame_channels, in_frames, out_frames,
                 base_channels=32, full_H=256, full_W=256,
                 use_grad=True, lam_max=0.8):
        super().__init__()
        self.in_frames   = in_frames
        self.out_frames  = out_frames
        self.frame_channels = frame_channels
        self.use_grad    = use_grad
        self.lam_max     = float(lam_max)   # dt*lam_max ≤ 0.8 建议

        # --- 输入通道数：是否拼接梯度 ---
        raw_in_ch = frame_channels * in_frames
        enc_in_ch = raw_in_ch * 2 if use_grad else raw_in_ch

        # === 编码器 ===
        self.enc1 = ConvBlock(enc_in_ch, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16)

        self.sobel_grad = SobelGrad(channels=raw_in_ch) if use_grad else None

        # === 解码器 A：delta_v ===
        self.up4_v  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_v = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_v = ConvBlock(base_channels*16, base_channels*8)
        self.up3_v  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_v = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_v = ConvBlock(base_channels*8, base_channels*4)
        self.up2_v  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_v = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_v = ConvBlock(base_channels*4, base_channels*2)
        self.up1_v  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_v = AttentionBlock(base_channels, base_channels)
        self.dec1_v = ConvBlock(base_channels*2, base_channels)
        self.head_delta_v = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)

        # === 解码器 B：λ_mix（源汇率, 单通道） ===
        self.up4_l  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_l = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_l = ConvBlock(base_channels*16, base_channels*8)
        self.up3_l  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_l = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_l = ConvBlock(base_channels*8, base_channels*4)
        self.up2_l  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_l = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_l = ConvBlock(base_channels*4, base_channels*2)
        self.up1_l  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_l = AttentionBlock(base_channels, base_channels)
        self.dec1_l = ConvBlock(base_channels*2, base_channels)
        self.head_lam = nn.Conv2d(base_channels, out_frames*1, 1)  # 单通道

        # === 解码器 C：mask（门控，输出logits→sigmoid） ===
        self.up4_m  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_m = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_m = ConvBlock(base_channels*16, base_channels*8)
        self.up3_m  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_m = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_m = ConvBlock(base_channels*8, base_channels*4)
        self.up2_m  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_m = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_m = ConvBlock(base_channels*4, base_channels*2)
        self.up1_m  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_m = AttentionBlock(base_channels, base_channels)
        self.dec1_m = ConvBlock(base_channels*2, base_channels)
        self.head_mask = nn.Conv2d(base_channels, out_frames*1, 1)  # 单通道 logits

        # === 全局刚体流头 ===
        self.global_head = GlobalRigidFlowHead(
            feat_ch=base_channels*16, T_out=out_frames, H=full_H, W=full_W
        )

    def forward(self, x):
        """
        x: [B, T_in, C, H, W]
        return:
          v:      [B, T_out, C, H, W, 2]  = v_global + delta_v
          lam:    [B, T_out, 1, H, W]     (0, lam_max]
          mask:   [B, T_out, 1, H, W]     (0,1) after sigmoid
          pars:   dict (global flow params)
        """
        B, Tin, C, H, W = x.shape
        x_cat = x.view(B, Tin*C, H, W)

        if self.use_grad:
            g_mag = self.sobel_grad(x_cat)
            x_in  = torch.cat([x_cat, g_mag], dim=1)
        else:
            x_in  = x_cat

        # 编码
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        # 解码 A：delta_v
        d4v = self.dec4_v(torch.cat([self.att4_v(e4, self.up4_v(b)), self.up4_v(b)], 1))
        d3v = self.dec3_v(torch.cat([self.att3_v(e3, self.up3_v(d4v)), self.up3_v(d4v)], 1))
        d2v = self.dec2_v(torch.cat([self.att2_v(e2, self.up2_v(d3v)), self.up2_v(d3v)], 1))
        d1v = self.dec1_v(torch.cat([self.att1_v(e1, self.up1_v(d2v)), self.up1_v(d2v)], 1))
        delta_v = self.head_delta_v(d1v).view(B, self.out_frames, self.frame_channels, 2, H, W)
        delta_v = delta_v.permute(0,1,2,4,5,3)  # [B,T,C,H,W,2]

        # 解码 B：λ_mix
        d4l = self.dec4_l(torch.cat([self.att4_l(e4, self.up4_l(b)), self.up4_l(b)], 1))
        d3l = self.dec3_l(torch.cat([self.att3_l(e3, self.up3_l(d4l)), self.up3_l(d4l)], 1))
        d2l = self.dec2_l(torch.cat([self.att2_l(e2, self.up2_l(d3l)), self.up2_l(d3l)], 1))
        d1l = self.dec1_l(torch.cat([self.att1_l(e1, self.up1_l(d2l)), self.up1_l(d2l)], 1))
        lam = torch.sigmoid(self.head_lam(d1l))  # [B, T_out*1, H, W]
        lam = lam.view(B, self.out_frames, 1, H, W) * self.lam_max

        # 解码 C：mask (logits→sigmoid)
        d4m = self.dec4_m(torch.cat([self.att4_m(e4, self.up4_m(b)), self.up4_m(b)], 1))
        d3m = self.dec3_m(torch.cat([self.att3_m(e3, self.up3_m(d4m)), self.up3_m(d4m)], 1))
        d2m = self.dec2_m(torch.cat([self.att2_m(e2, self.up2_m(d3m)), self.up2_m(d3m)], 1))
        d1m = self.dec1_m(torch.cat([self.att1_m(e1, self.up1_m(d2m)), self.up1_m(d2m)], 1))
        mask_logits = self.head_mask(d1m).view(B, self.out_frames, 1, H, W)
        mask = torch.sigmoid(mask_logits)

        # 全局参数 & 刚体流
        pars = self.global_head(b)                  # dict: each [B,T,1]
        v_global = compose_global_flow(pars, H, W)  # [B,T,1,H,W,2]
        v_global = v_global.expand(-1, -1, self.frame_channels, -1, -1, -1)

        v = v_global + delta_v
        return v, lam, mask, pars


# 动力学特征融合的Intensity Decoder
# ---------- 差分算子 ----------
def _cdiff_x(x):
    dx = 0.5 * (x[..., :, 2:] - x[..., :, :-2])
    return F.pad(dx, (1,1,0,0), mode='replicate')

def _cdiff_y(x):
    dy = 0.5 * (x[..., 2:, :] - x[..., :-2, :])
    return F.pad(dy, (0,0,1,1), mode='replicate')


# ---------- 动力学特征提取 ----------
def compute_dyn_feats(u, v, return_feats=True):
    dudx = _cdiff_x(u); dudy = _cdiff_y(u)
    dvdx = _cdiff_x(v); dvdy = _cdiff_y(v)

    div   = dudx + dvdy
    vort  = dvdx - dudy
    e1    = dudx - dvdy
    e2    = dvdx + dudy
    shear = torch.sqrt(e1 * e1 + e2 * e2 + 1e-6)
    strain= torch.sqrt(div * div + shear * shear + 1e-6)
    speed = torch.sqrt(u * u + v * v + 1e-6)

    if return_feats:
        return {
            "div": div,
            "vort": vort,
            "e1": e1,
            "e2": e2,
            "shear": shear,
            "strain": strain,
            "speed": speed
        }
    else:
        feats = torch.stack([div, vort, e1, e2, shear, strain, speed], dim=1)
        return feats


# ---------- 主模型 ----------
class LightweightUNetDual_Rotation_Grad_Dyn(nn.Module):
    """
    双分支结构：
      A：delta_v (残差流)
      B：s (强度残差)
    在B分支的不同层注入不同的动力学特征：
      - d4r/d3r：早融合 (speed, e1, e2, shear)
      - d2r：中层融合 (vort)
      - d1r：晚融合 (div, strain)
    手动指定不同阶段的融合通道数 dyn_ch_list = [a,b,c]
      a -> early (d4r, d3r)
      b -> mid   (d2r)
      c -> late  (d1r)
    """
    def __init__(self, frame_channels, in_frames, out_frames,
                 base_channels=32, full_H=256, full_W=256,
                 use_grad=True,
                 use_dyn_feats=True,
                 detach_flow=True,
                 dyn_ch_list=[32*4, 32*2,32*1]):  # 手动控制每阶段的通道数 [a,b,c]
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels
        self.use_grad = use_grad
        self.use_dyn_feats = use_dyn_feats
        self.detach_flow = detach_flow

        # unpack
        self.dyn_ch_early, self.dyn_ch_mid, self.dyn_ch_late = dyn_ch_list
        print(f"[Dyn_Fusion] 使用动力学投影通道数 early={self.dyn_ch_early}, mid={self.dyn_ch_mid}, late={self.dyn_ch_late}")

        # === 输入通道 ===
        raw_in_ch = frame_channels * in_frames
        enc_in_ch = raw_in_ch * 2 if use_grad else raw_in_ch

        # === 编码器 ===
        self.enc1 = ConvBlock(enc_in_ch, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16)

        self.sobel_grad = SobelGrad(channels=raw_in_ch) if use_grad else None

        # === 解码器 A：delta_v ===
        self.up4_v  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_v = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_v = ConvBlock(base_channels*16, base_channels*8)
        self.up3_v  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_v = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_v = ConvBlock(base_channels*8, base_channels*4)
        self.up2_v  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_v = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_v = ConvBlock(base_channels*4, base_channels*2)
        self.up1_v  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_v = AttentionBlock(base_channels, base_channels)
        self.dec1_v = ConvBlock(base_channels*2, base_channels)
        self.head_delta_v = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)

        # === 解码器 B：强度残差 ===
        self.up4_r  = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.att4_r = AttentionBlock(base_channels*8, base_channels*8)
        self.dec4_r = ConvBlock(base_channels*16 + (self.dyn_ch_early if use_dyn_feats else 0), base_channels*8)

        self.up3_r  = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.att3_r = AttentionBlock(base_channels*4, base_channels*4)
        self.dec3_r = ConvBlock(base_channels*8 + (self.dyn_ch_early if use_dyn_feats else 0), base_channels*4)

        self.up2_r  = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.att2_r = AttentionBlock(base_channels*2, base_channels*2)
        self.dec2_r = ConvBlock(base_channels*4 + (self.dyn_ch_mid if use_dyn_feats else 0), base_channels*2)

        self.up1_r  = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.att1_r = AttentionBlock(base_channels, base_channels)
        self.dec1_r = ConvBlock(base_channels*2 + (self.dyn_ch_late if use_dyn_feats else 0), base_channels)
        self.head_residual = nn.Conv2d(base_channels, out_frames*frame_channels, 1)

        # === 全局刚体流头 ===
        self.global_head = GlobalRigidFlowHead(
            feat_ch=base_channels*16, T_out=out_frames, H=full_H, W=full_W
        )

        # === 动力学特征投影卷积 ===
        if use_dyn_feats:
            self.proj_early = nn.Sequential(nn.Conv2d(4,  self.dyn_ch_early, 3, padding=1), nn.ReLU(inplace=True))
            self.proj_mid   = nn.Sequential(nn.Conv2d(1,  self.dyn_ch_mid,   3, padding=1), nn.ReLU(inplace=True))
            self.proj_late  = nn.Sequential(nn.Conv2d(2,  self.dyn_ch_late,  3, padding=1), nn.ReLU(inplace=True))
            # 下采样匹配空间分辨率
            self.down2 = nn.AvgPool2d(2)
            self.down4 = nn.AvgPool2d(4)
            self.down8 = nn.AvgPool2d(8)

    def forward(self, x):
        B, Tin, C, H, W = x.shape
        x_cat = x.view(B, Tin*C, H, W)

        if self.use_grad:
            g_mag = self.sobel_grad(x_cat)
            x_in  = torch.cat([x_cat, g_mag], dim=1)
        else:
            x_in  = x_cat

        # === 编码 ===
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        # === delta_v ===
        d4v = self.dec4_v(torch.cat([self.att4_v(e4, self.up4_v(b)), self.up4_v(b)], 1))
        d3v = self.dec3_v(torch.cat([self.att3_v(e3, self.up3_v(d4v)), self.up3_v(d4v)], 1))
        d2v = self.dec2_v(torch.cat([self.att2_v(e2, self.up2_v(d3v)), self.up2_v(d3v)], 1))
        d1v = self.dec1_v(torch.cat([self.att1_v(e1, self.up1_v(d2v)), self.up1_v(d2v)], 1))
        delta_v = self.head_delta_v(d1v).view(B, self.out_frames, self.frame_channels, 2, H, W)
        delta_v = delta_v.permute(0,1,2,4,5,3)

        pars = self.global_head(b)
        v_global = compose_global_flow(pars, H, W)
        v_global = v_global.expand(-1, -1, self.frame_channels, -1, -1, -1)
        v = v_global + delta_v

        # === 动力学特征计算 ===
        dyn_feats_dict = {}
        if self.use_dyn_feats:
            uv = v.mean(2)
            u = uv[..., 0]; vxy = uv[..., 1]
            if self.detach_flow:
                u = u.detach(); vxy = vxy.detach()

            dyn_dict_t = compute_dyn_feats(u[:, -1], vxy[:, -1], return_feats=True)
            dyn_feats_dict = dyn_dict_t

            early = torch.stack([dyn_dict_t[k] for k in ["speed", "e1", "e2", "shear"]], dim=1)
            mid   = dyn_dict_t["vort"].unsqueeze(1)
            late  = torch.stack([dyn_dict_t[k] for k in ["div", "strain"]], dim=1)

            early_emb = self.proj_early(early)
            mid_emb   = self.proj_mid(mid)
            late_emb  = self.proj_late(late)

            early_emb_32 = self.down8(early_emb)
            early_emb_64 = self.down4(early_emb)
            mid_emb_128  = self.down2(mid_emb)
            late_emb_256 = late_emb
        else:
            early_emb_32 = early_emb_64 = mid_emb_128 = late_emb_256 = None

        # === 解码B：s ===
        d4r = self.up4_r(b)
        e4r = self.att4_r(e4, d4r)
        d4r = torch.cat([e4r, d4r], 1)
        if early_emb_32 is not None:
            d4r = torch.cat([d4r, early_emb_32], 1)
        d4r = self.dec4_r(d4r)

        d3r = self.up3_r(d4r)
        e3r = self.att3_r(e3, d3r)
        d3r = torch.cat([e3r, d3r], 1)
        if early_emb_64 is not None:
            d3r = torch.cat([d3r, early_emb_64], 1)
        d3r = self.dec3_r(d3r)

        d2r = self.up2_r(d3r)
        e2r = self.att2_r(e2, d2r)
        d2r = torch.cat([e2r, d2r], 1)
        if mid_emb_128 is not None:
            d2r = torch.cat([d2r, mid_emb_128], 1)
        d2r = self.dec2_r(d2r)

        d1r = self.up1_r(d2r)
        e1r = self.att1_r(e1, d1r)
        d1r = torch.cat([e1r, d1r], 1)
        if late_emb_256 is not None:
            d1r = torch.cat([d1r, late_emb_256], 1)
        d1r = self.dec1_r(d1r)

        s = self.head_residual(d1r).view(B, self.out_frames, self.frame_channels, H, W)

        return {
            "v": v,
            "s": s,
            "pars": pars,
            "dyn_feats": dyn_feats_dict
        }


