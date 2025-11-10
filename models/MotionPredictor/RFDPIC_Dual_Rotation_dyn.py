import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    from .softsplat import softsplat as _softsplat_op, LogitNet
    HAS_SOFTSPLAT = True
except Exception:
    HAS_SOFTSPLAT = False

from .DPIC_models import (
    LightweightUNet, 
    EarthformerUNet, 
    InceptionUNet, 
    LightweightInceptionUNet,
    SwinTransformerUNet,
    GradLightweightUNet,
    LightweightUNetDual,
    LightweightUNetDual_Rotation,
    LightweightUNetDual_Rotation_Grad,
    LightweightUNetDual_Rotation_Grad_Dyn)

from .RF_models import (
    LightweightRefinementNet, 
    WarpSharpen, 
    WarpSharpen_VAE)

from torchdiffeq import odeint
import torch.nn.functional as F
try:
    from h8LM.utils.jacobi_iter import jacobi_diffuse_iso
except Exception:
    print("无法导入 jacobi_diffuse_iso, 请确保工具在您的路径中。")

# --- tiny gate: 把两个分支(前向splat/后向warp)的均值图做门控，输出 [B,1,H,W] ---
class Gate2to1(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, k, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, 1, 3, 1, 1)
        )
    def forward(self, x2):  # x2: [B,2,H,W]
        return torch.sigmoid(self.net(x2))


def _grad_mag(x):
    # x: [B,C,H,W] → [B,1,H,W], 简单一阶差分
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    # pad 回原尺寸
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    mag = torch.sqrt(dx*dx + dy*dy + 1e-8).mean(1, keepdim=True)
    return mag


def _forward_splat(I, flow, metric_logit=None, splat_type='softmax', alpha=20.0):
    """
    I:     [B,C,H,W]
    flow:  [B,2,H,W]  (像素位移, 前向: x->x+flow)
    metric_logit: [B,1,H,W] 或 None
    """
    if not HAS_SOFTSPLAT:
        raise ImportError("softsplat 未安装。请先: pip install softsplat")
    if splat_type == 'softmax':
        if metric_logit is None:
            metric_logit = _grad_mag(I)  # 简单启发式：高纹理处更可信
        metric = alpha * torch.clamp(metric_logit, -10.0, 10.0)
        return _softsplat_op(input=I, flow=flow, metric=metric, type='softmax')  # [B,C,H,W]
    elif splat_type in ('sum', 'avg'):
        metric = None
        out = _softsplat_op(input=I, flow=flow, metric=metric, type='sum')  # 累加
        if splat_type == 'avg':
            # 用一个全1图做“计数器”，避免除0
            ones = torch.ones_like(I[:, :1])
            count = _softsplat_op(input=ones, flow=flow, metric=None, type='sum').clamp_min(1e-6)
            out = out / count
        return out
    else:
        raise ValueError(f"splat_type 仅支持 'softmax'|'sum'|'avg'，但得到 {splat_type}")


class AdvectionWithSource(nn.Module):
    """通道独立平流方程：dI_c/dt = - (v_c·∇I_c) + λ·S_c"""
    def __init__(self, v_key, r_key, key_times, scale):
        super().__init__()
        # v_key: [B, T, C, 2, H, W]
        self.v_key = v_key
        self.r_key = r_key
        self.key_times = key_times
        self.scale = scale

    def _interp_key(self, t):
        t = t.clamp(self.key_times[0], self.key_times[-1])
        idx = torch.searchsorted(self.key_times, t)
        idx = torch.clamp(idx, 1, len(self.key_times) - 1)
        t0, t1 = self.key_times[idx - 1], self.key_times[idx]
        w = ((t - t0) / (t1 - t0)).clamp(0, 1)
        v0, v1 = self.v_key[:, idx - 1], self.v_key[:, idx]  # [B,C,2,H,W]
        v_t = (1 - w) * v0 + w * v1                          # [B,C,2,H,W]
        v_t = v_t.permute(0, 2, 1, 3, 4).contiguous()         # -> [B,2,C,H,W]

        r_t = None
        if self.r_key is not None and self.r_key.dim() == 5:
            r0, r1 = self.r_key[:, idx - 1], self.r_key[:, idx]
            r_t = (1 - w) * r0 + w * r1
        return v_t, r_t

    def forward(self, t, I):
        """
        I: [B, C, H, W]  每个通道独立平流
        """
        B, C, H, W = I.shape
        I_pad = F.pad(I, (1,1,1,1), mode="replicate")
        dIdx = 0.5 * (I_pad[:, :, 1:-1, 2:] - I_pad[:, :, 1:-1, :-2])
        dIdy = 0.5 * (I_pad[:, :, 2:, 1:-1] - I_pad[:, :, :-2, 1:-1])

        v_t, r_t = self._interp_key(t)  # [B,2,C,H,W]
        adv = -(v_t[:, 0] * dIdx + v_t[:, 1] * dIdy)         # [B,C,H,W]
        if r_t is not None:
            adv = adv + self.scale * r_t
        return adv
    

class RFDPIC_Dual_Rotation_Dyn(nn.Module):
    """
    并行速度场预测器 + (可选)Softmax Splatting 前向采样分支
    输入过去6帧 -> 一次性预测6个速度场 -> 并行生成6个未来帧
    """
    def __init__(self, 
                 dp_name: str=None,
                 rf_name: str=None,
                 rf_dp_config_file: str=None,
                 dp_mode='autoregrad',
                 rf_mode='autoregrad',
                 interpolation_mode="bilinear",
                 padding_mode="border",
                 alpha = 0.,
                 # intenisty scale
                 use_res_scale: bool = False,
                 res_scale_init: float = 0.05,
                 # advect mode
                 advect_mode:str ='semi-lagrangian', # or ode
                 ode_method:str ='rk4',
                 ode_substeps:int =6,
                 # jacobi diffusion
                 use_implicit_diffusion = False,   # 开关
                 jacobi_iters = 2,                # 1–3 通常够
                 D_max = 0.2,                     # 最大扩散系数
                 k_edge = 0.1,                    # 梯度调制阈值
                 diff_dt = 1.0,
                 diff_dx = 1.0,
                 D_const = None,                  # 如果想固定 D，可赋值 torch.tensor(0.1)
                 # --- 新增 softmax splatting 相关选项 ---
                 use_splat: bool = False,
                 splat_type: str = "softmax",   # 'softmax' | 'sum' | 'avg'
                 splat_alpha: float = 20.0,     # softmax 温度
                 splat_blend: float = 0.5,      # 固定融合权重: pred = w*FW + (1-w)*BW
                 use_gate: bool = False         # 是否用轻量门控代替固定融合
                 ):
        super().__init__()
        cfg = self.get_base_config(rf_dp_config_file)
        dp_cfg = cfg['dp_model']
        rf_cfg = cfg['rf_model']

        # 轻量化Attention-UNet
        if dp_name == "LightweightUNetDual_Rotation_Grad_Dyn":
            # 这是你的新版 Fusion 网络
            model_cfg = dp_cfg["LightweightUNetDual_Rotation_Grad_Dyn"]
            self.displacement_net = LightweightUNetDual_Rotation_Grad_Dyn(
                frame_channels=model_cfg['frame_channels'],
                in_frames=model_cfg['in_frames'],
                out_frames=model_cfg['out_frames'],
                base_channels=model_cfg['base_channels'],
                full_H=model_cfg['full_H'],
                full_W=model_cfg['full_W'],
                use_grad=True,
                use_dyn_feats=True,
                detach_flow=True,
                dyn_ch_list=model_cfg['dyn_ch_list']
            )
            self.dp_is_dual = True
            self.dp_returns_pars = True
            self.dp_returns_dyn_feats = True

        elif dp_name == "LightweightUNet":
            model_cfg = dp_cfg[dp_name]
            self.displacement_net = LightweightUNet(
                frame_channels=model_cfg['frame_channels'],
                in_frames=model_cfg['in_frames'],
                out_frames=model_cfg['out_frames'],
                base_channels=model_cfg['base_channels']
            )
        elif dp_name == "GradLightweightUNet":
            model_cfg = dp_cfg[dp_name]
            self.displacement_net = GradLightweightUNet(
                frame_channels=model_cfg['frame_channels'],
                in_frames=model_cfg['in_frames'],
                out_frames=model_cfg['out_frames'],
                base_channels=model_cfg['base_channels'],
                vel_proj=model_cfg['vel_proj']
            )
        elif dp_name == "SwinUNet":
            model_cfg = dp_cfg[dp_name]
            self.displacement_net = SwinTransformerUNet(
                img_size=model_cfg['img_size'],
                num_classes=model_cfg['num_classes'],
                patch_size=model_cfg['patch_size'],
                in_chans=model_cfg['in_chans'],
                embed_dim=model_cfg['embed_dim'],
                depths=model_cfg['depths'],
                num_heads=model_cfg['num_heads'],
                window_size=model_cfg['window_size'],
                mlp_ratio=model_cfg['mlp_ratio'],
                qkv_bias=model_cfg['qkv_bias'],
                qk_scale=model_cfg['qk_scale'],
                drop_rate=model_cfg['drop_rate'],
                attn_drop_rate=model_cfg['attn_drop_rate'],
                drop_path_rate=model_cfg['drop_path_rate'],
                ape=model_cfg['ape'],
                patch_norm=model_cfg['patch_norm'],
                use_checkpoint=model_cfg['use_checkpoint'],
                final_upsample=model_cfg['final_upsample'],
                vel_mode=model_cfg['vel_mode'],
                frame_channels=model_cfg['frame_channels'],
            )
        elif dp_name == "LightweightUNetDual":
            model_cfg = dp_cfg[dp_name]
            # dual: 同时预测位移 vel 和 残差 res
            self.displacement_net = LightweightUNetDual(
                frame_channels=model_cfg['frame_channels'],
                in_frames=model_cfg['in_frames'],
                out_frames=model_cfg['out_frames'],
                base_channels=model_cfg['base_channels']
            )
            self.dp_is_dual = True
        elif dp_name == "LightweightUNetDual_Rotation":
            model_cfg = dp_cfg[dp_name]
            self.displacement_net = LightweightUNetDual_Rotation(
                frame_channels=model_cfg['frame_channels'],
                in_frames=model_cfg['in_frames'],
                out_frames=model_cfg['out_frames'],
                base_channels=model_cfg['base_channels'],
                full_H=model_cfg['full_H'],
                full_W=model_cfg['full_W'])
            self.dp_is_dual = True
            self.dp_returns_pars = True
        elif dp_name == "LightweightUNetDual_Rotation_Grad":
            model_cfg = dp_cfg[dp_name]
            self.displacement_net = LightweightUNetDual_Rotation_Grad(
                frame_channels=model_cfg['frame_channels'],
                in_frames=model_cfg['in_frames'],
                out_frames=model_cfg['out_frames'],
                base_channels=model_cfg['base_channels'],
                full_H=model_cfg['full_H'],
                full_W=model_cfg['full_W'])
            self.dp_is_dual = True
            self.dp_returns_pars = True
            self.use_res_scale = use_res_scale
            self.res_gate = nn.Parameter(torch.tensor(float(res_scale_init)))
        elif dp_name in ["EarthformerUNet","InceptionUNet","LightweightInceptionUNet","SimVP"]:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        # 放在 if/elif 之后加一行：默认非 dual
        if not hasattr(self, "dp_is_dual"):
            self.dp_is_dual = False
        if not hasattr(self, "dp_returns_pars"):
            self.dp_returns_pars = False
        
        # Refinement 分支
        if rf_name == "LightweightRefinementNet":
            model_cfg = rf_cfg[rf_name]
            self.refine_net = LightweightRefinementNet(channels=model_cfg['channels'])
        elif rf_name == "WarpSharpen":
            model_cfg = rf_cfg[rf_name] 
            self.refine_net = WarpSharpen(
                channels=model_cfg['channles'],
                base_ch=model_cfg['base_ch'],
                num_blocks=model_cfg['num_blocks'],
                res_scale=model_cfg['res_scale']
            )
        elif rf_name == "WarpSharpen_VAE":
            model_cfg = rf_cfg[rf_name]
            self.refine_net = WarpSharpen_VAE(
                in_channels=model_cfg['in_channels'],
                out_channels=model_cfg['out_channels'],
                down_block_types=model_cfg['down_block_types'],
                up_block_types=model_cfg['up_block_types'],
                block_out_channels=model_cfg['block_out_channels'],
                layers_per_block=model_cfg['layers_per_block'],
                act_fn=model_cfg['act_fn'],
                latent_channels=model_cfg['latent_channels'],
                norm_num_groups=model_cfg['norm_num_groups'],
            )
        else:
            self.refine_net = nn.Identity()
            
        assert dp_mode in ('parallel','autoregrad')
        assert rf_mode in ('parallel','autoregrad')
        self.dp_mode = dp_mode
        self.rf_mode = rf_mode
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.alpha = float(alpha)
        self.advect_mode = advect_mode
        self.ode_method = ode_method
        self.ode_substeps = ode_substeps

        # --- splat 相关配置 ---
        self.use_splat = bool(use_splat)
        self.splat_type = splat_type
        self.splat_alpha = float(splat_alpha)
        self.splat_blend = float(splat_blend)
        self.use_gate = bool(use_gate)
        self.logit_net = LogitNet(in_ch=2, base_ch=16) if self.use_splat else None
        self.gate = Gate2to1() if (self.use_splat and self.use_gate) else None
        
        # jacobi diffusion 相关配置
        self.use_implicit_diffusion = use_implicit_diffusion # 开关
        self.jacobi_iters = jacobi_iters # 1–3 通常够
        self.D_max = D_max # 最大扩散系数
        self.k_edge = k_edge # 梯度调制阈值
        self.diff_dt = diff_dt
        self.diff_dx = diff_dx
        self.D_const = D_const # 如果想固定 D，可赋值 torch.tensor(0.1)

        print(f"======================= RFDPIC Working Mode =======================")
        print(f"motion predictor: {dp_name}" )
        print(f"refinement net: {rf_name}")
        print(f"displacement predictor: {dp_mode}")
        print(f"refinement network: {rf_mode}")
        print(f"advect mode: {advect_mode}")
        if advect_mode == 'semi-lagrangian':
            print(f"interpolation mode: {interpolation_mode}")
            print(f"padding mode: {padding_mode}")
        elif advect_mode == 'ode':
            print(f"ode method: {ode_method}")
            print(f"ode substeps: {ode_substeps}")
        else: 
            raise ValueError(f"未知 advect_mode: {advect_mode}")
        if self.use_splat:
            print(f"use splat: {self.use_splat} (type={self.splat_type}, alpha={self.splat_alpha})")
            print(f"splat fusion: {'GateNet' if self.use_gate else f'fixed w={self.splat_blend:.2f}'}")
        if self.use_implicit_diffusion:
            print(f"implicit diffusion: True (jacobi iters={self.jacobi_iters}, D_max={self.D_max}, k_edge={self.k_edge})")
        print(f"===================================================================")
    
    def get_base_config(self, config_file: str):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    # def forward(self, past_frames):
    #     """
    #     Args:
    #         past_frames: [B, 6, C, H, W]
    #     Returns:
    #         future_frames:       [B, 6, C, H, W]
    #         displacement_fields: [B, 6, C, H, W, 2]
    #         residual_fields:     [B, 6, C, H, W] 或 None
    #         pars:                dict 或 None（仅 Global 版本返回）
    #     """
    #     B, T, C, H, W = past_frames.shape
    #     last_frame = past_frames[:, -1]  # [B, C, H, W]

    #     if self.dp_is_dual:
    #         out = self.displacement_net(past_frames)
    #         if self.dp_returns_pars:
    #             displacement_fields, residual_fields, pars = out
    #         else:
    #             displacement_fields, residual_fields = out
    #             pars = None
    #     else:
    #         displacement_fields = self.displacement_net(past_frames)
    #         residual_fields, pars = None, None

    #     future_frames = self._warp(last_frame, displacement_fields, residual_fields)

    #     if self.rf_mode == 'parallel':
    #         future_frames = self.refine_net(future_frames)

    #     return future_frames, displacement_fields, residual_fields, pars

    def forward(self, past_frames):
        """
        Args:
            past_frames: [B, 6, C, H, W]
        Returns:
            future_frames:       [B, 6, C, H, W]
            displacement_fields: [B, 6, C, H, W, 2]
            residual_fields:     [B, 6, C, H, W] 或 None
            pars:                dict 或 None（全局参数）
            dyn_feats:           dict 或 None（Fusion 网络特征）
        """
        B, T, C, H, W = past_frames.shape
        last_frame = past_frames[:, -1]

        dyn_feats = None  # 预定义

        # ========== 通用预测器接口 ==========
        out = self.displacement_net(past_frames)

        # ✅ 新版 Fusion 网络输出为 dict
        if isinstance(out, dict):
            displacement_fields = out["v"]
            residual_fields = out.get("s", None)
            pars = out.get("pars", None)
            dyn_feats = out.get("dyn_feats", None)
        # ✅ 老版 Dual_Rotation_Grad 输出为 tuple
        elif isinstance(out, (list, tuple)):
            if self.dp_returns_pars:
                displacement_fields, residual_fields, pars = out
            else:
                displacement_fields, residual_fields = out
                pars = None
        else:
            raise TypeError(f"未知 displacement_net 输出类型: {type(out)}")

        # ========== 半拉格朗日/ODE 传播 ==========
        future_frames = self._warp(last_frame, displacement_fields, residual_fields)

        if self.rf_mode == 'parallel':
            future_frames = self.refine_net(future_frames)

        return future_frames, displacement_fields, residual_fields, pars, dyn_feats
    
    # def _warp(self, last_frame, displacement_fields, residual_fields=None):
    #     B, C, H, W = last_frame.shape
    #     T = displacement_fields.size(1)
    #     device = last_frame.device

    #     # base grid（保持原样）
    #     grid_y, grid_x = torch.meshgrid(
    #         torch.arange(H, dtype=torch.float32, device=device),
    #         torch.arange(W, dtype=torch.float32, device=device),
    #         indexing='ij'
    #     )
    #     base_grid = torch.stack([grid_x, grid_y], dim=-1)                    # [H,W,2]
    #     base_grid = base_grid[None, None, None].expand(B, T, C, -1, -1, -1)  # [B,T,C,H,W,2]

    #     future_frames = []
    #     cur_frame = last_frame

    #     for t in range(T):
    #         # stop-grad
    #         state_in = cur_frame.detach()

    #         # --- 构建采样网格 ---
    #         new_pos = base_grid[:, t] + displacement_fields[:, t]            # [B,C,H,W,2]
    #         new_pos_x = 2.0 * new_pos[..., 0] / (W - 1) - 1.0
    #         new_pos_y = 2.0 * new_pos[..., 1] / (H - 1) - 1.0
    #         new_pos = torch.stack([new_pos_x, new_pos_y], dim=-1)            # 重新 stack，确保连续
    #         grid_t  = new_pos.reshape(B * C, H, W, 2)                        # ← 用 reshape

    #         # 输入帧
    #         input_t = state_in.contiguous().reshape(B * C, 1, H, W)          # ← contiguous()+reshape

    #         # backward warp
    #         I_bw = F.grid_sample(
    #             input_t, grid_t,
    #             mode=self.interpolation_mode,
    #             padding_mode=self.padding_mode,
    #             align_corners=True
    #         ).reshape(B, C, H, W)

    #         # forward splat（如果启用）
    #         if self.use_splat:
    #             # [B,T,C,H,W,2] -> [B*C,2,H,W]
    #             flow_tc = displacement_fields[:, t].permute(0,1,4,2,3).contiguous() \
    #                                                 .reshape(B * C, 2, H, W)
    #             I_in = input_t  # 已是 [B*C,1,H,W]

    #             # logits 输入 [B,C,2,H,W]
    #             logit_in = torch.stack([state_in, I_bw], dim=2).contiguous()
    #             metric_logit = self.logit_net(logit_in)                      # [B,C,H,W]
    #             metric_logit = metric_logit.contiguous().reshape(B * C, 1, H, W)

    #             I_fw = _forward_splat(
    #                 I=I_in, flow=flow_tc, metric_logit=metric_logit,
    #                 splat_type=self.splat_type, alpha=self.splat_alpha
    #             ).reshape(B, C, H, W)

    #             if self.use_gate and (self.gate is not None):
    #                 gate_in = torch.stack([
    #                     I_fw.mean(1, keepdim=True).squeeze(1),
    #                     I_bw.mean(1, keepdim=True).squeeze(1)
    #                 ], dim=1).contiguous()                                   # [B,2,H,W]
    #                 w = self.gate(gate_in)                                   # [B,1,H,W]
    #                 x_adv = w * I_fw + (1.0 - w) * I_bw
    #             else:
    #                 x_adv = self.splat_blend * I_fw + (1.0 - self.splat_blend) * I_bw
    #         else:
    #             x_adv = I_bw

    #         # 残差叠加
    #         if residual_fields is not None:
    #             s_t = residual_fields[:, t]                        # [B,C,H,W]
    #             if getattr(self, 'use_res_gate', False):
    #                 gate_tc = self.res_gate[:, t]                  # [1,C,1,1]
    #                 x_pp = x_adv + gate_tc * s_t                  # 广播乘
    #             else:
    #                 x_pp = x_adv + s_t
    #         else:
    #             x_pp = x_adv

    #         future_frame = self.alpha * state_in + (1.0 - self.alpha) * x_pp

    #         if self.dp_mode == 'autoregrad':
    #             if self.rf_mode == 'autoregrad':
    #                 if 'WarpSharpen' in str(type(self.refine_net)):
    #                     future_frame = self.refine_net(state_in, future_frame)
    #                 else:
    #                     future_frame = self.refine_net(future_frame)
    #             cur_frame = future_frame

    #         future_frames.append(future_frame)

    #     return torch.stack(future_frames, dim=1)
    
    def _warp(self, last_frame, displacement_fields, residual_fields=None):
        """
        Args:
            last_frame: [B, C, H, W]
            displacement_fields: [B, T, C, H, W, 2]
            residual_fields: [B, T, C, H, W] or None
            advect_mode: 'semi-lagrangian' | 'ode'
        Returns:
            future_frames: [B, T, C, H, W]
        """
        if self.advect_mode == 'semi-lagrangian':
            B, C, H, W = last_frame.shape
            T = displacement_fields.size(1)
            device = last_frame.device

            # base grid（保持原样）
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, dtype=torch.float32, device=device),
                torch.arange(W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            base_grid = torch.stack([grid_x, grid_y], dim=-1)                    # [H,W,2]
            base_grid = base_grid[None, None, None].expand(B, T, C, -1, -1, -1)  # [B,T,C,H,W,2]

            future_frames = []
            cur_frame = last_frame

            for t in range(T):
                # stop-grad
                state_in = cur_frame.detach()

                # --- 构建采样网格 ---
                new_pos = base_grid[:, t] + displacement_fields[:, t]            # [B,C,H,W,2]
                new_pos_x = 2.0 * new_pos[..., 0] / (W - 1) - 1.0
                new_pos_y = 2.0 * new_pos[..., 1] / (H - 1) - 1.0
                new_pos = torch.stack([new_pos_x, new_pos_y], dim=-1)            # 重新 stack，确保连续
                grid_t  = new_pos.reshape(B * C, H, W, 2)                        # ← 用 reshape

                # 输入帧
                input_t = state_in.contiguous().reshape(B * C, 1, H, W)          # ← contiguous()+reshape

                # backward warp
                I_bw = F.grid_sample(
                    input_t, grid_t,
                    mode=self.interpolation_mode,
                    padding_mode=self.padding_mode,
                    align_corners=True
                ).reshape(B, C, H, W)

                # forward splat（如果启用）
                if self.use_splat:
                    # [B,T,C,H,W,2] -> [B*C,2,H,W]
                    flow_tc = displacement_fields[:, t].permute(0,1,4,2,3).contiguous() \
                                                        .reshape(B * C, 2, H, W)
                    I_in = input_t  # 已是 [B*C,1,H,W]

                    # logits 输入 [B,C,2,H,W]
                    logit_in = torch.stack([state_in, I_bw], dim=2).contiguous()
                    metric_logit = self.logit_net(logit_in)                      # [B,C,H,W]
                    metric_logit = metric_logit.contiguous().reshape(B * C, 1, H, W)

                    I_fw = _forward_splat(
                        I=I_in, flow=flow_tc, metric_logit=metric_logit,
                        splat_type=self.splat_type, alpha=self.splat_alpha
                    ).reshape(B, C, H, W)

                    if self.use_gate and (self.gate is not None):
                        gate_in = torch.stack([
                            I_fw.mean(1, keepdim=True).squeeze(1),
                            I_bw.mean(1, keepdim=True).squeeze(1)
                        ], dim=1).contiguous()                                   # [B,2,H,W]
                        w = self.gate(gate_in)                                   # [B,1,H,W]
                        x_adv = w * I_fw + (1.0 - w) * I_bw
                    else:
                        x_adv = self.splat_blend * I_fw + (1.0 - self.splat_blend) * I_bw
                else:
                    x_adv = I_bw

                # 残差叠加
                if residual_fields is not None:
                    s_t = residual_fields[:, t]                        # [B,C,H,W]
                    if getattr(self, 'use_res_gate', False):
                        gate_tc = self.res_gate[:, t]                  # [1,C,1,1]
                        x_pp = x_adv + gate_tc * s_t                  # 广播乘
                    else:
                        x_pp = x_adv + s_t
                else:
                    x_pp = x_adv

                # =====================================================
                # === 半隐式扩散 Jacobi 修正 ===
                # =====================================================
                if getattr(self, "use_implicit_diffusion", False):
                    # Step 1: 构造 D_map
                    if hasattr(self, "D_const") and self.D_const is not None:
                        D_map = self.D_const.to(x_pp.device).to(x_pp.dtype).expand_as(x_pp)
                    else:
                        # 纹理自适应 D_map
                        gx = x_pp[..., :, 1:] - x_pp[..., :, :-1]
                        gy = x_pp[..., 1:, :] - x_pp[..., :-1, :]
                        gx = F.pad(gx, (0,1,0,0))
                        gy = F.pad(gy, (0,0,0,1))
                        grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
                        c = torch.exp(-(grad_mag / getattr(self, "k_edge", 0.1))**2)
                        D_map = c * getattr(self, "D_max", 0.2)

                    dt = getattr(self, "diff_dt", 1.0)
                    dx = getattr(self, "diff_dx", 1.0)
                    jacobi_iters = getattr(self, "jacobi_iters", 2)
                    x_pp = jacobi_diffuse_iso(x_pp, D_map, dt=dt, dx=dx, iters=jacobi_iters)

                future_frame = self.alpha * state_in + (1.0 - self.alpha) * x_pp

                if self.dp_mode == 'autoregrad':
                    if self.rf_mode == 'autoregrad':
                        if 'WarpSharpen' in str(type(self.refine_net)):
                            future_frame = self.refine_net(state_in, future_frame)
                        else:
                            future_frame = self.refine_net(future_frame)
                    cur_frame = future_frame

                future_frames.append(future_frame)

            return torch.stack(future_frames, dim=1)

        # ===============   ODE 版本（推荐模式）   ===================
        elif self.advect_mode == 'ode':
            B, T, C, H, W, _ = displacement_fields.shape
            device = last_frame.device
            dtype  = last_frame.dtype

            # === (1) 保留每通道独立速度场 ===
            # [B,T,C,H,W,2] -> [B,T,C,2,H,W]
            v_key = displacement_fields.permute(0, 1, 2, 5, 3, 4).contiguous()

            # === (2) 源汇场 ===
            r_key = residual_fields if residual_fields is not None else None

            # === (3) 可学习缩放 ===
            scale = getattr(self, "res_gate", torch.tensor(1.0, device=device, dtype=dtype))
            if not getattr(self, "use_res_scale", False):
                scale = torch.tensor(1.0, device=device, dtype=dtype)

            # === (4) 时间轴 ===
            key_times = torch.linspace(0, T - 1, steps=T, device=device, dtype=dtype)
            N = getattr(self, "ode_substeps", 1)
            fine_steps = (T - 1) * N + 1
            integ_times = torch.linspace(0, T - 1, steps=fine_steps, device=device, dtype=dtype)

            # === (5) 构造 RHS 并积分 ===
            ode_rhs = AdvectionWithSource(v_key, r_key, key_times, scale)
            I_fine = odeint(ode_rhs, last_frame, integ_times, method=getattr(self, "ode_method", "rk4"))
            pick_idx = torch.arange(0, fine_steps, step=N, device=device)
            I_all = I_fine.index_select(0, pick_idx)  # [T,B,C,H,W]
            I_seq = I_all.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,C,H,W]
            return I_seq

        else:
            raise ValueError(f"Unknown advect_mode: {self.advect_mode}")
