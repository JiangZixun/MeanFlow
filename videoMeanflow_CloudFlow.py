# videoMeanflow.py
import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    # --- 保持不变 ---
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3, 4), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    # --- __init__ 保持您提供的版本不变 ---
    def __init__(
        self,
        channels=8,
        time_dim=6,
        height_dim=256,
        width_dim=256,
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.use_cond = True 

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    # sample_t_r (无修改)
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None): # x=x_future
        # ... (前序检查和 t, r 采样代码保持不变) ...
        # ... (直到 defined e 和 z) ...

        if not isinstance(c, (tuple, list)) or len(c) != 2:
            raise ValueError("c must be (c_start, c_cond)")
        c_start, c_cond = c
        
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1 1").detach().clone()

        z = (1 - t_) * x + t_ * c_start
        v = c_start - x # Ground Truth Velocity
        
        # --- 🔴 核心修改开始: x-prediction 转换 ---
        
        # 定义一个 wrapper，将模型输出(x)转换为速度(v)
        # 这样 jvp 计算的导数也是基于 velocity 的，保持原有一致性
        def get_velocity_from_x_model(z_in, t_in, r_in, cond_in):
            # 1. 模型直接预测干净的 x (Target) [cite: 11, 48]
            x_pred = model(z_in, t_in, r_in, cond_in)
            
            # 2. 根据公式 v = (z - x) / t 将 x 转换为 v
            #  论文建议 clip 分母以防止除零
            # 注意：你的代码中 t=0 是 clean，所以分母是 t
            t_in_clamped = t_in.clamp(min=1e-3) 
            t_expand = rearrange(t_in_clamped, "b -> b 1 1 1 1")
            
            v_pred = (z_in - x_pred) / t_expand
            return v_pred

        # --- 处理 CFG (在 Velocity 空间进行混合) ---
        if c_cond is not None:
            assert self.cfg_ratio is not None
            uncond = torch.zeros_like(c_cond)
            cfg_mask = torch.rand(batch_size, device=device) < self.cfg_ratio
            cfg_mask_expanded = rearrange(cfg_mask, "b -> b 1 1 1 1")
            c_cond_input = torch.where(cfg_mask_expanded, uncond, c_cond)
            
            # 计算目标 v_hat (用于蒸馏或作为 jvp 的目标)
            if self.w is not None: 
                with torch.no_grad():
                    # 获取 unconditional 的速度
                    v_uncond = get_velocity_from_x_model(z, t, t, uncond)
                
                # 这里 v 是 Ground Truth。
                # 如果要做 CFG Distill，通常是 mix(v_cond_pred, v_uncond_pred)
                # 但原代码逻辑是 mix(GT_v, Model_uncond_v)。保持原逻辑：
                v_hat = self.w * v + (1 - self.w) * v_uncond
                
                if self.cfg_uncond == 'v':
                     cfg_mask_v = rearrange(cfg_mask, "b -> b 1 1 1 1").bool()
                     v_hat = torch.where(cfg_mask_v, v, v_hat)
            else:
                v_hat = v
        else:
            c_cond_input = None
            v_hat = v

        # --- JVP 计算 ---
        # 将 wrapper 传入 jvp，而不是原始 model
        # 这样 u 就是 velocity, dudt 也是 velocity 对时间的导数
        model_partial_v = partial(get_velocity_from_x_model, cond_in=c_cond_input)
        
        jvp_args = (
            lambda z, t, r: model_partial_v(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()
        
        return loss, mse_val
    
    @torch.no_grad()
    def sample_prediction(self, model, c_past_and_cond, sample_steps=5, device='cuda'):
        # ... (前序检查保持不变) ...
        c_start, c_cond = c_past_and_cond
        model.eval()
        
        z = c_start
        
        # t 从 1.0 (噪声/过去) -> 0.0 (干净/未来)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t_curr = t_vals[i]
            t_next = t_vals[i + 1]
            
            t = torch.full((z.size(0),), t_curr, device=device)
            r = torch.full((z.size(0),), t_next, device=device)
            
            # --- 🔴 核心修改: 采样时的转换 ---
            # 1. 模型预测 x (干净数据)
            x_pred = model(z, t, r, c_cond)
            
            # 2. 转换为速度 v = (z - x) / t
            # 处理 t=0 的情况 (最后一元)，防止除零
            if t_curr < 1e-5:
                # 如果当前已经是 t=0 (理论上循环最后一步是到达0，不是从0开始)，
                # 或者非常接近0，直接认为已经到达目标。
                # 但由于这里是 Euler 步进 z -> z_next，如果在 t=0 处计算 v 可能会不稳定。
                # 在 Euler 积分中：z_next = z + (t_next - t_curr) * v
                # 代入 v = (z - x_pred) / t_curr
                # z_next = z + (t_next - t_curr) * (z - x_pred) / t_curr
                # 当 t_curr 很小时，数值不稳定。
                # 策略：Clip t 
                denom = max(t_curr, 1e-3)
                v = (z - x_pred) / denom
            else:
                v = (z - x_pred) / t_curr
                
            t_ = rearrange(t, "b -> b 1 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1 1")
            
            # 3. 更新 z (Standard Euler step)
            # z_next = z + dt * v = z + (r - t) * v
            z = z + (r_ - t_) * v 

        return z