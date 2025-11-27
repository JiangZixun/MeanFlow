# videoMeanflow.py
import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


class Normalizer:
    # --- 保持不变 ---
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode
        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(1, 1, -1, 1, 1)
            self.std = torch.tensor(std).view(1, 1, -1, 1, 1)
    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)
    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


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
        num_classes=None, # 废弃
        normalizer=['minmax', None, None],
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
        self.num_classes = num_classes 
        self.use_cond = True 

        self.normer = Normalizer.from_list(normalizer)
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

    # def loss(self, model, x, c=None): # x=x_future
    #     """
    #     🔴 核心修改：
    #     c 不再是单个张量，而是一个元组 (c_start, c_cond)
    #     c_start: 流的起点 (c_past)
    #     c_cond: 模型的条件 (c_rfdpic)
    #     """
    #     if not isinstance(c, (tuple, list)) or len(c) != 2:
    #         raise ValueError("`c` must be a tuple `(c_start, c_cond)`."
    #                          " `c_start` is the past distribution (t=1), "
    #                          " `c_cond` is the external condition (e.g., RFDPIC pred).")
        
    #     c_start, c_cond = c
        
    #     # 确保 x 和 c_start 形状相同
    #     if x.shape != c_start.shape:
    #          raise ValueError(f"x (x_future) and c_start (c_past) must have the same shape. "
    #                          f"Got x: {x.shape} and c_start: {c_start.shape}")

    #     batch_size = x.shape[0]
    #     device = x.device

    #     t, r = self.sample_t_r(batch_size, device)
    #     t_ = rearrange(t, "b -> b 1 1 1 1").detach().clone()
    #     r_ = rearrange(r, "b -> b 1 1 1 1").detach().clone()

    #     # --- 🔴 1. 修改：流的起点 e 是 c_start (c_past) ---
    #     e = self.normer.norm(c_start) # 归一化 t=1 的点
    #     x = self.normer.norm(x)       # 归一化 t=0 的点 (x_future)

    #     z = (1 - t_) * x + t_ * e
    #     # --- 🔴 2. 修改：v 是从 x 到 e 的速度 ---
    #     v = e - x
        
    #     # --- CFG 逻辑现在作用于 c_cond (c_rfdpic) ---
    #     if c_cond is not None:
    #         assert self.cfg_ratio is not None
            
    #         # 'uncond' 必须与 c_cond (c_rfdpic) 形状相同
    #         uncond = torch.zeros_like(c_cond)
            
    #         cfg_mask = torch.rand(batch_size, device=device) < self.cfg_ratio
    #         cfg_mask_expanded = rearrange(cfg_mask, "b -> b 1 1 1 1")
            
    #         # c_cond_input 是 c_rfdpic 和 uncond (零张量) 之间的选择
    #         c_cond_input = torch.where(cfg_mask_expanded, uncond, c_cond)
            
    #         if self.w is not None: # CFG 蒸馏
    #             with torch.no_grad():
    #                 u_t = model(z, t, t, uncond)
    #             v_hat = self.w * v + (1 - self.w) * u_t
    #             if self.cfg_uncond == 'v':
    #                 cfg_mask_v = rearrange(cfg_mask, "b -> b 1 1 1 1").bool()
    #                 v_hat = torch.where(cfg_mask_v, v, v_hat)
    #         else:
    #             v_hat = v
    #     else:
    #         # 如果没有 c_cond (例如 c_rfdpic=None)
    #         c_cond_input = None
    #         v_hat = v

    #     # forward pass
    #     # model_partial 使用 c_cond_input (c_rfdpic 或 uncond) 作为条件
    #     model_partial = partial(model, y=c_cond_input)
    #     jvp_args = (
    #         lambda z, t, r: model_partial(z, t, r),
    #         (z, t, r),
    #         (v_hat, torch.ones_like(t), torch.zeros_like(r)),
    #     )

    #     if self.create_graph:
    #         u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
    #     else:
    #         u, dudt = self.jvp_fn(*jvp_args)

    #     u_tgt = v_hat - (t_ - r_) * dudt
    #     error = u - stopgrad(u_tgt)
    #     loss = adaptive_l2_loss(error)
    #     mse_val = (stopgrad(error) ** 2).mean()
        
    #     return loss, mse_val

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

        e = torch.randn_like(x) # 🔴 将起点 e 设为标准高斯噪声
        x = self.normer.norm(x)       

        z = (1 - t_) * x + t_ * e
        v = e - x # Ground Truth Velocity
        
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

    # @torch.no_grad()
    # def sample_prediction(self, model, c_past_and_cond, sample_steps=5, device='cuda'):
    #     """
    #     🔴 核心修改：
    #     c_past_and_cond 是一个元组 (c_start, c_cond)
    #     c_start: 流的起点 (c_past)
    #     c_cond: 模型的条件 (c_rfdpic)
    #     """
    #     if not isinstance(c_past_and_cond, (tuple, list)) or len(c_past_and_cond) != 2:
    #         raise ValueError("`c_past_and_cond` must be a tuple `(c_start, c_cond)`.")
        
    #     c_start, c_cond = c_past_and_cond

    #     model.eval()
    #     batch_size = c_start.shape[0]

    #     # --- 🔴 3. 修改：采样的起点 z 是 c_start (c_past) ---
    #     z = self.normer.norm(c_start) # 归一化 t=1 的点

    #     t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

    #     for i in range(sample_steps):
    #         t = torch.full((z.size(0),), t_vals[i], device=device)
    #         r = torch.full((z.size(0),), t_vals[i + 1], device=device)

    #         t_ = rearrange(t, "b -> b 1 1 1 1").detach().clone()
    #         r_ = rearrange(r, "b -> b 1 1 1 1").detach().clone()

    #         # --- 🔴 4. 修改：使用 c_cond (c_rfdpic) 作为条件 y 传入 ---
    #         v = model(z, t, r, c_cond)
    #         z = z - (t_-r_) * v

    #     z = self.normer.unnorm(z)
    #     return z
    
    @torch.no_grad()
    def sample_prediction(self, model, c_past_and_cond, sample_steps=5, device='cuda'):
        # ... (前序检查保持不变) ...
        c_start, c_cond = c_past_and_cond
        model.eval()
        
        z = torch.randn_like(c_start)
        
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

        z = self.normer.unnorm(z)
        return z