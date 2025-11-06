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
    def __init__(
        self,
        channels=8,
        time_dim=6,
        height_dim=256,
        width_dim=256,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        # --- 🔴 移除所有 CFG 相关参数 ---
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        self.height_dim = height_dim
        self.width_dim = width_dim
        
        # --- 🔴 移除了 num_classes, use_cond ---
        # --- 🔴 移除了 cfg_ratio, cfg_scale, cfg_uncond ---

        self.normer = Normalizer.from_list(normalizer)
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
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

    def loss(self, model, x, c=None): # x=x_future, c=c_past
        """
        🔴 核心修改：
        - e (t=1 的点) 不再是噪声，而是 c (c_past)
        - v (目标速度) 变为 c - x
        - 移除了所有 CFG 逻辑
        """
        if c is None:
            raise ValueError("c (c_past) must be provided as the starting distribution (t=1)")
            
        # 假设：x_future (x) 和 c_past (c) 具有完全相同的形状
        if x.shape != c.shape:
            raise ValueError(f"In this mode, x (x_future) and c (c_past) must have the same shape. "
                             f"Got x: {x.shape} and c: {c.shape}")
                             
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1 1").detach().clone()

        # --- 🔴 核心修改在这里 ---
        # e 是 t=1 的分布 (c_past)，x 是 t=0 的分布 (x_future)
        e = c # e 是 c_past
        
        # 归一化两个端点
        x = self.normer.norm(x) # x (x_future) 被归一化到 [-1, 1]
        e = self.normer.norm(e) # e (c_past) 也被归一化到 [-1, 1]
        
        # z 是 x 和 e 之间的插值
        z = (1 - t_) * x + t_ * e
        
        # v 是从 x 到 e 的恒定速度向量
        v = e - x
        # --- 修改结束 ---

        # --- 🔴 移除所有 CFG 逻辑 ---
        # v_hat 现在就是 v
        v_hat = v

        # forward pass
        # --- 🔴 修改：模型不再接收 y=c_cond ---
        # model_partial = partial(model, y=c_cond) # <-- 移除
        
        jvp_args = (
            # --- 🔴 修改：直接调用 model，不带 y ---
            lambda z, t, r: model(z, t, r),
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

    # 6. --- 🔴 替换采样函数 ---
    @torch.no_grad()
    def sample_prediction(self, model, c_past, sample_steps=5, device='cuda'):
        """
        🔴 核心修改：
        - 采样的起点 z 不再是高斯噪声，而是归一化后的 c_past
        - 模型调用不再传入 y=c_past
        """
        model.eval()
        batch_size = c_past.shape[0]

        # --- 🔴 核心修改在这里 ---
        # 采样的起点 (t=1) 是 c_past
        # 我们必须像训练时一样对其进行归一化
        z = self.normer.norm(c_past)
        # --- 修改结束 ---

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1 1").detach().clone()

            # --- 🔴 核心修改在这里 ---
            # 模型不再需要条件 c_past，因为它已经是流的一部分
            v = model(z, t, r) # <-- 移除了 c_past
            # --- 修改结束 ---
            
            z = z - (t_-r_) * v

        # 最后一步 unnorm 保持不变
        z = self.normer.unnorm(z)
        return z