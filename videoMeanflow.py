import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            # 1. 调整 view 以便在 (B, T, C, H, W) 的 C 维度上广播
            self.mean = torch.tensor(mean).view(1, 1, -1, 1, 1)
            self.std = torch.tensor(std).view(1, 1, -1, 1, 1)
        
        # 你的 dataset_btchw.py 已经做了 [0, 1] 归一化
        # 'minmax' 模式将把它转换为 [-1, 1]，这是 DiT 训练的常见做法

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
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
    """
    Adaptive L2 loss
    Args:
        error: Tensor of shape (B, T, C, H, W)
    """
    # 2. 维度修改为 (1, 2, 3, 4) 以匹配 5D 视频张量
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
        # 3. 更新维度参数
        self.channels = channels
        self.time_dim = time_dim
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.num_classes = num_classes # 保留但不再使用
        self.use_cond = True # 现在总是有条件的

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

    def loss(self, model, x, c=None): # x=x_future, c=c_past
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        # 4. 维度修改为 "b -> b 1 1 1 1" 以匹配 (B, T, C, H, W)
        t_ = rearrange(t, "b -> b 1 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1 1").detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x) # x (x_future) 被归一化到 [-1, 1]

        z = (1 - t_) * x + t_ * e
        v = e - x

        if c is not None:
            assert self.cfg_ratio is not None
            
            # 5. --- 重写 CFG 逻辑 ---
            # 'uncond' 不再是类别标签，而是全零的视频张量
            uncond = torch.zeros_like(c)
            
            # mask 应该是 (B,) 形状
            cfg_mask = torch.rand(batch_size, device=device) < self.cfg_ratio
            
            # 将 mask 扩展到 5D 以便 torch.where 广播
            cfg_mask_expanded = rearrange(cfg_mask, "b -> b 1 1 1 1")
            
            # c 是 c_past (真实条件) 和 uncond (零张量) 之间的选择
            c_cond = torch.where(cfg_mask_expanded, uncond, c)
            
            if self.w is not None:
                with torch.no_grad():
                    # 使用 uncond (零张量) 作为 y 传入
                    u_t = model(z, t, t, uncond)
                
                v_hat = self.w * v + (1 - self.w) * u_t
                
                if self.cfg_uncond == 'v':
                    # 将 mask 扩展到 v (即 z) 的 5D 维度
                    cfg_mask_v = rearrange(cfg_mask, "b -> b 1 1 1 1").bool()
                    v_hat = torch.where(cfg_mask_v, v, v_hat)
            else:
                v_hat = v
        else:
            # 如果没有提供 c (c_past)，则无法训练
            raise ValueError("Conditional video 'c' (c_past) must be provided.")

        # forward pass
        # model_partial 使用 c_cond (可能是 c_past 或 uncond)
        model_partial = partial(model, y=c_cond)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
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

    # 6. --- 替换采样函数 ---
    @torch.no_grad()
    def sample_prediction(self, model, c_past, sample_steps=5, device='cuda'):
        """
        使用 c_past (过去视频) 作为条件来采样 x_future (未来视频)
        c_past: (B, T_past, C, H, W)
        """
        model.eval()
        batch_size = c_past.shape[0]

        # 从 c_past 获取维度，或从 self 获取目标维度
        # 假设 c_past 和 z 具有相同的 T, C, H, W
        z = torch.randn(batch_size, 
                        self.time_dim, 
                        self.channels, 
                        self.height_dim, 
                        self.width_dim,
                        device=device)

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            # 维度修改为 5D
            t_ = rearrange(t, "b -> b 1 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1 1").detach().clone()

            # 使用 c_past 作为条件 y 传入
            v = model(z, t, r, c_past)
            z = z - (t_-r_) * v

        z = self.normer.unnorm(z)
        return z