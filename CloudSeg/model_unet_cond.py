import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, nfreq: int = 256):
        super().__init__()
        self.nfreq = nfreq
        self.mlp = nn.Sequential(
            nn.Linear(nfreq, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor):
        t = t * 1000.0
        return self.mlp(self.timestep_embedding(t, self.nfreq))


class _InjectBlock(nn.Module):
    def __init__(self, out_channels: int, time_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        b = x.shape[0]
        return x + self.proj(t_emb).view(b, -1, 1, 1)


class UnetCond(nn.Module):
    """Official hzy Unet backbone with minimal conditional/time extension for MeanFlow."""

    def __init__(
        self,
        image_channels: int = 16,
        num_classes: int = 10,
        time_dim: int = 128,
        encoder_channels=None,
        bottleneck_channels: int = 1024,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must be a list of 4 ints, e.g. [64,128,256,512]")

        c1, c2, c3, c4 = encoder_channels
        cb = bottleneck_channels
        in_channels = image_channels + num_classes
        self.num_classes = num_classes

        self.t_embed = TimestepEmbedder(time_dim)
        self.r_embed = TimestepEmbedder(time_dim)

        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c1 // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1 // 2), nn.ReLU(),
            nn.Conv2d(c1 // 2, c1, 3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(),
        )
        self.stage_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(),
        )
        self.stage_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(),
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(),
        )
        self.stage_4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1),
            nn.BatchNorm2d(c4), nn.ReLU(),
            nn.Conv2d(c4, c4, 3, padding=1),
            nn.BatchNorm2d(c4), nn.ReLU(),
        )
        self.stage_5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c4, cb, 3, padding=1),
            nn.BatchNorm2d(cb), nn.ReLU(),
            nn.Conv2d(cb, cb, 3, padding=1),
            nn.BatchNorm2d(cb), nn.ReLU(),
        )

        self.inj1 = _InjectBlock(c1, time_dim)
        self.inj2 = _InjectBlock(c2, time_dim)
        self.inj3 = _InjectBlock(c3, time_dim)
        self.inj4 = _InjectBlock(c4, time_dim)
        self.inj5 = _InjectBlock(cb, time_dim)

        self.upsample_4 = nn.ConvTranspose2d(cb, c4, kernel_size=4, stride=2, padding=1)
        self.upsample_3 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1)
        self.upsample_2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.upsample_1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)

        self.stage_up_4 = nn.Sequential(
            nn.Conv2d(c4 + c4, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(),
            nn.Conv2d(c4, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(),
        )
        self.stage_up_3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),
        )
        self.stage_up_2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
        )
        self.stage_up_1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
        )

        self.final = nn.Conv2d(c1, num_classes, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, y: torch.Tensor):
        tr = self.t_embed(t) + self.r_embed(r)
        x = torch.cat([z, y], dim=1).float()

        s1 = self.inj1(self.stage_1(x), tr)
        s2 = self.inj2(self.stage_2(s1), tr)
        s3 = self.inj3(self.stage_3(s2), tr)
        s4 = self.inj4(self.stage_4(s3), tr)
        s5 = self.inj5(self.stage_5(s4), tr)

        u4 = self.stage_up_4(torch.cat([self.upsample_4(s5), s4], dim=1))
        u3 = self.stage_up_3(torch.cat([self.upsample_3(u4), s3], dim=1))
        u2 = self.stage_up_2(torch.cat([self.upsample_2(u3), s2], dim=1))
        u1 = self.stage_up_1(torch.cat([self.upsample_1(u2), s1], dim=1))

        return self.final(u1)
