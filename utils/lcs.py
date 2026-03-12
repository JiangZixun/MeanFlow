import torch
import torch.nn.functional as F
from typing import Optional


def _to_bt_hw2(displacement_fields: torch.Tensor) -> torch.Tensor:
    """
    Normalize displacement tensor to shape [B, T, H, W, 2].
    Supports:
    - [B, T, C, H, W, 2]
    - [B, T, H, W, 2]
    - [T, C, H, W, 2]
    - [T, H, W, 2]
    """
    if displacement_fields.ndim == 6:
        # [B, T, C, H, W, 2] -> average over channel C
        return displacement_fields.mean(dim=2)
    if displacement_fields.ndim == 5:
        # [B, T, H, W, 2]
        if displacement_fields.shape[-1] != 2:
            raise ValueError(f"Expected last dim=2, got {displacement_fields.shape}")
        return displacement_fields
    if displacement_fields.ndim == 4:
        # [T, H, W, 2] -> [1, T, H, W, 2]
        if displacement_fields.shape[-1] != 2:
            raise ValueError(f"Expected last dim=2, got {displacement_fields.shape}")
        return displacement_fields.unsqueeze(0)
    raise ValueError(f"Unsupported displacement shape: {displacement_fields.shape}")


def _spatial_gradients(x: torch.Tensor):
    """
    x: [B, T, H, W]
    returns dx, dy with same shape
    """
    pad_x = F.pad(x, (1, 1, 0, 0), mode="replicate")
    dx = 0.5 * (pad_x[..., 2:] - pad_x[..., :-2])

    pad_y = F.pad(x, (0, 0, 1, 1), mode="replicate")
    dy = 0.5 * (pad_y[..., 2:, :] - pad_y[..., :-2, :])
    return dx, dy


def compute_lcs_weight_map(
    displacement_fields: torch.Tensor,
    ridge_quantile: float = 0.9,
    smooth_kernel: int = 5,
    smooth_sigma: float = 1.0,
    eps: float = 1e-6,
):
    """
    Build an FTLE-like LCS map from displacement fields.

    Args:
        displacement_fields: velocity/displacement tensor.
        ridge_quantile: quantile threshold for ridge mask.
        smooth_kernel: odd kernel size for gaussian smoothing.
        smooth_sigma: gaussian sigma.
    Returns:
        lcs_weight: [B, T, 1, H, W], in [0, 1]
        ridge_mask: [B, T, 1, H, W], binary float
    """
    v = _to_bt_hw2(displacement_fields)  # [B, T, H, W, 2]
    vx = v[..., 0]
    vy = v[..., 1]

    dvx_dx, dvx_dy = _spatial_gradients(vx)
    dvy_dx, dvy_dy = _spatial_gradients(vy)

    # A = I + grad(v)
    a11 = 1.0 + dvx_dx
    a12 = dvx_dy
    a21 = dvy_dx
    a22 = 1.0 + dvy_dy

    # C = A^T A (2x2)
    c11 = a11 * a11 + a21 * a21
    c12 = a11 * a12 + a21 * a22
    c22 = a12 * a12 + a22 * a22

    trace = c11 + c22
    det = c11 * c22 - c12 * c12
    delta = (trace * trace - 4.0 * det).clamp(min=0.0)
    lambda_max = 0.5 * (trace + torch.sqrt(delta + eps))

    # FTLE-like intensity
    sigma = 0.5 * torch.log(lambda_max.clamp(min=eps))
    sigma = sigma.abs()

    # Optional smoothing to stabilize ridge extraction
    if smooth_kernel > 1:
        sigma_ = sigma.reshape(-1, 1, sigma.shape[-2], sigma.shape[-1])
        radius = smooth_kernel // 2
        coords = torch.arange(-radius, radius + 1, device=sigma.device, dtype=sigma.dtype)
        g = torch.exp(-(coords ** 2) / (2 * smooth_sigma * smooth_sigma))
        g = g / g.sum()
        kx = g.view(1, 1, 1, -1)
        ky = g.view(1, 1, -1, 1)
        sigma_ = F.pad(sigma_, (radius, radius, 0, 0), mode="reflect")
        sigma_ = F.conv2d(sigma_, kx)
        sigma_ = F.pad(sigma_, (0, 0, radius, radius), mode="reflect")
        sigma_ = F.conv2d(sigma_, ky)
        sigma = sigma_.view_as(sigma)

    # Normalize per (B, T)
    bt = sigma.shape[0] * sigma.shape[1]
    sigma_flat = sigma.view(bt, -1)
    s_min = sigma_flat.min(dim=1)[0].view(sigma.shape[0], sigma.shape[1], 1, 1)
    s_max = sigma_flat.max(dim=1)[0].view(sigma.shape[0], sigma.shape[1], 1, 1)
    lcs = (sigma - s_min) / (s_max - s_min + eps)

    # Ridge mask (top quantile)
    q = float(min(max(ridge_quantile, 0.0), 1.0))
    thresh = torch.quantile(lcs.view(bt, -1), q, dim=1).view(sigma.shape[0], sigma.shape[1], 1, 1)
    ridge_mask = (lcs >= thresh).float()

    return lcs.unsqueeze(2), ridge_mask.unsqueeze(2)


def apply_lcs_conditioning(
    cond: torch.Tensor,
    lcs_weight: torch.Tensor,
    alpha: float = 0.3,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    cond: [B, T, C, H, W]
    lcs_weight: [B, T, 1, H, W]
    """
    if cond.ndim != 5 or lcs_weight.ndim != 5:
        raise ValueError(f"Unexpected shapes cond={cond.shape}, lcs_weight={lcs_weight.shape}")
    if cond.shape[0] != lcs_weight.shape[0] or cond.shape[1] != lcs_weight.shape[1]:
        raise ValueError(f"Batch/time mismatch cond={cond.shape}, lcs_weight={lcs_weight.shape}")

    cond_lcs = cond * (1.0 + alpha * lcs_weight)
    if clamp_min is not None and clamp_max is not None:
        cond_lcs = cond_lcs.clamp(min=clamp_min, max=clamp_max)
    return cond_lcs


def spatial_gradient_magnitude(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B, T, C, H, W]
    returns grad magnitude with same shape
    """
    if x.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {x.shape}")
    b, t, c, h, w = x.shape
    x_ = x.reshape(b * t * c, 1, h, w)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x_, (1, 1, 1, 1), mode="reflect"), sobel_x)
    gy = F.conv2d(F.pad(x_, (1, 1, 1, 1), mode="reflect"), sobel_y)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    return mag.view(b, t, c, h, w)


def ridge_weighted_gradient_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    lcs_weight: torch.Tensor,
    ridge_mask: Optional[torch.Tensor] = None,
    ridge_boost: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    pred/target: [B,T,C,H,W]
    lcs_weight: [B,T,1,H,W]
    ridge_mask: [B,T,1,H,W] or None
    """
    g_pred = spatial_gradient_magnitude(pred, eps=eps)
    g_tgt = spatial_gradient_magnitude(target, eps=eps)

    weight = 1.0 + lcs_weight
    if ridge_mask is not None:
        weight = weight + ridge_boost * ridge_mask

    return (weight * (g_pred - g_tgt).abs()).mean()
