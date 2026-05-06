import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_mmap import CloudSegMMapDataset
from model_unet_cond import UnetCond


@dataclass
class MeanFlowConfig:
    flow_ratio: float = 0.5
    time_mu: float = -0.4
    time_sigma: float = 1.0


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, ignore_index: int = 10):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        t = target.clamp(0, self.n_classes - 1)
        onehot = F.one_hot(t, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        valid4d = valid.unsqueeze(1)
        probs = probs * valid4d
        onehot = onehot * valid4d
        smooth = 1e-5
        inter = (probs * onehot).sum(dim=(0, 2, 3))
        den = (probs * probs).sum(dim=(0, 2, 3)) + (onehot * onehot).sum(dim=(0, 2, 3))
        dice = (2 * inter + smooth) / (den + smooth)
        return 1.0 - dice.mean()


class AttentionWeightedCELoss(nn.Module):
    def __init__(self, num_classes: int, lambda_uncertainty: float = 0.5, ignore_index: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_uncertainty = lambda_uncertainty
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)
        valid_mask = targets != self.ignore_index
        n_valid = valid_mask.sum().item()

        w_base = torch.zeros(self.num_classes, device=logits.device)
        entropy_mean = torch.zeros(self.num_classes, device=logits.device)
        for c in range(self.num_classes):
            cls_mask = (targets == c) & valid_mask
            n_c = cls_mask.sum().item()
            if n_c > 0 and n_valid > 0:
                w_base[c] = (n_valid - n_c) / n_valid
                entropy_mean[c] = entropy[cls_mask].mean()

        w_combined = w_base * (1 + self.lambda_uncertainty * entropy_mean)
        weight_map = w_combined[targets.clamp(0, self.num_classes - 1)] * valid_mask
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        return (ce * weight_map).sum() / (valid_mask.sum() + 1e-6)


class MeanFlowSeg:
    def __init__(self, cfg: MeanFlowConfig):
        self.cfg = cfg

    def sample_t_r(self, bsz: int, device: torch.device):
        normal = np.random.randn(bsz, 2).astype(np.float32) * self.cfg.time_sigma + self.cfg.time_mu
        samples = 1 / (1 + np.exp(-normal))
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        k = int(self.cfg.flow_ratio * bsz)
        if k > 0:
            idx = np.random.permutation(bsz)[:k]
            r_np[idx] = t_np[idx]
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(
        self,
        model,
        image_cond,
        mask_onehot,
        mask_label,
        criterion_att_ce,
        criterion_dice,
        lambda_seg=0.2,
        seg_supervision_steps=3,
    ):
        b = mask_onehot.shape[0]
        device = mask_onehot.device
        t, r = self.sample_t_r(b, device)

        t_ = t[:, None, None, None]
        r_ = r[:, None, None, None]

        x = mask_onehot * 2.0 - 1.0
        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        u = model(z, t, r, image_cond)
        with torch.no_grad():
            u_t = model(z, t, t, image_cond)
        u_tgt = v - (t_ - r_) * u_t
        loss_flow = ((u - u_tgt) ** 2).mean()

        # Endpoint supervision: roll from z_t to z_0 with a short deterministic chain.
        z_cur = z.detach().clone()
        k = max(1, int(seg_supervision_steps))
        for i in range(k):
            alpha0 = 1.0 - (i / k)
            alpha1 = 1.0 - ((i + 1) / k)
            t_cur = t * alpha0
            r_cur = t * alpha1
            t_cur_ = t_cur[:, None, None, None]
            r_cur_ = r_cur[:, None, None, None]
            v_cur = model(z_cur, t_cur, r_cur, image_cond)
            z_cur = z_cur - (t_cur_ - r_cur_) * v_cur

        x0_hat = z_cur
        loss_att_ce = criterion_att_ce(x0_hat, mask_label)
        loss_dice = criterion_dice(x0_hat, mask_label)
        loss_seg = 0.5 * loss_att_ce + 0.5 * loss_dice
        loss = loss_flow + lambda_seg * loss_seg
        return loss, loss_flow.detach(), loss_att_ce.detach(), loss_dice.detach(), loss_seg.detach()

    @torch.no_grad()
    def sample(self, model, image_cond, sample_steps=10):
        model.eval()
        b, _, h, w = image_cond.shape
        ncls = model.num_classes
        z = torch.randn((b, ncls, h, w), device=image_cond.device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=image_cond.device)
        for i in range(sample_steps):
            t = torch.full((b,), t_vals[i], device=image_cond.device)
            r = torch.full((b,), t_vals[i + 1], device=image_cond.device)
            t_ = t[:, None, None, None]
            r_ = r[:, None, None, None]
            v = model(z, t, r, image_cond)
            z = z - (t_ - r_) * v
        x = (z + 1.0) * 0.5
        return x.argmax(dim=1)


def fast_hist(pred, label, ncls=10, ignore_idx=10):
    mask = label != ignore_idx
    pred = pred[mask]
    label = label[mask]
    return torch.bincount(label * ncls + pred, minlength=ncls * ncls).reshape(ncls, ncls)


def miou_from_hist(hist):
    inter = torch.diag(hist)
    union = hist.sum(0) + hist.sum(1) - inter
    iou = inter / union.clamp(min=1)
    return iou.mean().item(), iou


def metrics_from_hist(hist: torch.Tensor):
    inter = torch.diag(hist).float()
    gt = hist.sum(dim=1).float()
    pred = hist.sum(dim=0).float()
    union = gt + pred - inter

    class_iou = inter / union.clamp(min=1.0)
    class_pixelacc = inter / gt.clamp(min=1.0)
    miou = class_iou.mean().item()
    pixel_acc = (inter.sum() / hist.sum().clamp(min=1.0)).item()
    f1_per_class = (2.0 * inter) / (gt + pred).clamp(min=1.0)
    f1 = f1_per_class.mean().item()
    return {
        "miou": float(miou),
        "f1": float(f1),
        "pixelacc": float(pixel_acc),
        "class_miou": [float(x) for x in class_iou.tolist()],
        "class_pixelacc": [float(x) for x in class_pixelacc.tolist()],
        "class_f1": [float(x) for x in f1_per_class.tolist()],
    }


def build_loaders(cfg):
    train_ds = CloudSegMMapDataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["train_split"],
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["data"].get("ignore_index", 10),
        input_channels=cfg["model"]["image_channels"],
    )
    val_ds = CloudSegMMapDataset(
        root=cfg["data"]["root"],
        split=cfg["data"]["val_split"],
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["data"].get("ignore_index", 10),
        input_channels=cfg["model"]["image_channels"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def evaluate(model, flow, val_loader, device, sample_steps=10, ncls=10):
    model.eval()
    hist = torch.zeros((ncls, ncls), dtype=torch.long, device=device)
    for img, lbl in tqdm(val_loader, desc="val", leave=False):
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        pred = flow.sample(model, img, sample_steps=sample_steps)
        hist += fast_hist(pred, lbl, ncls=ncls, ignore_idx=10)
    m = metrics_from_hist(hist)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample_steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sample_steps = args.sample_steps if args.sample_steps is not None else cfg["eval"].get("sample_steps", 10)
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["train"]["save_dir"], "ckpt")
    metrics_dir = os.path.join(cfg["train"]["save_dir"], "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncls = cfg["model"]["num_classes"]

    model = UnetCond(
        image_channels=cfg["model"]["image_channels"],
        num_classes=ncls,
        time_dim=cfg["model"]["time_dim"],
        encoder_channels=cfg["model"].get("encoder_channels", [64, 128, 256, 512]),
        bottleneck_channels=cfg["model"].get("bottleneck_channels", 1024),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion_att_ce = AttentionWeightedCELoss(
        num_classes=ncls,
        lambda_uncertainty=cfg["train"].get("lambda_uncertainty", 0.5),
        ignore_index=cfg["data"].get("ignore_index", 10),
    )
    criterion_dice = DiceLoss(n_classes=ncls, ignore_index=cfg["data"].get("ignore_index", 10))

    flow = MeanFlowSeg(
        MeanFlowConfig(
            flow_ratio=cfg["meanflow"]["flow_ratio"],
            time_mu=cfg["meanflow"]["time_mu"],
            time_sigma=cfg["meanflow"]["time_sigma"],
        )
    )

    train_loader, val_loader = build_loaders(cfg)
    epochs = cfg["train"]["epochs"]
    save_every = int(cfg["train"].get("save_every", 0))
    lambda_seg = float(cfg["train"].get("lambda_seg", 0.2))
    seg_supervision_steps = int(cfg["train"].get("seg_supervision_steps", 3))

    best_miou = -1.0
    best_payload = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        tbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}")
        for img, lbl in tbar:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            onehot = F.one_hot(lbl.clamp(max=ncls - 1), num_classes=ncls).permute(0, 3, 1, 2).float()
            ignore = (lbl == 10).unsqueeze(1)
            onehot = torch.where(ignore, torch.zeros_like(onehot), onehot)

            loss, lf, latt, ldice, lseg = flow.loss(
                model,
                img,
                onehot,
                lbl,
                criterion_att_ce=criterion_att_ce,
                criterion_dice=criterion_dice,
                lambda_seg=lambda_seg,
                seg_supervision_steps=seg_supervision_steps,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tbar.set_postfix(
                loss=f"{loss.item():.4f}",
                flow=f"{lf.item():.4f}",
                att_ce=f"{latt.item():.4f}",
                dice=f"{ldice.item():.4f}",
                seg=f"{lseg.item():.4f}",
            )

        train_loss = running_loss / max(1, len(train_loader))

        val_metrics = evaluate(model, flow, val_loader, device, sample_steps=sample_steps, ncls=ncls)
        val_acc = val_metrics["pixelacc"]
        val_miou = val_metrics["miou"]
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.6f} "
            f"val_pixelAcc={val_acc:.6f} val_mIoU={val_miou:.6f} val_F1={val_metrics['f1']:.6f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_miou": val_miou,
            "cfg": cfg,
        }

        epoch_payload = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "sample_steps": int(sample_steps),
            "metrics": val_metrics,
        }
        with open(os.path.join(metrics_dir, f"epoch_{epoch:04d}.json"), "w", encoding="utf-8") as f:
            json.dump(epoch_payload, f, ensure_ascii=False, indent=2)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(state, os.path.join(ckpt_dir, "best.pth"))
            best_payload = epoch_payload
            with open(os.path.join(metrics_dir, "best.json"), "w", encoding="utf-8") as f:
                json.dump(best_payload, f, ensure_ascii=False, indent=2)

        if save_every > 0 and epoch % save_every == 0:
            torch.save(state, os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"))

    if best_payload is None:
        with open(os.path.join(metrics_dir, "best.json"), "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
