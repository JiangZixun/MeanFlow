import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import yaml
from einops import rearrange
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_btchw import Himawari8LightningDataModule
from evaluation.fvd.torchmetrics_wrap import FrechetVideoDistance
from evaluation.psd.psd_tools import calculate_psd_error_metrics, plot_sample_psd
from train_JiT_RFDPIC import VideoLightningModule as BaseVideoLightningModule
from utils.transform import data_transform, inverse_data_transform
from visualize import plot_metrics_curve, vis_himawari8_seq_btchw


class RebuttalStage1PerturbationModule(BaseVideoLightningModule):
    def __init__(
        self,
        model_config,
        data_config,
        meanflow_config,
        optimizer_config,
        scheduler_config,
        training_config,
        logging_config,
        eval_config,
        rfdpic_config_path,
        rfdpic_ckpt_path,
        sample_steps,
        args,
    ):
        super().__init__(
            model_config=model_config,
            data_config=data_config,
            meanflow_config=meanflow_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            training_config=training_config,
            logging_config=logging_config,
            eval_config=eval_config,
            rfdpic_config_path=rfdpic_config_path,
            rfdpic_ckpt_path=rfdpic_ckpt_path,
            sample_steps=sample_steps,
        )
        self.args = args
        self.perturbations = [p.strip().lower() for p in args.perturbations.split(",") if p.strip()]
        if not self.perturbations:
            self.perturbations = ["none"]

        self.coarse_d_mse = torchmetrics.MeanSquaredError()
        self.coarse_d_mae = torchmetrics.MeanAbsoluteError()
        self.coarse_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.coarse_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)

        self.coarse_d_mse_metric = nn.ModuleList([
            nn.ModuleList([torchmetrics.MeanSquaredError() for _ in range(self.num_timesteps)])
            for _ in range(self.num_channels)
        ])
        self.coarse_d_mae_metric = nn.ModuleList([
            nn.ModuleList([torchmetrics.MeanAbsoluteError() for _ in range(self.num_timesteps)])
            for _ in range(self.num_channels)
        ])
        self.coarse_d_rmse_metric = nn.ModuleList([
            nn.ModuleList([torchmetrics.MeanSquaredError(squared=False) for _ in range(self.num_timesteps)])
            for _ in range(self.num_channels)
        ])
        self.coarse_fvd_list = nn.ModuleList([
            FrechetVideoDistance(feature=400, normalize=False) for _ in range(self.num_channels)
        ])

        self.register_buffer("running_mean_coarse", torch.zeros(self.num_channels, self.num_timesteps))

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        if self.trainer.is_global_zero:
            with open(os.path.join(self.metrics_save_dir, "perturbation_config.json"), "w") as f:
                json.dump(vars(self.args), f, indent=2, ensure_ascii=False)
            if getattr(self.args, "save_examples", False):
                self.example_save_dir = os.path.join(self.trainer.logger.save_dir, "examples_test")
                os.makedirs(self.example_save_dir, exist_ok=True)

    def _flatten_video(self, data):
        b, t, c, h, w = data.shape
        return data.reshape(b * t, c, h, w), (b, t, c, h, w)

    def _restore_video(self, data, shape):
        b, t, c, h, w = shape
        return data.reshape(b, t, c, h, w)

    def _gaussian_kernel(self, kernel_size, sigma, device, dtype):
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * max(sigma, 1e-6) ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d

    def _apply_blur(self, coarse):
        if self.args.blur_kernel <= 1 or self.args.blur_sigma <= 0:
            return coarse

        frames, shape = self._flatten_video(coarse)
        kernel_2d = self._gaussian_kernel(
            self.args.blur_kernel,
            self.args.blur_sigma,
            frames.device,
            frames.dtype,
        )
        kernel = kernel_2d.view(1, 1, self.args.blur_kernel, self.args.blur_kernel)
        kernel = kernel.repeat(frames.shape[1], 1, 1, 1)
        padding = self.args.blur_kernel // 2
        blurred = F.conv2d(frames, kernel, padding=padding, groups=frames.shape[1])
        return self._restore_video(blurred, shape)

    def _apply_noise(self, coarse):
        if self.args.noise_std <= 0:
            return coarse

        noise = torch.randn_like(coarse) * self.args.noise_std
        if self.args.noise_mode == "additive":
            return coarse + noise
        if self.args.noise_mode == "multiplicative":
            return coarse * (1 + noise)
        raise ValueError(f"Unsupported noise mode: {self.args.noise_mode}")

    def _build_affine_theta(self, batch_size, device, dtype):
        theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)

        angles = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * self.args.affine_rotate
        angles = angles * math.pi / 180.0
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        scale_min = self.args.affine_scale_min
        scale_max = self.args.affine_scale_max
        scales = torch.ones(batch_size, device=device, dtype=dtype)
        if abs(scale_max - scale_min) > 1e-8:
            scales = torch.empty(batch_size, device=device, dtype=dtype).uniform_(scale_min, scale_max)
        else:
            scales = scales * scale_min

        shear_deg = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * self.args.affine_shear
        shear = torch.tan(shear_deg * math.pi / 180.0)

        tx = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * self.args.affine_translate
        ty = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * self.args.affine_translate

        theta[:, 0, 0] = scales * cos_a
        theta[:, 0, 1] = -scales * sin_a + shear
        theta[:, 1, 0] = scales * sin_a
        theta[:, 1, 1] = scales * cos_a
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        return theta

    def _apply_affine(self, coarse):
        has_affine = any(
            [
                self.args.affine_rotate > 0,
                self.args.affine_translate > 0,
                abs(self.args.affine_scale_min - 1.0) > 1e-8,
                abs(self.args.affine_scale_max - 1.0) > 1e-8,
                self.args.affine_shear > 0,
            ]
        )
        if not has_affine:
            return coarse

        frames, shape = self._flatten_video(coarse)
        theta = self._build_affine_theta(frames.shape[0], frames.device, frames.dtype)
        grid = F.affine_grid(theta, size=frames.shape, align_corners=False)
        deformed = F.grid_sample(
            frames,
            grid,
            mode=self.args.deform_interp,
            padding_mode=self.args.deform_padding,
            align_corners=False,
        )
        return self._restore_video(deformed, shape)

    def _apply_bias(self, coarse):
        if abs(self.args.bias_value) <= 0:
            return coarse
        return coarse + self.args.bias_value

    def _apply_scale(self, coarse):
        if abs(self.args.scale_factor - 1.0) <= 1e-8:
            return coarse
        return coarse * self.args.scale_factor

    def _apply_dropout(self, coarse):
        if self.args.dropout_prob <= 0:
            return coarse
        mask = torch.rand_like(coarse) > self.args.dropout_prob
        return coarse * mask.to(coarse.dtype)

    def apply_stage1_perturbations(self, coarse):
        perturbed = coarse.clone()
        for perturbation in self.perturbations:
            if perturbation in {"none", "clean"}:
                continue
            if perturbation == "blur":
                perturbed = self._apply_blur(perturbed)
            elif perturbation == "noise":
                perturbed = self._apply_noise(perturbed)
            elif perturbation in {"deform", "affine"}:
                perturbed = self._apply_affine(perturbed)
            elif perturbation == "bias":
                perturbed = self._apply_bias(perturbed)
            elif perturbation == "scale":
                perturbed = self._apply_scale(perturbed)
            elif perturbation == "dropout":
                perturbed = self._apply_dropout(perturbed)
            else:
                raise ValueError(f"Unsupported perturbation: {perturbation}")
        return perturbed.clamp(0.0, 1.0)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        c_past, x_future = batch

        c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
        coarse_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
        coarse = inverse_data_transform(coarse_norm, rescaled=self.rfdpic_rescaled).detach()
        coarse = self.apply_stage1_perturbations(coarse)

        preds_norm = self.meanflow.sample_prediction(
            self.model,
            (coarse, c_past),
            sample_steps=self.sample_steps,
            device=self.device,
        )
        targets_norm = x_future

        preds_denorm = self.denormalize(preds_norm, mode="test")
        coarse_denorm = self.denormalize(coarse, mode="test")
        targets_denorm = self.denormalize(targets_norm, mode="test")

        self.test_d_mse(preds_denorm, targets_denorm)
        self.test_d_mae(preds_denorm, targets_denorm)
        self.coarse_d_mse(coarse_denorm, targets_denorm)
        self.coarse_d_mae(coarse_denorm, targets_denorm)

        preds_bchw = rearrange(preds_norm, "b t c h w -> (b t) c h w")
        coarse_bchw = rearrange(coarse, "b t c h w -> (b t) c h w")
        targets_bchw = rearrange(targets_norm, "b t c h w -> (b t) c h w")
        self.test_ssim(preds_bchw, targets_bchw)
        self.test_psnr(preds_bchw, targets_bchw)
        self.coarse_ssim(coarse_bchw, targets_bchw)
        self.coarse_psnr(coarse_bchw, targets_bchw)

        real_video_t12 = torch.cat([c_past, targets_norm], dim=1)
        pred_video_t12 = torch.cat([c_past, preds_norm], dim=1)
        coarse_video_t12 = torch.cat([c_past, coarse], dim=1)
        for c in range(self.num_channels):
            real_ch_video = real_video_t12[:, :, c:c + 1, :, :]
            pred_ch_video = pred_video_t12[:, :, c:c + 1, :, :]
            coarse_ch_video = coarse_video_t12[:, :, c:c + 1, :, :]
            self.test_fvd_list[c].update(real_ch_video, real=True)
            self.test_fvd_list[c].update(pred_ch_video, real=False)
            self.coarse_fvd_list[c].update(real_ch_video, real=True)
            self.coarse_fvd_list[c].update(coarse_ch_video, real=False)

        self.test_psd_gt.update(targets_norm)
        self.test_psd_pred.update(preds_norm)
        self.test_psd_rfdpic.update(coarse)

        for c in range(self.num_channels):
            for t in range(self.num_timesteps):
                pred_t = preds_denorm[:, t, c].contiguous()
                coarse_t = coarse_denorm[:, t, c].contiguous()
                target_t = targets_denorm[:, t, c].contiguous()
                self.test_d_mae_metric[c][t](pred_t, target_t)
                self.test_d_mse_metric[c][t](pred_t, target_t)
                self.test_d_rmse_metric[c][t](pred_t, target_t)
                self.coarse_d_mae_metric[c][t](coarse_t, target_t)
                self.coarse_d_mse_metric[c][t](coarse_t, target_t)
                self.coarse_d_rmse_metric[c][t](coarse_t, target_t)

        batch_mean_pred = preds_denorm.mean(dim=(0, 3, 4))
        batch_mean_coarse = coarse_denorm.mean(dim=(0, 3, 4))
        batch_mean_target = targets_denorm.mean(dim=(0, 3, 4))
        self.running_mean_pred += batch_mean_pred.T
        self.running_mean_coarse += batch_mean_coarse.T
        self.running_mean_target += batch_mean_target.T
        self.test_step_count += 1.0

        micro_batch_size = c_past.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if getattr(self.args, "save_examples", False) and data_idx in self.test_example_data_idx_list:
            c_past_sample = c_past[0].cpu()
            preds_sample = preds_norm[0].cpu()
            coarse_sample = coarse[0].cpu()
            targets_sample = targets_norm[0].cpu()

            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [preds_sample[t] for t in range(preds_sample.shape[0])]
            target_list = [targets_sample[t] for t in range(targets_sample.shape[0])]
            coarse_list = [coarse_sample[t] for t in range(coarse_sample.shape[0])]

            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}")
            vis_himawari8_seq_btchw(
                save_dir=save_dir,
                context_seq=context_list,
                pred_seq=pred_list,
                target_seq=target_list,
            )
            vis_himawari8_seq_btchw(
                save_dir=os.path.join(save_dir, "perturbed_stage1"),
                context_seq=context_list,
                pred_seq=coarse_list,
                target_seq=target_list,
            )

            try:
                plot_sample_psd(
                    pred_4d=preds_sample.numpy(),
                    rfdpic_4d=coarse_sample.numpy(),
                    gt_4d=targets_sample.numpy(),
                    save_dir=save_dir,
                )
            except Exception as e:
                print(f"Failed to generate sample PSD plot for sample {data_idx}: {e}")

    def _compute_mean_fvd(self, metric_list):
        values = []
        for idx, metric in enumerate(metric_list):
            try:
                values.append(metric.compute())
            except Exception as e:
                print(f"Could not compute FVD for channel {idx}: {e}")
            metric.reset()
        if not values:
            return None
        return torch.stack(values).mean()

    def _collect_metric_matrix(self, metric_grid):
        values = np.zeros((self.num_channels, self.num_timesteps))
        for c in range(self.num_channels):
            for t in range(self.num_timesteps):
                values[c, t] = metric_grid[c][t].compute().detach().cpu().item()
                metric_grid[c][t].reset()
        return values

    def _to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
        return obj

    def _append_summary_csv(self, summary):
        summary_csv = getattr(self.args, "summary_csv", None)
        if not summary_csv:
            return

        row = {
            "case_name": getattr(self.args, "case_name", ""),
            "severity": getattr(self.args, "severity", ""),
            "perturbations": ",".join(summary["perturbations"]),
            "blur_kernel": self.args.blur_kernel,
            "blur_sigma": self.args.blur_sigma,
            "noise_mode": self.args.noise_mode,
            "noise_std": self.args.noise_std,
            "affine_rotate": self.args.affine_rotate,
            "affine_translate": self.args.affine_translate,
            "affine_scale_min": self.args.affine_scale_min,
            "affine_scale_max": self.args.affine_scale_max,
            "affine_shear": self.args.affine_shear,
            "bias_value": self.args.bias_value,
            "scale_factor": self.args.scale_factor,
            "dropout_prob": self.args.dropout_prob,
            "coarse_mse": summary["coarse"].get("d_mse"),
            "coarse_mae": summary["coarse"].get("d_mae"),
            "coarse_psnr": summary["coarse"].get("psnr"),
            "coarse_ssim": summary["coarse"].get("ssim"),
            "coarse_fvd": summary["coarse"].get("fvd"),
            "meanflow_mse": summary["meanflow"].get("d_mse"),
            "meanflow_mae": summary["meanflow"].get("d_mae"),
            "meanflow_psnr": summary["meanflow"].get("psnr"),
            "meanflow_ssim": summary["meanflow"].get("ssim"),
            "meanflow_fvd": summary["meanflow"].get("fvd"),
            "mse_reduction": summary["gain"].get("d_mse_reduction"),
            "mae_reduction": summary["gain"].get("d_mae_reduction"),
            "psnr_improvement": summary["gain"].get("psnr_improvement"),
            "ssim_improvement": summary["gain"].get("ssim_improvement"),
            "fvd_reduction": summary["gain"].get("fvd_reduction"),
        }
        row = self._to_jsonable(row)
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        file_exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def on_test_epoch_end(self):
        refined_summary = {
            "d_mse": float(self.test_d_mse.compute().detach().cpu().item()),
            "d_mae": float(self.test_d_mae.compute().detach().cpu().item()),
            "ssim": float(self.test_ssim.compute().detach().cpu().item()),
            "psnr": float(self.test_psnr.compute().detach().cpu().item()),
        }
        coarse_summary = {
            "d_mse": float(self.coarse_d_mse.compute().detach().cpu().item()),
            "d_mae": float(self.coarse_d_mae.compute().detach().cpu().item()),
            "ssim": float(self.coarse_ssim.compute().detach().cpu().item()),
            "psnr": float(self.coarse_psnr.compute().detach().cpu().item()),
        }

        self.log("test/d_mse_epoch", refined_summary["d_mse"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/d_mae_epoch", refined_summary["d_mae"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/ssim_epoch", refined_summary["ssim"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/psnr_epoch", refined_summary["psnr"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/coarse_d_mse_epoch", coarse_summary["d_mse"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/coarse_d_mae_epoch", coarse_summary["d_mae"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/coarse_ssim_epoch", coarse_summary["ssim"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/coarse_psnr_epoch", coarse_summary["psnr"], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        self.test_d_mse.reset()
        self.test_d_mae.reset()
        self.test_ssim.reset()
        self.test_psnr.reset()
        self.coarse_d_mse.reset()
        self.coarse_d_mae.reset()
        self.coarse_ssim.reset()
        self.coarse_psnr.reset()

        refined_fvd = self._compute_mean_fvd(self.test_fvd_list)
        coarse_fvd = self._compute_mean_fvd(self.coarse_fvd_list)
        if refined_fvd is not None:
            self.log("test/fvd_epoch_mean", refined_fvd, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            refined_summary["fvd"] = float(refined_fvd.detach().cpu().item())
        if coarse_fvd is not None:
            self.log("test/coarse_fvd_epoch_mean", coarse_fvd, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            coarse_summary["fvd"] = float(coarse_fvd.detach().cpu().item())

        k_axis_gt, psd_gt = self.test_psd_gt.compute()
        _, psd_pred = self.test_psd_pred.compute()
        _, psd_rfdpic = self.test_psd_rfdpic.compute()

        k_axis_np = k_axis_gt.cpu().numpy()
        psd_gt_np = psd_gt.cpu().numpy()
        psd_pred_np = psd_pred.cpu().numpy()
        psd_rfdpic_np = psd_rfdpic.cpu().numpy()

        self.test_psd_gt.reset()
        self.test_psd_pred.reset()
        self.test_psd_rfdpic.reset()

        if self.trainer.is_global_zero:
            splits_list = [0.50, 0.75, 0.90, 0.95]
            try:
                psd_metrics_pred = calculate_psd_error_metrics(psd_gt_np, psd_pred_np, k_axis_np, splits_list)
                psd_metrics_coarse = calculate_psd_error_metrics(psd_gt_np, psd_rfdpic_np, k_axis_np, splits_list)
                for key, value in psd_metrics_pred.items():
                    self.log(f"test/psd_pred_{key}", value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
                for key, value in psd_metrics_coarse.items():
                    self.log(f"test/psd_rfdpic_{key}", value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
            except Exception as e:
                print(f"Failed to compute PSD metrics: {e}")
                psd_metrics_pred = {}
                psd_metrics_coarse = {}
        else:
            psd_metrics_pred = {}
            psd_metrics_coarse = {}

        refined_mse = self._collect_metric_matrix(self.test_d_mse_metric)
        refined_mae = self._collect_metric_matrix(self.test_d_mae_metric)
        refined_rmse = self._collect_metric_matrix(self.test_d_rmse_metric)
        coarse_mse = self._collect_metric_matrix(self.coarse_d_mse_metric)
        coarse_mae = self._collect_metric_matrix(self.coarse_d_mae_metric)
        coarse_rmse = self._collect_metric_matrix(self.coarse_d_rmse_metric)

        avg_mean_pred = self.all_gather(self.running_mean_pred).mean(dim=0)
        avg_mean_coarse = self.all_gather(self.running_mean_coarse).mean(dim=0)
        avg_mean_target = self.all_gather(self.running_mean_target).mean(dim=0)

        if self.trainer.is_global_zero:
            np.savetxt(os.path.join(self.metrics_save_dir, "meanflow_mse.csv"), refined_mse, delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "meanflow_mae.csv"), refined_mae, delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "meanflow_rmse.csv"), refined_rmse, delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "coarse_mse.csv"), coarse_mse, delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "coarse_mae.csv"), coarse_mae, delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "coarse_rmse.csv"), coarse_rmse, delimiter=",")

            np.savetxt(os.path.join(self.metrics_save_dir, "improvement_mse.csv"), coarse_mse - refined_mse, delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "improvement_mae.csv"), coarse_mae - refined_mae, delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "improvement_rmse.csv"), coarse_rmse - refined_rmse, delimiter=",")

            np.savetxt(os.path.join(self.metrics_save_dir, "mean_pred.csv"), avg_mean_pred.cpu().numpy(), delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "mean_coarse.csv"), avg_mean_coarse.cpu().numpy(), delimiter=",")
            np.savetxt(os.path.join(self.metrics_save_dir, "mean_target.csv"), avg_mean_target.cpu().numpy(), delimiter=",")

            summary = {
                "perturbations": self.perturbations,
                "coarse": coarse_summary,
                "meanflow": refined_summary,
                "gain": {
                    "d_mse_reduction": coarse_summary["d_mse"] - refined_summary["d_mse"],
                    "d_mae_reduction": coarse_summary["d_mae"] - refined_summary["d_mae"],
                    "ssim_improvement": refined_summary["ssim"] - coarse_summary["ssim"],
                    "psnr_improvement": refined_summary["psnr"] - coarse_summary["psnr"],
                    "fvd_reduction": coarse_summary.get("fvd", float("nan")) - refined_summary.get("fvd", float("nan")),
                },
                "psd": {
                    "coarse": psd_metrics_coarse,
                    "meanflow": psd_metrics_pred,
                },
            }
            with open(os.path.join(self.metrics_save_dir, "summary.json"), "w") as f:
                json.dump(self._to_jsonable(summary), f, indent=2, ensure_ascii=False)
            self._append_summary_csv(summary)
            print(f"Rebuttal metrics saved to {self.metrics_save_dir}")

        self.running_mean_pred.zero_()
        self.running_mean_coarse.zero_()
        self.running_mean_target.zero_()
        self.test_step_count.zero_()


def build_parser():
    parser = argparse.ArgumentParser(description="ICML rebuttal test: perturb stage-1 coarse predictions before MeanFlow refinement")
    parser.add_argument("--config", type=str, default="configs/JiT-B_RFDPIC.yaml")
    parser.add_argument("--log_dir", type=str, default="logs/2026ICMLRebuttal/stage1_perturbation")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--rfdpic_config", type=str, default="configs/rfdpic_config.yaml")
    parser.add_argument("--rfdpic_ckpt", type=str, default="pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt")
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--save_examples", action="store_true")
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--case_name", type=str, default="")
    parser.add_argument("--severity", type=str, default="")

    parser.add_argument("--perturbations", type=str, default="none", help="Comma-separated perturbations: none, blur, noise, deform, bias, scale, dropout")
    parser.add_argument("--blur_kernel", type=int, default=5)
    parser.add_argument("--blur_sigma", type=float, default=1.0)
    parser.add_argument("--noise_mode", type=str, default="additive", choices=["additive", "multiplicative"])
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--affine_rotate", type=float, default=0.0, help="Max random rotation in degrees")
    parser.add_argument("--affine_translate", type=float, default=0.0, help="Max normalized translation ratio in [-1, 1]")
    parser.add_argument("--affine_scale_min", type=float, default=1.0)
    parser.add_argument("--affine_scale_max", type=float, default=1.0)
    parser.add_argument("--affine_shear", type=float, default=0.0, help="Max shear angle in degrees")
    parser.add_argument("--deform_interp", type=str, default="bilinear", choices=["bilinear", "nearest"])
    parser.add_argument("--deform_padding", type=str, default="border", choices=["zeros", "border", "reflection"])
    parser.add_argument("--bias_value", type=float, default=0.0)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    effective_batch_size = args.batch_size or config["training"]["batch_size"]
    pl.seed_everything(42, workers=True)

    dataset_cfg = config["data"]
    datamodule = Himawari8LightningDataModule(
        dataset_name=dataset_cfg["dataset_name"],
        train_data_path=dataset_cfg["train_data_path"],
        train_json_path=dataset_cfg["train_json_path"],
        train_dataset_prefix=dataset_cfg["train_dataset_prefix"],
        train_ratio=dataset_cfg["train_ratio"],
        train_random_flip=dataset_cfg["train_random_flip"],
        test_data_path=dataset_cfg["test_data_path"],
        test_json_path=dataset_cfg["test_json_path"],
        test_dataset_prefix=dataset_cfg["test_dataset_prefix"],
        batch_size=effective_batch_size,
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
    )

    model = RebuttalStage1PerturbationModule(
        model_config=config["model"],
        data_config=config["data"],
        meanflow_config=config["meanflow"],
        optimizer_config=config["optimizer"],
        scheduler_config=config["scheduler"],
        training_config=config["training"],
        logging_config=config["logging"],
        eval_config=config["eval"],
        rfdpic_config_path=args.rfdpic_config,
        rfdpic_ckpt_path=args.rfdpic_ckpt,
        sample_steps=args.sample_steps,
        args=args,
    )

    if args.use_wandb:
        logger_instance = WandbLogger(
            project=config["logging"]["project_name"],
            save_dir=args.log_dir,
            name=os.path.basename(args.log_dir),
        )
    else:
        logger_instance = pl_loggers.TensorBoardLogger(save_dir=args.log_dir, name="")

    trainer = pl.Trainer(
        logger=logger_instance,
        devices=args.gpus,
        accelerator="gpu",
        precision=32,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
    )

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
