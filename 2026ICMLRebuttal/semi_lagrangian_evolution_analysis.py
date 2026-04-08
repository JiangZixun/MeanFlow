import argparse
import csv
import json
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_btchw import Himawari8LightningDataModule
from train_CloudFlow_JiU_add import VideoLightningModule
from utils.transform import data_transform, inverse_data_transform


class ErrorAccumulator:
    def __init__(self, num_channels=8, num_timesteps=6):
        shape = (num_channels, num_timesteps)
        self.count = np.zeros(shape, dtype=np.float64)
        self.sum_abs = np.zeros(shape, dtype=np.float64)
        self.sum_sq = np.zeros(shape, dtype=np.float64)

    def update(self, pred, gt):
        pred_np = pred.detach().cpu().numpy().astype(np.float64)
        gt_np = gt.detach().cpu().numpy().astype(np.float64)
        diff = pred_np - gt_np

        for c in range(pred_np.shape[2]):
            for t in range(pred_np.shape[1]):
                d = diff[:, t, c].reshape(-1)
                self.count[c, t] += d.size
                self.sum_abs[c, t] += np.abs(d).sum()
                self.sum_sq[c, t] += np.square(d).sum()

    def finalize(self):
        n = np.maximum(self.count, 1.0)
        return {
            "mae": self.sum_abs / n,
            "rmse": np.sqrt(self.sum_sq / n),
        }


class PairAccumulator:
    def __init__(self, num_channels=8, num_timesteps=6):
        shape = (num_channels, num_timesteps)
        self.count = np.zeros(shape, dtype=np.float64)
        self.sum_x = np.zeros(shape, dtype=np.float64)
        self.sum_y = np.zeros(shape, dtype=np.float64)
        self.sum_x2 = np.zeros(shape, dtype=np.float64)
        self.sum_y2 = np.zeros(shape, dtype=np.float64)
        self.sum_xy = np.zeros(shape, dtype=np.float64)

    def update(self, pred, gt):
        pred_np = pred.detach().cpu().numpy().astype(np.float64)
        gt_np = gt.detach().cpu().numpy().astype(np.float64)

        for c in range(pred_np.shape[2]):
            for t in range(pred_np.shape[1]):
                x = pred_np[:, t, c].reshape(-1)
                y = gt_np[:, t, c].reshape(-1)
                self.count[c, t] += x.size
                self.sum_x[c, t] += x.sum()
                self.sum_y[c, t] += y.sum()
                self.sum_x2[c, t] += np.square(x).sum()
                self.sum_y2[c, t] += np.square(y).sum()
                self.sum_xy[c, t] += (x * y).sum()

    def finalize(self):
        eps = 1e-12
        n = np.maximum(self.count, 1.0)
        cov_num = n * self.sum_xy - self.sum_x * self.sum_y
        cov_den = np.sqrt(
            np.maximum(n * self.sum_x2 - np.square(self.sum_x), eps)
            * np.maximum(n * self.sum_y2 - np.square(self.sum_y), eps)
        )
        pearson_r = cov_num / np.maximum(cov_den, eps)
        cosine = self.sum_xy / np.maximum(np.sqrt(self.sum_x2 * self.sum_y2), eps)
        return {
            "pearson_r": pearson_r,
            "cosine_similarity": cosine,
        }


def save_matrix_csv(path, array):
    np.savetxt(path, array, delimiter=",", fmt="%.6f")


def save_per_timestep_csv(path, metric_dict):
    metric_names = list(metric_dict.keys())
    num_timesteps = next(iter(metric_dict.values())).shape[1]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestep"] + metric_names)
        writer.writeheader()
        for t in range(num_timesteps):
            row = {"timestep": t + 1}
            for metric_name, values in metric_dict.items():
                row[metric_name] = float(values[:, t].mean())
            writer.writerow(row)


def summarize_metric_dict(metric_dict):
    summary = {}
    for metric_name, values in metric_dict.items():
        summary[metric_name] = {
            "overall_mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def save_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "metric", "overall_mean", "min", "max"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Semi-Lagrangian persistence baseline and evolution residual analysis for ICML rebuttal"
    )
    parser.add_argument("--config", type=str, default="configs/CloudFlow_JiU.yaml")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/CloudFlow_JiU_add_alpha_0.125/checkpoints/step_100000-loss_0.0915.ckpt",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument(
        "--baseline_source",
        type=str,
        default="predicted_displacement",
        choices=["predicted_displacement", "stage1_prediction"],
        help="How to construct the advective baseline.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "semi_lagrangian_evolution_metrics",
        ),
    )
    return parser


def build_datamodule(config, batch_size):
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
        batch_size=batch_size,
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
    )
    datamodule.setup(stage="test")
    return datamodule


def build_module(config, ckpt_path, sample_steps, device):
    module = VideoLightningModule(
        model_config=config["model"],
        data_config=config["data"],
        meanflow_config=config["meanflow"],
        optimizer_config=config["optimizer"],
        scheduler_config=config["scheduler"],
        training_config=config["training"],
        logging_config=config["logging"],
        eval_config=config["eval"],
        rfdpic_config=config["rfdpic_model"],
        sample_steps=sample_steps,
        loss_alpha=0.125,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=False)
    module = module.to(device)
    module.eval()
    module.model.eval()
    module.rfdpic_model.eval()
    return module


def build_advective_baseline(module, c_past_norm, c_rfdpic_norm, displacement_fields, baseline_source):
    if baseline_source == "stage1_prediction":
        return c_rfdpic_norm.detach(), "stage1_prediction"

    last_frame_norm = c_past_norm[:, -1]
    try:
        adv_norm = module.rfdpic_model._warp(
            last_frame_norm,
            displacement_fields,
            residual_fields=None,
        )
        if adv_norm.shape != c_rfdpic_norm.shape:
            raise ValueError(
                f"Unexpected advective baseline shape {tuple(adv_norm.shape)} vs {tuple(c_rfdpic_norm.shape)}"
            )
        return adv_norm.detach(), "predicted_displacement"
    except Exception as exc:
        print(f"[warn] predicted_displacement baseline failed, fallback to stage1_prediction: {exc}")
        return c_rfdpic_norm.detach(), f"stage1_fallback:{type(exc).__name__}"


def main():
    parser = build_parser()
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)
    batch_size = args.batch_size or config["training"]["batch_size"]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")

    datamodule = build_datamodule(config, batch_size)
    test_loader = datamodule.test_dataloader()
    module = build_module(config, args.ckpt_path, args.sample_steps, device)

    baseline_error_acc = ErrorAccumulator(num_channels=8, num_timesteps=6)
    final_error_acc = ErrorAccumulator(num_channels=8, num_timesteps=6)
    residual_pair_acc = PairAccumulator(num_channels=8, num_timesteps=6)
    source_counter = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            c_past, x_future = batch
            c_past = c_past.to(device)
            x_future = x_future.to(device)

            c_past_norm = data_transform(c_past, rescaled=module.rfdpic_rescaled)
            c_rfdpic_norm, displacement_fields, _, _, _ = module.rfdpic_model(c_past_norm)

            preds_residual_norm = module.meanflow.sample_prediction(
                module.model,
                (c_rfdpic_norm, c_past_norm),
                sample_steps=args.sample_steps,
                device=device,
            )
            final_preds_norm = c_rfdpic_norm + preds_residual_norm

            adv_norm, source_name = build_advective_baseline(
                module=module,
                c_past_norm=c_past_norm,
                c_rfdpic_norm=c_rfdpic_norm,
                displacement_fields=displacement_fields,
                baseline_source=args.baseline_source,
            )
            source_counter[source_name] = source_counter.get(source_name, 0) + 1

            adv = inverse_data_transform(adv_norm, rescaled=module.rfdpic_rescaled)
            final_preds = inverse_data_transform(final_preds_norm, rescaled=module.rfdpic_rescaled)

            c_last = module.denormalize(c_past[:, -1:], mode="test")
            adv_denorm = module.denormalize(adv, mode="test")
            final_denorm = module.denormalize(final_preds, mode="test")
            gt_denorm = module.denormalize(x_future, mode="test")

            residual_gt = gt_denorm - adv_denorm
            residual_pred = final_denorm - adv_denorm

            baseline_error_acc.update(adv_denorm, gt_denorm)
            final_error_acc.update(final_denorm, gt_denorm)
            residual_pair_acc.update(residual_pred, residual_gt)

            if batch_idx % 20 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    os.makedirs(args.out_dir, exist_ok=True)

    baseline_metrics = baseline_error_acc.finalize()
    final_metrics = final_error_acc.finalize()
    residual_metrics = residual_pair_acc.finalize()
    gain_metrics = {
        "mae_improvement": baseline_metrics["mae"] - final_metrics["mae"],
        "rmse_improvement": baseline_metrics["rmse"] - final_metrics["rmse"],
        "relative_mae_improvement": (baseline_metrics["mae"] - final_metrics["mae"])
        / np.maximum(baseline_metrics["mae"], 1e-12),
        "relative_rmse_improvement": (baseline_metrics["rmse"] - final_metrics["rmse"])
        / np.maximum(baseline_metrics["rmse"], 1e-12),
    }

    metric_groups = {
        "advective_baseline": baseline_metrics,
        "final_prediction": final_metrics,
        "evolution_residual": residual_metrics,
        "transport_gain": gain_metrics,
    }

    summary_rows = []
    summary_json = {
        "config": vars(args),
        "baseline_usage": source_counter,
        "notes": {
            "advective_baseline": "Warp the last observed frame using the model-predicted displacement field, without residual/source term.",
            "evolution_residual_gt": "X_gt - X_adv",
            "evolution_residual_pred": "X_pred - X_adv",
        },
        "groups": {},
    }

    for group_name, metric_dict in metric_groups.items():
        per_group_dir = os.path.join(args.out_dir, group_name)
        os.makedirs(per_group_dir, exist_ok=True)

        for metric_name, values in metric_dict.items():
            save_matrix_csv(os.path.join(per_group_dir, f"{metric_name}.csv"), values)

        save_per_timestep_csv(os.path.join(per_group_dir, "per_timestep.csv"), metric_dict)

        group_summary = summarize_metric_dict(metric_dict)
        summary_json["groups"][group_name] = group_summary
        for metric_name, stats in group_summary.items():
            summary_rows.append(
                {
                    "group": group_name,
                    "metric": metric_name,
                    "overall_mean": stats["overall_mean"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )

    save_summary_csv(os.path.join(args.out_dir, "summary.csv"), summary_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved semi-Lagrangian evolution analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()
