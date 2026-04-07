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


class TrendAccumulator:
    def __init__(self, num_channels=8, num_timesteps=6):
        shape = (num_channels, num_timesteps)
        self.count = np.zeros(shape, dtype=np.float64)
        self.sum_x = np.zeros(shape, dtype=np.float64)
        self.sum_y = np.zeros(shape, dtype=np.float64)
        self.sum_x2 = np.zeros(shape, dtype=np.float64)
        self.sum_y2 = np.zeros(shape, dtype=np.float64)
        self.sum_xy = np.zeros(shape, dtype=np.float64)
        self.sum_abs = np.zeros(shape, dtype=np.float64)
        self.sum_sq = np.zeros(shape, dtype=np.float64)
        self.sign_match = np.zeros(shape, dtype=np.float64)

    def update(self, pred_trend, gt_trend):
        # pred_trend / gt_trend: [B, T, C, H, W], already denormalized
        pred_np = pred_trend.detach().cpu().numpy().astype(np.float64)
        gt_np = gt_trend.detach().cpu().numpy().astype(np.float64)

        for c in range(pred_np.shape[2]):
            for t in range(pred_np.shape[1]):
                x = pred_np[:, t, c].reshape(-1)
                y = gt_np[:, t, c].reshape(-1)
                diff = x - y

                self.count[c, t] += x.size
                self.sum_x[c, t] += x.sum()
                self.sum_y[c, t] += y.sum()
                self.sum_x2[c, t] += np.square(x).sum()
                self.sum_y2[c, t] += np.square(y).sum()
                self.sum_xy[c, t] += (x * y).sum()
                self.sum_abs[c, t] += np.abs(diff).sum()
                self.sum_sq[c, t] += np.square(diff).sum()
                self.sign_match[c, t] += (np.sign(x) == np.sign(y)).sum()

    def finalize(self):
        eps = 1e-12
        n = np.maximum(self.count, 1.0)

        cov_num = n * self.sum_xy - self.sum_x * self.sum_y
        cov_den = np.sqrt(
            np.maximum(n * self.sum_x2 - np.square(self.sum_x), eps)
            * np.maximum(n * self.sum_y2 - np.square(self.sum_y), eps)
        )
        pearson_r = cov_num / np.maximum(cov_den, eps)

        sst = self.sum_y2 - np.square(self.sum_y) / n
        r2 = 1.0 - self.sum_sq / np.maximum(sst, eps)

        cosine = self.sum_xy / np.maximum(np.sqrt(self.sum_x2 * self.sum_y2), eps)
        trend_mae = self.sum_abs / n
        trend_rmse = np.sqrt(self.sum_sq / n)
        sign_accuracy = self.sign_match / n

        return {
            "pearson_r": pearson_r,
            "r2": r2,
            "cosine_similarity": cosine,
            "trend_mae": trend_mae,
            "trend_rmse": trend_rmse,
            "sign_accuracy": sign_accuracy,
        }


def save_matrix_csv(path, array):
    np.savetxt(path, array, delimiter=",", fmt="%.6f")


def save_summary_csv(path, summary):
    fieldnames = [
        "metric",
        "overall_mean",
        "min",
        "max",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name, values in summary.items():
            writer.writerow(
                {
                    "metric": metric_name,
                    "overall_mean": values["overall_mean"],
                    "min": values["min"],
                    "max": values["max"],
                }
            )


def build_parser():
    parser = argparse.ArgumentParser(description="CloudFlow trend analysis for ICML rebuttal")
    parser.add_argument("--config", type=str, default="configs/CloudFlow_JiU.yaml")
    parser.add_argument("--ckpt_path", type=str, default="logs/CloudFlow_JiU_add_alpha_0.125/checkpoints/step_100000-loss_0.0915.ckpt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflow_trend_metrics"),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)
    effective_batch_size = args.batch_size or config["training"]["batch_size"]

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
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

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
        sample_steps=args.sample_steps,
        loss_alpha=0.125,
    )

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    module = module.to(device)
    module.eval()
    module.model.eval()
    module.rfdpic_model.eval()

    stage1_accumulator = TrendAccumulator(num_channels=8, num_timesteps=6)
    final_accumulator = TrendAccumulator(num_channels=8, num_timesteps=6)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            c_past, x_future = batch
            c_past = c_past.to(device)
            x_future = x_future.to(device)

            c_past_norm = data_transform(c_past, rescaled=module.rfdpic_rescaled)
            c_rfdpic_norm, _, _, _, _ = module.rfdpic_model(c_past_norm)
            preds_residual_norm = module.meanflow.sample_prediction(
                module.model,
                (c_rfdpic_norm, c_past_norm),
                sample_steps=args.sample_steps,
                device=device,
            )

            stage1_preds = inverse_data_transform(c_rfdpic_norm, rescaled=module.rfdpic_rescaled)
            final_preds_norm = c_rfdpic_norm + preds_residual_norm
            final_preds = inverse_data_transform(final_preds_norm, rescaled=module.rfdpic_rescaled)

            # All trend metrics are computed in the denormalized physical space.
            c_last_denorm = module.denormalize(c_past[:, -1:], mode="test")
            stage1_preds_denorm = module.denormalize(stage1_preds, mode="test")
            final_preds_denorm = module.denormalize(final_preds, mode="test")
            x_future_denorm = module.denormalize(x_future, mode="test")

            stage1_trend = stage1_preds_denorm - c_last_denorm
            pred_trend = final_preds_denorm - c_last_denorm
            gt_trend = x_future_denorm - c_last_denorm
            stage1_accumulator.update(stage1_trend, gt_trend)
            final_accumulator.update(pred_trend, gt_trend)

            if batch_idx % 20 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    stage1_metrics = stage1_accumulator.finalize()
    final_metrics = final_accumulator.finalize()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for metric_name, values in stage1_metrics.items():
        save_matrix_csv(os.path.join(args.out_dir, f"stage1_{metric_name}.csv"), values)
    for metric_name, values in final_metrics.items():
        save_matrix_csv(os.path.join(args.out_dir, f"final_{metric_name}.csv"), values)

    summary["stage1_vs_gt"] = {
        metric_name: {
            "overall_mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for metric_name, values in stage1_metrics.items()
    }
    summary["final_vs_gt"] = {
        metric_name: {
            "overall_mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for metric_name, values in final_metrics.items()
    }

    summary["trend_definition"] = {
        "description": "Residuals between each future frame and the last input frame, i.e., trend_t = x_future_t - x_input_last. We report the same trend metrics for both the stage-1 RFDPIC prediction and the final stage-2 CloudFlow prediction against GT.",
        "sequence_length": 6,
        "trend_steps": 6,
        "computed_on_denormalized_physical_values": True,
        "reference_frame": "last input frame",
        "stage1_prediction": "RFDPIC coarse prediction",
        "final_prediction": "CloudFlow final prediction",
    }
    summary["recommended_metrics"] = {
        "pearson_r": "Linear agreement between predicted and GT trend fields",
        "r2": "Explained variance of the GT trend by the predicted trend",
        "cosine_similarity": "Directional agreement of the trend vector in high-dimensional space",
        "sign_accuracy": "Agreement on increase/decrease direction per pixel",
        "trend_mae": "Absolute error of trend magnitude",
        "trend_rmse": "RMSE of trend magnitude",
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_summary_csv(
        os.path.join(args.out_dir, "summary.csv"),
        {
            f"stage1_{k}": v for k, v in summary["stage1_vs_gt"].items()
        } | {
            f"final_{k}": v for k, v in summary["final_vs_gt"].items()
        },
    )

    print(f"Trend metrics saved to {args.out_dir}")
    print("Saved matrices: stage1_*.csv and final_*.csv for pearson_r, r2, cosine_similarity, sign_accuracy, trend_mae, trend_rmse")


if __name__ == "__main__":
    main()
