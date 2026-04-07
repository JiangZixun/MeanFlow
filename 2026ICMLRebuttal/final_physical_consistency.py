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
from train_JiT_RFDPIC import VideoLightningModule
from utils.transform import data_transform, inverse_data_transform


class PairwiseAccumulator:
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
        self.pred_norm_sq = np.zeros(shape, dtype=np.float64)
        self.gt_norm_sq = np.zeros(shape, dtype=np.float64)

    def update(self, pred_tensor, gt_tensor):
        pred_np = pred_tensor.detach().cpu().numpy().astype(np.float64)
        gt_np = gt_tensor.detach().cpu().numpy().astype(np.float64)

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
                self.pred_norm_sq[c, t] += np.square(x).sum()
                self.gt_norm_sq[c, t] += np.square(y).sum()

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
        cosine_similarity = self.sum_xy / np.maximum(np.sqrt(self.sum_x2 * self.sum_y2), eps)
        mae = self.sum_abs / n
        rmse = np.sqrt(self.sum_sq / n)
        sign_accuracy = self.sign_match / n
        relative_l2_error = np.sqrt(self.sum_sq) / np.maximum(np.sqrt(self.gt_norm_sq), eps)
        norm_ratio = np.sqrt(self.pred_norm_sq) / np.maximum(np.sqrt(self.gt_norm_sq), eps)

        return {
            "pearson_r": pearson_r,
            "r2": r2,
            "cosine_similarity": cosine_similarity,
            "mae": mae,
            "rmse": rmse,
            "sign_accuracy": sign_accuracy,
            "relative_l2_error": relative_l2_error,
            "norm_ratio": norm_ratio,
        }


def save_matrix_csv(path, array):
    np.savetxt(path, array, delimiter=",", fmt="%.6f")


def summarize(metric_block):
    out = {}
    for metric_name, values in metric_block.items():
        out[metric_name] = {
            "overall_mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def build_parser():
    parser = argparse.ArgumentParser(description="Final prediction physical consistency against GT")
    parser.add_argument("--config", type=str, default="configs/JiT-B_RFDPIC.yaml")
    parser.add_argument("--rfdpic_config", type=str, default="configs/rfdpic_config.yaml")
    parser.add_argument("--rfdpic_ckpt", type=str, default="pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt")
    parser.add_argument("--ckpt_path", type=str, default="logs/JiT-B/checkpoints/step_1000000-loss_0.1542.ckpt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_physical_consistency"),
    )
    return parser


def main():
    args = build_parser().parse_args()

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

    model = VideoLightningModule(
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
    )
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    model = model.to(device)
    model.eval()
    model.model.eval()
    model.rfdpic_model.eval()

    accumulator = PairwiseAccumulator(num_channels=8, num_timesteps=6)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            c_past, x_future = batch
            c_past = c_past.to(device)
            x_future = x_future.to(device)

            c_past_norm = data_transform(c_past, rescaled=model.rfdpic_rescaled)
            coarse_norm, displacement_fields, _, _, _ = model.rfdpic_model(c_past_norm)
            coarse = inverse_data_transform(coarse_norm, rescaled=model.rfdpic_rescaled).detach()

            preds_norm = model.meanflow.sample_prediction(
                model.model,
                (coarse, c_past),
                sample_steps=args.sample_steps,
                device=device,
            )

            last_input_norm = c_past_norm[:, -1]
            transport_norm = model.rfdpic_model._warp(last_input_norm, displacement_fields, residual_fields=None)
            transport = inverse_data_transform(transport_norm, rescaled=model.rfdpic_rescaled)

            pred_phys = model.denormalize(preds_norm, mode="test")
            gt_phys = model.denormalize(x_future, mode="test")
            transport_phys = model.denormalize(transport, mode="test")

            pred_residual = pred_phys - transport_phys
            gt_residual = gt_phys - transport_phys
            accumulator.update(pred_residual, gt_residual)

            if batch_idx % 20 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    metrics = accumulator.finalize()
    os.makedirs(args.out_dir, exist_ok=True)
    for metric_name, values in metrics.items():
        save_matrix_csv(os.path.join(args.out_dir, f"{metric_name}.csv"), values)

    summary = {
        "definition": {
            "description": "Physical consistency is evaluated by first constructing a pure transport baseline from the last input frame using the stage-1 displacement field with no source term, and then comparing the final prediction residual (prediction minus transport baseline) against the GT residual (GT minus transport baseline), all in denormalized physical space.",
            "transport_baseline": "Warp(last_input_frame, displacement_field, residual_fields=None)",
            "physical_space": True,
        },
        "final_prediction_vs_gt_residual": summarize(metrics),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "overall_mean", "min", "max"])
        for metric_name, values in summary["final_prediction_vs_gt_residual"].items():
            writer.writerow([metric_name, values["overall_mean"], values["min"], values["max"]])

    print(f"Saved final physical consistency results to {args.out_dir}")


if __name__ == "__main__":
    main()
