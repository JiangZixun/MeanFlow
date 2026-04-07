import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_btchw import Himawari8LightningDataModule
from models.MotionPredictor.RFDPIC_Dual_Rotation_dyn import RFDPIC_Dual_Rotation_Dyn
from utils.transform import data_transform


def load_frozen_rfdpic(rfdpic_config_path, rfdpic_ckpt_path):
    with open(rfdpic_config_path, "r") as f:
        rfdpic_main_cfg = yaml.safe_load(f)
    rfdpic_rescaled = rfdpic_main_cfg.get("dataset", {}).get("rescaled", True)
    rfdpic_base_dir = os.path.dirname(rfdpic_config_path)
    model_cfg_path = rfdpic_main_cfg.get("model", {}).get("rf_dp_config_file")
    rfdpic_model_cfg_path = os.path.join(rfdpic_base_dir, model_cfg_path)
    if not os.path.exists(rfdpic_model_cfg_path):
        rfdpic_model_cfg_path = model_cfg_path

    with open(rfdpic_config_path, "r") as f:
        rfdpic_cfg = yaml.safe_load(f)["model"]

    model = RFDPIC_Dual_Rotation_Dyn(
        dp_name=rfdpic_cfg["dp_name"],
        rf_name=rfdpic_cfg["rf_name"],
        rf_dp_config_file=rfdpic_model_cfg_path,
        dp_mode=rfdpic_cfg["dp_mode"],
        rf_mode=rfdpic_cfg["rf_mode"],
        alpha=rfdpic_cfg["alpha"],
        interpolation_mode=rfdpic_cfg.get("interpolation_mode", "bilinear"),
        padding_mode=rfdpic_cfg.get("padding_mode", "border"),
        use_res_scale=rfdpic_cfg.get("use_res_scale", False),
        res_scale_init=rfdpic_cfg.get("res_scale_init", 0.05),
        advect_mode=rfdpic_cfg.get("advect_mode", "semi-lagrangian"),
        ode_method=rfdpic_cfg.get("ode_method", "rk4"),
        ode_substeps=rfdpic_cfg.get("ode_substeps", 1),
        use_splat=rfdpic_cfg.get("use_splat", False),
        splat_type=rfdpic_cfg.get("splat_type", "softmax"),
        splat_alpha=rfdpic_cfg.get("splat_alpha", 20.0),
        splat_blend=rfdpic_cfg.get("splat_blend", 0.5),
        use_gate=rfdpic_cfg.get("use_gate", False),
        use_implicit_diffusion=rfdpic_cfg.get("use_implicit_diffusion", False),
        jacobi_iters=rfdpic_cfg.get("jacobi_iters", 2),
        D_max=rfdpic_cfg.get("D_max", 0.2),
        k_edge=rfdpic_cfg.get("k_edge", 0.1),
        diff_dt=rfdpic_cfg.get("diff_dt", 1.0),
        diff_dx=rfdpic_cfg.get("diff_dx", 1.0),
        D_const=rfdpic_cfg.get("D_const", None),
    )

    state_dict = torch.load(rfdpic_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key.startswith("model."):
            model_state_dict[key.replace("model.", "", 1)] = val
        else:
            model_state_dict[key] = val
    model.load_state_dict(model_state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, rfdpic_rescaled


def build_test_loader(config, batch_size):
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
    return datamodule.test_dataloader()


class StreamingStats:
    def __init__(self, num_timesteps=6, num_channels=8, active_threshold=0.25):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.active_threshold = active_threshold
        self.active_count = 0
        self.active_sum = 0.0
        self.active_sum_sq = 0.0
        self.active_min = float("inf")
        self.active_max = float("-inf")

        self.step_count = np.zeros(num_timesteps, dtype=np.float64)
        self.step_sum = np.zeros(num_timesteps, dtype=np.float64)
        self.step_sum_sq = np.zeros(num_timesteps, dtype=np.float64)

        self.channel_count = np.zeros(num_channels, dtype=np.float64)
        self.channel_sum = np.zeros(num_channels, dtype=np.float64)
        self.channel_sum_sq = np.zeros(num_channels, dtype=np.float64)

    def update(self, speed):
        # speed: [B, T, C, H, W]
        speed = speed.detach().double()
        self.count += speed.numel()
        self.sum += speed.sum().item()
        self.sum_sq += torch.square(speed).sum().item()
        self.min = min(self.min, speed.min().item())
        self.max = max(self.max, speed.max().item())

        mask = speed >= self.active_threshold
        if mask.any():
            active_speed = speed[mask]
            self.active_count += active_speed.numel()
            self.active_sum += active_speed.sum().item()
            self.active_sum_sq += torch.square(active_speed).sum().item()
            self.active_min = min(self.active_min, active_speed.min().item())
            self.active_max = max(self.active_max, active_speed.max().item())

        step_vals = speed.sum(dim=(0, 2, 3, 4)).cpu().numpy()
        step_sq_vals = torch.square(speed).sum(dim=(0, 2, 3, 4)).cpu().numpy()
        step_n = np.full(speed.shape[1], speed.shape[0] * speed.shape[2] * speed.shape[3] * speed.shape[4], dtype=np.float64)
        self.step_count += step_n
        self.step_sum += step_vals
        self.step_sum_sq += step_sq_vals

        ch_vals = speed.sum(dim=(0, 1, 3, 4)).cpu().numpy()
        ch_sq_vals = torch.square(speed).sum(dim=(0, 1, 3, 4)).cpu().numpy()
        ch_n = np.full(speed.shape[2], speed.shape[0] * speed.shape[1] * speed.shape[3] * speed.shape[4], dtype=np.float64)
        self.channel_count += ch_n
        self.channel_sum += ch_vals
        self.channel_sum_sq += ch_sq_vals

    def finalize(self):
        mean = self.sum / max(self.count, 1)
        var = self.sum_sq / max(self.count, 1) - mean ** 2
        std = np.sqrt(max(var, 0.0))

        if self.active_count > 0:
            active_mean = self.active_sum / self.active_count
            active_var = self.active_sum_sq / self.active_count - active_mean ** 2
            active_std = np.sqrt(max(active_var, 0.0))
            active_min = float(self.active_min)
            active_max = float(self.active_max)
        else:
            active_mean = None
            active_var = None
            active_std = None
            active_min = None
            active_max = None

        step_mean = self.step_sum / np.maximum(self.step_count, 1.0)
        step_var = self.step_sum_sq / np.maximum(self.step_count, 1.0) - np.square(step_mean)
        step_std = np.sqrt(np.maximum(step_var, 0.0))

        channel_mean = self.channel_sum / np.maximum(self.channel_count, 1.0)
        channel_var = self.channel_sum_sq / np.maximum(self.channel_count, 1.0) - np.square(channel_mean)
        channel_std = np.sqrt(np.maximum(channel_var, 0.0))

        return {
            "count": int(self.count),
            "mean": float(mean),
            "var": float(var),
            "std": float(std),
            "min": float(self.min),
            "max": float(self.max),
            "active_count": int(self.active_count),
            "active_mean": None if active_mean is None else float(active_mean),
            "active_var": None if active_var is None else float(active_var),
            "active_std": None if active_std is None else float(active_std),
            "active_min": active_min,
            "active_max": active_max,
            "active_threshold_grid": float(self.active_threshold),
            "per_timestep_mean": step_mean.tolist(),
            "per_timestep_std": step_std.tolist(),
            "per_channel_mean": channel_mean.tolist(),
            "per_channel_std": channel_std.tolist(),
        }


def compute_speed(displacement_fields):
    # displacement_fields: [B, T, C, H, W, 2]
    return torch.linalg.vector_norm(displacement_fields, dim=-1)


def first_pass(loader, model, device, rfdpic_rescaled, active_threshold):
    stats = StreamingStats(active_threshold=active_threshold)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            c_past, _ = batch
            c_past = c_past.to(device)
            c_past_norm = data_transform(c_past, rescaled=rfdpic_rescaled)
            _, displacement_fields, _, _, _ = model(c_past_norm)
            speed = compute_speed(displacement_fields)
            stats.update(speed)
            if batch_idx % 20 == 0:
                print(f"First pass: processed batch {batch_idx + 1}/{len(loader)}")
    return stats.finalize()


def second_pass_histogram(loader, model, device, rfdpic_rescaled, min_value, max_value, num_bins):
    if max_value <= min_value:
        max_value = min_value + 1e-6

    hist = np.zeros(num_bins, dtype=np.float64)
    edges = np.linspace(min_value, max_value, num_bins + 1)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            c_past, _ = batch
            c_past = c_past.to(device)
            c_past_norm = data_transform(c_past, rescaled=rfdpic_rescaled)
            _, displacement_fields, _, _, _ = model(c_past_norm)
            speed = compute_speed(displacement_fields).detach().cpu().numpy().reshape(-1)
            batch_hist, _ = np.histogram(speed, bins=edges)
            hist += batch_hist
            if batch_idx % 20 == 0:
                print(f"Second pass: processed batch {batch_idx + 1}/{len(loader)}")

    return hist, edges


def save_vector_csv(path, values, header_name):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", header_name])
        for idx, value in enumerate(values):
            writer.writerow([idx, value])


def save_histogram_csv(path, hist, edges):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "bin_right", "count"])
        for i in range(len(hist)):
            writer.writerow([edges[i], edges[i + 1], hist[i]])


def plot_histogram(hist, edges, out_png, out_pdf):
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    plt.figure(figsize=(8.5, 5.5))
    plt.bar(centers, hist, width=widths, color="#1f77b4", edgecolor="white", linewidth=0.3, align="center")
    plt.xlabel("Velocity Magnitude")
    plt.ylabel("Count")
    plt.title("Distribution of Velocity Magnitude")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description="RFDPIC speed magnitude distribution analysis for rebuttal")
    parser.add_argument("--config", type=str, default="configs/JiT-B_RFDPIC.yaml")
    parser.add_argument("--rfdpic_config", type=str, default="configs/rfdpic_config.yaml")
    parser.add_argument("--rfdpic_ckpt", type=str, default="pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_bins", type=int, default=120)
    parser.add_argument(
        "--active_threshold",
        type=float,
        default=0.25,
        help="Threshold on velocity magnitude in grid units for active-motion-only statistics. Default 0.25 grid ~= 1 km / 30 min ~= 0.56 m/s for 4 km grids.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rfdpic_speed_distribution"),
    )
    return parser


def main():
    args = build_parser().parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)
    effective_batch_size = args.batch_size or config["training"]["batch_size"]
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")

    model, rfdpic_rescaled = load_frozen_rfdpic(args.rfdpic_config, args.rfdpic_ckpt)
    model = model.to(device)
    model.eval()

    loader = build_test_loader(config, effective_batch_size)
    stats = first_pass(loader, model, device, rfdpic_rescaled, args.active_threshold)

    loader = build_test_loader(config, effective_batch_size)
    hist, edges = second_pass_histogram(
        loader,
        model,
        device,
        rfdpic_rescaled,
        stats["min"],
        stats["max"],
        args.num_bins,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                **stats,
                "quantity": "RFDPIC displacement-field speed magnitude",
                "unit": "pixel displacement magnitude",
                "computed_streamingly": True,
                "histogram_bins": args.num_bins,
                "active_threshold_comment": "0.25 grid corresponds to 1 km displacement in 30 min. For a 4 km grid, this is about 0.56 m/s and is a more appropriate near-static threshold than 1 grid (2.22 m/s).",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic", "value"])
        for key in [
            "count",
            "mean",
            "var",
            "std",
            "min",
            "max",
            "active_count",
            "active_mean",
            "active_var",
            "active_std",
            "active_min",
            "active_max",
            "active_threshold_grid",
        ]:
            writer.writerow([key, stats[key]])

    save_vector_csv(os.path.join(args.out_dir, "per_timestep_mean.csv"), stats["per_timestep_mean"], "mean_speed")
    save_vector_csv(os.path.join(args.out_dir, "per_timestep_std.csv"), stats["per_timestep_std"], "std_speed")
    save_vector_csv(os.path.join(args.out_dir, "per_channel_mean.csv"), stats["per_channel_mean"], "mean_speed")
    save_vector_csv(os.path.join(args.out_dir, "per_channel_std.csv"), stats["per_channel_std"], "std_speed")
    save_histogram_csv(os.path.join(args.out_dir, "histogram.csv"), hist, edges)
    plot_histogram(
        hist,
        edges,
        os.path.join(args.out_dir, "speed_distribution.png"),
        os.path.join(args.out_dir, "speed_distribution.pdf"),
    )

    print(f"Saved RFDPIC speed distribution analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
