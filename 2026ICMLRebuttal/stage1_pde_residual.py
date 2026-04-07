import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_btchw import Himawari8LightningDataModule
from models.MotionPredictor.RFDPIC_Dual_Rotation_dyn import RFDPIC_Dual_Rotation_Dyn
from utils.transform import data_transform


class ResidualAccumulator:
    def __init__(self, num_channels=8, num_timesteps=6):
        shape = (num_channels, num_timesteps)
        self.count = np.zeros(shape, dtype=np.float64)
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sum_abs = np.zeros(shape, dtype=np.float64)
        self.sum_sq = np.zeros(shape, dtype=np.float64)
        self.ref_sq = np.zeros(shape, dtype=np.float64)

    def update(self, residual_tensor, ref_tensor):
        residual_np = residual_tensor.detach().cpu().numpy().astype(np.float64)
        ref_np = ref_tensor.detach().cpu().numpy().astype(np.float64)

        for c in range(residual_np.shape[2]):
            for t in range(residual_np.shape[1]):
                r = residual_np[:, t, c].reshape(-1)
                y = ref_np[:, t, c].reshape(-1)
                self.count[c, t] += r.size
                self.sum[c, t] += r.sum()
                self.sum_abs[c, t] += np.abs(r).sum()
                self.sum_sq[c, t] += np.square(r).sum()
                self.ref_sq[c, t] += np.square(y).sum()

    def finalize(self):
        eps = 1e-12
        n = np.maximum(self.count, 1.0)
        mae = self.sum_abs / n
        rmse = np.sqrt(self.sum_sq / n)
        bias = self.sum / n
        relative_l2_error = np.sqrt(self.sum_sq) / np.maximum(np.sqrt(self.ref_sq), eps)
        return {
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "relative_l2_error": relative_l2_error,
        }


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

    def finalize(self):
        eps = 1e-12
        n = np.maximum(self.count, 1.0)
        cov_num = n * self.sum_xy - self.sum_x * self.sum_y
        cov_den = np.sqrt(
            np.maximum(n * self.sum_x2 - np.square(self.sum_x), eps)
            * np.maximum(n * self.sum_y2 - np.square(self.sum_y), eps)
        )
        pearson_r = cov_num / np.maximum(cov_den, eps)
        mae = self.sum_abs / n
        rmse = np.sqrt(self.sum_sq / n)
        cosine_similarity = self.sum_xy / np.maximum(np.sqrt(self.sum_x2 * self.sum_y2), eps)
        return {
            "pearson_r": pearson_r,
            "cosine_similarity": cosine_similarity,
            "mae": mae,
            "rmse": rmse,
        }


def save_matrix_csv(path, array):
    np.savetxt(path, array, delimiter=",", fmt="%.6f")


def summarize_metrics(metric_dict):
    summary = {}
    for metric_name, values in metric_dict.items():
        summary[metric_name] = {
            "overall_mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def save_summary_csv(path, summary):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "metric", "overall_mean", "min", "max"])
        for group_name, metrics in summary.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, values in metrics.items():
                if isinstance(values, dict) and "overall_mean" in values:
                    writer.writerow([group_name, metric_name, values["overall_mean"], values["min"], values["max"]])


def build_parser():
    parser = argparse.ArgumentParser(description="Semi-Lagrangian integral residual analysis for stage-1 RFDPIC outputs")
    parser.add_argument("--rfdpic_config", type=str, default="configs/rfdpic_config.yaml")
    parser.add_argument("--rfdpic_ckpt", type=str, default="pretrained_models/pretrained_RFDPIC_Dual_Rotation_Dyn.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--source_integral_rule",
        type=str,
        default="trapezoid",
        choices=["trapezoid", "right", "left"],
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_pde_residual"),
    )
    return parser


def resolve_model_cfg_path(rfdpic_config_path, model_cfg_path):
    base_dir = os.path.dirname(rfdpic_config_path)
    candidate = os.path.join(base_dir, model_cfg_path)
    if os.path.exists(candidate):
        return candidate
    return model_cfg_path


def load_frozen_rfdpic(rfdpic_config_path, rfdpic_ckpt_path):
    with open(rfdpic_config_path, "r") as f:
        rfdpic_main_cfg = yaml.safe_load(f)

    model_cfg = rfdpic_main_cfg["model"]
    model_cfg_path = resolve_model_cfg_path(rfdpic_config_path, model_cfg["rf_dp_config_file"])
    model = RFDPIC_Dual_Rotation_Dyn(
        dp_name=model_cfg["dp_name"],
        rf_name=model_cfg["rf_name"],
        rf_dp_config_file=model_cfg_path,
        dp_mode=model_cfg["dp_mode"],
        rf_mode=model_cfg["rf_mode"],
        alpha=model_cfg["alpha"],
        interpolation_mode=model_cfg.get("interpolation_mode", "bilinear"),
        padding_mode=model_cfg.get("padding_mode", "border"),
        use_res_scale=model_cfg.get("use_res_scale", False),
        res_scale_init=model_cfg.get("res_scale_init", 0.05),
        advect_mode=model_cfg.get("advect_mode", "semi-lagrangian"),
        ode_method=model_cfg.get("ode_method", "rk4"),
        ode_substeps=model_cfg.get("ode_substeps", 1),
        use_splat=model_cfg.get("use_splat", False),
        splat_type=model_cfg.get("splat_type", "softmax"),
        splat_alpha=model_cfg.get("splat_alpha", 20.0),
        splat_blend=model_cfg.get("splat_blend", 0.5),
        use_gate=model_cfg.get("use_gate", False),
        use_implicit_diffusion=model_cfg.get("use_implicit_diffusion", False),
        jacobi_iters=model_cfg.get("jacobi_iters", 2),
        D_max=model_cfg.get("D_max", 0.2),
        k_edge=model_cfg.get("k_edge", 0.1),
        diff_dt=model_cfg.get("diff_dt", 1.0),
        diff_dx=model_cfg.get("diff_dx", 1.0),
        D_const=model_cfg.get("D_const", None),
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

    return model, rfdpic_main_cfg


def backward_sample(field, displacement_t, interpolation_mode="bilinear", padding_mode="border"):
    b, c, h, w = field.shape
    device = field.device
    dtype = field.dtype

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)[None, None].expand(b, c, -1, -1, -1)
    backtraced = base_grid + displacement_t
    backtraced_x = 2.0 * backtraced[..., 0] / max(w - 1, 1) - 1.0
    backtraced_y = 2.0 * backtraced[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([backtraced_x, backtraced_y], dim=-1).reshape(b * c, h, w, 2)
    sampled = torch.nn.functional.grid_sample(
        field.contiguous().reshape(b * c, 1, h, w),
        grid,
        mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return sampled.reshape(b, c, h, w)


def source_integral(prev_source, cur_source, displacement_t, dt, rule, interpolation_mode, padding_mode):
    if prev_source is None:
        prev_source_at_old = torch.zeros_like(cur_source)
    else:
        prev_source_at_old = backward_sample(
            prev_source,
            displacement_t,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
        )

    if rule == "trapezoid":
        return 0.5 * (prev_source_at_old + cur_source) * dt
    if rule == "right":
        return cur_source * dt
    if rule == "left":
        return prev_source_at_old * dt
    raise ValueError(f"Unsupported source integral rule: {rule}")


def get_source_scale(model, cur_source):
    if not getattr(model, "use_res_scale", False):
        return torch.tensor(1.0, device=cur_source.device, dtype=cur_source.dtype)
    scale = getattr(model, "res_gate", None)
    if scale is None:
        return torch.tensor(1.0, device=cur_source.device, dtype=cur_source.dtype)
    return scale.to(device=cur_source.device, dtype=cur_source.dtype)


def stack_time_list(tensors):
    return torch.stack(tensors, dim=1)


def main():
    args = build_parser().parse_args()

    model, rfdpic_cfg = load_frozen_rfdpic(args.rfdpic_config, args.rfdpic_ckpt)
    rescaled = rfdpic_cfg.get("dataset", {}).get("rescaled", True)
    dataset_cfg = rfdpic_cfg["dataset"]

    pl.seed_everything(42, workers=True)
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
        batch_size=args.batch_size,
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    model = model.to(device)
    model.eval()

    gt_lagrangian_acc = ResidualAccumulator()
    stage1_lagrangian_acc = ResidualAccumulator()
    advection_only_acc = ResidualAccumulator()
    source_match_acc = PairwiseAccumulator()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            c_past, x_future = batch
            c_past = c_past.to(device)
            x_future = x_future.to(device)

            c_past_norm = data_transform(c_past, rescaled=rescaled)
            x_future_norm = data_transform(x_future, rescaled=rescaled)

            coarse_norm, displacement_fields, residual_fields, _, _ = model(c_past_norm)

            gt_lagrangian_steps = []
            stage1_lagrangian_steps = []
            advection_only_steps = []
            pred_integral_steps = []
            gt_integral_steps = []

            prev_stage1 = c_past_norm[:, -1]
            prev_ref = c_past_norm[:, -1]
            prev_source = None

            for t in range(x_future_norm.size(1)):
                disp_t = displacement_fields[:, t]
                cur_source = (
                    residual_fields[:, t]
                    if residual_fields is not None
                    else torch.zeros_like(prev_ref)
                )

                gt_old = backward_sample(
                    prev_ref,
                    disp_t,
                    interpolation_mode=model.interpolation_mode,
                    padding_mode=model.padding_mode,
                )
                stage1_old = backward_sample(
                    prev_stage1,
                    disp_t,
                    interpolation_mode=model.interpolation_mode,
                    padding_mode=model.padding_mode,
                )
                integral = source_integral(
                    prev_source=prev_source,
                    cur_source=cur_source,
                    displacement_t=disp_t,
                    dt=args.dt,
                    rule=args.source_integral_rule,
                    interpolation_mode=model.interpolation_mode,
                    padding_mode=model.padding_mode,
                )
                integral = get_source_scale(model, cur_source) * integral

                gt_lagrangian_steps.append(x_future_norm[:, t] - gt_old - integral)
                stage1_lagrangian_steps.append(coarse_norm[:, t] - stage1_old - integral)
                advection_only_steps.append(x_future_norm[:, t] - gt_old)

                pred_integral_steps.append(integral)
                gt_integral_steps.append(x_future_norm[:, t] - gt_old)

                prev_stage1 = coarse_norm[:, t]
                prev_ref = x_future_norm[:, t]
                prev_source = cur_source

            gt_lagrangian_residual = stack_time_list(gt_lagrangian_steps)
            stage1_lagrangian_residual = stack_time_list(stage1_lagrangian_steps)
            advection_only_residual = stack_time_list(advection_only_steps)
            pred_integral = stack_time_list(pred_integral_steps)
            gt_integral = stack_time_list(gt_integral_steps)

            gt_lagrangian_acc.update(gt_lagrangian_residual, x_future_norm)
            stage1_lagrangian_acc.update(stage1_lagrangian_residual, coarse_norm)
            advection_only_acc.update(advection_only_residual, x_future_norm)
            source_match_acc.update(pred_integral, gt_integral)

            if batch_idx % 20 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    os.makedirs(args.out_dir, exist_ok=True)

    gt_lagrangian_metrics = gt_lagrangian_acc.finalize()
    stage1_lagrangian_metrics = stage1_lagrangian_acc.finalize()
    advection_only_metrics = advection_only_acc.finalize()
    source_match_metrics = source_match_acc.finalize()

    for metric_name, values in gt_lagrangian_metrics.items():
        save_matrix_csv(os.path.join(args.out_dir, f"gt_lagrangian_residual_{metric_name}.csv"), values)
    for metric_name, values in stage1_lagrangian_metrics.items():
        save_matrix_csv(os.path.join(args.out_dir, f"stage1_lagrangian_residual_{metric_name}.csv"), values)
    for metric_name, values in advection_only_metrics.items():
        save_matrix_csv(os.path.join(args.out_dir, f"gt_advection_only_increment_{metric_name}.csv"), values)
    for metric_name, values in source_match_metrics.items():
        save_matrix_csv(os.path.join(args.out_dir, f"source_integral_match_{metric_name}.csv"), values)

    summary = {
        "definition": {
            "description": "Semi-Lagrangian integral residual analysis for the stage-1 RFDPIC model. We backtrace each arrival grid point with the predicted displacement field, sample the previous state at the departure point, and compare the arrival state against the source-term path integral approximation.",
            "gt_lagrangian_residual": "R_SL^gt(t) = x_gt(t) - x_gt_old(t) - Integral[S d tau], where x_gt_old(t) is the GT previous frame sampled at the backtraced departure point.",
            "stage1_lagrangian_residual": "R_SL^stage1(t) = x_stage1(t) - x_stage1_old(t) - Integral[S d tau], where x_stage1_old(t) is the previous stage-1 frame sampled at the same departure point.",
            "source_integral": "Integral[S d tau] is approximated from the predicted residual/source fields using the selected quadrature rule.",
            "source_scale": float(getattr(model, "res_gate", torch.tensor(1.0)).detach().cpu().item())
            if getattr(model, "use_res_scale", False)
            else 1.0,
            "use_res_scale": bool(getattr(model, "use_res_scale", False)),
            "gt_advection_only_increment": "Delta_gt(t) = x_gt(t) - x_gt_old(t). This is the implied GT source accumulation without subtracting the predicted source integral.",
            "source_match": "Compare predicted source integral against the GT implied increment.",
            "dt": float(args.dt),
            "source_integral_rule": args.source_integral_rule,
            "physical_space": False,
            "rescaled_input": bool(rescaled),
        },
        "gt_lagrangian_residual": summarize_metrics(gt_lagrangian_metrics),
        "stage1_lagrangian_residual": summarize_metrics(stage1_lagrangian_metrics),
        "gt_advection_only_increment": summarize_metrics(advection_only_metrics),
        "source_integral_match": summarize_metrics(source_match_metrics),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_summary_csv(os.path.join(args.out_dir, "summary.csv"), summary)

    print(f"Saved stage-1 PDE residual results to {args.out_dir}")


if __name__ == "__main__":
    main()
