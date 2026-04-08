#!/usr/bin/env python3
import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_btchw import Himawari8LightningDataModule
from evaluation.psd.psd_tools import plot_sample_psd
from train_JiT_RFDPIC import VideoLightningModule as BaseVideoLightningModule
from visualize import vis_himawari8_seq_btchw


class PersistenceBaselineLightningModule(BaseVideoLightningModule):
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        c_past, x_future = batch

        c_past_norm = self._to_rfdpic_input(c_past)
        c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
        c_rfdpic = self._from_rfdpic_output(c_rfdpic_norm).detach()

        last_context = c_past[:, -1:, :, :, :]
        preds_norm = last_context.repeat(1, x_future.shape[1], 1, 1, 1)
        targets_norm = x_future

        preds_denorm = self.denormalize(preds_norm, mode="test")
        targets_denorm = self.denormalize(targets_norm, mode="test")

        self.test_d_mse(preds_denorm, targets_denorm)
        self.test_d_mae(preds_denorm, targets_denorm)

        preds_bchw = rearrange(preds_norm, "b t c h w -> (b t) c h w")
        targets_bchw = rearrange(targets_norm, "b t c h w -> (b t) c h w")
        self.test_ssim(preds_bchw, targets_bchw)
        self.test_psnr(preds_bchw, targets_bchw)

        real_video_t12 = torch.cat([c_past, targets_norm], dim=1)
        fake_video_t12 = torch.cat([c_past, preds_norm], dim=1)

        for c in range(self.num_channels):
            real_ch_video = real_video_t12[:, :, c : c + 1, :, :]
            fake_ch_video = fake_video_t12[:, :, c : c + 1, :, :]
            self.test_fvd_list[c].update(real_ch_video, real=True)
            self.test_fvd_list[c].update(fake_ch_video, real=False)

        self.test_psd_gt.update(targets_norm)
        self.test_psd_pred.update(preds_norm)
        self.test_psd_rfdpic.update(c_rfdpic)

        for c in range(self.num_channels):
            for t in range(self.num_timesteps):
                pred_slice = preds_denorm[:, t, c].contiguous()
                target_slice = targets_denorm[:, t, c].contiguous()
                self.test_d_mae_metric[c][t](pred_slice, target_slice)
                self.test_d_mse_metric[c][t](pred_slice, target_slice)
                self.test_d_rmse_metric[c][t](pred_slice, target_slice)

        batch_mean_p = preds_denorm.mean(dim=(0, 3, 4))
        batch_mean_t = targets_denorm.mean(dim=(0, 3, 4))
        self.running_mean_pred += batch_mean_p.T
        self.running_mean_target += batch_mean_t.T
        self.test_step_count += 1.0

        micro_batch_size = c_past.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if data_idx in self.test_example_data_idx_list:
            c_past_sample = c_past[0].cpu()
            preds_sample = preds_norm[0].cpu()
            targets_sample = targets_norm[0].cpu()

            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [preds_sample[t] for t in range(preds_sample.shape[0])]
            target_list = [targets_sample[t] for t in range(targets_sample.shape[0])]

            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}")
            vis_himawari8_seq_btchw(
                save_dir=save_dir,
                context_seq=context_list,
                pred_seq=pred_list,
                target_seq=target_list,
            )

            try:
                plot_sample_psd(
                    pred_4d=preds_sample.numpy(),
                    rfdpic_4d=c_rfdpic[0].cpu().numpy(),
                    gt_4d=targets_sample.numpy(),
                    save_dir=save_dir,
                )
            except Exception as e:
                print(f"Failed to generate sample PSD plot for sample {data_idx}: {e}")

    def _to_rfdpic_input(self, c_past):
        from utils.transform import data_transform

        return data_transform(c_past, rescaled=self.rfdpic_rescaled)

    def _from_rfdpic_output(self, c_rfdpic_norm):
        from utils.transform import inverse_data_transform

        return inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled)


def build_parser():
    parser = argparse.ArgumentParser(description="Meteorological persistence baseline test")
    parser.add_argument("--config", type=str, default="configs/JiT-B_RFDPIC.yaml")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mode", type=str, default="test", choices=["test"])
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--rfdpic_config", type=str, required=True)
    parser.add_argument("--rfdpic_ckpt", type=str, required=True)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
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
        batch_size=args.batch_size,
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
    )

    model = PersistenceBaselineLightningModule(
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

    if args.use_wandb:
        logger_instance = WandbLogger(
            project=config["logging"]["project_name"],
            save_dir=args.log_dir,
            name=os.path.basename(args.log_dir),
        )
        logger_instance.watch(model, log="all", log_freq=500)
    else:
        logger_instance = TensorBoardLogger(save_dir=args.log_dir, name="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="step_{step:06d}-loss_{train/loss:.4f}",
        every_n_train_steps=config["logging"]["save_step_frequency"],
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    use_gpu = args.gpus > 0 and torch.cuda.is_available()
    trainer = pl.Trainer(
        logger=logger_instance,
        callbacks=[checkpoint_callback],
        max_steps=config["training"]["n_steps"],
        devices=args.gpus if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        precision=32,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
    )

    print("--- Starting Persistence Baseline Testing ---")
    print("--- Prediction rule: repeat the last context frame across all future lead times ---")
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
