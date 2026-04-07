import argparse
import json
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_btchw import Himawari8LightningDataModule
from train_JiT_RFDPIC import VideoLightningModule as BaseVideoLightningModule
from videoMeanflow_RFDPIC_JiU_adaptive import AdaptiveMeanFlow


class AdaptiveSamplingVideoLightningModule(BaseVideoLightningModule):
    def __init__(self, *args, adaptive_method="dopri5", ode_rtol=1e-4, ode_atol=1e-5, **kwargs):
        self.adaptive_method = adaptive_method
        self.adaptive_ode_rtol = float(ode_rtol)
        self.adaptive_ode_atol = float(ode_atol)
        super().__init__(*args, **kwargs)

        self.meanflow = AdaptiveMeanFlow(
            channels=self.hparams.model_config["out_channels_c"],
            time_dim=self.hparams.model_config["input_size"][0],
            height_dim=self.hparams.model_config["input_size"][1],
            width_dim=self.hparams.model_config["input_size"][2],
            normalizer=["minmax", None, None],
            flow_ratio=self.hparams.meanflow_config["flow_ratio"],
            time_dist=self.hparams.meanflow_config["time_dist"],
            cfg_ratio=self.hparams.meanflow_config["cfg_ratio"],
            cfg_scale=self.hparams.meanflow_config["cfg_scale"],
            cfg_uncond=self.hparams.meanflow_config["cfg_uncond"],
            adaptive_method=self.adaptive_method,
            ode_rtol=self.adaptive_ode_rtol,
            ode_atol=self.adaptive_ode_atol,
        )
        print(
            f"--- Using adaptive second-stage sampler: method={self.adaptive_method}, "
            f"rtol={self.adaptive_ode_rtol}, atol={self.adaptive_ode_atol} ---"
        )

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.meanflow.nfe_history = []

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if self.trainer.is_global_zero:
            history = np.asarray(self.meanflow.nfe_history, dtype=np.float64)
            summary = {
                "adaptive_method": self.adaptive_method,
                "ode_rtol": self.adaptive_ode_rtol,
                "ode_atol": self.adaptive_ode_atol,
                "num_batches": int(history.size),
            }
            if history.size > 0:
                summary.update(
                    {
                        "nfe_mean": float(history.mean()),
                        "nfe_std": float(history.std()),
                        "nfe_min": float(history.min()),
                        "nfe_max": float(history.max()),
                    }
                )

            with open(os.path.join(self.metrics_save_dir, "adaptive_sampling_summary.json"), "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Saved adaptive sampling summary to {self.metrics_save_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="JiT-RFDPIC test with adaptive second-stage sampling")
    parser.add_argument("--config", type=str, default="configs/JiT-B_RFDPIC.yaml")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--rfdpic_config", type=str, required=True)
    parser.add_argument("--rfdpic_ckpt", type=str, required=True)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--adaptive_method", type=str, default="dopri5")
    parser.add_argument("--ode_rtol", type=float, default=1e-4)
    parser.add_argument("--ode_atol", type=float, default=1e-5)
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

    model = AdaptiveSamplingVideoLightningModule(
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
        adaptive_method=args.adaptive_method,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
    )

    if args.use_wandb:
        print("--- Using Weights & Biases Logger ---")
        logger_instance = WandbLogger(
            project=config["logging"]["project_name"],
            save_dir=args.log_dir,
            name=os.path.basename(args.log_dir),
        )
        logger_instance.watch(model, log="all", log_freq=500)
    else:
        print(f"--- WandbLogger is disabled, using TensorBoardLogger at {args.log_dir} ---")
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

    if args.mode == "train":
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path if args.ckpt_path else None)
        return

    if args.ckpt_path is None:
        raise ValueError("Must provide --ckpt_path for testing.")

    print(f"--- Starting Adaptive Second-Stage Testing ---")
    print(f"--- Manually loading checkpoint with strict=False: {args.ckpt_path} ---")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
