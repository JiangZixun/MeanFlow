import os
import copy
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset_btchw import Himawari8LightningDataModule
from models.ViT import JiT
from videoMeanflow_RFDPIC_JiU import MeanFlow
from train_JiT_RFDPIC import VideoLightningModule as BaseVideoLightningModule
from utils.transform import data_transform, inverse_data_transform


class VideoLightningModule(BaseVideoLightningModule):
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
        rfdpic_config_path: str,
        rfdpic_ckpt_path: str,
        sample_steps: int,
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

        self.rollout_steps = 6
        self.cond_steps = 2
        self.frame_channels = model_config["out_channels_c"]

        iter_model_config = copy.deepcopy(model_config)
        iter_model_config["input_size"] = [1, model_config["input_size"][1], model_config["input_size"][2]]
        iter_model_config["in_channels_c"] = self.frame_channels + self.cond_steps * self.frame_channels

        print("--- Replacing MeanFlow stage with iterative 2->1 prediction ---")
        print(f"Iterative JiT input_size: {iter_model_config['input_size']}")
        print(f"Iterative JiT in_channels_c: {iter_model_config['in_channels_c']}")

        self.meanflow = MeanFlow(
            channels=self.frame_channels,
            time_dim=1,
            height_dim=model_config["input_size"][1],
            width_dim=model_config["input_size"][2],
            normalizer=["minmax", None, None],
            flow_ratio=meanflow_config["flow_ratio"],
            time_dist=meanflow_config["time_dist"],
            cfg_ratio=meanflow_config["cfg_ratio"],
            cfg_scale=meanflow_config["cfg_scale"],
            cfg_uncond=meanflow_config["cfg_uncond"],
        )

        self.model = JiT(
            input_size=tuple(iter_model_config["input_size"]),
            in_channels_c=iter_model_config["in_channels_c"],
            out_channels_c=iter_model_config["out_channels_c"],
            time_emb_dim=iter_model_config["time_emb_dim"],
            patch_size=iter_model_config["patch_size"],
            hidden_size=iter_model_config["hidden_size"],
            depth=iter_model_config["depth"],
            num_heads=iter_model_config["num_heads"],
            mlp_ratio=iter_model_config["mlp_ratio"],
            bottleneck_dim=iter_model_config["bottleneck_dim"],
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        current_state = self.state_dict()
        filtered_state = {}
        skipped = []

        for key, value in state_dict.items():
            if key not in current_state:
                continue
            if current_state[key].shape != value.shape:
                skipped.append(key)
                continue
            filtered_state[key] = value

        if skipped:
            print(f"Skipped loading {len(skipped)} mismatched keys for iterative model.")

        super().load_state_dict(filtered_state, strict=False)

    def _pack_condition_window(self, cond_window):
        b, t, c, h, w = cond_window.shape
        if t != self.cond_steps:
            raise ValueError(f"Expected {self.cond_steps} condition frames, got {t}")
        return cond_window.reshape(b, 1, t * c, h, w)

    def _iter_teacher_forcing_triplets(self, c_past, x_future, c_rfdpic):
        for step in range(self.rollout_steps):
            if step == 0:
                cond_window = c_past[:, -2:, ...]
            elif step == 1:
                cond_window = torch.cat([c_past[:, -1:, ...], x_future[:, 0:1, ...]], dim=1)
            else:
                cond_window = x_future[:, step - 2:step, ...]

            yield (
                x_future[:, step:step + 1, ...],
                c_rfdpic[:, step:step + 1, ...],
                self._pack_condition_window(cond_window),
            )

    @torch.no_grad()
    def rollout_iterative_prediction(self, c_past, c_rfdpic, sample_steps=None):
        sample_steps = self.sample_steps if sample_steps is None else sample_steps
        history = [c_past[:, -2:-1, ...], c_past[:, -1:, ...]]
        preds = []

        for step in range(self.rollout_steps):
            cond_window = torch.cat(history[-2:], dim=1)
            c_cond = self._pack_condition_window(cond_window)
            c_start = c_rfdpic[:, step:step + 1, ...]
            pred = self.meanflow.sample_prediction(
                self.model,
                (c_start, c_cond),
                sample_steps=sample_steps,
                device=self.device,
            )
            preds.append(pred)
            history.append(pred)

        return torch.cat(preds, dim=1)

    def training_step(self, batch, batch_idx):
        c_past, x_future = batch

        with torch.no_grad():
            c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
            c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
            c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled).detach()

        total_loss = 0.0
        total_mse = 0.0

        # 顺序计算 6 个单步损失，避免把 batch 在显存里放大 6 倍。
        for x_target, c_start, c_cond in self._iter_teacher_forcing_triplets(c_past, x_future, c_rfdpic):
            step_loss, step_mse = self.meanflow.loss(self.model, x_target, c=(c_start, c_cond))
            total_loss = total_loss + step_loss
            total_mse = total_mse + step_mse

        loss = total_loss / self.rollout_steps
        mse_val = total_mse / self.rollout_steps

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/mse_loss", mse_val, on_step=True, on_epoch=False)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        save_freq = self.hparams.logging_config["save_step_frequency"]

        if self.trainer.is_global_zero and self.global_step > 0 and self.global_step % save_freq == 0:
            self.model.eval()

            if self.val_loader_iter is None:
                self.val_loader_iter = iter(self.trainer.datamodule.val_dataloader())
            try:
                val_batch = next(self.val_loader_iter)
            except StopIteration:
                self.val_loader_iter = iter(self.trainer.datamodule.val_dataloader())
                val_batch = next(self.val_loader_iter)

            c_past_val, x_future_val = val_batch
            c_past_val = c_past_val.to(self.device)
            x_future_val = x_future_val.to(self.device)

            with torch.no_grad():
                c_past_norm_val = data_transform(c_past_val, rescaled=self.rfdpic_rescaled)
                c_rfdpic_norm_val, _, _, _, _ = self.rfdpic_model(c_past_norm_val)
                c_rfdpic_val = inverse_data_transform(c_rfdpic_norm_val, rescaled=self.rfdpic_rescaled)
                z = self.rollout_iterative_prediction(c_past_val, c_rfdpic_val, sample_steps=10)

            c_past_sample = c_past_val[0].cpu()
            z_sample = z[0].cpu()
            x_future_sample = x_future_val[0].cpu()
            c_rfdpic_sample = c_rfdpic_val[0].cpu()
            del c_past_val, x_future_val, z, val_batch, c_rfdpic_val
            torch.cuda.empty_cache()

            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [z_sample[t] for t in range(z_sample.shape[0])]
            target_list = [x_future_sample[t] for t in range(x_future_sample.shape[0])]
            pred_rfdpic_list = [c_rfdpic_sample[t] for t in range(c_rfdpic_sample.shape[0])]

            save_dir = os.path.join(self.trainer.logger.save_dir, "images", f"step_{self.global_step}")
            from visualize import vis_himawari8_seq_btchw

            vis_himawari8_seq_btchw(save_dir=save_dir, context_seq=context_list, pred_seq=pred_list, target_seq=target_list)
            vis_himawari8_seq_btchw(
                save_dir=os.path.join(self.trainer.logger.save_dir, "images", f"step_{self.global_step}_rfdpic_cond"),
                context_seq=context_list,
                pred_seq=pred_rfdpic_list,
                target_seq=target_list,
            )
            self.model.train()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        c_past, x_future = batch

        c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
        c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
        c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled).detach()

        preds_norm = self.rollout_iterative_prediction(c_past, c_rfdpic)
        targets_norm = x_future

        preds_denorm = self.denormalize(preds_norm, mode="test")
        targets_denorm = self.denormalize(targets_norm, mode="test")

        self.test_d_mse(preds_denorm, targets_denorm)
        self.test_d_mae(preds_denorm, targets_denorm)

        from einops import rearrange

        preds_bchw = rearrange(preds_norm, "b t c h w -> (b t) c h w")
        targets_bchw = rearrange(targets_norm, "b t c h w -> (b t) c h w")
        self.test_ssim(preds_bchw, targets_bchw)
        self.test_psnr(preds_bchw, targets_bchw)

        real_video_t12 = torch.cat([c_past, targets_norm], dim=1)
        fake_video_t12 = torch.cat([c_past, preds_norm], dim=1)

        for c in range(self.num_channels):
            real_ch_video = real_video_t12[:, :, c:c + 1, :, :]
            fake_ch_video = fake_video_t12[:, :, c:c + 1, :, :]
            self.test_fvd_list[c].update(real_ch_video, real=True)
            self.test_fvd_list[c].update(fake_ch_video, real=False)

        self.test_psd_gt.update(targets_norm)
        self.test_psd_pred.update(preds_norm)
        self.test_psd_rfdpic.update(c_rfdpic)

        for c in range(self.num_channels):
            for t in range(self.num_timesteps):
                self.test_d_mae_metric[c][t](preds_denorm[:, t, c].contiguous(), targets_denorm[:, t, c].contiguous())
                self.test_d_mse_metric[c][t](preds_denorm[:, t, c].contiguous(), targets_denorm[:, t, c].contiguous())
                self.test_d_rmse_metric[c][t](preds_denorm[:, t, c].contiguous(), targets_denorm[:, t, c].contiguous())

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
            from visualize import vis_himawari8_seq_btchw
            from evaluation.psd.psd_tools import plot_sample_psd

            vis_himawari8_seq_btchw(
                save_dir=save_dir,
                context_seq=context_list,
                pred_seq=pred_list,
                target_seq=target_list,
            )

            try:
                rfdpic_sample_np = c_rfdpic[0].cpu().numpy()
                preds_sample_np = preds_sample.numpy()
                targets_sample_np = targets_sample.numpy()
                plot_sample_psd(
                    pred_4d=preds_sample_np,
                    rfdpic_4d=rfdpic_sample_np,
                    gt_4d=targets_sample_np,
                    save_dir=save_dir,
                )
            except Exception as e:
                print(f"Failed to generate sample PSD plot for sample {data_idx}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Iterative 2->1 MeanFlow over 6-step RFDPIC coarse predictions")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the MeanFlow config.yaml file")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs and checkpoints")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config if set)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Training or testing mode")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming MeanFlow training or testing")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--rfdpic_config", type=str, required=True, help="Path to the RFDPIC main config.yaml")
    parser.add_argument("--rfdpic_ckpt", type=str, required=True, help="Path to the RFDPIC model checkpoint (.pt or .ckpt)")
    parser.add_argument("--sample_steps", type=int, default=10, help="Number of sampling steps for MeanFlow")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandbLogger (default: False)")
    args = parser.parse_args()

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

    trainer = pl.Trainer(
        logger=logger_instance,
        callbacks=[checkpoint_callback],
        max_steps=config["training"]["n_steps"],
        devices=args.gpus,
        accelerator="gpu",
        precision=32,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
    )

    if args.mode == "train":
        ckpt_to_resume = args.ckpt_path if args.ckpt_path else None
        if ckpt_to_resume:
            print(f"--- 正在从 MeanFlow checkpoint 延续训练: {ckpt_to_resume} ---")
        else:
            print("--- 从头开始训练 MeanFlow iterative 2->1 ---")
        trainer.fit(model, datamodule, ckpt_path=ckpt_to_resume)
    elif args.mode == "test":
        if args.ckpt_path is None:
            raise ValueError("Must provide --ckpt_path for testing.")

        print(f"--- Starting Testing ---")
        print(f"--- Manually loading checkpoint with shape filtering: {args.ckpt_path} ---")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
