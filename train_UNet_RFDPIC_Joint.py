# train_UNet_RFDPIC_Joint.py

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_linear_schedule_with_warmup
from collections import OrderedDict
import torchmetrics
import torch.nn as nn
from einops import rearrange
import numpy as np
import json
from evaluation.fvd.torchmetrics_wrap import FrechetVideoDistance
from evaluation.psd.psd_metric import PSDAverageMetric
from evaluation.psd.psd_tools import plot_sample_psd, calculate_psd_error_metrics

from models.MotionPredictor.RFDPIC_Dual_Rotation_dyn import RFDPIC_Dual_Rotation_Dyn
from utils.transform import data_transform, inverse_data_transform
from dataset_btchw import Himawari8LightningDataModule
from models.UNet import UNet
from videoMeanflow_RFDPIC import MeanFlow 
from visualize import vis_himawari8_seq_btchw, plot_metrics_curve, vis_dynamics_feats, visualize_channel_fields, visualize_channel_residuals, plot_himawari8_seq_btchw, save_distribution_plot_final


class VideoLightningModule(pl.LightningModule):
    def __init__(self, model_config, data_config, meanflow_config, optimizer_config, scheduler_config, training_config, logging_config, eval_config,
                 rfdpic_config_path: str,
                 rfdpic_ckpt_path: str,
                 sample_steps: int):
        
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 加载 RFDPIC
        print(f"Loading RFDPIC config from: {rfdpic_config_path}")
        with open(rfdpic_config_path, 'r') as f:
            rfdpic_main_cfg = yaml.safe_load(f) 
            self.rfdpic_rescaled = rfdpic_main_cfg.get('dataset', {}).get('rescaled', True)
            rfdpic_base_dir = os.path.dirname(rfdpic_config_path)
            model_cfg_path = rfdpic_main_cfg.get('model', {}).get('rf_dp_config_file')
            self.rfdpic_model_cfg_path = os.path.join(rfdpic_base_dir, model_cfg_path)
            if not os.path.exists(self.rfdpic_model_cfg_path):
                self.rfdpic_model_cfg_path = model_cfg_path
            print(f"Using RFDPIC model config: {self.rfdpic_model_cfg_path}")
            
        # 🟢 修改：调用解冻加载函数
        self.rfdpic_model = self.load_trainable_rfdpic(
            rfdpic_config_path=rfdpic_config_path,
            rfdpic_ckpt_path=rfdpic_ckpt_path
        )

        # 2. 实例化 MeanFlow
        self.meanflow = MeanFlow(
            channels=model_config['out_channels_c'],
            time_dim=model_config['input_size'][0],
            height_dim=model_config['input_size'][1],
            width_dim=model_config['input_size'][2],
            normalizer=['minmax', None, None],
            flow_ratio=meanflow_config['flow_ratio'],
            time_dist=meanflow_config['time_dist'],
            cfg_ratio=meanflow_config['cfg_ratio'],
            cfg_scale=meanflow_config['cfg_scale'],
            cfg_uncond=meanflow_config['cfg_uncond']
        )
        
        self.sample_steps = sample_steps
        print(f"--- Using {self.sample_steps} sampling steps for prediction ---")

        # 3. 实例化 U-Net
        print(f"Initializing UNet with condition input_t (in_channels_c): {model_config['in_channels_c']}")
        self.model = UNet(
            input_size=model_config['input_size'],
            in_channels_c=model_config['in_channels_c'], 
            out_channels_c=model_config['out_channels_c'],
            time_emb_dim=model_config['time_emb_dim']
        )
        
        self.val_loader_iter = None

        # 4. 初始化指标
        self.num_channels = self.hparams.model_config['out_channels_c']
        self.num_timesteps = 6 
        
        self.test_d_mse = torchmetrics.MeanSquaredError()
        self.test_d_mae = torchmetrics.MeanAbsoluteError()
        self.test_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        
        self.test_d_mse_metric = nn.ModuleList([
            nn.ModuleList([torchmetrics.MeanSquaredError() for _ in range(self.num_timesteps)]) 
            for _ in range(self.num_channels)
        ])
        self.test_d_mae_metric = nn.ModuleList([
            nn.ModuleList([torchmetrics.MeanAbsoluteError() for _ in range(self.num_timesteps)]) 
            for _ in range(self.num_channels)
        ])
        self.test_d_rmse_metric = nn.ModuleList([
            nn.ModuleList([torchmetrics.MeanSquaredError(squared=False) for _ in range(self.num_timesteps)]) 
            for _ in range(self.num_channels)
        ])
        
        self.test_fvd_list = nn.ModuleList([
            FrechetVideoDistance(feature=400, normalize=False) 
            for _ in range(self.num_channels)
        ])

        H_W = self.hparams.model_config['input_size'][1]
        self.test_psd_gt = PSDAverageMetric(H=H_W, W=H_W)
        self.test_psd_pred = PSDAverageMetric(H=H_W, W=H_W)
        self.test_psd_rfdpic = PSDAverageMetric(H=H_W, W=H_W)
        
        self.train_example_data_idx_list = eval_config['train_example_data_idx_list']
        self.val_example_data_idx_list = eval_config['val_example_data_idx_list']
        self.test_example_data_idx_list = eval_config['test_example_data_idx_list']

        with open(data_config['train_json_path'], 'r') as f:
            train_global_stats = json.load(f)
        train_global_max_values = torch.tensor(train_global_stats['Global Max'], dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        train_global_min_values = torch.tensor(train_global_stats['Global Min'], dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        self.register_buffer('train_global_min_values', train_global_min_values)
        self.register_buffer('train_global_range', train_global_max_values - train_global_min_values)
        
        with open(data_config['test_json_path'], 'r') as f:
            test_global_stats = json.load(f)
        test_global_max_values = torch.tensor(test_global_stats['Global Max'], dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        test_global_min_values = torch.tensor(test_global_stats['Global Min'], dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        self.register_buffer('test_global_min_values', test_global_min_values)
        self.register_buffer('test_global_range', test_global_max_values - test_global_min_values)
        
        self.plot_dict = {
            'train': {'max': train_global_stats['Global Max'], 'min': train_global_stats['Global Min']},
            'val': {'max': train_global_stats['Global Max'], 'min': train_global_stats['Global Min']},
            'test': {'max': test_global_stats['Global Max'], 'min': test_global_stats['Global Min']}
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=False)
    
    def load_trainable_rfdpic(self, rfdpic_config_path, rfdpic_ckpt_path):
        """🟢 修改：解冻 RFDPIC 模型以便联合训练"""
        print(f"Loading RFDPIC model for joint training from {rfdpic_ckpt_path}")
        with open(rfdpic_config_path, 'r') as f:
            rfdpic_cfg = yaml.safe_load(f)['model']
        model = RFDPIC_Dual_Rotation_Dyn(
            dp_name=rfdpic_cfg['dp_name'],
            rf_name=rfdpic_cfg['rf_name'],
            rf_dp_config_file=self.rfdpic_model_cfg_path,
            dp_mode=rfdpic_cfg['dp_mode'],
            rf_mode=rfdpic_cfg['rf_mode'],
            alpha=rfdpic_cfg['alpha'],
            interpolation_mode=rfdpic_cfg.get('interpolation_mode', "bilinear"),
            padding_mode=rfdpic_cfg.get('padding_mode', "border"),
            use_res_scale=rfdpic_cfg.get('use_res_scale', False),
            res_scale_init=rfdpic_cfg.get('res_scale_init', 0.05),
            advect_mode=rfdpic_cfg.get('advect_mode', 'semi-lagrangian'),
            ode_method=rfdpic_cfg.get('ode_method', 'rk4'),
            ode_substeps=rfdpic_cfg.get('ode_substeps', 1),
            use_splat=rfdpic_cfg.get('use_splat', False),
            splat_type=rfdpic_cfg.get('splat_type', 'softmax'),
            splat_alpha=rfdpic_cfg.get('splat_alpha', 20.0),
            splat_blend=rfdpic_cfg.get('splat_blend', 0.5),
            use_gate=rfdpic_cfg.get('use_gate', False),
            use_implicit_diffusion=rfdpic_cfg.get('use_implicit_diffusion', False),
            jacobi_iters=rfdpic_cfg.get('jacobi_iters', 2),
            D_max=rfdpic_cfg.get('D_max', 0.2),
            k_edge=rfdpic_cfg.get('k_edge', 0.1),
            diff_dt=rfdpic_cfg.get('diff_dt', 1.0),
            diff_dx=rfdpic_cfg.get('diff_dx', 1.0),
            D_const=rfdpic_cfg.get('D_const', None),
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
        
        # 🟢 确保参数可训练且默认不强制 eval
        for param in model.parameters():
            param.requires_grad = True
        
        print(f"{'='*10} RFDPIC model loaded and UNFROZEN for joint training. {'='*10}")
        return model

    def denormalize(self, data: torch.Tensor, mode: str=''):
        if mode == 'train' or mode == 'val':
            return data * self.train_global_range + self.train_global_min_values
        elif mode == 'test':
            return data * self.test_global_range + self.test_global_min_values
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        """🟢 修改：同时优化 UNet 和 RFDPIC 的参数"""
        # 可以为 RFDPIC 设置不同的学习率参数组，这里默认使用相同配置
        all_params = list(self.model.parameters()) + list(self.rfdpic_model.parameters())
        
        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.hparams.optimizer_config['lr'],
            weight_decay=self.hparams.optimizer_config['weight_decay']
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.scheduler_config['num_warmup_steps'],
            num_training_steps=self.hparams.training_config['n_steps']
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def training_step(self, batch, batch_idx):
        c_past, x_future = batch 
        
        # 🟢 修改：移除 torch.no_grad() 并移除 .detach()，保持梯度回传给 RFDPIC
        self.rfdpic_model.train() # 确保处于训练模式
        
        c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
        c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
        c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled)

        # 传入 (c_start, c_cond)
        loss, mse_val = self.meanflow.loss(self.model, x_future, c=(c_rfdpic, c_past))
        
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/mse_loss', mse_val, on_step=True, on_epoch=False)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        save_freq = self.hparams.logging_config['save_step_frequency']
        if self.trainer.is_global_zero and self.global_step > 0 and self.global_step % save_freq == 0:
            self.model.eval()
            self.rfdpic_model.eval() # 🟢 采样时切换到 eval
            
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
                
                c_tuple_val = (c_rfdpic_val, c_past_val)
                z = self.meanflow.sample_prediction(
                    self.model, 
                    c_tuple_val, 
                    sample_steps=10,
                    device=self.device
                )

            c_past_sample = c_past_val[0].cpu()
            z_sample = z[0].cpu()
            x_future_sample = x_future_val[0].cpu()
            c_rfdpic_sample = c_rfdpic_val[0].cpu()
            del c_past_val, x_future_val, z, val_batch, c_rfdpic_val, c_tuple_val
            torch.cuda.empty_cache()
            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [z_sample[t] for t in range(z_sample.shape[0])]
            target_list = [x_future_sample[t] for t in range(x_future_sample.shape[0])]
            pred_rfdpic_list = [c_rfdpic_sample[t] for t in range(c_rfdpic_sample.shape[0])]
            save_dir = os.path.join(self.trainer.logger.save_dir, "images", f"step_{self.global_step}")
            vis_himawari8_seq_btchw(save_dir=save_dir, context_seq=context_list, pred_seq=pred_list, target_seq=target_list)
            vis_himawari8_seq_btchw(save_dir=os.path.join(self.trainer.logger.save_dir, "images", f"step_{self.global_step}_rfdpic_cond"), context_seq=context_list, pred_seq=pred_rfdpic_list, target_seq=target_list)
            
            self.model.train()
            self.rfdpic_model.train() # 🟢 恢复训练模式

    def on_test_epoch_start(self):
        if self.trainer.is_global_zero:
            self.metrics_save_dir = os.path.join(self.trainer.logger.save_dir, "metrics")
            os.makedirs(self.metrics_save_dir, exist_ok=True)
            self.example_save_dir = os.path.join(self.trainer.logger.save_dir, "examples_test")
            os.makedirs(self.example_save_dir, exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.rfdpic_model.eval()
        c_past, x_future = batch
        
        c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
        c_rfdpic_norm, displacement_fields, residual_fields, pars, dyn_feats = self.rfdpic_model(c_past_norm)
        c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled)
        rfdpic_denorm = self.denormalize(data=c_rfdpic, mode='test')

        c_tuple = (c_rfdpic, c_past)
        preds_norm = self.meanflow.sample_prediction(
            self.model, 
            c_tuple,
            sample_steps=self.sample_steps, 
            device=self.device
        )
        targets_norm = x_future
        
        preds_denorm = self.denormalize(preds_norm, mode='test')
        targets_denorm = self.denormalize(targets_norm, mode='test')
        
        self.test_d_mse(preds_denorm, targets_denorm)
        self.test_d_mae(preds_denorm, targets_denorm)
        
        preds_bchw = rearrange(preds_norm, 'b t c h w -> (b t) c h w')
        targets_bchw = rearrange(targets_norm, 'b t c h w -> (b t) c h w')
        self.test_ssim(preds_bchw, targets_bchw)
        self.test_psnr(preds_bchw, targets_bchw)
        
        real_video_T12 = torch.cat([c_past, targets_norm], dim=1)
        fake_video_T12 = torch.cat([c_past, preds_norm], dim=1)
        
        for c in range(self.num_channels):
            real_ch_video = real_video_T12[:, :, c:c+1, :, :]
            fake_ch_video = fake_video_T12[:, :, c:c+1, :, :]
            self.test_fvd_list[c].update(real_ch_video, real=True)
            self.test_fvd_list[c].update(fake_ch_video, real=False)

        self.test_psd_gt.update(targets_norm)
        self.test_psd_pred.update(preds_norm)
        self.test_psd_rfdpic.update(c_rfdpic) 

        for c in range(self.num_channels):
            for t in range(self.num_timesteps):
                self.test_d_mae_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())
                self.test_d_mse_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())
                self.test_d_rmse_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())

        micro_batch_size = c_past.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if data_idx in self.test_example_data_idx_list:
            c_past_sample = c_past[0].cpu()
            preds_sample = preds_norm[0].cpu() 
            preds_rfdpic_denorm = rfdpic_denorm[0].cpu()
            targets_sample = targets_norm[0].cpu() 
            
            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [preds_sample[t] for t in range(preds_sample.shape[0])]
            target_list = [targets_sample[t] for t in range(targets_sample.shape[0])]
            
            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "prediction")
            vis_himawari8_seq_btchw(save_dir=save_dir, context_seq=context_list, pred_seq=pred_list, target_seq=target_list)

            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "dsitribution")
            save_distribution_plot_final(tensor=c_past_sample, save_path=os.path.join(save_dir, "context_distribution.pdf"))
            save_distribution_plot_final(tensor=preds_sample, save_path=os.path.join(save_dir, "flowmatching_pred_distribution.pdf"))
            save_distribution_plot_final(tensor=preds_rfdpic_denorm, save_path=os.path.join(save_dir, "determinstic_distribution.pdf"))
            save_distribution_plot_final(tensor=targets_sample, save_path=os.path.join(save_dir, "target_distribution.pdf"))

            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "feature_dyn")
            vis_dynamics_feats(save_dir=save_dir, dyn_feats=dyn_feats, batch_idx=data_idx%micro_batch_size)
            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "displacement_fields")
            visualize_channel_fields(displacement_fields=displacement_fields[data_idx%micro_batch_size], field_type="displacement", save_dir=save_dir, size=4)
            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "residual_fields")
            visualize_channel_residuals(residual_fields=residual_fields[data_idx%micro_batch_size], save_dir=save_dir)
            
            try:
                rfdpic_sample_np = c_rfdpic[0].cpu().numpy()
                preds_sample_np = preds_sample.numpy()
                targets_sample_np = targets_sample.numpy()
                plot_sample_psd(pred_4d=preds_sample_np, rfdpic_4d=rfdpic_sample_np, gt_4d=targets_sample_np, save_dir=save_dir)
            except Exception as e:
                print(f"Failed to generate sample PSD plot for sample {data_idx}: {e}")

    def on_test_epoch_end(self):
        test_d_mse = self.test_d_mse.compute()
        test_d_mae = self.test_d_mae.compute()
        test_ssim = self.test_ssim.compute()
        test_psnr = self.test_psnr.compute()
        
        self.log('test/d_mse_epoch', test_d_mse, prog_bar=True, sync_dist=True)
        self.log('test/d_mae_epoch', test_d_mae, prog_bar=True, sync_dist=True)
        self.log('test/ssim_epoch', test_ssim, prog_bar=True, sync_dist=True)
        self.log('test/psnr_epoch', test_psnr, prog_bar=True, sync_dist=True)
        
        self.test_d_mse.reset()
        self.test_d_mae.reset()
        self.test_ssim.reset()
        self.test_psnr.reset()
        
        all_fvd_values = [] 
        for c in range(self.num_channels):
            try:
                test_fvd_c = self.test_fvd_list[c].compute()
                all_fvd_values.append(test_fvd_c)
            except Exception as e:
                print(f"Could not compute FVD for channel {c}: {e}")
            self.test_fvd_list[c].reset()
        
        if all_fvd_values:
            mean_fvd = torch.stack(all_fvd_values).mean() 
            self.log('test/fvd_epoch_mean', mean_fvd, prog_bar=True, sync_dist=True)

        k_axis_gt, psd_gt = self.test_psd_gt.compute()
        _, psd_pred = self.test_psd_pred.compute()
        _, psd_rfdpic = self.test_psd_rfdpic.compute()
        
        k_axis_np, psd_gt_np, psd_pred_np, psd_rfdpic_np = k_axis_gt.cpu().numpy(), psd_gt.cpu().numpy(), psd_pred.cpu().numpy(), psd_rfdpic.cpu().numpy()
        self.test_psd_gt.reset(); self.test_psd_pred.reset(); self.test_psd_rfdpic.reset()

        if self.trainer.is_global_zero:
            try:
                splits_list = [0.50, 0.75, 0.90, 0.95]
                psd_metrics_pred = calculate_psd_error_metrics(psd_gt_np, psd_pred_np, k_axis_np, splits_list)
                for key, value in psd_metrics_pred.items():
                    self.log(f'test/psd_pred_{key}', value, sync_dist=False)
                
                psd_metrics_rfdpic = calculate_psd_error_metrics(psd_gt_np, psd_rfdpic_np, k_axis_np, splits_list)
                for key, value in psd_metrics_rfdpic.items():
                    self.log(f'test/psd_rfdpic_{key}', value, sync_dist=False)
            except Exception as e:
                print(f"Failed to compute PSD metrics: {e}")

            d_mse_metric = np.zeros((self.num_channels, self.num_timesteps))
            d_mae_metric = np.zeros((self.num_channels, self.num_timesteps))
            d_rmse_metric = np.zeros((self.num_channels, self.num_timesteps))
            
            for c in range(self.num_channels):
                for t in range(self.num_timesteps):
                    d_mse_metric[c][t] = self.test_d_mse_metric[c][t].compute().cpu().item()
                    self.test_d_mse_metric[c][t].reset()
                    d_mae_metric[c][t] = self.test_d_mae_metric[c][t].compute().cpu().item()
                    self.test_d_mae_metric[c][t].reset()
                    d_rmse_metric[c][t] = self.test_d_rmse_metric[c][t].compute().cpu().item()
                    self.test_d_rmse_metric[c][t].reset()
            
            np.savetxt(f"{self.metrics_save_dir}/mse.csv", d_mse_metric, delimiter=",")
            np.savetxt(f"{self.metrics_save_dir}/mae.csv", d_mae_metric, delimiter=",")
            np.savetxt(f"{self.metrics_save_dir}/rmse.csv", d_rmse_metric, delimiter=",")
            plot_metrics_curve(self.metrics_save_dir, "mse", d_mse_metric)
            plot_metrics_curve(self.metrics_save_dir, "mae", d_mae_metric)
            plot_metrics_curve(self.metrics_save_dir, "rmse", d_rmse_metric)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Lightning Video Prediction Training")
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--rfdpic_config', type=str, required=True)
    parser.add_argument('--rfdpic_ckpt', type=str, required=True)
    parser.add_argument('--sample_steps', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    pl.seed_everything(42, workers=True)

    dataset_cfg = config['data']
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
        batch_size=args.batch_size if args.batch_size else dataset_cfg["batch_size"],
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
    )

    model = VideoLightningModule(
        model_config=config['model'],
        data_config=config['data'],
        meanflow_config=config['meanflow'],
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler'],
        training_config=config['training'],
        logging_config=config['logging'],
        eval_config=config['eval'],
        rfdpic_config_path=args.rfdpic_config,
        rfdpic_ckpt_path=args.rfdpic_ckpt,
        sample_steps=args.sample_steps,
    )

    if args.use_wandb:
        logger_instance = WandbLogger(project=config['logging']['project_name'], save_dir=args.log_dir, name=os.path.basename(args.log_dir))
    else:
        logger_instance = TensorBoardLogger(save_dir=args.log_dir, name="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="step_{step:06d}-loss_{train/loss:.4f}",
        every_n_train_steps=config['logging']['save_step_frequency'],
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    
    trainer = pl.Trainer(
        logger=logger_instance,
        callbacks=[checkpoint_callback],
        max_steps=config['training']['n_steps'],
        devices=args.gpus,
        accelerator="gpu",
        precision=32,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=config['training']['gradient_clip_val']
    )

    if args.mode == 'train':
        trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
    elif args.mode == 'test':
        if args.ckpt_path is None:
            raise ValueError("Must provide --ckpt_path for testing.")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        trainer.test(model, datamodule)

if __name__ == '__main__':
    main()