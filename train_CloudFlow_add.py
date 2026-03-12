# train_lightning.py (DiffCast Residual Joint Training Version)

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
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import json
from evaluation.fvd.torchmetrics_wrap import FrechetVideoDistance
from evaluation.psd.psd_metric import PSDAverageMetric
from evaluation.psd.psd_tools import plot_sample_psd, calculate_psd_error_metrics

from models.MotionPredictor.RFDPIC_Dual_Rotation_dyn import RFDPIC_Dual_Rotation_Dyn
from utils.transform import data_transform, inverse_data_transform
from dataset_btchw import Himawari8LightningDataModule
from models.ViT import JiT
from videoMeanflow_CloudFlow import MeanFlow 
from visualize import vis_himawari8_seq_btchw, plot_metrics_curve


class VideoLightningModule(pl.LightningModule):

    def __init__(self, model_config, data_config, meanflow_config, optimizer_config, scheduler_config, training_config, logging_config, eval_config,
                 rfdpic_config: dict, 
                 sample_steps: int,
                 loss_alpha: float = 0.5):
        
        super().__init__()
        self.save_hyperparameters()
        self.loss_alpha = loss_alpha
        
        # 1. 初始化 RFDPIC
        self.rfdpic_rescaled = rfdpic_config.get('rescaled', True)
        self.rfdpic_model = self.init_rfdpic_from_config(rfdpic_config)

        # 2. 实例化 MeanFlow
        self.meanflow = MeanFlow(
            channels=model_config['out_channels_c'],
            time_dim=model_config['input_size'][0],
            height_dim=model_config['input_size'][1],
            width_dim=model_config['input_size'][2],
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
        self.model = JiT(
            input_size=tuple(model_config['input_size']),
            in_channels_c=model_config['in_channels_c'],
            out_channels_c=model_config['out_channels_c'], 
            time_emb_dim=model_config['time_emb_dim'],
            patch_size=model_config['patch_size'],
            hidden_size=model_config['hidden_size'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            bottleneck_dim=model_config['bottleneck_dim'],
        )
        
        self.val_loader_iter = None

        # 4. 初始化所有测试指标
        self.num_channels = self.hparams.model_config['out_channels_c']
        self.num_timesteps = 6 
        
        # 指标 (反归一化)
        self.test_d_mse = torchmetrics.MeanSquaredError()
        self.test_d_mae = torchmetrics.MeanAbsoluteError()

        # 指标 (归一化, data_range=1.0)
        self.test_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        
        # 逐通道、逐时间步指标
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
        
        # FVD 指标
        self.test_fvd_list = nn.ModuleList([
            FrechetVideoDistance(feature=400, normalize=False) 
            for _ in range(self.num_channels)
        ])

        # PSD 指标
        H_W = self.hparams.model_config['input_size'][1]
        self.test_psd_gt = PSDAverageMetric(H=H_W, W=H_W)
        self.test_psd_pred = PSDAverageMetric(H=H_W, W=H_W)
        self.test_psd_rfdpic = PSDAverageMetric(H=H_W, W=H_W)
        
        self.train_example_data_idx_list = eval_config['train_example_data_idx_list']
        self.val_example_data_idx_list = eval_config['val_example_data_idx_list']
        self.test_example_data_idx_list = eval_config['test_example_data_idx_list']

        # 加载统计数据
        with open(data_config['train_json_path'], 'r') as f:
            train_global_stats = json.load(f)
        train_global_max_values = torch.tensor(train_global_stats['Global Max'], 
                                                   dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        train_global_min_values = torch.tensor(train_global_stats['Global Min'], 
                                                   dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        train_global_range = train_global_max_values - train_global_min_values
        self.register_buffer('train_global_max_values', train_global_max_values)
        self.register_buffer('train_global_min_values', train_global_min_values)
        self.register_buffer('train_global_range', train_global_range)
        
        with open(data_config['test_json_path'], 'r') as f:
            test_global_stats = json.load(f)
        test_global_max_values = torch.tensor(test_global_stats['Global Max'], 
                                                   dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        test_global_min_values = torch.tensor(test_global_stats['Global Min'], 
                                                   dtype=torch.float32).reshape(1, 1, 8, 1, 1)
        test_global_range = test_global_max_values - test_global_min_values
        self.register_buffer('test_global_max_values', test_global_max_values)
        self.register_buffer('test_global_min_values', test_global_min_values)
        self.register_buffer('test_global_range', test_global_range)
        
        self.plot_dict = {
            'train': {'max': train_global_stats['Global Max'], 'min': train_global_stats['Global Min']},
            'val': {'max': train_global_stats['Global Max'], 'min': train_global_stats['Global Min']},
            'test': {'max': test_global_stats['Global Max'], 'min': test_global_stats['Global Min']}
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=False)
    
    def init_rfdpic_from_config(self, rfdpic_cfg: dict):
        print(f"Initializing RFDPIC model from merged config (Random Init)...")
        sub_config_path = rfdpic_cfg.get('rf_dp_config_file', None)
        model = RFDPIC_Dual_Rotation_Dyn(
            dp_name=rfdpic_cfg['dp_name'],
            rf_name=rfdpic_cfg['rf_name'],
            rf_dp_config_file=sub_config_path,
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
        print(f"{'='*10} RFDPIC model initialized from scratch (Random Weights). {'='*10}")
        return model

    def denormalize(self, data: torch.Tensor, mode: str=''):
        if mode == 'train' or mode == 'val':
            return data * self.train_global_range + self.train_global_min_values
        elif mode == 'test':
            return data * self.test_global_range + self.test_global_min_values
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        combined_params = list(self.model.parameters()) + list(self.rfdpic_model.parameters())
        optimizer = torch.optim.AdamW(
            combined_params, 
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
        
        c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
        x_future_norm = data_transform(x_future, rescaled=self.rfdpic_rescaled)

        # 1. RFDPIC 前向 (Base Predictor)
        c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
        
        # 2. Base Loss (Deterministic)
        loss_base = F.l1_loss(c_rfdpic_norm, x_future_norm)

        # 3. 计算残差 (Residual) [DiffCast Eq.7: r = y - mu]
        # 这是关键修改：MeanFlow 现在学习去预测这个残差，而不是完整的未来帧
        residual_norm = x_future_norm - c_rfdpic_norm

        # 4. MeanFlow Loss (Condition: RFDPIC Output, Target: Residual)
        c_tuple = (c_rfdpic_norm, c_past_norm) 
        
        # [Residual Modification] 这里的 target 传入的是 residual_norm
        loss_flow, mse_val_flow = self.meanflow.loss(self.model, residual_norm, c=c_tuple)
        
        # 5. Total Loss
        total_loss = self.loss_alpha * loss_flow + (1 - self.loss_alpha) * loss_base
        
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss_flow', loss_flow, on_step=True, on_epoch=False)
        self.log('train/loss_base', loss_base, on_step=True, on_epoch=False)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        save_freq = self.hparams.logging_config['save_step_frequency']
        
        if self.trainer.is_global_zero and self.global_step > 0 and self.global_step % save_freq == 0:
            self.model.eval()
            self.rfdpic_model.eval()
            
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
                # RFDPIC 输出反归一化用于可视化（作为 Base）
                c_rfdpic_val = inverse_data_transform(c_rfdpic_norm_val, rescaled=self.rfdpic_rescaled)
                
                # 条件保持归一化空间
                c_tuple_val = (c_rfdpic_norm_val, c_past_norm_val)

                # MeanFlow 采样残差
                z_residual = self.meanflow.sample_prediction(
                    self.model, 
                    c_tuple_val, 
                    sample_steps=self.sample_steps,
                    device=self.device
                )
                
                # [Residual Modification] 最终预测 = Base + Residual (DiffCast Eq.13)
                # z_residual 是归一化的残差，c_rfdpic_norm_val 是归一化的 Base
                final_pred_norm = c_rfdpic_norm_val + z_residual
                
                # 反归一化最终结果用于可视化
                final_pred_val = inverse_data_transform(final_pred_norm, rescaled=self.rfdpic_rescaled)

            c_past_sample = c_past_val[0].cpu()
            z_sample = final_pred_val[0].cpu() # 已经是反归一化[0,1]的了
            x_future_sample = x_future_val[0].cpu()
            c_rfdpic_sample = c_rfdpic_val[0].cpu()
            
            del c_past_val, x_future_val, z_residual, final_pred_norm, val_batch, c_rfdpic_val, c_tuple_val
            torch.cuda.empty_cache()
            
            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [z_sample[t] for t in range(z_sample.shape[0])]
            target_list = [x_future_sample[t] for t in range(x_future_sample.shape[0])]
            pred_rfdpic_list = [c_rfdpic_sample[t] for t in range(c_rfdpic_sample.shape[0])]
            
            save_dir = os.path.join(self.trainer.logger.save_dir, "images", f"step_{self.global_step}")
            vis_himawari8_seq_btchw(save_dir=save_dir, context_seq=context_list, pred_seq=pred_list, target_seq=target_list)
            # 可视化 RFDPIC 基础预测
            vis_himawari8_seq_btchw(save_dir=os.path.join(self.trainer.logger.save_dir, "images", f"step_{self.global_step}_base_pred"), context_seq=context_list, pred_seq=pred_rfdpic_list, target_seq=target_list)
            
            self.model.train()
            self.rfdpic_model.train()

    def on_test_epoch_start(self):
        if self.trainer.is_global_zero:
            self.metrics_save_dir = os.path.join(self.trainer.logger.save_dir, "metrics")
            os.makedirs(self.metrics_save_dir, exist_ok=True)
            self.example_save_dir = os.path.join(self.trainer.logger.save_dir, "examples_test")
            os.makedirs(self.example_save_dir, exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        c_past, x_future = batch
        
        # 1. RFDPIC
        c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
        c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
        
        # 用于PSD/可视化对比的纯 RFDPIC 结果
        c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled)
        c_rfdpic = c_rfdpic.detach()

        # 2. MeanFlow (预测 Residual)
        c_tuple = (c_rfdpic_norm, c_past_norm)
        
        preds_residual_norm = self.meanflow.sample_prediction(
            self.model, 
            c_tuple,
            sample_steps=self.sample_steps, 
            device=self.device
        )
        
        # [Residual Modification] 3. 合并: Final = Base + Residual
        # 在归一化空间进行加和
        final_preds_norm = c_rfdpic_norm + preds_residual_norm # [-1,1]
        final_preds = inverse_data_transform(final_preds_norm, rescaled=self.rfdpic_rescaled) # 反归一化用于指标计算和可视化
        
        targets = x_future # x_future 已经在 dataloader 里处理过(虽然这里没显式调 transform，但在 metrics 需要一致性)
        
        # 4. 反归一化用于 MSE/MAE 计算
        preds_denorm = self.denormalize(final_preds, mode='test')
        targets_denorm = self.denormalize(targets, mode='test')
        
        # 5. 指标计算
        self.test_d_mse(preds_denorm, targets_denorm)
        self.test_d_mae(preds_denorm, targets_denorm)
        
        # SSIM/PSNR 需要归一化到 [0, 1] (如果 dataset 是 -1~1, 这里需要调整)
        # 假设 final_preds 是 -1~1, 我们在 transform 里通常也是 -1~1
        # 简单起见，这里传入 normalized data, torchmetrics 会自己处理
        preds_bchw = rearrange(final_preds, 'b t c h w -> (b t) c h w')
        targets_bchw = rearrange(targets, 'b t c h w -> (b t) c h w')
        
        self.test_ssim(preds_bchw, targets_bchw)
        self.test_psnr(preds_bchw, targets_bchw)
        
        real_video_T12 = torch.cat([c_past, targets], dim=1)
        fake_video_T12 = torch.cat([c_past, final_preds], dim=1)
        
        for c in range(self.num_channels):
            real_ch_video = real_video_T12[:, :, c:c+1, :, :]
            fake_ch_video = fake_video_T12[:, :, c:c+1, :, :]
            self.test_fvd_list[c].update(real_ch_video, real=True)
            self.test_fvd_list[c].update(fake_ch_video, real=False)

        self.test_psd_gt.update(targets)
        self.test_psd_pred.update(final_preds)
        self.test_psd_rfdpic.update(c_rfdpic) 

        for c in range(self.num_channels):
            for t in range(self.num_timesteps):
                self.test_d_mae_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())
                self.test_d_mse_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())
                self.test_d_rmse_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())

        # 6. 可视化
        micro_batch_size = c_past.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if data_idx in self.test_example_data_idx_list:
            c_past_sample = c_past[0].cpu()
            # 注意：可视化使用归一化的数据，vis 函数内部可能会处理
            preds_sample = final_preds[0].cpu() 
            targets_sample = targets[0].cpu() 
            
            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [preds_sample[t] for t in range(preds_sample.shape[0])]
            target_list = [targets_sample[t] for t in range(targets_sample.shape[0])]
            
            save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}")
            vis_himawari8_seq_btchw(
                save_dir=save_dir, 
                context_seq=context_list, 
                pred_seq=pred_list, 
                target_seq=target_list
            )
            
            try:
                # PSD 绘图使用 normalized numpy array
                rfdpic_sample_np = c_rfdpic[0].cpu().numpy() 
                preds_sample_np = final_preds[0].cpu().numpy()
                targets_sample_np = targets[0].cpu().numpy()
                plot_sample_psd(
                    pred_4d=preds_sample_np,
                    rfdpic_4d=rfdpic_sample_np,
                    gt_4d=targets_sample_np,
                    save_dir=save_dir
                )
            except Exception as e:
                print(f"Failed to generate sample PSD plot for sample {data_idx}: {e}")

    def on_test_epoch_end(self):
        test_d_mse = self.test_d_mse.compute()
        test_d_mae = self.test_d_mae.compute()
        test_ssim = self.test_ssim.compute()
        test_psnr = self.test_psnr.compute()
        
        self.log('test/d_mse_epoch', test_d_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/d_mae_epoch', test_d_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/ssim_epoch', test_ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/psnr_epoch', test_psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

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
            all_fvd_tensor = torch.stack(all_fvd_values) 
            mean_fvd = all_fvd_tensor.mean() 
            self.log('test/fvd_epoch_mean', mean_fvd, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else:
            print("\n--- Warning: No FVD values were successfully computed. ---\n")

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
            try:
                splits_list = [0.50, 0.75, 0.90, 0.95]
                psd_metrics_pred = calculate_psd_error_metrics(psd_gt_np, psd_pred_np, k_axis_np, splits_list)
                print("\n--- PSD Metrics (MeanFlow Pred vs. GT) ---")
                for key, value in psd_metrics_pred.items():
                    self.log(f'test/psd_pred_{key}', value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
                    print(f"  {key}: {value:.4f}")
                
                psd_metrics_rfdpic = calculate_psd_error_metrics(psd_gt_np, psd_rfdpic_np, k_axis_np, splits_list)
                print("\n--- PSD Metrics (RFDPIC vs. GT) ---")
                for key, value in psd_metrics_rfdpic.items():
                    self.log(f'test/psd_rfdpic_{key}', value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
                    print(f"  {key}: {value:.4f}")

            except Exception as e:
                print(f"Failed to compute PSD metrics: {e}")

        if self.trainer.is_global_zero:
            d_mse_metric = np.zeros((self.num_channels, self.num_timesteps))
            d_mae_metric = np.zeros((self.num_channels, self.num_timesteps))
            d_rmse_metric = np.zeros((self.num_channels, self.num_timesteps))
            
            for c in range(self.num_channels):
                for t in range(self.num_timesteps):
                    mse = self.test_d_mse_metric[c][t].compute()
                    d_mse_metric[c][t] = mse.cpu().item()
                    self.test_d_mse_metric[c][t].reset()
                    
                    mae = self.test_d_mae_metric[c][t].compute()
                    d_mae_metric[c][t] = mae.cpu().item()
                    self.test_d_mae_metric[c][t].reset()
                    
                    rmse = self.test_d_rmse_metric[c][t].compute()
                    d_rmse_metric[c][t] = rmse.cpu().item()
                    self.test_d_rmse_metric[c][t].reset()
            
            np.savetxt(f"{self.metrics_save_dir}/mse.csv", d_mse_metric, delimiter=",")
            np.savetxt(f"{self.metrics_save_dir}/mae.csv", d_mae_metric, delimiter=",")
            np.savetxt(f"{self.metrics_save_dir}/rmse.csv", d_rmse_metric, delimiter=",")
            
            plot_metrics_curve(self.metrics_save_dir, "mse", d_mse_metric)
            plot_metrics_curve(self.metrics_save_dir, "mae", d_mae_metric)
            plot_metrics_curve(self.metrics_save_dir, "rmse", d_rmse_metric)
            print(f"Test metrics saved and plotted in {self.metrics_save_dir}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Lightning Video Prediction Training")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to the MeanFlow config.yaml file")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory to save logs and checkpoints")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size (overrides config if set)")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="Training or testing mode")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to checkpoint for resuming MeanFlow training or testing")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--sample_steps', type=int, default=10, help="Number of sampling steps for MeanFlow (overrides config if set)")
    parser.add_argument('--alpha', type=float, default=0.5, help="Weighting factor for total loss (overrides config if set)")
    parser.add_argument('--use_wandb', action='store_true', help="Use WandbLogger (default: False)")
    
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
        batch_size=args.batch_size,
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
        rfdpic_config=config['rfdpic_model'], 
        sample_steps=args.sample_steps,
        loss_alpha=args.alpha 
    )

    if args.use_wandb:
        print("--- Using Weights & Biases Logger ---")
        logger_instance = WandbLogger(
            project=config['logging']['project_name'], 
            save_dir=args.log_dir, 
            name=os.path.basename(args.log_dir)
        )
        logger_instance.watch(model, log="all", log_freq=500)
    else:
        print(f"--- WandbLogger is disabled, using TensorBoardLogger at {args.log_dir} ---")
        logger_instance = TensorBoardLogger(
            save_dir=args.log_dir,
            name="" 
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="step_{step:06d}-loss_{train/total_loss:.4f}",
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
        print(f"--- Starting Training ---")
        print(f"Log dir: {args.log_dir}")
        ckpt_to_resume = None
        if args.ckpt_path:
            print(f"--- 正在从 MeanFlow checkpoint 延续训练: {args.ckpt_path} ---")
            ckpt_to_resume = args.ckpt_path
        else:
            print(f"--- 从头开始训练 MeanFlow + RFDPIC (Joint Training, Residual Mode) ---")
        
        trainer.fit(model, datamodule, ckpt_path=ckpt_to_resume)
        
    elif args.mode == 'test':
        if args.ckpt_path is None:
            raise ValueError("Must provide --ckpt_path for testing.")
            
        print(f"--- Starting Testing ---")
        print(f"--- Manually loading checkpoint with strict=False: {args.ckpt_path} ---")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        trainer.test(model, datamodule)

if __name__ == '__main__':
    main()