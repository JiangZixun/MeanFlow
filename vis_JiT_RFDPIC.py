# train_lightning.py (修改版)

# ... (所有 import 保持不变, 包括 RFDPIC 的) ...
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
from models.ViT import JiT
from videoMeanflow_RFDPIC_JiU import MeanFlow 
from visualize import vis_himawari8_seq_btchw, plot_metrics_curve, vis_dynamics_feats, visualize_channel_fields, visualize_channel_residuals, plot_himawari8_seq_btchw, save_distribution_plot_final


class VideoLightningModule(pl.LightningModule):
    # ... (load_frozen_rfdpic 和 configure_optimizers 保持不变) ...

    def __init__(self, model_config, data_config, meanflow_config, optimizer_config, scheduler_config, training_config, logging_config, eval_config,
                 rfdpic_config_path: str,
                 rfdpic_ckpt_path: str,
                 sample_steps: int):
        
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 加载 RFDPIC (不变)
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
        self.rfdpic_model = self.load_frozen_rfdpic(
            rfdpic_config_path=rfdpic_config_path,
            rfdpic_ckpt_path=rfdpic_ckpt_path
        )

        # 2. 实例化 MeanFlow (使用新修改的 videoMeanflow.py)
        #    __init__ 本身不需要改动
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
        # 🔴 这里的 model_config['in_channels_c'] 必须是 6 (由 main() 传入)
        print(f"Initializing UNet with condition input_t (in_channels_c): {model_config['in_channels_c']}")
        self.model = JiT(
            input_size=tuple(model_config['input_size']),
            in_channels_c=model_config['in_channels_c'],  # 条件输入通道数 (6)
            out_channels_c=model_config['out_channels_c'], # 预测输出通道数 (8)
            time_emb_dim=model_config['time_emb_dim'],
            patch_size=model_config['patch_size'],
            hidden_size=model_config['hidden_size'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            bottleneck_dim=model_config['bottleneck_dim'],
        )
        
        self.val_loader_iter = None

        # 🔴 4. 新增：初始化所有测试指标
        # 从配置中获取通道数和时间步数
        self.num_channels = self.hparams.model_config['out_channels_c']
        # (B, 6, C, H, W) -> 6 步
        self.num_timesteps = 6 
        
        # 指标 (反归一化)
        self.test_d_mse = torchmetrics.MeanSquaredError()
        self.test_d_mae = torchmetrics.MeanAbsoluteError()

        # 指标 (归一化, data_range=1.0)
        self.test_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0)
        
        # 逐通道、逐时间步指标 (反归一化)
        # 使用 ModuleList 包装，使其能被 .to(device) 自动移动
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

        # 🔴 新增：(回答您的指标要求 1)
        # 像 FVD 一样，我们定义三个流式指标，它们不会爆内存
        H_W = self.hparams.model_config['input_size'][1] # 应该是 256
        self.test_psd_gt = PSDAverageMetric(H=H_W, W=H_W)
        self.test_psd_pred = PSDAverageMetric(H=H_W, W=H_W)
        self.test_psd_rfdpic = PSDAverageMetric(H=H_W, W=H_W)
        
        # 可视化index
        self.train_example_data_idx_list = eval_config['train_example_data_idx_list']
        self.val_example_data_idx_list = eval_config['val_example_data_idx_list']
        self.test_example_data_idx_list = eval_config['test_example_data_idx_list']

        # 🔴 4. 新增：从 train_UNet_RFDPIC.py 照搬统计数据加载逻辑
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

    # 🔴 新增：覆盖此方法以允许从旧 checkpoint 恢复
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        覆盖 Pytorch Lightning 的默认行为。
        我们强制使用 strict=False，这样在加载一个旧的、
        没有 FVD 或 global_stats 缓冲区的 checkpoint 时，
        它不会因为 "Missing key(s)" 错误而崩溃。
        
        这使得 `trainer.fit(ckpt_path=...)` 能够成功加载模型权重，
        同时也能正确恢复 optimizer、scheduler 和 global_step。
        """
        # 强制使用 strict=False 来忽略缺失的键 (FVD, global_stats 等)
        super().load_state_dict(state_dict, strict=False)
    
    def load_frozen_rfdpic(self, rfdpic_config_path, rfdpic_ckpt_path):
        # ... (此辅助函数保持不变) ...
        print(f"Loading pretrained RFDPIC model from {rfdpic_ckpt_path}")
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
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"{'='*10} RFDPIC model loaded and frozen. {'='*10}")
        return model

    # 🔴 新增：反归一化辅助函数
    def denormalize(self, data: torch.Tensor, mode: str=''):
        if mode == 'train' or mode == 'val':
            # (注意：这里我们访问在 __init__ 中注册的 'train' buffer)
            return data * self.train_global_range + self.train_global_min_values
        elif mode == 'test':
            # (注意：这里我们访问在 __init__ 中注册的 'test' buffer)
            return data * self.test_global_range + self.test_global_min_values
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        # ... (此函数保持不变) ...
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
        c_past, x_future = batch # (B, 6, C, H, W), (B, 6, C, H, W)
        
        with torch.no_grad():
            c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
            c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
            c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled)
            c_rfdpic = c_rfdpic.detach() # (B, 6, C, H, W)

        # --- 🔴 1. 修改：将 c 作为元组 (c_start, c_cond) 传递 ---
        # c_start = c_rfpic
        # c_cond = c_past
        loss, mse_val = self.meanflow.loss(self.model, x_future, c=(c_rfdpic, c_past))
        
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/mse_loss', mse_val, on_step=True, on_epoch=False)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        save_freq = self.hparams.logging_config['save_step_frequency']
        
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
                # --- 🔴 2. 修改：为采样准备 (c_start, c_cond) ---
                c_past_norm_val = data_transform(c_past_val, rescaled=self.rfdpic_rescaled)
                c_rfdpic_norm_val, _, _, _, _ = self.rfdpic_model(c_past_norm_val)
                c_rfdpic_val = inverse_data_transform(c_rfdpic_norm_val, rescaled=self.rfdpic_rescaled)
                
                # c_start = c_rfdpic_val
                # c_cond = c_past_val
                c_tuple_val = (c_rfdpic_val, c_past_val)

                z = self.meanflow.sample_prediction(
                    self.model, 
                    c_tuple_val, # <-- 传入元组
                    sample_steps=10,
                    device=self.device
                )

            # ... (可视化和清理不变) ...
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

    # 🔴 新增：在测试开始时创建目录
    def on_test_epoch_start(self):
        if self.trainer.is_global_zero:
            self.metrics_save_dir = os.path.join(self.trainer.logger.save_dir, "metrics")
            os.makedirs(self.metrics_save_dir, exist_ok=True)
            self.example_save_dir = os.path.join(self.trainer.logger.save_dir, "examples_test")
            os.makedirs(self.example_save_dir, exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        c_past, x_future = batch
        micro_batch_size = c_past.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if data_idx in self.test_example_data_idx_list:
            # c_past, x_future = batch
            # --- 1. 获取 RFDPIC 预测 (c_start) ---
            c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
            c_rfdpic_norm, displacement_fields, residual_fields, pars, dyn_feats = self.rfdpic_model(c_past_norm)
            c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled)
            c_rfdpic = c_rfdpic.detach()
            rfdpic_denorm = self.denormalize(data=c_rfdpic, mode='test')

            # --- 2. 获取 MeanFlow 预测 (归一化) ---
            c_tuple = (c_rfdpic, c_past)
            preds_norm = self.meanflow.sample_prediction(
                self.model, 
                c_tuple,
                sample_steps=self.sample_steps, # 注意：采样步数应与验证时一致
                device=self.device
            )
            targets_norm = x_future
            
            # --- 3. 反归一化用于计算 MSE/MAE/RMSE ---
            preds_denorm = self.denormalize(preds_norm, mode='test')
            targets_denorm = self.denormalize(targets_norm, mode='test')
            
            # --- 4. 计算指标 ---
            # 总体指标 (在反归一化数据上)
            self.test_d_mse(preds_denorm, targets_denorm)
            self.test_d_mae(preds_denorm, targets_denorm)
            
            # SSIM/PSNR (在归一化数据上)
            # (B, T, C, H, W) -> (B*T, C, H, W)
            preds_bchw = rearrange(preds_norm, 'b t c h w -> (b t) c h w')
            targets_bchw = rearrange(targets_norm, 'b t c h w -> (b t) c h w')
            self.test_ssim(preds_bchw, targets_bchw)
            self.test_psnr(preds_bchw, targets_bchw)
            
            # 🟢 按照你的提议：拼接 (Cat) 输入和输出来创建 T=12 的视频
            # c_past, targets_norm, 和 preds_norm 此时都在 GPU 上
            # (B, 6, C, H, W) + (B, 6, C, H, W) -> (B, 12, C, H, W)
            real_video_T12 = torch.cat([c_past, targets_norm], dim=1)
            fake_video_T12 = torch.cat([c_past, preds_norm], dim=1)
            
            # 🟢 修改：循环遍历 8 个通道，分别更新 FVD 指标
            for c in range(self.num_channels):
                # real_video_T12 是 (B, 12, 8, H, W)
                # 我们提取第 c 个通道，得到 (B, 12, 1, H, W)
                real_ch_video = real_video_T12[:, :, c:c+1, :, :]
                fake_ch_video = fake_video_T12[:, :, c:c+1, :, :]
                
                # 这里的 FVD 指标会自动处理 C=1 -> C=3
                self.test_fvd_list[c].update(real_ch_video, real=True)
                self.test_fvd_list[c].update(fake_ch_video, real=False)

            # 🔴 更新 PSD 指标 (全局)
            # 这就像 FVD.update()，它只累加总和，不保存数据，不会爆内存
            self.test_psd_gt.update(targets_norm)
            self.test_psd_pred.update(preds_norm)
            self.test_psd_rfdpic.update(c_rfdpic) # (B, 6, C, H, W)

            # 逐通道/逐时间步指标 (在反归一化数据上)
            for c in range(self.num_channels):
                for t in range(self.num_timesteps):
                    self.test_d_mae_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())
                    self.test_d_mse_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())
                    self.test_d_rmse_metric[c][t](preds_denorm[:,t,c].contiguous(), targets_denorm[:,t,c].contiguous())

            # save_npy for extreme case study
            def save_prediction(pred_tensor, save_dir="results", name="sample"):
                """
                pred_tensor: (T, C, H, W) 的 PyTorch Tensor
                """
                assert pred_tensor.ndim == 4, "the shape of pred_tensor should be (T, C, H, W)"

                os.makedirs(save_dir, exist_ok=True)
                
                # 1. 转移到 CPU 并转为 NumPy
                # 如果数据是归一化的 (0,1) 或 (-1,1)，通常保持 float32 即可
                data = pred_tensor.detach().cpu().numpy()
                
                # 2. 构造文件名：模型名_样本索引.npy
                file_path = os.path.join(save_dir, f"{name}.npy")
                
                # 3. 保存
                np.save(file_path, data)
                print(f"Saved prediction to {file_path}")

            extreme_save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}")
            
            # save_prediction(
            #     pred_tensor=self.denormalize(data=c_rfdpic, mode='test')[0],
            #     save_dir=extreme_save_dir,
            #     name="context"
            # )
            # save_prediction(
            #     pred_tensor=rfdpic_denorm[0],
            #     save_dir=extreme_save_dir,
            #     name="rfdpic_pred"
            # )
            save_prediction(
                pred_tensor=preds_denorm[0],
                save_dir=extreme_save_dir,
                name=f"meanflow_pred-{self.sample_steps}_steps"
            )
            save_prediction(
                pred_tensor=targets_denorm[0],
                save_dir=extreme_save_dir,
                name="target"
            )
            
            # save_prediction(
            #     pred_tensor=preds_denorm[0],
            #     save_dir=extreme_save_dir,
            #     name=f"sample_{data_idx}"
            # )
            
            # # --- 5. 可视化 (例如：前 4 个 batch) ---
            # # Visualiozation
            
            # c_past_sample = c_past[0].cpu()
            # preds_sample = preds_norm[0].cpu() # 可视化归一化数据
            # preds_rfdpic = c_rfdpic[0].cpu()
            # targets_sample = targets_norm[0].cpu() 
            
            # context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            # rfdpic_list = [preds_rfdpic[t] for t in range(preds_rfdpic.shape[0])]
            # pred_list = [preds_sample[t] for t in range(preds_sample.shape[0])]
            # target_list = [targets_sample[t] for t in range(targets_sample.shape[0])]
            
            # save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "prediction")

            # plot_himawari8_seq_btchw(
            #     save_dir=save_dir,
            #     context_seq=context_list,
            #     pred_dict={'deterministic': rfdpic_list, 'flowmatching': pred_list},
            #     target_seq=target_list
            # )

            # save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "dsitribution")
            # save_distribution_plot_final(tensor=c_past_sample, save_path=os.path.join(save_dir, "context_distribution.pdf"))
            # save_distribution_plot_final(tensor=preds_sample, save_path=os.path.join(save_dir, "flowmatching_pred_distribution.pdf"))
            # save_distribution_plot_final(tensor=preds_rfdpic, save_path=os.path.join(save_dir, "determinstic_distribution.pdf"))
            # save_distribution_plot_final(tensor=targets_sample, save_path=os.path.join(save_dir, "target_distribution.pdf"))

            # save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "feature_dyn")
            # vis_dynamics_feats(
            #     save_dir=save_dir, 
            #     dyn_feats=dyn_feats, 
            #     batch_idx=data_idx%micro_batch_size
            # )

            # save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "displacement_fields")
            # visualize_channel_fields(
            #     displacement_fields=displacement_fields[data_idx%micro_batch_size], 
            #     field_type="displacement",
            #     save_dir=save_dir,
            #     size=4
            # )

            # save_dir = os.path.join(self.example_save_dir, f"test_sample_{data_idx}", "residual_fields")
            # visualize_channel_residuals(
            #     residual_fields=residual_fields[data_idx%micro_batch_size],
            #     save_dir=save_dir
            # )

            # # 🔴 新增：(回答您的可视化要求 2)
            # # 调用我们简单的“一体化”绘图函数
            # try:
            #     rfdpic_sample_np = c_rfdpic[0].cpu().numpy() # (T, C, H, W)
            #     preds_sample_np = preds_sample.numpy()
            #     targets_sample_np = targets_sample.numpy()
                
            #     # 调用一个函数，完成所有计算和绘图
            #     plot_sample_psd(
            #         pred_4d=preds_sample_np,
            #         rfdpic_4d=rfdpic_sample_np,
            #         gt_4d=targets_sample_np,
            #         save_dir=save_dir
            #     )

            # except Exception as e:
            #     print(f"Failed to generate sample PSD plot for sample {data_idx}: {e}")
        

    def on_test_epoch_end(self):
        # 聚合所有 GPU 上的指标
        test_d_mse = self.test_d_mse.compute()
        test_d_mae = self.test_d_mae.compute()
        test_ssim = self.test_ssim.compute()
        test_psnr = self.test_psnr.compute()
        
        
        # 记录日志
        self.log('test/d_mse_epoch', test_d_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/d_mae_epoch', test_d_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/ssim_epoch', test_ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/psnr_epoch', test_psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        

        # 重置指标
        self.test_d_mse.reset()
        self.test_d_mae.reset()
        self.test_ssim.reset()
        self.test_psnr.reset()
        
        # 🔴 新增：循环计算、记录和重置 8 个通道的 FVD
        all_fvd_values = [] 
        for c in range(self.num_channels):
            try:
                test_fvd_c = self.test_fvd_list[c].compute()
                # 日志名称类似: test/fvd_epoch_ch0, test/fvd_epoch_ch1, ...
                # self.log(f'test/fvd_epoch_ch{c}', test_fvd_c, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
                # 记录成功计算的FVD值
                all_fvd_values.append(test_fvd_c)
            except Exception as e:
                print(f"Could not compute FVD for channel {c}: {e}")
            
            self.test_fvd_list[c].reset()
        
        if all_fvd_values: # 确保列表中有值，防止除以零
            # 将列表转换为张量
            all_fvd_tensor = torch.stack(all_fvd_values) 
            # 计算平均值
            mean_fvd = all_fvd_tensor.mean() 
            # 记录整体的平均FVD
            self.log('test/fvd_epoch_mean', mean_fvd, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else:
            print("\n--- Warning: No FVD values were successfully computed for averaging. ---\n")

        # 🔴 新增：计算、记录和重置 PSD 指标
        # .compute() 会自动 DDP 同步，返回全局平均曲线
        k_axis_gt, psd_gt = self.test_psd_gt.compute()
        _, psd_pred = self.test_psd_pred.compute()
        _, psd_rfdpic = self.test_psd_rfdpic.compute()
        
        # 转换为 NumPy 用于指标计算
        k_axis_np = k_axis_gt.cpu().numpy()
        psd_gt_np = psd_gt.cpu().numpy()
        psd_pred_np = psd_pred.cpu().numpy()
        psd_rfdpic_np = psd_rfdpic.cpu().numpy()

        # 重置 (像 FVD.reset())
        self.test_psd_gt.reset()
        self.test_psd_pred.reset()
        self.test_psd_rfdpic.reset()

        # (仅在 rank 0 上保存和绘图)
        if self.trainer.is_global_zero:
            
            # 🔴 新增：调用 PSD 指标计算 (Part 2)
            try:
                # 按照您的要求设置百分比
                splits_list = [0.50, 0.75, 0.90, 0.95] # 对应 top 50%, 25%, 10%, 5%
                
                # 计算 MeanFlow vs GT
                psd_metrics_pred = calculate_psd_error_metrics(
                    psd_gt_np, psd_pred_np, k_axis_np, splits_list
                )
                print("\n--- PSD Metrics (MeanFlow Pred vs. GT) ---")
                for key, value in psd_metrics_pred.items():
                    self.log(f'test/psd_pred_{key}', value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
                    print(f"  {key}: {value:.4f}")
                
                # 计算 RFDPIC vs GT (作为对比)
                psd_metrics_rfdpic = calculate_psd_error_metrics(
                    psd_gt_np, psd_rfdpic_np, k_axis_np, splits_list
                )
                print("\n--- PSD Metrics (RFDPIC vs. GT) ---")
                for key, value in psd_metrics_rfdpic.items():
                    self.log(f'test/psd_rfdpic_{key}', value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
                    print(f"  {key}: {value:.4f}")

            except Exception as e:
                print(f"Failed to compute PSD metrics: {e}")

        # 处理逐通道指标 (仅在 rank 0 上执行)
        if self.trainer.is_global_zero:
            d_mse_metric = np.zeros((self.num_channels, self.num_timesteps))
            d_mae_metric = np.zeros((self.num_channels, self.num_timesteps))
            d_rmse_metric = np.zeros((self.num_channels, self.num_timesteps))
            
            # 遍历所有通道和时间步
            for c in range(self.num_channels):
                for t in range(self.num_timesteps):
                    # 计算 MSE
                    mse = self.test_d_mse_metric[c][t].compute()
                    d_mse_metric[c][t] = mse.cpu().item()
                    self.test_d_mse_metric[c][t].reset()  # 重置
                    # 计算 MAE
                    mae = self.test_d_mae_metric[c][t].compute()
                    d_mae_metric[c][t] = mae.cpu().item()
                    self.test_d_mae_metric[c][t].reset()  # 重置
                    # 计算 RMSE
                    rmse = self.test_d_rmse_metric[c][t].compute()
                    d_rmse_metric[c][t] = rmse.cpu().item()
                    self.test_d_rmse_metric[c][t].reset()  # 重置
            
            # 保存到 CSV
            np.savetxt(f"{self.metrics_save_dir}/mse.csv", d_mse_metric, delimiter=",")
            np.savetxt(f"{self.metrics_save_dir}/mae.csv", d_mae_metric, delimiter=",")
            np.savetxt(f"{self.metrics_save_dir}/rmse.csv", d_rmse_metric, delimiter=",")
            
            # 绘制曲线图
            plot_metrics_curve(self.metrics_save_dir, "mse", d_mse_metric)
            plot_metrics_curve(self.metrics_save_dir, "mae", d_mae_metric)
            plot_metrics_curve(self.metrics_save_dir, "rmse", d_rmse_metric)
            print(f"Test metrics saved and plotted in {self.metrics_save_dir}")


def main():
    # ... (parser 和 args 不变, 仍然需要 --rfdpic_config 和 --rfdpic_ckpt) ...
    parser = argparse.ArgumentParser(description="PyTorch Lightning Video Prediction Training")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to the MeanFlow config.yaml file")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory to save logs and checkpoints")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size (overrides config if set)")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="Training or testing mode")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to checkpoint for resuming MeanFlow training or testing")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use")
    parser.add_argument('--rfdpic_config', type=str, required=True, help="Path to the RFDPIC main config.yaml")
    parser.add_argument('--rfdpic_ckpt', type=str, required=True, help="Path to the RFDPIC model checkpoint (.pt or .ckpt)")
    parser.add_argument('--sample_steps', type=int, default=10, help="Number of sampling steps for MeanFlow (overrides config if set)")
    parser.add_argument('--use_wandb', action='store_true', help="Use WandbLogger (default: False)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    pl.seed_everything(42, workers=True)

    # 4. 初始化 DataModule (不变)
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

    # 5. 初始化 LightningModule (不变)
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

    # ... (Logger, Checkpoint, Trainer, trainer.fit/test 均保持不变) ...
    # 🔴 修改 Logger 初始化逻辑
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
        # 🔴 明确创建 TensorBoardLogger 并指定 save_dir
        # 我们设置 name="" 来避免多余的 "default" 子目录
        # 日志将保存在: args.log_dir/version_X
        logger_instance = TensorBoardLogger(
            save_dir=args.log_dir,
            name="" 
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="step_{step:06d}-loss_{train/loss:.4f}",
        every_n_train_steps=config['logging']['save_step_frequency'],
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    
    trainer = pl.Trainer(
        logger=logger_instance, # <-- 现在 logger_instance 始终是一个配置好的 logger
        callbacks=[checkpoint_callback],
        max_steps=config['training']['n_steps'],
        devices=args.gpus,
        accelerator="gpu",
        precision=32,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=config['training']['gradient_clip_val']
    )
    # if args.mode == 'train':
    #     print(f"--- Starting Training ---")
    #     print(f"Log dir: {args.log_dir}")
    #     ckpt_to_resume = None
    #     if args.ckpt_path:
    #         print(f"--- 正在从 MeanFlow checkpoint 延续训练: {args.ckpt_path} ---")
    #         ckpt_to_resume = args.ckpt_path
    #     else:
    #         print(f"--- 从头开始训练 MeanFlow ---")
    #     trainer.fit(model, datamodule, ckpt_path=ckpt_to_resume)
    if args.mode == 'train':
        print(f"--- Starting Training ---")
        print(f"Log dir: {args.log_dir}")
        ckpt_to_resume = None
        if args.ckpt_path:
            # 🟢 现在这个可以正常工作了！
            print(f"--- 正在从 MeanFlow checkpoint 延续训练: {args.ckpt_path} ---")
            ckpt_to_resume = args.ckpt_path
        else:
            print(f"--- 从头开始训练 MeanFlow ---")
        
        # 🟢 Lightning 会自动加载 ckpt_to_resume
        # 它会调用我们覆盖的 load_state_dict(..., strict=False)
        # 并且它还会成功加载 optimizer, scheduler, 和 global_step
        trainer.fit(model, datamodule, ckpt_path=ckpt_to_resume)
    elif args.mode == 'test':
        if args.ckpt_path is None:
            raise ValueError("Must provide --ckpt_path for testing.")
            
        print(f"--- Starting Testing ---")
        print(f"--- Manually loading checkpoint with strict=False: {args.ckpt_path} ---")
        # 1. 手动加载 checkpoint
        # 此时 model (VideoLightningModule) 已经被初始化了,
        # 它从 JSON 中读取的 buffer (train_global_min_values 等) 已经存在。
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")

        # 2. 使用 strict=False 加载 state_dict
        # 这将加载所有匹配的 key (例如 U-Net 的权重)
        # 并自动忽略 checkpoint 中缺失的 buffer keys (train_global_min_values 等)
        # model 中已经从 JSON 加载的 buffer 值将被保留，不会被覆盖。
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        
        # 3. 调用 trainer.test，但不传入 ckpt_path
        # 因为模型状态已经手动加载完毕
        trainer.test(model, datamodule)

if __name__ == '__main__':
    main()