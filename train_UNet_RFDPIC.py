# train_lightning.py (修改版)

# ... (所有 import 保持不变, 包括 RFDPIC 的) ...
import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_linear_schedule_with_warmup
from collections import OrderedDict

from models.MotionPredictor.RFDPIC_Dual_Rotation_dyn import RFDPIC_Dual_Rotation_Dyn
from utils.transform import data_transform, inverse_data_transform
from dataset_btchw import Xiaoshan_6steps_30min_Dataset, Xiaoshan_6steps_30min_Test_Dataset
from models.UNet import UNet
from videoMeanflow_RFDPIC import MeanFlow 
from visualize import vis_himawari8_seq_btchw


class VideoDataModule(pl.LightningDataModule):
    # ... (此类完全不变) ...
    def __init__(self, data_config: dict, batch_size: int, num_workers: int):
        super().__init__()
        self.config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Xiaoshan_6steps_30min_Dataset(
                data_path=self.config['data_path'],
                json_path=self.config['json_path'],
                dataset_prefix=self.config['dataset_prefix'],
                train_ratio=self.config['train_ratio'],
                split='train',
                random_flip=self.config['random_flip']
            )
            self.val_dataset = Xiaoshan_6steps_30min_Dataset(
                data_path=self.config['data_path'],
                json_path=self.config['json_path'],
                dataset_prefix=self.config['dataset_prefix'],
                train_ratio=self.config['train_ratio'],
                split='valid',
                random_flip=0.0
            )
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")
        if stage == 'test' or stage is None:
            self.test_dataset = Xiaoshan_6steps_30min_Test_Dataset(
                data_path=self.config['data_path'],
                json_path=self.config['json_path'],
                dataset_prefix=self.config['dataset_prefix'],
            )
            print(f"Test dataset size: {len(self.test_dataset)}")
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False)


class VideoLightningModule(pl.LightningModule):
    # ... (load_frozen_rfdpic 和 configure_optimizers 保持不变) ...

    def __init__(self, model_config, meanflow_config, optimizer_config, scheduler_config, training_config, logging_config,
                 rfdpic_config_path: str,
                 rfdpic_ckpt_path: str):
        
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
        
        # 3. 实例化 U-Net
        # 🔴 这里的 model_config['in_channels_c'] 必须是 6 (由 main() 传入)
        print(f"Initializing UNet with condition input_t (in_channels_c): {model_config['in_channels_c']}")
        self.model = UNet(
            input_size=model_config['input_size'],
            in_channels_c=model_config['in_channels_c'], # 应该是 6
            out_channels_c=model_config['out_channels_c'],
            time_emb_dim=model_config['time_emb_dim']
        )
        
        self.val_loader_iter = None
    
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
        state_dict = torch.load(rfdpic_ckpt_path, map_location="cpu")
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
        # c_start = c_past
        # c_cond = c_rfdpic
        loss, mse_val = self.meanflow.loss(self.model, x_future, c=(c_past, c_rfdpic))
        
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
                
                # c_start = c_past_val
                # c_cond = c_rfdpic_val
                c_tuple_val = (c_past_val, c_rfdpic_val)

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

    def test_step(self, batch, batch_idx):
        c_past, x_future = batch
        with torch.no_grad():
            c_past_norm = data_transform(c_past, rescaled=self.rfdpic_rescaled)
            c_rfdpic_norm, _, _, _, _ = self.rfdpic_model(c_past_norm)
            c_rfdpic = inverse_data_transform(c_rfdpic_norm, rescaled=self.rfdpic_rescaled)
            c_rfdpic = c_rfdpic.detach()
        
        # --- 🔴 3. 修改：将 c 作为元组传递 ---
        loss, mse_val = self.meanflow.loss(self.model, x_future, c=(c_past, c_rfdpic))
        
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/mse_loss', mse_val, on_step=False, on_epoch=True)


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
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    pl.seed_everything(42, workers=True)

    # 4. 初始化 DataModule (不变)
    datamodule = VideoDataModule(
        data_config=config['data'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # 5. 初始化 LightningModule (不变)
    model = VideoLightningModule(
        model_config=config['model'],
        meanflow_config=config['meanflow'],
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler'],
        training_config=config['training'],
        logging_config=config['logging'],
        rfdpic_config_path=args.rfdpic_config,
        rfdpic_ckpt_path=args.rfdpic_ckpt
    )

    # ... (Logger, Checkpoint, Trainer, trainer.fit/test 均保持不变) ...
    wandb_logger = WandbLogger(project=config['logging']['project_name'], save_dir=args.log_dir, name=os.path.basename(args.log_dir))
    wandb_logger.watch(model, log="all", log_freq=500)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="step_{step:06d}-loss_{train/loss:.4f}",
        every_n_train_steps=config['logging']['save_step_frequency'],
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
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
            print(f"--- 从头开始训练 MeanFlow ---")
        trainer.fit(model, datamodule, ckpt_path=ckpt_to_resume)
    elif args.mode == 'test':
        if args.ckpt_path is None:
            raise ValueError("Must provide --ckpt_path for testing.")
        print(f"--- Starting Testing ---")
        print(f"Loading MeanFlow checkpoint: {args.ckpt_path}")
        trainer.test(model, datamodule, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()