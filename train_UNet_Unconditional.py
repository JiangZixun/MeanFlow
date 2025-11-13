# train_lightning.py
import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_linear_schedule_with_warmup
import matplotlib
matplotlib.use('Agg')

# 导入你项目中的模块
from dataset_btchw import Xiaoshan_6steps_30min_Dataset, Xiaoshan_6steps_30min_Test_Dataset
from models.UNet_Unconditional import UNet
from videoMeanflow_Unconditional import MeanFlow
from visualize import vis_himawari8_seq_btchw

# --- VideoDataModule 保持不变 ---
class VideoDataModule(pl.LightningDataModule):
    """
    封装数据加载的 Lightning DataModule
    """
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
            # 假设测试集使用完整数据
            self.test_dataset = Xiaoshan_6steps_30min_Test_Dataset(
                data_path=self.config['data_path'],
                json_path=self.config['json_path'],
                dataset_prefix=self.config['dataset_prefix'],
            )
            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False
        )


class VideoLightningModule(pl.LightningModule):
    """
    封装模型、损失和训练/验证逻辑的 Lightning Module
    """
    def __init__(self, model_config, meanflow_config, optimizer_config, scheduler_config, training_config, logging_config):
        super().__init__()
        self.save_hyperparameters()

        # --- 🔴 修改 UNet 实例化 ---
        self.model = UNet(
            input_size=model_config['input_size'],
            # 假设 'out_channels_c' 是 c_past 和 x_future 的通道数
            data_channels_c=model_config['out_channels_c'], 
            time_emb_dim=model_config['time_emb_dim']
        )
        
        # --- 🔴 修改 MeanFlow 实例化 (移除 CFG) ---
        self.meanflow = MeanFlow(
            channels=model_config['out_channels_c'],
            time_dim=model_config['input_size'][0],
            height_dim=model_config['input_size'][1],
            width_dim=model_config['input_size'][2],
            normalizer=['minmax', None, None],
            flow_ratio=meanflow_config['flow_ratio'],
            time_dist=meanflow_config['time_dist'],
            # 假设 jvp_api 在 config 中，如果不在，MeanFlow 会使用默认值 'autograd'
            jvp_api=meanflow_config.get('jvp_api', 'autograd')
        )
        
        self.val_loader_iter = None

    def configure_optimizers(self):
        # --- 保持不变 ---
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def training_step(self, batch, batch_idx):
        # --- 保持不变 (逻辑已移至 MeanFlow.loss) ---
        c_past, x_future = batch
        loss, mse_val = self.meanflow.loss(self.model, x_future, c_past)
        
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/mse_loss', mse_val, on_step=True, on_epoch=False)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # --- 保持不变 (逻辑已移至 MeanFlow.sample_prediction) ---
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
                z = self.meanflow.sample_prediction(
                    self.model, 
                    c_past_val, 
                    sample_steps=1000, # modify
                    device=self.device
                )

            c_past_sample = c_past_val[0].cpu()
            z_sample = z[0].cpu()
            x_future_sample = x_future_val[0].cpu()
            
            del c_past_val, x_future_val, z, val_batch
            torch.cuda.empty_cache()

            context_list = [c_past_sample[t] for t in range(c_past_sample.shape[0])]
            pred_list = [z_sample[t] for t in range(z_sample.shape[0])]
            target_list = [x_future_sample[t] for t in range(x_future_sample.shape[0])]

            save_dir = os.path.join(self.trainer.logger.save_dir, "images", f"step_{self.global_step}")
            
            vis_himawari8_seq_btchw(
                save_dir=save_dir,
                context_seq=context_list,
                pred_seq=pred_list,
                target_seq=target_list
            )
            
            self.model.train()

    def test_step(self, batch, batch_idx):
        # --- 保持不变 ---
        c_past, x_future = batch
        loss, mse_val = self.meanflow.loss(self.model, x_future, c_past)
        
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/mse_loss', mse_val, on_step=False, on_epoch=True)


def main():
    # --- 整个 main 函数保持不变 ---
    parser = argparse.ArgumentParser(description="PyTorch Lightning Video Prediction Training")
    
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to the config.yaml file")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory to save logs and checkpoints")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size (overrides config if set)")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="Training or testing mode")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to checkpoint for testing")
    parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    pl.seed_everything(42, workers=True)

    datamodule = VideoDataModule(
        data_config=config['data'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    model = VideoLightningModule(
        model_config=config['model'],
        meanflow_config=config['meanflow'],
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler'],
        training_config=config['training'],
        logging_config=config['logging']
    )

    wandb_logger = WandbLogger(
        project=config['logging']['project_name'],
        save_dir=args.log_dir,
        name=os.path.basename(args.log_dir)
    )
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
        print(f"Config: {config}")
        print(f"Log dir: {args.log_dir}")
        if args.ckpt_path:
            print(f"--- 正在从 checkpoint 延续训练: {args.ckpt_path} ---")
            trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
        else:
            print(f"--- 从头开始训练 ---")
            trainer.fit(model, datamodule)
    elif args.mode == 'test':
        if args.ckpt_path is None:
            raise ValueError("Must provide --ckpt_path for testing.")
        print(f"--- Starting Testing ---")
        print(f"Loading checkpoint: {args.ckpt_path}")
        trainer.test(model, datamodule, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()