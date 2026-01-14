import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, seed_everything
import h5py
import json
import numpy as np
import os

# --- 工具函数保持不变 ---
def random_flip_tensor(data: torch.Tensor, p: float):
    if p <= 0:
        return data
    if torch.rand(1).item() < p:
        data = torch.flip(data, dims=[-2])
    if torch.rand(1).item() < p:
        data = torch.flip(data, dims=[-1])
    return data.contiguous()

class Xiaoshan_6steps_30min_Test_Dataset(Dataset):
    def __init__(self, data_path, json_path, dataset_prefix):
        super().__init__()
        self.data_path = data_path
        self.dataset_names = []
        self.dataset_lengths = []
        
        with open(json_path, 'r') as f:
            global_stats = json.load(f)
        self.global_max_values = np.array(global_stats['Global Max'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_min_values = np.array(global_stats['Global Min'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_range = self.global_max_values - self.global_min_values

        with h5py.File(self.data_path, 'r') as file:
            for month in range(1, 13):
                dataset_name = f'{dataset_prefix}{month:02}'
                if dataset_name in file:
                    self.dataset_names.append(dataset_name)
                    self.dataset_lengths.append(file[dataset_name].shape[0])

        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))
        total_samples = self.cumulative_lengths[-1]
        self.indices = np.arange(0, total_samples)
        
        # 新增：用于在子进程中保存文件句柄
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 优化点：不再使用 with 语句，而是判断句柄是否存在
        # 这样每个进程只会打开一次文件，直到训练结束
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r', swmr=True, libver='latest')

        global_idx = self.indices[idx]
        dataset_index = next(i for i, cl in enumerate(self.cumulative_lengths) if cl > global_idx) - 1
        dataset_idx = global_idx - self.cumulative_lengths[dataset_index]
        dataset_name = self.dataset_names[dataset_index]

        # 从持久化的句柄中读取
        dataset = self.file[dataset_name]
        data = dataset[dataset_idx]

        data = (data - self.global_min_values) / self.global_range
        tensor = torch.tensor(data, dtype=torch.float32).contiguous()

        input_data = tensor[:6]
        target_data = tensor[6:]
        
        return input_data, target_data


class Xiaoshan_6steps_30min_Dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 json_path, 
                 dataset_prefix, 
                 train_ratio, 
                 split, 
                 random_flip: float =0.0):
        super().__init__()
        self.data_path = data_path
        self.dataset_names = []
        self.dataset_lengths = []
        self.random_flip = float(random_flip) if split == 'train' else 0.0

        with open(json_path, 'r') as f:
            global_stats = json.load(f)
        self.global_max_values = np.array(global_stats['Global Max'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_min_values = np.array(global_stats['Global Min'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_range = self.global_max_values - self.global_min_values

        with h5py.File(self.data_path, 'r') as file:
            for month in range(1, 13):
                dataset_name = f'{dataset_prefix}{month:02}'
                if dataset_name in file:
                    self.dataset_names.append(dataset_name)
                    self.dataset_lengths.append(file[dataset_name].shape[0])

        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))
        self.train_indices = []
        self.valid_indices = []

        for month_index in range(len(self.dataset_lengths)):
            month_length = self.dataset_lengths[month_index]
            train_size = int(train_ratio * month_length)
            self.train_indices.extend(range(self.cumulative_lengths[month_index], 
                                            self.cumulative_lengths[month_index] + train_size))
            self.valid_indices.extend(range(self.cumulative_lengths[month_index] + train_size, 
                                            self.cumulative_lengths[month_index] + month_length))

        if split == 'train':
            self.indices = np.array(self.train_indices)
        else:
            self.indices = np.array(self.valid_indices)

        # 新增：用于在子进程中保存文件句柄
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 优化点：每个子进程只打开一次文件
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r', swmr=True, libver='latest')

        global_idx = self.indices[idx]
        dataset_index = next(i for i, cl in enumerate(self.cumulative_lengths) if cl > global_idx) - 1
        dataset_idx = global_idx - self.cumulative_lengths[dataset_index]
        dataset_name = self.dataset_names[dataset_index]

        # 从持久化的句柄中读取
        dataset = self.file[dataset_name]
        data = dataset[dataset_idx]

        data = (data - self.global_min_values) / self.global_range
        tensor = torch.tensor(data, dtype=torch.float32).contiguous()
        tensor = random_flip_tensor(tensor, self.random_flip)

        input_data = tensor[:6]
        target_data = tensor[6:]
        
        return input_data, target_data
    

class Himawari8LightningDataModule(LightningDataModule):
    def __init__(self, 
                 dataset_name='Himawari8',
                 train_data_path='/mnt/data1/Dataset/Himawari8/train/xiaoshan_6steps_30min_Data_day.h5',
                 train_json_path='/mnt/data1/Dataset/Himawari8/train/xiaoshan_6steps_30min_Data_day_all.json',
                 train_dataset_prefix='2022',
                 train_ratio=0.9,
                 train_random_flip=0.0,
                 test_data_path='/mnt/data1/Dataset/Himawari8/test/full_2023_xiaoshan_6steps_30min_Data_day.h5',
                 test_json_path='/mnt/data1/Dataset/Himawari8/test/full_2023_xiaoshan_6steps_30min_Data_day_all.json',
                 test_dataset_prefix='2023',
                 batch_size=1,
                 num_workers=8,
                 pin_memory=False,
                 seed=0):
        super(Himawari8LightningDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.train_data_path = train_data_path
        self.train_json_path = train_json_path
        self.train_dataset_prefix =train_dataset_prefix
        self.train_ratio = train_ratio
        self.train_random_flip=train_random_flip
        self.test_data_path = test_data_path
        self.test_json_path = test_json_path
        self.test_dataset_prefix = test_dataset_prefix
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory=pin_memory
        self.seed = seed

    def prepare_data(self) -> None:
        assert os.path.exists(self.train_data_path), f"train data not found!"
        assert os.path.exists(self.test_data_path), f"test data not found!"

    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            self.train_set =  Xiaoshan_6steps_30min_Dataset(self.train_data_path, self.train_json_path, self.train_dataset_prefix, 
                                                            self.train_ratio, split='train', random_flip=self.train_random_flip)
            self.val_set =  Xiaoshan_6steps_30min_Dataset(self.train_data_path, self.train_json_path, self.train_dataset_prefix, 
                                                            self.train_ratio, split='val')
        if stage in (None, "test"):
            self.test_set =  Xiaoshan_6steps_30min_Test_Dataset(self.test_data_path, self.test_json_path, self.test_dataset_prefix)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          # 优化点：保持 worker 进程不销毁，避免句柄重置
                          persistent_workers=(self.num_workers > 0))

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=(self.num_workers > 0))

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=(self.num_workers > 0))

    @property
    def num_train_samples(self):
        return len(self.train_set)

    @property
    def num_val_samples(self):
        return len(self.val_set)

    @property
    def num_test_samples(self):
        return len(self.test_set)