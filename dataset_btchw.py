import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, seed_everything
import h5py
import json
import numpy as np
import os
# from functools import partial


# --- 工具函数：对 (T, C, H, W) 的 data 在 H/W 维度随机翻转 ---
def random_flip_tensor(data: torch.Tensor, p: float):
    """
    对 (T,C,H,W) tensor 在 H/W 维度上随机翻转
    p: 每个维度独立触发的概率
    """
    if p <= 0:
        return data

    # # 时间翻转 (T)
    # if torch.rand(1).item() < p:
    #     data = torch.flip(data, dims=[0])

    # 垂直翻转 (H)
    if torch.rand(1).item() < p:
        data = torch.flip(data, dims=[-2])

    # 水平翻转 (W)
    if torch.rand(1).item() < p:
        data = torch.flip(data, dims=[-1])
    return data.contiguous()


class Xiaoshan_6steps_30min_Test_Dataset(Dataset):
    def __init__(self, data_path, json_path, dataset_prefix):
        r"""
            初始化数据集，并根据 train_ratio 进行顺序划分。
            :param args.data_path: HDF5 文件的路径。
            :param args.dataset_prefix: 数据集名称的前缀，例如 '2022'。
            :param args.split: 'train' 或 'valid'，指定数据集的划分部分。
        """
        super().__init__()
        self.data_path = data_path
        self.dataset_names = []
        self.dataset_lengths = []
        
        # 数据集的最大最小值和范围的计算
        with open(json_path, 'r') as f:
            global_stats = json.load(f)
        self.global_max_values = np.array(global_stats['Global Max'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_min_values = np.array(global_stats['Global Min'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_range = self.global_max_values - self.global_min_values

        # 读取 HDF5 文件中的数据集名称和长度信息
        with h5py.File(self.data_path, 'r') as file:
            for month in range(1, 13):
                dataset_name = f'{dataset_prefix}{month:02}'
                if dataset_name in file:
                    self.dataset_names.append(dataset_name)
                    self.dataset_lengths.append(file[dataset_name].shape[0])

        # 记录每个数据集的累积长度，以便在 __getitem__ 中进行索引映射
        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))  # [0, 148, 678, 943, ...]

        # 获取总样本数量
        total_samples = self.cumulative_lengths[-1]

        self.indices = np.arange(0, total_samples)

    def __len__(self):
        """
        返回数据集中样本的总数量。
        """
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        dataset_index = next(i for i, cl in enumerate(self.cumulative_lengths) if cl > global_idx) - 1
        dataset_idx = global_idx - self.cumulative_lengths[dataset_index]
        dataset_name = self.dataset_names[dataset_index]

        # 仅在需要时打开文件
        with h5py.File(self.data_path, 'r') as file:
            dataset = file[dataset_name]
            data = dataset[dataset_idx]

        data = (data - self.global_min_values) / self.global_range
        tensor = torch.tensor(data, dtype=torch.float32).contiguous() # T C H W

        # 分割数据：前6个为input，后6个为target
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
        r"""
            初始化数据集，并根据 train_ratio 进行顺序划分。
            :param args: 包含数据路径、训练比例等信息的参数。
            :param split: 'train' 或 'valid'，指定数据集的划分部分。
        """
        super().__init__()
        self.data_path = data_path
        self.dataset_names = []
        self.dataset_lengths = []
        # 只在训练集启用
        self.random_flip = float(random_flip) if split == 'train' else 0.0

        # 加载全局统计信息
        with open(json_path, 'r') as f:
            global_stats = json.load(f)
        self.global_max_values = np.array(global_stats['Global Max'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_min_values = np.array(global_stats['Global Min'], dtype=np.float32).reshape(1, 8, 1, 1)
        self.global_range = self.global_max_values - self.global_min_values

        # 读取 HDF5 文件中的数据集名称和长度信息
        with h5py.File(self.data_path, 'r') as file:
            for month in range(1, 13):
                dataset_name = f'{dataset_prefix}{month:02}'
                if dataset_name in file:
                    self.dataset_names.append(dataset_name)
                    self.dataset_lengths.append(file[dataset_name].shape[0])

        # 记录每个数据集的累积长度
        self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))

        # 根据每个月划分训练集和验证集
        self.train_indices = []
        self.valid_indices = []

        for month_index in range(len(self.dataset_lengths)):
            month_length = self.dataset_lengths[month_index]
            train_size = int(train_ratio * month_length)
            
            # 将每个月的训练集和验证集索引添加到对应列表
            self.train_indices.extend(range(self.cumulative_lengths[month_index], 
                                            self.cumulative_lengths[month_index] + train_size))
            self.valid_indices.extend(range(self.cumulative_lengths[month_index] + train_size, 
                                            self.cumulative_lengths[month_index] + month_length))

        # 根据 split 参数，选择训练集或验证集的索引
        if split == 'train':
            self.indices = np.array(self.train_indices)
        else:
            self.indices = np.array(self.valid_indices)

    def __len__(self):
        """
        返回数据集中样本的总数量。
        """
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        dataset_index = next(i for i, cl in enumerate(self.cumulative_lengths) if cl > global_idx) - 1
        dataset_idx = global_idx - self.cumulative_lengths[dataset_index]
        dataset_name = self.dataset_names[dataset_index]

        # 仅在需要时打开文件
        with h5py.File(self.data_path, 'r') as file:
            dataset = file[dataset_name]
            data = dataset[dataset_idx]

        data = (data - self.global_min_values) / self.global_range
        tensor = torch.tensor(data, dtype=torch.float32).contiguous() # T C H W
        tensor = random_flip_tensor(tensor, self.random_flip)

        # 分割数据：前6个为input，后6个为target
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
        assert os.path.exists(self.train_data_path), f"train data not found! Should be located at {self.train_data_path}"
        assert os.path.exists(self.train_json_path), f"train json not found! Should be located at {self.train_json_path}"
        assert os.path.exists(self.test_data_path), f"test data not found! Should be located at {self.test_data_path}"
        assert os.path.exists(self.test_json_path), f"test json not found! Should be located at {self.test_json_path}"

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
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    @property
    def num_train_samples(self):
        return len(self.train_set)

    @property
    def num_val_samples(self):
        return len(self.val_set)

    @property
    def num_test_samples(self):
        return len(self.test_set)


