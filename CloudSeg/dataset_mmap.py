import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalize16:
    """Same min-max normalization settings as /mnt/data1/hzy/data/dataset.py."""

    def __init__(self, band_num: int = 16):
        self.band_num = band_num
        self.min_val = np.array([
            0, 0, 0, 0, 0, 0,
            207.1699981689453, 187.04998779296875, 183.6999969482422,
            183.8599853515625, 182.22999572753906, 208.33999633789062,
            183.83999633789062, 181.66000366210938, 181.54998779296875,
            184.88999938964844,
        ], dtype=np.float32).reshape(16, 1, 1)
        self.max_val = np.array([
            1.2105, 1.21420002, 1.19639993, 1.22599995, 1.23109996, 1.23199999,
            370.76998901, 261.57998657, 271.57998657, 274.79998779,
            313.61999512, 281.30999756, 317.63000488, 317.38998413,
            310.6000061, 283.77999878,
        ], dtype=np.float32).reshape(16, 1, 1)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        x = img[: self.band_num].astype(np.float32, copy=False)
        x = (x - self.min_val) / (self.max_val - self.min_val + 1e-8)
        x[x < 0] = 0.0
        return torch.from_numpy(x)


class CloudSegMMapDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        num_classes: int = 10,
        ignore_index: int = 10,
        input_channels: int = 16,
    ):
        super().__init__()
        mmap_dir = os.path.join(root, split, "mmap")
        manifest_path = os.path.join(mmap_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._norm = Normalize16(band_num=input_channels)

        self.shards: List[Tuple[np.memmap, np.memmap, int]] = []
        for s in self.manifest["shards"]:
            img_file = os.path.join(mmap_dir, s["image_file"])
            lbl_file = os.path.join(mmap_dir, s["label_file"])
            cnt = int(s["count"])
            img_mm = np.load(img_file, mmap_mode="r")
            lbl_mm = np.load(lbl_file, mmap_mode="r")
            self.shards.append((img_mm, lbl_mm, cnt))

        self.index_map: List[Tuple[int, int]] = []
        for si, (_, _, cnt) in enumerate(self.shards):
            for li in range(cnt):
                self.index_map.append((si, li))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self.index_map[idx]
        img_mm, lbl_mm, _ = self.shards[shard_idx]

        img = np.asarray(img_mm[local_idx])
        label = np.asarray(lbl_mm[local_idx]).astype(np.int64)

        x = self._norm(img)
        y = torch.from_numpy(label)
        y[y >= self.num_classes] = self.ignore_index
        y[y < 0] = self.ignore_index
        return x, y
