"""Dataset và DataLoader đọc từ split.csv.

Định dạng split.csv (Son xuất ra, cả nhóm dùng chung):
    filepath,label,split
    classification_task/train/glioma/brisc2025_train_00001_gl_ax_t1.jpg,glioma,train
- filepath: đường dẫn tương đối so với --data-root (để chạy được trên Colab/Kaggle/máy cá nhân)
- label   : một trong glioma, meningioma, no_tumor, pituitary
- split   : train, val hoặc test
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import CLASS_TO_IDX, DEFAULT_BATCH_SIZE, SEED
from split_utils import read_split
from transforms import get_eval_transforms, get_train_transforms
from utils import seed_worker


class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.labels = [CLASS_TO_IDX[l] for l in self.df["label"]]

    def __len__(self):
        return len(self.df)

    def path_of(self, idx):
        return os.path.join(self.data_root, self.df.loc[idx, "filepath"])

    def __getitem__(self, idx):
        # Ảnh MRI thường là ảnh xám -> chuyển sang RGB 3 kênh để dùng trọng số ImageNet
        img = Image.open(self.path_of(idx)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def build_datasets(split_csv: str, data_root: str):
    df = read_split(split_csv)
    return {
        "train": MRIDataset(df[df.split == "train"], data_root, get_train_transforms()),
        "val": MRIDataset(df[df.split == "val"], data_root, get_eval_transforms()),
        "test": MRIDataset(df[df.split == "test"], data_root, get_eval_transforms()),
    }


def get_loaders(split_csv: str, data_root: str, batch_size: int = DEFAULT_BATCH_SIZE,
                num_workers: int = 2, seed: int = SEED):
    """Trả về (loaders, datasets). Train shuffle=True, Val/Test shuffle=False."""
    datasets = build_datasets(split_csv, data_root)
    g = torch.Generator()
    g.manual_seed(seed)
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin,
                  worker_init_fn=seed_worker, persistent_workers=num_workers > 0)
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=g, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, datasets
