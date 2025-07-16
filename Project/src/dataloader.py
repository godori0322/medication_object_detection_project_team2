# src/dataloader.py

import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import PillDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def PillDataloader(DataLoader):
    # 1. 데이터셋 및 데이터로더
    print("Loading data...")
    dataset = PillDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        annotation_dir=config.TRAIN_ANNOTATION_DIR
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 # CPU 코어 수에 맞게 조절
    )

    return data_loader