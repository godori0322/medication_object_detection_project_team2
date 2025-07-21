<<<<<<< HEAD
=======
import pandas as pd
import json
from pathlib import Path
>>>>>>> feature/arc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.dataset import PillDataset, PillTestDataset
from src.transforms import get_train_transforms, get_valid_transforms, get_test_transforms
from src.utils.tensor_utils import collate_fn
from src.utils.data_utils import load_filtered_df, load_mappings


def create_dataloaders(cfg):
    print("Loading data...")
    # 1. 어노테이션 및 매핑 로딩
    filtered_df = load_filtered_df(cfg)
    mappings = load_mappings(cfg)

    # 2. 학습/검증 분할
    unique_ids = filtered_df['image_id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    
    train_df = filtered_df[filtered_df['image_id'].isin(train_ids)].reset_index(drop=True)
    val_df = filtered_df[filtered_df['image_id'].isin(val_ids)].reset_index(drop=True)

    # 3. Transform 구성
    train_tf = get_train_transforms()
    val_tf = get_valid_transforms()
    test_tf = get_test_transforms()

    # 4. Dataset 생성
    print(f"train데이터 생성 : {cfg.train_image_dir}")
    train_dataset = PillDataset(
        img_dir=cfg.train_image_dir,
        labels_df=train_df,
        mappings=mappings,
        transforms=train_tf
    )

    val_dataset = PillDataset(
        img_dir=cfg.train_image_dir,
        labels_df=val_df,
        mappings=mappings,
        transforms=val_tf
    )

    test_dataset = PillTestDataset(
        img_dir=cfg.test_image_dir,
        transforms=test_tf
    )

    # 5. DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

<<<<<<< HEAD
    return train_loader, val_loader, test_loader, mappings
=======
    return train_loader, val_loader, test_loader, mappings
>>>>>>> feature/arc
