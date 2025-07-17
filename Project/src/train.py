# src/train.py

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.dataset import PillDataset
from config import get_optimizer
from .utils.visualizer import save_loss_curve

def collate_fn(batch):
    return tuple(zip(*batch))

def create_dataloaders(config):
    print("Loading data...")
    full_dataset = PillDataset(config.train_image_dir, config.train_annotation_dir)
    
    print(f"Total dataset size: {len(full_dataset)}")
    
    # 데이터셋이 비어있는지 확인
    if len(full_dataset) == 0:
        raise ValueError(f"Dataset is empty! Check paths:\n"
                        f"Image dir: {config.train_image_dir}\n"
                        f"Annotation dir: {config.train_annotation_dir}")
    
    # 데이터 분할
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=config.num_workers)

    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs):
    model.train()
    train_loop = tqdm(train_loader, leave=True)
    total_loss = 0
    
    for images, targets in train_loop:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        train_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loop.set_postfix(train_loss=losses.item())
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, cfg):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델 객체 (예: models.faster_rcnn(num_classes=100))
        train_loader: 훈련 데이터로더
        val_loader: 검증 데이터로더
        config: 하이퍼 파라미터 관리 config
    """
    # 옵티마이저 생성
    optimizer = get_optimizer(model, cfg)

    model_name = model.__class__.__name__
    optimizer_type = type(optimizer).__name__
    
    print(f"{model_name.upper()} 모델 학습 시작")
    print(f"Device: {cfg.device}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Optimizer: {optimizer_type} (lr={cfg.learning_rate}, wd={cfg.weight_decay})")
    
    # 모델을 디바이스로 이동
    model = model.to(cfg.device)
    
    # 모델 타입에 따라 파라미터 추출
    all_params = model.parameters()
    params = [p for p in all_params if p.requires_grad]
    
    print(f"학습 가능한 파라미터 수: {len(params)}")
    if len(params) == 0:
        raise ValueError("학습 가능한 파라미터가 없습니다. 모델 구조를 확인해주세요.")
    
    # 체크포인트 경로
    checkpoint_path = cfg.output_dir / f"{model_name}_best.pth"
    
    print(f"학습 시작: ({cfg.num_epochs} epochs)")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        avg_train_loss = train_epoch(model, train_loader, optimizer, cfg.device, epoch, cfg.num_epochs)
        avg_val_loss = validate_epoch(model, val_loader, cfg.device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"모델저장. validation loss: {best_val_loss:.4f}")

    # 손실 곡선 저장
    loss_curve_path = cfg.output_dir / f"{model_name}_loss_curve.png"
    save_loss_curve(train_losses, val_losses, cfg.num_epochs, loss_curve_path)
    
    print(f"{model_name.upper()} 모델 학습 완료")
    return model, checkpoint_path