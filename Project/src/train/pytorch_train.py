# src/train.py

import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.config import get_optimizer, get_lr_scheduler
from ..utils.logger import create_experiment_dir, Logger
from ..utils.visualizer import save_loss_curve

def validate_metrics_epoch(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            
            # torchmetrics는 dict(list[Tensor]) 형식을 요구함
            # 즉, targets: list[dict], outputs: list[dict] → 그대로 사용 가능
            metric.update(outputs, targets)
    
    return metric.compute()  # dict 형태로 반환 (mAP, recall, precision 등 포함)

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
            targets = [t.to(device) for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    return total_loss / len(val_loader)

def validate_loss_epoch(model, val_loader, device):
    model.train()  # loss 계산을 위해 train 모드
    total_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # loss 반환

            losses = sum(
                v.item() for v in loss_dict.values()
                if torch.is_tensor(v) and v.dim() == 0
            )
            total_loss += losses

    return total_loss / len(val_loader)

def train_pytorch(model, train_loader, val_loader, cfg):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델 객체 (예: models.faster_rcnn(num_classes=100))
        train_loader: 훈련 데이터로더
        val_loader: 검증 데이터로더
        config: 하이퍼 파라미터 관리 config
    """

    # 실험 결과 저장용 디렉토리 생성 (output_dir 기반)
    experiment_dir = create_experiment_dir(cfg.output_dir, model.__class__.__name__)
    print(f"Experiment directory created at: {experiment_dir}")
    
    # cfg 객체의 output_dir 경로를 실험별 폴더로 교체
    cfg.output_dir = Path(experiment_dir)

    # best model state dict 저장하는 checkpoint_path 설정
    checkpoint_path = cfg.output_dir / f"{model.__class__.__name__}_best.pth"

    # 실험 결과 저장용 디렉토리에 config의 하이퍼 파라미터 정보 csv 파일로 저장
    num_train_images = len(train_loader.dataset)
    
    logger = Logger(cfg.output_dir)
    hyperparams = {
        "Number of Used Images": num_train_images, 
        "device": cfg.device,
        "num_epochs": cfg.num_epochs,
        "num_classes": cfg.num_classes, 
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "lrf": cfg.lrf,
        "lr_scheduler": cfg.lr_scheduler,
        "optimizer": cfg.optimizer,
        "num_workers": cfg.num_workers, 
        "weight_decay": cfg.weight_decay,
        "confidence_threshold": cfg.confidence_threshold, 
        "momentum": cfg.momentum
    }
    logger.save_hyperparameters_csv(hyperparams)

    # 옵티마이저 생성
    optimizer = get_optimizer(model, cfg)

    # 학습률 스케줄러 생성
    scheduler = get_lr_scheduler(optimizer, cfg)

    model_name = model.__class__.__name__
    optimizer_type = type(optimizer).__name__
    
    print(f"{model_name.upper()} 모델 학습 시작")
    print(f"Device: {cfg.device}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Optimizer: {optimizer_type} (lr={cfg.lr}, wd={cfg.weight_decay})")
    
    # 모델을 디바이스로 이동
    model = model.to(cfg.device)
    
    # 모델 타입에 따라 파라미터 추출
    all_params = model.parameters()
    params = [p for p in all_params if p.requires_grad]
    
    print(f"학습 가능한 파라미터 수: {len(params)}")
    if len(params) == 0:
        raise ValueError("학습 가능한 파라미터가 없습니다. 모델 구조를 확인해주세요.")
    
    print(f"학습 시작: ({cfg.num_epochs} epochs)")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # GPU 메모리 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics_log = []  # <- 성능 기록용 리스트

    for epoch in range(cfg.num_epochs):
        avg_train_loss = train_epoch(model, train_loader, optimizer, cfg.device, epoch, cfg.num_epochs)
        avg_val_loss = validate_loss_epoch(model, val_loader, cfg.device)
        metrics = validate_metrics_epoch(model, val_loader, cfg.device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 현재 learning rate 로그 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"mAP: {metrics['map']:.4f}, mAP@50: {metrics['map_50']:.4f}, mAP@75: {metrics['map_75']:.4f}")

        # 저장용 dict 생성
        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "map": metrics['map'].item(),
            "map_50": metrics['map_50'].item()
        })

        # best model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"모델저장. validation loss: {best_val_loss:.4f}")

        # 스케줄러 step
        if scheduler:
            if cfg.scheduler == 'ReduceLROnPlateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    # 성능 기록 CSV로 저장
    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = cfg.output_dir / f"{model_name}_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[✓] Validation 성능 기록 저장 완료: {metrics_csv_path}")

    # 손실 곡선 저장
    loss_curve_path = cfg.output_dir / f"{model_name}_loss_curve.png"
    save_loss_curve(train_losses, val_losses, cfg.num_epochs, loss_curve_path)
    
    # 손실 저장
    logger.save_loss_history_csv(train_losses, val_losses)

    print(f"{model_name.upper()} 모델 학습 완료")
    return model
