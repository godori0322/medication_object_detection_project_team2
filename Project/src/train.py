import torch
from tqdm import tqdm
import random
from pathlib import Path

from . import config
from .utils.visualizer import save_loss_curve
from . import data_loader
from models.model import get_detection_model

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

def train_model(model, train_loader, val_loader, training_config=None):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델 객체 (예: models.faster_rcnn(num_classes=100))
        train_loader: 훈련 데이터로더
        val_loader: 검증 데이터로더
        training_config: 학습 설정 딕셔너리
    """
    
    # 기본 설정
    if training_config is None:
        training_config = {}
    
    # 설정값 가져오기 (전달된 값이 있으면 사용, 없으면 기본값)
    num_epochs = training_config.get('num_epochs', config.NUM_EPOCHS)
    #device_name = training_config.get('device', config.DEVICE)
    test_image_dir = training_config.get('test_image_dir', config.TEST_IMAGE_DIR)
    
    # Optimizer 설정
    optimizer_config = training_config.get('optimizer', {})
    learning_rate = optimizer_config.get('learning_rate', config.LEARNING_RATE)
    weight_decay = optimizer_config.get('weight_decay', config.WEIGHT_DECAY)
    momentum = optimizer_config.get('momentum', 0.9)
    optimizer_type = optimizer_config.get('type', 'SGD')
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model.__class__.__name__.lower()
    
    print(f"{model_name.upper()} 모델 학습 시작")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimizer: {optimizer_type} (lr={learning_rate}, wd={weight_decay})")
    
    # 모델을 디바이스로 이동
    model = model.to(device)
    
    # 모델 타입에 따라 파라미터 추출
    all_params = model.parameters()
    params = [p for p in all_params if p.requires_grad]
    
    print(f"학습 가능한 파라미터 수: {len(params)}")
    if len(params) == 0:
        raise ValueError("학습 가능한 파라미터가 없습니다. 모델 구조를 확인해주세요.")
    
    # Optimizer 생성
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # 체크포인트 경로
    checkpoint_path = config.OUTPUT_DIR / f"{model_name}_best.pth"
    
    print(f"학습 시작: ({num_epochs} epochs)")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        avg_val_loss = validate_epoch(model, val_loader, device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"모델저장. validation loss: {best_val_loss:.4f}")

    # 손실 곡선 저장
    loss_curve_path = config.OUTPUT_DIR / f"{model_name}_loss_curve.png"
    save_loss_curve(train_losses, val_losses, num_epochs, loss_curve_path)
    
    print(f"{model_name.upper()} 모델 학습 완료")
    return model, checkpoint_path