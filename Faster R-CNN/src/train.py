# src/train.py

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# src 폴더의 다른 모듈과 설정 파일 임포트
from . import config
from .dataset import PillDataset
from .model import get_detection_model

# DataLoader는 배치 내 이미지 크기가 다를 수 있으므로 collate_fn이 필요합니다.
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device(config.DEVICE)
    
    # 1. 전체 데이터셋 로드
    print("Loading data...")
    full_dataset = PillDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        annotation_dir=config.TRAIN_ANNOTATION_DIR
    )

    # 2. 데이터를 학습용과 검증용으로 분할 (90% 학습, 10% 검증)
    # 데이터셋의 인덱스를 기준으로 분할합니다.
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    
    # 분할된 인덱스를 사용해 Subset 생성
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 3. 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    # 4. 모델 및 옵티마이저 설정
    model = get_detection_model(num_classes=config.NUM_CLASSES).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY
    )
    
    print("🚀 Training started!")
    
    # 각 에포크의 손실을 저장할 리스트
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # 가장 낮은 검증 손실을 기록하기 위한 변수

    # 5. 학습 및 검증 루프
    for epoch in range(config.NUM_EPOCHS):
        # --- 학습 단계 ---
        model.train()
        train_loop = tqdm(train_loader, leave=True)
        total_train_loss = 0
        
        for images, targets in train_loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_train_loss += losses.item()
            train_loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
            train_loop.set_postfix(train_loss=losses.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- 검증 단계 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        # 6. 가장 좋은 성능의 모델 저장 (Best Checkpoint)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_CHECKPOINT)
            print(f"✨ Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

    # 7. 학습 완료 후 손실 그래프 생성 및 저장
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.NUM_EPOCHS + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
    plt.plot(range(1, config.NUM_EPOCHS + 1), val_losses, marker='o', linestyle='--', label='Validation Loss')
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.xticks(range(1, config.NUM_EPOCHS + 1))
    
    loss_curve_path = config.OUTPUT_DIR / "training_loss_curve.png"
    plt.savefig(loss_curve_path)
    print(f"📈 Loss curve saved to {loss_curve_path}")


if __name__ == '__main__':
    main()