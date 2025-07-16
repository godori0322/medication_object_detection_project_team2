# src/train.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # 학습 진행률 시각화
import matplotlib.pyplot as plt # Matplotlib 임포트

# src 폴더의 다른 모듈과 설정 파일 임포트
from . import config
from .dataloader import data_loader
from models.model import get_detection_model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device(config.DEVICE)
    
    model = get_detection_model(num_classes=config.NUM_CLASSES).to(device)

    # 3. 옵티마이저
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY
    )
    
    print("🚀 Training started!")

    # 각 에포크의 평균 손실을 저장할 리스트
    epoch_losses = [] 
    
    # 4. 학습 루프
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        loop = tqdm(data_loader, leave=True)
        total_loss = 0
        
        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
            loop.set_postfix(loss=losses.item())
        
        avg_loss = total_loss / len(data_loader)
        # 현재 에포크의 평균 손실을 리스트에 추가
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 5. 모델 저장
    torch.save(model.state_dict(), config.MODEL_CHECKPOINT)
    print(f"✅ Model saved to {config.MODEL_CHECKPOINT}")

    # 학습 완료 후 손실 그래프 생성 및 저장
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.NUM_EPOCHS + 1), epoch_losses, marker='o', linestyle='-')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.xticks(range(1, config.NUM_EPOCHS + 1))
    
    loss_curve_path = config.OUTPUT_DIR / "training_loss_curve.png"
    plt.savefig(loss_curve_path)
    print(f"📈 Loss curve saved to {loss_curve_path}")

if __name__ == '__main__':
    main()