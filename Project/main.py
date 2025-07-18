# src/main.py

from src.train import create_dataloaders, train_model
from src import models
from src.config import get_config, get_device
from src.utils import visualizer

def main():
    # 현재 사용 중인 device 확인
    device = get_device()
    print(f'현재 사용 중인 device: {device}')
    
    # config로 경로 및 하이퍼파라미터 설정
    cfg = get_config()

    # 데이터로더 생성
    train_loader, val_loader = create_dataloaders(cfg)
    
    # 모델 객체 생성
    model = models.yolo_v5(num_classes=cfg.num_classes)
    
    # 모델 학습
    trained_model, checkpoint_path = train_model(model, train_loader, val_loader, cfg)
    
    # 모델 결과 시각화
    

    print("\n✅ 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()

## 실행 예시(CLI에서)
## python src/main.py --num_epochs 30 --lr 0.0005 --batch_size 64