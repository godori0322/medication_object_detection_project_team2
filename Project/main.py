# src/main.py
from src.train import train_model
from src import models
from src.config import get_config, get_device
from src.utils.evaluater import evaluate_map_50
from src.utils.logger import save_metric_result
from src.dataloader import create_dataloaders

def main():
    # 현재 사용 중인 device 확인
    device = get_device()
    print(f'현재 사용 중인 device: {device}')
    
    # config로 경로 및 하이퍼파라미터 설정
    cfg = get_config()

    # 데이터로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # 모델 객체 생성
    model = models.yolo_v5(num_classes=cfg.num_classes)
    
    # 모델 학습
    trained_model, checkpoint_path = train_model(model, train_loader, val_loader, cfg)
    
    # 모델 성능 평가(mAP@50)
    # metrics = evaluate_map_50(trained_model, val_loader, cfg)
    # save_metric_result(metrics, cfg.output_dir / "metrics.csv")

    print("\n✅ 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()

## 실행 예시(CLI에서)
## python src/main.py --num_epochs 30 --lr 0.0005 --batch_size 64