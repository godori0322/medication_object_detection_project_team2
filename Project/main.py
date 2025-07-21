# src/main.py
from src import models
from src.train import train_model
from src.test import run_test
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
    train_loader, val_loader, test_loader, mappings = create_dataloaders(cfg)
    
    # 모델 객체 생성
    if cfg.model_type.lower() == 'yolo':
        model = models.yolo_v5(num_classes=cfg.num_classes)
    elif cfg.model_type.lower() == 'rcnn':
        model = models.faster_rcnn(num_classes=cfg.num_classes)
    elif cfg.model_type.lower() == 'ssd':
        model = models.ssd(num_classes=cfg.num_classes)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {cfg.model_type}")
    
    # 모델 학습
    trained_model = train_model(model, train_loader, val_loader, cfg)
    
    # 모델 성능 평가(mAP@50)
    # metrics = evaluate_map_50(trained_model, val_loader, cfg)
    # save_metric_result(metrics, cfg.output_dir / "metrics.csv")

    # test 데이터 기반으로 결과 예측
    run_test(trained_model, test_loader, cfg)

    print("\n✅ 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()

## 실행 예시(CLI에서)
## python src/main.py --model_type yolo --num_epochs 30 --lr 0.0005 --batch_size 64
