# src/main.py
from src.models import yolo_v5, yolo_v8, yolo_v11, faster_rcnn, ssd
from src.train import train_model
from src.test import run_test
from src.yolo_test import run_test_yolo
from src.config import get_config, get_device
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

    # 모델 객체 생성(모델 변경을 위해 이 부분을 수정하세요)
    model = yolo_v5(num_classes=cfg.num_classes, pretrained=True)
    # model = yolo_v8(num_classes=cfg.num_classes, pretrained=True)
    # model = yolo_v11(num_classes=cfg.num_classes, pretrained=True)
    # model = faster_rcnn(num_classes=cfg.num_classes, backbone='resnet50', pretrained=True) # resnet18, resnet50, vgg16 중 선택 가능
    
    # 모델 학습(YOLO 모델일 때와 아닐 때 파이프라인 분리) & outputs 디렉토리에 결과 저장
    trained_model = train_model(model, train_loader, val_loader, cfg)
    
    model_name = trained_model.__class__.__name__.lower()
    # YOLO 모델은 ultralytics 라이브러리로 성능평가 및 시각화까지
    if model_name == "yolo":
        run_test_yolo(trained_model, cfg)

    # 다른 모델은 직접 예측 결과를 저장하는 방식으로 진행
    else:
        # test 데이터 기반으로 결과 예측
        run_test(trained_model, test_loader, cfg)

        # 모델 성능 평가(mAP@50)
        # metrics = evaluate_map_50(trained_model, val_loader, cfg)
        # save_metric_result(metrics, cfg.output_dir / "metrics.csv")

    print("\n✅ 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()

## 실행 예시(CLI에서)
## python src/main.py --model_type yolo --num_epochs 30 --lr 0.0005 --batch_size 64
