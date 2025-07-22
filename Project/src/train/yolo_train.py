import os
from pathlib import Path
from ultralytics import YOLO

def train_yolo(model, cfg):
    # data.yaml 경로 설정
    data_yaml = cfg.data_dir / "data.yaml"

    # 학습 실행
    model.train(
        data=str(data_yaml),
        epochs=cfg.num_epochs,
        imgsz=640,
        device=cfg.device,
        batch=cfg.batch_size,
        lr0=cfg.lr,
        project=str(cfg.output_dir),
        name=f"yolo_experiment"
    )

    # 모델 평가
    metrics = model.val()
    print(f"Evaluation metrics: {metrics}")
    
    print(f"YOLO 학습 완료")
    return model
