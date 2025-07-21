from ultralytics import YOLO
from pathlib import Path
from ..config import DATA_DIR

def train_yolo(model, cfg):
    # data.yaml 경로 설정
    data_yaml = DATA_DIR / "data.yaml"
    
    # 학습 실행
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.num_epochs,
        imgsz=640,
        device=cfg.device,
        batch=cfg.batch_size,
        lr0=cfg.lr,
        project=str(cfg.output_dir),
        name=f"yolo_experiment"
    )
    
    print(f"YOLO 학습 완료")
    return model, results
