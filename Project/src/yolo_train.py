from ultralytics import YOLO
from pathlib import Path
from .config import DATA_DIR

def train_yolo(cfg):
    # data.yaml 경로 설정
    data_yaml = DATA_DIR / "data.yaml"

    # YOLO 모델 로드
    model = YOLO(f'{cfg.yolo_model_name}.pt')
    
    
    # 학습 실행
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.num_epochs,
        imgsz=640,
        device=cfg.device,
        batch=cfg.batch_size,
        lr0=cfg.lr,
        project=str(cfg.output_dir),
        name=f"{cfg.yolo_model_name}_experiment"
    )
    
    print(f"YOLO 학습 완료")
    return model, results

if __name__ == "__main__":
    import sys
    sys.argv = ['']
    from .config import get_config
    
    cfg = get_config()
    model, results = train_yolo(cfg)
