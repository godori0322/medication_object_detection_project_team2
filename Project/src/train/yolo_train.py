import os
from pathlib import Path
from ultralytics import YOLO
from ..utils.logger import create_experiment_dir

def train_yolo(model, cfg):
    # data.yaml 경로 설정
    data_yaml = cfg.data_dir / "data.yaml"

    # 실험 결과 저장용 디렉토리 생성 (output_dir 기반)
    experiment_dir = create_experiment_dir(cfg.output_dir, model.__class__.__name__)
    print(f"Experiment directory created at: {experiment_dir}")
    
    # cfg 객체의 output_dir 경로를 실험별 폴더로 교체
    cfg.output_dir = Path(experiment_dir)

    os.chdir(cfg.output_dir)  # 현재 작업 디렉토리를 output_dir로 변경(상대경로 오작동 방지)
    
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
