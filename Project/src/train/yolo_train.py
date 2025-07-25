import yaml
import os
from pathlib import Path
from ultralytics import YOLO

def train_yolo(model, cfg):
    # data.yaml 경로 설정
    data_yaml = cfg.data_dir / "data.yaml"
    
    custom_augmentation = get_default_augmentation()
    custom_augmentation['flipud'] = 0.1
    custom_augmentation['auto_augment'] = "randaugment"
    
    # 학습 실행
    if getattr(cfg, 'tune', False):
        # 하이퍼파라미터 진화(evolutionary search)
        model.tune(
            data=str(data_yaml),
            epochs=cfg.tune_epochs,
            imgsz=640,
            device=cfg.device,
            batch=cfg.batch_size,
            iterations=cfg.iterations, 
            project=str(cfg.output_dir),
            name="yolo_tune"
        )
        
        tune_root = Path(cfg.output_dir)
        tune_dirs = sorted(
            [p for p in tune_root.iterdir() if p.name.startswith("yolo_tune") and p.is_dir()],
            key=lambda p: p.stat().st_mtime
            )
        if not tune_dirs:
             raise FileNotFoundError("yolo_tune* 폴더를 찾을 수 없습니다.")
        best_tune_dir = tune_dirs[-1]
        
        print(best_tune_dir)

        hyp_file = best_tune_dir / "best_hyperparameters.yaml"
        with open(hyp_file, 'r', encoding='utf-8') as f:
            hyp_dict = yaml.safe_load(f)
        
        for k in list(custom_augmentation):
            if k in hyp_dict:
                custom_augmentation.pop(k)
        
        model.train(
            data=str(data_yaml),
            epochs=cfg.num_epochs,
            imgsz=640,
            device=cfg.device,
            batch=cfg.batch_size,
            project=str(cfg.output_dir),
            name="yolo_experiment",  
            patience=20,
            **hyp_dict,
            **custom_augmentation
        )
    else:
        model.train(
            data=str(data_yaml),
            epochs=cfg.num_epochs,
            imgsz=640,
            device=cfg.device,
            batch=cfg.batch_size,
            lr0=cfg.lr,
            lrf=cfg.lrf,
            optimizer=cfg.optimizer, # SGD, Adam, AdamW, NAdam, RAdam, RMSProp
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            project=str(cfg.output_dir),
            name=f"yolo_experiment",
            patience=20,
            **custom_augmentation
            )
        
    print(f"YOLO 학습 완료")
    return model


def get_default_augmentation():
    return {
        "hsv_h": 0.015,             # 0.0 - 1.0
        "hsv_s": 0.7,               # 0.0 - 1.0
        "hsv_v": 0.4,               # 0.0 - 1.0
        "degrees": 0.0,             # 0.0 - 180
        "translate": 0.1,           # 0.0 - 1.0
        "scale": 0.5,               # >=0.0
        "shear": 0.0,               # -180 - +180
        "perspective": 0.0,         # 0.0 - 0.001
        "flipud": 0.0,              # 0.0 - 1.0
        "fliplr": 0.5,              # 0.0 - 1.0
        "bgr": 0.0,                 # 0.0 - 1.0
        "mosaic": 1.0,              # 0.0 - 1.0
        "mixup": 0.0,               # 0.0 - 1.0
        "cutmix": 0.0,              # 0.0 - 1.0
        "copy_paste": 0.0,          # 0.0 - 1.0
        "copy_paste_mode": "flip",  # flip or mixup
        "auto_augment": None,       # randaugment, autoaugment 또는 augmix
        "erasing": 0.4,             # 0.0 - 0.9
    }
