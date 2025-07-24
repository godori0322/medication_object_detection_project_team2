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
