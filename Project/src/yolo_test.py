#src/yolo_test.py

import os
import numpy as np

def test_yolo(model, cfg):
    data_dir = cfg.test_image_dir
    output_dir = cfg.output_dir
    save_dir = os.path.join(output_dir, 'yolo_test')
    os.makedirs(save_dir, exist_ok=True)

    # 예측 수행 및 결과 저장
    model.predict(
        source=data_dir,
        imgsz=640,
        conf=cfg.confidence_threshold,
        save=True, 
        save_txt=True,
        save_conf=True, 
        save_dir=save_dir,
        device=cfg.device,
        project='results', 
        name='yolo_test',
        exist_ok=True
    )
    print(f"YOLO test completed. Results saved to {save_dir}")