# src/evaluate_final.py

# SGD, AdamW 두 옵티마이저 기준으로 튜닝된 하이퍼파라미터를 적용하여 학습 완료된 모델 2개에 대한 성능 평가 진행
# (mAP@50, mAP@50-95, Precision, Recall, F1-score)
# 이후 해당 결과 csv 파일로 저장, test 이미지에 대한 예측 결과를 csv 파일로 저장
# 이후 kaggle 제출용 submission.csv 파일 생성

import os
import pandas as pd
from ultralytics import YOLO
from yolo_test import run_test_yolo
from pathlib import Path
import argparse
from config import get_config

def evaluate_and_save(model_path, cfg, model_name):
    model = YOLO(model_path)
    
    # 모델 테스트 및 submission.csv 저장
    run_test_yolo(model, cfg)

    # 성능 지표 추출
    metrics = model.val(data=cfg.data_dir / "data.yaml", imgsz=640, device=cfg.device)
    
    # 평가 결과 저장
    metrics_dict = {
        'model': model_name,
        'mAP50': round(metrics.box.map50, 4),
        'mAP50-95': round(metrics.box.map, 4),
        'Precision': round(metrics.box.precision, 4),
        'Recall': round(metrics.box.recall, 4),
        'F1': round(2 * metrics.box.precision * metrics.box.recall / (metrics.box.precision + metrics.box.recall + 1e-6), 4)
    }
    
    print(f"📊 Evaluation done for {model_name}: {metrics_dict}")
    return metrics_dict

def main(sgd_model_path, adamw_model_path, cfg):
    results = []
    
    results.append(evaluate_and_save(sgd_model_path, cfg, "SGD"))
    results.append(evaluate_and_save(adamw_model_path, cfg, "AdamW"))
    
    # 평가 결과 CSV 저장
    metrics_df = pd.DataFrame(results)
    metrics_csv_path = cfg.output_dir / "final_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✅ Final evaluation metrics saved at: {metrics_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate final YOLO models")
    parser.add_argument("--sgd_model", type=str, required=True, help="Absolute path to SGD model (e.g., /path/to/sgd.pt)")
    parser.add_argument("--adamw_model", type=str, required=True, help="Absolute path to AdamW model (e.g., /path/to/adamw.pt)")
    args = parser.parse_args()

    # 설정 파일 로드
    cfg = get_config()

    # 모델 경로
    sgd_model_path = Path(args.sgd_model)
    adamw_model_path = Path(args.adamw_model)

    main(sgd_model_path, adamw_model_path, cfg)

# 실행 예시
# python evaluate_final.py --sgd_model /path/to/sgd.pt --adamw_model /path/to/adamw.pt