# src/yolo_evaluate.py
from ultralytics import YOLO
from config import get_config
import os
import csv

def evaluate_yolo(model_path: str, data_yaml: str, iou: float = 0.5, imgsz: int = 640):
    """
    Ultralytics YOLO 모델 성능 평가 (val set 기준)
    
    Args:
        model_path (str): best.pt 모델 가중치 경로
        data_yaml (str): 데이터셋 경로를 포함한 data.yaml 파일 경로
        imgsz (int): 평가 시 이미지 크기 (default=640)
        iou (float): IoU threshold for NMS
    
    Returns:
        dict: 평가 메트릭 결과 (precision, recall, mAP50, mAP50-95)
    """
    cfg = get_config()

    # 모델 로드
    model = YOLO(model_path)

    # 검증
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=cfg.confidence_threshold,
        iou=iou,
        device=cfg.device,
        split='val',
        save_json=False,
        verbose=True
    )

    results = {
        "precision": metrics.results_dict.get("metrics/precision(B)", 0.0),
        "recall": metrics.results_dict.get("metrics/recall(B)", 0.0),
        "mAP50": metrics.results_dict.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)
    }
    return results


if __name__ == "__main__":
    cfg = get_config()

    # 모델 경로와 데이터셋 yaml 경로
    model_path = os.path.join(cfg.output_dir, "best.pt")
    data_yaml = cfg.data_yaml

    # 평가 실행
    results = evaluate_yolo(model_path, data_yaml)

    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # 결과 저장 (CSV)
    os.makedirs(cfg.output_dir, exist_ok=True)
    csv_path = os.path.join(cfg.output_dir, "evaluation_results.csv")
    with open(csv_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "mAP50", "mAP50-95"])
        writer.writerow([
            results["precision"],
            results["recall"],
            results["mAP50"],
            results["mAP50-95"]
        ])

    print(f"[✓] Results saved to {csv_path}")