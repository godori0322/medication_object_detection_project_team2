# src/test.py

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import csv

def f1_score(p, r):
    return 2 * p * r / (p + r + 1e-6)

def run_test(trained_model, test_loader, cfg):
    result_dir = Path(cfg.output_dir) / "test"
    pred_dir = result_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    trained_model.eval()

    submission = []
    annotation_id = 1

    # 성능 평가를 위한 mAP 메트릭 초기화
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

    for images, image_ids in tqdm(test_loader, desc="Testing"):
        images = images.to(cfg.device)
        outputs = trained_model(images)

        for idx, output in enumerate(outputs):
            image_id_str = str(image_ids[idx])
            image_id = int(Path(image_id_str).stem)

            boxes = output['boxes'].detach().cpu()
            scores = output['scores'].detach().cpu()
            labels = output['labels'].detach().cpu()

            # Draw and save box image
            img_with_boxes = draw_bounding_boxes(
                (images[idx].cpu() * 255).byte(), boxes, labels=[str(l.item()) for l in labels]
            )
            to_pil_image(img_with_boxes).save(pred_dir / f"{image_id:04d}.jpg")

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                submission.append([
                    annotation_id,
                    image_id,
                    label.item(),
                    int(x1), int(y1), int(w), int(h),
                    float(score)
                ])
                annotation_id += 1

        # mAP 메트릭 업데이트
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
        outputs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]
        metric.update(outputs_cpu, targets_cpu)

    # Save submission CSV
    df = pd.DataFrame(submission, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    df.to_csv(result_dir / "submission.csv", index=False)
    print(f"[✓] submission.csv 저장 완료: {result_dir / 'submission.csv'}")

    # Compute metrics
    results = metric.compute()
    precision = results["precision"][0].item()
    recall = results["recall"][0].item()
    f1 = f1_score(precision, recall)
    map50 = results["map_50"].item()
    map_all = results["map"].item()

    # Save metrics to CSV
    map_result_path = result_dir / "map_result.csv"
    with open(map_result_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mAP@0.5", map50])
        writer.writerow(["mAP@0.5:0.95", map_all])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        writer.writerow(["F1 Score", f1])
    print(f"[✓] map_result.csv 저장 완료: {map_result_path}")