# src/test.py

import os
from pathlib import Path
import torch
from tqdm import tqdm
import csv
import pandas as pd
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from src.utils.evaluater import evaluate_map_50
from src.dataset import get_test_dataloader


def run_test(trained_model, test_loader, cfg):
    result_dir = Path(cfg.output_dir) / "test"
    pred_dir = result_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    trained_model.eval()

    submission = []
    annotation_id = 1

    for images, image_ids in tqdm(test_loader, desc="Testing"):
        images = images.to(cfg.device)
        outputs = trained_model(images)

        for idx, output in enumerate(outputs):
            image_id_str = str(image_ids[idx])
            image_id = int(Path(image_id_str).stem)  # 이미지 파일명에서 숫자 추출

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

    # Save submission CSV
    df = pd.DataFrame(submission, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    df.to_csv(result_dir / "submission.csv", index=False)
    print(f"[✓] submission.csv 저장 완료: {result_dir / 'submission.csv'}")

    # mAP 평가
    metrics = evaluate_map_50(trained_model, test_loader, cfg)
    map_result_path = result_dir / "map_result.csv"
    with open(map_result_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    print(f"[✓] mAP 결과 저장 완료: {map_result_path}")