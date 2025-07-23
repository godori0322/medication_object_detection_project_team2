# src/test.py

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import csv

def run_test(trained_model, test_loader, cfg):
    # YOLO 모델은 자체 평가 루틴(val())을 사용하므로 여기서 평가하지 않음
    if getattr(trained_model, 'is_yolo', False):
        print("[i] YOLO 모델은 run_test를 건너뜁니다(자체 평가 루틴 사용).")
        return
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
            image_id_str = image_ids[idx]["image_name"]
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

    # Save submission CSV
    df = pd.DataFrame(submission, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    df.to_csv(result_dir / "submission.csv", index=False)
    print(f"[✓] submission.csv 저장 완료: {result_dir / 'submission.csv'}")
