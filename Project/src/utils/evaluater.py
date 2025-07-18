# utils/evaluator.py

from mean_average_precision import MetricBuilder
import torch
from tqdm import tqdm

def evaluate_map_50(model, dataloader, cfg):
    model.eval()
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=cfg.num_classes)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(cfg.device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # 예측값 준비
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()

                # 정답 준비
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()

                if len(pred_boxes) > 0:
                    # mean_average_precision 요구 형식: [x, y, w, h, score, label]
                    pred = [
                        [x1, y1, x2 - x1, y2 - y1, score, label]
                        for (x1, y1, x2, y2), score, label in zip(pred_boxes, pred_scores, pred_labels)
                    ]
                else:
                    pred = []

                if len(gt_boxes) > 0:
                    gt = [
                        [x1, y1, x2 - x1, y2 - y1, label]
                        for (x1, y1, x2, y2), label in zip(gt_boxes, gt_labels)
                    ]
                else:
                    gt = []

                metric_fn.add(pred, gt)

    metrics = metric_fn.value(iou_thresholds=[0.5])
    map_50 = metrics['mAP']  # mean_average_precision 기준으로 mAP@0.5

    return {
        "mAP@0.5": round(map_50, 4)
    }