#src/yolo_test.py

import os
import csv

def create_submission_csv(results, output_csv_path):
    annotation_id = 1
    submission_rows = []

    for result in results:
        filename = os.path.basename(result.path)
        # 이미지 파일명에서 숫자만 추출 (예: '123.jpg' -> 123)
        image_id = int(''.join(filter(str.isdigit, filename)))

        boxes = result.boxes.xyxy.cpu().numpy()    # (xmin, ymin, xmax, ymax)
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            submission_rows.append([
                annotation_id,
                image_id,
                int(cls),
                int(xmin),
                int(ymin),
                int(bbox_w),
                int(bbox_h),
                round(float(score), 4)
            ])
            annotation_id += 1

    # CSV 저장
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
        writer.writerows(submission_rows)

    print(f"✅ Submission CSV saved at: {output_csv_path}")

def run_test_yolo(model, cfg):
    data_dir = cfg.test_image_dir
    output_dir = cfg.output_dir
    save_dir = os.path.join(output_dir, 'yolo_test')
    os.makedirs(save_dir, exist_ok=True)

    # 예측 수행 및 결과 저장
    results = model.predict(
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

    # submission.csv 생성
    output_csv_path = os.path.join(save_dir, 'submission.csv')
    create_submission_csv(results, output_csv_path)