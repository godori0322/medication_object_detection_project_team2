#src/yolo_test.py

import os
import csv
import yaml
import json

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
            
            class_name = class_names[int(cls)]
            category_id = name_to_id_map[class_name]

            submission_rows.append([
                annotation_id,
                image_id,
                category_id,
                int(xmin),
                int(ymin),
                int(bbox_w),
                int(bbox_h),
                round(float(score), 4)
            ])
            annotation_id += 1

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'])
        writer.writerows(submission_rows)

    print(f"✅ Submission CSV saved at: {output_csv_path}")

def run_test_yolo(model, cfg):
    data_dir = cfg.test_image_dir
    output_dir = cfg.output_dir
    save_dir = os.path.join(output_dir, 'yolo_test')
    os.makedirs(save_dir, exist_ok=True)
    
    # 클래스 이름 로드
    data_yaml_path = cfg.data_dir / "data.yaml"
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']
    
    # 매핑 정보 로드 및 역매핑 생성
    mappings_path = "Project/data_csv/mappings.json"
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    name_to_id_map = {name: cat_id for cat_id, name in mappings['category_id_map'].items()}

    # 예측 수행 및 결과 저장
    results = model.predict(
        source=data_dir,
        imgsz=640,
        conf=cfg.confidence_threshold,
        save=False,
        save_txt=False,
        save_conf=False,
        device=cfg.device
    )
    print(f"YOLO test completed. Results saved to {save_dir}")

    # submission.csv 생성
    output_csv_path = os.path.join(save_dir, 'submission.csv')
    create_submission_csv(results, output_csv_path, class_names, name_to_id_map)