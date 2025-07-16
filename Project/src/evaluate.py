# src/evaluate.py

import torch
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

from . import config
from .model import get_detection_model

def main():
    device = torch.device(config.DEVICE)
    
    # 1. 모델 불러오기
    model = get_detection_model(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # 2. 테스트 이미지 목록
    test_image_files = [f for f in os.listdir(config.TEST_IMAGE_DIR) if f.endswith('.png')]
    
    results = []
    annotation_id_counter = 1

    print("🔍 Starting evaluation...")
    with torch.no_grad():
        for image_file in tqdm(test_image_files, desc="Evaluating"):
            image_path = os.path.join(config.TEST_IMAGE_DIR, image_file)
            image = Image.open(image_path).convert("RGB")
            image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
            
            prediction = model(image_tensor)
            
            image_id = int(os.path.splitext(image_file)[0])
            boxes = prediction[0]['boxes']
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']
            
            for box, label, score in zip(boxes, labels, scores):
                if score > config.CONFIDENCE_THRESHOLD:
                    x, y, w, h = box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()
                    results.append({
                        'annotation_id': annotation_id_counter,
                        'image_id': image_id, 'category_id': label.item(),
                        'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
                        'score': score.item()
                    })
                    annotation_id_counter += 1

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"🎉 Submission file created at {config.SUBMISSION_FILE}")

if __name__ == '__main__':
    main()