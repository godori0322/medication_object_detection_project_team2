import os
import json

import cv2
import numpy as np
from PIL import Image


import torch
from torch.utils.data import Dataset
from pathlib import Path

# 데이터 증강을 위한 Albumentations 라이브러리 (선택 사항)
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split


class PillDataset(Dataset):
    def __init__(self, image_files, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = image_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.image_dir, img_name)

        # OpenCV로 이미지 로딩
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        ann_dir = os.path.join(self.annotation_dir, f"{img_id.split('_')[0]}_json")
        bboxes = []
        labels = []

        for dirname in os.listdir(ann_dir):
            for file in os.listdir(os.path.join(ann_dir, dirname)):
                if file.endswith(".json") and img_id in file:
                    with open(os.path.join(ann_dir, dirname, file), "r") as f:
                        ann = json.load(f)
                        label = ann["categories"][0]["name"]
                        class_id = ann["annotations"][0]["category_id"]
                        if class_id == -1:
                            continue
                        x, y, bw, bh = ann["annotations"][0]["bbox"]
                        x_min, y_min = x, y
                        x_max, y_max = x + bw, y + bh
                        bboxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        # 변환된 bbox (pascal_voc) → YOLO 포맷으로 변환
        targets = []
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox
            bw = x_max - x_min
            bh = y_max - y_min
            x_c = x_min + bw / 2
            y_c = y_min + bh / 2
            x_c /= image.shape[2]  # width
            y_c /= image.shape[1]  # height
            bw /= image.shape[2]
            bh /= image.shape[1]
            targets.append([label, x_c, y_c, bw, bh])

        return image, torch.tensor(targets, dtype=torch.float32)

# --- 데이터 증강 및 변환 정의 ---
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.3),
        ToTensorV2() # 이미지를 PyTorch 텐서로 변환
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transform():
    return A.Compose([
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


if __name__ == '__main__':
    # --- 테스트 코드 ---
    # config.py에서 경로를 가져와 사용
    from config import TRAIN_IMAGE_DIR, TRAIN_ANNOTATION_DIR
    
    image_files = []
    for f in os.listdir(TRAIN_IMAGE_DIR):
        if f.endswith(".png"):
            image_files.append(f)

    train_images, val_images = train_test_split(image_files, test_size=0.1, random_state=42, shuffle=True)

    print(len(train_images), len(val_images))

    # # 1. 데이터셋 인스턴스 생성 (증강 적용)
    # dataset = PillDataset(
    #     image_files=[],
    #     image_dir=TRAIN_IMAGE_DIR,
    #     annotation_dir=TRAIN_ANNOTATION_DIR,
    #     transform=get_train_transform()
    # )
    
    # # 2. 데이터셋 길이 확인
    # print(f"Dataset size: {len(dataset)}")
    
    # # 3. 첫 번째 샘플 데이터 확인
    # if len(dataset) > 0:
    #     image, target = dataset[0]
    #     print("\n--- Sample 0 ---")
    #     print("Image shape:", image.shape)
    #     print("Target boxes:\n", target['boxes'])
    #     print("Target labels:\n", target['labels'])
    #     print("Image ID:", target['image_id'])
        
    #     # 바운딩 박스가 제대로 변환되었는지 확인
    #     assert target['boxes'].dtype == torch.float32, "Boxes dtype should be float32"
    #     assert target['labels'].dtype == torch.int64, "Labels dtype should be int64"
    #     # With ToTensorV2, the image tensor is of type torch.uint8 by default.
    #     # Let's normalize it to float and check the type.
    #     image = image.float() / 255.0
    #     assert image.dtype == torch.float32, f"Image tensor dtype is {image.dtype}, but expected float32"
        
    # print("\nDataset test completed successfully!")