# src/dataset.py

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

# 데이터 증강을 위한 Albumentations 라이브러리 (선택 사항)
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PillDataset(Dataset):
    """
    개별 JSON 파일로 구성된 경구약제 데이터셋을 위한 커스텀 클래스
    """
    def __init__(self, image_dir, annotation_dir, transforms=None):
        """
        Args:
            image_dir (string): 이미지 파일들이 있는 디렉토리 경로
            annotation_dir (string): 개별 JSON 어노테이션 파일들이 있는 상위 디렉토리 경로
            transforms (callable, optional): 샘플에 적용될 전처리(transform)
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transforms = transforms
        
        # annotation_dir 및 모든 하위 디렉토리에서 .json 파일을 찾습니다.
        self.json_paths = sorted(list(self.annotation_dir.glob('**/*.json')))
        
    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        # 해당 인덱스의 JSON 파일 경로를 가져옵니다.
        json_path = self.json_paths[idx]
        
        # JSON 파일에서 어노테이션 정보 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 파일명은 JSON 파일명에서 확장자만 변경하여 구성
        image_filename = json_path.stem + '.png'
        image_path = self.image_dir / image_filename
        
        # 이미지를 Numpy 배열로 불러오기 (transforms 적용을 위해)
        image = np.array(Image.open(image_path).convert("RGB"))
        
        boxes = []
        labels = []
        
        # JSON 파일 내의 'annotations' 리스트에서 정보 추출
        for ann in data.get('annotations', []):
            bbox = ann.get('bbox')
            category_id = ann.get('category_id')
            
            if bbox and category_id is not None:
                # COCO 형식 [x, y, w, h] -> [xmin, ymin, xmax, ymax] 형식으로 변환
                xmin, ymin = bbox[0], bbox[1]
                xmax, ymax = xmin + bbox[2], ymin + bbox[3]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(category_id)

        target = {"boxes": boxes, "labels": labels}

        # --- 데이터 증강(Data Augmentation) 적용 부분 ---
        if self.transforms:
            # Albumentations 라이브러리를 사용할 경우
            transformed = self.transforms(image=image, bboxes=target['boxes'], labels=target['labels'])
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
            
            # 증강 후 바운딩 박스가 사라졌을 경우 처리
            if len(target['boxes']) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        else:
            # 기본 전처리 (Numpy -> Tensor)
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)

        # 바운딩 박스가 없는 이미지에 대한 예외 처리
        if len(target['boxes']) == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            
        # image_id 등 추가 정보 (필요시)
        target["image_id"] = torch.tensor([idx])
        
        return image, target