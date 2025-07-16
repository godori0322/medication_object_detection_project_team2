import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class PillDataset(Dataset):
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
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 이미지 파일명은 JSON 파일명과 동일하다고 가정 (확장자만 다름)
        image_filename = json_path.stem + '.png' # 또는 '.jpg' 등
        image_path = self.image_dir / image_filename
        image = Image.open(image_path).convert("RGB")
        
        boxes = []
        labels = []
        
        # JSON 파일 내의 'annotations' 리스트에서 정보 추출
        for ann in data.get('annotations', []):
            bbox = ann['bbox']
            # COCO 형식 [x, y, w, h] -> PyTorch 형식 [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        # 데이터를 torch.Tensor로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # image_id는 파일명에서 추출하거나 인덱스를 사용할 수 있습니다.
        target["image_id"] = torch.tensor([idx])

        # 바운딩 박스가 없는 경우, 빈 텐서를 할당하여 에러 방지
        if len(boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        # transforms 유무와 관계없이 이미지를 항상 Tensor로 변환
        # (간단한 ToTensor 역할)
        image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # 만약 복잡한 transforms가 있다면 여기서 추가 적용 가능
        # if self.transforms:
        #     image, target = self.transforms(image, target)

        return image, target