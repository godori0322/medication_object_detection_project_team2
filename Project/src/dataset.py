# src/dataset
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset


def coco_to_voc(boxes):
    if len(boxes) == 0:
        return boxes
    return np.array([[x, y, x + w, y + h] for x, y, w, h in boxes])


def safe_tensor(data, shape, dtype):
    return torch.as_tensor(data, dtype=dtype) if len(data) else torch.zeros(shape, dtype=dtype)



#datasets/pill_dataset.py
# YOLO / FastRCNN 등의 구조에 적합한 클래스형태
class PillDataset(Dataset):
    def __init__(self, img_dir, labels_df, mappings, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.labels_df = labels_df
        self.image_id_map = mappings.get('image_id_map', {})
        self.img_ids = self.labels_df['image_id'].unique()
        self.records_dict = labels_df.groupby('image_id')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image_id = self.img_ids[idx]
        file_name = self.image_id_map.get(image_id)
        img_path = os.path.join(self.img_dir, file_name)
        image_np = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        # 해당 이미지의 모든 객체 라벨
        records = self.records_dict.get_group(image_id)
        boxes_coco = records[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].values
        labels = records['category_id'].values

        # Albumentations transform이 있으면 bbox_voc 변환 및 동기 처리
        if self.transforms is not None:
            boxes_voc = coco_to_voc(boxes_coco)
            transformed = self.transforms(
                image=image_np,
                bboxes=boxes_voc.tolist(),
                category_ids=labels.tolist()
            )
            image_tensor = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['category_ids']
        else:
            image_tensor = torch.as_tensor(image_np).permute(2, 0, 1).float() / 255.
            boxes = boxes_coco # 원본 bbox 형식 유지

        # tensor 안전하게 변환
        boxes = safe_tensor(boxes, (0, 4), torch.float32)
        labels = safe_tensor(labels, (0,), torch.int64)

        target = {
            "boxes" : boxes,
            "labels" : labels,
            "image_id" : torch.tensor([image_id]),
        }

        return image_tensor, target



class PillTestDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        """
        테스트 이미지셋 전용 Dataset 클래스 (라벨 없음)

        Args:
            img_dir (str): 이미지 경로 디렉토리
            transforms: Albumentations 이미지 변환
        """
        self.img_dir = img_dir
        self.transforms = transforms
        self.file_names = sorted(os.listdir(img_dir))  # 파일명 기준으로 처리

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, file_name)

        image_np = np.array(Image.open(img_path).convert('RGB'))

        if self.transforms is not None:
            image_tensor = self.transforms(image=image_np)['image']
        else:
            image_tensor = torch.as_tensor(image_np).permute(2, 0, 1).float() / 255.

        target = {
            "image_name": file_name
        }

        return image_tensor, target
