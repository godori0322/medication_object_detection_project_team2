
#src/transforms.py
from albumentations.pytorch import ToTensorV2
import albumentations as A

def get_train_transforms():
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
        #A.HorizontalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.5),
        # 추가 증강: A.ShiftScaleRotate(), A.Blur(), A.CoarseDropout() 등
    ], bbox_params=A.BboxParams(
        format='pascal_voc',             # 'pascal_voc' 형식: [xmin, ymin, xmax, ymax]
        label_fields=['category_ids']    # 바운딩박스와 함께 라벨 함께 변환
    ))

def get_valid_transforms():
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
        # 검증용: 증강 없이 정규화/텐서 변환만 데이터셋클래스내에서
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['category_ids']
    ))

def get_test_transforms():
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])