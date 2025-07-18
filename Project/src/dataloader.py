import os
from sklearn.model_selection import train_test_split
import config

from torch.utils.data import DataLoader

from dataset import PillDataset, get_train_transform


def collate_fn(batch):
    return tuple(zip(*batch))

def getPillDataloader(dataset, shuffle=False):
    # 데이터셋 및 데이터로더
    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0 # CPU 코어 수에 맞게 조절
    )

    return data_loader


if __name__ == '__main__':
    image_files = []
    for f in os.listdir(config.TRAIN_IMAGE_DIR):
        if f.endswith(".png"):
            image_files.append(f)

    train_images, val_images = train_test_split(image_files, test_size=0.1, random_state=42, shuffle=True)

    
    train_dataset = PillDataset(
        image_files=train_images,
        image_dir=config.TRAIN_IMAGE_DIR,
        annotation_dir=config.TRAIN_ANNOTATION_DIR,
        transform=get_train_transform()
    )
    val_dataset = PillDataset(
        image_files=val_images,
        image_dir=config.TRAIN_IMAGE_DIR,
        annotation_dir=config.TRAIN_ANNOTATION_DIR,
    )

    train_dl = getPillDataloader(train_dataset)
    val_dl = getPillDataloader(val_dataset)
    print("Data loader created successfully.")
    
    # 데이터로더에서 배치 데이터 확인
    for i, (images, targets) in enumerate(train_dl):
        print(f"\n--- Batch {i+1} ---")
        print(f"Number of images in batch: {len(images)}")
        print(f"Number of targets in batch: {len(targets)}")
        
        # 첫 번째 이미지와 타겟 정보 확인
        print("Image shape:", images[0].shape)
        print("Target:", targets[0])
        
        if i == 0: # 첫 번째 배치만 확인하고 종료
            break