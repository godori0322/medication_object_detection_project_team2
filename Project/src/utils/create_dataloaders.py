import os
from src.config import TRAIN_IMAGE_DIR, TRAIN_ANNOTATION_DIR

from src.dataset import PillDataset, get_train_transform
from src.dataloader import getPillDataloader

from sklearn.model_selection import train_test_split


def create_dataloaders():
    print("Loading data...")
    image_files = []
    for f in os.listdir(TRAIN_IMAGE_DIR):
        if f.endswith(".png"):
            image_files.append(f)

    train_images, val_images = train_test_split(image_files, test_size=0.1, random_state=42, shuffle=True)
    
    train_dataset = PillDataset(
        image_files=train_images,
        image_dir=TRAIN_IMAGE_DIR,
        annotation_dir=TRAIN_ANNOTATION_DIR,
        transform=get_train_transform()
    )
    val_dataset = PillDataset(
        image_files=val_images,
        image_dir=TRAIN_IMAGE_DIR,
        annotation_dir=TRAIN_ANNOTATION_DIR,
    )

    train_loader = getPillDataloader(train_dataset, True)
    val_loader = getPillDataloader(val_dataset)

    return train_loader, val_loader