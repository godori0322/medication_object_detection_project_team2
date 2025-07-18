import os
from config import get_config

from src.dataset import PillDataset, get_train_transform
from src.dataloader import getPillDataloader

from sklearn.model_selection import train_test_split


config = get_config()

def create_dataloaders():
    print("Loading data...")
    image_files = []
    for f in os.listdir(config.train_image_dir):
        if f.endswith(".png"):
            image_files.append(f)

    train_images, val_images = train_test_split(image_files, test_size=0.1, random_state=42, shuffle=True)
    
    train_dataset = PillDataset(
        image_files=train_images,
        image_dir=config.train_image_dir,
        annotation_dir=config.annotation_dir,
        transform=get_train_transform()
    )
    val_dataset = PillDataset(
        image_files=val_images,
        image_dir=config.train_image_dir,
        annotation_dir=config.annotation_dir,
    )

    train_loader = getPillDataloader(train_dataset, True)
    val_loader = getPillDataloader(val_dataset)

    return train_loader, val_loader