from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.train import train_model
from src import models, config
from src.dataloader import PillDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def create_dataloaders():
    print("Loading data...")
    full_dataset = PillDataset(config.TRAIN_IMAGE_DIR, config.TRAIN_ANNOTATION_DIR)
    
    print(f"Total dataset size: {len(full_dataset)}")
    
    # 데이터셋이 비어있는지 확인
    if len(full_dataset) == 0:
        raise ValueError(f"Dataset is empty! Check paths:\n"
                        f"Image dir: {config.TRAIN_IMAGE_DIR}\n"
                        f"Annotation dir: {config.TRAIN_ANNOTATION_DIR}")
    
    # 데이터 분할
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)

    return train_loader, val_loader

def main():
    
    # 데이터로더 생성
    train_loader, val_loader = create_dataloaders()
    
    # 모델 객체 생성
    model = models.yolo_v5(num_classes=config.NUM_CLASSES)
    
    # 학습 설정
    training_config = {
        'num_epochs': 1,
        'optimizer': {
            'type': 'ADAM',
            'learning_rate': 0.005,
            'weight_decay': 0.0005,
            'momentum': 0.9
        }
    }
    
    # 모델 학습
    train_model(model, train_loader, val_loader, training_config)
    
    print("\n✅ 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()