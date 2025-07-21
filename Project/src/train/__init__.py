from .pytorch_train import train_pytorch, create_dataloaders
from .yolo_train import train_yolo

def train_model(model, train_loader, val_loader, cfg):
    # 모델 객체로부터 적절한 학습기 연결
    model_name = model.__class__.__name__.lower()
    print(f"모델 이름: {model_name}")
    
    if 'yolo' in model_name:
        return train_yolo(model, cfg)
    else:
        return train_pytorch(model, train_loader, val_loader, cfg)