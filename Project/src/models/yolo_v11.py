from ultralytics import YOLO

def yolo_v11(num_classes, pretrained=True):
    if pretrained:
        model_name = f'yolo11s.pt'
    else:
        model_name = f'yolo11s.yaml'

    model = YOLO(model_name)
    
    return model