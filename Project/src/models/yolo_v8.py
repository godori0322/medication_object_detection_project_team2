from ultralytics import YOLO

def yolo_v8(num_classes, pretrained=True):
    if pretrained:
        model_name = f'yolov8s.pt'
    else:
        model_name = f'yolov8s.yaml'

    model = YOLO(model_name)
    model.is_yolo = True
    
    return model