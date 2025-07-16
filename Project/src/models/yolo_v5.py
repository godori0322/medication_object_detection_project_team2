from ultralytics import YOLO

def yolo_v5(num_classes, pretrained=True):
    if pretrained:
        model = YOLO('yolov5s.pt')
    else:
        model = YOLO('yolov5s.yaml')
    
    # Modify for custom number of classes
    if hasattr(model.model, 'model') and hasattr(model.model.model[-1], 'nc'):
        model.model.model[-1].nc = num_classes
    
    return model