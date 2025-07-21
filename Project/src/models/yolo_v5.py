import torch

def yolo_v5(num_classes, pretrained=True):
    # torch.hub를 통해 YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 
                           model_name = "yolov5s", 
                           num_classes=num_classes, 
                           pretrained=pretrained)

    return model