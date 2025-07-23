from ultralytics import YOLO

def yolo_v5(num_classes, pretrained=True):
    # ultralytics 라이브러리를 통해 YOLOv5 모델 로드
    model = YOLO('yolov5s.pt')
    model.is_yolo = True
    
    return model