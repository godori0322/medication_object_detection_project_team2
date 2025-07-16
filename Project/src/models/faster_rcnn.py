import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def faster_rcnn(num_classes, pretrained=True):
    weights = 'FasterRCNN_ResNet50_FPN_Weights.DEFAULT' if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model