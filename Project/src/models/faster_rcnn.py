import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def faster_rcnn(num_classes, backbone='resnet50', pretrained=True):
    
    if backbone == 'resnet18':
        weights = 'FasterRCNN_ResNet18_FPN_Weights.DEFAULT' if pretrained else None
        model = torchvision.models.detection.fasterrcnn_resnet18_fpn(weights=weights)
    elif backbone == 'resnet50':
        weights = 'FasterRCNN_ResNet50_FPN_Weights.DEFAULT' if pretrained else None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    elif backbone == 'vgg16':
        weights = 'FasterRCNN_VGG16_Weights.DEFAULT' if pretrained else None
        model = torchvision.models.detection.fasterrcnn_vgg16(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model