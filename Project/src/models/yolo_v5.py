import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def yolo_v5(num_classes, pretrained=True):
    if pretrained:
        weights = 'RetinaNet_ResNet50_FPN_Weights.DEFAULT'
    else:
        weights = None
    
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)
    
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model