import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead

def ssd(num_classes, pretrained=True):
    # VGG16 backbone
    if pretrained:
        weights = 'SSD300_VGG16_Weights.DEFAULT'
    else:
        weights = None
    
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)
    
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = SSDClassificationHead(
        in_channels=[512, 1024, 512, 256, 256, 256],
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model