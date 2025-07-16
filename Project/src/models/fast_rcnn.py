import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

def fast_rcnn(num_classes, pretrained=True):
    """
    Fast R-CNN model implementation
    Uses pre-computed region proposals (no RPN)
    """
    class FastRCNN(nn.Module):
        def __init__(self, num_classes, pretrained=True):
            super(FastRCNN, self).__init__()
            
            # CNN backbone (ResNet50 without final layers)
            if pretrained:
                backbone = resnet50(weights='ResNet50_Weights.DEFAULT')
            else:
                backbone = resnet50(weights=None)
            
            # Remove final classification layers
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            
            # ROI pooling
            self.roi_pool = torchvision.ops.RoIPool(output_size=(7, 7), spatial_scale=1.0)
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
            
            # Bbox regressor
            self.bbox_regressor = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes * 4)
            )
        
        def forward(self, images, rois=None):
            # Extract features
            features = self.backbone(images)
            
            if rois is not None:
                # Apply ROI pooling
                pooled_features = self.roi_pool(features, rois)
                
                # Classification and bbox regression
                class_scores = self.classifier(pooled_features)
                bbox_pred = self.bbox_regressor(pooled_features)
                
                return class_scores, bbox_pred
            else:
                # Just return features for region proposal
                return features
    
    return FastRCNN(num_classes, pretrained)