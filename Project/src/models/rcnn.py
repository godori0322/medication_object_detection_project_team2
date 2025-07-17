import torch
import torch.nn as nn
from torchvision.models import resnet50

def rcnn(num_classes, pretrained=True):
    """
    R-CNN model implementation
    Traditional R-CNN with separate CNN for each region proposal
    """
    class RCNN(nn.Module):
        def __init__(self, num_classes, pretrained=True):
            super(RCNN, self).__init__()
            
            # CNN backbone (ResNet50)
            if pretrained:
                backbone = resnet50(weights='ResNet50_Weights.DEFAULT')
            else:
                backbone = resnet50(weights=None)
            
            # Remove final classification layer
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            
            # Classifier for object detection
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
            
            # Bounding box regressor
            self.bbox_regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes * 4)
            )
        
        def forward(self, x):
            # Extract features
            features = self.features(x)
            
            # Classification
            class_scores = self.classifier(features)
            
            # Bounding box regression
            bbox_pred = self.bbox_regressor(features)
            
            return class_scores, bbox_pred
    
    return RCNN(num_classes, pretrained)