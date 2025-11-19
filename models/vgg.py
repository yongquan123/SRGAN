import torch
import torch.nn as nn
from torchvision import models


class TruncatedVGG19(nn.Module):
    """
    Simplified VGG feature extractor to avoid create_feature_extractor issues
    """
    def __init__(self, output_layer_idx=35):
        super().__init__()
        
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg19.features.children())[:output_layer_idx+1])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # SRGAN paper scaling
        self.vgg_scale = 1.0 / 12.75
        
        self.eval()

    def forward(self, x):
        """
        Input x: [-1, +1]
        Convert to: [0,1], then normalize, then extract features.
        """
        # [-1,1] -> [0,1]
        x = (x + 1) / 2

        # Imagenet normalization
        x = (x - self.mean) / self.std

        # Extract features and apply SRGAN scaling
        features = self.features(x)
        return features * self.vgg_scale


