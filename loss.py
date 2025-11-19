import torch
import torch.nn as nn
from models.vgg import TruncatedVGG19

class FixedContentLoss(nn.Module):
    """
    Content loss using the simplified VGG feature extractor
    """
    def __init__(self):
        super().__init__()
        self.vgg = TruncatedVGG19(output_layer_idx=35)
        self.mse = nn.MSELoss()

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr).detach()  # detach targets
        return self.mse(sr_features, hr_features)

class TensorAdversarialLoss(nn.Module):
    """
    Adversarial loss that works with tensor targets instead of booleans
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_tensor):
        """
        Args:
            pred: Discriminator predictions
            target_tensor: Tensor of target labels (0.0 or 1.0)
        """
        return self.loss(pred, target_tensor)

