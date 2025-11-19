import torch
import torch.nn as nn
from models.srresnet import SRResNet  

class SRGANGenerator(SRResNet):
    """SRResNet used as SRGAN Generator."""
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_residual_blocks=16, upscale_factor=4):
        super(SRGANGenerator, self).__init__(in_channels, out_channels, channels, num_residual_blocks, upscale_factor)

    def initialize_with_srresnet(self, srresnet_checkpoint_path):
        """Load pretrained SRResNet weights into this generator (except final layers)."""
        checkpoint = torch.load(srresnet_checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['model']
        own_state = self.state_dict()
        
        # Load matching parameters only
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
        print("Initialized SRGAN Generator with SRResNet weights.")