import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.video as video_models
from torchvision.models.video import R3D_18_Weights

class SurfManeuverModel(nn.Module):
    def __init__(self, num_classes=7, mode='dev', dropout_rate=0.5, freeze_backbone=True):
        super(SurfManeuverModel, self).__init__()
        self.mode = mode
        
        # Use 3D ResNet (R3D-18) for spatio-temporal feature extraction
        # This processes the entire sequence as a single 3D volume
        self.model = video_models.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        
        # Modify the first conv layer based on input channels (grayscale vs RGB)
        if self.mode == 'dev':
            # For grayscale, use 1 input channel
            # We need to modify the first layer of the model
            old_conv = self.model.stem[0]
            self.model.stem[0] = nn.Conv3d(
                1,  # 1 input channel for grayscale
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            
            # Copy weights from the pretrained model for the first channel
            with torch.no_grad():
                self.model.stem[0].weight[:, 0:1, :, :, :] = old_conv.weight[:, 0:1, :, :, :].clone()
        
        # Freeze most of the backbone layers if specified
        if freeze_backbone:
            # Freeze all layers except the final fully connected layer
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # Don't freeze the fully connected layer
                    param.requires_grad = False
                    
            print("Model backbone frozen. Only training the classifier head.")
        
        # Modify the classification head with dropout for better generalization
        in_features = self.model.fc.in_features
        
        # Replace with a simpler classifier network
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass for the 3D CNN model
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
            hidden: Ignored parameter (kept for backward compatibility)
            
        Returns:
            output: Classification scores
            None: Placeholder to maintain compatibility with previous interface
        """
        # Reshape input from [batch_size, seq_len, channels, height, width] 
        # to [batch_size, channels, seq_len, height, width] for 3D CNN
        batch_size, seq_len, channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        
        # Forward pass through the 3D CNN
        output = self.model(x)
        
        # Return output and None to maintain compatibility with LSTM interface
        # This allows us to use the same inference code without changes
        return output, None
