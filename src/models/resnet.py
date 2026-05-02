import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50WithExtras(nn.Module):
    def __init__(self, num_classes=2, in_chans=3, pretrained=True):
        super().__init__()
        self.in_chans = in_chans
        
        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
        
        # Handle extra input channels
        if in_chans != 3:
            self._modify_first_conv(in_chans)
        
        # Modify classifier (fc in ResNet)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def load_pretrained_weights(self, state_dict):
        """Load pretrained weights handling channel mismatch safely."""
        # Handle first conv layer for different input channels
        if self.in_chans != 3:
            # ResNet's first conv layer is 'conv1.weight'
            if 'conv1.weight' in state_dict:
                first_conv_weight = state_dict['conv1.weight']
                new_first_conv = torch.randn_like(first_conv_weight)
                new_first_conv[:, :3] = first_conv_weight  # Copy RGB weights
                
                # Initialize extra channels
                for i in range(3, self.in_chans):
                    new_first_conv[:, i] = first_conv_weight.mean(dim=1)
                state_dict['conv1.weight'] = new_first_conv
        
        # Load with strict=False to ignore any deep architectural mismatches safely
        self.load_state_dict(state_dict, strict=False)

    def _modify_first_conv(self, in_chans):
        """Modify first conv layer to handle extra channels."""
        original_conv = self.backbone.conv1
        
        # Create new conv layer with correct input channels
        new_conv = nn.Conv2d(
            in_chans, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights: copy RGB channels, average for extra channels
        with torch.no_grad():
            if in_chans >= 3:
                new_conv.weight[:, :3] = original_conv.weight
                # For extra channels, use average of RGB weights
                for i in range(3, in_chans):
                    new_conv.weight[:, i] = original_conv.weight.mean(dim=1)
            else:
                # If fewer than 3 channels, take first channels
                new_conv.weight = original_conv.weight[:, :in_chans]
        
        self.backbone.conv1 = new_conv
    
    def forward(self, x):
        return self.backbone(x)

class ResNet50Standard(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet50(weights=None)
        
        # Modify classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)