import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

class EfficientNetV2MWithExtras(nn.Module):
    def __init__(self, num_classes=2, in_chans=3, pretrained=True):
        super().__init__()
        self.in_chans = in_chans
        
        # Load pretrained EfficientNetV2-M
        if pretrained:
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_v2_m(weights=weights)
        else:
            self.backbone = efficientnet_v2_m(weights=None)
        
        # Handle extra input channels
        if in_chans != 3:
            self._modify_first_conv(in_chans)
        
        # Modify classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def load_pretrained_weights(self, state_dict):
        """Load pretrained weights handling channel mismatch."""
        # Handle first conv layer for different input channels
        if self.in_chans != 3:
            first_conv_weight = state_dict['features.0.0.weight']
            new_first_conv = torch.randn_like(first_conv_weight)
            new_first_conv[:, :3] = first_conv_weight  # Copy RGB weights
            # Initialize extra channels
            for i in range(3, self.in_chans):
                new_first_conv[:, i] = first_conv_weight.mean(dim=1)
            state_dict['features.0.0.weight'] = new_first_conv
        
        # Load with strict=False to ignore extra channel mismatches
        self.load_state_dict(state_dict, strict=False)

    def _modify_first_conv(self, in_chans):
        """Modify first conv layer to handle extra channels."""
        original_conv = self.backbone.features[0][0]
        
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
        
        self.backbone.features[0][0] = new_conv
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetV2MStandard(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
            self.model = efficientnet_v2_m(weights=weights)
        else:
            self.model = efficientnet_v2_m(weights=None)
        
        # Modify classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)