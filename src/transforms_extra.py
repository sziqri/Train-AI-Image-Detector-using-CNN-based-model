import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from features import extract_ela_feature, extract_fft_feature, extract_prnu_feature
from PIL import Image
import numpy as np
import io

# --- 1. CORRECTED JPEG AUGMENTATION ---
class RandomJPEGCompression:
    """Corrected: This only performs compression and returns a PIL Image."""
    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img
            
        # Ensure image is RGB to prevent JPEG save errors
        if img.mode != 'RGB':
            img = img.convert('RGB')

        q = np.random.randint(self.quality_range[0], self.quality_range[1])
        output = io.BytesIO()
        img.save(output, 'JPEG', quality=q)
        output.seek(0)
        return Image.open(output)

# --- 2. SMART CROP & PAD (PRESERVES FORENSIC FEATURES) ---
class SmartCropPad:
    """
    Intelligently pads small images and crops large images.
    NO INTERPOLATION OR RESIZING IS USED. This preserves pixel-level forensic artifacts.
    """
    def __init__(self, size, train=True):
        self.size = size
        self.train = train

    def __call__(self, img):
        w, h = img.size
        
        # Calculate padding needed if image is smaller than target size
        pad_w = max(0, self.size - w)
        pad_h = max(0, self.size - h)
        
        # Apply reflection padding to avoid harsh black borders
        if pad_w > 0 or pad_h > 0:
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = TF.pad(img, padding, padding_mode='reflect')
        
        # Crop to the exact target size
        if self.train:
            return transforms.RandomCrop(self.size)(img)
        else:
            return transforms.CenterCrop(self.size)(img)

# --- 3. SYNCHRONIZED FORENSIC TRANSFORM ---
class ForensicTransform:
    def __init__(self, img_size: int, train: bool = True, 
                 use_fft: bool = True, use_ela: bool = True, use_prnu: bool = True):
        self.img_size = img_size
        self.train = train
        self.use_fft = use_fft
        self.use_ela = use_ela
        self.use_prnu = use_prnu
        
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _process_forensic_feature(self, feature):
        if feature is None: return None
        
        # 1. Convert numpy to tensor if needed
        if not isinstance(feature, torch.Tensor):
            feature = torch.from_numpy(feature).float()
            
        # 2. STRICT DIMENSION ENFORCEMENT
        # If the feature is 2D (H, W), add the channel dimension to make it 3D (1, H, W)
        if feature.dim() == 2:
            feature = feature.unsqueeze(0)
        
        # 3. Safety net: Ensure spatial synchronization to the target size
        if feature.shape[-2:] != (self.img_size, self.img_size):
            feature = F.interpolate(feature.unsqueeze(0), size=(self.img_size, self.img_size), 
                                    mode='bilinear', align_corners=False).squeeze(0)
        return feature

    def __call__(self, img):
        # Fallback safety: Ensure the image is correctly sized before feature extraction
        if img.size != (self.img_size, self.img_size):
            img = SmartCropPad(self.img_size, train=False)(img)
        
        img_tensor = self.final_transform(img)
        
        # Apply Random Erasing to the tensor (Training only)
        if self.train:
            img_tensor = transforms.RandomErasing(p=0.2)(img_tensor)
        
        extra_channels = []
        
        # Extract features from the PRISTINE 384x384 PIL Image
        if self.use_fft:
            extra_channels.append(self._process_forensic_feature(extract_fft_feature(img)))
        if self.use_ela:
            extra_channels.append(self._process_forensic_feature(extract_ela_feature(img)))
        if self.use_prnu:
            extra_channels.append(self._process_forensic_feature(extract_prnu_feature(img)))
            
        if extra_channels:
            extra_tensor = torch.cat(extra_channels, dim=0)
            return torch.cat([img_tensor, extra_tensor], dim=0)
        
        return img_tensor
    
# --- 4. UPDATED BUILDER ---
def build_transforms_with_extras(img_size: int, train: bool = True, 
                                 use_fft: bool = True, use_ela: bool = True, use_prnu: bool = True,
                                 rand_augment: bool = False, color_jitter: float = 0.0, hflip_prob: float = 0.5):
    aug = []
    
    # 1. ALWAYS apply Smart Crop & Pad first (Preserves pixel integrity for forensic features)
    aug.append(SmartCropPad(size=img_size, train=train))
    
    if train:
        # 2. STRICT 1-TO-1 PAPER AUGMENTATION MATCH
        aug.extend([
            transforms.RandomAffine(
                degrees=15,             
                translate=(0.1, 0.1),   
                scale=(0.8, 1.2),       
                shear=10                
            ),
            transforms.RandomHorizontalFlip(p=hflip_prob),
            transforms.ColorJitter(brightness=(0.8, 1.2))
        ])

    return transforms.Compose([
        transforms.Compose(aug),
        ForensicTransform(img_size=img_size, train=train, 
                          use_fft=use_fft, use_ela=use_ela, use_prnu=use_prnu)
    ])