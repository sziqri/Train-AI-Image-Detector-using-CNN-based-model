import torch                 
import io                     
from PIL import Image        
from torchvision import transforms

class RandomJPEGCompression:
    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # Ensure image is RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            quality = torch.randint(*self.quality_range, (1,)).item()
            buffer = io.BytesIO()
            img.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        return img

def build_transforms(img_size: int, train: bool = True, **kwargs):
    """
    1-to-1 Replication of Paper 4 Augmentations for Baseline.
    Applies Rotation, Shear, Zoom, Flip, and Brightness adjustments.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            
            # 1. Rotation, Shift, Zoom, and Shear [Paper 4 specific]
            transforms.RandomAffine(
                degrees=15,             # Image rotation (typically 15-20 degrees)
                translate=(0.1, 0.1),   # Horizontal & Vertical shifts
                scale=(0.8, 1.2),       # Zooming in (1.2) and out (0.8)
                shear=10                # Shear transformation
            ),
            
            # 2. Horizontal flipping
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 3. Brightness adjustment exactly within a range of 0.8 to 1.2
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            
            # Finalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Validation/Testing data MUST NOT be augmented, only resized and normalized.
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])