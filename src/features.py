import numpy as np
import io
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance

def extract_ela_feature(image: Image):
    """Extract REAL ELA feature using Memory Buffer (Fast)."""
    try:
        # 1. Prepare image
        image = image.convert('RGB')
        
        # 2. Save to Memory (RAM) instead of Disk (Faster)
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=90)
        buffer.seek(0)
        
        # 3. Reload and Calculate Difference
        jpeg_image = Image.open(buffer)
        ela_image = ImageChops.difference(image, jpeg_image)
        
        # 4. Enhance Brightness (Otherwise it's too dark for the CNN to easily learn)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # 5. Convert to Array and Normalize
        # Output shape: (H, W)
        ela_array = np.array(ela_image.convert('L')).astype(np.float32) / 255.0
        
        return ela_array
    except Exception as e:
        print(f"ELA extraction failed: {e}")
        return None

def extract_fft_feature(image: Image):
    """Extract FFT (Fast Fourier Transform) feature from an image."""
    try:
        # Better luminance conversion than simple mean(axis=2)
        gray_image = image.convert('L')
        img_array = np.asarray(gray_image).astype(np.float32)
        
        # Compute FFT and get magnitude
        fft = np.fft.fft2(img_array)
        fft_shift = np.fft.fftshift(fft)
        
        # Add epsilon (1e-8) to prevent log(0) which causes -inf tensors
        magnitude = np.log(1 + np.abs(fft_shift) + 1e-8)
        
        # Normalize to [0, 1]
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Output shape: (H, W)
        return magnitude
    except Exception as e:
        print(f"FFT extraction failed: {e}")
        return None

def extract_prnu_feature(image: Image):
    """
    Extract PRNU using PyTorch optimized Convolution.
    """
    try:
        # 1. Convert PIL Image to PyTorch Tensor (Grayscale)
        img_array = np.array(image.convert('L'))
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
        
        # 2. Define the High-Pass Filter Kernel (Laplacian/Edge Detection)
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        # 3. Apply Convolution
        # padding=1 ensures a 384x384 input remains exactly 384x384
        prnu_tensor = F.conv2d(img_tensor, kernel, padding=1)
        
        # 4. Post-processing
        prnu_tensor = torch.abs(prnu_tensor)
        
        # Robust Normalization: Clamp extreme outliers before min/max normalization
        # This prevents a single dead/hot pixel from destroying the feature map contrast
        prnu_tensor = torch.clamp(prnu_tensor, min=0.0, max=torch.quantile(prnu_tensor, 0.99))
        
        min_val = prnu_tensor.min()
        max_val = prnu_tensor.max()
        if max_val - min_val > 0:
            prnu_tensor = (prnu_tensor - min_val) / (max_val - min_val)
        
        # Output shape: (H, W) - Fully squeezed to match ELA and FFT dimensions
        return prnu_tensor.squeeze() 
    
    except Exception as e:
        print(f"PRNU extraction failed: {e}")
        return None