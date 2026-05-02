import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transforms_extra import build_transforms_with_extras

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transforms_extra import build_transforms_with_extras

class ForensicMultiSourceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        root_dir = os.path.abspath(root_dir)
        
        # 1. Find the Fake folder
        fake_folder = next((d for d in os.listdir(root_dir) if d.lower() == 'fake'), None)
        if fake_folder:
            fake_root = os.path.join(root_dir, fake_folder)
            # Recursively find ALL images in ANY subfolder of Fake
            for root, dirs, files in os.walk(fake_root):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Use the immediate parent folder name as the source (e.g., 'adm')
                        source_name = os.path.basename(root)
                        self.samples.append((os.path.join(root, file), 1, source_name))

        # 2. Find the Real folder
        real_folder = next((d for d in os.listdir(root_dir) if d.lower() == 'real'), None)
        if real_folder:
            real_root = os.path.join(root_dir, real_folder)
            for root, dirs, files in os.walk(real_root):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(root, file), 0, "Real"))
                        
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, source = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: 
            img = self.transform(img)
        return img, label, source

def build_dataloaders_with_extras(train_dir, val_dir, img_size, batch_size, num_workers, 
                                  rand_augment, color_jitter, hflip_prob, 
                                  use_fft=True, use_ela=True, use_prnu=True):
    
    # Correctly map the 11 arguments from train.py to the transform builder
    train_tfm = build_transforms_with_extras(
        img_size, True, use_fft, use_ela, use_prnu, 
        rand_augment, color_jitter, hflip_prob
    )
    val_tfm = build_transforms_with_extras(
        img_size, False, use_fft, use_ela, use_prnu, 
        rand_augment, color_jitter, hflip_prob
    )
    
    train_ds = ForensicMultiSourceDataset(train_dir, transform=train_tfm)
    val_ds = ForensicMultiSourceDataset(val_dir, transform=val_tfm)
    
    # DataLoader will now find the len() method correctly
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
                            
    return train_loader, val_loader, ["Real", "AI-Generated"]