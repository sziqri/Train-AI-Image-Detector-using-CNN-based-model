import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transforms import build_transforms

class ForensicMultiSourceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        # 1. Process Real Images
        real_path = os.path.join(root_dir, 'Real')
        for img in os.listdir(real_path):
            self.samples.append((os.path.join(real_path, img), 0, "Real"))
            
        # 2. Process Fake Sub-folders (ADM, BigGAN, etc.)
        fake_root = os.path.join(root_dir, 'Fake')
        for model_name in os.listdir(fake_root):
            model_path = os.path.join(fake_root, model_name)
            if os.path.isdir(model_path):
                for img in os.listdir(model_path):
                    self.samples.append((os.path.join(model_path, img), 1, model_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label, source = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label, source

def build_dataloaders(train_dir, val_dir, img_size, batch_size, num_workers, **kwargs):
    # Pass kwargs (rand_augment, etc.) to your build_transforms
    train_ds = ForensicMultiSourceDataset(train_dir, transform=build_transforms(img_size, train=True, **kwargs))
    val_ds = ForensicMultiSourceDataset(val_dir, transform=build_transforms(img_size, train=False))

    # Optimization for your RTX 3060
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, ["Real", "AI-Generated"]