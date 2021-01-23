import os
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RetinalDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = [ Path(image_dir, fname) for fname in os.listdir(image_dir) ]
        self.mask_paths = [ Path(mask_dir, fname) for fname in os.listdir(mask_dir) ]
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks.")
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        x_path = self.image_paths[idx]
        y_path = self.mask_paths[idx]

        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')

        x = self.transform(x)
        y = self.transform(y)

        return x, y

if __name__ == '__main__':
    ds = RetinalDataset('training/images/', 'training/1st_manual/')
    print(len(ds))
    print(ds[0][0].shape)
