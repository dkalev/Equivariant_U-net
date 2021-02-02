from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import zipfile
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional


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

        # CHECK if resizing is necessary and if larger/smaller size should be used
        x = Image.open(x_path).resize((512,512)).convert('RGB')
        y = Image.open(y_path).resize((512,512))

        x = self.transform(x)
        y = self.transform(y)

        return x, y


class RetinalDataModule(LightningDataModule):

    def __init__(self):
        super().__init__()

    def prepare_data(self, datapath='data/datasets.zip', output_dir='data'):
        with zipfile.ZipFile(datapath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        for fname in ['training', 'test']:
            path = Path(output_dir, f'{fname}.zip')
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(path)

        train_dir = Path(output_dir, 'training')
        valid_dir = Path(output_dir, 'valid')
        os.makedirs(valid_dir)

        for dirname in os.listdir(train_dir):
            source_dir = Path(train_dir, dirname)
            targ_dir = Path(valid_dir, dirname)
            os.makedirs(targ_dir)
            for fname in os.listdir(source_dir)[-4:]:
                os.rename(Path(source_dir, fname), Path(targ_dir, fname))

    def setup(self, stage: Optional[str] = None):
        self.paths = {
            'train': {
                'images': 'data/training/images',
                'labels': 'data/training/1st_manual',
            },
            'valid': {
                'images': 'data/valid/images',
                'labels': 'data/valid/1st_manual',
            }
        }

    def train_dataloader(self):
        train_split = RetinalDataset(self.paths['train']['images'], self.paths['train']['labels'])
        return DataLoader(train_split)

    def val_dataloader(self):
        valid_split = RetinalDataset(self.paths['valid']['images'], self.paths['valid']['labels'])
        return DataLoader(valid_split)
