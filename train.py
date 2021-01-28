from dataset import RetinalDataset
from models import C4UNet, UNet
from torch.utils.data import DataLoader
from e2cnn import gspaces
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a U-Net model')
    parser.add_argument('--n_epochs', type=int, default=10, help='Max number of epochs')
    parser.add_argument('--bs', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image_dir', type=str, default='training/images', help='Path to training dataset')
    parser.add_argument('--label_dir', type=str, default='training/1st_manual', help='Path to training dataset manually segmented masks')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory path to save results')

    config = parser.parse_args()

    ds = RetinalDataset(config.image_dir, config.label_dir)
    train_loader = DataLoader(ds, batch_size=config.bs, shuffle=True, num_workers=12, pin_memory=True)

    r2_act = gspaces.Rot2dOnR2(8)
    model = C4UNet(r2_act, 3,1, n_features=8)

    checkpoint_cb = ModelCheckpoint(monitor='train_loss')

    trainer = pl.Trainer(gpus=1, max_epochs=300, callbacks=[checkpoint_cb])
    # trainer = pl.Trainer(auto_lr_find=True, gpus=1, callbacks=[checkpoint_cb])
    # trainer.tune(model, train_loader)
    trainer.fit(model, train_loader)

