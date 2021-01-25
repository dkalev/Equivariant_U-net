from dataset import RetinalDataset
from loss import DiceLoss
from model import C4UNet
from torch.utils.data import DataLoader
from torch.optim import AdamW
from e2cnn import gspaces
import torch
import argparse


def eval():
    pass

def train(config):
    for epoch in range(config.n_epochs):
        for i, (x, targs) in enumerate(train_loader):
            model.train()
            optim.zero_grad()

            x, targs = x.to(device), targs.to(device)
            preds = model(x)

            loss = crit(preds, targs)
            loss.backward()
            optim.step()
        print(f'Epoch: {epoch}, loss: {loss.detach().item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a U-Net model')
    parser.add_argument('--n_epochs', type=int, default=10, help='Max number of epochs')
    parser.add_argument('--bs', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image_dir', type=str, default='training/images', help='Path to training dataset')
    parser.add_argument('--label_dir', type=str, default='training/1st_manual', help='Path to training dataset manually segmented masks')

    config = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ds = RetinalDataset(config.image_dir, config.label_dir)
    train_loader = DataLoader(ds, batch_size=config.bs, shuffle=True)

    r2_act = gspaces.Rot2dOnR2(4)
    model = C4UNet(r2_act, 3,1, features=2).to(device)

    optim = AdamW(model.parameters(), lr=config.lr)
    crit = DiceLoss()

    train(config)

    torch.save(model.state_dict(), 'C4Unet.pth')

