import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def forward(self, pred, targ):
        assert pred.size() == targ.size()

        intersection = (pred * targ).sum()
        dice_score = (2. * intersection + self.smooth) / ( pred.sum() + targ.sum() + self.smooth )
        return 1. - dice_score
