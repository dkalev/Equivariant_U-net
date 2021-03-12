import sys
from e2cnn import gspaces
from e2cnn.nn import FieldType, R2Conv, InnerBatchNorm, ReLU, GroupPooling, GeometricTensor
from e2cnn.nn import NormNonLinearity, GNormBatchNorm, NormPool, NormBatchNorm, GatedNonLinearity1
from e2cnn.gspaces import Rot2dOnR2

from torch.nn.modules.container import Sequential
sys.path.append('..')
from loss import DiceLoss
import torch
import torch.nn as nn
import pytorch_lightning as pl
from loss import DiceLoss
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

class BaseCNN(pl.LightningModule):
    def __init__(self, *args, lr=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = DiceLoss()
        # self.crit = nn.BCEWithLogitsLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1()
        self.lr = lr
        
    def training_step(self, batch, batch_idx):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss)
        return loss
   
    def validation_step(self, batch, batch_idx):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss, split='valid')
    
    def log_metrics(self, preds, targs, loss, split='train'):
        preds = torch.sigmoid(preds)
        self.log(f'{split}_loss', loss)
        self.log(f'{split}_acc', self.accuracy(preds, targs))
        self.log(f'{split}_f1', self.f1(preds, targs))
        if split == 'valid':
            targs_numpy = targs.cpu().flatten().numpy().astype(int)
            preds_numpy = preds.cpu().flatten()
            self.log(f'valid_auc', roc_auc_score(targs_numpy, preds_numpy))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class BaselineCNN(BaseCNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 64, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=(3,3), padding=1),
        )

    def forward(self, x):
        return self.model(x)



class BaselineRegularCNN(BaseCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gspace = Rot2dOnR2(4)
        self.input_type = FieldType(self.gspace, 3*[self.gspace.trivial_repr]) 

        self.small_type = FieldType(self.gspace, 4*[self.gspace.regular_repr])
        self.mid_type = FieldType(self.gspace, 16*[self.gspace.regular_repr])

        self.model = nn.Sequential(
            R2Conv(self.input_type, self.small_type, kernel_size=3, padding=1, bias=False),
            InnerBatchNorm(self.small_type),
            ReLU(self.small_type),

            R2Conv(self.small_type, self.small_type, kernel_size=3, padding=1, bias=False),
            InnerBatchNorm(self.small_type),
            ReLU(self.small_type),

            R2Conv(self.small_type, self.small_type, kernel_size=3, padding=1, bias=False),
            InnerBatchNorm(self.small_type),
            ReLU(self.small_type),

            R2Conv(self.small_type, self.mid_type, kernel_size=3, padding=1, bias=False),
            InnerBatchNorm(self.mid_type),
            ReLU(self.mid_type),

            R2Conv(self.mid_type, self.small_type, kernel_size=3, padding=1, bias=False),
            InnerBatchNorm(self.small_type),
            ReLU(self.small_type),
        )

        self.pool = GroupPooling(self.small_type)
        pool_out = self.pool.out_type.size
        self.final = nn.Conv2d(pool_out, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = GeometricTensor(x, self.input_type)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x



class BaselineSteerableCNN(BaseCNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gspace = Rot2dOnR2(-1, maximum_frequency=2)
        self.input_type = FieldType(self.gspace, 3*[self.gspace.trivial_repr]) 

        self.small_type = FieldType(self.gspace, 4*list(self.gspace.irreps.values()))
        self.mid_type = FieldType(self.gspace, 16*list(self.gspace.irreps.values()))

        self.model = nn.Sequential(
            R2Conv(self.input_type, self.small_type, kernel_size=3, padding=1, bias=False),
            GNormBatchNorm(self.small_type),
            NormNonLinearity(self.small_type),

            R2Conv(self.small_type, self.small_type, kernel_size=3, padding=1, bias=False),
            GNormBatchNorm(self.small_type),
            NormNonLinearity(self.small_type),
            
            R2Conv(self.small_type, self.small_type, kernel_size=3, padding=1, bias=False),
            GNormBatchNorm(self.small_type),
            NormNonLinearity(self.small_type),

            R2Conv(self.small_type, self.mid_type, kernel_size=3, padding=1, bias=False),
            GNormBatchNorm(self.mid_type),
            NormNonLinearity(self.mid_type),

            R2Conv(self.mid_type, self.small_type, kernel_size=3, padding=1, bias=False),
            GNormBatchNorm(self.small_type),
            NormNonLinearity(self.small_type),
        )

        self.pool = NormPool(self.small_type)
        pool_out = self.pool.out_type.size
        self.final = nn.Conv2d(pool_out, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = GeometricTensor(x, self.input_type)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x

from e2cnn.nn import MultipleModule

class GatedSteerableCNN(BaseCNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gspace = Rot2dOnR2(-1, maximum_frequency=2)
        self.input_type = FieldType(self.gspace, 3*[self.gspace.trivial_repr]) 

        layers = []

        irreps = [ v for k, v in self.gspace.irreps.items() if k != self.gspace.trivial_repr.name]

        trivials = FieldType(self.gspace, [self.gspace.trivial_repr]*10)
        gates = FieldType(self.gspace, len(irreps) * [self.gspace.trivial_repr]*10)
        gated = FieldType(self.gspace, irreps*10).sorted()
        gate = gates + gated

        self.small_type = trivials + gate

        layers.append(
            R2Conv(self.input_type, self.small_type, kernel_size=3, padding=1, bias=False)
        )
        layers.append( 
            MultipleModule(layers[-1].out_type,
            labels=[
                 *(["trivial"] * (len(trivials) + len(gates)) + ["gated"] * len(gated))
            ],
            modules=[
                (InnerBatchNorm(trivials + gates), 'trivial'),
                (NormBatchNorm(gated), 'gated')
            ])
        )
        layers.append(
            MultipleModule(layers[-1].out_type,
            labels=[
                *(["trivial"] * len(trivials) + ["gate"] * len(gate))
            ], 
            modules=[
                (ReLU(trivials), 'trivial'),
                (GatedNonLinearity1(gate), 'gate')
            ])

        )

        self.model = nn.Sequential(*layers)

        self.pool = NormPool(layers[-1].out_type)
        pool_out = self.pool.out_type.size
        self.final = nn.Conv2d(pool_out, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = GeometricTensor(x, self.input_type)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x

if __name__ == '__main__':
    from dataset import RetinalDataModule
    rdm = RetinalDataModule()

    # model = BaselineCNN()
    # model = BaselineRegularCNN()
    # model = BaselineSteerableCNN()
    model = GatedSteerableCNN()

    trainer = pl.Trainer(gpus=1,
                        min_epochs=50,
                        max_epochs=1000)

    trainer.fit(model, rdm)