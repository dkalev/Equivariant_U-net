import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from dataset import RetinalDataModule
from pathlib import Path
from parse_hyperparams import parse_hparams

from train import get_model

def train_tune(hparams, rdm):

    model = get_model(hparams)

    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False)
    logger.log_hyperparams(hparams, {
        'train_acc': 0,
        'train_f1': 0,
        'train_loss': 0,
        'valid_acc': 0,
        'valid_f1': 0,
        'valid_loss': 0,
    })

    trainer = pl.Trainer(
        max_epochs=hparams['n_epochs'],
        gpus=1,
        logger=logger,
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                ['valid_acc', 'valid_f1', 'valid_loss'],
                on="validation_end")
        ])
    trainer.fit(model, rdm)

def grid_search(hparams):
    scheduler = ASHAScheduler(
        max_t=hparams['n_epochs'],
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        parameter_columns=hparams['param_cols'],
        metric_columns=['valid_acc', 'valid_f1', 'valid_loss'])

    rdm = RetinalDataModule()

    analysis = tune.run(
        tune.with_parameters(train_tune, rdm=rdm),
        resources_per_trial={ "cpu": 1, "gpu":1 },
        metric="valid_loss",
        mode="min",
        config=hparams,
        local_dir=Path(hparams['output_dir'], 'ray_tune'),
        num_samples=5,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=f"tune_{hparams['model']}_DRIVE")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grid search for a U-Net model")
    hparams = parse_hparams(parser)
    # hparams['lr'] = tune.loguniform(1e-5, 1e-1)
    hparams['lr'] = 1e-3
    hparams['n_features'] = tune.grid_search([2, 8, 16, 32])
    # hparams['batch_size'] = tune.choice([2, 4, 8])
    hparams['batch_size'] = 4

    if hparams['model'] == 'steerable':
        hparams['irrep_type'] = tune.grid_search(['all', 'even', 'odd'])
        hparams['n_blocks'] = tune.grid_search([2, 4, 8, 16])
        hparams['max_freq'] = tune.grid_search([2, 4, 8, 16])
    elif hparams['model'] == 'discrete':
        hparams['N'] = tune.grid_search([2,4,8,16])
        hparams['group'] = tune.grid_search(['circle', 'dihedral'])

    param_cols = ["lr", "batch_size", "irrep_type", "n_features", "n_blocks", "max_freq", "N", "group"]
    param_cols = list(set(param_cols).intersection(hparams.keys()))
    hparams['param_cols'] = param_cols

    grid_search(hparams)