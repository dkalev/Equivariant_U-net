import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from dataset import RetinalDataModule
from pathlib import Path

from train import get_model

def train_tune(config, rdm):

    model = get_model(config)

    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False)
    logger.log_hyperparams(config, {
        'train_acc': 0,
        'train_f1': 0,
        'train_loss': 0,
        'valid_acc': 0,
        'valid_f1': 0,
        'valid_loss': 0,
    })

    trainer = pl.Trainer(
        max_epochs=config['n_epochs'],
        gpus=1,
        logger=logger,
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                ['valid_acc', 'valid_f1', 'valid_loss'],
                on="validation_end")
        ])
    trainer.fit(model, rdm)

def grid_search(config):
    scheduler = ASHAScheduler(
        max_t=config['n_epochs'],
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        parameter_columns=config['param_cols'],
        metric_columns=['valid_acc', 'valid_f1', 'valid_loss'])

    rdm = RetinalDataModule()

    analysis = tune.run(
        tune.with_parameters(train_tune, rdm=rdm),
        resources_per_trial={ "cpu": 1, "gpu":1 },
        metric="valid_loss",
        mode="min",
        config=config,
        local_dir=Path(config['output_dir'], 'ray_tune'),
        num_samples=5,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=f"tune_{config['model']}_DRIVE")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a U-Net model")
    parser.add_argument('--model', type=str, default='steerable', help='Model type. One of [discrete, standard, harmonic, steerable]')
    parser.add_argument('--n_epochs', type=int, default=10, help='Max number of epochs')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory path to save results')

    config = parser.parse_args()
    config = vars(config)

    # config['lr'] = tune.loguniform(1e-5, 1e-1)
    config['lr'] = 1e-3
    config['n_features'] = tune.choice([2, 8, 16, 32])
    # config['batch_size'] = tune.choice([2, 4, 8])
    config['batch_size'] = 4

    if config['model'] == 'steerable':
        config['irrep_type'] = tune.grid_search(['all', 'even', 'odd'])
        config['n_blocks'] = tune.grid_search([2, 4, 8, 16])
        config['max_freq'] = tune.grid_search([2, 4, 8, 16])
    elif config['model'] == 'discrete':
        config['N'] = tune.grid_search([2,4,8,16])
        config['group'] = tune.grid_search(['circle', 'dihedral'])

    param_cols = ["lr", "batch_size", "irrep_type", "n_features", "n_blocks", "max_freq", "N", "group"]
    param_cols = list(set(param_cols).intersection(config.keys()))
    config['param_cols'] = param_cols

    grid_search(config)