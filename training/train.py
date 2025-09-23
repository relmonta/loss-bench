import torch
from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import numpy as np
import os
import argparse
import pickle
import fcntl
from data.data_module import DatasetSetup
from models.vision_transformer import VisionTransformer
from training.lightning_module import DownscalingModel
from models.losses import *
from training.utils import *

"""
Training script for the downscaling experiments.

This script:
- Loads experiment configuration from YAML
- Prepares datasets, model, optimizer, and loss function
- Sets up metrics and callbacks
- Trains using PyTorch Lightning with DDP support

Usage:
    >>> python train.py -var_name 'pr' --criterion "mse"
"""


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def set_asym_params(criterion, train_ds):
    var_name_ = "pr_log" if train_ds.apply_log_flag else "pr"
    start_year = train_ds.train_start_date.year
    # load norm stats
    with open(train_ds.stats_file, "rb") as f:
        # Acquire an exclusive lock before reading
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data_stats = pickle.load(f)
        finally:
            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)
    alpha = data_stats[var_name_][start_year]["alpha"]
    beta = data_stats[var_name_][start_year]["beta"]
    for year in range(start_year + 1, train_ds.train_end_date.year + 1):
        alpha += data_stats[var_name_][year]["alpha"]
        beta += data_stats[var_name_][year]["beta"]

    lon_min, lon_max, lat_min, lat_max = train_ds.extent
    lats = train_ds.latitudes
    lons = train_ds.longitudes

    # Handle case where latitudes are descending (e.g., ERA5)
    dx = lats[1] - lats[0]
    if lats[0] > lats[-1]:
        lat_mask = (lats <= lat_max) & (lats >= lat_min - dx)
    else:
        lat_mask = (lats >= lat_min - dx) & (lats <= lat_max)

    lon_mask = (lons >= lon_min) & (lons <= lon_max + dx)

    # Indices
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    # Crop
    alpha_crop = alpha[np.ix_(lat_idx, lon_idx)]
    beta_crop = beta[np.ix_(lat_idx, lon_idx)]

    alpha_crop = torch.from_numpy(alpha_crop)/train_ds.train_num_years
    beta_crop = torch.from_numpy(beta_crop)/train_ds.train_num_years
    if isinstance(criterion, AsymmetricLoss):
        criterion.set_params(alpha=alpha_crop.to(
            'cuda'), beta=beta_crop.to('cuda'))
    elif isinstance(criterion, LossCombination):
        for loss in criterion.losses:
            if isinstance(loss, AsymmetricLoss):
                loss.set_params(alpha=alpha_crop.to('cuda'),
                                beta=beta_crop.to('cuda'))
                print(f"Set alpha and beta for {loss} loss")
    else:
        raise ValueError(
            f"Criterion {criterion_name} is not an instance of AsymmetricLoss or LossCombination")


if __name__ == '__main__':
    # -----------------------------
    # Arguments
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Train VisionTransformer downscaling model")
    parser.add_argument('-var_name', type=str, required=True,
                        help='Target variable name (e.g., uas, pr)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint if available')
    parser.add_argument('--criterion', type=str,
                        default=None, help='Loss function override')
    parser.add_argument('--apply_log', type=str, default=None,
                        help='Apply log scaling to input data (true/false)')
    args = parser.parse_args()

    var_name = args.var_name
    resume = args.resume

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # -----------------------------
    # Load configuration
    # -----------------------------
    exp_config = load_yaml(f'configs/exp_config_{var_name}.yaml')

    batch_size = exp_config['training']['batch_size']
    accumulate_grad_batches = exp_config['training']['accumulate_grad_batches']
    num_epochs = exp_config['training']['epochs']
    learning_rate = exp_config['training']['learning_rate']
    # get number of available CPUs
    num_cpus = os.cpu_count()
    num_workers = min(exp_config['training']['num_workers'], num_cpus)

    criterion_name = args.criterion or exp_config['training']['criterion']

    # -----------------------------
    # Dataset setup
    # -----------------------------
    if "nllbg" in criterion_name.lower():
        exp_config['data']['kwargs_train_val']['normalize'] = False
        exp_config['data']['kwargs_train_val']['standardize'] = False

    if args.apply_log is not None:
        exp_config['data']['kwargs_train_val']['apply_log'] = args.apply_log.lower(
        ) == "true"

    # -----------------------------
    # Model setup
    # -----------------------------
    model_args = exp_config["model"]["params"]
    model_args["bernoulli_gamma"] = "nllbg" in criterion_name.lower()
    model = VisionTransformer(**model_args).cuda()

    # Load YAML config containing loss details (losses_var.yaml)
    loss_config = load_yaml(exp_config['training']['loss_config_path'])
    metrics = {}
    for metric_name in exp_config['training']['metrics']:
        metric_args = loss_config['losses'].get(metric_name.lower(), {}) or {}
        metrics[metric_name] = get_criterion(metric_name, **metric_args)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=exp_config['training']['weight_decay'])

    # Get loss args from config
    loss_args = loss_config['losses'].get(criterion_name.lower(), {}) or {}
    print("=====================================================================")
    if criterion_name.lower().startswith('combo'):
        # For combination losses, gather individual loss args
        wargs_dict = {loss: loss_config['losses'].get(loss, {}) or {}
                      for loss in loss_config['losses'][criterion_name]["losses"]}
        loss_args['losses'] = wargs_dict
        print(f"Losses args: {wargs_dict}")
        criterion = get_criterion("combination", **loss_args)
        print(
            f"Training using a combination of : {[loss_config['display'][loss] for loss in loss_args['losses']] } losses")
    else:
        # For single losses, use args directly
        criterion = get_criterion(criterion_name, **loss_args)
        print(f"Training using {loss_config['display'][criterion_name]} loss")
    print("=====================================================================")
    require_gamma_params = "asym" in criterion_name.lower()
    dss = DatasetSetup(exp_config, require_gamma_params=require_gamma_params)
    dss.setup()
    train_dataset = dss.get_train_ds()
    val_dataset = dss.get_val_ds()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers)

    # -----------------------------
    # Experiment naming
    # -----------------------------
    description = criterion_name
    if train_dataset.apply_log_flag:
        description = "log_" + description

    weight_path = os.path.join(exp_config['training']['weights_path'],
                               exp_config['name'])
    os.makedirs(weight_path, exist_ok=True)

    filename = f"weights-{description}"
    ckpt_path = os.path.join(weight_path, filename + ".ckpt")

    if not resume and os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} already exists. Deleting it.")
        os.remove(ckpt_path)
    if resume and not os.path.exists(ckpt_path):
        print(
            f"Resume file {ckpt_path} does not exist. Starting training from scratch.")
        resume = False

    if require_gamma_params:
        # Get asym params from train dataset
        set_asym_params(criterion, train_dataset)

    downscaling_model = DownscalingModel(
        model, criterion, optimizer, learning_rate, metrics=metrics
    )

    # -----------------------------
    # Logging & callbacks
    # -----------------------------
    if not resume:
        tb_path = f"training/logs/tensorboard/{exp_config['name']}/{description}"
        if os.path.exists(tb_path):
            os.system(f"rm -rf {tb_path}/*")
            print(f"Deleted previous tensorboard logs at {tb_path}")

    logger = TensorBoardLogger(
        save_dir='training/logs/tensorboard/',
        name=exp_config['name'],
        version=description
    )
    logger.log_hyperparams(exp_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_path,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(
        max_epochs=num_epochs,
        devices='auto',
        accelerator='auto',
        precision=32,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=1,
        strategy='ddp_find_unused_parameters_true',
        accumulate_grad_batches=accumulate_grad_batches,
        num_sanity_val_steps=0,
        detect_anomaly=False
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(downscaling_model, train_loader, [val_loader],
                ckpt_path=ckpt_path if resume else None)
