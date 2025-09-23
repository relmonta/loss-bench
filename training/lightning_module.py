from training.callbacks import linear_warmup_scheduler
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import os
from training.utils import plot_predictions, plot_predictions_err
from training.utils import unstandardize_data
from models.losses import *
import numpy as np
import torch.distributed as dist
import math


class DownscalingModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, learning_rate, metrics: dict = None):
        super(DownscalingModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        # Get metric instances to device
        self.done = False
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_has_nllbg = isinstance(self.criterion, NLLBernoulliGammaLoss) or isinstance(
            self.criterion, LossCombination)

    def forward(self, x):
        # gradient checkpointing
        return checkpoint(self.model, x)

    def training_step(self, batch, batch_idx):
        # Get metric instances to device
        if self.metrics is not None and not self.done:
            for metric_name, metric_fn in self.metrics.items():
                if isinstance(metric_fn, nn.Module):
                    self.metrics[metric_name] = metric_fn.to(self.device)
            self.done = True
        # high_res shape: (B, V, H, W)
        if len(batch) == 4:
            low_res, static_vars, time, high_res = batch
        else:
            low_res, static_vars, time, high_res, mean_x, std_x, orig_y = batch
        # Model forward pass
        # Expected shape: (B, V, H, W)
        outputs = self.model(low_res, static_vars, time)
        if isinstance(self.criterion, AsymmetricLoss):
            loss = self.criterion(outputs, high_res.clone(),
                                  orig_y_true=orig_y)
        else:
            loss = self.criterion(outputs, high_res.clone())
        # Log NLL BG loss

        if self.loss_has_nllbg:
            self._log_losses_(
                outputs, high_res, "train", prog_bar=True, on_step=True,
                criterion=self.criterion, criterion_name="loss")

        if isinstance(outputs, list):  # Bernoulli-Gamma output
            # Compute expected value without tracking gradients
            pi = outputs[0].detach()
            alpha = outputs[1].detach()
            beta = outputs[2].detach()
            outputs = pi * alpha * beta
        else:
            outputs = outputs.detach()

        # If the dataset is standardized, unstandardize the data
        # to log consistent values
        if not self.trainer.sanity_checking:
            train_ds = self.trainer.train_dataloader.dataset
            if train_ds.standardize:
                # Unstandardize the data
                _, high_res, outputs = unstandardize_data(
                    None, high_res, outputs, mean_x, std_x, self.device)
            high_res = train_ds.denormalize_y(
                high_res)
            outputs = train_ds.denormalize_y(
                outputs)
            # Do not log  NLL BG loss as the ouputs are not
            # pi, alpha, beta anymore, but the expected value.
            if not self.loss_has_nllbg:
                self._log_losses_(
                    outputs, high_res, "train", prog_bar=True, on_step=True,
                    criterion=self.criterion, criterion_name="loss")
            for metric_name, metric_fn in self.metrics.items():
                self._log_losses_(outputs, high_res, "train",
                                  criterion=metric_fn, criterion_name=metric_name)
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("nan_steps", 1, prog_bar=True)
            print(f"Skipping step {batch_idx} due to NaN loss")
            return None
        return loss

    def on_before_optimizer_step(self, optimizer):
        skip = False
        for name, param in self.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                print(f"Bad grad in {name}, skipping optimizer step")
                skip = True
                break
        if skip:
            optimizer.zero_grad()

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            low_res, static_vars, time, high_res = batch
        else:
            low_res, static_vars, time, high_res, mean_x, std_x, _ = batch
        # Model forward pass
        outputs = self.model(low_res, static_vars, time)
        loss = self.criterion(outputs, high_res)

        # Log  NLL BG loss
        if self.loss_has_nllbg:
            self._log_losses_(
                outputs, high_res, "val", prog_bar=True, on_step=True,
                criterion=self.criterion, criterion_name="loss")

        if isinstance(outputs, list):  # Bernoulli-Gamma output
            # Compute expected value without tracking gradients
            pi = outputs[0].detach()
            alpha = outputs[1].detach()
            beta = outputs[2].detach()
            outputs = pi * alpha * beta
        else:
            outputs = outputs.detach()

        if not self.trainer.sanity_checking:
            train_ds = self.trainer.train_dataloader.dataset
            if train_ds.standardize:
                _, high_res, outputs = unstandardize_data(
                    None, high_res, outputs, mean_x, std_x, self.device)

            high_res = train_ds.denormalize_y(
                high_res)
            outputs = train_ds.denormalize_y(
                outputs)
            # Do not log  NLL BG loss as the ouputs are not
            # pi, alpha, beta anymore, but the expected value.
            if not self.loss_has_nllbg:
                self._log_losses_(
                    outputs, high_res, "val", prog_bar=True, on_step=True,
                    criterion=self.criterion, criterion_name="loss")

            for metric_name, metric_fn in self.metrics.items():
                self._log_losses_(outputs, high_res, "val",
                                  criterion=metric_fn, criterion_name=metric_name)
        return loss

    @torch.no_grad()
    def on_validation_epoch_end(self):
        # log learning rate
        if self.trainer.is_global_zero:
            lr = self.optimizer.param_groups[0]['lr']
            self.log('learning_rate', lr, on_step=False, on_epoch=True,
                     prog_bar=False, logger=True, sync_dist=True)
        if not self.trainer.sanity_checking:

            # Only the first process will execute this (global rank = 0)
            if self.trainer.is_global_zero:
                train_ds = self.trainer.train_dataloader.dataset
                # Get a sample from the validation dataset
                batch = next(iter(self.trainer.val_dataloaders[0]))

                if len(batch) == 4:
                    low_res, static_vars, time, high_res = batch
                else:
                    low_res, static_vars, time, high_res, mean_x, std_x, _ = batch
                # print(low_res.shape, static_vars.shape, time.shape, high_res.shape,
                #       aux.shape, text_id.shape, mean_x.shape, std_x.shape)
                # Move to device
                low_res, high_res = low_res.to(
                    self.device), high_res.to(self.device)
                static_vars, time = static_vars.to(
                    self.device), time.to(self.device)

                # Model forward pass
                outputs = self.model(low_res, static_vars, time)

                if isinstance(outputs, list):  # Bernoulli-Gamma output
                    # Compute expected value without tracking gradients
                    pi = outputs[0].detach()
                    alpha = outputs[1].detach()
                    beta = outputs[2].detach()
                    outputs = pi * alpha * beta
                else:
                    outputs = outputs.detach()

                if train_ds.standardize:  # Unstandardize
                    low_res, high_res, outputs = unstandardize_data(
                        low_res, high_res, outputs, mean_x, std_x, self.device
                    )

                # Denormalize
                low_res = train_ds.denormalize_x(low_res)
                high_res = train_ds.denormalize_y(high_res)
                outputs = train_ds.denormalize_y(outputs)

                plot_predictions(low_res.cpu().numpy(),
                                 high_res.cpu().numpy(),
                                 outputs.cpu().numpy(),
                                 extent=train_ds.extent,
                                 path=None, logger=self.logger,
                                 current_epoch=self.current_epoch,
                                 var_details=train_ds.var_details
                                 )
                plot_predictions_err(high_res.cpu().numpy(),
                                     outputs.cpu().numpy(),
                                     extent=train_ds.extent,
                                     path=None, logger=self.logger,
                                     current_epoch=self.current_epoch,
                                     var_details=train_ds.var_details
                                     )
            if dist.is_initialized():
                dist.barrier()

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = linear_warmup_scheduler(
            optimizer,
            (0.1 * self.trainer.max_epochs),
            self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _log_losses_(self, y_pred, y_true, title, prog_bar=False, on_step=False,
                     criterion=None, criterion_name=None):
        loss = criterion(y_pred, y_true)
        # check if not nan
        if not torch.isnan(loss):
            self.log(f'{title}_{criterion_name}', loss, on_step=on_step, on_epoch=True, prog_bar=prog_bar,
                     logger=True, sync_dist=True)
        return loss
