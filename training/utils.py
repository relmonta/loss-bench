import numpy as np
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cmocean
import torch
import os
import math
from contextlib import contextmanager
import time


@contextmanager
def inference_mode(desc=""):
    start_time = time.time()
    with torch.no_grad():
        yield
    elapsed_time = time.time() - start_time
    print(f"[{desc}] Completed in {elapsed_time:.3f} seconds.")


def unstandardize_data(low_res, high_res, outputs, mean_x, std_x, device):
    """
    Unstandardize the data using the provided mean and standard deviation.

    Args:
        low_res (torch.Tensor): Low-resolution data.
        high_res (torch.Tensor): High-resolution data.
        outputs (torch.Tensor): Model outputs.
        mean_x (torch.Tensor): Mean used for standardization.
        std_x (torch.Tensor): Standard deviation used for standardization.
        device (torch.device): Device to perform the operation on.

    Returns:
        tuple: Unstandardized low_res, high_res, and outputs.
    """
    if low_res is not None:
        low_res_ = low_res * std_x.to(device) + mean_x.to(device)
    else:
        low_res_ = low_res
    high_res_ = high_res * std_x.to(device) + mean_x.to(device)
    outputs_ = outputs * std_x.to(device) + mean_x.to(device)
    return low_res_, high_res_, outputs_


def inference_step(batch, model, train_dataset, device, dates=False):
    if len(batch) == 4:
        low_res, static_vars, time, high_res = batch
    else:
        low_res, static_vars, time, high_res, mean_x, std_x, *_ = batch

    # Move to device
    low_res, high_res = low_res.to(device), high_res.to(device)
    static_vars, time = static_vars.to(
        device), time.to(device)

    # Model forward pass
    outputs = model(low_res, static_vars, time)
    if isinstance(outputs, list):  # Bernoulli Gamma output
        # Compute bernoulli gamma expected value (pi * alpha * beta)
        outputs = outputs[0] * outputs[1] * outputs[2]

    if train_dataset.standardize:  # Unstandardize
        low_res, high_res, outputs = unstandardize_data(
            low_res, high_res, outputs, mean_x, std_x, device
        )

    # Denormalize
    low_res = train_dataset.denormalize_x(low_res)
    high_res = train_dataset.denormalize_y(high_res)
    outputs = train_dataset.denormalize_y(outputs)

    if dates:
        return low_res, high_res, outputs, time
    else:
        return low_res, high_res, outputs


def get_cmap_norm(var, vmin, vmax, err=False, lognorm=True):
    """Load a .rgb colormap file and return a Matplotlib colormap."""
    rain_colors = ["#fdfdfd",  # white
                   "#04e9e7", "#01c5f4", "#019ff4", "#0077f4", "#0300f4",  # light to deep blue
                   "#02fd02", "#01c501", "#008e00",                         # greens
                   "#ccfd02", "#fdf802", "#f0c802", "#e5bc00",              # yellow-gold
                   "#fd9500", "#fd5a00", "#fd0000", "#d40000", "#bc0000",   # oranges to red
                   "#f800fd", "#c479f2", "#9854c6"               # magenta, purple
                   ]
    rain_cmap = mcolors.ListedColormap(rain_colors)
    path = os.path.join(os.path.expanduser(
        "~"), "projects/downscaling_total/palettes/")
    var_cmap_disct = {"rlds": cmocean.cm.solar,
                      "rsds": cmocean.cm.solar,
                      "rsns": cmocean.cm.solar,
                      "zg500": cmocean.cm.haline,
                      "pr": rain_cmap,  # cmocean.cm.rain,
                      "ps": cmocean.cm.haline,
                      "psl": cmocean.cm.haline}
    if err or var not in list(var_cmap_disct.keys()):
        try:
            if vmin < 0 and vmax > 0:
                norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        except ValueError:
            print(f"vmin: {vmin:.3f}, vmax:{vmax:.3f}")
            raise ValueError
        if err:
            disc_cmap = discrete_cmap("RdBu_r")
        elif var.startswith("tas"):
            disc_cmap = discrete_cmap(cmocean.cm.thermal)
        elif var.startswith("ua") or var.startswith("va"):
            disc_cmap = discrete_cmap(cmocean.cm.curl)
        else:
            disc_cmap = discrete_cmap(cmocean.cm.balance)
    else:
        disc_cmap = discrete_cmap(var_cmap_disct[var])
        norm = None
    if lognorm and var == 'pr' and vmax > 100 or -vmin > 100:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        norm = mcolors.SymLogNorm(
            linthresh=5, linscale=1, vmin=vmin, vmax=vmax)

    return {'cmap': disc_cmap, 'norm': norm,
            'vmin': vmin if not norm else None,
            'vmax': vmax if not norm else None}


def discrete_cmap(cmap, N=21):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, N))  # Sample `N` colors
    cmap_discrete = mcolors.ListedColormap(colors)
    return cmap_discrete


def finalize_plot(fig, path=None, logger=None, logger_tag=None, current_epoch=None):
    """
    Finalize the plot by saving, showing, and logging it.

    Args:
        fig (matplotlib.figure.Figure): The figure to finalize.
        path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
        logger (object, optional): Logger for experiment tracking. Defaults to None.
        logger_tag (str, optional): Tag for logging the figure. Defaults to "Validation Example".
        current_epoch (int, optional): Current epoch for logging. Defaults to None.
    """
    plt.tight_layout()

    # Save the plot if needed
    if path:
        # make sure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight', format=path.split('.')[-1])
        print(f"Plot saved to {'/'.join(path.split('/')[-4:])}")

    # Log the plot if a logger is provided
    if logger:
        logger.experiment.add_figure(
            logger_tag, fig, global_step=current_epoch)

    # Close the plot to free resources
    plt.close()


def fmt(var):
    abs_var = abs(var)
    digits_before = 0 if abs_var == 0 else int(math.log10(abs_var)) + 1
    decimals = min(max(3 - digits_before, 0), 3)  # Adjust decimals dynamically
    return f"{var:.{decimals}f}"


def cartopy_setup(ax):
    ax.coastlines(resolution='110m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5, linewidth=0.6)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.3)


def plot_predictions(low_res_b, high_res_b, output_b, extent, path=None,
                     logger=None, current_epoch=None, var_details=None):
    """
    Plots N variables with (N, H, W) shape as subplots, focusing on a given sub-region.

    Args:
        low_res_batch (numpy.ndarray or torch.Tensor): Low-resolution input of shape (N, H, W)
        high_res_batch (numpy.ndarray or torch.Tensor): High-resolution ground truth of shape (N, H, W)
        output_batch (numpy.ndarray or torch.Tensor): Model output of shape (N, H, W)
        extent (list): Full geographical extent [lon_min, lon_max, lat_min, lat_max]
        path (str, optional): Path to save the figure
        logger (object, optional): Logger for TensorBoard
        current_epoch (int, optional): Epoch number for logging
        var_details (dict, optional): Metadata for variables (units, titles, scaling)
    """
    # print("high_res", high_res.shape)

    fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 3),
                             subplot_kw={'projection': ccrs.Robinson()})

    # Handle the case where there is only one variable
    v = var_details['name']
    low_res, high_res, output, unit = low_res_b[0,
                                                0], high_res_b[0, 0], output_b[0, 0], var_details['unit']
    vmin_i = high_res.min()
    vmax_i = high_res.max()

    for ax, data, title in zip(axes, [low_res, high_res, output], ["L-Res", "H-Res", "Pred"]):
        arg_dict = get_cmap_norm(v, vmin=vmin_i, vmax=vmax_i)
        cf = ax.imshow(
            data,
            extent=extent,
            origin='upper',
            transform=ccrs.PlateCarree(),
            **arg_dict
        )

        # Add coastlines and borders
        cartopy_setup(ax)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(
            title +
            f" | min: {fmt(data.min())} max: {fmt(data.max())}, mean: {fmt(data.mean())}",
            fontsize=max(10, int(7 * fig.get_size_inches()[0] / 10))
        )
        ax.axis('off')

        # Add colour bar
        cbar = fig.colorbar(
            cf, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
        cbar.set_label(
            var_details['title'] + f" ({unit})", fontsize=14 + fig.get_size_inches()[1] / 5)
        cbar.ax.tick_params(labelsize=10)

    finalize_plot(fig, path=path, logger=logger,
                  logger_tag="Validation Example", current_epoch=current_epoch)


def plot_predictions_err(high_res, output, extent, path=None, logger=None,
                         current_epoch=None, var_details=None):
    """
    Plots a single variable (N=1) with (H, W) shape as subplots: H-Res, Pred, and Pred-H-Res.

    Args:
        high_res (numpy.ndarray or torch.Tensor): High-resolution ground truth of shape (1, H, W)
        output (numpy.ndarray or torch.Tensor): Model output of shape (1, H, W)
        extent (list): Full geographical extent [lon_min, lon_max, lat_min, lat_max]
        path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the plot
        logger (object, optional): Logger for TensorBoard
        current_epoch (int, optional): Epoch number for logging
        var_details (dict, optional): Metadata for variables (units, titles, scaling)
    """
    # N=1, so we don't need variables list, just use var_details['name']
    fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 3),
                             subplot_kw={'projection': ccrs.Robinson()})

    # Extract the single variable
    v = var_details['name']
    high_res, output, unit = high_res[0, 0], output[0, 0], var_details['unit']
    delta = output - high_res

    vmin_i = high_res.min()
    vmax_i = high_res.max()

    titles = ["H-Res", "Pred", "Pred - H-Res"]
    for ax, data, title in zip(axes, [high_res, output, delta], titles):
        err_map = title == "Pred - H-Res"
        arg_dict = get_cmap_norm(v,
                                 vmin=vmin_i if not err_map else data.min(),
                                 vmax=vmax_i if not err_map else data.max(),
                                 err=err_map)

        cf = ax.imshow(
            data,
            extent=extent,
            origin='upper',
            transform=ccrs.PlateCarree(),
            **arg_dict
        )

        # Add coastlines and borders
        cartopy_setup(ax)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(
            title +
            f" | min: {fmt(data.min())} max: {fmt(data.max())}, mean: {fmt(data.mean())}",
            fontsize=max(10, int(7 * fig.get_size_inches()[0] / 10)))
        ax.axis('off')

        # Add colour bar
        cbar = fig.colorbar(
            cf, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
        cbar.set_label(
            var_details['title'] + f" ({unit})", fontsize=14 + fig.get_size_inches()[1] / 5)
        cbar.ax.tick_params(labelsize=10)

    finalize_plot(fig, path=path, logger=logger,
                  logger_tag="Validation Example Err", current_epoch=current_epoch)
