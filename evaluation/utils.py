import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, Bbox
from itertools import cycle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
from contextlib import contextmanager
import time
from training.utils import get_cmap_norm

COLORS_CB_FRIENDLY = [
    "#1f77b4",             # blue
    "#ff7f0e",             # orange
    "#2ca02c",             # green
    "#d62728",             # red
    "#9467bd",             # purple
    "xkcd:dark turquoise",  # cyan/teal
    "xkcd:olive",          # olive green
    "xkcd:dark cyan",      # cyan
    "#bcbd22",             # yellow-green
    "#e377c2",             # pink
]


@contextmanager
def inference_mode(desc=""):
    start_time = time.time()
    with torch.no_grad():
        yield
    elapsed_time = time.time() - start_time
    print(f"[{desc}] Completed in {elapsed_time:.3f} seconds.")


def download_weights_from_zenodo(var_name, file_name, save_path):
    """
    Download model weights from Zenodo.

    Args:
        var_name (str): Variable name (e.g., 'pr', 'tas').
        file_name (str): Name of the weights file to download.
        save_path (str): Local path to save the downloaded file.
    """
    var_to_id = {
        "pr": "17182594",
        "uas": "17182398"
    }
    id = var_to_id.get(var_name)
    if not id:
        raise ValueError(f"Unknown variable name: {var_name}")
    # Zenodo url
    zenodo_url = f"https://zenodo.org/records/{id}/files/{file_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        # try using wget
        try:
            os.system(f"wget -O {save_path} {zenodo_url}")
        except Exception as e:
            print(f"Error downloading {zenodo_url} using wget: {e}")
            print("Trying using urllib ... If this fails or take too long, please download the file manually from zenodo.")
            try:
                import urllib.request as requests
                requests.urlretrieve(zenodo_url, save_path)
                print(f"Downloaded weights from {zenodo_url} to {save_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"File {save_path} not found and download failed: {e}")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download weights from {zenodo_url}: {e}")


def loss_display_name(config, loss, weights=True):
    """
    Convert loss name to a more readable format.

    Args:
        config (dict): Configuration dictionary containing loss details.
        loss (str): The loss name.
        weights (bool): Whether to add weights or not

    Returns:
        str: A more readable version of the loss name.
    """
    use_log = loss.startswith('log_')
    log_label = ' ' + r'$(\log)$' if use_log else ""
    log_label = ' ' + '(log)' if use_log else ""
    if use_log:
        loss = loss[4:]

    if loss.startswith('combo'):
        name = ""
        for loss, lda in zip(config['losses'][loss]['losses'], config['losses'][loss]['lambdas']):
            # write lambda as a fraction (e.g., 0.1 -> 1/10, 0.5 -> 1/2, 0.3 -> 3/10)
            lda_frac = get_fraction(lda) if weights else ""
            name += " + " if name != "" else ""
            name += f"{lda_frac}{config['short'][loss]}"
        return name + log_label
    else:
        return config['display'][loss] + log_label


def plot_power_spectra(path, y_preds, y_true, exp_list, exp_names,
                       line_styles_dict, colors_dict, figformat='png', hp=40):
    """
    Plot power spectra for model predictions and ground truth.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    spectra_results = {}

    # Compute power spectrum for ground truth
    k_ref, mean_ref, std_ref = average_power_spectrum(y_true)
    spectra_results["truth"] = (k_ref, mean_ref, std_ref)
    relative_high_freq_delta = {}
    # Compute power spectra for each model prediction
    for exp in exp_list:
        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        k, mean_ps, std_ps = average_power_spectrum(y_pred)
        spectra_results[exp] = (k, mean_ps, std_ps)
        # Compute difference at high frequencies
        relative_high_freq_delta[exp] = np.mean(
            np.abs(mean_ps[k > hp] - mean_ref[k > hp])) / np.mean(mean_ref[k > hp]) * 100

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    line_styles = cycle(line_styles_dict.values())
    for exp, (k, mean_ps, std_ps) in spectra_results.items():
        c = colors_dict.get(exp, None)
        ls = next(line_styles)
        delta = ""
        if exp != "truth":
            delta = r" | $\Delta_{HF}=$" + \
                f"{relative_high_freq_delta[exp]:.1f}%" + r"$\,\downarrow$"
        ax.plot(k, mean_ps, label=exp_names[exp] + delta,
                linewidth=2, alpha=0.9, linestyle=ls, color=c)

    i = 0
    line_styles = cycle(line_styles_dict.values())

    # Add zoomed inset ("spy") at a manual location: upper right, slightly outside the main axes
    # x0, y0, width, height in axes coords
    bbox = Bbox.from_bounds(1.05, 0, 0.8, 1)
    axins = inset_axes(
        ax,
        width="100%", height="100%",
        bbox_to_anchor=bbox,
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    min_y = np.min(mean_ref)
    for exp, (k, mean_ps, _) in spectra_results.items():
        c = colors_dict.get(exp, None)
        ls = next(line_styles)

        axins.plot(k, mean_ps, color=c, linestyle=ls,
                   linewidth=2, alpha=0.8)
        min_y = min(min_y, np.min(mean_ps))

    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_xlim(hp, np.max(k) + (np.max(k)-hp)*0.05)
    axins.set_ylim(min_y*0.95, 1e6)
    customise_ax(ax=axins, top_right=True)

    axins.tick_params(axis="both", which="both",
                      length=0, labelbottom=False, labelleft=False)
    axins.grid(True, which="both", ls="--", lw=0.5,
               alpha=0.6)  # keep grid visible

    mark_inset(ax, axins, loc1=2, loc2=3,
               fc="none", ec="teal")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber", fontsize=14)
    ax.set_ylabel("Power spectral density", fontsize=14)

    ax.legend(frameon=True, fontsize=12, loc="lower left")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    customise_ax(ax=ax)
    plt.tight_layout()
    outpath = os.path.join(path, f"average_power_spectra.{figformat}")
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Average power spectra plot saved to {os.path.basename(outpath)}")
    # save relative_high_freq_delta to a CSV file
    with open(os.path.join(path, "relative_high_freq_delta.csv"), "w") as f:
        f.write("model,rel_hp_delta\n")
        for exp, delta in relative_high_freq_delta.items():
            f.write(f"{exp},{delta:.2f}\n")
    print(f"Relative high-frequency differences saved to 'relative_high_freq_delta.csv'")
    return relative_high_freq_delta


def plot_event_zoom(path, y_preds, y_true, exp_list, exp_names,
                    common_data, variable=None, title=None, label=None,
                    figformat='png', zoom_extent=[-8, 21, 34, 54], sel_criteria=["median", "std"]):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    lon = common_data["longitudes"]
    lat = common_data["latitudes"]
    dates = common_data["dates"]
    [lon_min, lon_max, lat_min, lat_max] = zoom_extent

    # Select indices inside this region
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    # Restrict y_true to this region
    y_true_region = y_true[:, lat_idx.min():lat_idx.max()+1,
                           lon_idx.min():lon_idx.max()+1]
    for cr in sel_criteria:
        if cr == "median":
            day_idx = np.argmax(np.median(y_true_region, axis=(1, 2)))
        elif cr == "mean":
            day_idx = np.argmax(np.mean(y_true_region, axis=(1, 2)))
        elif cr == "std":
            day_idx = np.argmax(np.std(y_true_region, axis=(1, 2)))
        else:  # specific date
            try:
                day_idx = np.where(dates == np.datetime64(cr))[0][0]
            except IndexError:
                print(f"Date {cr} not found in the dataset. Skipping.")
                continue
        print(
            f"Maximum {cr} event in the region: {dates[day_idx]}")

        # Extract the chosen day
        maps = {"Ground truth": y_true_region[day_idx, :, :]}
        for exp in exp_list:
            y_pred = y_preds[exp]
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.numpy()
            maps[exp_names[exp]] = y_pred[day_idx, lat_idx.min(
            ):lat_idx.max()+1, lon_idx.min():lon_idx.max()+1]

        # --- Color scale (shared) ---
        vmin = y_true[day_idx, :, :].min()
        vmax = y_true[day_idx, :, :].max()
        cmap_dict = get_cmap_norm(variable, vmin, vmax, lognorm=False)
        vmin = cmap_dict['vmin']
        vmax = cmap_dict['vmax']
        cmap = cmap_dict['cmap']
        norm = cmap_dict['norm']

        # --- Subplot config ---
        n_models = len(maps)
        ncols = 3
        nrows = int(np.ceil(n_models / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5*ncols, 3*nrows),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # Plot maps
        for ax, (name, data) in zip(axes.flat, maps.items()):
            im = ax.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                extent=zoom_extent, origin="upper"
            )
            ax.set_extent(zoom_extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4,
                           edgecolor="gray", linestyle="--")
            ax.add_feature(cfeature.RIVERS.with_scale(
                "50m"), linewidth=0.4, alpha=0.5)
            ax.set_title(name, fontsize=13, weight="bold")

        # Hide unused subplots
        for ax in axes.flat[len(maps):]:
            ax.axis("off")

        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.9, 0.07, 0.015, 0.82])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
        cbar.set_label(label, fontsize=16, labelpad=1)

        # leave space for colorbar on the right
        plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.94])

        tmp = "cr_" + cr + "_" if cr in ["median", "mean", "std"] else ""
        out_path = os.path.join(
            path, f"event_zoom_{tmp}{dates[day_idx].strftime('%Y-%m-%d')}.{figformat}")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.show()
        plt.close()

        print(
            f"{title} event map comparison saved to {os.path.basename(out_path)}")


def plot_event_full(path, y_preds, y_true, exp_list, exp_names, common_data, variable=None, title=None, label=None, figformat='png', sel_criteria=["median", "mean"]):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    extent = common_data["extent"]
    dates = common_data["dates"]
    n_models = len(exp_list) + 1  # +1 for ground truth
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    for cr in sel_criteria:
        if cr == "median":
            day_idx = np.argmax(np.median(y_true, axis=(1, 2)))
        elif cr == "mean":
            day_idx = np.argmax(np.mean(y_true, axis=(1, 2)))
        elif cr == "std":
            day_idx = np.argmax(np.std(y_true, axis=(1, 2)))
        else:  # specific date
            try:
                day_idx = np.where(dates == np.datetime64(cr))[0][0]
            except IndexError:
                print(f"Date {cr} not found in the dataset. Skipping.")
                continue

        print(
            f"Selected date in the region: {dates[day_idx]}")
        vmin = y_true[day_idx].min()
        vmax = y_true[day_idx].max()
        cmap_dict = get_cmap_norm(variable, vmin, vmax, lognorm=False)
        vmin = cmap_dict['vmin']
        vmax = cmap_dict['vmax']
        cmap = cmap_dict['cmap']
        norm = cmap_dict['norm']
        exp_maps = {"Ground truth": y_true[day_idx]}
        for exp in exp_list:
            y_pred = y_preds[exp]
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.numpy()
            exp_maps[exp_names[exp]] = y_pred[day_idx]

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5*ncols, 4.9*nrows),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        for ax, (name, data) in zip(axes.flat, exp_maps.items()):
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                           extent=extent, origin="upper")
            ax.set_title(name, fontsize=13, weight="bold")
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            # Features
            ax.add_feature(cfeature.COASTLINE.with_scale(
                "50m"), linewidth=0.6)
            ax.add_feature(cfeature.BORDERS.with_scale(
                "50m"), linewidth=0.4, edgecolor="gray", linestyle="--")
            ax.add_feature(cfeature.RIVERS.with_scale(
                "50m"), linewidth=0.4, alpha=0.5)

        for ax in axes.flat[len(exp_maps):]:
            ax.axis("off")

        cbar_ax = fig.add_axes([0.9, 0.07, 0.015, 0.82])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
        cbar.set_label(label, fontsize=16, labelpad=1)
        # leave space for colorbar on the right
        plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.94])
        tmp = "cr_" + cr + "_" if cr in ["median", "mean", "std"] else ""
        out_path = os.path.join(
            path, f"event_full_{tmp}{dates[day_idx].strftime('%Y-%m-%d')}.{figformat}")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.show()
        plt.close()
        print(
            f"{title} map comparison saved to {os.path.basename(out_path)}")


def plot_qq(path, y_preds, y_true, exp_list, exp_names, colors_dict, threshold=None, scales=["linear", "log"], var_title=None, unit=None, figformat='png'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    y_true_flat = y_true.flatten()
    if threshold is not None:
        y_true_flat = y_true_flat[y_true_flat > threshold]
    # Compute quantiles
    # smoother curve (200 instead of 100)
    quantiles = np.linspace(0, 1, 200)
    true_q = np.quantile(y_true_flat, quantiles)
    for scale in scales:
        plt.figure(figsize=(8, 7))
        for exp in exp_list:
            y_pred = y_preds[exp]
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.numpy()
            y_pred_flat = y_pred.flatten()
            if threshold is not None:
                y_pred_flat = y_pred_flat[y_pred_flat > threshold]
            pred_q = np.quantile(y_pred_flat, quantiles)

            # Plot
            c = colors_dict.get(exp, None)
            plt.plot(true_q, pred_q, color=c, linewidth=1.5, alpha=0.5, marker='o',
                     markersize=4, label=exp_names[exp])

        # Reference 1:1 line
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val],
                 '--', color="black", lw=1.5, alpha=0.7, label="$y=x$")

        # Axis scaling
        if scale == "log":
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(
                f"Ground truth {var_title} {unit}", fontsize=14)
            plt.ylabel(f"Predicted {var_title} {unit}", fontsize=14)
            title = f"Q-Q plot of {var_title} values"
            filename = f"qq_plot_log.{figformat}"
        else:
            plt.xlabel(
                f"Ground truth {var_title} {unit}", fontsize=14)
            plt.ylabel(f"Predicted {var_title} {unit}", fontsize=14)
            title = f"Q-Q plot of {var_title} values"
            filename = f"qq_plot.{figformat}"
        # ---- only left & bottom axes in teal ----
        customise_ax()
        # Titles & aesthetics
        plt.title(title, fontsize=16, pad=15)
        plt.legend(frameon=False, fontsize=13, framealpha=0.2,
                   markerscale=2, handlelength=3, handleheight=2, loc="upper left")
        # both major and minor grid on log-scale
        plt.grid(alpha=0.3, which="both")
        # enforce same aspect ratio for fair comparison
        # plt.axis("square")

        plt.tight_layout()
        plt.savefig(os.path.join(path, filename),
                    bbox_inches="tight", pad_inches=0, dpi=300)
        plt.show()
        plt.close()
        print(f"Q-Q plot saved to {filename}")


def plot_distributions(path, y_preds, y_true, exp_list, exp_names, results, colors_dict, var_title=None, unit=None, emd_unit=None, log_scale=False, figformat='png'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    y_true_flat = y_true.ravel()
    emd_unit = emd_unit or unit
    # Subplots
    ncols = 3
    nrows = int(np.ceil(len(exp_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5*ncols, 4*nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    # Bins
    bins = np.linspace(np.min(y_true_flat), np.max(y_true_flat), 50)
    i = 0
    for j, exp in enumerate(exp_list):
        ax = axes[j]
        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred_flat = y_pred.ravel()

        # Predictions histogram (coloured bars)
        ax.hist(
            y_pred_flat, bins=bins, density=True,
            alpha=0.6, color=colors_dict.get(exp, None)
        )

        # Ground truth histogram (black line, transparent fill)
        ax.hist(
            y_true_flat, bins=bins, density=True,
            histtype="step", linewidth=1.8, color="gray", label="Ground truth"
        )

        # Styling
        # get wasserstein distance
        w_dist = results[exp]["wasserstein"]
        # rel_w_dist = w_dist / np.std(y_true_flat)
        ax.set_title(f"{exp_names[exp]} | EMD={w_dist:.2f}" + r"$\,$" + emd_unit + r"$\,\downarrow$",
                     fontsize=12, pad=8, weight="bold")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5, which="both")

        if j % ncols == 0:
            ax.set_ylabel(
                f"Density", fontsize=12)
        if j >= (nrows-1)*ncols:
            ax.set_xlabel(rf"{var_title} [{unit}]", fontsize=12)
        customise_ax(ax=ax, minor=True)
        ax.legend(loc="upper right", fontsize=14, frameon=False)

    # Remove empty subplots if any
    for j in range(len(exp_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"distributions.{figformat}"),
                bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()
    plt.close()

    print(f"'distributions.{figformat}' saved")


def compute_power_spectrum(field):
    """
    Compute isotropic power spectrum of a 2D field.
    Returns (k, spectrum).
    """
    fft2 = np.fft.fftshift(np.fft.fft2(field))
    psd2D = np.abs(fft2)**2

    ny, nx = field.shape
    y, x = np.indices((ny, nx))
    center = (nx//2, ny//2)
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2).astype(int)

    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / np.maximum(nr, 1)

    return np.arange(len(radial_prof)), radial_prof


def average_power_spectrum(data):
    """Compute mean ± std power spectrum over time dimension (t, lat, lon)."""
    spectra = []
    for i in range(data.shape[0]):
        k, ps = compute_power_spectrum(data[i])
        spectra.append(ps)
    spectra = np.array(spectra)
    return k, spectra.mean(axis=0), spectra.std(axis=0)


def customise_ax(ax=None, tick_labelsize=12, minor=True, top_right=False):
    ax = plt.gca() if ax is None else ax
    # Remove top/right spines
    if not top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Set left/bottom spines color
    ax.spines["left"].set_color("teal")
    ax.spines["bottom"].set_color("teal")
    if top_right:
        ax.spines["top"].set_color("teal")
        ax.spines["right"].set_color("teal")
    # Axis label and title colors
    ax.xaxis.label.set_color("teal")
    ax.yaxis.label.set_color("teal")
    ax.title.set_color("teal")
    # Tick colors and size
    # ax.ticklabel_format(useOffset=False)
    ax.tick_params(axis="x", colors="teal",
                   labelsize=tick_labelsize, width=1.2)
    ax.tick_params(axis="y", colors="teal",
                   labelsize=tick_labelsize, width=1.2)
    # Minor ticks
    if minor:
        ax.minorticks_on()
        ax.tick_params(which='minor', length=4, color='teal')


def get_fraction(x, latex=True):
    """Convert a float to a fraction string."""
    if x == 0:
        return "0"
    from fractions import Fraction
    frac = Fraction(x).limit_denominator()
    if latex:
        return f"$\\frac{{{frac.numerator}}}{{{frac.denominator}}}$" if frac.denominator != 1 else str(frac.numerator)
    else:
        # Return as a string in the form "numerator/denominator" or just "numerator" if denominator is 1
        return f"{frac.numerator}/{frac.denominator}" if frac.denominator != 1 else str(frac.numerator)
