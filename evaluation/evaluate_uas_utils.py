
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from evaluation.utils import customise_ax
from scipy.stats import wasserstein_distance
import torch


def plot_temporal_evolution(path, y_preds, y_true, exp_list, exp_names,
                            dates, start_date_plot, end_date_plot, colors_dict, figformat='png'):
    """
    Plot the temporal evolution of maximum zonal wind over value the domain for each model and ground truth.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    fig, axes = plt.subplots((len(exp_list) + 1)//2, 2, figsize=(
        20, 1.5*len(exp_list)), sharex=False, sharey=False)

    if axes.ndim > 1:
        axes = axes.flatten()

    for j, exp in enumerate(exp_list):
        ax = axes[j]
        mask = (dates >= start_date_plot) & (dates <= end_date_plot)
        dates_sel = dates[mask]

        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        y_true_flat = y_true.reshape(y_true.shape[0], -1)

        # Eastward (max) and westward (min) daily extremes
        y_pred_east_sel = y_pred_flat.max(axis=1)[mask]
        y_true_east_sel = y_true_flat.max(axis=1)[mask]
        # if < 0 set to 0
        y_true_east_sel[y_true_east_sel < 0] = 0
        y_pred_east_sel[y_pred_east_sel < 0] = 0

        y_pred_west_sel = y_pred_flat.min(axis=1)[mask]
        y_true_west_sel = y_true_flat.min(axis=1)[mask]
        # if > 0 set to 0
        y_true_west_sel[y_true_west_sel > 0] = 0
        y_pred_west_sel[y_pred_west_sel > 0] = 0

        # --- Plot ground truth ---
        ax.plot(
            dates_sel, y_true_east_sel,
            label="Ground truth (east)",
            color=colors_dict["truth"],
            linewidth=1.8, alpha=0.9,
            marker='o', markersize=5
        )
        ax.plot(
            dates_sel, y_true_west_sel,
            label="Ground truth (west)",
            color=colors_dict["truth"],
            linewidth=1.8, alpha=0.9,
            marker='s', markersize=4
        )

        # --- Plot model predictions ---
        ax.plot(
            dates_sel, y_pred_east_sel,
            label=f"{exp_names[exp]} (east)", color=colors_dict[exp],
            linestyle="--", linewidth=1.5, alpha=0.7,
            marker='o', markersize=5
        )
        ax.plot(
            dates_sel, y_pred_west_sel,
            label=f"{exp_names[exp]} (west)", color=colors_dict[exp],
            linestyle="--", linewidth=1.5, alpha=0.7,
            marker='s', markersize=4
        )

        # --- Labels and formatting ---
        ax.set_title(
            f"Daily zonal wind extremes over time | {exp_names[exp]}",
            fontsize=14, pad=10
        )
        ax.set_ylabel(r"Zonal wind [m.s$^{-1}$]", fontsize=12)
        ax.grid(alpha=0.3, which="both")
        customise_ax(ax=ax, minor=False)
        ax.legend(frameon=True, fontsize=12, loc="upper right")

    for ax in axes:
        ax.tick_params(axis="x", rotation=10, labelsize=12)
        plt.setp(ax.get_xticklabels(), ha="right")

    for ax in axes[j+1:]:
        ax.remove()

    plt.tight_layout()
    plt.savefig(os.path.join(
        path, f"temporal_evolution.{figformat}"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    print(
        f"Temporal evolution plot saved to 'temporal_evolution.{figformat}'")


def plot_distributions_max(path, y_preds, y_true, exp_list, exp_names, colors_dict, figformat='png'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    emd_scores = {}
    # Ground truth daily maximum and minimum zonal wind
    y_true_east = np.maximum(y_true.max(axis=(1, 2)), 0)
    y_true_west = np.minimum(y_true.min(axis=(1, 2)), 0)
    y_true_flat = np.concatenate([y_true_east, y_true_west])

    # Subplots
    ncols = 3
    nrows = int(np.ceil(len(exp_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5*ncols, 4*nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    # Bins
    bins = np.linspace(np.min(y_true_flat), np.max(y_true_flat), 100)
    i = 0
    for j, exp in enumerate(exp_list):
        ax = axes[j]
        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred_east = np.maximum(y_pred.max(axis=(1, 2)), 0)
        y_pred_west = np.minimum(y_pred.min(axis=(1, 2)), 0)
        y_pred_flat = np.concatenate([y_pred_east, y_pred_west])

        # Predictions histogram (coloured bars)
        ax.hist(
            y_pred_flat, bins=bins, density=True,
            alpha=0.6, color=colors_dict[exp]
        )

        # Ground truth histogram (black line, transparent fill)
        ax.hist(
            y_true_flat, bins=bins, density=True,
            histtype="step", linewidth=1.8, color="gray", label="Ground truth"
        )

        # Styling
        w_dist = wasserstein_distance(y_true_flat, y_pred_flat)
        emd_scores[exp] = w_dist
        ax.set_title(f"{exp_names[exp]} | EMD={w_dist:.2f}" + r"$\,m.s^{-1}\,\downarrow$",
                     fontsize=12, pad=8, weight="bold")
        # ax.set_yscale("log")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

        if j % ncols == 0:
            ax.set_ylabel("Density", fontsize=12)
        if j >= (nrows-1)*ncols:
            ax.set_xlabel(r"Zonal wind [m.s$^{-1}$]", fontsize=12)
        customise_ax(ax=ax, minor=False)
        ax.legend(fontsize=14, frameon=False)

    # Remove empty subplots if any
    for j in range(len(exp_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(
        path, f"distributions_max_uwind.{figformat}"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    print(f"'distributions_max_uwind.{figformat}' saved")
    return emd_scores


def plot_div_plot(path, y_preds, y_true, exp_list, exp_names, estimation_thresholds=[1.0, 2.0, 3.0], figformat='png'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    e_div_scores = {}
    for estimation_threshold in estimation_thresholds:
        stats = []
        e_div_scores[estimation_threshold] = {}
        # Ground truth daily maximum and minimum zonal wind
        u_true_max = y_true.max(axis=(1, 2))  # eastward
        # if max < 0, set to 0
        u_true_max[u_true_max < 0] = 0
        u_true_min = y_true.min(axis=(1, 2))  # westward
        # if min > 0, set to 0
        u_true_min[u_true_min > 0] = 0

        for exp in exp_list:
            y_pred = y_preds[exp]
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.numpy()
            u_pred_max = y_pred.max(axis=(1, 2))
            # if max < 0, set to 0
            u_pred_max[u_pred_max < 0] = 0
            u_pred_min = y_pred.min(axis=(1, 2))
            # if min > 0, set to 0
            u_pred_min[u_pred_min > 0] = 0

            diff_max = u_pred_max - u_true_max
            diff_min = u_pred_min - u_true_min

            n_days = len(diff_max)

            # percentage of days exceeding threshold
            over_max = np.sum(
                diff_max > estimation_threshold) / n_days * 100
            under_max = np.sum(
                diff_max < -estimation_threshold) / n_days * 100
            over_min = np.sum(
                diff_min < -estimation_threshold) / n_days * 100
            under_min = np.sum(
                diff_min > estimation_threshold) / n_days * 100

            stats.append({
                "Model": exp_names[exp],
                "East Under": -under_max,
                "East Over": over_max,
                "West Under": -under_min,
                "West Over": over_min
            })
            e_div_scores[estimation_threshold][exp] = {
                "under": under_max, "over": over_max}

        df_stats = pd.DataFrame(stats)

        # Bar width
        bar_width = 0.3
        x = np.arange(len(df_stats))

        # Colors
        colors = {
            "East Under": "#e74c3c",  # red
            "West Under": "#f39c12",   # orange
            "East Over": "#00b4be",  # blue
            "West Over": "#ae01f3"    # green
        }

        fig, ax = plt.subplots(figsize=(12, 6))

        # Ensure correct signs: east under <0, east over >0, west under <0, west over >0
        east_under = -np.abs(df_stats["East Under"])
        east_over = np.abs(df_stats["East Over"])
        west_under = -np.abs(df_stats["West Under"])
        west_over = np.abs(df_stats["West Over"])

        # Plot bars
        bars = [
            ax.bar(x - 0.5*bar_width, east_under, bar_width,
                   color=colors["East Under"], label="Underestimation (E)"),
            ax.bar(x - 0.5*bar_width, east_over, bar_width,
                   color=colors["East Over"], label="Overestimation (E)"),
            ax.bar(x + 0.5*bar_width, west_under, bar_width,
                   color=colors["West Under"], label="Underestimation (W)"),
            ax.bar(x + 0.5*bar_width, west_over, bar_width,
                   color=colors["West Over"], label="Overestimation (W)")
        ]

        # Add labels
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                if height != 0:
                    va = "bottom" if height > 0 else "top"
                    offset = 0  # height/100 if height >= 0 else -height/100
                    ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                            f"{abs(height):.1f}", ha="center", va=va, fontsize=14)

        # Style
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(df_stats["Model"],
                           rotation=-20, fontsize=12, ha="left")
        ax.set_ylabel("Percentage of days [%]", fontsize=14)
        ax.set_title(
            f"Frequency of daily E/W maximum value over- and underestimation of {estimation_threshold} " + r"m.s$^{-1}$", fontsize=15, pad=15)
        ax.legend(frameon=True, fontsize=14, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        customise_ax(ax=ax, minor=False)

        plt.tight_layout()
        out_file = os.path.join(
            path, f"misestimation_{estimation_threshold}.{figformat}")
        plt.savefig(out_file, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
        print(
            f"Bias frequency plot saved to {os.path.basename(out_file)}")
    return e_div_scores
