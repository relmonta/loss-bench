
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from evaluation.utils import customise_ax
from scipy.stats import wasserstein_distance
import seaborn as sns
import torch


def plot_temporal_evolution(path, y_preds, y_true, exp_list, exp_names,
                            dates, start_date_plot, end_date_plot, colors_dict, figformat='png'):
    """
    Plot the temporal evolution of maximum precipitation for different experiments.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    fig, axes = plt.subplots((len(exp_list) + 1)//2, 2, figsize=(
        20, 1.5*len(exp_list)), sharex=False, sharey=False)

    if axes.ndim > 1:
        axes = axes.flatten()

    for j, exp in enumerate(exp_list):
        ax = axes[j]
        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred_max = y_pred.reshape(y_pred.shape[0], -1).max(axis=1)
        y_true_max = y_true.reshape(y_true.shape[0], -1).max(axis=1)

        mask = (dates >= start_date_plot) & (dates <= end_date_plot)
        dates_sel = dates[mask]
        y_pred_sel = y_pred_max[mask]
        y_true_sel = y_true_max[mask]

        # plot ground truth
        ax.plot(
            dates_sel, y_true_sel,
            label="Ground truth",
            color=colors_dict.get("truth", "gray"),
            linewidth=1.8, alpha=0.9,
            marker='o', markersize=5
        )

        # plot model
        c = colors_dict.get(exp, None)
        ax.plot(
            dates_sel, y_pred_sel,
            label="Model pred", color=c,
            linestyle="--",
            linewidth=1.5, alpha=0.7,
            marker='o', markersize=5
        )

        ax.set_title(exp_names[exp], fontsize=16, pad=10)
        if j % 2 == 0:
            ax.set_ylabel("Max precip. [mm/day]", fontsize=12)
        ax.grid(alpha=0.3, which="both")
        customise_ax(ax=ax, minor=False)
        ax.legend(frameon=True, fontsize=13,
                  loc="upper right", ncols=2, framealpha=0.4)

    for ax in axes:
        ax.tick_params(axis="x", rotation=10, labelsize=12)
        plt.setp(ax.get_xticklabels(), ha="right")

    for ax in axes[j+1:]:
        ax.remove()

    plt.tight_layout()
    plt.savefig(os.path.join(
        path, f"temporal_evolution.{figformat}"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()
    plt.close()
    print(
        f"Temporal evolution plot saved to temporal_evolution.{figformat}")


def plot_rx1day(path, df, rx1day_true, exp_names, color_dict, figformat='png'):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    bar_width = 0.6
    color_palette = [color_dict.get(exp, "gray") for exp in df.index]
    # change df index to exp_names
    df.index = [exp_names.get(exp, exp) for exp in df.index]
    # Bars
    bars = ax.bar(
        x, df["rx1day"], width=bar_width, color=color_palette, edgecolor="gray", linewidth=0.6, alpha=0.75
    )

    # Ground truth line
    ax.axhline(rx1day_true, color="gray",
               linestyle="--", linewidth=2, label="Ground truth")

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height +
            0.02*max(df["rx1day"]),
            f"{height:.1f} mm", ha="center", va="bottom", fontsize=8.5, color="teal"
        )

    # Style
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=-20, fontsize=8, ha="left")
    ax.set_ylabel("Rx1day value (mm)", fontsize=13)
    ax.set_title(
        "Rx1day: Largest daily maximum value", fontsize=15, pad=15)

    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    customise_ax(ax=ax, minor=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"rx1day_comparison.{figformat}"),
                bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()
    plt.close()

    print(
        f"Rx1day comparison plot saved to 'Rx1day_comparison.{figformat}'")


def plot_distributions_max(path, y_preds, y_true, exp_list, exp_names, colors_dict, figformat='png'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    y_true_flat = y_true.max(axis=(1, 2))

    # Subplots
    ncols = 3
    nrows = int(np.ceil(len(exp_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5*ncols, 4*nrows), sharex=True, sharey=True)
    axes = axes.flatten()
    emd_scores = {}
    # Bins
    bins = np.linspace(np.min(y_true_flat), np.max(y_true_flat), 50)
    for j, exp in enumerate(exp_list):
        ax = axes[j]
        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred_flat = y_pred.max(axis=(1, 2))

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
        # get wasserstein distance
        w_dist = wasserstein_distance(y_true_flat, y_pred_flat)
        emd_scores[exp] = w_dist
        ax.set_title(f"{exp_names[exp]} | EMD={w_dist:.2f}" + r"$\,mm\,\downarrow$",
                     fontsize=12, pad=8, weight="bold")
        ax.set_yscale("log")
        # minor grid on y axis
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5, which="both")

        if j % ncols == 0:
            ax.set_ylabel("Density", fontsize=12)
        if j >= (nrows-1)*ncols:
            ax.set_xlabel("Precipitation [mm/day]", fontsize=12)
        customise_ax(ax=ax, minor=True)
        ax.legend(loc="upper right", fontsize=14, frameon=False)

    # Remove empty subplots if any
    for j in range(len(exp_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(
        path, f"distributions_max.{figformat}"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()
    plt.close()
    print(f"'distributions_max.{figformat}' saved")
    return emd_scores


def plot_violin_max(path, y_preds, y_true, exp_list, exp_names, color_dict, figformat='png'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()

    # Ground truth maxima
    y_true_max = y_true.max(axis=(1, 2))

    # Model maxima
    records = []
    for exp in exp_list:
        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred_max = y_pred.max(axis=(1, 2))
        records.extend([(val, exp_names[exp]) for val in y_pred_max])
    df_models = pd.DataFrame(records, columns=["Precipitation", "Model"])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Background ground truth violin for each model position ---
    positions = np.arange(len(exp_list))
    parts = ax.violinplot(
        [y_true_max for _ in exp_list],  # repeat GT for each model
        positions=positions,
        showmeans=False, showextrema=False, showmedians=False
    )
    for pc in parts['bodies']:
        pc.set_facecolor("lightgray")
        pc.set_alpha(0.5)
        pc.set_edgecolor("gray")

    # --- Overlay model violins ---
    sns.violinplot(
        data=df_models, x="Model", y="Precipitation",
        palette={exp_names[exp]: color_dict[exp] for exp in exp_list},
        inner=None, cut=0, alpha=0.5, width=0.6, ax=ax, hue="Model"
    )

    # --- Scatter points ---
    sns.stripplot(
        data=df_models, x="Model", y="Precipitation",
        jitter=0.15, alpha=0.25, size=4,
        palette={exp_names[exp]: color_dict[exp] for exp in exp_list},
        edgecolor="none", ax=ax, hue="Model"
    )

    # Styling
    ax.set_ylabel("Precipitation [mm/day]", fontsize=14)
    ax.set_title("Daily maximum precipitation values", fontsize=18, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-20,
                       fontsize=12, ha="left")
    ax.grid(alpha=0.3, axis="y")

    customise_ax(minor=False)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"violin_plot_max.{figformat}"),
                bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()
    plt.close()

    print(f"'violin_plot_max.{figformat}' saved")


def plot_div_plot(path, y_preds, y_true, exp_list, exp_names, estimation_threshold=10, figformat='png'):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    stats = []
    # Ground truth max
    y_true_max = y_true.max(axis=(1, 2))
    div_scores = {}
    for exp in exp_list:
        y_pred = y_preds[exp]
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred_max = y_pred.max(axis=(1, 2))
        diff = y_pred_max - y_true_max
        n_days = len(diff)

        under = np.sum(diff < -estimation_threshold) / n_days * 100
        over = np.sum(diff > estimation_threshold) / n_days * 100

        stats.append(
            {"Model": exp_names[exp], "Under": -under, "Over": over})
        div_scores[exp] = {"under": under, "over": over}

    df_stats = pd.DataFrame(stats)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_stats))
    bar_width = 0.5
    colors = {"Under": "#e74c3c", "Over": "#3498db"}

    # Bars
    bars_under = ax.bar(x, df_stats["Under"], bar_width,
                        color=colors["Under"], label=f"Underestimation > {estimation_threshold} mm", alpha=0.75)
    bars_over = ax.bar(x, df_stats["Over"], bar_width,
                       color=colors["Over"], label=f"Overestimation > {estimation_threshold} mm", alpha=0.75)

    # Add values as labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        height + (1.5 if height > 0 else -1.5),
                        f"{abs(height):.1f}%",
                        ha="center", va="bottom" if height > 0 else "top",
                        fontsize=10, color="black")

    add_labels(bars_under)
    add_labels(bars_over)

    # Style
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df_stats["Model"],
                       rotation=-20, fontsize=8.5, ha="left")
    ax.set_ylabel("Percentage of days [%]", fontsize=14)
    ax.set_title(
        f"Frequency of daily maximum precipitation over- and underestimation",
        fontsize=15, pad=15
    )

    ax.legend(frameon=True, fontsize=11)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    customise_ax(ax=ax, minor=False)

    plt.tight_layout()
    plt.savefig(os.path.join(
        path, f"misestimation_{estimation_threshold}.{figformat}"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.show()
    plt.close()
    print(
        f"Bias frequency bar plot saved to 'misestimation_{estimation_threshold}.{figformat}'")
    return div_scores
