import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from matplotlib import rcParams
from evaluation.utils import customise_ax

# --- Plot style ---
rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2
})

# --- Generate skewed distribution ---
a = 4  # skewness parameter
y_min, y_max = 0, 4
y = np.linspace(y_min, y_max, 1000)
pdf = skewnorm.pdf(y, a)
pdf /= np.trapz(pdf, y)  # Normalize PDF

# Prediction range
y_pred = np.linspace(y_min, y_max, 500)

# Values of p
p_list = [1, 2, 10, 50, 100, 500]
colors = plt.cm.plasma(np.linspace(0, 1, len(p_list)))

# --- Setup plot ---
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Plot distribution (PDF)
ax1.plot(y, pdf, color="black", linestyle="--", label="PDF")

# Plot losses
for i, p in enumerate(p_list):
    losses = []
    for yp in y_pred:
        loss = np.sum(np.abs(y - yp)**p * pdf) * (y[1] - y[0])
        losses.append(loss)
    losses = np.log(np.array(losses))
    losses /= np.max(losses)  # Normalize

    # Minimiser
    y_star = y_pred[np.argmin(losses)]
    q = np.sum(pdf * (y <= y_star)) * (y[1] - y[0])  # CDF at minimiser

    ax2.plot(y_pred, losses, label=rf"$p={p}$, $\hat y^*={y_star:.2f}$ (q={100*q:.0f}%)",
             color=colors[i], alpha=0.9)
    ax2.scatter([y_star], [np.min(losses)], color=colors[i],
                s=60, marker="o", edgecolor="k", zorder=5)

# --- Reference stats ---
mean = np.sum(y * pdf) * (y[1] - y[0])
median = y[np.where(np.cumsum(pdf)/np.sum(pdf) >= 0.5)[0][0]]
mode = y[np.argmax(pdf)]

# --- Formatting ---
ax1.set_ylabel(r"$p(y)$", fontsize=16)
ax2.set_ylabel(r"Normalized $\log(\mathbb{E}[|y - \hat{y}|^p])$", fontsize=16)

ax1.set_title(
    rf"$\mathcal{{L}}_p$ minimisers | median={median:.2f}, mean={mean:.2f}, mode={mode:.2f}",
    fontsize=16, pad=15
)

ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=False)
ax1.set_ylim(0, np.max(pdf) + 0.2)
ax1.grid(True, linestyle=":", linewidth=1)
customise_ax(ax1, tick_labelsize=12, minor=True, top_right=False)
customise_ax(ax2, tick_labelsize=12, minor=True, top_right=False)
fig.tight_layout()
plt.savefig("data/plot_path/lp_loss_minimisers.png",
            dpi=300, bbox_inches="tight")
plt.show()
