from torch.distributions.gamma import Gamma
from pytorch_msssim import ssim
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DTCWTForward
from scipy.stats import wasserstein_distance
import numpy as np
import warnings


class SpectralLoss(nn.Module):
    def __init__(self, p=1, calibration=5e-3, f_min=None, f_max=None,
                 use_mask=False, use_power=False, reduction='mean', energy_pct=None):
        """
        Spectral loss with frequency range restriction or energy-based high-pass filtering.

        Args:
            p (int): Order of Lp norm (1 for L1, 2 for L2, etc).
            calibration (float): Global scaling factor for the loss.
            f_min (float): Minimum normalised frequency (0 to 1, relative to Nyquist).
            f_max (float): Maximum normalised frequency.
            use_mask (bool): Whether to apply a frequency-domain mask.
            use_power (bool): If True, use power spectrum (|FFT|^2), else magnitude.
            reduction (str): 'mean' (default), 'sum', or 'none'.
            energy_pct (float or None): If set (0–100), applies a high-pass filter keeping
                                        only frequencies that contribute at least this % of energy.
        """
        super().__init__()
        self.p = p
        self.calibration = calibration
        self.f_min = f_min
        self.f_max = f_max
        self.use_mask = use_mask
        self.use_power = use_power
        self.reduction = reduction
        self.energy_pct = energy_pct  # in %

        self.mask = None

    def _compute_energy_mask(self, spectrum, freq_r):
        """
        Builds a high-pass frequency mask that retains a target energy percentage.
        Computes the mask per batch and per channel (B, C, H, W).

        Args:
            spectrum (Tensor): |FFT| or |FFT|², shape (B, C, H, W)
            freq_r (Tensor): Radial frequency, shape (B, C, H, W)
        Returns:
            mask (Tensor): Binary mask, shape (B, C, H, W)
        """
        B, C, H, W = spectrum.shape
        flat_spec = spectrum.view(B, C, -1)
        flat_freq = freq_r.view(B, C, -1)

        # Sort by ascending frequency
        sorted_freq, sorted_idx = torch.sort(flat_freq, dim=-1)
        sorted_energy = flat_spec.gather(-1, sorted_idx)

        cumsum_energy = torch.cumsum(sorted_energy, dim=-1)
        total_energy = cumsum_energy[:, :, -1]
        target_energy = total_energy * (self.energy_pct / 100.0)

        # Find the frequency threshold that achieves the target energy
        over_target = cumsum_energy >= target_energy.unsqueeze(-1)
        idx_cutoff = over_target.float().argmax(dim=-1)  # shape (B, C)
        freq_cutoff = sorted_freq.gather(-1,
                                         idx_cutoff.unsqueeze(-1)).squeeze(-1)

        # Broadcast and apply the frequency threshold to build mask
        freq_cutoff = freq_cutoff.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        mask = (freq_r >= freq_cutoff).float()  # High-pass

        return mask

    def _update_freq_mask(self, target, spec_ref=None):
        H, W = target.shape[-2:]
        device = target.device

        freq_y = torch.fft.fftfreq(H, d=1.0, device=device)
        freq_x = torch.fft.fftfreq(W, d=1.0, device=device)
        fx, fy = torch.meshgrid(freq_x, freq_y, indexing='ij')
        freq_r = torch.sqrt(fx ** 2 + fy ** 2)
        # expand to match target shape (B, C, H, W)
        freq_r = freq_r.unsqueeze(0).unsqueeze(0).expand(
            target.shape[0], target.shape[1], H, W)

        if self.energy_pct is not None:
            if spec_ref is None:
                raise ValueError(
                    "spec_ref must be provided when energy_pct is set")
            # Use reference spectrum to compute energy-based mask
            energy_mask = self._compute_energy_mask(spec_ref, freq_r)
            # Expand mask to include batch and channel axes
            self.mask = energy_mask.to(device)
        else:
            # Range-based frequency mask
            fmin_abs = self.f_min * freq_r.max()
            fmax_abs = self.f_max * freq_r.max()
            mask = (freq_r >= fmin_abs) & (freq_r <= fmax_abs)
            self.mask = mask.float().to(device)

    def forward(self, pred, target):
        assert pred.shape == target.shape, "pred and target must have the same shape"

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        if self.use_power:
            pred_spec = torch.abs(pred_fft) ** 2
            target_spec = torch.abs(target_fft) ** 2
        else:
            pred_spec = torch.abs(pred_fft)
            target_spec = torch.abs(target_fft)

        # Update mask if shape changed or not set
        if self.use_mask:
            # print(
            #     f"Using mask: {self.use_mask}, cached shape: {self._cached_shape}, current shape: {(H, W)}")
            #
            if self.mask is None or self.energy_pct is not None:
                self._update_freq_mask(target, spec_ref=target_spec)

            mask = self.mask
            while mask.dim() < pred_spec.dim():
                mask = mask.unsqueeze(0)
            pred_spec = pred_spec * mask
            target_spec = target_spec * mask

        # Compute loss
        loss = (pred_spec - target_spec).abs() ** self.p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return self.calibration * loss


class LossCombination(nn.Module):
    def __init__(self, losses, lambdas):
        super(LossCombination, self).__init__()
        self.losses = []
        self.lambdas = []
        assert len(losses) == len(
            lambdas), "Number of losses and lambdas must match"

        for i, loss in enumerate(losses):
            self.losses.append(get_criterion(loss, **losses[loss]))
            self.lambdas.append(lambdas[i])
        self.losses = nn.ModuleList(self.losses)

    def forward(self, pred, target, idx_pr=None):
        loss = 0
        for i, loss_fn in enumerate(self.losses):
            if isinstance(pred, list):  # Output are Bernoulli-Gamma parameters
                if isinstance(loss_fn, NLLBernoulliGammaLoss):
                    loss += self.lambdas[i] * \
                        loss_fn([p.clone() for p in pred], target)
                else:
                    # Compute the expected value
                    if len(pred) != 3:
                        raise ValueError(
                            "Expected pred to be a list of 3 elements for NLLBernoulliGammaLoss")
                    expected_value = pred[0] * pred[1] * pred[2]
                    if isinstance(loss_fn, WaveletLoss):
                        expected_value = expected_value.to(dtype=pred[0].dtype)
                    loss += self.lambdas[i] * \
                        loss_fn(expected_value.clone(), target)
            else:
                loss += self.lambdas[i] * loss_fn(pred.clone(), target)
        return loss


class FilteredLpLoss(nn.Module):
    """
    Computes the Lp loss after applying Gaussian smoothing to both predictions and targets.
    """

    def __init__(self, sigma=2.0, p=2, calibration=1):
        """
        Initializes the Gaussian kernel.

        Args:
        - sigma (float): Standard deviation of the Gaussian distribution.
        - p (int): Order of the Lp norm (default is 2 for L2 loss).
        """
        super(FilteredLpLoss, self).__init__()
        self.sigma = sigma
        self.p = p
        self.calibration = calibration
        # Kernel size based on sigma
        self.kernel_size = int(6 * sigma + 1)

        # Create a 1D Gaussian kernel
        coords = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        g_kernel = torch.exp(-coords**2 / (2 * sigma**2))
        g_kernel /= g_kernel.sum()  # Normalize

        # Convert to 2D separable kernel
        g_kernel_2d = g_kernel[:, None] * g_kernel[None, :]
        # Save kernel as buffer
        self.register_buffer("g_kernel_2d", g_kernel_2d[None, None, :, :])

    def gaussian_blur(self, x):
        """
        Applies a Gaussian blur using a 2D convolution.

        Args:
        - x (tensor): Input tensor (batch_size, channels, height, width).

        Returns:
        - Smoothed tensor.
        """
        _, channels, _, _ = x.shape
        kernel = self.g_kernel_2d.expand(
            channels, 1, self.kernel_size, self.kernel_size).to(x.dtype).to(x.device)
        padding = self.kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=channels)

    def forward(self, pred, target):
        """
        Args:
        - pred (tensor): Predicted values (batch_size, channels, height, width).
        - target (tensor): Ground truth values.

        Returns:
        - Filtered MSE loss.
        """
        pred_smooth = self.gaussian_blur(pred)
        target_smooth = self.gaussian_blur(target)
        return self.calibration * torch.mean(
            (pred_smooth - target_smooth).abs() ** self.p)


class WaveletLoss(nn.Module):
    def __init__(self, wavelet='sym4', n_levels=4, p=1,
                 limited_levels=None, calibration=1e-1, complex=False,
                 biort='near_sym_b', qshift='qshift_b'):
        """
        Wavelet-based loss function using pytorch_wavelets.

        Parameters
        ----------
        wavelet : str
            Type of wavelet for DWT (e.g., 'haar', 'db4', 'bior3.5').
        n_levels : int
            Number of decomposition levels.
        p : int
            Base Lp loss function (1=L1, 2=L2, ...).
        limited_levels : list or None
            List of levels to take into account (0=low-pass, 1.., n_levels=detail).
        calibration : float
            Global scaling factor for the loss.
        complex : bool
            If True, uses Dual-Tree Complex Wavelet Transform (DTCWT).
        biort, qshift : str
            Filters for DTCWT.
        """
        super(WaveletLoss, self).__init__()

        if complex:
            self.wtransform = DTCWTForward(
                J=n_levels, biort=biort, qshift=qshift)
        else:
            self.wtransform = DWTForward(
                J=n_levels, wave=wavelet, mode='symmetric')

        self.p = p
        self.calibration = calibration

        if limited_levels is None:
            self.levels_to_use = list(range(n_levels + 1))  # include low-pass
        else:
            # remove out-of-range levels and sort
            # check levels are in range
            for l in limited_levels:
                if l < 0 or l > n_levels:
                    warnings.warn(
                        f"Level {l} is out of range for n_levels={n_levels}, ignoring.")

            levels_to_use = [l for l in limited_levels if 0 <= l <= n_levels]
            if len(levels_to_use) == 0:
                raise ValueError("No valid levels provided for wavelet loss.")
            self.levels_to_use = sorted(set(levels_to_use))

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        yl_pred, yh_pred = self.wtransform(pred)
        yl_target, yh_target = self.wtransform(target)

        loss = 0.0
        for i in self.levels_to_use:
            if i == 0:
                # low-pass coefficients
                loss += torch.mean((yl_pred - yl_target).abs() ** self.p)
            else:
                diff = yh_pred[i - 1] - yh_target[i - 1]
                loss += torch.mean(diff.abs() ** self.p)

        loss /= len(self.levels_to_use)
        return self.calibration * loss


class SSIMLoss(nn.Module):
    def __init__(self, calibration=30):
        super(SSIMLoss, self).__init__()
        self.calibration = calibration

    def forward(self, pred, target):
        dr = target.max() - target.min()
        loss = 1 - ssim(pred.clone(), target, data_range=dr, size_average=True)
        return self.calibration * loss


class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss (GDL) using a configurable finite difference stencil (5 or 7).
    """

    def __init__(self, p=1, kernel_size=5, calibration=1):
        """
        Args:
            p (int): Norm degree (1 for L1, 2 for L2, etc.)
            kernel_size (int): Size of the finite difference stencil (must be 3, 5, or 7)
        """
        super(GradientDifferenceLoss, self).__init__()
        assert kernel_size in [
            3, 5, 7], "Only kernel sizes 3, 5, or 7 are supported."
        self.p = p
        self.kernel_size = kernel_size
        self.calibration = calibration
        self.register_kernels()

    def register_kernels(self):
        # Define 1D finite difference kernels based on standard central difference weights
        if self.kernel_size == 3:
            dx = torch.tensor([[-1, 0, 1]], dtype=torch.float32) / 2
        elif self.kernel_size == 5:
            dx = torch.tensor([[1, -8, 0, 8, -1]], dtype=torch.float32) / 12
        elif self.kernel_size == 7:
            dx = torch.tensor([[-1, 9, -45, 0, 45, -9, 1]],
                              dtype=torch.float32) / 60

        # Vertical kernel is transpose of horizontal
        dy = dx.T

        # Register as buffers to put on correct device automatically
        self.register_buffer("kernel_x", dx[None, None, :, :])
        self.register_buffer("kernel_y", dy[None, None, :, :])

    def compute_gradient(self, x, kernel):
        B, C, H, W = x.shape
        kernel = kernel.expand(C, 1, *kernel.shape[-2:])
        kernel = kernel.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, kernel,
                        padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2),
                        groups=C)

    def forward(self, pred, target):
        grad_pred_x = self.compute_gradient(pred, self.kernel_x)
        grad_pred_y = self.compute_gradient(pred, self.kernel_y)
        grad_targ_x = self.compute_gradient(target, self.kernel_x)
        grad_targ_y = self.compute_gradient(target, self.kernel_y)

        diff_x = torch.abs(grad_pred_x - grad_targ_x)
        diff_y = torch.abs(grad_pred_y - grad_targ_y)

        loss = torch.mean(diff_x ** self.p + diff_y ** self.p)
        return self.calibration * loss


class WassersteinDistance(nn.Module):
    """
    Wasserstein distance loss function.
    """

    def __init__(self, divide_by_std=False):
        super(WassersteinDistance, self).__init__()
        self.divide_by_std = divide_by_std

    def forward(self, pred, target):
        """
        Compute the Wasserstein distance between two distributions.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (B, C, H, W)
            target (torch.Tensor): Ground truth tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Computed Wasserstein distance.
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().detach().numpy()
        pred = pred.flatten()
        target = target.flatten()
        std_target = np.std(target)
        if self.divide_by_std:
            if std_target < 1e-6:
                std_target = 1e-6
            return wasserstein_distance(pred, target) / std_target
        return wasserstein_distance(pred, target)


class CalibratedL1Loss(nn.Module):
    """
    Calibrated L1 loss function with a calibration factor.
    """

    def __init__(self, calibration=1):
        """
        Args:
            calibration (float): Calibration factor to scale the loss.
        """
        super(CalibratedL1Loss, self).__init__()
        self.calibration = calibration

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (Tensor): Predicted values (B, C, H, W)
            y_true (Tensor): Ground truth values (B, C, H, W)
        Returns:
            loss (Tensor): Scalar loss
        """
        assert y_pred.shape == y_true.shape, "Shape mismatch"
        loss = torch.abs(y_pred - y_true).mean()
        return self.calibration * loss


class CalibratedL2Loss(nn.Module):
    """
    Calibrated L2 loss function with a calibration factor.
    """

    def __init__(self, calibration=1):
        """
        Args:
            calibration (float): Calibration factor to scale the loss.
        """
        super(CalibratedL2Loss, self).__init__()
        self.calibration = calibration

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (Tensor): Predicted values (B, C, H, W)
            y_true (Tensor): Ground truth values (B, C, H, W)
        Returns:
            loss (Tensor): Scalar loss
        """
        assert y_pred.shape == y_true.shape, "Shape mismatch"
        loss = torch.mean((y_pred - y_true) ** 2)
        return self.calibration * loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric MAE loss for precipitation underestimation,
    with gamma CDF-based penalty.
    """

    def __init__(self, calibration=1):
        super().__init__()
        self.calibration = calibration

    def set_params(self, alpha: torch.Tensor, beta: torch.Tensor):
        # Compute gamma CDF at y_true
        # Need to broadcast alpha/beta to match shape (B, 1, H, W)
        self.gamma_dist = Gamma(
            alpha[None, None, :, :], beta[None, None, :, :])

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, orig_y_true: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            y_pred (Tensor): Predicted precipitation (B, 1, H, W)
            y_true (Tensor): Ground-truth precipitation (B, 1, H, W)
            orig_y_true (Tensor): Original unnormalized ground-truth precipitation.
        Returns:
            loss (Tensor): Scalar loss
        """
        assert y_pred.shape == y_true.shape, "Shape mismatch"
        B, C, H, W = y_true.shape
        device = y_true.device

        # Absolute error
        abs_error = torch.abs(y_true - y_pred)

        # Mask for rainy grid points (y_true > 0)
        rainy_mask = (y_true > 0)

        # Underestimation mask (y_pred < y_true)
        under_mask = (y_pred < y_true)
        if orig_y_true is not None:
            gamma_cdf = self.gamma_dist.cdf(
                orig_y_true.clamp(min=1e-6))  # Avoid 0 for CDF
        else:
            gamma_cdf = self.gamma_dist.cdf(
                y_true.clamp(min=1e-6))  # Avoid 0 for CDF

        # Penalty only where there is underestimation during rainy conditions
        gamma_weight = gamma_cdf ** 2 * rainy_mask * under_mask

        # Final weighted loss: L = |y - ŷ| + γ * |y - ŷ|
        weighted_error = abs_error + gamma_weight * abs_error

        return self.calibration * weighted_error.mean()


class NLLBernoulliGammaLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-3, calibration=10):
        super().__init__()
        self.reduction = reduction
        self.calibration = calibration
        self.eps = eps

    def forward(self, pred, y):
        """
        pi:    (B, ...) Bernoulli probability, in (0,1)
        alpha: (B, ...) Gamma shape > 0
        beta:  (B, ...) Gamma scale > 0
        y:     (B, ...) target values, either 0 or strictly > 0
        """
        pi = pred[0]
        alpha = pred[1]
        beta = pred[2]

        if torch.any(y < - self.eps):
            warnings.warn(
                "Some target values are negative, check if any normalisation is applied"
                f"min={y.min()}, max={y.max()}", UserWarning)
        #  Runtime checks with warnings
        if torch.any((pi <= 0) | (pi >= 1)):
            warnings.warn(
                "Some pi values are outside (0,1). They will be clamped."
                f"min={pi.detach().min().item()}, max={pi.detach().max().item()}", UserWarning)
            pi = torch.clamp(pi, 1e-6, 1 - self.eps)
        if torch.any(alpha <= 0):
            warnings.warn(
                "Some alpha values are <= 0. They will be clamped."
                f"min={alpha.detach().min().item()}, max={alpha.detach().max().item()}", UserWarning)
            alpha = torch.clamp(alpha, 0, None)
        if torch.any(beta <= 1e-6):
            warnings.warn(
                f"Some beta values are <= {1e-6}. They will be clamped."
                f"min={beta.detach().min().item()}, max={beta.detach().max().item()}", UserWarning)
            beta = torch.clamp(beta, self.eps, None)

        #  Case y == 0
        loss_zero = -torch.log1p(-pi)
        loss_pos = (
            - torch.log(pi)
            + torch.lgamma(alpha)
            + alpha * torch.log(beta)
            - (alpha - 1) * torch.log(torch.clamp(y, min=self.eps))
            + y / beta
        )

        occurence_mask = (y > self.eps).float()
        loss = (1 - occurence_mask) * loss_zero + occurence_mask * loss_pos

        if self.reduction == "mean":
            return self.calibration * loss.mean()
        elif self.reduction == "sum":
            return self.calibration * loss.sum()
        else:
            return loss


def get_criterion(name: str, **kwargs) -> Callable:
    # Map criterion names to corresponding loss functions

    name = name.lower()

    if name == 'mse' or name == 'l2':
        criterion = CalibratedL2Loss(**kwargs)
    elif name == 'mae' or name == 'l1':
        criterion = CalibratedL1Loss(**kwargs)
    elif name.startswith("spectral"):
        # Spectral Loss ||F(y) - F(y_pred)||_{Lp}
        criterion = SpectralLoss(**kwargs)
    elif name == "f_mae":
        criterion = FilteredLpLoss(p=1, **kwargs)
    elif name == "f_mse":
        criterion = FilteredLpLoss(p=2, **kwargs)
    elif name.startswith("wavelet"):
        # ||DWT(y) - DWT(y_pred)||_{Lp}
        criterion = WaveletLoss(**kwargs)
    elif name == 'ssim':
        # Structural Similarity Index (SSIM) Loss
        # SSIMLoss = 1 - SSIM(x, y)
        criterion = SSIMLoss(**kwargs)
    elif name.startswith("gdl"):
        # Gradient Difference Loss
        # GDL = ||∇(y) - ∇(y_pred)||_{Lp}
        criterion = GradientDifferenceLoss(**kwargs)
    elif name == 'emd' or name == 'wasserstein':
        # Wasserstein Distance Loss or Earth Mover's Distance (EMD)
        # EMD = intergral(|CDF(y) - CDF(y_pred)|)
        criterion = WassersteinDistance(**kwargs)
    elif name == "asym" or name == "asymmetric":
        # MAE loss with asymmetric penalty for precipitation underestimation
        # AsymmetricLoss = |y_pred - y_true| + gamma * max(0, y_true - y_pred)
        criterion = AsymmetricLoss(**kwargs)
    elif name == "nllbg":
        # Negative log likelihood of bernoulli-gamma dist
        criterion = NLLBernoulliGammaLoss(**kwargs)
    elif name == 'combination':
        # General combination of losses
        criterion = LossCombination(**kwargs)
    else:
        raise ValueError(
            f"Criterion '{name.lower()}' not recognized. Supported: mse, mae, spectral, f_mae, f_mse, wavelet, ssim, gdl, emd, asym, nllbg, combination."
        )

    return criterion
