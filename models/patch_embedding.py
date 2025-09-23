import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np


class SpatioTemporalPatchEncoder(nn.Module):
    def __init__(self, patch_size: tuple, emb_dim: int, stride: tuple = None):
        super().__init__()
        self.t_emb_dim = 7
        self.pos_emb_dim = 4
        # Subtract the dimensions used for time, position
        self.emb_dim = emb_dim - self.t_emb_dim - self.pos_emb_dim
        # Patch Embedding
        self.stride = stride if stride else tuple(p//2 for p in patch_size)
        self.padding = tuple((p - s) // 2 for p,
                             s in zip(patch_size, self.stride))

        # Patch Embedding (Uses Conv2D)
        self.patch_emb = PatchEmbedding2D(patch_size, self.emb_dim)

        self.time_encoder = TimeEmbedding()

    def extract_patch_means(self, grid):
        patches_means = F.avg_pool2d(
            grid, kernel_size=self.stride, stride=self.stride, padding=0)
        patches_means = patches_means.flatten(start_dim=1)
        return patches_means.unsqueeze(-1)  # Shape (B, num_patches, 1)

    def sine_cosine_encoding(self, values, min_val=-180, max_val=180):
        values = (values - min_val) / (max_val -
                                       min_val)  # Normalize to [0, 1]
        values = 2 * torch.pi * values  # Scale to [0, 2π]
        sin_enc = torch.sin(values)[:, None, :, :]
        cos_enc = torch.cos(values)[:, None, :, :]
        return torch.cat([sin_enc, cos_enc], dim=-1)

    def forward(self, x, static_inputs, time):
        lat_patches = self.extract_patch_means(static_inputs[:, :1, ...])
        lon_patches = self.extract_patch_means(static_inputs[:, 1:, ...])

        # Sine-Cosine encoding for latitude & longitude
        position_enc = self.sine_cosine_encoding(
            torch.cat([lat_patches, lon_patches], dim=-1))
        position_enc = rearrange(position_enc, 'b v p n -> b (v p) n')

        time_enc = self.time_encoder(time)  # Shape: (B, emb_dim)

        # Encode patches
        patches, h_patches, w_patches = self.patch_emb(x)
        # Repeat time vector to match number of patches  (B, num_patches, 7)
        time_enc = time_enc.expand(-1, patches.shape[-2], -1)
        embeddings = [patches, position_enc, time_enc]
        patches = torch.cat(embeddings, dim=-1)

        return patches, h_patches, w_patches


class PatchEmbedding2D(nn.Module):
    def __init__(self, patch_size, emb_dim, stride=None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride else tuple(s // 2 for s in patch_size)
        self.padding = tuple((p - s) // 2 for p,
                             s in zip(patch_size, self.stride))

        self.projection = nn.Conv2d(
            1, emb_dim,
            kernel_size=patch_size, stride=self.stride,
            padding=self.padding
        )

    def forward(self, x):
        # x shape: (batch_size, 1, H, W)
        patches = self.projection(x)
        h_patches, w_patches = patches.shape[2:]

        # Flatten into (batch_size, num_patches, emb_dim)
        patches = rearrange(patches, "b e h w -> b (h w) e")
        return patches, h_patches, w_patches


class TimeEmbedding(nn.Module):
    def __init__(self):
        super(TimeEmbedding, self).__init__()

        # Define attention weights as learnable parameters
        # 7 features: year, month_sin, month_cos, etc.
        self.attention_weights = nn.Parameter(torch.ones(7))

        # Attention MLP (optional, for more flexibility)
        self.attention_mlp = nn.Sequential(
            nn.Linear(7, 16),  # Input: 7 features
            nn.ReLU(),
            nn.Linear(16, 7),  # Output: Attention scores for each feature
            nn.Softmax(dim=-1)  # Normalise to get attention scores
        )

    def forward(self, t):
        year, month, day, hour = t[..., :1], t[...,
                                               1:2], t[..., 2:3], t[..., 3:4]
        reference_year = 1940
        elapsed_years = (year - reference_year)*1e-2

        # Cyclical encoding for periodic features
        month_sin = torch.sin(2 * np.pi * (month - 1) / 12)
        month_cos = torch.cos(2 * np.pi * (month - 1) / 12)

        day_sin = torch.sin(2 * np.pi * (day - 1) /
                            31)
        day_cos = torch.cos(2 * np.pi * (day - 1) /
                            31)

        hour_sin = torch.sin(2 * np.pi * hour /
                             24)
        hour_cos = torch.cos(2 * np.pi * hour /
                             24)

        # Combine features
        time_features = torch.cat([
            elapsed_years,
            month_sin, month_cos,
            day_sin, day_cos,
            hour_sin, hour_cos], dim=-1)
        # Apply attention
        attention_scores = self.attention_mlp(time_features)
        attended_features = time_features * attention_scores

        return attended_features
