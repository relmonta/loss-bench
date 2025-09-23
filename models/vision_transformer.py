import torch
import torch.nn as nn
from einops import rearrange
from models.patch_embedding import SpatioTemporalPatchEncoder
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)  # Self-attention
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for Spatio-Temporal Data.

    Args:
        emb_dim (int): Embedding dimension.
        patch_size (int or tuple): Patch size for tokenization.
        num_layers (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        stride (int, optional): Stride for patch embedding.
    """

    def __init__(self, emb_dim=128, patch_size=8, num_layers=6, num_heads=8,
                 dropout=0.1, stride=None, bernoulli_gamma=False):
        super().__init__()

        # Ensure patch_size is always a tuple
        self.patch_size = (patch_size, patch_size) if isinstance(
            patch_size, int) else tuple(patch_size)

        # Patch embedding layer
        self.patch_emb = SpatioTemporalPatchEncoder(
            self.patch_size, emb_dim, stride
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            emb_dim, num_layers, num_heads, dropout)

        # Store stride and padding from patch embedding
        self.stride = self.patch_emb.stride
        self.padding = self.patch_emb.padding
        self.bernoulli_gamma = bernoulli_gamma

        outchannels = 3 if bernoulli_gamma else 1

        self.reconstruction_2dhead = nn.ConvTranspose2d(
            emb_dim, outchannels, kernel_size=self.patch_size,
            stride=self.stride, padding=self.padding
        )

    def forward(self, x, static_vars, time):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (Tensor): Input tensor of shape (B, C=1, 1, H, W).
            static_vars (Tensor): Static variables associated with input.
            time (Tensor): Time information for embeddings.

        Returns:
            Tensor: Reconstructed output of shape (B, 1 or 3, H, W).
        """
        # Encode patches
        # print(x.shape)
        patches, h_patches, w_patches = self.patch_emb(x, static_vars, time)
        # print(patches.shape)
        encoded_patches = self.encoder(patches)

        # Reshape back to (B, emb_dim, H, W)
        patches = rearrange(
            encoded_patches, "b (h w) e -> b e h w", h=h_patches, w=w_patches)

        # Apply reconstruction heads per variable
        outputs = self.reconstruction_2dhead(patches)
        if self.bernoulli_gamma:
            pi = torch.sigmoid(outputs[:, 0:1, :, :])
            alpha = F.softplus(outputs[:, 1:2, :, :])
            beta = F.softplus(outputs[:, 2:3, :, :])
            return [pi, alpha, beta]

        return outputs
