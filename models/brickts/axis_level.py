"""
BrickTS - Interaction Level Axis
Defines how models handle relationships between different channels.

Level types based on PDF p7 Eq.(19)~(26):
- Direct: Raw channel processing without transformation
- Decomposition: Trend/Seasonal/Residual decomposition
- Spectral: Frequency domain transformation via FFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelDirect(nn.Module):
    """
    Direct Channel Interaction (PDF p7 Eq.19)
    Y = F_direct(X) = F_direct(x1, x2, ..., xC)

    Simply projects input to d_model dimension.
    Input: (B, L, C) -> Output: (B, L, D)
    """

    def __init__(self, seq_len, enc_in, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(enc_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C) -> (B, L, D)
        out = self.proj(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class LevelDecomposition(nn.Module):
    """
    Decomposition Channel Interaction (PDF p7 Eq.20-23)
    X_trend = MovingAvg(X; k)
    X_seasonal = X - X_trend
    X_residual = learned residual pattern
    Y = F_decomp(X_trend, X_seasonal, X_residual)

    Input: (B, L, C) -> Output: (B, L, D)
    """

    def __init__(self, seq_len, enc_in, d_model, dropout=0.1,
                 moving_avg_kernel=25):
        super().__init__()
        self.moving_avg_kernel = moving_avg_kernel

        # Moving average for trend extraction
        self.avg_pool = nn.AvgPool1d(
            kernel_size=moving_avg_kernel,
            stride=1,
            padding=0
        )

        # Projections for each component
        self.trend_proj = nn.Linear(enc_in, d_model // 3)
        self.seasonal_proj = nn.Linear(enc_in, d_model // 3)
        self.residual_proj = nn.Linear(enc_in, d_model - 2 * (d_model // 3))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _moving_avg(self, x):
        # x: (B, L, C)
        # Padding for same output length
        pad_len = (self.moving_avg_kernel - 1) // 2
        front = x[:, :1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # Apply avg pool: (B, L, C) -> (B, C, L) -> pool -> (B, C, L) -> (B, L, C)
        x_t = x_padded.permute(0, 2, 1)
        trend = self.avg_pool(x_t)
        trend = trend.permute(0, 2, 1)
        return trend

    def forward(self, x):
        # x: (B, L, C)
        trend = self._moving_avg(x)
        seasonal = x - trend
        residual = x - trend - seasonal  # This is essentially zeros, but we keep for structure

        # Project each component
        trend_feat = self.trend_proj(trend)
        seasonal_feat = self.seasonal_proj(seasonal)
        residual_feat = self.residual_proj(x)  # Use original for residual learning

        # Concatenate features
        out = torch.cat([trend_feat, seasonal_feat, residual_feat], dim=-1)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class LevelSpectral(nn.Module):
    """
    Spectral Channel Interaction (PDF p7 Eq.24-26)
    S_c = FFT(x_c)
    Y = F_spectral(S1, S2, ..., SC)

    Transforms input to frequency domain and extracts spectral features.
    Input: (B, L, C) -> Output: (B, L, D)
    """

    def __init__(self, seq_len, enc_in, d_model, dropout=0.1,
                 topk_freq=None, spectral_mode='mag'):
        super().__init__()
        self.seq_len = seq_len
        self.spectral_mode = spectral_mode  # 'mag' or 'realimag'

        # Number of frequency bins (rfft output)
        n_freq = seq_len // 2 + 1
        self.topk_freq = topk_freq if topk_freq else n_freq
        self.topk_freq = min(self.topk_freq, n_freq)

        # Feature dimension based on mode
        if spectral_mode == 'mag':
            freq_feat_dim = self.topk_freq * enc_in
        else:  # realimag
            freq_feat_dim = self.topk_freq * enc_in * 2

        # Project spectral features to d_model
        self.freq_proj = nn.Linear(freq_feat_dim, d_model)
        self.time_proj = nn.Linear(enc_in, d_model)

        # Combine time and frequency features
        self.combine = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape

        # FFT along time dimension: (B, L, C) -> (B, n_freq, C)
        x_freq = torch.fft.rfft(x, dim=1)

        # Select top-k frequencies
        x_freq = x_freq[:, :self.topk_freq, :]

        # Extract features based on mode
        if self.spectral_mode == 'mag':
            freq_feat = x_freq.abs()  # (B, topk, C)
        else:
            freq_feat = torch.cat([x_freq.real, x_freq.imag], dim=-1)  # (B, topk, 2C)

        # Flatten and project: (B, topk, C) -> (B, topk*C) -> (B, D)
        freq_feat = freq_feat.reshape(B, -1)
        freq_feat = self.freq_proj(freq_feat)  # (B, D)

        # Expand to sequence length: (B, D) -> (B, L, D)
        freq_feat = freq_feat.unsqueeze(1).expand(-1, L, -1)

        # Time domain features
        time_feat = self.time_proj(x)  # (B, L, D)

        # Combine
        out = torch.cat([time_feat, freq_feat], dim=-1)
        out = self.combine(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out


# Registry for level types
LEVEL_REGISTRY = {
    'direct': LevelDirect,
    'decomposition': LevelDecomposition,
    'spectral': LevelSpectral,
}


def build_level(level_type, seq_len, enc_in, d_model, dropout=0.1, **kwargs):
    """Build a level module from registry."""
    level_cls = LEVEL_REGISTRY[level_type]
    return level_cls(seq_len, enc_in, d_model, dropout, **kwargs)
