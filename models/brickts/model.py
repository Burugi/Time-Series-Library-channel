"""
BrickTS - Brick-like Time Series Model

A modular forecasting model that combines three orthogonal axes:
- Level: How to handle channel information (direct/decomposition/spectral)
- Scope: How to capture temporal dependencies (global/local/hierarchical/sparse)
- Architecture: The backbone network (mlp/rnn/cnn/transformer)

Supports both Channel-Independent (CI) and Channel-Dependent (CD) strategies.
CI/CD mode is controlled externally via run_training.py/run_hyperopt.py --mode flag.
"""

import torch
import torch.nn as nn

from .axis_level import build_level, LEVEL_REGISTRY
from .axis_scope import build_scope, SCOPE_REGISTRY
from .axis_arch import build_arch, ARCH_REGISTRY


class BrickTS(nn.Module):
    """
    BrickTS: Modular Time Series Forecasting Model

    Pipeline:
        Input X: (B, L, C)
        -> Level(X) -> (B, L, D)
        -> Scope(X) -> (B, L, D)
        -> concat([level_feat, scope_feat]) -> (B, L, 2D)
        -> in_proj -> (B, L, D_backbone)
        -> Arch(z) -> (B, H, C_out)

    Note: CI/CD mode is handled by run_training.py/run_hyperopt.py externally.
          This model assumes CD mode by default.
    """

    def __init__(self, configs):
        super().__init__()

        # Basic parameters
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', configs.seq_len)
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, 'c_out', configs.enc_in)

        # Model dimension
        self.d_model = getattr(configs, 'd_model', 64)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # Axis selection
        self.level_type = getattr(configs, 'level_type', 'direct')
        self.scope_type = getattr(configs, 'scope_type', 'global')
        self.arch_type = getattr(configs, 'arch_type', 'mlp')

        # Build axis modules
        self._build_modules(configs)

    def _build_modules(self, configs):
        """Build Level, Scope, and Architecture modules."""

        # Level-specific kwargs
        level_kwargs = {}
        if self.level_type == 'decomposition':
            # Ensure moving_avg_kernel is smaller than seq_len
            default_kernel = min(25, max(3, self.seq_len // 2))
            moving_avg_kernel = getattr(configs, 'moving_avg_kernel', default_kernel)
            # Adjust if too large for seq_len
            if moving_avg_kernel >= self.seq_len:
                moving_avg_kernel = max(3, self.seq_len // 2)
            # Ensure odd number for symmetric padding
            if moving_avg_kernel % 2 == 0:
                moving_avg_kernel += 1
            level_kwargs['moving_avg_kernel'] = moving_avg_kernel
        elif self.level_type == 'spectral':
            level_kwargs['topk_freq'] = getattr(configs, 'topk_freq', None)
            level_kwargs['spectral_mode'] = getattr(configs, 'spectral_mode', 'mag')

        # Scope-specific kwargs
        scope_kwargs = {}
        if self.scope_type == 'local':
            # Ensure kernel fits in seq_len
            default_kernel = min(3, self.seq_len)
            kernel_size = getattr(configs, 'local_kernel_size', default_kernel)
            if kernel_size > self.seq_len:
                kernel_size = min(3, self.seq_len)
            scope_kwargs['kernel_size'] = kernel_size
            scope_kwargs['dilation'] = getattr(configs, 'local_dilation', 1)
        elif self.scope_type == 'hierarchical':
            # Filter kernel sizes that fit in seq_len
            default_kernels = [k for k in [3, 5, 7] if k <= self.seq_len]
            if not default_kernels:
                default_kernels = [min(3, self.seq_len)]
            kernel_sizes = getattr(configs, 'hierarchical_kernel_sizes', default_kernels)
            # Filter to valid sizes
            kernel_sizes = [k for k in kernel_sizes if k <= self.seq_len]
            if not kernel_sizes:
                kernel_sizes = [min(3, self.seq_len)]
            scope_kwargs['kernel_sizes'] = kernel_sizes
        elif self.scope_type == 'sparse':
            scope_kwargs['topk_ratio'] = getattr(configs, 'sparse_topk_ratio', 0.5)

        # Arch-specific kwargs
        arch_kwargs = {}
        if self.arch_type == 'mlp':
            arch_kwargs['n_layers'] = getattr(configs, 'mlp_n_layers', 2)
            arch_kwargs['hidden_dim'] = getattr(configs, 'mlp_hidden_dim', None)
        elif self.arch_type == 'rnn':
            arch_kwargs['hidden_size'] = getattr(configs, 'rnn_hidden_size', None)
            arch_kwargs['num_layers'] = getattr(configs, 'rnn_num_layers', 2)
        elif self.arch_type == 'cnn':
            arch_kwargs['n_layers'] = getattr(configs, 'cnn_n_layers', 3)
            # Ensure kernel fits
            cnn_kernel = getattr(configs, 'cnn_kernel_size', 3)
            if cnn_kernel > self.seq_len:
                cnn_kernel = min(3, self.seq_len)
            arch_kwargs['kernel_size'] = cnn_kernel
            arch_kwargs['dilation_base'] = getattr(configs, 'cnn_dilation_base', 2)
        elif self.arch_type == 'transformer':
            arch_kwargs['n_heads'] = getattr(configs, 'n_heads', 4)
            arch_kwargs['e_layers'] = getattr(configs, 'e_layers', 2)
            arch_kwargs['d_ff'] = getattr(configs, 'd_ff', None)

        # Build Level module
        self.level = build_level(
            self.level_type,
            seq_len=self.seq_len,
            enc_in=self.enc_in,
            d_model=self.d_model,
            dropout=self.dropout,
            **level_kwargs
        )

        # Build Scope module
        self.scope = build_scope(
            self.scope_type,
            seq_len=self.seq_len,
            enc_in=self.enc_in,
            d_model=self.d_model,
            dropout=self.dropout,
            **scope_kwargs
        )

        # Input projection: concat(level_feat, scope_feat) -> d_backbone
        d_backbone = self.d_model
        self.in_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, d_backbone),
            nn.LayerNorm(d_backbone),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # Build Architecture module
        self.arch = build_arch(
            self.arch_type,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=d_backbone,
            c_out=self.c_out,
            dropout=self.dropout,
            **arch_kwargs
        )

    def _forward_core(self, x):
        """
        Core forward pass through Level, Scope, and Architecture.
        Input: (B, L, C)
        Output: (B, H, C_out)
        """
        # Parallel feature extraction
        level_feat = self.level(x)   # (B, L, D)
        scope_feat = self.scope(x)   # (B, L, D)

        # Combine features
        z = torch.cat([level_feat, scope_feat], dim=-1)  # (B, L, 2D)
        z = self.in_proj(z)  # (B, L, D_backbone)

        # Architecture for prediction
        out = self.arch(z)  # (B, H, C_out)
        return out

    def forecast(self, x_enc):
        """Forecasting forward pass."""
        return self._forward_core(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Standard forward interface compatible with Time-Series-Library.

        Args:
            x_enc: Input sequence (B, L, C)
            x_mark_enc: Input timestamps (unused)
            x_dec: Decoder input (unused)
            x_mark_dec: Decoder timestamps (unused)
            mask: Optional mask (unused)

        Returns:
            Predictions (B, pred_len, c_out)
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None


class Model(BrickTS):
    """Alias for BrickTS to match Time-Series-Library naming convention."""
    pass
