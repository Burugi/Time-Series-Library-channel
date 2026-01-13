"""
BrickTS - Interaction Scope Axis
Defines how models capture dependencies across different temporal positions.

Scope types based on PDF p6 Eq.(13)~(18):
- Global: Processes all temporal steps uniformly
- Local: Limited temporal window with fixed kernel
- Hierarchical: Multi-scale temporal processing
- Sparse: Selective focus on important time steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScopeGlobal(nn.Module):
    """
    Global Temporal Interaction (PDF p6 Eq.13)
    Y = F_global(X_{1:T}) = F_global(X_1, X_2, ..., X_T)

    Processes all temporal positions using token-mixing MLP.
    Input: (B, L, C) -> Output: (B, L, D)
    """

    def __init__(self, seq_len, enc_in, d_model, dropout=0.1):
        super().__init__()
        # Token mixing MLP (mix across time dimension)
        self.time_proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )

        # Channel projection
        self.channel_proj = nn.Linear(enc_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape

        # Token mixing: (B, L, C) -> (B, C, L) -> mix -> (B, C, L) -> (B, L, C)
        x_t = x.permute(0, 2, 1)
        x_t = self.time_proj(x_t)
        x_mixed = x_t.permute(0, 2, 1)

        # Channel projection
        out = self.channel_proj(x_mixed)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class ScopeLocal(nn.Module):
    """
    Local Temporal Interaction (PDF p6 Eq.14)
    Y_t = F_local(X_{t-w+1:t})

    Uses fixed-size convolutional kernel for local patterns.
    Input: (B, L, C) -> Output: (B, L, D)
    """

    def __init__(self, seq_len, enc_in, d_model, dropout=0.1,
                 kernel_size=3, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Causal padding
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=enc_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape

        # Conv: (B, L, C) -> (B, C, L) -> conv -> (B, D, L') -> (B, L, D)
        x_t = x.permute(0, 2, 1)
        out = self.conv(x_t)

        # Truncate to original length (causal)
        out = out[:, :, :L]
        out = out.permute(0, 2, 1)

        out = self.act(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class ScopeHierarchical(nn.Module):
    """
    Hierarchical Temporal Interaction (PDF p6 Eq.15)
    Y = F_hierarchical(⊕_{w∈W} α_w · F_w(X_{t-w+1:t}))

    Multi-scale processing with multiple kernel sizes.
    Input: (B, L, C) -> Output: (B, L, D)
    """

    def __init__(self, seq_len, enc_in, d_model, dropout=0.1,
                 kernel_sizes=None):
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 5, 7]
        self.kernel_sizes = kernel_sizes
        n_scales = len(kernel_sizes)

        # Per-scale convolutions
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k - 1) // 2
            self.convs.append(
                nn.Conv1d(enc_in, d_model // n_scales, k, padding=padding)
            )

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Final projection to ensure exact d_model
        concat_dim = (d_model // n_scales) * n_scales
        self.out_proj = nn.Linear(concat_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        x_t = x.permute(0, 2, 1)  # (B, C, L)

        # Multi-scale features
        scale_feats = []
        weights = F.softmax(self.scale_weights, dim=0)
        for i, conv in enumerate(self.convs):
            feat = conv(x_t)  # (B, D//n, L)
            feat = feat * weights[i]
            scale_feats.append(feat)

        # Concatenate scales
        out = torch.cat(scale_feats, dim=1)  # (B, D', L)
        out = out.permute(0, 2, 1)  # (B, L, D')

        out = self.act(out)
        out = self.out_proj(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class ScopeSparse(nn.Module):
    """
    Sparse Temporal Interaction (PDF p6 Eq.16-18)
    Score(t) = S_model(X, t; θ)
    I = arg top-K Score(t)
    Y = F_sparse({X_t | t ∈ I})

    Selectively processes top-k important time steps.
    Input: (B, L, C) -> Output: (B, L, D)
    """

    def __init__(self, seq_len, enc_in, d_model, dropout=0.1,
                 topk_ratio=0.5):
        super().__init__()
        self.topk = max(1, int(seq_len * topk_ratio))
        self.seq_len = seq_len

        # Score network: simple linear score
        self.score_net = nn.Linear(enc_in, 1)

        # Process selected positions
        self.sparse_proj = nn.Linear(enc_in, d_model)

        # Full sequence projection (for residual)
        self.full_proj = nn.Linear(enc_in, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape

        # Compute importance scores: (B, L, C) -> (B, L, 1) -> (B, L)
        scores = self.score_net(x).squeeze(-1)

        # Get top-k indices
        _, topk_indices = torch.topk(scores, self.topk, dim=1)  # (B, topk)

        # Gather top-k features
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        sparse_x = torch.gather(x, dim=1, index=topk_indices_exp)  # (B, topk, C)

        # Process sparse features
        sparse_feat = self.sparse_proj(sparse_x)  # (B, topk, D)
        sparse_feat = self.act(sparse_feat)

        # Aggregate sparse features (mean pooling)
        sparse_agg = sparse_feat.mean(dim=1, keepdim=True)  # (B, 1, D)
        sparse_agg = sparse_agg.expand(-1, L, -1)  # (B, L, D)

        # Full sequence features
        full_feat = self.full_proj(x)  # (B, L, D)

        # Combine
        out = full_feat + sparse_agg
        out = self.norm(out)
        out = self.dropout(out)
        return out


# Registry for scope types
SCOPE_REGISTRY = {
    'global': ScopeGlobal,
    'local': ScopeLocal,
    'hierarchical': ScopeHierarchical,
    'sparse': ScopeSparse,
}


def build_scope(scope_type, seq_len, enc_in, d_model, dropout=0.1, **kwargs):
    """Build a scope module from registry."""
    scope_cls = SCOPE_REGISTRY[scope_type]
    return scope_cls(seq_len, enc_in, d_model, dropout, **kwargs)
