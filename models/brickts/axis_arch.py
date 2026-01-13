"""
BrickTS - Interaction Architecture Axis
Defines the backbone architecture for processing combined features.

Architecture types based on PDF p5 Eq.(7)~(12):
- MLP: Multi-Layer Perceptron
- RNN: Recurrent Neural Network (GRU)
- CNN: Temporal Convolutional Network
- Transformer: Self-attention based
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArchMLP(nn.Module):
    """
    MLP-based Architecture (PDF p5 Eq.7)
    Y = W2 * σ(W1 * X + b1) + b2

    Simple feedforward network for prediction.
    Input: (B, L, D_in) -> Output: (B, H, C_out)
    """

    def __init__(self, seq_len, pred_len, d_model, c_out, dropout=0.1,
                 n_layers=2, hidden_dim=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_out = c_out

        hidden_dim = hidden_dim or d_model * 2

        layers = []
        in_dim = seq_len * d_model
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, pred_len * c_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, D)
        B = x.size(0)
        x = x.reshape(B, -1)  # (B, L*D)
        out = self.mlp(x)  # (B, H*C)
        out = out.reshape(B, self.pred_len, self.c_out)  # (B, H, C)
        return out


class ArchRNN(nn.Module):
    """
    RNN-based Architecture (PDF p5 Eq.8-9)
    h_t = W_h * X_t + W_hh * h_{t-1} + b_h
    Y_t = W_out * h_t + b_out

    GRU-based recurrent processing.
    Input: (B, L, D_in) -> Output: (B, H, C_out)
    """

    def __init__(self, seq_len, pred_len, d_model, c_out, dropout=0.1,
                 hidden_size=None, num_layers=2):
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out

        hidden_size = hidden_size or d_model

        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.pred_head = nn.Linear(hidden_size, pred_len * c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        B = x.size(0)
        rnn_out, _ = self.rnn(x)  # (B, L, H)
        last_hidden = rnn_out[:, -1, :]  # (B, H)
        last_hidden = self.dropout(last_hidden)
        out = self.pred_head(last_hidden)  # (B, H*C)
        out = out.reshape(B, self.pred_len, self.c_out)  # (B, H, C)
        return out


class ArchCNN(nn.Module):
    """
    CNN-based Architecture (PDF p5 Eq.10)
    Y_{t,c} = Σ W_{i,j} · X_{t-i,c-j} + b

    TCN-style dilated convolutions.
    Input: (B, L, D_in) -> Output: (B, H, C_out)
    """

    def __init__(self, seq_len, pred_len, d_model, c_out, dropout=0.1,
                 n_layers=3, kernel_size=3, dilation_base=2):
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out

        # Dilated conv stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            self.convs.append(
                nn.Conv1d(d_model, d_model, kernel_size,
                          padding=padding, dilation=dilation)
            )
            self.norms.append(nn.LayerNorm(d_model))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Prediction head
        self.pred_head = nn.Linear(seq_len * d_model, pred_len * c_out)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape

        # Conv processing: (B, L, D) -> (B, D, L)
        out = x.permute(0, 2, 1)

        for conv, norm in zip(self.convs, self.norms):
            residual = out
            out = conv(out)
            out = out[:, :, :L]  # Truncate to original length
            out = out.permute(0, 2, 1)  # (B, L, D)
            out = norm(out)
            out = self.act(out)
            out = self.dropout(out)
            out = out.permute(0, 2, 1)  # (B, D, L)
            out = out + residual

        out = out.permute(0, 2, 1)  # (B, L, D)
        out = out.reshape(B, -1)  # (B, L*D)
        out = self.pred_head(out)  # (B, H*C)
        out = out.reshape(B, self.pred_len, self.c_out)  # (B, H, C)
        return out


class ArchTransformer(nn.Module):
    """
    Transformer-based Architecture (PDF p5 Eq.11-12)
    Q = X * W_Q, K = X * W_K, V = X * W_V
    Attention(Q, K, V) = softmax(QK^T / √d_k) * V

    Encoder-only self-attention.
    Input: (B, L, D_in) -> Output: (B, H, C_out)
    """

    def __init__(self, seq_len, pred_len, d_model, c_out, dropout=0.1,
                 n_heads=4, e_layers=2, d_ff=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_out = c_out

        d_ff = d_ff or d_model * 4

        # Positional encoding
        self.pos_encoding = self._generate_pe(seq_len, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Prediction head
        self.pred_head = nn.Linear(seq_len * d_model, pred_len * c_out)
        self.dropout = nn.Dropout(dropout)

    def _generate_pe(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2]) if d_model % 2 == 1 else torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape

        # Add positional encoding
        pe = self.pos_encoding[:, :L, :D].to(x.device)
        x = x + pe

        # Transformer encoding
        out = self.encoder(x)  # (B, L, D)
        out = self.dropout(out)

        # Prediction
        out = out.reshape(B, -1)  # (B, L*D)
        out = self.pred_head(out)  # (B, H*C)
        out = out.reshape(B, self.pred_len, self.c_out)  # (B, H, C)
        return out


# Registry for architecture types
ARCH_REGISTRY = {
    'mlp': ArchMLP,
    'rnn': ArchRNN,
    'cnn': ArchCNN,
    'transformer': ArchTransformer,
}


def build_arch(arch_type, seq_len, pred_len, d_model, c_out, dropout=0.1, **kwargs):
    """Build an architecture module from registry."""
    arch_cls = ARCH_REGISTRY[arch_type]
    return arch_cls(seq_len, pred_len, d_model, c_out, dropout, **kwargs)
