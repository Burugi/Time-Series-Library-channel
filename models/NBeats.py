import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NBeatsBlock(nn.Module):
    """
    N-BEATS block - the basic building block.
    """

    def __init__(self, input_size, theta_size, hidden_size, num_layers, block_type='generic'):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.block_type = block_type

        # Fully connected stack
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc_stack = nn.Sequential(*layers)

        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

    def forward(self, x, backcast_linspace, forecast_linspace):
        # x: (B, input_size)
        h = self.fc_stack(x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        if self.block_type == 'trend':
            # Polynomial trend
            backcast = self._polynomial_basis(theta_b, backcast_linspace)
            forecast = self._polynomial_basis(theta_f, forecast_linspace)
        elif self.block_type == 'seasonality':
            # Fourier seasonality
            backcast = self._fourier_basis(theta_b, backcast_linspace)
            forecast = self._fourier_basis(theta_f, forecast_linspace)
        else:
            # Generic - use learned linear basis
            backcast = self._generic_basis(theta_b, backcast_linspace)
            forecast = self._generic_basis(theta_f, forecast_linspace)

        return backcast, forecast

    def _polynomial_basis(self, theta, t):
        # theta: (B, degree+1), t: (length,)
        B = theta.size(0)
        degree = theta.size(1)
        t = t.unsqueeze(0).expand(B, -1)  # (B, length)

        # Polynomial: sum(theta_i * t^i)
        powers = torch.stack([t ** i for i in range(degree)], dim=2)  # (B, length, degree)
        return torch.einsum('bld,bd->bl', powers, theta)

    def _fourier_basis(self, theta, t):
        # theta: (B, num_harmonics * 2), t: (length,)
        B = theta.size(0)
        num_harmonics = theta.size(1) // 2
        t = t.unsqueeze(0).expand(B, -1)  # (B, length)

        # Fourier basis
        cos_terms = []
        sin_terms = []
        for k in range(1, num_harmonics + 1):
            cos_terms.append(torch.cos(2 * np.pi * k * t))
            sin_terms.append(torch.sin(2 * np.pi * k * t))

        cos_stack = torch.stack(cos_terms, dim=2)  # (B, length, num_harmonics)
        sin_stack = torch.stack(sin_terms, dim=2)

        theta_cos = theta[:, :num_harmonics]
        theta_sin = theta[:, num_harmonics:]

        return torch.einsum('blk,bk->bl', cos_stack, theta_cos) + \
               torch.einsum('blk,bk->bl', sin_stack, theta_sin)

    def _generic_basis(self, theta, t):
        # Generic: linear interpolation
        B = theta.size(0)
        length = t.size(0)
        theta_size = theta.size(1)

        # Simple linear projection
        basis = torch.linspace(0, 1, theta_size, device=theta.device)
        basis = basis.unsqueeze(0).unsqueeze(0).expand(B, length, -1)
        theta_expanded = theta.unsqueeze(1).expand(-1, length, -1)

        return (basis * theta_expanded).sum(dim=2)


class NBeatsStack(nn.Module):
    """
    N-BEATS stack - collection of blocks.
    """

    def __init__(self, input_size, pred_len, hidden_size, num_layers, num_blocks, block_type='generic'):
        super(NBeatsStack, self).__init__()
        self.input_size = input_size
        self.pred_len = pred_len
        self.block_type = block_type

        # Determine theta size based on block type
        if block_type == 'trend':
            theta_size = 4  # Polynomial degree
        elif block_type == 'seasonality':
            theta_size = 8  # Number of harmonics * 2
        else:
            theta_size = max(input_size, pred_len)

        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, hidden_size, num_layers, block_type)
            for _ in range(num_blocks)
        ])

    def forward(self, x, backcast_linspace, forecast_linspace):
        # x: (B, input_size)
        residual = x
        forecast_sum = torch.zeros(x.size(0), self.pred_len, device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residual, backcast_linspace, forecast_linspace)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast

        return residual, forecast_sum


class Model(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
    Paper link: https://arxiv.org/abs/1905.10437
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # N-BEATS hyperparameters
        self.hidden_size = getattr(configs, 'd_model', 256)
        self.num_layers = getattr(configs, 'e_layers', 4)
        self.num_blocks = getattr(configs, 'num_blocks', 3)
        self.stack_types = getattr(configs, 'stack_types', ['trend', 'seasonality', 'generic'])
        self.individual = getattr(configs, 'individual', True)

        # Create linspace for basis functions
        self.register_buffer('backcast_linspace',
                             torch.linspace(0, 1, self.seq_len))
        self.register_buffer('forecast_linspace',
                             torch.linspace(0, 1, self.pred_len))

        if self.individual:
            # Separate model for each channel
            self.stacks = nn.ModuleList()
            for _ in range(self.channels):
                channel_stacks = nn.ModuleList([
                    NBeatsStack(self.seq_len, self.pred_len, self.hidden_size,
                               self.num_layers, self.num_blocks, stack_type)
                    for stack_type in self.stack_types
                ])
                self.stacks.append(channel_stacks)
        else:
            # Shared model across channels
            self.stacks = nn.ModuleList([
                NBeatsStack(self.seq_len, self.pred_len, self.hidden_size,
                           self.num_layers, self.num_blocks, stack_type)
                for stack_type in self.stack_types
            ])

    def forecast_channel(self, x, channel_idx=None):
        """
        Forecast for a single channel.
        x: (B, seq_len)
        """
        B = x.size(0)
        residual = x
        forecast_sum = torch.zeros(B, self.pred_len, device=x.device)

        stacks = self.stacks[channel_idx] if self.individual else self.stacks

        for stack in stacks:
            residual, stack_forecast = stack(residual, self.backcast_linspace, self.forecast_linspace)
            forecast_sum = forecast_sum + stack_forecast

        return forecast_sum

    def forecast(self, x_enc):
        """
        Forecast using N-BEATS model.
        x_enc: (B, L, N)
        """
        B, L, N = x_enc.shape

        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Forecast each channel
        outputs = []
        for i in range(N):
            x_channel = x_enc[:, :, i]  # (B, L)
            pred = self.forecast_channel(x_channel, channel_idx=i if self.individual else None)
            outputs.append(pred)

        # Stack: (B, pred_len, N)
        dec_out = torch.stack(outputs, dim=2)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
