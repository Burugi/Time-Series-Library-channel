import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.api import VAR as StatsVAR
import warnings
warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    VAR (Vector Autoregression) model using statsmodels library.
    This model does not require training - it fits VAR on each input sequence.
    VAR captures linear interdependencies among multiple time series.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # VAR order (lag)
        self.p = getattr(configs, 'var_order', 5)

        # Ensure p doesn't exceed seq_len
        self.p = min(self.p, self.seq_len // 2)

        # Trend option for statsmodels VAR
        self.use_trend = getattr(configs, 'use_trend', True)
        self.trend = 'c' if self.use_trend else 'n'

        # Dummy parameter to make it a valid nn.Module (required for framework)
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def fit_var_sample(self, x_np):
        """
        Fit VAR model on a single sample (all channels together).
        x_np: numpy array of shape (seq_len, n_channels)
        Returns: numpy array of shape (pred_len, n_channels)
        """
        try:
            model = StatsVAR(x_np)
            fitted = model.fit(maxlags=self.p, ic=None, trend=self.trend)
            forecast = fitted.forecast(x_np[-fitted.k_ar:], steps=self.pred_len)
            return forecast
        except Exception:
            # Fallback: use last values
            return np.tile(x_np[-1:, :], (self.pred_len, 1))

    def fit_ar_channel(self, x_np):
        """
        Fallback: Fit simple AR model for single channel case.
        x_np: numpy array of shape (seq_len,)
        Returns: numpy array of shape (pred_len,)
        """
        try:
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(x_np, lags=self.p, trend=self.trend)
            fitted = model.fit()
            forecast = fitted.forecast(steps=self.pred_len)
            return forecast
        except Exception:
            return np.full(self.pred_len, x_np[-1])

    def forecast(self, x_enc):
        """
        Forecast using statsmodels VAR.
        x_enc: (B, L, N) tensor
        """
        B, L, N = x_enc.shape
        device = x_enc.device

        # Convert to numpy
        x_np = x_enc.detach().cpu().numpy()

        # Forecast each sample
        forecasts = np.zeros((B, self.pred_len, N))

        for b in range(B):
            if N >= 2:
                # VAR for multivariate case
                forecasts[b] = self.fit_var_sample(x_np[b])
            else:
                # AR for univariate case (VAR requires at least 2 variables)
                forecasts[b, :, 0] = self.fit_ar_channel(x_np[b, :, 0])

        # Convert back to tensor
        return torch.from_numpy(forecasts).float().to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
