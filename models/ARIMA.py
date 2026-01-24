import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


class Model(nn.Module):
    """
    ARIMA model using statsmodels library.
    This model does not require training - it fits ARIMA on each input sequence.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # ARIMA parameters (p, d, q)
        self.p = getattr(configs, 'ar_order', 5)  # AR order
        self.d = getattr(configs, 'diff_order', 1)  # Differencing order
        self.q = getattr(configs, 'ma_order', 1)  # MA order

        # Ensure valid orders
        self.p = min(self.p, self.seq_len // 2)
        self.q = min(self.q, self.seq_len // 2)
        self.d = min(self.d, 2)

        # Dummy parameter to make it a valid nn.Module (required for framework)
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def fit_arima_channel(self, x_np):
        """
        Fit ARIMA model on a single time series and forecast.
        x_np: numpy array of shape (seq_len,)
        Returns: numpy array of shape (pred_len,)
        """
        try:
            model = ARIMA(x_np, order=(self.p, self.d, self.q))
            fitted = model.fit(method_kwargs={'maxiter': 100})
            forecast = fitted.forecast(steps=self.pred_len)
            return forecast
        except Exception:
            # Fallback: use last value
            return np.full(self.pred_len, x_np[-1])

    def forecast(self, x_enc):
        """
        Forecast using statsmodels ARIMA.
        x_enc: (B, L, N) tensor
        """
        B, L, N = x_enc.shape
        device = x_enc.device

        # Convert to numpy
        x_np = x_enc.detach().cpu().numpy()

        # Forecast each sample and channel
        forecasts = np.zeros((B, self.pred_len, N))

        for b in range(B):
            for n in range(N):
                forecasts[b, :, n] = self.fit_arima_channel(x_np[b, :, n])

        # Convert back to tensor
        return torch.from_numpy(forecasts).float().to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
