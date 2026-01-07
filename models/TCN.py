import torch
import torch.nn as nn
from layers.tcn import TemporalBlock


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.num_channels = getattr(configs, 'num_channels', [32, 64, 128])
        self.kernel_size = getattr(configs, 'kernel_size', 3)
        self.dropout = getattr(configs, 'dropout', 0.2)

        self.tcn_layers = nn.ModuleList()
        num_levels = len(self.num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.channels if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]

            self.tcn_layers.append(
                TemporalBlock(
                    in_channels, out_channels, self.kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(self.kernel_size-1) * dilation_size,
                    dropout=self.dropout
                )
            )

        self.pred_network = nn.Sequential(
            nn.Linear(self.num_channels[-1] * self.seq_len, self.pred_len * self.channels),
            nn.Dropout(self.dropout)
        )

    def forecast(self, x_enc):
        batch_size = x_enc.size(0)

        x = x_enc.transpose(1, 2)

        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)

        x = x.reshape(batch_size, -1)
        x = self.pred_network(x)
        x = x.reshape(batch_size, self.pred_len, self.channels)

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
