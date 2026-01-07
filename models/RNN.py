import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.hidden_size = getattr(configs, 'hidden_size', 128)
        self.num_layers = getattr(configs, 'num_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.rnn_type = getattr(configs, 'rnn_type', 'LSTM')

        rnn_class = getattr(nn, self.rnn_type)
        self.rnn = rnn_class(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )

        self.pred_network = nn.Sequential(
            nn.Linear(self.hidden_size * self.seq_len, self.pred_len * self.channels),
            nn.Dropout(self.dropout)
        )

    def forecast(self, x_enc):
        batch_size = x_enc.size(0)

        rnn_out, _ = self.rnn(x_enc)
        rnn_out = rnn_out.reshape(batch_size, -1)
        x = self.pred_network(rnn_out)
        x = x.reshape(batch_size, self.pred_len, self.channels)

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
