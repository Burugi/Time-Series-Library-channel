import torch
import torch.nn as nn
from layers.ssrnn_layers import CustomRNNCell


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.hidden_size = getattr(configs, 'hidden_size', 128)
        self.num_layers = getattr(configs, 'num_layers', 1)

        self.rnn_cell = CustomRNNCell(input_size=self.seq_len, hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.pred_len)

    def forecast(self, x_enc):
        batch_size = x_enc.size(0)

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)

        h = torch.zeros(batch_size, self.hidden_size).to(x_enc.device)
        out = []
        for t in range(self.channels):
            h = self.rnn_cell(x_enc[:, t, :], h)
            out.append(h.unsqueeze(1))
        out = torch.cat(out, dim=1)

        out = self.fc(out)
        out = out.permute(0, 2, 1)

        out = out * stdev + means

        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
