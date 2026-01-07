import torch
import torch.nn as nn
from einops import rearrange
from layers.duet.linear_extractor_cluster import Linear_extractor_cluster
from utils.duet.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

    def forward(self, input):
        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature, L_importance = self.cluster(input)

        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
            changed_input = rearrange(input, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        return output, L_importance


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        configs.CI = getattr(configs, 'CI', False)
        configs.factor = getattr(configs, 'factor', 1)
        configs.output_attention = getattr(configs, 'output_attention', 0)
        configs.d_model = getattr(configs, 'd_model', 512)
        configs.d_ff = getattr(configs, 'd_ff', 2048)
        configs.n_heads = getattr(configs, 'n_heads', 8)
        configs.e_layers = getattr(configs, 'e_layers', 2)
        configs.dropout = getattr(configs, 'dropout', 0.1)
        configs.fc_dropout = getattr(configs, 'fc_dropout', 0.1)
        configs.activation = getattr(configs, 'activation', 'gelu')
        configs.moving_avg = getattr(configs, 'moving_avg', 25)
        configs.num_experts = getattr(configs, 'num_experts', 4)
        configs.noisy_gating = getattr(configs, 'noisy_gating', True)
        configs.k = getattr(configs, 'k', 1)
        configs.hidden_size = getattr(configs, 'hidden_size', 256)

        self.model = DUETModel(configs)

    def forecast(self, x_enc):
        output, L_importance = self.model(x_enc)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return None
