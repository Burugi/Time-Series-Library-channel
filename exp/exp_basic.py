import os
import torch
from models import (
    Autoformer, SegRNN, TimeMixer, SCINet,
    DLinear, Transformer, TimesNet, TSMixer, TiDE,
    Informer, Pyraformer, MICN, LightTS, TimeFilter,
    MultiPatchFormer, WPMixer,
    Linear, TCN, RNN, DSSRNN, SSRNN, DUET, BrickTS,
    TimeXer, iTransformer, ARIMA, NBeats, VAR
)
from utils.tools import dotdict


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'SegRNN': SegRNN,
            'TimeMixer': TimeMixer,
            'SCINet': SCINet,
            'DLinear': DLinear,
            'Transformer': Transformer,
            'TimesNet': TimesNet,
            'TSMixer': TSMixer,
            'TiDE': TiDE,
            'Informer': Informer,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'LightTS': LightTS,
            'TimeFilter': TimeFilter,
            'MultiPatchFormer': MultiPatchFormer,
            'WPMixer': WPMixer,
            'Linear': Linear,
            'TCN': TCN,
            'RNN': RNN,
            'DSSRNN': DSSRNN,
            'SSRNN': SSRNN,
            'DUET': DUET,
            'BrickTS': BrickTS,
            'TimeXer': TimeXer,
            'iTransformer': iTransformer,
            'ARIMA': ARIMA,
            'NBeats': NBeats,
            'VAR': VAR
        }
        self.device = self._acquire_device()
        self._prepare_model_configs()
        self.model = self._build_model().to(self.device)

    def _prepare_model_configs(self):
        """Prepare model configuration based on mode and data."""
        # Set task name
        if not hasattr(self.args, 'task_name'):
            self.args.task_name = 'long_term_forecast'

        # Set number of input/output channels based on mode
        if self.args.mode == 'CI':
            # CI mode: single feature
            self.args.enc_in = 1
            self.args.dec_in = 1
            self.args.c_out = 1
        else:
            # CD mode: multiple features
            # These will be set dynamically when data is loaded
            if not hasattr(self.args, 'enc_in'):
                self.args.enc_in = len(self.args.features) if self.args.features else 7

            self.args.dec_in = self.args.enc_in
            self.args.c_out = self.args.enc_in


    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        self.args.device = device
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
