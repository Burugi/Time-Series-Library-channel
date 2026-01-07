from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        train_start_time = time.time()
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (iter_count % 100 == 0):
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iter_count)
                    print(f"\titers: {iter_count}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        # Delete checkpoint after loading
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        # Remove checkpoint directory if empty
        if os.path.exists(path) and not os.listdir(path):
            os.rmdir(path)

        # Calculate training metrics
        train_time = time.time() - train_start_time
        train_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0

        return self.model, train_time, train_memory

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # Reset peak memory stats for inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        preds = []
        trues = []
        inputs = []

        inference_start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()

                # Inverse transform
                if test_data.scale and self.args.inverse:
                    if self.args.mode == 'CD':
                        # For CD mode, inverse transform all features
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                        # Inverse transform input
                        input_shape = batch_x.shape
                        batch_x = test_data.inverse_transform(batch_x.reshape(input_shape[0] * input_shape[1], -1)).reshape(input_shape)
                    else:
                        # For CI mode, inverse transform single feature
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(
                            outputs.reshape(shape[0] * shape[1], -1),
                            feature_name=self.args.target_feature
                        ).reshape(shape)
                        batch_y = test_data.inverse_transform(
                            batch_y.reshape(shape[0] * shape[1], -1),
                            feature_name=self.args.target_feature
                        ).reshape(shape)

                        # Inverse transform input
                        input_shape = batch_x.shape
                        batch_x = test_data.inverse_transform(
                            batch_x.reshape(input_shape[0] * input_shape[1], -1),
                            feature_name=self.args.target_feature
                        ).reshape(input_shape)

                preds.append(outputs)
                trues.append(batch_y)
                inputs.append(batch_x)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)

        # Calculate inference metrics
        inference_time = time.time() - inference_start_time
        inference_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0

        print('test shape:', preds.shape, trues.shape, inputs.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

        # result save
        folder_path = os.path.join(
            './results/',
            self.args.data,
            f'{self.args.seq_len}_{self.args.pred_len}',
            self.args.model,
            setting
        )
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse: {mse}, mae: {mae}, rmse: {rmse}')

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        np.save(os.path.join(folder_path, 'input.npy'), inputs)

        # Save scalability metrics
        np.save(os.path.join(folder_path, 'scalability.npy'), np.array([inference_time, inference_memory]))

        # Prepare per-channel metrics and channel names for CD mode
        per_channel_metrics_dict = None
        channel_names = None
        if self.args.mode == 'CD':
            n_channels = preds.shape[-1]
            per_channel_metrics = []

            # Get channel names from test_data
            channel_names = test_data.all_features if hasattr(test_data, 'all_features') else [f'ch{i}' for i in range(n_channels)]

            for ch in range(n_channels):
                mae_ch, mse_ch, rmse_ch, mape_ch, mspe_ch = metric(preds[:, :, ch:ch+1], trues[:, :, ch:ch+1])
                per_channel_metrics.append([mae_ch, mse_ch, rmse_ch, mape_ch, mspe_ch])

            per_channel_metrics = np.array(per_channel_metrics)
            np.save(os.path.join(folder_path, 'per_channel_metrics.npy'), per_channel_metrics)

            # Create dictionary with channel names
            per_channel_metrics_dict = {}
            for i, ch_name in enumerate(channel_names):
                per_channel_metrics_dict[ch_name] = {
                    'mae': float(per_channel_metrics[i, 0]),
                    'mse': float(per_channel_metrics[i, 1]),
                    'rmse': float(per_channel_metrics[i, 2]),
                    'mape': float(per_channel_metrics[i, 3]),
                    'mspe': float(per_channel_metrics[i, 4])
                }

        return mae, mse, rmse, mape, mspe, inference_time, inference_memory, per_channel_metrics_dict, channel_names
