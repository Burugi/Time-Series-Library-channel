import argparse
import os
import time
import json
import numpy as np
import torch
from data_provider.data_factory import data_provider
from utils.metrics import metric
import warnings
warnings.filterwarnings('ignore')


class StatsModelRunner:
    """
    Runner for statistical models (ARIMA, VAR) that don't require training.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')  # Stats models run on CPU

        # Import model
        if args.model == 'ARIMA':
            from models.ARIMA import Model
        elif args.model == 'VAR':
            from models.VAR import Model
        else:
            raise ValueError(f"Unknown model: {args.model}")

        self.model = Model(args)
        self.model.eval()

    def run(self, setting):
        """Run inference on test data."""
        test_data, test_loader = data_provider(self.args, flag='test')

        preds = []
        trues = []
        inputs = []

        inference_start_time = time.time()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                # decoder input (not used by stats models, but kept for interface compatibility)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # model inference
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                outputs = outputs.numpy()
                batch_y = batch_y.numpy()
                batch_x = batch_x.numpy()

                # Inverse transform
                if test_data.scale and self.args.inverse:
                    if self.args.mode == 'CD':
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        input_shape = batch_x.shape
                        batch_x = test_data.inverse_transform(batch_x.reshape(input_shape[0] * input_shape[1], -1)).reshape(input_shape)
                    else:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(
                            outputs.reshape(shape[0] * shape[1], -1),
                            feature_name=self.args.target_feature
                        ).reshape(shape)
                        batch_y = test_data.inverse_transform(
                            batch_y.reshape(shape[0] * shape[1], -1),
                            feature_name=self.args.target_feature
                        ).reshape(shape)
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

        inference_time = time.time() - inference_start_time
        inference_memory = 0  # CPU-based, no GPU memory

        print('test shape:', preds.shape, trues.shape, inputs.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

        # Result save
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
        np.save(os.path.join(folder_path, 'scalability.npy'), np.array([inference_time, inference_memory]))

        # Per-channel metrics for CD mode
        per_channel_metrics_dict = None
        channel_names = None
        if self.args.mode == 'CD':
            n_channels = preds.shape[-1]
            per_channel_metrics = []
            channel_names = test_data.all_features if hasattr(test_data, 'all_features') else [f'ch{i}' for i in range(n_channels)]

            for ch in range(n_channels):
                mae_ch, mse_ch, rmse_ch, mape_ch, mspe_ch = metric(preds[:, :, ch:ch+1], trues[:, :, ch:ch+1])
                per_channel_metrics.append([mae_ch, mse_ch, rmse_ch, mape_ch, mspe_ch])

            per_channel_metrics = np.array(per_channel_metrics)
            np.save(os.path.join(folder_path, 'per_channel_metrics.npy'), per_channel_metrics)

            per_channel_metrics_dict = {}
            for i, ch_name in enumerate(channel_names):
                per_channel_metrics_dict[ch_name] = {
                    'mae': float(per_channel_metrics[i, 0]),
                    'mse': float(per_channel_metrics[i, 1]),
                    'rmse': float(per_channel_metrics[i, 2]),
                    'mape': float(per_channel_metrics[i, 3]),
                    'mspe': float(per_channel_metrics[i, 4])
                }

        return mae, mse, rmse, mape, mspe, inference_time, per_channel_metrics_dict, channel_names


def main():
    parser = argparse.ArgumentParser(description='Statistical Models for Time Series Forecasting')

    # Basic config
    parser.add_argument('--model', type=str, required=True,
                        choices=['ARIMA', 'VAR'],
                        help='model name')

    # Data config
    parser.add_argument('--data', type=str, required=True, help='dataset name')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='milano_6165.csv', help='data file')
    parser.add_argument('--features', type=str, nargs='*', default=None,
                        help='feature names to use (None for all features)')
    parser.add_argument('--target_features', type=str, nargs='*', default=None,
                        help='target features for CI mode (None for all features in features list)')

    # Mode config
    parser.add_argument('--mode', type=str, required=True, choices=['CD', 'CI'],
                        help='CD: Channel Dependency, CI: Channel Independency')

    # Forecasting config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Data loader config
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    # Other config
    parser.add_argument('--scale', type=bool, default=True, help='scale data')
    parser.add_argument('--timeenc', type=int, default=0, help='time encoding')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')

    # Number of repeats
    parser.add_argument('--n_repeats', type=int, default=1, help='number of experiment repeats')

    # ARIMA parameters
    parser.add_argument('--ar_order', type=int, default=5, help='AR order for ARIMA')
    parser.add_argument('--diff_order', type=int, default=1, help='Differencing order for ARIMA')
    parser.add_argument('--ma_order', type=int, default=1, help='MA order for ARIMA')

    # VAR parameters
    parser.add_argument('--var_order', type=int, default=5, help='VAR order (lag)')
    parser.add_argument('--use_trend', type=bool, default=True, help='Use trend in VAR')

    args = parser.parse_args()

    # Set task_name for model compatibility
    args.task_name = 'long_term_forecast'
    args.enc_in = None  # Will be set by data loader

    # Set label_len
    if args.label_len >= args.pred_len:
        args.label_len = args.seq_len // 2
        print(f'Warning: label_len adjusted to {args.label_len} (seq_len // 2)')

    print('=' * 80)
    print(f'Statistical Model Inference')
    print(f'Model: {args.model}')
    print(f'Mode: {args.mode}')
    print(f'Dataset: {args.data}')
    print(f'Seq Len: {args.seq_len}, Pred Len: {args.pred_len}')
    print('=' * 80)

    # Get enc_in from data
    test_data, _ = data_provider(args, flag='test')
    args.enc_in = test_data.data_x.shape[-1]
    print(f'Number of channels: {args.enc_in}')

    # Run experiments
    if args.mode == 'CI':
        # CI mode: run for each target feature
        if args.target_features is None:
            if args.features is None:
                raise ValueError("Either features or target_features must be specified for CI mode")
            target_features = args.features
        else:
            target_features = args.target_features

        print(f'\nCI mode: Running for {len(target_features)} features separately')

        all_results = {}
        for feature in target_features:
            print(f'\n{"=" * 80}')
            print(f'Running for feature: {feature}')
            print(f'{"=" * 80}')

            args.target_feature = feature
            feature_results = []

            for repeat in range(args.n_repeats):
                setting = f'{args.model}_CI_{feature}_repeat{repeat}'
                print(f'\nRepeat {repeat + 1}/{args.n_repeats}')

                runner = StatsModelRunner(args)
                mae, mse, rmse, mape, mspe, inference_time, _, _ = runner.run(setting)
                feature_results.append({
                    'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse),
                    'mape': float(mape), 'mspe': float(mspe),
                    'inference_time': float(inference_time)
                })

            all_results[feature] = feature_results

        # Save summary results
        results_dir = os.path.join('./results/', args.data, f'{args.seq_len}_{args.pred_len}')
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'{args.model}_CI_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f'\nResults saved to: {results_file}')

    else:  # CD mode
        print(f'\nCD mode: Running for all features together')

        all_results = []
        for repeat in range(args.n_repeats):
            setting = f'{args.model}_CD_repeat{repeat}'
            print(f'\nRepeat {repeat + 1}/{args.n_repeats}')

            runner = StatsModelRunner(args)
            mae, mse, rmse, mape, mspe, inference_time, per_channel_metrics, channel_names = runner.run(setting)

            result = {
                'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse),
                'mape': float(mape), 'mspe': float(mspe),
                'inference_time': float(inference_time)
            }
            if per_channel_metrics:
                result['per_channel_metrics'] = per_channel_metrics
            all_results.append(result)

        # Save summary results
        results_dir = os.path.join('./results/', args.data, f'{args.seq_len}_{args.pred_len}')
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'{args.model}_CD_results.json')

        summary = {
            'results': all_results,
            'avg_mae': float(np.mean([r['mae'] for r in all_results])),
            'avg_mse': float(np.mean([r['mse'] for r in all_results])),
            'avg_rmse': float(np.mean([r['rmse'] for r in all_results])),
            'std_mae': float(np.std([r['mae'] for r in all_results])),
            'std_mse': float(np.std([r['mse'] for r in all_results])),
            'std_rmse': float(np.std([r['rmse'] for r in all_results])),
            'channel_names': channel_names
        }

        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f'\nResults saved to: {results_file}')

    print('\n' + '=' * 80)
    print('All experiments completed!')
    print('=' * 80)


if __name__ == '__main__':
    main()
