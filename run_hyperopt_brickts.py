import argparse
import os
import json
import itertools
import shutil
import pandas as pd
import torch
import numpy as np
from exp.exp_forecasting import Exp_Forecast


# BrickTS 3-axis values
ARCH_TYPES = ['mlp', 'rnn', 'cnn', 'transformer']
SCOPE_TYPES = ['global', 'local', 'hierarchical', 'sparse']
LEVEL_TYPES = ['direct', 'decomposition', 'spectral']


def detect_enc_in(args):
    """Detect the number of input channels from the actual data file."""
    df = pd.read_csv(os.path.join(args.root_path, args.data_path))
    cols = list(df.columns)
    if 'date' in cols[0].lower():
        cols = cols[1:]
    if args.features is not None:
        cols = [f for f in args.features if f in cols]
    return len(cols)


def delete_npy_folder(args, setting):
    """Delete .npy result folder created by exp.test()."""
    folder_path = os.path.join(
        './results/', args.data,
        f'{args.seq_len}_{args.pred_len}',
        'BrickTS', setting
    )
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def run_experiment_cd(args, combo_name):
    """Run single CD experiment: train + test. Returns results dict."""
    exp = Exp_Forecast(args)
    setting = f'BrickTS_{combo_name}_{args.mode}'

    print('  >>> Training')
    _, train_time, train_memory = exp.train(setting)

    print('  >>> Testing')
    mae, mse, rmse, mape, mspe, inference_time, inference_memory, per_ch_dict, _ = exp.test(setting)

    delete_npy_folder(args, setting)

    num_params = sum(p.numel() for p in exp.model.parameters())

    results = {
        'overall': {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
        },
        'scalability': {
            'train_time': float(train_time),
            'train_memory_gb': float(train_memory),
            'inference_time': float(inference_time),
            'inference_memory_gb': float(inference_memory),
            'num_params': num_params,
        }
    }

    if per_ch_dict is not None:
        results['per_channel'] = per_ch_dict

    torch.cuda.empty_cache()
    return results


def run_experiment_ci(args, combo_name, target_features):
    """Run CI experiment: train + test per feature. Returns results dict."""
    all_results = {}

    for feature in target_features:
        print(f'  Feature: {feature}')
        args.target_feature = feature

        exp = Exp_Forecast(args)
        setting = f'BrickTS_{combo_name}_{args.mode}_{feature}'

        print('    >>> Training')
        _, train_time, train_memory = exp.train(setting)

        print('    >>> Testing')
        mae, mse, rmse, mape, mspe, inference_time, inference_memory, _, _ = exp.test(setting)

        delete_npy_folder(args, setting)

        num_params = sum(p.numel() for p in exp.model.parameters())

        all_results[feature] = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'scalability': {
                'train_time': float(train_time),
                'train_memory_gb': float(train_memory),
                'inference_time': float(inference_time),
                'inference_memory_gb': float(inference_memory),
                'num_params': num_params,
            }
        }

        torch.cuda.empty_cache()

    return all_results


def save_results(results, args, arch_type, scope_type, level_type):
    """Save results.json to the specified folder structure."""
    combo_name = f'{arch_type}_{scope_type}_{level_type}'
    results_dir = os.path.join(
        'results_BrickTS', args.data,
        f'{args.seq_len}_{args.pred_len}',
        'BrickTS', combo_name
    )
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'{combo_name}_{args.mode}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'  Results saved to: {results_file}')


def main():
    parser = argparse.ArgumentParser(description='BrickTS Model Combination Experiments')

    # Data config
    parser.add_argument('--data', type=str, required=True, help='dataset name')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='milano_6165.csv', help='data file')
    parser.add_argument('--features', type=str, nargs='*', default=None,
                        help='feature names to use (None for all features)')
    parser.add_argument('--target_features', type=str, nargs='*', default=None,
                        help='target features for CI mode')

    # Mode config
    parser.add_argument('--mode', type=str, required=True, choices=['CD', 'CI'],
                        help='CD: Channel Dependency, CI: Channel Independency')

    # Forecasting config
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Training config (fixed)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--d_model', type=int, default=64, help='model dimension (fixed)')
    parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--patience', type=int, default=2, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    # GPU config
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # Axis filter (optional, to run specific combinations)
    parser.add_argument('--arch_types', type=str, nargs='*', default=None,
                        help='filter arch_type (default: all)')
    parser.add_argument('--scope_types', type=str, nargs='*', default=None,
                        help='filter scope_type (default: all)')
    parser.add_argument('--level_types', type=str, nargs='*', default=None,
                        help='filter level_type (default: all)')

    # Other config
    parser.add_argument('--scale', type=bool, default=True, help='scale data')
    parser.add_argument('--timeenc', type=int, default=0, help='time encoding')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--seed', type=int, default=2, help='seed')

    args = parser.parse_args()

    args.model = 'BrickTS'
    args.task_name = 'long_term_forecast'

    # Adjust label_len
    if args.label_len >= args.pred_len:
        args.label_len = args.seq_len // 2
        print(f'Warning: label_len adjusted to {args.label_len} (seq_len // 2)')

    # Detect enc_in from actual data to avoid shape mismatch
    enc_in = detect_enc_in(args)
    args.enc_in = enc_in
    args.dec_in = enc_in
    args.c_out = enc_in
    print(f'Detected enc_in={enc_in} from data')

    # Determine axis combinations
    arch_types = args.arch_types if args.arch_types else ARCH_TYPES
    scope_types = args.scope_types if args.scope_types else SCOPE_TYPES
    level_types = args.level_types if args.level_types else LEVEL_TYPES

    combos = list(itertools.product(arch_types, scope_types, level_types))

    # CI mode target features
    target_features = None
    if args.mode == 'CI':
        if args.target_features is not None:
            target_features = args.target_features
        elif args.features is not None:
            target_features = args.features
        else:
            raise ValueError("Either features or target_features must be specified for CI mode")

    print('=' * 80)
    print(f'BrickTS Model Combination Experiments')
    print(f'Mode: {args.mode}')
    print(f'Dataset: {args.data}')
    print(f'Seq Len: {args.seq_len}, Pred Len: {args.pred_len}')
    print(f'enc_in: {enc_in}, d_model: {args.d_model}')
    print(f'lr: {args.learning_rate}, batch_size: {args.batch_size}, dropout: {args.dropout}')
    print(f'Epochs: {args.train_epochs}, Patience: {args.patience}')
    print(f'Total combinations: {len(combos)}')
    print('=' * 80)

    for i, (arch_type, scope_type, level_type) in enumerate(combos):
        combo_name = f'{arch_type}_{scope_type}_{level_type}'
        print(f'\n{"=" * 80}')
        print(f'[{i + 1}/{len(combos)}] {combo_name}')
        print(f'{"=" * 80}')

        args.arch_type = arch_type
        args.scope_type = scope_type
        args.level_type = level_type

        if args.mode == 'CD':
            results = run_experiment_cd(args, combo_name)
        else:
            results = run_experiment_ci(args, combo_name, target_features)

        save_results(results, args, arch_type, scope_type, level_type)

        if 'overall' in results:
            print(f'  MAE: {results["overall"]["mae"]:.4f}')
            print(f'  MSE: {results["overall"]["mse"]:.4f}')

    print('\n' + '=' * 80)
    print('All combinations completed!')
    print('=' * 80)


if __name__ == '__main__':
    main()
