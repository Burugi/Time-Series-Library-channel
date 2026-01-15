import argparse
import os
import json
import torch
import numpy as np
from exp.exp_forecasting import Exp_Forecast
from utils.tools import dotdict


def load_hyperparameters(hyperopt_dir, mode, target_feature=None):
    """
    Load best hyperparameters from hyperopt results.

    Args:
        hyperopt_dir: Directory containing hyperopt results
        mode: 'CD' or 'CI'
        target_feature: For CI mode, which feature's hyperparameters to load

    Returns:
        best_params: Dictionary of best hyperparameters
    """
    if mode == 'CI' and target_feature is not None:
        result_path = os.path.join(hyperopt_dir, target_feature, 'hyperopt_results.json')
    else:
        result_path = os.path.join(hyperopt_dir, 'hyperopt_results.json')

    with open(result_path, 'r') as f:
        results = json.load(f)

    return results['best_params']


def main():
    parser = argparse.ArgumentParser(description='Train Time Series Forecasting Models')

    # Basic config
    parser.add_argument('--model', type=str, required=True,
                        choices=[
                            'Autoformer',
                            'SegRNN',
                            'TimeMixer',
                            'SCINet',
                            'WPMixer',
                            'DLinear',
                            'Transformer',
                            'LightTS',
                            'Pyraformer',
                            'Informer',
                            'MICN',
                            'TiDE',
                            'TimesNet',
                            'TSMixer',
                            'MultiPatchFormer',
                            'Linear',
                            'DSSRNN',
                            'SSRNN',
                            'TCN',
                            'RNN',
                            'DUET',
                            'ModernTCN',
                            'BrickTS'
                        ],
                        help='model name')

    # Data config
    parser.add_argument('--data', type=str, required=True, help='dataset name')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.csv', help='data file')
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
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (will be overridden by hyperopt)')

    # Hyperparameter loading
    parser.add_argument('--use_hyperopt', action='store_true', default=True,
                        help='use hyperparameters from hyperopt')
    parser.add_argument('--hyperopt_dir', type=str, default=None,
                        help='directory containing hyperopt results (auto-generated if not specified)')

    # Training config
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--patience', type=int, default=25, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU config
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # Other config
    parser.add_argument('--scale', type=bool, default=True, help='scale data')
    parser.add_argument('--timeenc', type=int, default=0, help='time encoding (0: manual, 1: learned)')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')

    # Experiment config
    parser.add_argument('--n_repeats', type=int, default=10, help='number of repeated experiments')

    # Model define (defaults will be overridden by hyperopt if use_hyperopt=True, from Time-Series-Library-origin/run.py)
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=0, help='channel independence')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decomposition')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None, help='down sampling method')
    parser.add_argument('--seg_len', type=int, default=96, help='segment length for SegRNN')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--hid_size', type=float, default=1.0, help='hidden size for SCINet')
    parser.add_argument('--num_levels', type=int, default=3, help='num levels for SCINet')
    parser.add_argument('--num_decoder_layer', type=int, default=1, help='num decoder layers for SCINet')
    parser.add_argument('--concat_len', type=int, default=0, help='concat length for SCINet')
    parser.add_argument('--groups', type=int, default=1, help='groups for SCINet')
    parser.add_argument('--kernel', type=int, default=5, help='kernel size for SCINet')
    parser.add_argument('--positionalE', type=bool, default=False, help='positional encoding for SCINet')
    parser.add_argument('--modified', type=bool, default=True, help='modified for SCINet')
    parser.add_argument('--RIN', type=bool, default=False, help='RIN for SCINet')

    # Additional model parameters from run.py
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio for imputation')
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--use_dtw', type=bool, default=False, help='use dtw metric')
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--node_dim', type=int, default=10, help='node dimension for GCN')
    parser.add_argument('--gcn_depth', type=int, default=2, help='GCN depth')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='GCN dropout')
    parser.add_argument('--propalpha', type=float, default=0.3, help='prop alpha for GCN')
    parser.add_argument('--conv_channel', type=int, default=32, help='conv channel')
    parser.add_argument('--skip_channel', type=int, default=32, help='skip channel')
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: individual')
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, default=1, help='Positional Embedding')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # DUET specific parameters
    parser.add_argument('--num_experts', type=int, default=4, help='number of experts for DUET')
    parser.add_argument('--noisy_gating', type=bool, default=True, help='noisy gating for DUET')
    parser.add_argument('--k', type=int, default=1, help='top k experts for DUET')
    parser.add_argument('--fc_dropout', type=float, default=0.1, help='fc dropout for DUET')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size for DUET')
    parser.add_argument('--output_attention', type=int, default=0, help='output attention for DUET')
    parser.add_argument('--CI', type=bool, default=False, help='Channel Independence for DUET')

    # ModernTCN specific parameters
    parser.add_argument('--patch_size', type=int, default=16, help='patch size for ModernTCN')
    parser.add_argument('--patch_stride', type=int, default=8, help='patch stride for ModernTCN')
    parser.add_argument('--stem_ratio', type=int, default=1, help='stem ratio for ModernTCN')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample ratio for ModernTCN')
    parser.add_argument('--ffn_ratio', type=int, default=4, help='ffn ratio for ModernTCN')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[1, 1, 1, 1], help='num blocks for ModernTCN')
    parser.add_argument('--large_size', type=int, nargs='+', default=[51, 49, 47, 13], help='large kernel size for ModernTCN')
    parser.add_argument('--small_size', type=int, nargs='+', default=[5, 5, 5, 5], help='small kernel size for ModernTCN')
    parser.add_argument('--dims', type=int, nargs='+', default=[64, 128, 256, 512], help='dimensions for ModernTCN')
    parser.add_argument('--dw_dims', type=int, nargs='+', default=[64, 128, 256, 512], help='dw dimensions for ModernTCN')
    parser.add_argument('--small_kernel_merged', type=bool, default=False, help='small kernel merged for ModernTCN')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout for ModernTCN')
    parser.add_argument('--use_multi_scale', type=bool, default=True, help='use multi scale for ModernTCN')
    parser.add_argument('--revin', type=bool, default=True, help='revin for ModernTCN')
    parser.add_argument('--affine', type=bool, default=True, help='affine for ModernTCN')
    parser.add_argument('--subtract_last', type=bool, default=False, help='subtract last for ModernTCN')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size for decomposition')
    parser.add_argument('--decomposition', type=bool, default=False, help='decomposition for ModernTCN')

    args = parser.parse_args()

    # Set label_len dynamically if not properly set
    if args.label_len >= args.pred_len:
        args.label_len = args.seq_len // 2
        print(f'Warning: label_len adjusted to {args.label_len} (seq_len // 2)')

    # Auto-generate hyperopt_dir if not specified
    if args.hyperopt_dir is None:
        args.hyperopt_dir = os.path.join(
            'hyperopt_archive',
            args.data,
            f'{args.seq_len}_{args.pred_len}',
            f'{args.model}_{args.mode}'
        )

    print('='*80)
    print(f'Training Configuration')
    print(f'Model: {args.model}')
    print(f'Mode: {args.mode}')
    print(f'Dataset: {args.data}')
    print(f'Seq Len: {args.seq_len}, Pred Len: {args.pred_len}')
    print(f'Use Hyperopt: {args.use_hyperopt}')
    if args.use_hyperopt:
        print(f'Hyperopt Dir: {args.hyperopt_dir}')
    print(f'N Repeats: {args.n_repeats}')
    print('='*80)

    # Results storage
    all_results = {}

    # For CI mode, train models for each target feature
    if args.mode == 'CI':
        # Determine target features
        if args.target_features is None:
            if args.features is None:
                raise ValueError("Either features or target_features must be specified for CI mode")
            target_features = args.features
        else:
            target_features = args.target_features

        print(f'\nCI mode: Training {len(target_features)} models separately')

        for feature in target_features:
            print(f'\n{"="*80}')
            print(f'Training model for feature: {feature}')
            print(f'{"="*80}')

            # Update args for this feature
            args.target_feature = feature

            # Set CI parameter for DUET model
            if args.model == 'DUET':
                args.CI = True

            # Load hyperparameters
            if args.use_hyperopt:
                best_params = load_hyperparameters(args.hyperopt_dir, args.mode, feature)
                print(f'\nLoaded hyperparameters: {best_params}')
                for key, value in best_params.items():
                    setattr(args, key, value)

            # Repeat experiments
            feature_results = []
            scalability_results = []
            for repeat in range(args.n_repeats):
                print(f'\n--- Repeat {repeat + 1}/{args.n_repeats} ---')

                # Set random seed for reproducibility
                torch.manual_seed(repeat)
                np.random.seed(repeat)

                # Create experiment
                exp = Exp_Forecast(args)
                setting = f'{args.model}_{args.mode}_{feature}_repeat{repeat}'

                # Train
                print('>>>>>>>start training>>>>>>>>>>>>>>')
                _, train_time, train_memory = exp.train(setting)

                # Test
                print('>>>>>>>testing>>>>>>>>>>>>>>')
                mae, mse, rmse, mape, mspe, inference_time, inference_memory, _, _ = exp.test(setting)

                feature_results.append({
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mape': float(mape),
                    'mspe': float(mspe)
                })

                num_params = sum(p.numel() for p in exp.model.parameters())
                scalability_results.append({
                    'train_time': float(train_time),
                    'train_memory_gb': float(train_memory),
                    'inference_time': float(inference_time),
                    'inference_memory_gb': float(inference_memory),
                    'num_params': num_params
                })

                torch.cuda.empty_cache()

            # Compute average results
            avg_results = {
                'mae': np.mean([r['mae'] for r in feature_results]),
                'mse': np.mean([r['mse'] for r in feature_results]),
                'rmse': np.mean([r['rmse'] for r in feature_results]),
                'mape': np.mean([r['mape'] for r in feature_results]),
                'mspe': np.mean([r['mspe'] for r in feature_results]),
                'std': {
                    'mae': np.std([r['mae'] for r in feature_results]),
                    'mse': np.std([r['mse'] for r in feature_results]),
                    'rmse': np.std([r['rmse'] for r in feature_results]),
                },
                'scalability': {
                    'train_time_mean': np.mean([r['train_time'] for r in scalability_results]),
                    'train_time_std': np.std([r['train_time'] for r in scalability_results]),
                    'train_memory_gb_mean': np.mean([r['train_memory_gb'] for r in scalability_results]),
                    'train_memory_gb_std': np.std([r['train_memory_gb'] for r in scalability_results]),
                    'inference_time_mean': np.mean([r['inference_time'] for r in scalability_results]),
                    'inference_time_std': np.std([r['inference_time'] for r in scalability_results]),
                    'inference_memory_gb_mean': np.mean([r['inference_memory_gb'] for r in scalability_results]),
                    'inference_memory_gb_std': np.std([r['inference_memory_gb'] for r in scalability_results]),
                    'num_params': scalability_results[0]['num_params']
                }
            }

            all_results[feature] = avg_results

            print(f'\nFeature {feature} Average Results:')
            print(f'MAE: {avg_results["mae"]:.4f} ± {avg_results["std"]["mae"]:.4f}')
            print(f'MSE: {avg_results["mse"]:.4f} ± {avg_results["std"]["mse"]:.4f}')
            print(f'RMSE: {avg_results["rmse"]:.4f} ± {avg_results["std"]["rmse"]:.4f}')

    else:  # CD mode
        print(f'\nCD mode: Training one model for all features')

        # Set CI parameter for DUET model
        if args.model == 'DUET':
            args.CI = False

        # Load hyperparameters
        if args.use_hyperopt:
            best_params = load_hyperparameters(args.hyperopt_dir, args.mode)
            print(f'\nLoaded hyperparameters: {best_params}')
            for key, value in best_params.items():
                setattr(args, key, value)

        # Repeat experiments
        cd_results = []
        scalability_results = []
        per_channel_results = {}
        channel_names = None

        for repeat in range(args.n_repeats):
            print(f'\n--- Repeat {repeat + 1}/{args.n_repeats} ---')

            # Set random seed for reproducibility
            torch.manual_seed(repeat)
            np.random.seed(repeat)

            # Create experiment
            exp = Exp_Forecast(args)
            setting = f'{args.model}_{args.mode}_repeat{repeat}'

            # Train
            print('>>>>>>>start training>>>>>>>>>>>>>>')
            _, train_time, train_memory = exp.train(setting)

            # Test
            print('>>>>>>>testing>>>>>>>>>>>>>>')
            mae, mse, rmse, mape, mspe, inference_time, inference_memory, per_channel_metrics_dict, ch_names = exp.test(setting)

            cd_results.append({
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe)
            })

            num_params = sum(p.numel() for p in exp.model.parameters())
            scalability_results.append({
                'train_time': float(train_time),
                'train_memory_gb': float(train_memory),
                'inference_time': float(inference_time),
                'inference_memory_gb': float(inference_memory),
                'num_params': num_params
            })

            # Collect per-channel results
            if per_channel_metrics_dict is not None:
                if channel_names is None:
                    channel_names = ch_names
                for ch_name, metrics in per_channel_metrics_dict.items():
                    if ch_name not in per_channel_results:
                        per_channel_results[ch_name] = []
                    per_channel_results[ch_name].append(metrics)

            torch.cuda.empty_cache()

        # Compute average overall results
        overall_results = {
            'mae': np.mean([r['mae'] for r in cd_results]),
            'mse': np.mean([r['mse'] for r in cd_results]),
            'rmse': np.mean([r['rmse'] for r in cd_results]),
            'mape': np.mean([r['mape'] for r in cd_results]),
            'mspe': np.mean([r['mspe'] for r in cd_results]),
            'std': {
                'mae': np.std([r['mae'] for r in cd_results]),
                'mse': np.std([r['mse'] for r in cd_results]),
                'rmse': np.std([r['rmse'] for r in cd_results]),
            }
        }

        # Compute average per-channel results
        per_channel_avg = {}
        for ch_name, results_list in per_channel_results.items():
            per_channel_avg[ch_name] = {
                'mae': np.mean([r['mae'] for r in results_list]),
                'mse': np.mean([r['mse'] for r in results_list]),
                'rmse': np.mean([r['rmse'] for r in results_list]),
                'mape': np.mean([r['mape'] for r in results_list]),
                'mspe': np.mean([r['mspe'] for r in results_list]),
                'std': {
                    'mae': np.std([r['mae'] for r in results_list]),
                    'mse': np.std([r['mse'] for r in results_list]),
                    'rmse': np.std([r['rmse'] for r in results_list]),
                }
            }

        # Compute scalability metrics
        scalability_avg = {
            'train_time_mean': np.mean([r['train_time'] for r in scalability_results]),
            'train_time_std': np.std([r['train_time'] for r in scalability_results]),
            'train_memory_gb_mean': np.mean([r['train_memory_gb'] for r in scalability_results]),
            'train_memory_gb_std': np.std([r['train_memory_gb'] for r in scalability_results]),
            'inference_time_mean': np.mean([r['inference_time'] for r in scalability_results]),
            'inference_time_std': np.std([r['inference_time'] for r in scalability_results]),
            'inference_memory_gb_mean': np.mean([r['inference_memory_gb'] for r in scalability_results]),
            'inference_memory_gb_std': np.std([r['inference_memory_gb'] for r in scalability_results]),
            'num_params': scalability_results[0]['num_params']
        }

        all_results['overall'] = overall_results
        all_results['per_channel'] = per_channel_avg
        all_results['scalability'] = scalability_avg

        print(f'\nCD Mode Average Results:')
        print(f'MAE: {overall_results["mae"]:.4f} ± {overall_results["std"]["mae"]:.4f}')
        print(f'MSE: {overall_results["mse"]:.4f} ± {overall_results["std"]["mse"]:.4f}')
        print(f'RMSE: {overall_results["rmse"]:.4f} ± {overall_results["std"]["rmse"]:.4f}')

    # Save all results
    results_dir = os.path.join('results', args.data, f'{args.seq_len}_{args.pred_len}')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'{args.model}_{args.mode}_results.json')

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print('\n' + '='*80)
    print(f'All experiments completed!')
    print(f'Results saved to: {results_file}')
    print('='*80)


if __name__ == '__main__':
    main()
