import argparse
import os
from optimization.hyperopt import HyperparameterOptimizer
from utils.tools import dotdict


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Time Series Forecasting')

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
                        ],
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

    # Optimization config
    parser.add_argument('--n_trials', type=int, default=10, help='number of trials for hyperparameter optimization')
    parser.add_argument('--timeout', type=int, default=None, help='timeout in seconds for optimization')
    parser.add_argument('--sampler', type=str, default='tpe', choices=['tpe', 'random'],
                        help='sampler type: tpe (Bayesian) or random')

    # GPU config
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # Other config
    parser.add_argument('--scale', type=bool, default=True, help='scale data')
    parser.add_argument('--timeenc', type=int, default=0, help='time encoding (0: manual, 1: learned)')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')

    # Training config (for objective function)
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs for each trial')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # Model define (defaults will be overridden by hyperopt, from Time-Series-Library-origin/run.py)
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

    # Create save directory
    save_dir = os.path.join(
        'hyperopt_archive',
        args.data,
        f'{args.seq_len}_{args.pred_len}',
        f'{args.model}_{args.mode}'
    )
    os.makedirs(save_dir, exist_ok=True)

    # Config paths
    common_config_path = 'configs/hyperopt_config.yaml'
    model_config_path = f'configs/models/{args.model}.yaml'

    print('='*80)
    print(f'Hyperparameter Optimization')
    print(f'Model: {args.model}')
    print(f'Mode: {args.mode}')
    print(f'Dataset: {args.data}')
    print(f'Seq Len: {args.seq_len}, Pred Len: {args.pred_len}')
    print(f'Sampler: {args.sampler}')
    print(f'N Trials: {args.n_trials}')
    print('='*80)

    # For CI mode, optimize for each target feature
    if args.mode == 'CI':
        # Determine target features
        if args.target_features is None:
            if args.features is None:
                raise ValueError("Either features or target_features must be specified for CI mode")
            target_features = args.features
        else:
            target_features = args.target_features

        print(f'\nCI mode: Optimizing for {len(target_features)} features separately')

        for feature in target_features:
            print(f'\n{"="*80}')
            print(f'Optimizing for feature: {feature}')
            print(f'{"="*80}')

            # Update args for this feature
            args.target_feature = feature
            feature_save_dir = os.path.join(save_dir, feature)

            # Run optimization
            optimizer = HyperparameterOptimizer(
                args=args,
                model_name=args.model,
                common_config_path=common_config_path,
                model_config_path=model_config_path,
                save_dir=feature_save_dir,
                sampler_type=args.sampler
            )

            best_params, best_value = optimizer.optimize(n_trials=args.n_trials, timeout=args.timeout)
            print(f'\nFeature {feature} optimization completed!')
            print(f'Best validation loss: {best_value:.6f}')

    else:  # CD mode
        print(f'\nCD mode: Optimizing for all features together')

        # Run optimization
        optimizer = HyperparameterOptimizer(
            args=args,
            model_name=args.model,
            common_config_path=common_config_path,
            model_config_path=model_config_path,
            save_dir=save_dir,
            sampler_type=args.sampler
        )

        best_params, best_value = optimizer.optimize(n_trials=args.n_trials, timeout=args.timeout)
        print(f'\nOptimization completed!')
        print(f'Best validation loss: {best_value:.6f}')

    print('\n' + '='*80)
    print('All optimizations completed!')
    print('='*80)


if __name__ == '__main__':
    main()
