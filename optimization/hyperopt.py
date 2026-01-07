import os
import yaml
import json
import optuna
from optuna.samplers import TPESampler, RandomSampler
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from utils.tools import EarlyStopping, dotdict
from data_provider.data_factory import data_provider


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna.

    Args:
        args: Arguments containing experiment configuration
        model_name: Name of the model to optimize
        common_config_path: Path to common hyperparameter config
        model_config_path: Path to model-specific hyperparameter config
        save_dir: Directory to save optimization results
        sampler_type: 'tpe' (Bayesian) or 'random'
    """
    def __init__(self, args, model_name, common_config_path, model_config_path,
                 save_dir, sampler_type='tpe'):
        self.args = args
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Load configs
        with open(common_config_path, 'r') as f:
            self.common_config = yaml.safe_load(f)

        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)

        # Create sampler
        if sampler_type == 'tpe':
            self.sampler = TPESampler(seed=42)
        elif sampler_type == 'random':
            self.sampler = RandomSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        # Device
        self.device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')

    def _suggest_parameters(self, trial):
        """Suggest hyperparameters for a trial."""
        params = {}

        # Common parameters
        for param_name, param_config in self.common_config['common_params'].items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )

        # Model-specific parameters
        if 'model_params' in self.model_config:
            for param_name, param_config in self.model_config['model_params'].items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['min'],
                        param_config['max'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

        # Validate and adjust length-dependent parameters
        if 'moving_avg' in params:
            # moving_avg should be less than seq_len
            if params['moving_avg'] >= self.args.seq_len:
                params['moving_avg'] = max(3, self.args.seq_len // 2)

        if 'seg_len' in params:
            # seg_len should divide seq_len and pred_len evenly
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            # Find valid seg_len values (common divisors)
            valid_seg_lens = []
            for sl in [6, 12, 24, 48, 96]:
                if sl <= min(seq_len, pred_len) and seq_len % sl == 0 and pred_len % sl == 0:
                    valid_seg_lens.append(sl)

            if valid_seg_lens and params['seg_len'] not in valid_seg_lens:
                # Pick the closest valid seg_len
                params['seg_len'] = min(valid_seg_lens, key=lambda x: abs(x - params['seg_len']))

        return params

    def objective(self, trial):
        """Optuna objective function."""
        # Suggest hyperparameters
        params = self._suggest_parameters(trial)

        # Update args with suggested parameters
        for key, value in params.items():
            setattr(self.args, key, value)

        # Import model
        from exp.exp_forecasting import Exp_Forecast

        # Train model
        exp = Exp_Forecast(self.args)
        setting = f'{self.model_name}_{self.args.mode}_trial_{trial.number}'

        # Get data
        train_data, train_loader = exp._get_data(flag='train')
        vali_data, vali_loader = exp._get_data(flag='val')

        # Training
        model_optim = exp._select_optimizer()
        criterion = exp._select_criterion()

        patience = self.common_config.get('patience', 10)
        max_epochs = self.common_config.get('max_epochs', 50)

        early_stopping = EarlyStopping(patience=patience, verbose=False)

        for epoch in range(max_epochs):
            exp.model.train()
            train_loss = []

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float().to(exp.device)
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)

                # encoder - decoder
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -exp.args.pred_len:, :]
                batch_y = batch_y[:, -exp.args.pred_len:, :]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            # Validation
            vali_loss = exp.vali(vali_data, vali_loader, criterion)

            # Early stopping check
            temp_path = os.path.join(self.save_dir, f'temp_trial_{trial.number}')
            os.makedirs(temp_path, exist_ok=True)
            early_stopping(vali_loss, exp.model, temp_path)

            if early_stopping.early_stop:
                break

            # Report intermediate value
            trial.report(vali_loss, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Delete temporary checkpoint
        checkpoint_path = os.path.join(temp_path, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        # Remove temp directory if empty
        if os.path.exists(temp_path) and not os.listdir(temp_path):
            os.rmdir(temp_path)

        return vali_loss

    def optimize(self, n_trials=None, timeout=None):
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Time limit in seconds

        Returns:
            best_params: Dictionary of best hyperparameters
            best_value: Best validation loss
        """
        if n_trials is None:
            n_trials = self.common_config.get('n_trials', 50)
        if timeout is None:
            timeout = self.common_config.get('timeout', None)

        study = optuna.create_study(
            direction='minimize',
            sampler=self.sampler,
            study_name=f'{self.model_name}_{self.args.mode}_optimization'
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Save results
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials)
        }

        result_path = os.path.join(self.save_dir, 'hyperopt_results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f'\nOptimization completed!')
        print(f'Best params: {results["best_params"]}')
        print(f'Best value: {results["best_value"]:.6f}')
        print(f'Results saved to: {result_path}')

        return results['best_params'], results['best_value']
