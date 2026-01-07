# Time Series Forecasting Library - Version 1

A comprehensive framework for time series forecasting with support for multiple models, Channel Dependency (CD) and Channel Independency (CI) modes, and automated hyperparameter optimization.

## Features

- **Multiple Model Support**: 20+ models including Autoformer, SegRNN, TimeMixer, SCINet, Linear, DLinear, RNN, TCN, DSSRNN, SSRNN, Transformer, TimesNet, TSMixer, TiDE, Informer, Pyraformer, MICN, LightTS, TimeFilter, MultiPatchFormer, WPMixer, and more
- **Flexible Channel Modes**:
  - **CD (Channel Dependency)**: Train one model using all features together
  - **CI (Channel Independency)**: Train separate models for each feature
- **Automated Hyperparameter Optimization**: Using Optuna with TPE or Random sampling
- **Comprehensive Visualization**: Jupyter notebook for analyzing and comparing results
- **Organized Results Structure**: Hierarchical directory structure for easy management

## Directory Structure

```
Time-Series-Library-ver1/
├── configs/
│   ├── hyperopt_config.yaml          # Common hyperparameters
│   └── models/                        # Model-specific configs
│       ├── Autoformer.yaml
│       ├── SegRNN.yaml
│       ├── TimeMixer.yaml
│       ├── SCINet.yaml
│       └── ... (17+ models)
├── data_provider/
│   ├── data_factory.py                # Data loader factory
│   └── data_loader.py                 # Custom dataset class
├── exp/
│   ├── exp_basic.py                   # Base experiment class
│   └── exp_forecasting.py             # Forecasting experiment
├── models/                            # Model implementations
├── optimization/
│   └── hyperopt.py                    # Hyperparameter optimization
├── utils/
│   ├── metrics.py                     # Evaluation metrics
│   └── tools.py                       # Utility functions
├── dataset/                           # Your datasets go here
├── run_hyperopt.py                    # Hyperparameter optimization script
├── run_training.py                    # Training script
├── test_dataloader.py                 # Data loader testing
├── results_viz.ipynb                  # Results visualization notebook
├── README.md
└── Troubleshooting.md
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install torch numpy pandas scikit-learn optuna pyyaml matplotlib seaborn jupyter
```

## Quick Start

### 1. Prepare Your Data

Place your CSV file in the `dataset/` directory:
```
dataset/
└── your_dataset.csv
```

Your CSV should have:
- First column: date/timestamp (optional)
- Other columns: feature values

Example:
```csv
date,feature1,feature2,feature3
2020-01-01,100,200,300
2020-01-02,110,210,310
...
```

### 2. Run Hyperparameter Optimization

**For CD mode (all features together):**
```bash
python run_hyperopt.py \
    --model Autoformer \
    --mode CD \
    --data your_dataset \
    --data_path your_dataset.csv \
    --features feature1 feature2 feature3 \
    --seq_len 96 \
    --pred_len 96 \
    --n_trials 50
```

**For CI mode (separate model per feature):**
```bash
python run_hyperopt.py \
    --model Autoformer \
    --mode CI \
    --data your_dataset \
    --data_path your_dataset.csv \
    --features feature1 feature2 feature3 \
    --seq_len 96 \
    --pred_len 96 \
    --n_trials 50
```

### 3. Train with Optimized Hyperparameters

```bash
python run_training.py \
    --model Autoformer \
    --mode CD \
    --data your_dataset \
    --data_path your_dataset.csv \
    --features feature1 feature2 feature3 \
    --seq_len 96 \
    --pred_len 96 \
    --n_repeats 3
```

### 4. Visualize Results

```bash
jupyter notebook results_viz.ipynb
```

Then configure the visualization filters:
```python
FILTER_DATASET = 'your_dataset'
FILTER_SEQ_LEN = 96
FILTER_PRED_LEN = 96
FILTER_MODELS = ['Autoformer', 'SegRNN']
FILTER_MODES = ['CD', 'CI']
```

## Usage Details

### Command-Line Arguments

#### Data Configuration
- `--data`: Dataset name
- `--root_path`: Root path of data file (default: `./dataset/`)
- `--data_path`: CSV filename
- `--features`: List of feature names (omit for all features)
- `--target_features`: For CI mode, which features to predict

#### Model Configuration
- `--model`: Model name (see supported models below)
- `--mode`: Training mode (`CD` or `CI`)
- `--seq_len`: Input sequence length (default: 96)
- `--label_len`: Decoder input length (default: 48, auto-adjusted)
- `--pred_len`: Prediction horizon (default: 96)

#### Training Configuration
- `--train_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--n_repeats`: Number of experiment repeats (default: 1)

#### Hyperparameter Optimization
- `--n_trials`: Number of Optuna trials (default: 50)
- `--sampler`: Sampler type (`tpe` or `random`)
- `--use_hyperopt`: Use hyperparameter optimization results (default: True)

#### GPU Configuration
- `--use_gpu`: Use GPU (default: True)
- `--gpu`: GPU device ID (default: 0)

### Supported Models

| Model | Config File | Description |
|-------|-------------|-------------|
| Autoformer | `Autoformer.yaml` | Auto-correlation mechanism |
| SegRNN | `SegRNN.yaml` | Segment-wise RNN |
| TimeMixer | `TimeMixer.yaml` | Multi-scale mixing |
| SCINet | `SCINet.yaml` | Sample convolution and interaction |
| Linear | `Linear.yaml` | Simple linear baseline |
| DLinear | `DLinear.yaml` | Decomposition linear |
| RNN | `RNN.yaml` | Vanilla RNN/GRU/LSTM |
| TCN | `TCN.yaml` | Temporal convolutional network |
| DSSRNN | `DSSRNN.yaml` | Dual-stage structured RNN |
| SSRNN | `SSRNN.yaml` | State space RNN |
| Transformer | `Transformer.yaml` | Standard transformer |
| TimesNet | `TimesNet.yaml` | TimesBlock-based |
| TSMixer | `TSMixer.yaml` | MLP-Mixer for time series |
| TiDE | `TiDE.yaml` | Time series dense encoder |
| Informer | `Informer.yaml` | ProbSparse attention |
| Pyraformer | `Pyraformer.yaml` | Pyramidal attention |
| MICN | `MICN.yaml` | Multi-scale isometric convolution |
| LightTS | `LightTS.yaml` | Lightweight time series |
| TimeFilter | `TimeFilter.yaml` | Frequency-based filtering |
| MultiPatchFormer | `MultiPatchFormer.yaml` | Multi-scale patch transformer |
| WPMixer | `WPMixer.yaml` | Wavelet-patch mixer |
| And more... | | |

### Modes Explained

#### CD (Channel Dependency)
- Trains **one model** that uses all features together
- Good when features are correlated
- More efficient (single model)
- Example: Predicting temperature, humidity, and pressure together

#### CI (Channel Independency)
- Trains **separate models** for each feature
- Good when features are independent
- More flexible (feature-specific optimization)
- Example: Predicting multiple unrelated KPIs

## Results Structure

Results are saved in a hierarchical structure:

```
results/
└── {DATASET}/
    └── {seq_len}_{pred_len}/
        └── {MODEL}/
            └── {setting}/
                ├── metrics.npy      # [MAE, MSE, RMSE, MAPE, MSPE]
                ├── pred.npy         # Predictions
                ├── true.npy         # Ground truth
                └── input.npy        # Input sequences
```

Example:
```
results/
└── milano_6165/
    └── 96_96/
        ├── Autoformer/
        │   ├── Autoformer_CD_repeat0/
        │   │   ├── metrics.npy
        │   │   ├── pred.npy
        │   │   ├── true.npy
        │   │   └── input.npy
        │   └── Autoformer_CI_OT_repeat0/
        │       └── ...
        └── SegRNN/
            └── SegRNN_CD_repeat0/
                └── ...
```

## Hyperparameter Optimization

### Configuration Files

**Common hyperparameters** (`configs/hyperopt_config.yaml`):
```yaml
common_params:
  learning_rate:
    type: float
    min: 1.0e-5
    max: 1.0e-2
    log: true
  batch_size:
    type: categorical
    choices: [16, 32, 64, 128, 256]
  dropout:
    type: float
    min: 0.0
    max: 0.5

patience: 10
max_epochs: 50
```

**Model-specific hyperparameters** (`configs/models/{MODEL}.yaml`):
```yaml
model_params:
  d_model:
    type: categorical
    choices: [64, 128, 256, 512]
  n_heads:
    type: categorical
    choices: [4, 8, 16]
  e_layers:
    type: int
    min: 1
    max: 3
```

### Results

Optimization results are saved in:
```
hyperopt_archive/
└── {DATASET}/
    └── {seq_len}_{pred_len}/
        └── {MODEL}_{MODE}/
            └── {feature}/              # For CI mode
                └── hyperopt_results.json
```

## Visualization

The `results_viz.ipynb` notebook provides:

1. **Automatic Loading**: Scans and loads all results
2. **Filtering**: Select specific datasets, models, and configurations
3. **Metrics Tables**:
   - Per-feature results
   - CI average across all features
   - CD vs CI comparison
4. **Visualizations**:
   - Metric comparison bar charts
   - Input + Prediction + Ground truth plots
   - Error distribution histograms
5. **Export**: Save results to CSV

### Example Usage

```python
# Section 1: Load all results (automatic)
# Section 2: Configure filters
FILTER_DATASET = 'milano_6165'
FILTER_SEQ_LEN = 96
FILTER_PRED_LEN = 96
FILTER_MODELS = ['Autoformer', 'SegRNN']
FILTER_MODES = ['CD', 'CI']

# Section 3-6: View results and visualizations
# Section 7: Export to CSV
```

## Examples

### Example 1: Basic Forecasting with CD Mode

```bash
# 1. Optimize hyperparameters
python run_hyperopt.py \
    --model Autoformer \
    --mode CD \
    --data weather \
    --data_path weather.csv \
    --seq_len 96 \
    --pred_len 96 \
    --n_trials 30

# 2. Train with optimized hyperparameters
python run_training.py \
    --model Autoformer \
    --mode CD \
    --data weather \
    --data_path weather.csv \
    --seq_len 96 \
    --pred_len 96 \
    --n_repeats 5
```

### Example 2: CI Mode with Specific Features

```bash
# Optimize for specific features only
python run_hyperopt.py \
    --model SegRNN \
    --mode CI \
    --data traffic \
    --data_path traffic.csv \
    --features sensor1 sensor2 sensor3 \
    --seq_len 168 \
    --pred_len 24 \
    --n_trials 50

# Train
python run_training.py \
    --model SegRNN \
    --mode CI \
    --data traffic \
    --data_path traffic.csv \
    --features sensor1 sensor2 sensor3 \
    --seq_len 168 \
    --pred_len 24 \
    --n_repeats 3
```

### Example 3: Multiple Models Comparison

```bash
# Run for multiple models
for model in Autoformer SegRNN TimeMixer SCINet; do
    python run_hyperopt.py --model $model --mode CD --data electricity \
        --seq_len 96 --pred_len 96 --n_trials 30

    python run_training.py --model $model --mode CD --data electricity \
        --seq_len 96 --pred_len 96 --n_repeats 3
done

# Then visualize all results in results_viz.ipynb
```

## Testing

### Test Data Loader

```bash
python test_dataloader.py
```

This will:
- Load your dataset
- Check data shapes
- Verify train/val/test splits
- Test `__getitem__` method

## Troubleshooting

Common issues and solutions are documented in [Troubleshooting.md](Troubleshooting.md).

Quick checklist:
- ✓ Data file exists in `dataset/` directory
- ✓ Feature names match column names in CSV
- ✓ `seq_len` and `pred_len` are appropriate for your data length
- ✓ GPU is available if `--use_gpu` is set
- ✓ Sufficient disk space for results and checkpoints

## Advanced Usage

### Custom Hyperparameter Spaces

Edit `configs/hyperopt_config.yaml` or `configs/models/{MODEL}.yaml`:

```yaml
model_params:
  my_parameter:
    type: float
    min: 0.1
    max: 1.0
    log: false
```

### Custom Models

1. Implement your model in `models/YourModel.py`
2. Add config file `configs/models/YourModel.yaml`
3. Register in `exp/exp_basic.py`:
```python
self.model_dict = {
    ...
    'YourModel': YourModel,
}
```

### Different Data Splits

Default split is 70% train, 10% val, 20% test. To change, modify `data_loader.py`:

```python
num_train = int(len(df_raw) * 0.7)  # Change this
num_val = int(len(df_raw) * 0.1)    # And this
```

## Performance Tips

1. **Use GPU**: Set `--use_gpu` for faster training
2. **Adjust batch size**: Larger batches (64, 128) for faster training
3. **Reduce trials**: Start with 20-30 trials for quick experiments
4. **Parallel experiments**: Run multiple models simultaneously on different GPUs

```bash
CUDA_VISIBLE_DEVICES=0 python run_hyperopt.py --model Autoformer ... &
CUDA_VISIBLE_DEVICES=1 python run_hyperopt.py --model SegRNN ... &
```

## Citation

If you use this code in your research, please cite the original Time-Series-Library:

```bibtex
@inproceedings{wu2022timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## License

This project is based on the Time-Series-Library and follows the same license.

## Acknowledgments

- Original Time-Series-Library: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- Optuna: [optuna/optuna](https://github.com/optuna/optuna)

## Contact

For questions or issues:
- Open an issue on GitHub
- Check [Troubleshooting.md](Troubleshooting.md)
- Review example commands in this README

---

**Happy Forecasting! 📈**
