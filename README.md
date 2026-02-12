# Reassembling Channel Dependency: A Systematic Benchmark through Modular Framework for Time Series Forecasting

## Abstract

Multivariate time series forecasting (MTSF) is a fundamental task that requires modeling both temporal dependencies and inter-channel relationships. Deep learning-based models have been widely adopted under two dominant learning strategies: **Channel-Dependent (CD)** approaches, which jointly leverage information from multiple channels, and **Channel-Independent (CI)** approaches, which process each channel separately. Despite their widespread use, limited understanding exists regarding when each strategy is most effective.

To address this gap, we propose a **three-dimensional analysis framework** comprising **Architecture**, **Scope**, and **Level**, which extends beyond conventional backbone-based categorization to enable fine-grained model characterization into **48 distinct types**. We conduct a comprehensive empirical study across **20 models** and **10 real-world datasets**, demonstrating that **no universal winner exists** among models or strategies; optimal choices are dataset-dependent.

Our framework reveals that the performance gaps between CD and CI vary systematically with model design, highlighting the necessity for automatic model selection. We validate this idea through **BrickTS**, a prototype modular forecasting model implemented under our three-dimensional framework. BrickTS not only confirms that performance variations align with the proposed dimensions but also identifies unexplored configurations that outperform existing baselines, demonstrating a promising direction to guide automatic agentic systems for MTSF.

**Key Contributions:**
- **Three-Dimensional Framework**: Classifies MTSF models along Architecture x Scope x Level axes (48 distinct types)
- **Comprehensive Benchmark**: Evaluates 20 models across 10 real-world datasets under both CD and CI strategies
- **BrickTS**: A modular prototype model for controlled evaluation of framework dimensions

## Directory Structure

```
.
├── configs/                    # Hyperparameter configurations
│   ├── hyperopt_config.yaml    # Common hyperparameter search space
│   └── models/                 # Model-specific hyperparameters
├── data_provider/              # Data loading with CD/CI support
│   ├── data_loader.py          # Dataset classes (CD/CI branching)
│   └── data_factory.py         # Data provider factory
├── exp/                        # Experiment logic
│   ├── exp_basic.py            # Base experiment class
│   └── exp_forecasting.py      # Forecasting experiment
├── layers/                     # Neural network layers
├── models/                     # Model implementations
│   ├── brickts/                # BrickTS modular model
│   └── *.py                    # Baseline models (DLinear, TimeMixer, etc.)
├── optimization/               # Hyperparameter optimization
│   └── hyperopt.py             # Optuna-based optimizer
├── utils/                      # Utilities (metrics, tools)
├── run_hyperopt.py             # Hyperparameter search script
├── run_training.py             # Training script
└── run_statsmodel.py           # Statistical models (ARIMA, VAR)
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n cdci python=3.11
conda activate cdci
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place CSV files in `./dataset/` directory.

### 3. Run Hyperparameter Optimization

```bash
# CD mode
python run_hyperopt.py \
    --model DLinear \
    --data ETTh1 \
    --mode CD \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --seq_len 96 \
    --pred_len 96 \
    --n_trials 10

# CI mode
python run_hyperopt.py \
    --model DLinear \
    --data ETTh1 \
    --mode CI \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --features OT \
    --target_features OT \
    --seq_len 96 \
    --pred_len 96 \
    --n_trials 10
```

### 4. Run Training

```bash
python run_training.py \
    --model DLinear \
    --data ETTh1 \
    --mode CD \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --seq_len 96 \
    --pred_len 96 \
    --n_repeats 5
```

## Experimental Settings

| Setting | Value |
|---------|-------|
| Data Split | Train:Val:Test = 7:1:2 |
| Input/Output Length | 96 (36 for short datasets) |
| Max Epochs | 30 |
| Early Stopping Patience | 10 |
| Repeated Runs | 5 |
| Hyperparameter Trials | 10 |
| Optimizer | Optuna (TPE Sampler) |
| Metric | MSE |

## Model Classification

### Three-Dimensional Framework

| Dimension | Categories | Description |
|-----------|------------|-------------|
| **Architecture** | MLP, RNN, CNN, Transformer | Backbone network type |
| **Scope** | Global, Local, Hierarchical, Sparse | Temporal dependency range |
| **Level** | Direct, Decomposition, Spectral | Channel feature extraction |

### Baseline Models

| Model | Architecture | Scope | Level |
|-------|-------------|-------|-------|
| Linear | MLP | Global | Direct |
| DLinear | MLP | Global | Decomposition |
| LightTS | MLP | Global | Direct |
| TiDE | MLP | Global | Direct |
| TSMixer | MLP | Global | Decomposition |
| TimeMixer | MLP | Hierarchical | Spectral |
| WPMixer | MLP | Hierarchical | Spectral |
| DUET | MLP | Sparse | Spectral |
| RNN | RNN | Global | Direct |
| SSRNN | RNN | Global | Direct |
| DSSRNN | RNN | Global | Decomposition |
| SegRNN | RNN | Local | Direct |
| TCN | CNN | Local | Direct |
| TimesNet | CNN | Hierarchical | Spectral |
| MICN | CNN | Hierarchical | Decomposition |
| SCINet | CNN | Hierarchical | Decomposition |
| Transformer | Transformer | Global | Direct |
| Informer | Transformer | Sparse | Direct |
| Autoformer | Transformer | Sparse | Spectral |
| Pyraformer | Transformer | Hierarchical | Decomposition |

## Datasets

| Dataset | Channels | Timesteps | Granularity | Description |
|---------|----------|-----------|-------------|-------------|
| ETTh1/h2 | 7 | 17,420 | 1 hour | Electricity transformer temperature |
| ETTm1/m2 | 7 | 69,680 | 15 min | Electricity transformer temperature |
| Traffic | 862 | 17,544 | 1 hour | Road occupancy rates |
| Electricity | 321 | 26,304 | 1 hour | Electricity consumption |
| Exchange | 8 | 7,588 | 1 day | Currency exchange rates |
| Weather | 21 | 52,696 | 10 min | Weather indicators |
| ILI | 7 | 966 | 1 week | Influenza-like illness rates |
| Milano-6165 | 5 | 4,320 | 10 min | Telecommunications activity |

## CD/CI Implementation

### Channel-Dependent (CD) Mode

Uses all channels as input to predict all channels simultaneously.

```python
# In data_loader.py
if self.mode == 'CD':
    cols_data = list(all_features)  # Use all features
```

### Channel-Independent (CI) Mode

Predicts each channel independently using only its own historical values.

```python
# In data_loader.py
if self.mode == 'CI':
    cols_data = [self.target_feature]  # Single feature only
```

### Usage

```bash
# CD: One model predicts all channels
python run_training.py --model DLinear --mode CD ...

# CI: Separate model per target channel
python run_training.py --model DLinear --mode CI --target_features OT ...
```

## BrickTS: Modular Framework

BrickTS is a prototype model for systematic exploration of the three-dimensional framework. It combines three orthogonal axes to create **48 possible configurations** (3 Level x 4 Scope x 4 Architecture).

### Architecture

```
Input X: (B, L, C)
         |
    +----+----+
    |         |
Level Module  Scope Module    <- Parallel feature extraction
    |         |
    +----+----+
         |
      Concat + Projection
         |
  Architecture Module         <- Backbone for prediction
         |
Output: (B, H, C_out)
```

### Module Options

| Axis | Options | Description |
|------|---------|-------------|
| **Level** | Direct, Decomposition, Spectral | How to extract channel features |
| **Scope** | Global, Local, Hierarchical, Sparse | How to capture temporal dependencies |
| **Architecture** | MLP, RNN, CNN, Transformer | Backbone network type |

### Configuration

```bash
python run_hyperopt.py --model BrickTS --mode CD --data ETTh1 ...
```

BrickTS hyperparameters (`configs/models/BrickTS.yaml`):
- `level_type`: [direct, decomposition, spectral]
- `scope_type`: [global, local, hierarchical, sparse]
- `arch_type`: [mlp, rnn, cnn, transformer]

## Hyperparameter Search Space

| Model | Parameter | Type | Range / Choices |
|-------|-----------|------|-----------------|
| **Common** | learning_rate | float (log) | 1e-5 ~ 1e-2 |
| | batch_size | categorical | [32, 64, 128, 256] |
| | dropout | float | 0.1 ~ 0.5 |
| **DLinear** | moving_avg | categorical | [7, 13, 25] |
| **RNN** | hidden_size | categorical | [64, 128, 256, 512] |
| | num_layers | int | 1 ~ 3 |
| | rnn_type | categorical | [LSTM, GRU] |
| **TCN** | num_channels | categorical | [[16,32,64], [32,64,128], [64,128,256]] |
| | kernel_size | int | 3 ~ 7 |
| **Transformer** | d_model | categorical | [64, 128, 256, 512] |
| | n_heads | categorical | [4, 8] |
| | e_layers | int | 1 ~ 3 |
| | d_ff | categorical | [256, 512, 1024] |
| | activation | categorical | [relu, gelu] |
| **TimeMixer** | d_model | categorical | [64, 128, 256, 512] |
| | d_ff | categorical | [128, 256, 512, 1024] |
| | e_layers | int | 1 ~ 3 |
| | down_sampling_layers | int | 1 ~ 3 |
| | down_sampling_method | categorical | [avg, max, conv] |
| **BrickTS** | level_type | categorical | [direct, decomposition, spectral] |
| | scope_type | categorical | [global, local, hierarchical, sparse] |
| | arch_type | categorical | [mlp, rnn, cnn, transformer] |


## Results

Results are saved in `results/{dataset}/{seq_len}_{pred_len}/{model}_{mode}_results.json`.

```json
{
    "overall": {
        "mse": 0.1234,
        "mae": 0.2345,
        "std": {"mse": 0.001, "mae": 0.002}
    },
    "scalability": {
        "inference_time_mean": 10.5,
        "inference_memory_gb_mean": 0.5,
        "num_params": 12345
    }
}
```

