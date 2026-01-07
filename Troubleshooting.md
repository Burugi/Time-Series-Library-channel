# Troubleshooting Guide

This document contains common issues encountered during development and their solutions.

---

## Table of Contents
1. [Model Input Dimension Errors](#1-model-input-dimension-errors)
2. [Length-Dependent Hyperparameter Issues](#2-length-dependent-hyperparameter-issues)
3. [Data Loading Errors](#3-data-loading-errors)
4. [SegRNN Hidden Size Mismatch](#4-segrnn-hidden-size-mismatch)
5. [Embedding Size Mismatch in Autoformer](#5-embedding-size-mismatch-in-autoformer)
6. [Results Path and Organization](#6-results-path-and-organization)

---

## 1. Model Input Dimension Errors

### Problem
```
RuntimeError: Given groups=1, weight of size [64, 7, 3], expected input[16, 5, 98] to have 7 channels, but got 5 channels instead
```

### Cause
- `enc_in` and `dec_in` were set as command-line arguments with default values (e.g., 7)
- These values didn't match the actual number of features in the dataset
- The model was initialized with wrong channel dimensions

### Solution
**Removed `enc_in` and `dec_in` from argument parsers:**
- File: `run_hyperopt.py`, `run_training.py`
- These values are now automatically set in `exp_basic.py` based on:
  - Mode (CD vs CI)
  - Actual number of features in the data

```python
# In exp_basic.py
def _prepare_model_configs(self):
    if self.args.mode == 'CI':
        # CI mode: single feature
        self.args.enc_in = 1
        self.args.dec_in = 1
        self.args.c_out = 1
    else:
        # CD mode: multiple features (auto-detected)
        if not hasattr(self.args, 'enc_in'):
            self.args.enc_in = len(self.args.features) if self.args.features else 7
        self.args.dec_in = self.args.enc_in
        self.args.c_out = self.args.enc_in
```

---

## 2. Length-Dependent Hyperparameter Issues

### Problem A: label_len >= pred_len
```
RuntimeError: The size of tensor a (72) must match the size of tensor b (84) at non-singleton dimension 1
```

**Cause:** Default `label_len=48` was too large when `seq_len=36` and `pred_len=36`.

**Solution:** Automatic adjustment in `run_hyperopt.py` and `run_training.py`:
```python
# Set label_len dynamically if not properly set
if args.label_len >= args.pred_len:
    args.label_len = args.seq_len // 2
    print(f'Warning: label_len adjusted to {args.label_len} (seq_len // 2)')
```

### Problem B: moving_avg >= seq_len
**Cause:** Default `moving_avg=25` hyperparameter choice could exceed `seq_len` for short sequences.

**Solution:** Validation in `optimization/hyperopt.py`:
```python
if 'moving_avg' in params:
    if params['moving_avg'] >= self.args.seq_len:
        params['moving_avg'] = max(3, self.args.seq_len // 2)
```

### Problem C: seg_len doesn't divide seq_len evenly
```
RuntimeError: Expected hidden size (1, 1792, 64), got [1, 1280, 64]
```

**Cause:** SegRNN requires `seg_len` to evenly divide both `seq_len` and `pred_len`.

**Solution:** Validation in `optimization/hyperopt.py`:
```python
if 'seg_len' in params:
    seq_len = self.args.seq_len
    pred_len = self.args.pred_len
    valid_seg_lens = []
    for sl in [6, 12, 24, 48, 96]:
        if sl <= min(seq_len, pred_len) and seq_len % sl == 0 and pred_len % sl == 0:
            valid_seg_lens.append(sl)

    if valid_seg_lens and params['seg_len'] not in valid_seg_lens:
        params['seg_len'] = min(valid_seg_lens, key=lambda x: abs(x - params['seg_len']))
```

---

## 3. Data Loading Errors

### Problem
```
ValueError: could not broadcast input array from shape (8640,) into shape (4320,)
```

### Cause
- Using `np.zeros_like()` with pandas DataFrame created unexpected shapes
- Duplicate features in the feature list (e.g., "callout callout")

### Solution
**Explicit shape specification in `data_loader.py`:**
```python
# Get number of rows and columns explicitly
n_rows = df_data.shape[0]
n_cols = df_data.shape[1]

# Create scaled_data with explicit shape
scaled_data = np.zeros((n_rows, n_cols), dtype=np.float32)

# Also remove duplicate features
if self.features is not None:
    seen = set()
    available_features = []
    for f in self.features:
        if f in cols_data and f not in seen:
            available_features.append(f)
            seen.add(f)
    cols_data = available_features
```

---

## 4. SegRNN Hidden Size Mismatch

### Problem
```
RuntimeError: Expected hidden size (1, 1792, 64), got [1, 1280, 64]
```

### Root Cause
- SegRNN calculates `seg_num_x = seq_len // seg_len` and `seg_num_y = pred_len // seg_len`
- Hidden state size depends on these values
- If `seg_len` doesn't evenly divide the sequence lengths, dimension mismatches occur

### Solution
See [Problem C](#problem-c-seg_len-doesnt-divide-seq_len-evenly) above for the validation logic.

---

## 5. Embedding Size Mismatch in Autoformer

### Problem
```
RuntimeError: The size of tensor a (72) must match the size of tensor b (84) at non-singleton dimension 1
```
When using `seq_len=36, pred_len=36, label_len=48`.

### Cause
- `label_len=48` was the default value
- For decoder input: `label_len + pred_len = 48 + 36 = 84`
- But model expected `seq_len + pred_len = 36 + 36 = 72`

### Solution
Automatic `label_len` adjustment (see [Problem A](#problem-a-label_len--pred_len)).

---

## 6. Results Path and Organization

### Initial Structure (Problematic)
```
results/
  └── Autoformer_CD_repeat0/
      ├── metrics.npy
      ├── pred.npy
      └── true.npy
```
- Hard to organize multiple datasets and configurations
- Can't distinguish between different models easily

### Final Structure (Improved)
```
results/
  └── {DATASET}/
      └── {seq_len}_{pred_len}/
          └── {MODEL}/
              └── {setting}/
                  ├── metrics.npy
                  ├── pred.npy
                  ├── true.npy
                  └── input.npy
```

**Example:**
```
results/
  └── milano_6165/
      └── 96_96/
          ├── Autoformer/
          │   ├── Autoformer_CD_repeat0/
          │   └── Autoformer_CI_OT_repeat0/
          └── SegRNN/
              └── SegRNN_CD_repeat0/
```

**Changes made:**
- `exp_forecasting.py`: Updated `folder_path` construction
- `results_viz.ipynb`: Updated scanning logic to match new structure
- Added `input.npy` to save input sequences for visualization

---

## 7. Checkpoint Management

### Problem
Checkpoint files accumulated in `./checkpoints/` directory after training.

### Solution
**Automatic cleanup after loading:**

In `exp_forecasting.py`:
```python
# After training
best_model_path = os.path.join(path, 'checkpoint.pth')
self.model.load_state_dict(torch.load(best_model_path))

# Delete checkpoint after loading
if os.path.exists(best_model_path):
    os.remove(best_model_path)
# Remove checkpoint directory if empty
if os.path.exists(path) and not os.listdir(path):
    os.rmdir(path)
```

In `optimization/hyperopt.py`:
```python
# After each trial
checkpoint_path = os.path.join(temp_path, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
if os.path.exists(temp_path) and not os.listdir(temp_path):
    os.rmdir(temp_path)
```

---

## 8. Visualization Issues

### Problem: "No filtered metrics to display"

**Cause:** Empty `filtered_df` after applying filters.

**Solution:** Added proper checks in `results_viz.ipynb`:
```python
if len(filtered_df) > 0:
    print(filtered_df.groupby(['Dataset', 'Seq_Len', 'Pred_Len', 'Model', 'Mode', 'Feature']).size())
else:
    print("No results match the filters!")
```

### Problem: CI vs CD comparison not fair

**Cause:** CI mode has per-feature results while CD has aggregate results.

**Solution:** Calculate CI average across all features:
```python
# For CI mode, calculate average across all features
for (dataset, seq_len, pred_len, model), group in ci_data.groupby(['Dataset', 'Seq_Len', 'Pred_Len', 'Model']):
    ci_avg_list.append({
        'Model': model,
        'Mode': 'CI_avg',
        'MAE_mean': group['MAE'].mean(),  # Average across all features
        ...
    })
```

---

## Best Practices

1. **Always check data dimensions** before training
   - Verify `enc_in`, `dec_in`, `c_out` match your actual data

2. **Validate hyperparameters** against sequence lengths
   - Ensure `label_len < pred_len`
   - Ensure `moving_avg < seq_len`
   - For SegRNN, ensure `seg_len` divides both `seq_len` and `pred_len`

3. **Use automatic configuration** where possible
   - Let the code infer dimensions from data
   - Use dynamic adjustment for length-dependent parameters

4. **Check results directory structure**
   - Ensure results are saved in the correct hierarchical structure
   - Use visualization notebook to verify results are properly loaded

5. **Monitor disk space**
   - Checkpoints are automatically cleaned up
   - But verify if you modify checkpoint saving logic

---

## Quick Reference Commands

### Test data loader
```bash
python test_dataloader.py
```

### Run hyperparameter optimization
```bash
python run_hyperopt.py --model Autoformer --mode CD --data milano_6165 \
    --features OT smsin smsout callin callout --seq_len 96 --pred_len 96 --n_trials 50
```

### Run training with optimized hyperparameters
```bash
python run_training.py --model Autoformer --mode CD --data milano_6165 \
    --features OT smsin smsout callin callout --seq_len 96 --pred_len 96 --n_repeats 3
```

### Visualize results
```bash
jupyter notebook results_viz.ipynb
```

---

## Getting Help

If you encounter an issue not covered here:

1. Check the error message carefully
2. Verify your command-line arguments
3. Check the results directory structure
4. Review the data loader configuration
5. Consult the README.md for usage examples

For model-specific issues, check the corresponding config file in `configs/models/`.
