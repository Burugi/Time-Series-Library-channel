import torch
import sys
from utils.tools import dotdict

# Test configuration
configs = dotdict({
    'task_name': 'long_term_forecast',
    'seq_len': 36,
    'label_len': 18,
    'pred_len': 36,
    'enc_in': 7,
    'dec_in': 7,
    'c_out': 7,
    'd_model': 128,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 256,
    'factor': 3,
    'dropout': 0.1,
    'activation': 'gelu',
    'embed': 'timeF',
    'freq': 'h',
    'moving_avg': 25,
    'individual': False,
    'top_k': 5,
    'num_kernels': 6,
    'feature_encode_dim': 2,
    'distil': True,
    'window_size': [4, 4],
    'inner_size': 5,
    'conv_kernel': [12, 16],
    'chunk_size': 24,
    'patch_len': 16,
    'alpha': 0.1,
    'top_p': 0.5,
    'pos': True,
    'd_hidden': 512,
    'patch_scales': [8, 16, 24, 32],
    'stride_scales': [8, 8, 7, 6],
    'wavelet': 'db2',
    'level': 1,
    'patch_stride': 8,
    'tfactor': 5,
    'dfactor': 5,
    'no_decomposition': False,
    'num_channels': [32, 64, 128],
    'kernel_size': 3,
    'hidden_size': 128,
    'num_layers': 2,
    'rnn_type': 'LSTM',
})

# Models to test
models_to_test = [
    'DLinear', 'Transformer', 'TimesNet', 'TSMixer', 'TiDE',
    'Informer', 'Pyraformer', 'MICN', 'LightTS', 'TimeFilter',
    'MultiPatchFormer', 'WPMixer',
    'Linear', 'TCN', 'RNN', 'DSSRNN', 'SSRNN'
]

# Test input shapes
batch_size = 32
x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)

print(f"Input shapes:")
print(f"  x_enc: {x_enc.shape}")
print(f"  x_mark_enc: {x_mark_enc.shape}")
print(f"  x_dec: {x_dec.shape}")
print(f"  x_mark_dec: {x_mark_dec.shape}")
print(f"\nExpected output shape: [{batch_size}, {configs.pred_len}, {configs.c_out}]")
print("\n" + "="*80)

results = {}

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")

    try:
        # Import model
        exec(f"from models.{model_name} import Model")
        model = eval("Model(configs)")

        # Forward pass
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Check output shape
        expected_shape = (batch_size, configs.pred_len, configs.c_out)

        if output.shape == expected_shape:
            print(f"✓ {model_name}: Output shape {output.shape} - CORRECT")
            results[model_name] = "PASS"
        else:
            print(f"✗ {model_name}: Output shape {output.shape} - INCORRECT (expected {expected_shape})")
            results[model_name] = f"FAIL - shape {output.shape}"

    except Exception as e:
        print(f"✗ {model_name}: ERROR - {str(e)}")
        results[model_name] = f"ERROR - {str(e)[:100]}"

print("\n" + "="*80)
print("\nSummary:")
print("-"*80)
passed = sum(1 for v in results.values() if v == "PASS")
total = len(results)
print(f"Passed: {passed}/{total}")
print("\nDetailed Results:")
for model_name, result in results.items():
    status = "✓" if result == "PASS" else "✗"
    print(f"{status} {model_name}: {result}")
