"""
BrickTS: Brick-like Time Series Forecasting Model

A modular framework for MTSF that combines three orthogonal axes:
- Level (Interaction Level): direct, decomposition, spectral
- Scope (Interaction Scope): global, local, hierarchical, sparse
- Architecture (Interaction Architecture): mlp, rnn, cnn, transformer

Reference:
- PDF p5~p7: Three-axis framework definition
- Scope: PDF p6 Eq.(13)~(18)
- Level: PDF p7 Eq.(19)~(26)

Note: CI/CD mode is controlled externally by run_training.py/run_hyperopt.py via --mode flag.
      The model assumes CD mode by default.
"""

from .model import Model, BrickTS
from .axis_level import (
    LevelDirect,
    LevelDecomposition,
    LevelSpectral,
    LEVEL_REGISTRY,
    build_level,
)
from .axis_scope import (
    ScopeGlobal,
    ScopeLocal,
    ScopeHierarchical,
    ScopeSparse,
    SCOPE_REGISTRY,
    build_scope,
)
from .axis_arch import (
    ArchMLP,
    ArchRNN,
    ArchCNN,
    ArchTransformer,
    ARCH_REGISTRY,
    build_arch,
)

__all__ = [
    # Main model
    'Model',
    'BrickTS',
    # Level modules
    'LevelDirect',
    'LevelDecomposition',
    'LevelSpectral',
    'LEVEL_REGISTRY',
    'build_level',
    # Scope modules
    'ScopeGlobal',
    'ScopeLocal',
    'ScopeHierarchical',
    'ScopeSparse',
    'SCOPE_REGISTRY',
    'build_scope',
    # Architecture modules
    'ArchMLP',
    'ArchRNN',
    'ArchCNN',
    'ArchTransformer',
    'ARCH_REGISTRY',
    'build_arch',
]
