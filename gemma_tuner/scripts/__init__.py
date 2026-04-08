#!/usr/bin/env python3
"""
Gemma Fine-Tuning Scripts Package

Provides scripts for Gemma model fine-tuning, evaluation, and dataset management.
Implements a complete workflow from training data to production-ready fine-tuned
models, with extensive support for Apple Silicon (MPS), CUDA, and CPU platforms.

Core Training and Evaluation:
- finetune.py: Fine-tuning orchestrator with Gemma routing
- evaluate.py: Evaluation system with metrics and analysis
- blacklist.py: Outlier detection and quality management
- system_check.py: Hardware compatibility validation

Data Management:
- prepare_data.py: Dataset preparation with audio processing
- gather.py: Multi-experiment result aggregation

Utilities:
- export.py: Model export and format conversion

Integration:
All scripts work with the `gemma-macos-tuner` CLI:
```bash
gemma-macos-tuner prepare dataset_name
gemma-macos-tuner finetune profile_name
gemma-macos-tuner evaluate profile_name
```
"""

# Package version and metadata
__version__ = "1.1.0"
__author__ = "Gemma macOS Tuner"
__description__ = "Scripts package for Gemma model fine-tuning and evaluation"

import warnings

# Export commonly used modules for package-level imports.
# Only ImportError is caught here — any other exception (SyntaxError, AttributeError,
# TypeError, etc.) is a real bug and must propagate so it is visible to the developer.
# ImportError is expected when optional dependencies (torch, jiwer, …) are absent.
try:  # pragma: no cover - import surface
    from . import (
        blacklist,
        evaluate,
        finetune,
        gather,
        inference_common,
        prepare_data,
        system_check,
    )
except ImportError as e:
    warnings.warn(
        f"Some script submodules could not be imported ({e}). "
        "This is expected if optional dependencies (e.g. torch, jiwer) are missing. "
        "Install them with: pip install gemma-macos-tuner[torch,eval]",
        ImportWarning,
        stacklevel=2,
    )

# Package-level constants for cross-script consistency.
from gemma_tuner.constants import FileSystem

DEFAULT_OUTPUT_DIR = FileSystem.OUTPUT_DIR_DEFAULT

__all__ = [
    # Core modules
    "blacklist",
    "evaluate",
    "inference_common",
    "finetune",
    "gather",
    "prepare_data",
    "system_check",
    # Package constants
    "DEFAULT_OUTPUT_DIR",
]
