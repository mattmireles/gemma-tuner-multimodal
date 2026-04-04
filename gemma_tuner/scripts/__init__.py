#!/usr/bin/env python3
"""
Gemma Fine-Tuning Scripts Package

Provides scripts for Gemma model fine-tuning, evaluation, and dataset management.
Implements a complete workflow from raw audio data to production-ready fine-tuned
models, with extensive support for Apple Silicon (MPS), CUDA, and CPU platforms.

Core Training and Evaluation:
- finetune.py: Fine-tuning orchestrator with Gemma routing
- evaluate.py: Evaluation system with metrics and analysis
- blacklist.py: Outlier detection and quality management
- system_check.py: Hardware compatibility validation

Data Management:
- prepare_data.py: Dataset preparation with audio processing
- pseudo_label.py: Pseudo-labeling for dataset augmentation
- gather.py: Multi-experiment result aggregation

Utilities:
- utils.py: Shared utilities for experiment tracking and metadata
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
__version__ = "1.0.0"
__author__ = "Gemma macOS Tuner"
__description__ = "Scripts package for Gemma model fine-tuning and evaluation"

# Export commonly used modules for package-level imports.
# Be tolerant of optional dependencies when imported in lightweight environments (CI fast tests).
try:  # pragma: no cover - import surface
    from . import (
        blacklist,
        evaluate,
        finetune,
        gather,
        inference_common,
        prepare_data,
        pseudo_label,
        system_check,
        utils,
    )
except Exception:
    # Allow partial import if some heavy submodules raise due to optional deps.
    pass

# Package-level constants for cross-script consistency.
from gemma_tuner.constants import (
    Evaluation,
    FileSystem,
    MemoryLimits,
)
from gemma_tuner.models.gemma.constants import AudioProcessingConstants

DEFAULT_OUTPUT_DIR = FileSystem.OUTPUT_DIR_DEFAULT
DEFAULT_SAMPLING_RATE = AudioProcessingConstants.DEFAULT_SAMPLING_RATE  # 16 kHz
UNKNOWN_LANGUAGE_TOKEN = "??"

# Device optimization constants
MPS_MEMORY_RATIO = MemoryLimits.MPS_DEFAULT_FRACTION
CUDA_MEMORY_RATIO = MemoryLimits.CUDA_DEFAULT_FRACTION
CPU_BATCH_SIZE = 8

# Quality assurance constants
DEFAULT_WER_THRESHOLD = Evaluation.WER_THRESHOLD_TRAINING
VALIDATION_WER_THRESHOLD = Evaluation.WER_THRESHOLD_VALIDATION
MIN_AUDIO_DURATION = 0.1

__all__ = [
    # Core modules
    "blacklist",
    "evaluate",
    "inference_common",
    "finetune",
    "gather",
    "prepare_data",
    "pseudo_label",
    "system_check",
    "utils",
    # Package constants
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SAMPLING_RATE",
    "UNKNOWN_LANGUAGE_TOKEN",
    "MPS_MEMORY_RATIO",
    "CUDA_MEMORY_RATIO",
    "CPU_BATCH_SIZE",
    "DEFAULT_WER_THRESHOLD",
    "VALIDATION_WER_THRESHOLD",
    "MIN_AUDIO_DURATION",
]
