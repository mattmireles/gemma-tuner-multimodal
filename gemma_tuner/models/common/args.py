"""
Training Arguments Builder for Gemma Fine-Tuning Models

This module provides common argument construction utilities for all Gemma
fine-tuning variants (standard, LoRA, distillation). It handles platform-specific
optimizations for data loading and processing, particularly for Apple Silicon's
unified memory architecture.

Key responsibilities:
- Platform-optimized worker count determination
- Common training argument construction
- Device-specific default configuration
- Configuration value normalization and validation

Called by:
- models.gemma.finetune.py for standard fine-tuning setup
- models.gemma.finetune.py for LoRA configuration
- models.gemma.finetune.py for distillation setup

Platform optimizations:
- MPS (Apple Silicon): Single preprocessing worker, no dataloader workers
- CUDA: Multiple workers for both preprocessing and dataloading
- CPU: Default worker counts for balanced performance

Worker count rationale:
Apple Silicon's unified memory architecture doesn't benefit from multiple
workers due to lack of CPU-GPU transfer overhead. Multiple workers can
actually hurt performance due to GIL contention and memory pressure.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

from gemma_tuner.constants import TrainingDefaults

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


# Training argument constants
class TrainingArgConstants:
    """Named constants for training argument defaults."""

    # Worker count defaults by platform
    MPS_PREPROCESSING_WORKERS = 1  # Single worker for unified memory
    MPS_DATALOADER_WORKERS = 0  # No multiprocessing for MPS
    DEFAULT_PREPROCESSING_WORKERS = None  # Let datasets library decide
    DEFAULT_DATALOADER_WORKERS = 4  # Standard for CUDA/CPU
    # Heuristic memory requirement per Dataset.map() worker (GB)
    DEFAULT_MEMORY_GB_PER_WORKER = 1.5

    # Evaluation strategy aliases
    EVAL_STRATEGY_ALIASES = ["evaluation_strategy", "eval_strategy"]
    DEFAULT_EVAL_STRATEGY = "no"

    # Training defaults
    # Use "none" to avoid requiring tensorboard unless explicitly requested
    DEFAULT_REPORT_TO = "none"

    # Boolean string values
    TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}


def get_effective_preprocessing_workers(profile_config: Dict, device) -> int | None:
    """
    Determines optimal preprocessing worker count using real-time system resources.

    Intelligent default: computes a safe number of parallel workers for
    HuggingFace Datasets' Dataset.map() by considering both available CPU
    cores and currently available system memory. Uses a conservative
    memory-per-worker heuristic to avoid unified memory pressure on Apple
    Silicon and similar resource-constrained systems.

    Called by:
    - models.gemma.finetune.py:prepare_dataset() for data preprocessing
    - models.gemma.finetune.py for teacher model preprocessing
    - Training scripts setting up data pipelines

    Behavior:
    - If profile_config explicitly sets a positive `preprocessing_num_workers`,
      that value is respected.
    - If set to 0 or negative, or not provided, dynamically determine workers as:
        safe_workers = max(1, min(cpu_cores, int(available_ram_gb / mem_per_worker_gb)))
      with mem_per_worker_gb = 1.5 (tunable via constant).

    Configuration override:
    Profile can specify "preprocessing_num_workers" to override defaults.
    Set to 0 or negative for auto-detection (dynamic resource-aware mode).

    Args:
        profile_config (Dict): Training configuration with optional worker count
        device: PyTorch device object indicating compute platform

    Returns:
        int | None: Number of workers (None uses library default)
    """
    logger = logging.getLogger(__name__)

    # MPS (Apple Silicon): single preprocessing worker avoids GIL contention
    # and unified-memory pressure; mirrors get_effective_dataloader_workers logic
    device_type = getattr(device, "type", None)
    if device_type == "mps":
        return TrainingArgConstants.MPS_PREPROCESSING_WORKERS

    configured = profile_config.get("preprocessing_num_workers")
    # Honor explicit configuration if provided
    if configured is not None:
        try:
            coerced = int(configured)
            if coerced <= 0:
                # Treat non-positive values as "auto": fall through to dynamic computation
                pass
            else:
                return coerced
        except Exception:
            # Fall through to auto on invalid config
            pass

    # Dynamic resource-aware computation
    cpu_cores = os.cpu_count() or 1
    mem_per_worker_gb = float(getattr(TrainingArgConstants, "DEFAULT_MEMORY_GB_PER_WORKER", 1.5))
    available_gb = None
    if psutil is not None:
        try:
            available_gb = float(psutil.virtual_memory().available) / float(1024**3)
        except Exception:
            available_gb = None

    # If memory could not be determined, default to conservative CPU-bound choice: ~50% of cores
    if available_gb is None:
        safe_workers = max(1, int(cpu_cores * 0.5))
        logger.info(
            "Preprocessing workers (fallback): cpu_cores=%s, using %s workers (memory unknown)",
            cpu_cores,
            safe_workers,
        )
        return safe_workers

    max_workers_by_mem = max(0, int(available_gb / mem_per_worker_gb))
    safe_workers = max(1, min(cpu_cores, max_workers_by_mem))
    logger.info(
        "Preprocessing workers (dynamic): available_ram_gb=%.2f, cpu_cores=%d, mem_per_worker_gb=%.2f => using %d workers",
        available_gb,
        cpu_cores,
        mem_per_worker_gb,
        safe_workers,
    )
    return safe_workers


def get_effective_dataloader_workers(profile_config: Dict, device) -> int:
    """
    Determines optimal DataLoader worker count for training based on platform.

    This function configures PyTorch DataLoader parallelism with platform-specific
    optimizations. Critical for training performance as it affects batch loading
    speed and GPU utilization.

    Called by:
    - All training scripts when creating DataLoaders
    - models/*/finetune.py when setting training arguments
    - Evaluation scripts for inference dataloading

    Platform optimization rationale:

    MPS (Apple Silicon) - 0 workers:
    - Multiprocessing overhead exceeds benefits
    - Unified memory eliminates transfer overhead
    - Python GIL contention with multiple processes
    - Main process loading is actually faster

    CUDA - 4 workers (default):
    - Hides PCIe transfer latency with parallel loading
    - Overlaps data preparation with GPU computation
    - Multiple workers keep GPU fed with data

    CPU - 4 workers:
    - Parallelizes I/O operations
    - Utilizes multiple CPU cores for preprocessing

    Args:
        profile_config (Dict): Configuration with optional dataloader_num_workers
        device: PyTorch device indicating platform

    Returns:
        int: Number of DataLoader workers (0 means main process)

    Example:
        >>> device = torch.device("cuda")
        >>> workers = get_effective_dataloader_workers({}, device)
        >>> print(workers)  # 4 (standard for CUDA)

    Performance impact:
    Wrong worker count can cause 2-10x training slowdown.
    MPS with workers>0 often causes 50% performance degradation.
    """
    device_type = getattr(device, "type", None)

    # On MPS, force 0 workers regardless of override to avoid multiprocessing issues and pickling constraints
    if device_type == "mps":
        return TrainingArgConstants.MPS_DATALOADER_WORKERS

    # Non-MPS: honor explicit configuration if valid, else use default
    configured = profile_config.get("dataloader_num_workers")
    if configured is not None:
        try:
            return int(configured)
        except Exception:
            return TrainingArgConstants.DEFAULT_DATALOADER_WORKERS

    return TrainingArgConstants.DEFAULT_DATALOADER_WORKERS
