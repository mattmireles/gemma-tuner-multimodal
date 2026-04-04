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


def build_common_training_kwargs(profile_config: Dict) -> Dict:
    """
    Constructs common training arguments shared across all Gemma fine-tuning variants.

    This function builds a standardized dictionary of training arguments that can be
    used with HuggingFace's Seq2SeqTrainingArguments. It handles configuration
    normalization, type conversion, and default value application.

    Called by:
    - models.gemma.finetune.py:main() when creating training arguments
    - models.gemma.finetune.py for LoRA training setup
    - models.gemma.finetune.py for distillation arguments

    Argument categories:

    Training hyperparameters:
    - per_device_train_batch_size: Samples per GPU/device
    - gradient_accumulation_steps: Steps before optimizer update
    - num_train_epochs: Total training epochs
    - learning_rate: AdamW learning rate
    - warmup_steps: Linear warmup steps

    Checkpointing and logging:
    - save_strategy: When to save checkpoints (steps/epoch)
    - save_steps: Frequency of checkpoint saves
    - save_total_limit: Maximum checkpoints to keep
    - logging_steps: Frequency of loss logging
    - report_to: Logging backend (tensorboard)

    Evaluation settings:
    - evaluation_strategy: When to run validation (steps/epoch/no)
    - predict_with_generate: Use generation for validation

    Memory optimization:
    - gradient_checkpointing: Trade compute for memory
    - remove_unused_columns: Keep all columns for custom processing

    Args:
        profile_config (Dict): Merged configuration from config.py

    Returns:
        Dict: Keyword arguments for Seq2SeqTrainingArguments

    Configuration aliases handled:
    - "evaluation_strategy" or "eval_strategy" → evaluation_strategy
    - Boolean strings ("true", "1", "yes") → Python bool

    Example:
        >>> config = {"per_device_train_batch_size": "8", "learning_rate": "5e-5"}
        >>> kwargs = build_common_training_kwargs(config)
        >>> training_args = Seq2SeqTrainingArguments(**kwargs, output_dir="...")

    Note:
        Model-specific implementations can override or extend these arguments
        before creating their TrainingArguments instance.

    NOTE: This function is not currently called by finetune.py:main() which builds
    TrainingArguments inline. If you are refactoring the training pipeline, consider
    using this function to centralize TrainingArguments construction.
    """
    # Handle evaluation_strategy key aliases for backward compatibility
    evaluation_strategy = (
        profile_config.get("evaluation_strategy")
        if "evaluation_strategy" in profile_config
        else profile_config.get("eval_strategy", TrainingArgConstants.DEFAULT_EVAL_STRATEGY)
    )

    # Provide robust defaults for optional keys when invoked outside INI profiles
    save_strategy = profile_config.get("save_strategy", "steps")
    save_steps = int(profile_config.get("save_steps", 1000))
    save_total_limit = int(profile_config.get("save_total_limit", 1))
    logging_steps = int(profile_config.get("logging_steps", 50))

    # Auto-enable optimizations for large models (medium and large)
    # Models >= 500M params benefit from gradient checkpointing and accumulation
    model_name = profile_config.get("base_model", profile_config.get("model", "")).lower()
    is_large_model = "medium" in model_name or "large" in model_name

    # Gradient checkpointing: trade compute for memory (allows larger batches)
    # If explicitly set in config, respect that; otherwise auto-enable for large models
    gradient_checkpointing = profile_config.get("gradient_checkpointing")
    if gradient_checkpointing is None:
        gradient_checkpointing = is_large_model
    else:
        gradient_checkpointing = str(gradient_checkpointing).lower() in TrainingArgConstants.TRUTHY_VALUES

    # Gradient accumulation: increase effective batch size without more memory
    # If explicitly set in config, respect that; otherwise auto-set to 2 for large models
    gradient_accumulation_steps = profile_config.get("gradient_accumulation_steps")
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 2 if is_large_model else 1
    else:
        gradient_accumulation_steps = int(gradient_accumulation_steps)

    return {
        # Training hyperparameters with type conversion
        "per_device_train_batch_size": int(profile_config["per_device_train_batch_size"]),
        "num_train_epochs": int(profile_config["num_train_epochs"]),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": float(profile_config["learning_rate"]),
        "warmup_steps": int(profile_config["warmup_steps"]),
        # Checkpointing and saving configuration
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        # Logging configuration
        "logging_steps": logging_steps,
        "report_to": TrainingArgConstants.DEFAULT_REPORT_TO,
        # Evaluation configuration (use eval_strategy per HF 4.53+ API)
        "eval_strategy": evaluation_strategy,
        "predict_with_generate": True,  # Required for Gemma generation
        # Memory optimization
        "gradient_checkpointing": gradient_checkpointing,
        # Training mode flags
        "do_train": True,
        "remove_unused_columns": False,  # Keep all columns for audio processing
    }
