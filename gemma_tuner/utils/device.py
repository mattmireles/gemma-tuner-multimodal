#!/usr/bin/env python3
"""
Device selection, memory management, and diagnostics for Gemma fine-tuning.

Consolidated module covering:
- Device detection and selection (MPS / CUDA / CPU)
- Profile config defaults for the selected device
- Memory management (watermark, cache clearing, stats)
- System diagnostics and environment info

All callers import from `gemma_tuner.utils.device` — no submodules needed.
"""

from __future__ import annotations

import functools
import logging
import os
import platform
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from gemma_tuner.core.profile_config import ProfileConfig

logger = logging.getLogger(__name__)

try:
    from gemma_tuner.constants import MemoryLimits
except ImportError:

    class MemoryLimits:
        MPS_DEFAULT_FRACTION = 0.8
        CUDA_DEFAULT_FRACTION = 0.9


class MPSMemoryConstants:
    """Constants used for MPS memory pressure monitoring."""

    BYTES_TO_GB = 1024**3
    MEMORY_PRESSURE_WARNING_THRESHOLD = 85  # percent
    MEMORY_PRESSURE_CRITICAL_THRESHOLD = 90  # percent


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Return the best available torch device (MPS > CUDA > CPU), cached."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device")
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device
    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device


def to_bool(value) -> bool:
    """Coerce config values (bool, int, float, str, None) to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_device_defaults(profile_config: ProfileConfig | dict[str, Any]) -> None:
    """
    Apply device-specific configuration defaults in-place.

    Called by config loading after profile_config is assembled. Overrides dtype
    and attn_implementation for MPS compatibility, and warns about unsupported
    mixed-precision settings on Apple Silicon.
    """
    device = get_device()
    if device.type == "mps":
        if profile_config.get("dtype") != "float32":
            logger.warning("Overriding dtype to 'float32' for MPS compatibility.")
            profile_config["dtype"] = "float32"

        if profile_config.get("attn_implementation") != "eager":
            logger.warning("Overriding attn_implementation to 'eager' for MPS compatibility.")
            profile_config["attn_implementation"] = "eager"

        if profile_config.get("gradient_checkpointing"):
            logger.warning(
                "Gradient checkpointing is enabled on an MPS device. "
                "While this saves memory, it may lead to slower training. "
                "Consider disabling it if performance is a concern."
            )
            try:
                batch_size = int(profile_config.get("per_device_train_batch_size", 0) or 0)
            except Exception:
                batch_size = 0
            if batch_size and batch_size >= 8:
                logger.warning(
                    "On Apple Silicon, prefer larger batches with gradient_accumulation_steps "
                    "over gradient_checkpointing for throughput. Try gradient_checkpointing=false."
                )
            elif batch_size and batch_size <= 2:
                logger.warning(
                    "Very small per_device_train_batch_size detected with gradient_checkpointing=true. "
                    "This often hurts throughput on MPS. Consider increasing batch size and disabling checkpointing."
                )

        if to_bool(profile_config.get("fp16", "")):
            logger.warning(
                "fp16 mixed precision is not supported on MPS execution paths here; using float32. "
                "Set fp16=false on Apple Silicon unless explicitly required by your model."
            )
            profile_config["fp16"] = False
        if to_bool(profile_config.get("bf16", "")):
            logger.warning(
                "bf16 on MPS may fall back or reduce stability depending on PyTorch version. "
                "If you observe instability, switch to dtype=float32."
            )

    elif device.type == "cpu":
        profile_config.setdefault("dtype", "float32")
        profile_config.setdefault("attn_implementation", "eager")


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------


def probe_bfloat16(device: torch.device) -> bool:
    """
    Test whether bfloat16 tensors can be created on the given device.

    Returns True if bfloat16 is usable on this device, False otherwise.
    Replaces three independent inline try/except probe loops that previously
    lived in gemma_generate.py, gemma_profiler.py, and gemma_preflight.py.

    Called by:
    - gemma_tuner.scripts.gemma_generate.main() for dtype selection
    - gemma_tuner.scripts.gemma_profiler.main() for dtype selection
    - gemma_tuner.scripts.gemma_preflight.main() for capability reporting

    Device-specific logic:
    - mps:  Allocates a single-element bfloat16 tensor; if that raises, bfloat16
            is not supported by this PyTorch/macOS combination.
    - cuda: Delegates to torch.cuda.is_bf16_supported(), which queries the GPU's
            compute capability (requires SM 8.0 / Ampere or newer for full support).
    - cpu:  Returns False — CPU bfloat16 support is inconsistent across platforms
            and not used in this pipeline.

    Args:
        device: The torch.device to probe.

    Returns:
        bool: True if bfloat16 is usable on the given device, False otherwise.
    """
    if device.type == "mps":
        try:
            test_tensor = torch.zeros(1, device=device, dtype=torch.bfloat16)
            del test_tensor
            return True
        except Exception:
            return False
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    return False


def set_memory_fraction(fraction=MemoryLimits.MPS_DEFAULT_FRACTION) -> None:
    """
    Set device memory fraction before the training run starts.

    For MPS: sets PYTORCH_MPS_HIGH_WATERMARK_RATIO env var. Set this before
    importing device.py (and therefore torch) to guarantee the allocator picks
    up the value at initialization. Setting it after torch is loaded still
    records the env var and may affect future allocations, but the allocator's
    initial watermark will already be fixed.
    For CUDA: calls torch.cuda.set_per_process_memory_fraction().
    """
    device = get_device()

    if device.type == "mps":
        current_value = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        target_value = str(fraction)
        if current_value is None:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = target_value
            logger.info("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=%s", fraction)
        elif current_value != target_value:
            logger.warning(
                f"MPS memory watermark already set to {current_value}, "
                f"cannot change to {fraction} after PyTorch import. "
                "Set PYTORCH_MPS_HIGH_WATERMARK_RATIO before importing PyTorch."
            )
        else:
            logger.info(f"MPS memory watermark confirmed at {fraction}")
    elif device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"Set CUDA memory fraction to {fraction}")


def synchronize() -> None:
    """Synchronize the active device (MPS or CUDA) — no-op on CPU."""
    device = get_device()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache() -> None:
    """Release cached (but not allocated) memory on MPS or CUDA."""
    device = get_device()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def get_memory_stats() -> dict[str, Any]:
    """Return a dict of current memory statistics for the active device."""
    device = get_device()
    if device.type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory()
            stats: dict[str, Any] = {
                "allocated": allocated,
                "allocated_gb": allocated / MPSMemoryConstants.BYTES_TO_GB,
                "device": "mps",
            }
            try:
                import psutil

                memory_info = psutil.virtual_memory()
                stats["memory_pressure"] = {
                    "total_gb": memory_info.total / MPSMemoryConstants.BYTES_TO_GB,
                    "available_gb": memory_info.available / MPSMemoryConstants.BYTES_TO_GB,
                    "used_percent": memory_info.percent,
                    "warning": memory_info.percent > MPSMemoryConstants.MEMORY_PRESSURE_WARNING_THRESHOLD,
                }
                if memory_info.percent > MPSMemoryConstants.MEMORY_PRESSURE_CRITICAL_THRESHOLD:
                    logger.warning(
                        f"CRITICAL: System memory usage at {memory_info.percent:.1f}%. "
                        f"Risk of swapping affecting MPS performance. "
                        f"Available: {memory_info.available / MPSMemoryConstants.BYTES_TO_GB:.1f} GB"
                    )
                elif memory_info.percent > MPSMemoryConstants.MEMORY_PRESSURE_WARNING_THRESHOLD:
                    logger.warning(
                        f"HIGH: System memory usage at {memory_info.percent:.1f}%. "
                        f"Approaching memory pressure. Available: "
                        f"{memory_info.available / MPSMemoryConstants.BYTES_TO_GB:.1f} GB"
                    )
            except ImportError:
                stats["memory_pressure"] = {"note": "Install psutil for memory pressure monitoring"}
            except Exception as exc:
                stats["memory_pressure"] = {"error": f"Memory pressure query failed: {exc}"}
            return stats
        except AttributeError:
            return {"device": "mps", "note": "Memory stats not available - requires PyTorch 2.0+"}
        except RuntimeError:
            return {"device": "mps", "note": "Memory stats not available - MPS not initialized"}
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "allocated_gb": torch.cuda.memory_allocated() / MPSMemoryConstants.BYTES_TO_GB,
            "reserved_gb": torch.cuda.memory_reserved() / MPSMemoryConstants.BYTES_TO_GB,
            "device": "cuda",
        }
    return {"device": "cpu", "note": "Memory managed by system virtual memory"}


def check_memory_pressure() -> bool:
    """Return True if system memory is under warning-level pressure (MPS only)."""
    stats = get_memory_stats()
    if stats.get("device") == "mps" and "memory_pressure" in stats:
        pressure_info = stats["memory_pressure"]
        if isinstance(pressure_info, dict) and "warning" in pressure_info:
            return pressure_info["warning"]
    return False


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def get_env_info() -> dict[str, Any]:
    """Return a snapshot of the current environment (versions, device)."""
    try:
        import datasets as _datasets  # type: ignore
    except Exception:
        _datasets = None
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None
    try:
        import transformers as _transformers  # type: ignore
    except Exception:
        _transformers = None

    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "transformers": getattr(_transformers, "__version__", None) if _transformers else None,
        "datasets": getattr(_datasets, "__version__", None) if _datasets else None,
        "numpy": getattr(_np, "__version__", None) if _np is not None else None,
        "device": str(get_device()),
    }


def verify_mps_setup() -> tuple[bool, str]:
    """Return (is_available, message) for MPS readiness."""
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            return False, "PyTorch wasn't built with MPS support"
        return False, "MPS not available - check macOS 12.3+ and Apple Silicon hardware"
    return True, "MPS is available and configured"


def get_device_info() -> dict[str, Any]:
    """Return hardware details for the active device."""
    device = get_device()
    info: dict[str, Any] = {"device": str(device), "device_type": device.type}

    if device.type == "mps":
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
        info["unified_memory"] = True
        info["description"] = "Apple Silicon GPU (Metal Performance Shaders)"
    elif device.type == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name()
        info["device_count"] = torch.cuda.device_count()
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()
    else:
        info["description"] = "CPU device"

    return info
