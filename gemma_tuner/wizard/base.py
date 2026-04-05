"""
Shared constants, types, and utility functions for the wizard package.

This module contains the foundational building blocks that all wizard submodules
depend on: named constants (WizardConstants), training method definitions
(TrainingMethod), model specification tables (ModelSpecs), and pure utility
functions that do not require any UI interaction (Rich console, questionary).

Extracted from the original monolithic wizard.py so that submodules can
    from gemma_tuner.wizard.base import WizardConstants, TrainingMethod, ModelSpecs, ...
without pulling in the entire wizard graph.

Why these items live here:
- WizardConstants: Referenced by nearly every wizard submodule for magic-number-free code.
- TrainingMethod / ModelSpecs: Shared data tables consumed by model selection,
  estimation, configuration generation, and confirmation screens.
- get_wizard_device_info(): Hardware detection used by welcome screen, model selection,
  training estimation, and confirmation screen.
- detect_datasets(): Dataset discovery used by the dataset selection step.

UI singletons (console, apple_style) are also defined here because every
submodule needs them and they must live in a single location to avoid
duplicate Rich Console instances.
"""

from pathlib import Path
from typing import Any, Dict, List

import questionary
from rich.console import Console

# Import existing utilities
from gemma_tuner.models.gemma.constants import AudioProcessingConstants
from gemma_tuner.utils.device import get_device

# ---------------------------------------------------------------------------
# Shared UI singletons — used by every wizard submodule
# ---------------------------------------------------------------------------

console = Console()

apple_style = questionary.Style(
    [
        ("qmark", "fg:#ff9500 bold"),
        ("question", "bold"),
        ("answer", "fg:#007aff bold"),
        ("pointer", "fg:#ff9500 bold"),
        ("highlighted", "fg:#007aff bold"),
        ("selected", "fg:#34c759 bold"),
        ("instruction", "fg:#8e8e93"),
        ("text", ""),
    ]
)


class WizardConstants:
    """Named constants for wizard configuration, user interface, and training estimation."""

    # Progressive Disclosure Timing Constants
    # These control the pacing of the Steve Jobs-inspired progressive disclosure UI
    ANIMATION_DELAY = 0.5  # Seconds between progressive UI reveals
    CONFIRMATION_WAIT = 2.0  # Seconds to display confirmation messages
    WELCOME_SCREEN_PAUSE = 1.0  # Seconds to display welcome screen animations

    # Training Estimation Constants
    # Used for calculating realistic training time and memory requirements
    BASE_SAMPLES_ESTIMATE = 100000  # Baseline sample count for time calculations
    SAMPLES_PER_FILE = 10  # Average samples per dataset file (rough estimate)
    MEMORY_SAFETY_BUFFER = 0.8  # Use only 80% of available memory (20% safety margin)
    HOURS_TO_MINUTES_CUTOFF = 1.0  # Show minutes instead of hours below this threshold

    # Apple Silicon Performance Multipliers
    # Device-specific optimization factors for training time estimation
    MPS_PERFORMANCE_MULTIPLIER = 1.0  # Apple Silicon baseline (unified memory architecture)
    CUDA_PERFORMANCE_MULTIPLIER = 0.7  # NVIDIA GPUs typically 30% faster than Apple Silicon
    CPU_PERFORMANCE_MULTIPLIER = 3.0  # CPU training is ~3x slower than Apple Silicon MPS

    # Audio Processing
    # Default audio sample rate for Gemma audio tower (USM-based).
    # Single source of truth lives in AudioProcessingConstants; aliased here for
    # local readability without magic numbers.
    DEFAULT_SAMPLING_RATE = AudioProcessingConstants.DEFAULT_SAMPLING_RATE

    # Dataset Detection Patterns
    # File extensions and patterns for automatic dataset discovery
    AUDIO_EXTENSIONS = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
    DATASET_FILES_PATTERN = "*.csv"
    SKIP_DIRECTORIES = {".cache", "__pycache__", ".git", ".DS_Store"}

    # Configuration Generation Constants
    # Defaults for wizard-generated training profiles
    DEFAULT_LORA_DROPOUT = 0.1  # Intentionally higher than GemmaTrainingConstants.LORA_DROPOUT (0.05) for wizard's conservative defaults
    # LoRA Configuration Presets
    # Pre-defined LoRA rank configurations with smart alpha defaults
    LORA_RANK_OPTIONS = [
        {"rank": 4, "description": "Ultra lightweight", "alpha": 8},
        {"rank": 8, "description": "Lightweight", "alpha": 16},
        {"rank": 16, "description": "Balanced ⭐ Recommended", "alpha": 32},
        {"rank": 32, "description": "High capacity", "alpha": 64},
        {"rank": 64, "description": "Maximum capacity", "alpha": 128},
    ]

    # BigQuery Import Constants
    # Default settings for BigQuery dataset import workflow
    DEFAULT_BQ_LIMIT = 1000  # Default row limit for BQ exports
    BQ_SAMPLING_OPTIONS = ["random", "first"]  # Available sampling strategies

    # Common HuggingFace Dataset Presets
    # Curated list of popular datasets for training
    RECOMMENDED_HF_DATASETS = [
        {"name": "mozilla-foundation/common_voice_13_0", "description": "Common Voice multilingual dataset"},
        {"name": "openslr/librispeech_asr", "description": "LibriSpeech English ASR dataset"},
        {"name": "facebook/voxpopuli", "description": "VoxPopuli multilingual dataset"},
    ]


class TrainingMethod:
    """Training method configurations with resource estimation multipliers.

    Gemma models use LoRA fine-tuning exclusively. Memory and time multipliers
    are calibrated from Apple Silicon benchmarking.
    """

    LORA = {
        "key": "lora",
        "name": "🎨 LoRA Fine-Tune",
        "description": "Memory-efficient parameter-efficient fine-tuning",
        "memory_multiplier": 0.4,  # ~60% memory savings through adapter architecture
        "time_multiplier": 0.8,  # 20% faster due to fewer parameters to update
        "quality": "high",  # 95-98% of standard fine-tuning quality
    }


class ModelSpecs:
    """Model specifications for estimation calculations."""

    MODELS = {
        # Gemma 3n Models
        "gemma-3n-e2b-it": {"params": "~2B", "memory_gb": 10.0, "hours_100k": 10.0, "hf_id": "google/gemma-3n-E2B-it"},
        "gemma-3n-e4b-it": {"params": "~4B", "memory_gb": 18.0, "hours_100k": 18.0, "hf_id": "google/gemma-3n-E4B-it"},
        # Gemma 4 Models
        "gemma-4-e2b-it": {"params": "~2B", "memory_gb": 10.0, "hours_100k": 9.0, "hf_id": "google/gemma-4-E2B-it"},
        "gemma-4-e4b-it": {"params": "~4B", "memory_gb": 18.0, "hours_100k": 16.0, "hf_id": "google/gemma-4-E4B-it"},
    }


def get_wizard_device_info() -> Dict[str, Any]:
    """
    Comprehensive device detection and performance profiling for training estimation.

    This is separate from utils/device.py:get_device_info() which returns a simpler
    schema without performance_multiplier or display_name. This version includes
    fields needed by the wizard estimator.

    This function provides detailed hardware analysis to enable accurate training time
    and memory requirements estimation. It handles the three primary training platforms
    (Apple Silicon MPS, NVIDIA CUDA, CPU) with platform-specific optimizations.

    Called by:
    - show_welcome_screen() for system status display
    - select_model() for memory constraint filtering
    - estimate_training_time() for performance multiplier application
    - show_confirmation_screen() for final hardware verification

    Calls to:
    - utils/device.py:get_device() for PyTorch device detection and MPS availability
    - psutil.virtual_memory() for system memory analysis and availability calculation
    - Platform-specific optimization lookup for performance multiplier determination

    Device-specific optimizations:

    Apple Silicon (MPS):
    - Unified memory architecture: CPU and GPU share same RAM pool
    - Performance multiplier: 1.0 (baseline for Apple-optimized training)
    - Memory efficiency: Excellent due to unified memory and Metal optimization
    - Thermal management: Integrated SoC design with shared thermal envelope

    NVIDIA CUDA:
    - Discrete GPU memory: Separate VRAM pool with high-bandwidth access
    - Performance multiplier: 0.7 (typically 30% faster than Apple Silicon)
    - Memory efficiency: Good with dedicated VRAM but transfer overhead
    - Scalability: Multi-GPU support and advanced optimization libraries

    CPU Fallback:
    - System RAM: Uses main memory with cache hierarchy optimization
    - Performance multiplier: 3.0 (significantly slower due to lack of parallel compute)
    - Memory efficiency: Poor due to lack of specialized ML compute units
    - Compatibility: Universal fallback for any hardware configuration

    Memory calculation considerations:
    - Total memory: Physical RAM available to the system
    - Available memory: Currently unused memory (accounting for OS and other processes)
    - Safety buffer: Reserve 20% of available memory to prevent system instability
    - Swap/virtual memory: Not considered due to severe performance penalties

    Returns:
        Dict containing comprehensive device information:
        {
            "type": PyTorch device type ("mps", "cuda", "cpu"),
            "name": Full device identifier string,
            "display_name": Human-readable device description,
            "total_memory_gb": Total system memory in GB,
            "available_memory_gb": Currently available memory in GB,
            "performance_multiplier": Relative training speed factor vs Apple Silicon
        }

    Example:
        device_info = get_wizard_device_info()
        if device_info["available_memory_gb"] > 16:
            # Sufficient memory for large model training
            enable_large_models = True
        training_hours *= device_info["performance_multiplier"]
    """
    # Primary device detection using shared utility with MPS availability checking
    device = get_device()

    # System memory analysis for training capacity planning
    # Uses psutil for cross-platform memory statistics
    import psutil

    memory_stats = psutil.virtual_memory()
    total_memory_gb = memory_stats.total / (1024**3)
    available_memory_gb = memory_stats.available / (1024**3)

    # Base device information structure
    device_info = {
        "type": device.type,
        "name": str(device),
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": available_memory_gb,
    }

    # Platform-specific optimization and performance characteristics
    if device.type == "mps":
        device_info["display_name"] = f"Apple Silicon ({device})"
        device_info["performance_multiplier"] = WizardConstants.MPS_PERFORMANCE_MULTIPLIER

        # Apple Silicon specific optimizations
        device_info["unified_memory"] = True
        device_info["memory_bandwidth"] = "High"  # 68-400+ GB/s depending on chip
        device_info["thermal_design"] = "Integrated SoC"

    elif device.type == "cuda":
        device_info["display_name"] = f"NVIDIA GPU ({device})"
        device_info["performance_multiplier"] = WizardConstants.CUDA_PERFORMANCE_MULTIPLIER

        # NVIDIA CUDA specific optimizations
        device_info["unified_memory"] = False
        device_info["memory_bandwidth"] = "Very High"  # 500-900+ GB/s for high-end cards
        device_info["thermal_design"] = "Discrete GPU"

    else:
        device_info["display_name"] = f"CPU ({device})"
        device_info["performance_multiplier"] = WizardConstants.CPU_PERFORMANCE_MULTIPLIER

        # CPU fallback characteristics
        device_info["unified_memory"] = True  # Shared with system
        device_info["memory_bandwidth"] = "Moderate"  # 50-100 GB/s typical
        device_info["thermal_design"] = "Traditional CPU"

    return device_info


def detect_datasets() -> List[Dict[str, Any]]:
    """Auto-detect available datasets under data/datasets plus curated sources.

    We intentionally scan only the immediate children of `data/datasets` to avoid
    treating the parent `data/` directory or the `datasets/` folder itself as a dataset.
    """
    datasets: List[Dict[str, Any]] = []

    # Prefer canonical layout: data/datasets/<name>
    # Anchored to project root so this works regardless of cwd.
    _project_root = Path(__file__).resolve().parent.parent.parent
    root = _project_root / "data" / "datasets"
    if root.exists():
        for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
            # Skip hidden and cache directories
            if subdir.name.startswith(".") or subdir.name in {".cache", "__pycache__"}:
                continue

            # Look for CSV files (common dataset format)
            csv_files = list(subdir.glob("*.csv"))
            if csv_files:
                datasets.append(
                    {
                        "name": subdir.name,
                        "type": "local_csv",
                        "path": str(subdir),
                        "files": len(csv_files),
                        "description": f"Local dataset with {len(csv_files)} CSV files",
                    }
                )

            # Look for audio files recursively inside this dataset folder
            audio_extensions = WizardConstants.AUDIO_EXTENSIONS
            audio_files: List[Path] = []
            for ext in audio_extensions:
                audio_files.extend(subdir.glob(f"**/{ext}"))
            if audio_files:
                datasets.append(
                    {
                        "name": subdir.name,
                        "type": "local_audio",
                        "path": str(subdir),
                        "files": len(audio_files),
                        "description": f"Local audio dataset with {len(audio_files)} files",
                    }
                )

    # Add BigQuery import option (virtual source)
    datasets.append(
        {
            "name": "Import from Google BigQuery",
            "type": "bigquery_import",
            "description": "Query BQ, export surgical slice to _prepared.csv",
        }
    )

    # Add Granary dataset setup option
    datasets.append(
        {
            "name": "Setup NVIDIA Granary Dataset",
            "type": "granary_setup",
            "description": "🚀 Large-scale multilingual dataset (~643k hours across 25 languages)",
        }
    )

    # Add common Hugging Face datasets
    hf_datasets = [
        {
            "name": "mozilla-foundation/common_voice_13_0",
            "type": "huggingface",
            "description": "Common Voice multilingual dataset",
        },
        {"name": "openslr/librispeech_asr", "type": "huggingface", "description": "LibriSpeech English ASR dataset"},
        {"name": "facebook/voxpopuli", "type": "huggingface", "description": "VoxPopuli multilingual dataset"},
    ]

    datasets.extend(hf_datasets)

    # Add custom dataset option
    datasets.append({"name": "custom", "type": "custom", "description": "I'll specify my dataset path manually"})

    # Ensure the BigQuery import option appears first in the wizard list
    # without changing the relative order of the remaining entries.
    bigquery_first: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for item in datasets:
        if item.get("type") == "bigquery_import":
            bigquery_first.append(item)
        else:
            others.append(item)
    return bigquery_first + others
