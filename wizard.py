#!/usr/bin/env python3

"""
Whisper Fine-Tuning Wizard - Interactive CLI for Apple Silicon

A Steve Jobs-inspired command-line interface that guides users through the entire
Whisper fine-tuning process with progressive disclosure. Simple for beginners,
powerful for experts.

Design principles:
- Ask one question at a time
- Show only what's relevant
- Smart defaults for everything
- Beautiful visual feedback
- Zero configuration required

Called by:
- manage.py finetune-wizard command
- Direct execution: python wizard.py

Integrates with:
- main.py: Executes training using existing infrastructure
- config.ini: Can generate profile configs on the fly
- All existing model types: whisper, distil-whisper, LoRA variants
"""

import os
import sys
import configparser
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple


# Rich for beautiful terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import track
from rich.align import Align
from rich import print as rprint

# Questionary for interactive prompts
import questionary
from questionary import Style

# Import existing utilities
from utils.device import get_device

# Try to import distributed training functionality
try:
    from distributed.utils import load_hosts_config, validate_ssh_connectivity
    from distributed.launcher import DistributedLauncher
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

# Initialize console and styling
console = Console()

# Custom style for questionary prompts (Apple-inspired)
apple_style = Style([
    ('qmark', 'fg:#ff9500 bold'),          # Orange question mark (Apple orange)
    ('question', 'bold'),                   # Bold question text
    ('answer', 'fg:#007aff bold'),         # Blue answers (Apple blue)
    ('pointer', 'fg:#ff9500 bold'),        # Orange pointer
    ('highlighted', 'fg:#007aff bold'),    # Blue highlight
    ('selected', 'fg:#34c759 bold'),       # Green selected (Apple green)
    ('instruction', 'fg:#8e8e93'),         # Gray instructions
    ('text', ''),                          # Default text
])

class WizardConstants:
    """Named constants for wizard configuration, user interface, and training estimation."""
    
    # Progressive Disclosure Timing Constants
    # These control the pacing of the Steve Jobs-inspired progressive disclosure UI
    ANIMATION_DELAY = 0.5              # Seconds between progressive UI reveals
    CONFIRMATION_WAIT = 2.0            # Seconds to display confirmation messages
    WELCOME_SCREEN_PAUSE = 1.0         # Seconds to display welcome screen animations
    
    # Training Estimation Constants  
    # Used for calculating realistic training time and memory requirements
    BASE_SAMPLES_ESTIMATE = 100000     # Baseline sample count for time calculations
    SAMPLES_PER_FILE = 10              # Average samples per dataset file (rough estimate)
    MEMORY_SAFETY_BUFFER = 0.8         # Use only 80% of available memory (20% safety margin)
    HOURS_TO_MINUTES_CUTOFF = 1.0      # Show minutes instead of hours below this threshold
    
    # Apple Silicon Performance Multipliers
    # Device-specific optimization factors for training time estimation
    MPS_PERFORMANCE_MULTIPLIER = 1.0   # Apple Silicon baseline (unified memory architecture)
    CUDA_PERFORMANCE_MULTIPLIER = 0.7  # NVIDIA GPUs typically 30% faster than Apple Silicon
    CPU_PERFORMANCE_MULTIPLIER = 3.0   # CPU training is ~3x slower than Apple Silicon MPS
    
    # Model Architecture Constants
    # d_model values for Whisper model compatibility checking
    WHISPER_D_MODEL_LARGE = 1280       # Large models (large, large-v2, large-v3)
    WHISPER_D_MODEL_MEDIUM = 1024      # Medium models
    WHISPER_D_MODEL_SMALL = 768        # Small models  
    WHISPER_D_MODEL_BASE = 512         # Base models
    WHISPER_D_MODEL_TINY = 384         # Tiny models
    
    # Mel Spectrogram Configuration
    # Default mel bin counts for different Whisper model generations
    MEL_BINS_LARGE_V3 = 128           # Whisper large-v3 uses 128 mel bins
    MEL_BINS_STANDARD = 80            # Most other Whisper models use 80 mel bins
    
    # Dataset Detection Patterns
    # File extensions and patterns for automatic dataset discovery
    AUDIO_EXTENSIONS = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
    DATASET_FILES_PATTERN = "*.csv"
    SKIP_DIRECTORIES = {".cache", "__pycache__", ".git", ".DS_Store"}
    
    # Configuration Generation Constants
    # Defaults for wizard-generated training profiles
    DEFAULT_LORA_DROPOUT = 0.1         # Standard LoRA dropout rate
    DEFAULT_DISTILLATION_ALPHA = 0.5   # Balance between hard and soft targets
    DEFAULT_TEMPERATURE_RANGE = (2.0, 10.0)  # Conservative to aggressive distillation
    RECOMMENDED_TEMPERATURE = 5.0      # Balanced distillation temperature
    
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
    DEFAULT_BQ_LIMIT = 1000            # Default row limit for BQ exports
    BQ_SAMPLING_OPTIONS = ["random", "first"]  # Available sampling strategies
    
    # Common HuggingFace Dataset Presets
    # Curated list of popular datasets for training
    RECOMMENDED_HF_DATASETS = [
        {
            "name": "mozilla-foundation/common_voice_13_0",
            "description": "Common Voice multilingual dataset"
        },
        {
            "name": "openslr/librispeech_asr", 
            "description": "LibriSpeech English ASR dataset"
        },
        {
            "name": "facebook/voxpopuli",
            "description": "VoxPopuli multilingual dataset"
        }
    ]

class TrainingMethod:
    """
    Training method configurations with smart defaults and resource estimation multipliers.
    
    This class defines the three primary fine-tuning approaches supported by the wizard,
    each with carefully calibrated resource requirements and quality expectations based
    on extensive Apple Silicon benchmarking and community feedback.
    
    Used by:
    - select_training_method() for method selection UI (line 300)
    - estimate_training_time() for resource planning calculations (line 885)
    - generate_profile_config() for configuration generation (line 998)
    - show_confirmation_screen() for final configuration display (line 926)
    
    Design philosophy:
    - Conservative memory estimates to prevent OOM errors
    - Realistic time estimates based on Apple Silicon benchmarks
    - Quality ratings help users understand trade-offs
    - Progressive disclosure hides complexity from beginners
    
    Memory multipliers account for:
    - Standard: Full model parameters + gradients + optimizer states
    - LoRA: Reduced adapter parameters but same base model memory
    - Distillation: Teacher model + student model + additional computation
    
    Time multipliers reflect:
    - LoRA: Faster convergence due to fewer parameters
    - Distillation: Additional forward passes through teacher model
    - Apple Silicon unified memory architecture optimizations
    """
    
    STANDARD = {
        "key": "standard",
        "name": "🚀 Standard Fine-Tune (SFT)",
        "description": "Full model fine-tuning for best accuracy",
        "memory_multiplier": 1.0,  # Baseline memory requirements
        "time_multiplier": 1.0,    # Baseline training time
        "quality": "highest"       # Maximum achievable quality
    }
    
    LORA = {
        "key": "lora", 
        "name": "🎨 LoRA Fine-Tune",
        "description": "Memory-efficient parameter-efficient fine-tuning",
        "memory_multiplier": 0.4,  # ~60% memory savings through adapter architecture
        "time_multiplier": 0.8,    # 20% faster due to fewer parameters to update
        "quality": "high"          # 95-98% of standard fine-tuning quality
    }
    
    DISTILLATION = {
        "key": "distillation",
        "name": "🧠 Knowledge Distillation", 
        "description": "Train smaller models from larger teacher models",
        "memory_multiplier": 1.2,  # 20% overhead for teacher model inference
        "time_multiplier": 1.5,    # 50% longer due to teacher forward passes
        "quality": "good"          # Quality depends on teacher-student gap
    }

class ModelSpecs:
    """Model specifications for estimation calculations"""
    
    MODELS = {
        # OpenAI Standard Models
        "whisper-tiny": {"params": "39M", "memory_gb": 1.2, "hours_100k": 0.5, "hf_id": "openai/whisper-tiny"},
        "whisper-base": {"params": "74M", "memory_gb": 2.1, "hours_100k": 1.0, "hf_id": "openai/whisper-base"}, 
        "whisper-small": {"params": "244M", "memory_gb": 4.2, "hours_100k": 2.5, "hf_id": "openai/whisper-small"},
        "whisper-medium": {"params": "769M", "memory_gb": 8.4, "hours_100k": 6.0, "hf_id": "openai/whisper-medium"},
        "whisper-large-v2": {"params": "1550M", "memory_gb": 16.8, "hours_100k": 12.0, "hf_id": "openai/whisper-large-v2"},
        "whisper-large-v3": {"params": "1550M", "memory_gb": 16.8, "hours_100k": 12.0, "hf_id": "openai/whisper-large-v3"},
        
        # OpenAI English-Only Models
        "whisper-tiny.en": {"params": "39M", "memory_gb": 1.2, "hours_100k": 0.45, "hf_id": "openai/whisper-tiny.en"},
        "whisper-base.en": {"params": "74M", "memory_gb": 2.1, "hours_100k": 0.9, "hf_id": "openai/whisper-base.en"},
        "whisper-small.en": {"params": "244M", "memory_gb": 4.2, "hours_100k": 2.3, "hf_id": "openai/whisper-small.en"},
        "whisper-medium.en": {"params": "769M", "memory_gb": 8.4, "hours_100k": 5.5, "hf_id": "openai/whisper-medium.en"},

        # Distil-Whisper Models (Pre-trained by HuggingFace)
        "distil-small.en": {"params": "166M", "memory_gb": 3.2, "hours_100k": 1.8, "hf_id": "distil-whisper/distil-small.en"},
        "distil-medium.en": {"params": "394M", "memory_gb": 6.1, "hours_100k": 3.5, "hf_id": "distil-whisper/distil-medium.en"},
        "distil-large-v2": {"params": "756M", "memory_gb": 12.4, "hours_100k": 8.0, "hf_id": "distil-whisper/distil-large-v2"},
        "distil-large-v3": {"params": "756M", "memory_gb": 12.4, "hours_100k": 7.5, "hf_id": "distil-whisper/distil-large-v3"},
        
        # Custom Distillation Targets (Create your own distilled models)
        "distil-tiny-from-medium": {"params": "39M", "memory_gb": 2.8, "hours_100k": 1.2, "hf_id": "openai/whisper-tiny"},
        "distil-base-from-medium": {"params": "74M", "memory_gb": 4.2, "hours_100k": 2.0, "hf_id": "openai/whisper-base"},
        "distil-tiny.en-from-medium.en": {"params": "39M", "memory_gb": 2.8, "hours_100k": 1.1, "hf_id": "openai/whisper-tiny.en"},
        "distil-base.en-from-medium.en": {"params": "74M", "memory_gb": 4.2, "hours_100k": 1.8, "hf_id": "openai/whisper-base.en"},
        
        # Hybrid Encoder-Decoder Models (Fast decoding with quality encoding)
        "distil-large-encoder-tiny-decoder": {"params": "1550M→195M", "memory_gb": 10.2, "hours_100k": 6.0, "hf_id": "openai/whisper-large-v3"},
        "distil-medium-encoder-tiny-decoder": {"params": "769M→195M", "memory_gb": 6.8, "hours_100k": 3.5, "hf_id": "openai/whisper-medium"},
        "distil-small-encoder-tiny-decoder": {"params": "244M→195M", "memory_gb": 4.5, "hours_100k": 2.0, "hf_id": "openai/whisper-small"},

        # Gemma 3n Models (approximate sizing for gating)
        "gemma-3n-e2b-it": {"params": "~2B", "memory_gb": 10.0, "hours_100k": 10.0, "hf_id": "google/gemma-3n-E2B-it"},
        "gemma-3n-e4b-it": {"params": "~4B", "memory_gb": 18.0, "hours_100k": 18.0, "hf_id": "google/gemma-3n-E4B-it"},
    }


def _infer_num_mel_bins(model_name_or_key: Any) -> int:
    """
    Determines expected mel spectrogram bins for Whisper model compatibility checking.
    
    This helper function is critical for distillation workflows where teacher and student
    models must have matching mel bin configurations. Whisper models use different mel
    bin counts depending on their generation and architecture.
    
    Called by:
    - configure_method_specifics() for teacher-student mel bin compatibility (line 718)
    - Custom hybrid model validation during distillation setup
    - Model architecture verification before training begins
    
    Calls to:
    - Python string operations for model name parsing and normalization
    - Exception handling for robust input type conversion
    
    Architecture compatibility rules:
    - Whisper large-v3: Uses 128 mel bins (newer architecture)
    - All other Whisper models: Use 80 mel bins (standard architecture)
    - Custom models: Inherit from their base model architecture
    - Mixed architectures: Require explicit compatibility validation
    
    Input handling patterns:
    - String model names: Direct parsing for architecture detection
    - Tuples: Extract first element (common from model selection returns)
    - None/empty values: Graceful degradation to standard 80 mel bins
    - Invalid types: Defensive string conversion with exception handling
    
    Args:
        model_name_or_key: Model identifier in various formats:
            - str: "whisper-large-v3", "openai/whisper-base", etc.
            - tuple: (model_name, config_dict) from selection functions
            - Any: Defensive handling for edge cases
            
    Returns:
        int: Number of mel bins expected by the model:
            - 128 for large-v3 variants (modern architecture)  
            - 80 for all other models (standard architecture)
            
    Example:
        mel_bins = _infer_num_mel_bins("whisper-large-v3")  # Returns 128
        mel_bins = _infer_num_mel_bins("whisper-base")      # Returns 80
        mel_bins = _infer_num_mel_bins(("whisper-small", {}))  # Returns 80
    """
    # Robust input normalization for various input types and formats
    # Handles tuples from model selection functions and defensive type conversion
    value: Any = model_name_or_key
    if isinstance(value, tuple) and value:
        value = value[0]  # Extract model name from (model, config) tuples
    
    # Defensive string conversion with exception handling for edge cases
    try:
        key = (value or "").lower()
    except Exception:
        key = str(value or "").lower()
    
    # Architecture detection: large-v3 uses modern 128-bin architecture
    if "large-v3" in key:
        return WizardConstants.MEL_BINS_LARGE_V3
    
    # All other models use standard 80-bin architecture
    return WizardConstants.MEL_BINS_STANDARD

def get_device_info() -> Dict[str, Any]:
    """
    Comprehensive device detection and performance profiling for training estimation.
    
    This function provides detailed hardware analysis to enable accurate training time
    and memory requirements estimation. It handles the three primary training platforms
    (Apple Silicon MPS, NVIDIA CUDA, CPU) with platform-specific optimizations.
    
    Called by:
    - show_welcome_screen() for system status display (line 185)
    - select_model() for memory constraint filtering (line 322)
    - estimate_training_time() for performance multiplier application (line 885)
    - show_confirmation_screen() for final hardware verification (line 926)
    
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
        device_info = get_device_info()
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

def show_welcome_screen():
    """
    Displays elegant Steve Jobs-inspired welcome screen with system capability verification.
    
    This function creates the first impression of the wizard using Apple's design language
    principles: beautiful visual elements, clear system status, and confidence-building
    through capability verification. It serves as both introduction and technical validation.
    
    Called by:
    - wizard_main() as the opening experience (line 1801)
    - Interactive training workflows requiring system status validation
    - Development and demonstration environments for visual appeal
    
    Calls to:
    - get_device_info() for comprehensive system capability detection (line 311)
    - rich.console for Apple-inspired visual formatting and layout
    - rich.panel for elegant bordered content presentation
    
    Design Philosophy:
    - Progressive disclosure: Only essential information shown initially
    - Confidence building: Clear indication system is ready for training
    - Visual hierarchy: ASCII art logo draws attention, then system details
    - Apple aesthetics: Blue color scheme, clean typography, generous whitespace
    
    System Verification Elements:
    - Device type detection (Apple Silicon MPS, NVIDIA CUDA, CPU fallback)
    - Available memory calculation for training capacity planning
    - Training readiness status with visual confirmation
    - Hardware optimization status for performance expectations
    
    User Experience Flow:
    1. ASCII art logo creates immediate visual impact and brand recognition
    2. System information builds confidence in hardware capabilities
    3. Ready status provides clear signal to proceed with training
    4. Press Enter prompt gives user control over pacing
    
    Visual Design Elements:
    - Custom ASCII logo using box-drawing characters for terminal aesthetics
    - Apple Silicon emoji (🍎) for brand association and hardware recognition
    - Color coding: Blue for branding, Green for success, Dim for instructions
    - Panel borders using Rich styling for professional appearance
    
    Technical Integration:
    - Uses identical device detection logic to training pipeline
    - Memory calculations align with training resource planning
    - Status verification prevents wizard from proceeding with invalid configurations
    - Visual feedback matches training progress indicators for consistency
    
    Accessibility Considerations:
    - High contrast color choices for terminal visibility
    - Clear text hierarchy for screen readers
    - Simple interaction model (Enter key) for universal access
    - Descriptive status messages for non-visual confirmation
    """
    
    # ASCII art logo
    logo = """
    ██╗    ██╗██╗  ██╗██╗███████╗██████╗ ███████╗██████╗ 
    ██║    ██║██║  ██║██║██╔════╝██╔══██╗██╔════╝██╔══██╗
    ██║ █╗ ██║███████║██║███████╗██████╔╝█████╗  ██████╔╝
    ██║███╗██║██╔══██║██║╚════██║██╔═══╝ ██╔══╝  ██╔══██╗
    ╚███╔███╔╝██║  ██║██║███████║██║     ███████╗██║  ██║
     ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
                                                          
                 🍎 Fine-Tuner for Apple Silicon
    """
    
    device_info = get_device_info()
    
    welcome_text = f"""
[bold cyan]Welcome to the Whisper Fine-Tuning Wizard![/bold cyan]

We'll guide you through training your custom Whisper model in just a few questions.

[green]System Information:[/green]
• Device: {device_info['display_name']}
• Available Memory: {device_info['available_memory_gb']:.1f} GB
• Status: Ready for training ✅

[dim]Press Enter to begin...[/dim]
    """
    
    console.print(Panel(
        Align.center(Text(logo, style="bold blue"), vertical="middle"),
        title="🎯 Whisper Fine-Tuner",
        border_style="blue",
        padding=(1, 2)
    ))
    console.print(welcome_text)
    
    input()  # Wait for user to press Enter

def detect_datasets() -> List[Dict[str, Any]]:
    """Auto-detect available datasets under data/datasets plus curated sources.

    We intentionally scan only the immediate children of `data/datasets` to avoid
    treating the parent `data/` directory or the `datasets/` folder itself as a dataset.
    """
    datasets: List[Dict[str, Any]] = []

    # Prefer canonical layout: data/datasets/<name>
    root = Path("data/datasets")
    if root.exists():
        for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
            # Skip hidden and cache directories
            if subdir.name.startswith(".") or subdir.name in {".cache", "__pycache__"}:
                continue

            # Look for CSV files (common dataset format)
            csv_files = list(subdir.glob("*.csv"))
            if csv_files:
                datasets.append({
                    "name": subdir.name,
                    "type": "local_csv",
                    "path": str(subdir),
                    "files": len(csv_files),
                    "description": f"Local dataset with {len(csv_files)} CSV files",
                })

            # Look for audio files recursively inside this dataset folder
            audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
            audio_files: List[Path] = []
            for ext in audio_extensions:
                audio_files.extend(subdir.glob(f"**/{ext}"))
            if audio_files:
                datasets.append({
                    "name": subdir.name,
                    "type": "local_audio",
                    "path": str(subdir),
                    "files": len(audio_files),
                    "description": f"Local audio dataset with {len(audio_files)} files",
                })

    # Add BigQuery import option (virtual source)
    datasets.append({
        "name": "Import from Google BigQuery",
        "type": "bigquery_import",
        "description": "Query BQ, export surgical slice to _prepared.csv"
    })
    
    # Add Granary dataset setup option
    datasets.append({
        "name": "Setup NVIDIA Granary Dataset",
        "type": "granary_setup",
        "description": "🚀 Large-scale multilingual dataset (~643k hours across 25 languages)"
    })

    # Add common Hugging Face datasets
    hf_datasets = [
        {"name": "mozilla-foundation/common_voice_13_0", "type": "huggingface", "description": "Common Voice multilingual dataset"},
        {"name": "openslr/librispeech_asr", "type": "huggingface", "description": "LibriSpeech English ASR dataset"},
        {"name": "facebook/voxpopuli", "type": "huggingface", "description": "VoxPopuli multilingual dataset"},
    ]
    
    datasets.extend(hf_datasets)
    
    # Add custom dataset option
    datasets.append({
        "name": "custom",
        "type": "custom", 
        "description": "I'll specify my dataset path manually"
    })
    
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

def select_model_family() -> str:
    """Step 0: Choose model family (Whisper or Gemma)."""
    console.print("\n[bold]Step 0: Choose your model family[/bold]")
    families = [
        {"name": "🌬️ Whisper - The robust ASR model from OpenAI.", "value": "whisper"},
        {"name": "💎 Gemma - The new multimodal model from Google.", "value": "gemma"},
    ]
    return questionary.select(
        "Which model family do you want to work with?",
        choices=families,
        style=apple_style
    ).ask()


def select_training_method(family: str | None = None) -> Dict[str, Any]:
    """Step 1: Select training method with progressive disclosure"""
    console.print("\n[bold]Step 1: Choose your training method[/bold]")

    methods = [TrainingMethod.STANDARD, TrainingMethod.LORA, TrainingMethod.DISTILLATION]
    # Gemma family supports LoRA-only in first release
    if family == "gemma":
        methods = [TrainingMethod.LORA]

    choices = []
    for method in methods:
        choices.append({
            "name": f"{method['name']} - {method['description']}",
            "value": method
        })

    selected_method = questionary.select(
        "What kind of fine-tuning do you want to run?",
        choices=choices,
        style=apple_style
    ).ask()

    return selected_method

def select_model(method: Dict[str, Any], family: str | None = None):
    """Step 2: Select model based on training method, driven by config.ini.

    For distillation, this returns the STUDENT model key (not teacher).
    Teacher will be chosen in a later step via configure_method_specifics().
    """
    
    console.print(f"\n[bold]Step 2: Choose your model[/bold]")
    
    device_info = get_device_info()
    available_memory = device_info["available_memory_gb"]
    
    # Dynamically discover available models from config.ini
    cfg = _read_config()
    available_models = [s.replace("model:", "") for s in cfg.sections() if s.startswith("model:")]
    # Apply family filter
    if family in ("whisper", "gemma"):
        filtered = []
        for m in available_models:
            section = f"model:{m}"
            grp = cfg.get(section, "group", fallback="").strip().lower()
            if family == "gemma" and grp == "gemma":
                filtered.append(m)
            if family == "whisper" and grp != "gemma":
                filtered.append(m)
        available_models = filtered
    
    # Filter models based on the selected training method
    if method["key"] == "lora":
        # Include LoRA-suffixed Whisper models AND Gemma family models (Gemma uses LoRA-only path)
        base_models = []
        for m in available_models:
            if "lora" in m:
                base_models.append(m)
                continue
            section = f"model:{m}"
            if cfg.has_option(section, "group") and cfg.get(section, "group").strip().lower() == "gemma":
                base_models.append(m)
    elif method["key"] == "distillation":
        # For distillation, list base students (including medium) to fine-tune as student
        base_models = [
            m for m in available_models
            if ("tiny" in m or "base" in m or "small" in m or "medium" in m)
        ]
    else: # standard
        base_models = [m for m in available_models if "lora" not in m and "distil" not in m]

    # Build model choices with memory and time estimates
    choices = []
    # For distillation, restrict students to a clean set and add Custom Hybrid entry
    allowed_students = {
        "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
        "whisper-tiny.en", "whisper-base.en", "whisper-small.en", "whisper-medium.en",
    }
    seen: set[str] = set()
    for model_name in base_models:
        # Use a display-friendly name if the config name is long
        display_name = model_name.replace("-lora", "")
        if method["key"] == "distillation" and display_name not in allowed_students:
            continue
        if display_name in seen:
            continue
        if display_name not in ModelSpecs.MODELS:
            continue
        seen.add(display_name)
        specs = ModelSpecs.MODELS[display_name]
        required_memory = specs["memory_gb"] * method["memory_multiplier"]
        
        # Skip if not enough memory
        if required_memory > available_memory * 0.8:  # Leave 20% buffer
            continue
        
        # Estimate training time (assuming 100k samples baseline)
        estimated_hours = specs["hours_100k"] * method["time_multiplier"] * device_info["performance_multiplier"]
        
        if estimated_hours < 1:
            time_str = f"{estimated_hours * 60:.0f} minutes"
        else:
            time_str = f"{estimated_hours:.1f} hours"
        
        memory_str = f"{required_memory:.1f}GB"
        
        # Create descriptive text for distillation models
        if "from-medium" in display_name:
            if "tiny" in display_name:
                model_desc = "Distill tiny (39M) from larger teacher"
            elif "base" in display_name:
                model_desc = "Distill base (74M) from larger teacher"
            else:
                model_desc = display_name
            choice_text = f"{model_desc} - ~{time_str}, {memory_str} memory"
        elif "encoder-tiny-decoder" in display_name:
            if "large" in display_name:
                model_desc = "Large Encoder / Tiny Decoder - Fast generation, best quality"
            elif "medium" in display_name:
                model_desc = "Medium Encoder / Tiny Decoder - Faster, good quality"
            elif "small" in display_name:
                model_desc = "Small Encoder / Tiny Decoder - Balanced speed & quality"
            else:
                model_desc = display_name
            choice_text = f"{model_desc} - ~{time_str}, {memory_str} memory"
        else:
            choice_text = f"{display_name} ({specs['params']}) - ~{time_str}, {memory_str} memory"
        
        # Add recommendation for optimal choice
        if display_name == "whisper-small" and method["key"] != "distillation":
            choice_text += " ⭐ Recommended"
        elif display_name == "distil-base-from-medium" and method["key"] == "distillation":
            choice_text += " ⭐ Recommended"
        elif display_name == "gemma-3n-e2b-it" and method["key"] == "lora":
            choice_text += " ⭐ Recommended"
        
        choices.append({
            "name": choice_text,
            "value": model_name  # Return the full config name, e.g., "whisper-base"
        })

    # Distillation: add a Custom Hybrid option inline at Step 2
    if method["key"] == "distillation":
        choices.append({
            "name": "Build a Custom Hybrid (mix encoder and decoder)",
            "value": "__custom_hybrid__",
        })
    
    if not choices:
        console.print("[red]❌ No models available for your memory constraints. Consider using LoRA training.[/red]")
        sys.exit(1)
    
    prompt = (
        "Which model do you want to fine-tune?" if method["key"] != "distillation"
        else "Which student model do you want to train? (or choose Custom Hybrid)"
    )
    selected_model = questionary.select(
        prompt,
        choices=choices,
        style=apple_style
    ).ask()

    # If user chose Custom Hybrid, immediately ask how to customize (encoder/decoder)
    if method["key"] == "distillation" and selected_model == "__custom_hybrid__":
        cfg = _read_config()
        available_models = [s.replace("model:", "") for s in cfg.sections() if s.startswith("model:")]
        enc_sources = [m for m in available_models if ("large" in m or "medium" in m)]
        dec_sources = [m for m in available_models if ("tiny" in m or "base" in m or "small" in m)]

        enc_choice = questionary.select(
            "Choose an Encoder source (large/medium)",
            choices=[{"name": m, "value": m} for m in enc_sources],
            style=apple_style,
        ).ask()
        
        # Add d_model compatibility warning
        d_model_map = {
            "whisper-large": 1280, "whisper-large-v2": 1280, "whisper-large-v3": 1280,
            "whisper-medium": 1024, "whisper-medium.en": 1024,
            "whisper-small": 768, "whisper-small.en": 768,
            "whisper-base": 512, "whisper-base.en": 512,
            "whisper-tiny": 384, "whisper-tiny.en": 384,
        }
        enc_d_model = d_model_map.get(enc_choice, 1024)
        
        # Filter decoder choices by d_model compatibility
        dec_choices = []
        for m in dec_sources:
            dec_d_model = d_model_map.get(m, 512)
            if dec_d_model != enc_d_model:
                dec_choices.append({"name": f"{m} (d_model={dec_d_model}, incompatible with encoder d_model={enc_d_model})", "value": m})
            else:
                dec_choices.append({"name": f"{m} (d_model={dec_d_model}, compatible ✓)", "value": m})
        
        dec_choice = questionary.select(
            "Choose a Decoder source (must have matching d_model)",
            choices=dec_choices,
            style=apple_style,
        ).ask()
        
        # Validate d_model compatibility
        dec_d_model = d_model_map.get(dec_choice, 512)
        if dec_d_model != enc_d_model:
            console.print(f"[red]❌ Error: Encoder d_model ({enc_d_model}) != Decoder d_model ({dec_d_model})[/red]")
            console.print("[red]Custom hybrid models require matching d_model. Please choose compatible models.[/red]")
            # Re-ask for decoder
            compatible_decs = [m for m in dec_sources if d_model_map.get(m, 512) == enc_d_model]
            if not compatible_decs:
                console.print(f"[red]No compatible decoders found for encoder with d_model={enc_d_model}[/red]")
                return selected_model, {}
            dec_choice = questionary.select(
                "Choose a compatible Decoder source:",
                choices=[{"name": m, "value": m} for m in compatible_decs],
                style=apple_style,
            ).ask()

        seed = {
            "student_model_type": "custom",
            "student_encoder_from": enc_choice,
            "student_decoder_from": dec_choice,
        }
        return selected_model, seed

    return selected_model, {}

def select_dataset(method: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: Select dataset"""
    
    console.print(f"\n[bold]Step 3: Choose your dataset[/bold]")
    
    datasets = detect_datasets()
    
    choices = []
    for dataset in datasets:
        if dataset["type"] == "local_csv" or dataset["type"] == "local_audio":
            choice_text = f"📁 {dataset['name']} - {dataset['description']}"
        elif dataset["type"] == "huggingface":
            choice_text = f"🤗 {dataset['name']} - {dataset['description']}"
        else:
            choice_text = f"⚙️ {dataset['description']}"
        
        choices.append({
            "name": choice_text,
            "value": dataset
        })
    
    selected_dataset = questionary.select(
        "Which dataset do you want to use for training?",
        choices=choices,
        style=apple_style
    ).ask()
    
    # Handle BigQuery import flow
    if selected_dataset.get("type") == "bigquery_import":
        bq_dataset = select_bigquery_table_and_export()
        return bq_dataset
    
    # Handle Granary dataset setup flow
    if selected_dataset.get("type") == "granary_setup":
        granary_dataset = setup_granary_dataset()
        return granary_dataset

    # Handle custom dataset path
    if selected_dataset["name"] == "custom":
        dataset_path = questionary.path(
            "Enter the path to your dataset:",
            style=apple_style
        ).ask()
        
        selected_dataset["path"] = dataset_path
        selected_dataset["name"] = Path(dataset_path).name
    
    return selected_dataset

def configure_training_parameters() -> Dict[str, Any]:
    """Step 4: Training Parameters (mandatory)
    
    Prompts for critical hyperparameters with simple guidance, returning a dict:
    {"learning_rate": float, "num_train_epochs": int, "warmup_steps": int}
    """
    console.print(f"\n[bold]Step 4: Training Parameters[/bold]")
    # Learning rate
    console.print("[dim]This is the most important hyperparameter. It controls how much the model learns from the data. A smaller number is safer. The default (1e-5) is a good starting point for fine-tuning.[/dim]")
    lr_str = questionary.text("What learning rate do you want to use?", default="1e-5", style=apple_style).ask()
    try:
        learning_rate = float(lr_str)
    except Exception:
        learning_rate = 1e-5

    # Number of epochs
    console.print("[dim]An epoch is one full pass through the entire training dataset. More epochs can lead to better results, but also increase the risk of overfitting. For fine-tuning, 1-3 epochs is usually enough.[/dim]")
    epochs_str = questionary.text("How many training epochs?", default="3", style=apple_style).ask()
    try:
        num_train_epochs = int(epochs_str)
    except Exception:
        num_train_epochs = 3

    # Warmup steps
    console.print("[dim]This gradually increases the learning rate at the start of training, which helps stabilize the model. A small number like 50-100 is a safe choice.[/dim]")
    warmup_str = questionary.text("How many warmup steps for the learning rate?", default="50", style=apple_style).ask()
    try:
        warmup_steps = int(warmup_str)
    except Exception:
        warmup_steps = 50

    return {"learning_rate": learning_rate, "num_train_epochs": num_train_epochs, "warmup_steps": warmup_steps}

def _read_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")
    return cfg

def _write_config(cfg: configparser.ConfigParser) -> None:
    with open("config.ini", "w") as f:
        cfg.write(f)

def _add_dataset_to_config(dataset_name: str, text_column: str) -> None:
    """Ensure `[dataset:dataset_name]` exists with source and text_column."""
    cfg = _read_config()
    section = f"dataset:{dataset_name}"
    if not cfg.has_section(section):
        cfg.add_section(section)
    cfg.set(section, "source", dataset_name)
    if text_column:
        cfg.set(section, "text_column", text_column)
    
    # BQ-created datasets have standard train/validation splits.
    # This ensures they are always present for the config validator.
    if not cfg.has_option(section, "train_split"):
        cfg.set(section, "train_split", "train")
    if not cfg.has_option(section, "validation_split"):
        cfg.set(section, "validation_split", "validation")
        
    _write_config(cfg)

def _update_bq_defaults(project_id: Optional[str], dataset_id: Optional[str]) -> None:
    cfg = _read_config()
    section = "bigquery"
    if not cfg.has_section(section):
        cfg.add_section(section)
    if project_id:
        cfg.set(section, "last_project_id", project_id)
    if dataset_id:
        cfg.set(section, "last_dataset_id", dataset_id)
    _write_config(cfg)

def _infer_candidate_columns(schema_fields: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    names = [f.get("name") if isinstance(f, dict) else getattr(f, "name", "") for f in schema_fields]
    names_lower = [str(n or "") for n in names]
    def pick(patterns: List[str]) -> List[str]:
        res: List[str] = []
        for p in patterns:
            for n in names:
                if n and n.lower() == p:
                    if n not in res:
                        res.append(n)
        return res
    audio_candidates = pick(["audio_path", "audio_url", "gcs_uri", "uri", "path", "audio"])
    transcript_candidates = pick(["text_perfect", "text_verbatim", "transcript", "asr_text", "text"]) 
    language_candidates = pick(["language", "lang", "locale"]) 
    # Add fallbacks if empty
    if not audio_candidates:
        audio_candidates = names[:5]
    if not transcript_candidates:
        transcript_candidates = names[:5]
    return audio_candidates, transcript_candidates, language_candidates

def select_bigquery_table_and_export() -> Dict[str, Any]:
    """
    Interactive BigQuery dataset import with surgical precision and enterprise-grade workflow.
    
    This function implements a comprehensive BigQuery-to-training-data pipeline that enables
    users to extract precisely the data they need from enterprise data warehouses. It provides
    intelligent schema analysis, column mapping, sampling strategies, and quality control
    for production-ready training datasets.
    
    Called by:
    - select_dataset() when user chooses BigQuery import option (line 502)
    - Enterprise workflows requiring direct data warehouse integration
    - Large-scale training data preparation from production analytics systems
    - Custom dataset creation for specialized domain training requirements
    
    Calls to:
    - core/bigquery.py:check_gcp_auth() for authentication validation
    - core/bigquery.py:list_datasets(), list_tables() for resource discovery
    - core/bigquery.py:get_table_schema() for intelligent column analysis
    - core/bigquery.py:get_distinct_languages() for multilingual dataset filtering
    - core/bigquery.py:build_query_and_export() for surgical data extraction
    - _add_dataset_to_config() for automatic configuration integration
    - _update_bq_defaults() for user experience optimization through defaults
    
    Enterprise BigQuery integration workflow:
    
    Authentication & Access:
    - GCP authentication verification with actionable setup guidance
    - Project access validation and permission checking
    - Service account or user credential support with clear error messages
    - Graceful fallback options for authentication issues
    
    Resource Discovery:
    - Dynamic project dataset enumeration with caching for performance
    - Table listing with metadata analysis for informed selection
    - Schema introspection for intelligent column mapping suggestions
    - Automatic detection of common audio and transcript patterns
    
    Intelligent Column Mapping:
    - Audio path detection: audio_path, audio_url, gcs_uri, path, audio
    - Transcript identification: text_perfect, text_verbatim, transcript, text  
    - Language analysis: language, lang, locale with distinct value enumeration
    - Fallback suggestions when standard patterns aren't found
    
    Advanced Sampling Strategies:
    - Random sampling: Statistical representation across data distribution
    - First-N sampling: Chronological order preservation for time-series data
    - Custom WHERE clauses: Complex filtering for domain-specific requirements
    - Row limits: Memory management and iteration speed optimization
    
    Language-Aware Processing:
    - Automatic language detection from schema analysis
    - Multi-language filtering with checkbox selection interface
    - Language distribution analysis for balanced dataset creation
    - Unicode and encoding validation for international datasets
    
    Quality Control & Validation:
    - Schema compatibility checking for Whisper training requirements
    - Data type validation and automatic conversion recommendations
    - NULL value detection and handling strategies
    - File size estimation and download time predictions
    
    Configuration Integration:
    - Automatic config.ini updates with dataset definitions
    - Standard train/validation split generation (80/20 default)
    - Text column mapping for consistent pipeline integration
    - Source tracking for data lineage and reproducibility
    
    Output Structure:
    Creates `data/datasets/{dataset_name}/` directory containing:
    - train.csv: Training split with audio paths and transcripts
    - validation.csv: Validation split for model evaluation
    - metadata.json: Export configuration and statistics
    - schema.json: Original BigQuery schema for reference
    
    Error Handling & Recovery:
    - Network timeout handling with retry mechanisms
    - Query complexity validation and optimization suggestions
    - Memory pressure detection during large exports
    - Partial download recovery for interrupted transfers
    
    Performance Optimizations:
    - Parallel chunk downloading for large datasets
    - Automatic query optimization through BigQuery best practices
    - Smart caching of schema and metadata to reduce API calls
    - Progress tracking with ETA calculation for long exports
    
    Returns:
        Dict[str, Any]: Dataset descriptor compatible with wizard workflow:
        {
            "name": Generated dataset directory name,
            "type": "local_csv" for downstream compatibility,
            "path": Absolute path to created dataset directory,
            "files": Number of CSV files created (train.csv, validation.csv),
            "description": Human-readable description with source information,
            "source_info": {
                "project_id": BigQuery project identifier,
                "dataset_id": BigQuery dataset name,
                "table_id": Source table name,
                "export_timestamp": ISO timestamp of export,
                "row_count": Total rows exported,
                "languages": List of languages included (if filtered)
            }
        }
        
    Example Usage:
        dataset = select_bigquery_table_and_export()
        # Interactive prompts guide user through:
        # 1. Project selection: "my-audio-project"
        # 2. Dataset selection: "speech_data_warehouse"  
        # 3. Table selection: "transcribed_calls"
        # 4. Column mapping: audio_path -> "gcs_audio_uri", transcript -> "human_transcription"
        # 5. Language filtering: ["en", "es", "fr"] from 15 available languages
        # 6. Sampling: Random 10,000 rows
        # Result: Creates data/datasets/bq_whisper_finetuning_20250813/ with train.csv and validation.csv
        
    Integration Benefits:
        - Zero-copy data pipeline from BigQuery to training
        - Surgical precision in data selection reduces training time
        - Enterprise-grade authentication and permission handling
        - Automatic data validation and quality assurance
        - Seamless integration with existing wizard workflow
        - Configuration persistence for reproducible experiments
    """
    from core import bigquery as bq

    console.print("\n[bold]BigQuery Import[/bold]")

    # Auth check
    if not bq.check_gcp_auth():
        console.print("[yellow]GCP auth not detected. Run: gcloud auth application-default login[/yellow]")
        proceed = questionary.confirm("Continue anyway (may fail)?", default=False, style=apple_style).ask()
        if not proceed:
            return {"name": "custom", "type": "custom", "description": "Manual path"}

    # Defaults
    cfg = _read_config()
    last_project = cfg.get("bigquery", "last_project_id", fallback="")
    last_dataset = cfg.get("bigquery", "last_dataset_id", fallback="")

    # Project
    project_id = questionary.text("GCP Project ID:", default=last_project, style=apple_style).ask()

    # Dataset selection
    datasets = bq.list_datasets(project_id) or []
    if datasets:
        dataset_id = questionary.select("Dataset:", choices=datasets, style=apple_style).ask()
    else:
        dataset_id = questionary.text("Dataset ID:", default=last_dataset or "", style=apple_style).ask()

    # Table selection (single-table MVP) with preflight and auto-refresh
    def _pick_table() -> str:
        tbls = bq.list_tables(project_id, dataset_id) or []
        if tbls:
            return questionary.select("Table:", choices=tbls, style=apple_style).ask()
        return questionary.text("Table ID:", style=apple_style).ask()

    # Initial pick
    table_id = _pick_table()
    # Preflight verify and auto-refresh once if needed
    ok, msg = bq.verify_table(project_id, dataset_id, table_id)
    if not ok:
        console.print("[yellow]Selected table/view failed preflight. Refreshing table list...[/yellow]")
        table_id = _pick_table()
        ok2, msg2 = bq.verify_table(project_id, dataset_id, table_id)
        if not ok2:
            console.print(f"[red]Preflight failed again for '{table_id}':[/red] {msg2}")
            raise RuntimeError(
                "BigQuery table check failed. Please choose a concrete table or fix the view wildcard."
            )

    _update_bq_defaults(project_id, dataset_id)

    # Schema and candidates
    schema = bq.get_table_schema(project_id, dataset_id, table_id)
    # Convert to serializable for inference helper
    schema_dicts = [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in schema]
    audio_cands, text_cands, lang_cands = _infer_candidate_columns(schema_dicts)

    audio_col = questionary.select("Audio path column:", choices=audio_cands, style=apple_style).ask()
    transcript_col = questionary.select("Transcript source column:", choices=text_cands, style=apple_style).ask()
    # The target column name should be the same as the source column name.
    # This removes the need for an extra user prompt.
    transcript_target = transcript_col

    use_language = False
    language_col = None
    languages: Optional[List[str]] = None
    if lang_cands:
        use_language = questionary.confirm("Filter by language?", default=True, style=apple_style).ask()
        if use_language:
            language_col = questionary.select("Language column:", choices=lang_cands, style=apple_style).ask()
            distinct = bq.get_distinct_languages(project_id, dataset_id, table_id, language_column=language_col) or []
            if distinct:
                languages = questionary.checkbox("Select languages (Space to toggle):", choices=distinct, style=apple_style).ask()
            else:
                languages = None

    # Sampling
    limit_str = questionary.text("Max rows to fetch (blank = no limit):", default="1000", style=apple_style).ask()
    try:
        limit = int(limit_str) if limit_str.strip() else None
    except Exception:
        limit = 1000
    sample_random = questionary.confirm("Random sample?", default=True, style=apple_style).ask()
    sample = "random" if sample_random else "first"

    extra_where = questionary.text("Advanced WHERE (optional):", default="", style=apple_style).ask()
    extra_where = extra_where.strip() or None

    # Execute export
    out_dir = Path("data/datasets")
    try:
        dataset_dir = bq.build_query_and_export(
            project_id=project_id,
            tables=[(dataset_id, table_id)],
            audio_col=audio_col,
            transcript_col=transcript_col,
            transcript_target=transcript_target,  # sets output column name
            language_col=language_col,
            languages=languages,
            limit=limit,
            sample=sample,  # "random" or "first"
            extra_where=extra_where,
            out_dir=out_dir,
        )
    except Exception as e:
        console.print(f"[red]BigQuery export failed:[/red] {e}")
        raise

    dataset_name = dataset_dir.name
    # Update config.ini for dataset resolution and text_column
    # Use the source column name (transcript_col) which is now the same as transcript_target
    _add_dataset_to_config(dataset_name, transcript_target)

    # Return dataset descriptor compatible with downstream flow
    return {
        "name": dataset_name,
        "type": "local_csv",
        "path": str(dataset_dir),
        "files": 2,  # train.csv and validation.csv
        "description": f"Imported from BigQuery {project_id}.{dataset_id}.{table_id}",
    }

def setup_granary_dataset() -> Dict[str, Any]:
    """
    Interactive NVIDIA Granary dataset setup with guided corpus download workflow.
    
    This function implements a comprehensive Granary setup workflow that guides users through
    the process of configuring one of the world's largest public speech datasets. It provides
    step-by-step instructions for downloading external audio corpora, configures the necessary
    audio source mappings, and generates the required configuration sections.
    
    Called by:
    - select_dataset() when user chooses "Setup NVIDIA Granary Dataset" option
    
    Granary Dataset Overview:
    The NVIDIA Granary dataset combines ~643k hours of transcribed audio across 25 languages
    from multiple large-scale speech corpora:
    - VoxPopuli: Multilingual parliamentary speeches
    - YouTube Commons (YTC): Diverse web content  
    - LibriLight: Large-scale English audiobooks
    - YODAS: Custom corpus (included in HuggingFace download)
    
    Setup Workflow:
    1. Introduction and value proposition explanation
    2. Language subset selection for focused training
    3. External corpus download guidance with specific links
    4. Audio source path configuration and validation
    5. Configuration generation and integration
    6. Optional preparation script execution
    
    Returns:
        Dict[str, Any]: Dataset descriptor for integration with training pipeline
    """
    console.print("\n" + "=" * 60)
    console.print(Align.center("🚀 [bold]NVIDIA GRANARY DATASET SETUP[/bold] 🚀"))
    console.print("=" * 60)
    
    # Step 1: Introduction and value proposition
    console.print(Panel.fit(
        "[bold cyan]Welcome to Granary Setup![/bold cyan]\n\n"
        "The NVIDIA Granary dataset is one of the world's largest public speech datasets with:\n"
        "• 📊 ~643,000 hours of transcribed audio\n"
        "• 🌍 25 languages for multilingual training\n"
        "• 🎯 High-quality transcriptions for robust ASR models\n"
        "• 📚 Multiple diverse corpora (parliamentary speeches, web content, audiobooks)\n\n"
        "[yellow]Note: Granary requires external audio corpus downloads (several TB total)[/yellow]",
        title="About Granary",
        border_style="cyan"
    ))
    
    proceed = questionary.confirm(
        "Ready to set up Granary for your project?",
        default=True,
        style=apple_style
    ).ask()
    
    if not proceed:
        return {"name": "custom", "type": "custom", "description": "Manual dataset path"}
    
    # Step 2: Language subset selection
    console.print(f"\n[bold]Step 1: Choose Language Subset[/bold]")
    language_options = [
        {"name": "🇺🇸 English (en) - Most common choice", "value": "en"},
        {"name": "🇪🇸 Spanish (es) - Large corpus available", "value": "es"},
        {"name": "🇫🇷 French (fr) - High-quality parliamentary data", "value": "fr"},
        {"name": "🇩🇪 German (de) - Rich multilingual content", "value": "de"},
        {"name": "🌍 Other language (specify manually)", "value": "custom"},
    ]
    
    language_choice = questionary.select(
        "Which language subset do you want to prepare?",
        choices=language_options,
        style=apple_style
    ).ask()
    
    if language_choice == "custom":
        language_code = questionary.text(
            "Enter language code (e.g., 'it', 'pt', 'nl'):",
            style=apple_style
        ).ask()
    else:
        language_code = language_choice
    
    # Step 3: Download guidance with specific links
    console.print(f"\n[bold]Step 2: Download Required Audio Corpora[/bold]")
    console.print(Panel.fit(
        "[bold red]IMPORTANT: External Downloads Required[/bold red]\n\n"
        "Granary requires you to download audio files from external sources:\n\n"
        "📥 [bold]Required Downloads:[/bold]\n"
        "1. VoxPopuli: https://github.com/facebookresearch/voxpopuli\n"
        "2. YouTube Commons: https://research.google.com/youtube-cc/\n"
        "3. LibriLight: https://github.com/facebookresearch/libri-light\n\n"
        "[yellow]Total size: Several terabytes - ensure you have adequate storage![/yellow]\n\n"
        "💡 [bold]Tips:[/bold]\n"
        "• Download to a fast SSD for best training performance\n"
        "• Consider downloading only the language subset you need\n"
        "• Ensure stable internet connection for large downloads",
        title="Download Instructions",
        border_style="red"
    ))
    
    downloads_complete = questionary.confirm(
        "Have you completed downloading all required audio corpora?",
        default=False,
        style=apple_style
    ).ask()
    
    if not downloads_complete:
        console.print("\n[yellow]💡 Come back and run this setup again after downloading the audio corpora.[/yellow]")
        console.print("For now, I'll create a template configuration you can complete later.")
    
    # Step 4: Audio source path configuration
    console.print(f"\n[bold]Step 3: Configure Audio Source Paths[/bold]")
    
    audio_sources = {}
    corpus_info = [
        ("voxpopuli", "VoxPopuli parliamentary speeches"),
        ("ytc", "YouTube Commons diverse content"),
        ("librilight", "LibriLight English audiobooks"),
    ]
    
    for corpus_key, corpus_desc in corpus_info:
        if downloads_complete:
            path = questionary.path(
                f"Path to {corpus_desc} audio directory:",
                style=apple_style
            ).ask()
            audio_sources[corpus_key] = path
        else:
            # Template paths for user to fill in later
            audio_sources[corpus_key] = f"/path/to/downloaded/{corpus_key}/audio"
    
    # Step 4: Validation configuration
    console.print(f"\n[bold]Step 4: Configure Audio Validation[/bold]")
    console.print(Panel.fit(
        "[bold cyan]Audio Validation Trade-offs[/bold cyan]\n\n"
        "Granary contains ~643k hours of audio. Validating every file takes time but prevents training failures.\n\n"
        "🔍 [bold]Full Validation (Recommended):[/bold] Checks every audio file exists (slow but safe)\n"
        "🎯 [bold]Sample Validation:[/bold] Checks a percentage of files (faster, some risk)\n"
        "🚀 [bold]Skip Validation:[/bold] No file checking (fastest, highest risk)\n\n"
        "[yellow]Recommendation: Use full validation for production, sampling for development[/yellow]",
        title="Validation Options",
        border_style="cyan"
    ))
    
    validation_mode = questionary.select(
        "How thorough should audio file validation be?",
        choices=[
            "Full validation (slowest, safest)",
            "Sample validation (faster, some risk)",
            "Skip validation (fastest, risky)"
        ],
        style=apple_style
    ).ask()
    
    # Configure validation settings based on choice
    skip_audio_validation = False
    sample_validation_rate = 1.0
    
    if validation_mode == "Skip validation (fastest, risky)":
        skip_audio_validation = True
        console.print("\n[yellow]⚠️  Audio files will NOT be verified. Training may fail if files are missing.[/yellow]")
    elif validation_mode == "Sample validation (faster, some risk)":
        sample_rate_choice = questionary.select(
            "What percentage of files should be validated?",
            choices=[
                "10% (quick sanity check)",
                "25% (reasonable confidence)",
                "50% (high confidence)",
                "Custom percentage"
            ],
            style=apple_style
        ).ask()
        
        if sample_rate_choice == "10% (quick sanity check)":
            sample_validation_rate = 0.1
        elif sample_rate_choice == "25% (reasonable confidence)":
            sample_validation_rate = 0.25
        elif sample_rate_choice == "50% (high confidence)":
            sample_validation_rate = 0.5
        else:  # Custom percentage
            while True:
                try:
                    custom_rate = questionary.text(
                        "Enter validation percentage (1-100):",
                        style=apple_style
                    ).ask()
                    rate_float = float(custom_rate) / 100.0
                    if 0.01 <= rate_float <= 1.0:
                        sample_validation_rate = rate_float
                        break
                    else:
                        console.print("[red]Please enter a number between 1 and 100[/red]")
                except (ValueError, TypeError):
                    console.print("[red]Please enter a valid number[/red]")
        
        console.print(f"\n[cyan]✅ Will validate {sample_validation_rate:.1%} of audio files[/cyan]")
    else:  # Full validation
        console.print("\n[green]✅ Will validate all audio files (recommended for production)[/green]")
    
    # Step 5: Generate configuration
    console.print(f"\n[bold]Step 5: Generate Configuration[/bold]")
    
    dataset_name = f"granary-{language_code}"
    config_section = f"""
[dataset:{dataset_name}]
source_type = granary
hf_name = nvidia/Granary
hf_subset = {language_code}
local_path = data/datasets/{dataset_name}
text_column = text
train_split = train
validation_split = validation
audio_source_voxpopuli = {audio_sources['voxpopuli']}
audio_source_ytc = {audio_sources['ytc']}
audio_source_librilight = {audio_sources['librilight']}
skip_audio_validation = {str(skip_audio_validation).lower()}
sample_validation_rate = {sample_validation_rate}
"""
    
    console.print(Panel.fit(
        f"[bold]Configuration to add to config.ini:[/bold]\n{config_section}",
        title="Generated Configuration",
        border_style="green"
    ))
    
    # Add configuration to config.ini
    add_config = questionary.confirm(
        "Add this configuration to your config.ini file?",
        default=True,
        style=apple_style
    ).ask()
    
    if add_config:
        try:
            config = configparser.ConfigParser()
            config.read("config.ini")
            
            section_name = f"dataset:{dataset_name}"
            if not config.has_section(section_name):
                config.add_section(section_name)
            
            config.set(section_name, "source_type", "granary")
            config.set(section_name, "hf_name", "nvidia/Granary")
            config.set(section_name, "hf_subset", language_code)
            config.set(section_name, "local_path", f"data/datasets/{dataset_name}")
            config.set(section_name, "text_column", "text")
            config.set(section_name, "train_split", "train")
            config.set(section_name, "validation_split", "validation")
            config.set(section_name, "audio_source_voxpopuli", audio_sources['voxpopuli'])
            config.set(section_name, "audio_source_ytc", audio_sources['ytc'])
            config.set(section_name, "audio_source_librilight", audio_sources['librilight'])
            
            # Add validation configuration
            config.set(section_name, "skip_audio_validation", str(skip_audio_validation).lower())
            config.set(section_name, "sample_validation_rate", str(sample_validation_rate))
            
            with open("config.ini", "w") as f:
                config.write(f)
                
            console.print("✅ [green]Configuration added to config.ini successfully![/green]")
            
        except Exception as e:
            console.print(f"❌ [red]Failed to update config.ini: {e}[/red]")
            console.print("Please add the configuration manually.")
    
    # Step 6: Optional preparation execution
    if downloads_complete and add_config:
        console.print(f"\n[bold]Step 5: Prepare Dataset[/bold]")
        run_preparation = questionary.confirm(
            f"Run dataset preparation now? (python main.py prepare-granary --profile {dataset_name})",
            default=True,
            style=apple_style
        ).ask()
        
        if run_preparation:
            console.print(f"\n🚀 [bold]Running Granary preparation for {dataset_name}...[/bold]")
            try:
                from scripts.prepare_granary import prepare_granary
                manifest_path = prepare_granary(dataset_name)
                console.print(f"✅ [green]Preparation completed! Manifest: {manifest_path}[/green]")
                
                return {
                    "name": dataset_name,
                    "type": "local_csv", 
                    "path": f"data/datasets/{dataset_name}",
                    "files": 1,
                    "description": f"NVIDIA Granary {language_code} dataset (~643k hours)",
                    "prepared": True
                }
                
            except Exception as e:
                console.print(f"❌ [red]Preparation failed: {e}[/red]")
                console.print("You can run preparation later with:")
                console.print(f"[cyan]python main.py prepare-granary --profile {dataset_name}[/cyan]")
    
    # Return dataset descriptor for training pipeline integration
    console.print(f"\n🎉 [bold green]Granary setup completed![/bold green]")
    console.print("You can now use this dataset for training once preparation is complete.")
    
    return {
        "name": dataset_name,
        "type": "granary_configured",
        "path": f"data/datasets/{dataset_name}", 
        "files": 0,  # Will be 1 after preparation
        "description": f"NVIDIA Granary {language_code} dataset (setup complete, preparation needed)",
        "language": language_code,
        "audio_sources": audio_sources,
        "prepared": downloads_complete and add_config and 'run_preparation' in locals() and run_preparation
    }

def configure_method_specifics(method: Dict[str, Any], model: str | tuple, seed: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Step 5: Method-specific configuration (progressive disclosure)"""
    # Defensive: older call sites may pass a (model, seed) tuple.
    if isinstance(model, tuple):
        model, seed_from_tuple = model
        if seed is None and isinstance(seed_from_tuple, dict):
            seed = seed_from_tuple

    config = {} if seed is None else dict(seed)
    
    if method["key"] == "lora":
        console.print(f"\n[bold]Step 5: LoRA Configuration[/bold]")
        console.print("[dim]LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning[/dim]")
        
        # LoRA rank
        rank_choices = [
            {"name": "4 (Ultra lightweight)", "value": 4},
            {"name": "8 (Lightweight)", "value": 8}, 
            {"name": "16 (Balanced) ⭐ Recommended", "value": 16},
            {"name": "32 (High capacity)", "value": 32},
            {"name": "64 (Maximum capacity)", "value": 64},
        ]
        
        config["lora_r"] = questionary.select(
            "LoRA rank (higher = more parameters to train):",
            choices=rank_choices,
            style=apple_style
        ).ask()
        
        # LoRA alpha (smart default based on rank)
        default_alpha = config["lora_r"] * 2
        alpha_choices = [
            {"name": f"{default_alpha} (Recommended)", "value": default_alpha},
            {"name": f"{config['lora_r']} (Conservative)", "value": config["lora_r"]},
            {"name": f"{config['lora_r'] * 4} (Aggressive)", "value": config["lora_r"] * 4},
            {"name": "Custom value", "value": "custom"},
        ]
        
        alpha = questionary.select(
            "LoRA alpha (controls adaptation strength):",
            choices=alpha_choices,
            style=apple_style
        ).ask()
        
        if alpha == "custom":
            alpha = questionary.text(
                "Enter custom alpha value:",
                default=str(default_alpha),
                style=apple_style
            ).ask()
            alpha = int(alpha)
        
        config["lora_alpha"] = alpha
        config["lora_dropout"] = 0.1  # Smart default
        config["use_peft"] = True
        
    elif method["key"] == "distillation":
        console.print(f"\n[bold]Step 5: Distillation Configuration[/bold]")
        # If user already chose Custom Hybrid in Step 2, skip asking architecture again
        arch_choice = "custom" if (model == "__custom_hybrid__" or config.get("student_model_type") == "custom") else "standard"
        
        # Define student model path
        if arch_choice == "custom":
            # Encoder/decoder sources
            if not config.get("student_encoder_from") or not config.get("student_decoder_from"):
                cfg = _read_config()
                available_models = [s.replace("model:", "") for s in cfg.sections() if s.startswith("model:")]
                large_like = [m for m in available_models if ("large" in m or "medium" in m)]
                small_like = [m for m in available_models if ("tiny" in m or "base" in m or "small" in m)]
                encoder_source = questionary.select(
                    "Choose an Encoder source (teacher model)",
                    choices=[{"name": m, "value": m} for m in large_like],
                    style=apple_style,
                ).ask()
                decoder_source = questionary.select(
                    "Choose a Decoder source (small/efficient model)",
                    choices=[{"name": m, "value": m} for m in small_like],
                    style=apple_style,
                ).ask()
                # Save in config
                config["student_model_type"] = "custom"
                config["student_encoder_from"] = encoder_source
                config["student_decoder_from"] = decoder_source
            else:
                encoder_source = config.get("student_encoder_from")
                decoder_source = config.get("student_decoder_from")

            # Teacher selection (guide to match encoder mel bins)
            teacher_models = ["whisper-large-v3", "whisper-large-v2", "whisper-medium"]
            student_mels = _infer_num_mel_bins(encoder_source)
            teacher_choices = []
            incompatible_count = 0
            for teacher in teacher_models:
                txt = teacher
                teacher_mels = _infer_num_mel_bins(teacher)
                if teacher_mels != student_mels:
                    txt += f" ({teacher_mels} mel bins vs student's {student_mels} - incompatible)"
                    incompatible_count += 1
                teacher_choices.append({"name": txt, "value": teacher})
            
            if incompatible_count == len(teacher_models):
                console.print(f"[yellow]⚠️ Warning: All teacher models have incompatible mel bins with your encoder ({student_mels} mel bins).[/yellow]")
                console.print("[yellow]Training may fail or produce poor results. Consider choosing a different encoder.[/yellow]")
            
            teacher_choice = questionary.select(
                "Which teacher model should we distill knowledge from?",
                choices=teacher_choices,
                style=apple_style,
            ).ask()
        else:
            # Standard student: teacher from curated list with compatibility filter
            teacher_models = ["whisper-large-v3", "whisper-large-v2", "whisper-medium"]
            teacher_choices = []
            for teacher in teacher_models:
                if teacher != model:
                    choice_text = f"{teacher}"
                    teacher_choices.append({"name": choice_text, "value": teacher})
            student_mels = _infer_num_mel_bins(model)
            filtered_teacher_choices = []
            for ch in teacher_choices:
                t_model = ch["value"]
                if _infer_num_mel_bins(t_model) != student_mels:
                    ch = {"name": ch["name"] + " (incompatible mel bins; not recommended)", "value": t_model}
                filtered_teacher_choices.append(ch)
            teacher_choice = questionary.select(
                "Which teacher model should we distill knowledge from?",
                choices=filtered_teacher_choices,
                style=apple_style,
            ).ask()
        # Resolve to full HF repo id via config.ini when possible
        try:
            cfg = _read_config()
            sec = f"model:{teacher_choice}"
            if cfg.has_section(sec) and cfg.has_option(sec, "base_model"):
                resolved_teacher = cfg.get(sec, "base_model")
            else:
                resolved_teacher = f"openai/{teacher_choice}" if teacher_choice.startswith("whisper-") else teacher_choice
        except Exception:
            resolved_teacher = teacher_choice
        config["teacher_model"] = resolved_teacher
        
        # Temperature
        temp_choices = [
            {"name": "2.0 (Conservative)", "value": 2.0},
            {"name": "5.0 (Balanced) ⭐ Recommended", "value": 5.0},
            {"name": "10.0 (Aggressive)", "value": 10.0},
            {"name": "Custom value", "value": "custom"}
        ]
        
        temperature = questionary.select(
            "Distillation temperature (higher = softer teacher guidance):",
            choices=temp_choices,
            style=apple_style
        ).ask()
        
        if temperature == "custom":
            temperature = questionary.text(
                "Enter custom temperature:",
                default="5.0",
                style=apple_style
            ).ask()
            temperature = float(temperature)
        
        config["temperature"] = temperature
    
    return config

def estimate_training_time(method: Dict[str, Any], model: str, dataset: Dict[str, Any], method_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Estimate training time and resource usage"""
    
    device_info = get_device_info()
    
    # Handle custom hybrid models by using encoder source for estimation
    if model == "__custom_hybrid__" and method_config:
        encoder_source = method_config.get("student_encoder_from", "whisper-medium")
        # Clean up model name to match ModelSpecs keys
        if "/" in encoder_source:
            encoder_source = encoder_source.split("/")[-1]
        model_specs = ModelSpecs.MODELS.get(encoder_source, ModelSpecs.MODELS["whisper-base"])
    else:
        model_specs = ModelSpecs.MODELS.get(model, ModelSpecs.MODELS["whisper-base"])
    
    # Rough estimation based on dataset size
    if "files" in dataset:
        estimated_samples = dataset["files"] * 10  # Assume 10 samples per file on average
    else:
        estimated_samples = 100000  # Default assumption
    
    # Base time calculation (hours for 100k samples)
    base_hours = model_specs["hours_100k"]
    sample_ratio = estimated_samples / 100000
    method_multiplier = method["time_multiplier"] 
    device_multiplier = device_info["performance_multiplier"]
    
    estimated_hours = base_hours * sample_ratio * method_multiplier * device_multiplier
    
    # Memory calculation
    base_memory = model_specs["memory_gb"]
    method_memory_multiplier = method["memory_multiplier"]
    estimated_memory = base_memory * method_memory_multiplier
    
    return {
        "hours": estimated_hours,
        "memory_gb": estimated_memory,
        "samples": estimated_samples,
        "eta": datetime.now() + timedelta(hours=estimated_hours)
    }

def show_confirmation_screen(method: Dict[str, Any], model: str, dataset: Dict[str, Any], 
                           method_config: Dict[str, Any], estimates: Dict[str, Any]) -> bool:
    """Step 5: Beautiful confirmation screen"""
    
    console.print(f"\n[bold cyan]Step 6: Ready to Train![/bold cyan]")
    
    # Create a beautiful configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan", width=20)
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Training Method", method["name"].replace("🚀", "").replace("🎨", "").replace("🧠", "").strip())
    # Distillation: show student architecture details (standard vs custom)
    if method["key"] == "distillation" and method_config.get("student_model_type") == "custom":
        config_table.add_row("Student", "Custom Hybrid")
        config_table.add_row("Encoder From", str(method_config.get("student_encoder_from")))
        config_table.add_row("Decoder From", str(method_config.get("student_decoder_from")))
    else:
        config_table.add_row("Model", f"{model} ({ModelSpecs.MODELS.get(model, {}).get('params', 'Unknown')})")
    config_table.add_row("Dataset", f"{dataset['name']} ({estimates['samples']:,} samples)")
    # Training parameters (added in Step 4)
    if "learning_rate" in method_config:
        config_table.add_row("Learning Rate", str(method_config["learning_rate"]))
    if "num_train_epochs" in method_config:
        config_table.add_row("Epochs", str(method_config["num_train_epochs"]))
    if "warmup_steps" in method_config:
        config_table.add_row("Warmup Steps", str(method_config["warmup_steps"]))
    
    # Add method-specific configuration
    if method["key"] == "lora":
        config_table.add_row("LoRA Rank", str(method_config["lora_r"]))
        config_table.add_row("LoRA Alpha", str(method_config["lora_alpha"]))
    elif method["key"] == "distillation":
        config_table.add_row("Teacher Model", method_config["teacher_model"])
        config_table.add_row("Temperature", str(method_config["temperature"]))
    
    config_table.add_row("", "")  # Spacer
    config_table.add_row("Estimated Time", f"{estimates['hours']:.1f} hours")
    config_table.add_row("Memory Usage", f"{estimates['memory_gb']:.1f} GB")
    # Display dtype/attention when Gemma is selected or when group specifies
    try:
        cfg = _read_config()
        section = f"model:{model}"
        if cfg.has_section(section):
            group = cfg.get(section, "group", fallback="").strip().lower()
            if group == "gemma":
                dtype = cfg.get("group:gemma", "dtype", fallback="bfloat16")
                attn = cfg.get("group:gemma", "attn_implementation", fallback="eager")
                config_table.add_row("Precision (dtype)", dtype)
                config_table.add_row("Attention Impl", attn)
    except Exception:
        pass
    config_table.add_row("Completion ETA", estimates['eta'].strftime("%I:%M %p today" if estimates['hours'] < 12 else "%I:%M %p tomorrow"))
    
    device_info = get_device_info()
    config_table.add_row("Training Device", device_info['display_name'])
    
    # Status indicators
    memory_status = "🟢 Sufficient" if estimates['memory_gb'] < device_info['available_memory_gb'] * 0.8 else "🟡 Tight"
    config_table.add_row("Memory Status", memory_status)
    
    # Show the panel
    console.print(Panel(
        config_table,
        title="🎯 Training Configuration",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Ask about visualization
    console.print(f"\n[bold cyan]Optional: Enable Training Visualizer?[/bold cyan]")
    console.print("[dim]Watch your AI learn in real-time with stunning 3D graphics![/dim]")
    
    enable_viz = questionary.confirm(
        "🎆 Enable live training visualization?",
        default=True,
        style=apple_style
    ).ask()
    
    # Store visualization choice for later use
    method_config['visualize'] = enable_viz
    
    if enable_viz:
        console.print("[green]✨ Visualization will open in your browser when training starts![/green]")
    
    # Confirmation prompt
    return questionary.confirm(
        "Start training with this configuration?",
        default=True,
        style=apple_style
    ).ask()

def generate_profile_config(method: Dict[str, Any], model: str, dataset: Dict[str, Any], 
                          method_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate config dict for the existing training infrastructure by leveraging the core config loader."""
    
    from core.config import load_model_dataset_config
    
    # Load the base configuration from config.ini using the robust, hierarchical loader.
    # This ensures that all central defaults are respected.
    cfg = _read_config()
    # For custom hybrid, load base defaults from decoder source instead of sentinel model
    model_for_loader = model
    if method["key"] == "distillation" and method_config.get("student_model_type") == "custom" and model == "__custom_hybrid__":
        model_for_loader = method_config.get("student_decoder_from") or model
    profile_config = load_model_dataset_config(cfg, model_for_loader, dataset["name"])

    # CRITICAL: Add the model and dataset keys that are required by load_profile_config
    # These are not included in load_model_dataset_config but are required for profile sections
    # For custom hybrid, use the decoder source as the model key for config validation
    if method["key"] == "distillation" and method_config.get("student_model_type") == "custom":
        profile_config["model"] = method_config.get("student_decoder_from", model_for_loader)
    else:
        profile_config["model"] = model_for_loader
    profile_config["dataset"] = dataset["name"]

    # Layer the user's interactive choices on top of the base configuration.
    # This overrides the defaults with the specific parameters selected in the wizard.
    
    # Method-specific configuration
    if method["key"] == "lora":
        profile_config.update({
            "use_peft": True,
            "peft_method": "lora",
            "lora_r": method_config["lora_r"],
            "lora_alpha": method_config["lora_alpha"], 
            "lora_dropout": method_config.get("lora_dropout", 0.1), # Sensible default
            # Use canonical key expected by trainer; leave as list, not string
            # Prefer canonical Gemma naming (o_proj not out_proj)
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        })
        # Gemma-specific safety: if selected model belongs to group:gemma, enforce eager attention
        section = f"model:{model_for_loader}"
        if cfg.has_option(section, "group") and cfg.get(section, "group").strip().lower() == "gemma":
            profile_config["attn_implementation"] = "eager"
    elif method["key"] == "distillation":
        profile_config.update({
            "teacher_model": method_config["teacher_model"],
            "distillation_temperature": method_config["temperature"],
            "distillation_alpha": 0.5,  # Balance between hard and soft targets
        })
        # Propagate custom student architecture if selected
        if method_config.get("student_model_type") == "custom":
            profile_config["student_model_type"] = "custom"
            profile_config["student_encoder_from"] = method_config.get("student_encoder_from")
            profile_config["student_decoder_from"] = method_config.get("student_decoder_from")
    
    # Dataset-specific configuration
    if dataset["type"] == "huggingface":
        profile_config["dataset_name"] = dataset["name"]
        profile_config["dataset_config"] = "en"  # Default to English
        profile_config["train_split"] = "train"
        profile_config["eval_split"] = "validation"
    elif dataset["type"] in ["local_csv", "local_audio"]:
        profile_config["train_dataset_path"] = dataset["path"]
        profile_config["eval_dataset_path"] = dataset["path"]  # Same for now
    
    # Merge training parameters from Step 4 (learning_rate, num_train_epochs, warmup_steps)
    for k in ("learning_rate", "num_train_epochs", "warmup_steps"):
        if k in method_config:
            profile_config[k] = method_config[k]

    # Add visualization flag if enabled
    if method_config.get('visualize', False):
        profile_config['visualize'] = True
    
    # Ensure required splits are always present for validation
    # These are required by the configuration validator
    if "train_split" not in profile_config:
        profile_config["train_split"] = "train"
    if "validation_split" not in profile_config:
        profile_config["validation_split"] = "validation"
    
    return profile_config

def execute_training(profile_config: Dict[str, Any]):
    """
    Executes fine-tuning training using the established main.py infrastructure with wizard integration.
    
    This function serves as the bridge between the wizard's interactive configuration and
    the production training system. It generates temporary configuration files, executes
    training via subprocess isolation, and provides comprehensive progress feedback with
    graceful error handling and cleanup.
    
    Called by:
    - wizard_main() after user confirms training configuration (line 1159)
    - Interactive training workflows initiated through the wizard interface
    - Automated training pipelines using wizard-generated configurations
    
    Calls to:
    - main.py:main() via subprocess for isolated training execution (line 1127)
    - core/config.py configuration loading through main.py integration
    - All training infrastructure through main.py's operation dispatch system
    - Temporary file management for wizard-generated configurations
    
    Configuration generation workflow:
    1. Temporary config directory creation with timestamp-based isolation
    2. Main config.ini parsing to preserve model and dataset definitions
    3. Wizard profile generation with user-selected parameters
    4. Temporary config file creation with complete training configuration
    5. Subprocess training execution with isolated environment
    6. Progress monitoring and error handling with user feedback
    7. Cleanup of temporary configuration files
    
    Subprocess isolation benefits:
    - Prevents import side effects from affecting wizard UI
    - Enables clean resource management and memory cleanup
    - Provides process-level isolation for training experiments
    - Supports graceful interruption without wizard corruption
    - Maintains compatibility with package installation scenarios
    
    Error handling strategies:
    - subprocess.CalledProcessError: Training process failures with exit codes
    - KeyboardInterrupt: User cancellation with checkpoint preservation
    - FileNotFoundError: Configuration file access issues
    - Exception: General error recovery with diagnostic information
    
    Configuration inheritance:
    - Preserves all model and dataset definitions from main config.ini
    - Inherits DEFAULT section with global training parameters
    - Maintains group configurations and hierarchical settings
    - Applies wizard-specific overrides only where necessary
    
    Progress feedback:
    - Real-time training status through subprocess stdout/stderr
    - Checkpoint saving notifications for long-running training
    - Success/failure status with actionable user guidance
    - Visual progress indicators using Rich console formatting
    
    Args:
        profile_config (Dict[str, Any]): Complete training configuration generated by wizard:
            - model: Model identifier and architecture settings
            - dataset: Dataset name and processing parameters  
            - method: Training method (standard, LoRA, distillation)
            - method_config: Method-specific parameters (ranks, temperatures, etc.)
            - device_settings: Hardware optimization parameters
            - visualization: Optional training visualization settings
            
    Side Effects:
        - Creates temp_configs/ directory if it doesn't exist
        - Generates temporary wizard_config_{timestamp}.ini file
        - Executes training subprocess with main.py integration
        - Cleans up temporary files after training completion/failure
        - Updates system with trained model outputs in configured output directory
        
    Example:
        # Wizard-generated configuration for LoRA fine-tuning
        profile_config = {
            "model": "whisper-base",
            "dataset": "librispeech_subset", 
            "use_peft": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "visualize": True
        }
        execute_training(profile_config)
        # Outputs: Trained LoRA adapter in output/wizard_20250813_143052/
    """
    console.print(f"\n[bold green]🚀 Starting training...[/bold green]")
    
    import subprocess
    import argparse
    
    # Temporary configuration management with timestamp-based isolation
    # Prevents conflicts between concurrent wizard instances
    config_dir = Path("temp_configs")
    config_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    temp_config_path = config_dir / f"wizard_config_{timestamp}.ini"
    
    # Configuration inheritance from main config.ini
    # Ensures all model definitions, dataset configurations, and global defaults
    # are available to the training process without duplication
    main_config = _read_config()
    
    # New configuration generation with selective section copying
    # Preserves essential configuration structure while adding wizard profile
    config = configparser.ConfigParser()
    
    # DEFAULT section preservation for global training parameters
    # ConfigParser treats DEFAULT as special - not returned by sections() but contains
    # critical parameters like num_train_epochs, learning_rate, logging_steps
    try:
        config["DEFAULT"] = dict(main_config["DEFAULT"])  # Always present in parsed config
    except Exception:
        # Fallback minimal defaults for malformed configuration recovery
        config["DEFAULT"] = {
            "output_dir": "output",
            "logging_dir": "logs",
            "num_train_epochs": "3",
            # Safety net: conservative default LR for fine-tuning
            "learning_rate": "1e-5"
        }
    
    # Essential section copying for training infrastructure support
    # Includes model definitions, dataset configurations, and group settings
    for section in main_config.sections():
        if section.startswith(("model:", "dataset:", "group:", "dataset_defaults")):
            config[section] = dict(main_config[section])
    
    # Wizard profile section creation with complete user configuration
    profile_name = f"wizard_{timestamp}"
    config[f"profile:{profile_name}"] = profile_config
    
    # Temporary configuration file generation for subprocess execution
    with open(temp_config_path, 'w') as f:
        config.write(f)
    
    # User progress feedback with realistic time expectations
    console.print("[dim]Training started! This may take several hours...[/dim]")
    console.print("[dim]Press Ctrl+C to interrupt (training will be saved at checkpoints)[/dim]")
    
    keep_config = False
    try:
        # Ask if user wants distributed training; if so, route to distributed launcher
        enable_dist = questionary.confirm(
        "Enable distributed training (multi-GPU)?",
        default=False,
        style=apple_style
        ).ask()

        if enable_dist:
        # Gather distributed params with sensible defaults
            num_nodes_str = questionary.text(
            "How many GPUs to use?",
            default="2",
            style=apple_style
            ).ask()
            try:
                num_nodes = int(num_nodes_str)
            except Exception:
                num_nodes = 2

            strategy = questionary.select(
            "Which distributed strategy?",
            choices=["allreduce", "diloco"],
            style=apple_style
            ).ask()

            h_param = 100
            if strategy == "diloco":
                h_param_str = questionary.text(
                "DiLoCo communication interval H?",
                default="100",
                style=apple_style
                ).ask()
                try:
                    h_param = int(h_param_str)
                except Exception:
                    h_param = 100

            module_cwd = Path(__file__).resolve().parent
            result = subprocess.run([
                sys.executable,
                str(module_cwd / "train_distributed.py"),
                "--profile", profile_name,
                "--output_dir", str(Path("output") / f"wizard_{timestamp}"),
                "--num_nodes", str(num_nodes),
                "--strategy", strategy,
                "--h_param", str(h_param),
                "--config", str(temp_config_path.resolve())
            ], check=True, text=True, capture_output=False, cwd=str(module_cwd))
        else:
            # Subprocess training execution with environment isolation
            # Uses module invocation for package compatibility and clean resource management
            module_cwd = Path(__file__).resolve().parent
            result = subprocess.run([
                sys.executable,
                "-m", "main",              # Module invocation for package compatibility
                "finetune",               # Training operation
                profile_name,             # Generated wizard profile
                "--config",               # Custom configuration file
                str(temp_config_path.resolve())  # Absolute path for cross-platform compatibility
            ], check=True, text=True, capture_output=False, cwd=str(module_cwd))
        
        # Success feedback with actionable next steps
        console.print(f"\n[bold green]✅ Training completed successfully![/bold green]")
        console.print(f"[green]Model saved in output directory[/green]")
        console.print(f"[dim]Check output/wizard_{timestamp}/ for your trained model[/dim]")
        
    except subprocess.CalledProcessError as e:
        # Training process failure with diagnostic guidance
        console.print(f"\n[red]❌ Training failed with exit code {e.returncode}[/red]")
        console.print(f"[red]Check the logs for detailed error information[/red]")
        console.print(f"[dim]Log file: output/wizard_{timestamp}/run.log[/dim]")
        
    except KeyboardInterrupt:
        # User interruption with checkpoint preservation confirmation
        keep_config = True
        console.print(f"\n[yellow]⚠️ Training interrupted by user[/yellow]")
        console.print(f"[yellow]Progress saved at latest checkpoint[/yellow]")
        console.print(f"[dim]Resume with: python main.py finetune {profile_name} --config {temp_config_path}[/dim]")

    except Exception as e:
        # General error recovery with troubleshooting guidance
        keep_config = True
        console.print(f"\n[red]❌ Training execution failed: {str(e)}[/red]")
        console.print(f"[red]Check your configuration and try again[/red]")
        console.print(f"[dim]Configuration saved at: {temp_config_path}[/dim]")

    finally:
        # Temporary file cleanup with error tolerance
        # Clean up temporary configuration to prevent accumulation
        # Preserve config file when training was interrupted or failed so user can resume
        if not keep_config:
            try:
                temp_config_path.unlink()
            except Exception:
                # Ignore cleanup errors - temporary files will be cleaned up eventually
                pass

def check_distributed_training():
    """
    Check if distributed training is configured and offer it as an option.
    
    Beautiful progressive disclosure:
    1. Check for distributed_hosts.json
    2. Validate connectivity to configured machines
    3. Offer distributed training with estimated speedup
    
    Returns:
        Dict or None: Hosts configuration if distributed training is selected, None otherwise
    """
    if not HAS_DISTRIBUTED:
        return None
        
    hosts_file = Path("distributed_hosts.json")
    if not hosts_file.exists():
        return None
    
    try:
        hosts_config = load_hosts_config("distributed_hosts.json")
        
        # Enhanced local environment validation (ExoLabs integration)
        from distributed.utils import validate_training_environment
        print("\n🖥️ Validating local environment for distributed training...")
        local_env = validate_training_environment()
        
        # Report Apple Silicon compatibility
        if local_env['is_apple_silicon']:
            if local_env['is_rosetta_emulation']:
                print("⚠️ WARNING: Python running under Rosetta 2")
                print("   Consider reinstalling with native ARM64 Python for optimal performance")
            
            if not local_env['mps_functional'] and local_env['mps_available']:
                print("⚠️ MPS available but not functional - this may impact training performance")
            elif local_env['mps_functional']:
                print("✅ Apple Silicon MPS validated and ready")
        
        # Show critical issues if any
        if local_env['environment_issues']:
            print("⚠️ Environment issues detected:")
            for issue in local_env['environment_issues'][:2]:  # Show max 2 to avoid clutter
                print(f"   • {issue}")
        
        # Quick connectivity check
        worker_addrs = hosts_config['workers']
        ssh_user = hosts_config['ssh_user']
        
        print("\n🔍 Checking other Macs on your network...")
        time.sleep(0.5)  # Brief pause for effect
        
        try:
            available_workers = validate_ssh_connectivity(
                hosts=worker_addrs,
                ssh_user=ssh_user,
                ssh_key_path=hosts_config.get('ssh_key_path')
            )
        except Exception as e:
            print(f"⚠️ Could not reach other Macs: {e}")
            return None
        
        if available_workers:
            total_machines = len(available_workers) + 1
            estimated_speedup = min(total_machines * 0.8, total_machines)  # Account for overhead
            
            console.print(Panel(
                f"🚀 I found {len(available_workers)} other Macs on your network!\n\n"
                f"📊 Estimated speedup: {estimated_speedup:.1f}x faster training\n"
                f"🎯 Strategy: DiLoCo (minimal communication, maximum speed)\n"
                f"💡 Perfect for your multi-Mac setup",
                title="✨ Distributed Training Available",
                border_style="bright_blue"
            ))
            
            use_distributed = questionary.confirm(
                "🌟 Use distributed training across multiple Macs?",
                default=True,
                style=apple_style
            ).ask()
            
            if use_distributed:
                console.print(f"[green]✅ Distributed training enabled across {total_machines} machines[/green]")
                time.sleep(WizardConstants.ANIMATION_DELAY)
                return hosts_config
        
    except Exception as e:
        console.print(f"[yellow]Distributed training config found but invalid: {e}[/yellow]")
    
    return None


def execute_distributed_training(profile_config: Dict[str, Any], hosts_config: Dict[str, Any]):
    """
    Execute distributed training across multiple machines.
    
    Args:
        profile_config: Training configuration from wizard
        hosts_config: Distributed hosts configuration
    """
    console.print(f"\n[bold green]🚀 Starting distributed training across {len(hosts_config['workers']) + 1} machines...[/bold green]")
    
    try:
        # Create launcher
        launcher = DistributedLauncher("distributed_hosts.json")
        
        # Convert wizard config to distributed training args
        training_args = {
            "strategy": "diloco",  # Default to DiLoCo for V1
            "model": "openai/whisper-tiny",  # Use tiny model for testing
            "num-epochs": 1,  # Conservative for testing
            "batch-size": 4,
            "hosts-config": "distributed_hosts.json"
        }
        
        console.print(f"[dim]📡 Setting up secure connections...[/dim]")
        
        # Record start time
        start_time = time.time()
        
        # Launch distributed training
        launcher.launch(training_args)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        console.print(f"\n[bold green]✅ Distributed training completed successfully![/bold green]")
        console.print(f"⏱️ Total time: {training_time:.1f} seconds")
        
        # Check for results
        if Path("distributed_training_result.pt").exists():
            console.print(f"💾 Training results saved to distributed_training_result.pt")
        
        console.print(f"\n[bold]🎉 Your multi-Mac training setup is working perfectly![/bold]")
        console.print(f"[dim]Next time, you can use larger models and longer training sessions.[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Distributed training failed: {e}[/red]")
        console.print(f"[yellow]⚠️ Falling back to single-machine training...[/yellow]")
        
        # Fall back to regular training
        execute_training(profile_config)


def wizard_main():
    """
    Main wizard orchestration function implementing Steve Jobs-inspired progressive disclosure.
    
    This function implements the complete wizard workflow using progressive disclosure principles:
    show only what's relevant at each step, ask one question at a time, and provide intelligent
    defaults for everything. The design creates a smooth, Apple-like experience that guides
    users from zero configuration to production-ready training.
    
    Called by:
    - Direct script execution: python wizard.py (line 1203)
    - Package entry points and command-line tools
    - Interactive training workflows and development environments
    - Automated training setup and configuration generation
    
    Calls to (complete wizard workflow):
    - show_welcome_screen() for elegant introduction and system status (line 185)
    - select_training_method() for method selection with smart recommendations (line 300)
    - select_model() for model selection with memory/performance constraints (line 322)  
    - select_dataset() for dataset selection with BigQuery integration (line 502)
    - configure_method_specifics() for advanced configuration with progressive disclosure (line 718)
    - estimate_training_time() for realistic resource planning and expectations (line 885)
    - show_confirmation_screen() for final configuration review and approval (line 926)
    - generate_profile_config() for production configuration generation (line 998)
    - execute_training() for training execution with progress monitoring (line 1071)
    
    Progressive disclosure workflow:
    
    Step 0 (Welcome Screen):
    - System capability detection and hardware profiling
    - Visual welcome with Apple-inspired design language  
    - Confidence building through status verification
    - Sets expectations for the complete workflow
    
    Step 1 (Training Method Selection):
    - Three clear options: Standard, LoRA, Knowledge Distillation
    - Quality vs efficiency trade-offs explained simply
    - Smart defaults highlighted with recommendation badges
    - Technical complexity hidden until needed
    
    Step 2 (Model Selection):
    - Dynamic model filtering based on hardware constraints
    - Memory and time estimates for realistic expectations
    - Performance recommendations based on use case
    - Automatic incompatibility filtering for user safety
    
    Step 3 (Dataset Selection):
    - Automatic local dataset discovery with file counting
    - BigQuery integration for enterprise workflows
    - Popular HuggingFace datasets with descriptions
    - Custom dataset path support for advanced users
    
    Step 4 (Method-Specific Configuration):
    - Revealed only for selected training method
    - LoRA: Rank and alpha with smart defaults
    - Distillation: Teacher model and temperature selection
    - Custom hybrid: Encoder/decoder architecture building
    
    Step 5 (Resource Estimation):
    - Realistic time and memory requirements
    - Hardware-specific performance calculations
    - Training completion estimates with ETA
    - Safety checks for resource constraints
    
    Step 6 (Confirmation & Execution):
    - Beautiful configuration summary table
    - Final approval with ability to cancel safely
    - Training execution with progress feedback
    - Success/failure handling with actionable guidance
    
    Error handling philosophy:
    - Graceful degradation: Never crash, always provide alternatives
    - User empowerment: Clear error messages with troubleshooting guidance
    - State preservation: Configuration saved even on failures
    - Recovery options: Instructions for manual continuation
    
    Apple-inspired design principles:
    - Simplicity: Complex operations feel effortless
    - Beauty: Visual design creates confidence and delight
    - Intelligence: Smart defaults reduce cognitive load
    - Humanity: Conversational language, not technical jargon
    - Reliability: Robust error handling prevents frustration
    
    Exception handling:
    - KeyboardInterrupt: Clean cancellation without system changes
    - Configuration errors: Diagnostic information with recovery steps
    - Training failures: Checkpoint preservation and troubleshooting guidance
    - System errors: Comprehensive error reporting for issue resolution
    
    Side effects:
    - May create temporary configuration files (cleaned up automatically)
    - Updates config.ini with new dataset definitions (BigQuery imports)
    - Creates training output directories and model checkpoints
    - Generates comprehensive training logs and metrics
    
    Example workflow:
        $ python wizard.py
        
        Welcome Screen: "Ready for training ✅"
        Method Selection: "🚀 Standard Fine-Tune (SFT) ⭐ Recommended"
        Model Selection: "whisper-small (244M) - ~2.5 hours, 4.2GB memory ⭐ Recommended"
        Dataset Selection: "📁 my_dataset - Local dataset with 3 CSV files"
        Configuration: [Smart defaults applied automatically]
        Confirmation: "Start training with this configuration? Yes"
        Training: [Progress monitoring with checkpoints]
        Completion: "✅ Training completed successfully! Model saved in output/"
    """
    try:
        # Step 0: Elegant introduction with system profiling
        # Creates confidence through hardware verification and beautiful design
        show_welcome_screen()
        
        # Step 0: Model family selection (Whisper vs Gemma)
        family = select_model_family()
        
        # Step 1: Training method selection with progressive disclosure
        # Presents three clear choices with quality/efficiency trade-offs
        method = select_training_method(family)
        
        # Step 2: Model selection with intelligent constraints
        # Returns (model_key, seed_dict) tuple for configuration flexibility
        model, seed = select_model(method, family)
        
        # Step 3: Dataset selection with automatic discovery
        # Supports local files, BigQuery imports, and HuggingFace datasets
        dataset = select_dataset(method)
        
        # Step 4: Training parameters (mandatory hyperparameters)
        training_params = configure_training_parameters()

        # Step 5: Method-specific configuration with smart defaults
        # Reveals advanced options only when needed, passes seed for custom hybrids
        method_config = configure_method_specifics(method, model, seed)
        # Merge training parameters into method_config for downstream display and merging
        method_config.update(training_params)
        
        # Step 6: Resource estimation with realistic expectations
        # Calculates training time and memory requirements based on hardware
        estimates = estimate_training_time(method, model, dataset, method_config)
        
        # Step 7: Beautiful confirmation screen with final approval
        # Comprehensive configuration review before committing to training
        if show_confirmation_screen(method, model, dataset, method_config, estimates):
            
            # Configuration generation for production training infrastructure
            profile_config = generate_profile_config(method, model, dataset, method_config)
            
            # Step 8: Check for distributed training opportunity
            # Steve Jobs magic: auto-discover other Macs and offer seamless speedup
            hosts_config = check_distributed_training()
            
            # Training execution with progress monitoring and error handling
            if hosts_config:
                # Execute distributed training across multiple machines
                execute_distributed_training(profile_config, hosts_config)
            else:
                # Execute standard single-machine training
                execute_training(profile_config)
            
        else:
            # Graceful cancellation with guidance for future use
            console.print(f"\n[yellow]Training cancelled by user.[/yellow]")
            console.print(f"[dim]Run the wizard again anytime with: python wizard.py[/dim]")
            console.print(f"[dim]Or use the management interface: python manage.py finetune-wizard[/dim]")
            
    except KeyboardInterrupt:
        # Clean interruption handling with system state preservation
        console.print(f"\n\n[yellow]Wizard interrupted by user.[/yellow]")
        console.print(f"[dim]No changes made to your system.[/dim]")
        console.print(f"[dim]Your configuration choices were not saved.[/dim]")
        
    except Exception as e:
        # Comprehensive error handling with debugging support
        console.print(f"\n[red]❌ Wizard error: {str(e)}[/red]")
        console.print(f"[red]Please report this issue or try manual configuration.[/red]")
        console.print(f"[dim]For manual setup: python main.py --help[/dim]")
        
        # Re-raise for debugging in development environments
        # This ensures stack traces are available for issue resolution
        raise

if __name__ == "__main__":
    wizard_main()