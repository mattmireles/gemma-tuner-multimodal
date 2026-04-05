#!/usr/bin/env python3
"""
System-wide Constants for Gemma Fine-Tuning

Centralizes all magic numbers, timing parameters, and configuration
constants used throughout the Gemma fine-tuning system.

Key categories:
- Memory management: Device-specific memory fractions and limits
- Training parameters: Default batch sizes, learning rates, epochs
- File system: Path constants, file extensions, directory structures
- Device optimization: Platform-specific performance parameters
- Timing parameters: Delays, timeouts, retry intervals

Usage pattern:
    from gemma_tuner.constants import MemoryLimits, DeviceOptimization

    # Instead of: set_memory_fraction(0.8)
    set_memory_fraction(MemoryLimits.MPS_DEFAULT_FRACTION)
"""

# ===== MEMORY MANAGEMENT CONSTANTS =====


class MemoryLimits:
    """
    Memory management constants for different compute platforms.

    These constants control memory usage across Apple Silicon (MPS), NVIDIA CUDA,
    and CPU platforms, accounting for architectural differences between unified
    and discrete memory systems.
    """

    # MPS (Apple Silicon) Memory Configuration
    # Apple Silicon uses unified memory architecture where CPU and GPU share
    # physical memory. These limits prevent system-wide memory pressure.
    MPS_DEFAULT_FRACTION = 0.8
    """
    Default memory fraction for MPS devices.
    
    Set to 0.8 (80%) to prevent system-wide memory swapping while maximizing
    GPU utilization. Apple Silicon's unified memory architecture means GPU
    memory pressure affects the entire system, not just GPU processes.
    
    Tuning guidelines:
    - Production inference: 0.8 (stable performance)
    - Development/debugging: 0.7 (safety margin for system tools)
    - Memory-constrained systems: 0.6 (multiple processes or limited RAM)
    - Aggressive training: 0.9 (risk of system instability)
    """

    MPS_CONSERVATIVE_FRACTION = 0.7
    """Conservative memory fraction for development and debugging."""

    MPS_AGGRESSIVE_FRACTION = 0.9
    """Aggressive memory fraction for maximum performance (risk of swapping)."""

    # CUDA (NVIDIA) Memory Configuration
    # CUDA uses discrete GPU memory separate from system RAM.
    CUDA_DEFAULT_FRACTION = 0.9
    """
    Default memory fraction for CUDA devices.
    
    Set to 0.9 (90%) to maximize GPU memory utilization. CUDA's discrete
    memory architecture means GPU out-of-memory errors don't affect system
    stability, allowing more aggressive memory usage.
    """

    CUDA_CONSERVATIVE_FRACTION = 0.8
    """Conservative CUDA memory fraction for multi-process scenarios."""


# ===== TRAINING CONFIGURATION CONSTANTS =====


class TrainingDefaults:
    """
    Default training parameters optimized for different hardware platforms.

    These constants provide reasonable starting points for training configuration,
    with platform-specific optimizations for Apple Silicon, NVIDIA CUDA, and CPU.
    """

    # Batch Size Configuration
    # Apple Silicon optimized batch sizes account for unified memory architecture
    MPS_BATCH_SIZE_SMALL = 4
    """Small batch size for MPS training (4 samples)."""

    MPS_BATCH_SIZE_MEDIUM = 8
    """Medium batch size for MPS training (8 samples)."""

    MPS_BATCH_SIZE_LARGE = 16
    """Large batch size for MPS training (16 samples)."""

    # CUDA batch sizes can be larger due to discrete GPU memory
    CUDA_BATCH_SIZE_SMALL = 8
    """Small batch size for CUDA training (8 samples)."""

    CUDA_BATCH_SIZE_MEDIUM = 16
    """Medium batch size for CUDA training (16 samples)."""

    CUDA_BATCH_SIZE_LARGE = 32
    """Large batch size for CUDA training (32 samples)."""

    # CPU batch sizes are limited by system memory and processing speed
    CPU_BATCH_SIZE_SMALL = 2
    """Small batch size for CPU training (2 samples)."""

    CPU_BATCH_SIZE_MEDIUM = 4
    """Medium batch size for CPU training (4 samples)."""

    # Learning Rate Configuration
    LEARNING_RATE_DEFAULT = 1e-5
    """
    Default learning rate for fine-tuning (1e-5).

    This learning rate works well for most fine-tuning scenarios,
    providing stable training without requiring extensive hyperparameter tuning.
    """

    LEARNING_RATE_AGGRESSIVE = 5e-5
    """Aggressive learning rate for faster convergence (5e-5)."""

    LEARNING_RATE_CONSERVATIVE = 5e-6
    """Conservative learning rate for sensitive datasets (5e-6)."""

    LEARNING_RATE_LORA = 1e-4
    """Default learning rate for LoRA fine-tuning (1e-4)."""

    # Training Duration
    EPOCHS_DEFAULT = 3
    """Default number of training epochs (3)."""

    EPOCHS_QUICK = 1
    """Single epoch for quick experimentation."""

    EPOCHS_THOROUGH = 5
    """Extended training for thorough fine-tuning."""

    # Training Steps Configuration
    WARMUP_STEPS_DEFAULT = 500
    """Default warmup steps for learning rate scheduling (500)."""

    WARMUP_STEPS_QUICK = 100
    """Quick warmup for shorter training runs (100)."""

    LOGGING_STEPS = 25
    """Frequency of training metrics logging (25 steps)."""

    SAVE_STEPS = 1000
    """Frequency of checkpoint saving (1000 steps)."""

    EVAL_STEPS = 500
    """Frequency of validation evaluation (500 steps)."""

    # LoRA Configuration
    LORA_RANK_DEFAULT = 32
    """Default LoRA rank for adapter layers (32)."""

    LORA_ALPHA_DEFAULT = 64
    """Default LoRA alpha scaling factor (64)."""

    LORA_DROPOUT_DEFAULT = 0.07
    """Default LoRA dropout rate (0.07)."""

    # Token Length Limits
    MAX_LABEL_LENGTH = 256
    """Maximum token length for labels (256 tokens)."""

    MAX_TARGET_LENGTH = 256
    """Maximum target sequence length (256 tokens)."""

    # Special Token Handling
    IGNORE_TOKEN_ID = -100
    """
    Token ID used to ignore positions in loss computation (-100).

    This is PyTorch's standard ignore index for cross-entropy loss.
    Used across all training backends (standard, distillation, LoRA, Gemma)
    to mask padding and non-target tokens during loss calculation.
    """

    # Model Configuration Defaults (cross-platform compatibility)
    DEFAULT_DTYPE = "float32"
    """
    Default tensor dtype string identifier for model loading.

    Float32 is the safest default across all platforms: Apple Silicon MPS
    has limited float64 support and some float16 edge cases, while float32
    works everywhere. Backend-specific overrides (e.g., float16 for CUDA,
    bfloat16 for Gemma) are handled in their respective modules.
    """

    DEFAULT_ATTENTION_IMPLEMENTATION = "eager"
    """
    Default attention implementation for model loading.

    'eager' is the baseline implementation compatible with all platforms
    (MPS, CUDA, CPU). Faster alternatives ('sdpa', 'flash_attention_2')
    are CUDA-only or have platform restrictions and should be set
    explicitly in profile configs when appropriate.
    """


# ===== FILE SYSTEM CONSTANTS =====


class FileSystem:
    """
    File system constants for consistent path handling and organization.

    These constants ensure consistent file organization across the entire
    fine-tuning system and provide clear documentation of directory structure.
    """

    # Directory Structure
    OUTPUT_DIR_DEFAULT = "output"
    """Default output directory for all training artifacts."""

    DATA_DIR = "data"
    """Base directory for dataset storage."""

    DATASETS_SUBDIR = "datasets"
    """Subdirectory within data for dataset files."""

    PATCHES_DIR = "data_patches"
    """Directory for data quality patches (overrides, blacklists)."""

    CACHE_SUBDIR = ".cache"
    """Subdirectory name for dataset caching."""

    # File Extensions and Patterns
    CSV_EXTENSION = ".csv"
    """Standard CSV file extension."""

    JSON_EXTENSION = ".json"
    """Standard JSON file extension."""

    LOCK_EXTENSION = ".lock"
    """File extension for lock files."""

    BLACKLIST_PREFIX = "blacklist-"
    """Prefix for blacklist files."""

    METADATA_FILENAME = "metadata.json"
    """Standard filename for run metadata."""

    COMPLETION_FILENAME = "completed"
    """Filename for run completion markers."""

    RUN_ID_FILENAME = "next_run_id.txt"
    """Filename for run ID tracking."""


# ===== TIMING AND RETRY CONSTANTS =====


class Timing:
    """
    Timing constants for delays, timeouts, and retry operations.

    These constants control temporal aspects of the system, including
    file operation retries, device initialization delays, and timeout values.
    """

    # File Operation Timeouts
    FILE_LOCK_TIMEOUT = 30.0
    """
    Timeout for file lock acquisition (30 seconds).
    
    Maximum time to wait for exclusive file access before giving up.
    Prevents indefinite blocking in concurrent execution scenarios.
    """

    FILE_OPERATION_RETRY_DELAY = 0.1
    """
    Delay between file operation retries (100ms).
    
    Brief delay to allow temporary file system issues to resolve
    before retrying failed operations.
    """

    # Device Initialization
    DEVICE_INITIALIZATION_DELAY = 1.0
    """
    Delay after device initialization (1 second).
    
    Some device operations benefit from a brief delay to ensure
    proper initialization, especially on Apple Silicon systems.
    """

    # Model Loading
    MODEL_LOADING_TIMEOUT = 300.0
    """
    Timeout for model loading operations (5 minutes).
    
    Maximum time to wait for large model loading before considering
    the operation failed. Accounts for slow storage or network loading.
    """


# ===== EVALUATION CONSTANTS =====


class Evaluation:
    """
    Constants for model evaluation and metrics calculation.

    These constants control evaluation behavior, metrics calculation,
    and output formatting for consistent evaluation across the system.
    """

    # Metrics Configuration
    WER_DECIMAL_PLACES = 4
    """
    Decimal places for WER reporting (4 places).
    
    Word Error Rate is reported with 4 decimal places for sufficient
    precision while maintaining readability in reports and logs.
    """

    # WER Thresholds for Quality Control
    WER_THRESHOLD_TRAINING = 75.0
    """
    Maximum acceptable WER for training samples (75%).
    
    Samples with WER above this threshold are considered too noisy
    for training and should be blacklisted or reviewed.
    """

    WER_THRESHOLD_VALIDATION = 80.0
    """
    Maximum acceptable WER for validation samples (80%).
    
    Slightly more permissive than training threshold to allow
    for dataset variability while still catching major issues.
    """

    # Sample Limiting
    EVAL_SAMPLE_LIMIT_DEBUG = 100
    """Sample limit for debug evaluation (100 samples)."""

    EVAL_SAMPLE_LIMIT_QUICK = 1000
    """Sample limit for quick evaluation (1000 samples)."""

    # Progress Reporting
    PROGRESS_UPDATE_FREQUENCY = 10
    """
    Progress update frequency during evaluation (10 samples).
    
    Controls how frequently progress updates are displayed during
    long evaluation runs to provide user feedback without overwhelming
    the output with too many updates.
    """

    # Evaluation Batch Size
    EVAL_BATCH_SIZE_DEFAULT = 16
    """Default batch size for evaluation (16 samples)."""


# ===== VISUALIZATION CONSTANTS =====


class VisualizationConstants:
    """
    Constants for training visualization and monitoring interface.

    These constants control the web-based visualization interface for
    real-time training monitoring and performance tracking.
    """

    # Web Server Configuration
    DEFAULT_PORT = 8080
    """Default port for visualization server (8080)."""

    # Buffer and Update Settings
    METRICS_BUFFER_SIZE = 1000
    """Maximum metrics buffer size (1000 entries)."""

    BATCH_BUFFER_SIZE = 10
    """Batch accumulation buffer size (10 batches)."""

    SMOOTHING_WINDOW = 100
    """Window size for metrics smoothing (100 samples)."""

    UPDATE_FREQUENCY = 10
    """UI update frequency in seconds (10s)."""

    # Performance Limits
    MAX_CONCURRENT_CONNECTIONS = 20
    """Maximum concurrent WebSocket connections (20)."""

    CONNECTION_TIMEOUT = 5
    """WebSocket connection timeout in seconds (5s)."""


# ===== LOGGING CONSTANTS =====


class LoggingDefaults:
    """Logging configuration constants. Delegates to core.logging.LoggingConstants
    for level values to avoid duplication."""

    DEFAULT_LEVEL = "INFO"
    RUN_LOG_FILENAME = "run.log"
    TRAINING_LOG_FILENAME = "training.log"


# ===== BACKWARDS COMPATIBILITY ALIASES =====

# Memory management aliases
DEFAULT_MPS_MEMORY_FRACTION = MemoryLimits.MPS_DEFAULT_FRACTION
DEFAULT_CUDA_MEMORY_FRACTION = MemoryLimits.CUDA_DEFAULT_FRACTION

# Training aliases
DEFAULT_LEARNING_RATE = TrainingDefaults.LEARNING_RATE_DEFAULT
DEFAULT_EPOCHS = TrainingDefaults.EPOCHS_DEFAULT

# File system aliases
DEFAULT_OUTPUT_DIR = FileSystem.OUTPUT_DIR_DEFAULT
METADATA_FILE = FileSystem.METADATA_FILENAME
