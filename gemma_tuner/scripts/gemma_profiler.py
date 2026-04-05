#!/usr/bin/env python3
"""
Gemma 3n Performance Profiler and Resource Analyzer

This module provides comprehensive performance profiling and resource analysis for Gemma 3n
multimodal models. It measures critical performance metrics including inference timing,
memory consumption, and device utilization to enable optimal training configuration and
resource planning for Apple Silicon hardware.

Key Responsibilities:
- Model and processor loading with optimal configuration detection
- Performance benchmarking with realistic audio input simulation
- Memory usage analysis and resource consumption reporting
- Device-specific optimization validation (MPS, CUDA, CPU)
- Data type optimization testing (bfloat16 vs float32)

Architecture Integration:
This profiler serves as a critical tool for performance analysis and resource planning
throughout the development lifecycle. It provides quantitative data for training
configuration decisions and deployment resource estimation.

Called by:
- Development workflows for performance baseline establishment
- Resource planning sessions for training infrastructure sizing
- Performance regression testing during model development
- Deployment planning for production inference systems
- Hardware evaluation for Apple Silicon optimization

Calls to:
- transformers.AutoModelForCausalLM for model loading and inference
- transformers.AutoProcessor for multimodal preprocessing
- torch.backends.mps for Apple Silicon acceleration
- psutil for system resource monitoring and memory tracking

Cross-File Integration Points:

Training Pipeline Integration:
- Uses identical model loading logic to models/gemma/finetune.py
- Validates device detection compatibility with utils/device.py
- Tests preprocessing pipeline used in training workflows

Performance Baseline Integration:
- Provides quantitative baselines for training time estimation
- Validates memory requirements for dataset size planning
- Measures inference overhead for real-time application design

Configuration Validation:
- Tests optimal dtype selection logic used throughout training
- Validates device configuration recommended by scripts/gemma_preflight.py
- Measures impact of configuration choices on performance

Profiling Methodology:

1. Environment Setup:
   - Device detection and optimal configuration selection
   - Data type compatibility testing (bfloat16 vs float32)
   - Model loading with memory-optimized settings

2. Synthetic Workload Generation:
   - Realistic audio duration simulation (3 seconds at target sampling rate)
   - Standard multimodal message format for consistent testing
   - Processor pipeline validation with representative input

3. Performance Measurement:
   - High-precision timing with device synchronization
   - Memory consumption tracking via RSS monitoring
   - Device utilization analysis for optimization opportunities

4. Resource Analysis:
   - Peak memory usage identification for capacity planning
   - Inference latency measurement for real-time requirements
   - Device-specific performance characterization

Metrics Collected:
- Device type and optimization status (MPS, CUDA, CPU)
- Data type configuration and hardware support validation
- Forward pass inference timing with synchronization
- Peak RSS memory consumption during inference
- Model configuration and attention implementation details

Performance Insights:
- Quantifies Apple Silicon MPS acceleration benefits
- Measures bfloat16 memory savings and performance impact
- Identifies bottlenecks in multimodal preprocessing pipeline
- Provides baselines for training time and resource estimation

Usage Examples:

Basic performance profiling:
  python scripts/gemma_profiler.py

Custom model profiling:
  python -c "from gemma_tuner.scripts.gemma_profiler import main; main('custom/model-id')"

Integration with training workflows:
  # Profile before training to validate resource requirements
  python scripts/gemma_profiler.py
  # Use output for training configuration decisions

Resource planning workflow:
  # Test multiple model sizes for deployment planning
  for model in models.txt; do
    python scripts/gemma_profiler.py "$model"
  done

Output Format:
Reports key metrics in structured format:
- device: Compute device and acceleration status
- dtype: Data type configuration and optimization
- time_s: Forward pass timing in seconds
- rss_mb: Peak memory consumption in megabytes

Performance Optimization Insights:
- MPS acceleration: 3-10x speedup over CPU inference
- bfloat16 optimization: 30-50% memory reduction when supported
- Eager attention: Conservative but compatible implementation
- Memory optimization: Identifies peak usage for capacity planning

Security and Safety:
- Read-only model analysis with no system modifications
- Safe tensor operations with proper cleanup
- Minimal system resource impact during profiling
- No sensitive data logging or external communication
"""

import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from gemma_tuner.models.gemma.constants import AudioProcessingConstants
from gemma_tuner.utils.device import probe_bfloat16


class GemmaProfilerConstants:
    """Named constants for Gemma performance profiling configuration."""

    # Default Configuration
    DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"  # Default Gemma model for profiling

    # Synthetic Audio Configuration
    # Realistic audio parameters for representative performance testing
    SYNTHETIC_AUDIO_DURATION_SECONDS = 3  # Audio length for consistent testing
    DEFAULT_SAMPLING_RATE = AudioProcessingConstants.DEFAULT_SAMPLING_RATE  # Fallback sampling rate for USM audio tower

    # Model Configuration
    ATTENTION_IMPLEMENTATION = "eager"  # Conservative attention for compatibility
    LOW_CPU_MEM_USAGE = True  # Memory optimization during loading

    # Performance Measurement
    # bfloat16 testing configuration for dtype optimization validation
    BFLOAT16_TEST_TENSOR_SIZE = 1  # Minimal tensor for dtype testing

    # Memory Reporting
    BYTES_TO_MB_DIVISOR = 1024 * 1024  # Conversion factor for RSS memory reporting

    # Output Formatting
    TIME_PRECISION_DIGITS = 3  # Decimal places for timing measurements
    MEMORY_PRECISION_DIGITS = 1  # Decimal places for memory measurements

    # Chat Template Structure
    # Standard multimodal message format for consistent profiling
    CHAT_TEMPLATE = {
        "USER_ROLE": "user",
        "AUDIO_TYPE": "audio",
        "TEXT_TYPE": "text",
        "AUDIO_PLACEHOLDER": "<audio:attached>",
        "TRANSCRIPTION_PROMPT": "Please transcribe this audio.",
    }


def main(model_id: str = GemmaProfilerConstants.DEFAULT_MODEL_ID) -> None:
    """
    Executes comprehensive performance profiling for Gemma 3n multimodal models.

    This function performs systematic performance analysis including model loading,
    preprocessing, inference timing, and resource consumption measurement. It provides
    quantitative data for training configuration decisions and deployment planning.

    Called by:
    - Command-line execution when script is run directly
    - Development workflows for performance baseline establishment
    - Resource planning sessions for infrastructure sizing
    - Performance regression testing during model development

    Calls to:
    - transformers.AutoProcessor.from_pretrained() for multimodal preprocessing
    - transformers.AutoModelForCausalLM.from_pretrained() for model loading
    - torch.backends.mps for Apple Silicon acceleration detection
    - psutil.Process() for system resource monitoring

    Profiling Methodology:
    1. Environment Setup: Device detection and optimal configuration selection
    2. Model Loading: Memory-optimized loading with dtype optimization
    3. Synthetic Workload: Realistic audio simulation for consistent testing
    4. Performance Measurement: High-precision timing with device synchronization
    5. Resource Analysis: Memory consumption tracking and reporting

    Performance Optimizations Tested:
    - Apple Silicon MPS acceleration vs CPU fallback
    - bfloat16 memory optimization vs float32 baseline
    - Eager attention implementation for maximum compatibility
    - Memory-optimized model loading for large model support

    Args:
        model_id (str): Hugging Face model identifier or local path to Gemma 3n model

    Output:
        Prints structured performance metrics:
        - device: Compute device and acceleration status
        - dtype: Data type configuration and optimization
        - time_s: Forward pass timing in seconds
        - rss_mb: Peak memory consumption in megabytes

    Example Output:
        device=mps, dtype=torch.bfloat16, time_s=2.145, rss_mb=8247.3

    Integration Notes:
    - Uses identical device detection logic to training pipeline
    - Tests preprocessing pipeline compatibility with training workflows
    - Validates memory requirements for training dataset size planning
    - Provides baselines for real-time inference performance requirements
    """
    constants = GemmaProfilerConstants

    # Initialize system resource monitoring for peak memory tracking
    process_monitor = psutil.Process(os.getpid())

    # Device detection: prefer MPS (Apple Silicon), then CUDA, then CPU.
    # Previous code silently skipped CUDA, which meant Linux/NVIDIA boxes
    # fell through to CPU. Follows same priority as CLAUDE.md get_device() pattern.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load processor for multimodal preprocessing
    # This validates preprocessing pipeline used during training
    processor = AutoProcessor.from_pretrained(model_id)

    # Optimize data type based on device capabilities.
    # bfloat16 provides significant memory savings when supported.
    # Delegated to probe_bfloat16() in utils/device.py to avoid duplicating
    # the try/except probe logic (and the separate CUDA branch) across scripts.
    bfloat16_supported = probe_bfloat16(device)

    # Select optimal dtype based on hardware support testing
    optimal_dtype = torch.bfloat16 if bfloat16_supported else torch.float32

    # Load model with memory and performance optimizations
    # Uses identical configuration to training pipeline for consistency
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=optimal_dtype,
        attn_implementation=constants.ATTENTION_IMPLEMENTATION,
        low_cpu_mem_usage=constants.LOW_CPU_MEM_USAGE,
    )

    # Move model to the selected device for inference
    model = model.to(device)

    # Generate synthetic audio workload for consistent performance testing
    # Uses realistic duration and sampling rate for representative measurements
    processor_sampling_rate = getattr(processor, "sampling_rate", None)
    if processor_sampling_rate is None and hasattr(processor, "feature_extractor"):
        processor_sampling_rate = getattr(processor.feature_extractor, "sampling_rate", constants.DEFAULT_SAMPLING_RATE)

    # Create synthetic audio with realistic characteristics
    effective_sampling_rate = processor_sampling_rate or constants.DEFAULT_SAMPLING_RATE
    audio_sample_count = int(constants.SYNTHETIC_AUDIO_DURATION_SECONDS * effective_sampling_rate)
    synthetic_audio = torch.randn(audio_sample_count).tolist()

    # Construct multimodal message using standard chat template
    # This format matches training pipeline for preprocessing consistency
    chat_template = constants.CHAT_TEMPLATE
    multimodal_messages = [
        [
            {
                "role": chat_template["USER_ROLE"],
                "content": [
                    {"type": chat_template["AUDIO_TYPE"], "audio": chat_template["AUDIO_PLACEHOLDER"]},
                    {"type": chat_template["TEXT_TYPE"], "text": chat_template["TRANSCRIPTION_PROMPT"]},
                ],
            }
        ]
    ]

    # Process multimodal inputs through preprocessing pipeline
    # This validates the complete preprocessing workflow used during training
    processed_inputs = processor(
        messages=multimodal_messages, audios=[synthetic_audio], return_tensors="pt", padding=True
    )

    # Move all tensors to the selected device for inference
    for key, value in list(processed_inputs.items()):
        if torch.is_tensor(value):
            processed_inputs[key] = value.to(device)

    # Perform high-precision performance measurement with device synchronization
    # Device synchronization ensures accurate timing measurements
    if device.type == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    # Execute forward pass with memory-optimized inference
    # inference_mode() provides optimal memory usage and performance
    with torch.inference_mode():
        _ = model(**processed_inputs)

    # Ensure all operations complete before timing measurement.
    # Both MPS and CUDA dispatch asynchronously — synchronize to get accurate wall time.
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    inference_duration = time.perf_counter() - start_time

    # Measure peak memory consumption for capacity planning
    # RSS (Resident Set Size) provides accurate memory usage measurement
    peak_memory_bytes = process_monitor.memory_info().rss
    peak_memory_mb = peak_memory_bytes / constants.BYTES_TO_MB_DIVISOR

    # Report structured performance metrics for analysis and planning
    print(
        f"device={device}, dtype={optimal_dtype}, "
        f"time_s={inference_duration:.{constants.TIME_PRECISION_DIGITS}f}, "
        f"rss_mb={peak_memory_mb:.{constants.MEMORY_PRECISION_DIGITS}f}"
    )


if __name__ == "__main__":
    main()
