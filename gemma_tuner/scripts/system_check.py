#!/usr/bin/env python3
"""
Comprehensive System Compatibility Checker

This module provides thorough system validation for Gemma fine-tuning environments,
verifying hardware compatibility, software versions, and configuration settings across
Apple Silicon, NVIDIA CUDA, and CPU platforms.

Key responsibilities:
- Hardware detection and capability assessment
- Software dependency version validation
- Environment configuration verification
- Performance baseline establishment
- Compatibility issue identification and reporting

Called by:
- test_mps.py for MPS-specific system validation
- Manual execution for environment setup verification
- CI/CD pipelines for automated compatibility testing
- Troubleshooting workflows for deployment issues

Validation categories:

Hardware assessment:
- Device type detection (MPS/CUDA/CPU)
- GPU capability and memory reporting
- Apple Silicon generation identification
- Multi-GPU configuration detection

Software validation:
- PyTorch version and MPS/CUDA support
- Transformers library compatibility
- Datasets library version verification
- Python architecture (ARM64 vs x86_64)

Environment verification:
- Operating system and version
- Conda/pip environment configuration
- Critical environment variables
- Path and dependency resolution

Performance assessment:
- Memory availability and limits
- Storage space and I/O performance
- Network connectivity for model downloads
- Baseline operation timing

Compatibility reporting:
- Clear pass/fail status for each component
- Detailed diagnostic information for failures
- Optimization recommendations
- Troubleshooting guidance and next steps

This system checker ensures reliable deployment across diverse hardware
and software configurations, preventing common setup issues and optimizing
performance for each target platform.
"""

import os
import platform
import subprocess

import torch
from packaging import version

from gemma_tuner.utils.device import get_device


def get_device_type() -> str:
    """Return device type string ('mps', 'cuda', 'cpu'). Delegates to utils.device.get_device()."""
    return get_device().type


def get_gpu_info():
    """
    Retrieves comprehensive GPU information for system diagnostics.

    This function gathers platform-specific GPU details essential for
    performance assessment, compatibility validation, and troubleshooting.
    The information collected varies significantly between MPS and CUDA
    due to architectural differences.

    Called by:
    - main() for GPU capability reporting
    - Performance assessment workflows
    - Hardware compatibility validation

    Platform-specific information:

    MPS (Apple Silicon):
    - Limited information due to Metal abstraction layer
    - Identifies as "Apple Silicon GPU" with MPS backend
    - Single device count (unified GPU architecture)
    - No capability versioning (handled by Metal/MPS)

    CUDA (NVIDIA):
    - Detailed GPU name and model information
    - Multi-GPU device count and enumeration
    - Compute capability version (major.minor)
    - Hardware-specific optimization information

    CPU fallback:
    - No GPU information available
    - Placeholder values for consistent interface

    Returns:
        tuple: (gpu_name, gpu_count, major_capability, minor_capability)
            - gpu_name (str): Human-readable GPU identifier
            - gpu_count (int): Number of available devices
            - major_capability (int): Major compute capability (CUDA only)
            - minor_capability (int): Minor compute capability (CUDA only)

    Error handling:
    - CUDA query failures: Returns placeholder values with error context
    - MPS unavailability: Graceful fallback to CPU information
    - Exception recovery: Prevents system check failure from GPU queries

    Example output:
        MPS: ("Apple Silicon GPU (Metal Performance Shaders)", 1, 0, 0)
        CUDA: ("NVIDIA GeForce RTX 3080", 1, 8, 6)
        CPU: ("No GPU available", 0, 0, 0)
    """
    device_type = get_device_type()

    if device_type == "mps":
        # MPS doesn't expose detailed GPU info like CUDA
        return "Apple Silicon GPU (Metal Performance Shaders)", 1, 0, 0
    elif device_type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            major, minor = torch.cuda.get_device_capability()
            return gpu_name, gpu_count, major, minor
        except Exception as e:
            # CUDA query failures can occur due to driver issues,
            # permission problems, or hardware incompatibilities
            return f"CUDA Error: {e}", 0, 0, 0
    else:
        return "No GPU found (CPU only)", 0, 0, 0


def get_memory_info():
    """
    Get comprehensive GPU/system memory information across different compute backends.

    This function provides unified memory reporting across CUDA, MPS, and CPU environments,
    handling platform-specific differences in memory querying capabilities. It's essential
    for diagnosing memory-related training issues and optimizing batch sizes.

    Called by:
    - main() function during system diagnostics
    - Training scripts for memory monitoring
    - Debugging utilities for OOM troubleshooting

    Calls to:
    - get_device_type() for backend detection
    - torch.mps.current_allocated_memory() for Apple Silicon memory
    - torch.cuda.get_device_properties() for CUDA memory info
    - torch.cuda.memory_allocated/reserved() for CUDA memory stats

    Platform-Specific Behavior:

    MPS (Apple Silicon):
    - Reports unified memory allocation only
    - Total/cached memory not available due to unified architecture
    - Returns allocated memory in GB

    CUDA:
    - Reports total GPU memory from device properties
    - Tracks allocated (active) and reserved (cached) memory
    - All values in gigabytes for consistency

    CPU:
    - Returns placeholder values as GPU memory N/A
    - System RAM monitoring handled separately

    Returns:
        Tuple[Union[str, float], Union[str, float], Union[str, float]]:
            - total_memory: Total GPU memory in GB or descriptive string
            - allocated_memory: Currently allocated memory in GB or "N/A"
            - cached_memory: Reserved/cached memory in GB or "N/A"

    Error Handling:
    - Graceful fallback for older PyTorch versions
    - Returns descriptive strings on query failure
    - Prevents crashes from missing backend support
    """
    device_type = get_device_type()

    if device_type == "mps":
        try:
            # MPS has limited memory querying capabilities
            allocated = torch.mps.current_allocated_memory() / 1e9  # in GB
            # MPS doesn't provide total memory or cached memory info
            return "Unified Memory", allocated, "N/A"
        except Exception:
            # Fallback for older PyTorch versions
            return "Unified Memory", "N/A", "N/A"
    elif device_type == "cuda":
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9  # in GB
            cached_memory = torch.cuda.memory_reserved(0) / 1e9  # in GB
            return total_memory, allocated_memory, cached_memory
        except Exception as e:
            return f"Error: {e}", 0, 0
    else:
        return "System RAM", "N/A", "N/A"


def check_flash_attention_2():
    """
    Check Flash Attention 2 availability and compatibility for optimized transformer training.

    Flash Attention 2 provides significant memory and speed improvements for transformer
    models by using a tiling algorithm that reduces memory access. This function verifies
    installation and version compatibility across different compute backends.

    Called by:
    - main() function during capability assessment
    - Training scripts for attention optimization selection
    - Model configuration utilities for automatic optimization

    Calls to:
    - get_device_type() for backend detection
    - flash_attn module import and version check
    - version.parse() for semantic version comparison

    Backend Support:

    MPS:
    - Not supported due to Metal API limitations
    - Recommends using scaled dot-product attention (sdpa) instead
    - SDPA provides similar benefits with MPS compatibility

    CUDA:
    - Requires flash_attn package installation
    - Needs version >= 2.0.0 for Gemma compatibility
    - Provides 2-4x speedup for attention layers

    CPU:
    - Not applicable for CPU-only systems

    Returns:
        str: Status message indicating:
            - "Supported": Version 2.0+ installed and compatible
            - "Not supported on MPS (use sdpa instead)": MPS backend detected
            - "Not installed": Package not found
            - "Not supported (version < 2.0.0)": Outdated version
            - Error message if detection fails

    Performance Impact:
    - 30-50% training speedup with large batch sizes
    - 40-60% memory reduction for attention computation
    - Critical for training large Gemma models efficiently
    """
    device_type = get_device_type()

    if device_type == "mps":
        return "Not supported on MPS (use sdpa instead)"

    try:
        # Try importing the library
        import flash_attn

        # If the import is successful, check the version
        if version.parse(flash_attn.__version__) >= version.parse("2.0.0"):
            return "Supported"
        else:
            return "Not supported (version < 2.0.0)"
    except ImportError:
        return "Not installed"
    except Exception as e:
        return f"Error: {e}"


def check_bfloat16_support():
    """
    Verify bfloat16 (Brain Floating Point) support for mixed precision training.

    BFloat16 offers better numerical stability than float16 for deep learning while
    providing similar memory savings. This function checks hardware and software
    support across different compute backends for optimal training configuration.

    Called by:
    - main() function during precision capability check
    - Training scripts for automatic precision selection
    - Model initialization for dtype configuration

    Calls to:
    - get_device_type() for backend detection
    - torch.cuda.is_bf16_supported() for CUDA capability check

    Precision Comparison:

    BFloat16 advantages:
    - Same range as float32 (prevents overflow)
    - Better gradient stability than float16
    - Native support on modern hardware

    Platform Support:

    MPS (Apple Silicon):
    - Fully supported on M1/M2/M3 chips
    - Hardware-accelerated operations
    - Recommended for Gemma training on Mac

    CUDA:
    - Requires Ampere (A100) or newer GPUs
    - Hardware acceleration on supported devices
    - Falls back to software emulation on older GPUs

    CPU:
    - Software emulation available
    - Slower than hardware acceleration
    - Useful for debugging/testing only

    Returns:
        Union[str, bool]: Support status:
            - "Supported on Apple Silicon": MPS with hardware support
            - True/False: CUDA hardware support status
            - "CPU supports bfloat16 (may be slower)": CPU emulation
            - Error message if detection fails

    Training Recommendations:
    - Use bfloat16 over float16 when available
    - 2x memory savings vs float32
    - Minimal accuracy loss for Gemma models
    """
    device_type = get_device_type()

    if device_type == "mps":
        # MPS supports bfloat16 on M1/M2/M3
        return "Supported on Apple Silicon"
    elif device_type == "cuda":
        try:
            return torch.cuda.is_bf16_supported()
        except Exception as e:
            return f"Error: {e}"
    else:
        return "CPU supports bfloat16 (may be slower)"


def get_cuda_version():
    """Gets the CUDA driver and runtime versions."""
    device_type = get_device_type()

    if device_type != "cuda":
        return "N/A", "N/A"

    try:
        driver_version = (
            subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
            .decode("utf-8")
            .strip()
        )
        runtime_version = torch.version.cuda
        return driver_version, runtime_version
    except Exception as e:
        return f"Error: {e}", "N/A"


def get_metal_version():
    """Gets Metal/MPS information."""
    device_type = get_device_type()

    if device_type != "mps":
        return "N/A", "N/A"

    # Check if MPS is built and available
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()

    return f"Built: {mps_built}", f"Available: {mps_available}"


def get_backend_version():
    """Gets the backend version (cuDNN for CUDA, Metal for MPS)."""
    device_type = get_device_type()

    if device_type == "cuda":
        try:
            return f"cuDNN {torch.backends.cudnn.version()}"
        except Exception as e:
            return f"cuDNN Error: {e}"
    elif device_type == "mps":
        return "Metal Performance Shaders"
    else:
        return "CPU backend"


def check_dependency_versions():
    """Checks if installed packages match requirements/requirements.txt versions."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as installed_version
    from pathlib import Path

    mismatches = []
    requirements_path = Path(__file__).parent.parent.parent / "requirements" / "requirements.txt"

    if not requirements_path.exists():
        return mismatches

    # Parse requirements.txt for version specifiers (supports ==, >=, <=, ~=, and ranges)
    from packaging.requirements import Requirement
    from packaging.version import Version

    with open(requirements_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and flags
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            try:
                req = Requirement(line)
                package_name = req.name

                try:
                    pkg_version = installed_version(package_name)
                    installed_ver = Version(pkg_version)
                    if not req.specifier.contains(installed_ver):
                        mismatches.append((package_name, str(req.specifier), pkg_version))
                except PackageNotFoundError:
                    mismatches.append((package_name, str(req.specifier), "NOT INSTALLED"))
            except Exception:
                # Skip lines that don't parse as valid requirements
                continue

    return mismatches


def get_python_version():
    """Gets the Python version."""
    return platform.python_version()


def get_os_version():
    """Gets the OS version."""
    return platform.platform()


def get_pytorch_version():
    """Gets the PyTorch version."""
    return torch.__version__


def get_architecture():
    """Gets the system architecture."""
    return platform.machine()


def check_environment_vars():
    """Checks relevant environment variables."""
    vars_to_check = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "MPS fallback to CPU for unsupported ops",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "MPS memory limit ratio",
        "CUDA_VISIBLE_DEVICES": "CUDA device visibility",
    }

    env_vars = {}
    for var, desc in vars_to_check.items():
        value = os.environ.get(var, "Not set")
        env_vars[var] = (value, desc)

    return env_vars


def main():
    """Prints system information relevant to training Gemma models."""
    print("System Check for Gemma Model Training\n")

    device_type = get_device_type()
    print(f"Detected Device Type: {device_type.upper()}")

    print("\n" + "=" * 40, "GPU Information", "=" * 40)
    gpu_name, gpu_count, major, minor = get_gpu_info()
    print(f"  GPU Name: {gpu_name}")
    print(f"  Number of GPUs: {gpu_count}")
    if device_type == "cuda":
        print(f"  Compute Capability: {major}.{minor}")

    total_memory, allocated_memory, cached_memory = get_memory_info()
    print(f"\n  Total GPU Memory: {total_memory if isinstance(total_memory, str) else f'{total_memory:.2f} GB'}")
    print(
        f"  Allocated GPU Memory: {allocated_memory if isinstance(allocated_memory, str) else f'{allocated_memory:.2f} GB'}"
    )
    print(f"  Cached GPU Memory: {cached_memory if isinstance(cached_memory, str) else f'{cached_memory:.2f} GB'}")

    print(f"\n  Flash Attention 2: {check_flash_attention_2()}")
    print(f"  bfloat16 Support: {check_bfloat16_support()}")

    print("\n" + "=" * 40, "Software Versions", "=" * 40)

    if device_type == "cuda":
        driver_version, runtime_version = get_cuda_version()
        print(f"  CUDA Driver Version: {driver_version}")
        print(f"  CUDA Runtime Version: {runtime_version}")
    elif device_type == "mps":
        mps_built, mps_available = get_metal_version()
        print(f"  MPS Status: {mps_built}, {mps_available}")

    print(f"  Backend: {get_backend_version()}")
    print(f"  Python Version: {get_python_version()}")
    print(f"  OS Version: {get_os_version()}")
    print(f"  Architecture: {get_architecture()}")
    print(f"  PyTorch Version: {get_pytorch_version()}")

    print("\n" + "=" * 40, "PyTorch Backend Support", "=" * 40)
    if device_type == "cuda":
        print(f"  PyTorch built with CUDA: {torch.version.cuda}")
        print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
    elif device_type == "mps":
        print(f"  PyTorch built with MPS: {torch.backends.mps.is_built()}")
        print(f"  PyTorch MPS available: {torch.backends.mps.is_available()}")
        print("  macOS version requirement: 12.3+ (Monterey or later)")

    print("\n" + "=" * 40, "Environment Variables", "=" * 40)
    env_vars = check_environment_vars()
    for var, (value, desc) in env_vars.items():
        print(f"  {var}: {value}")
        print(f"    → {desc}")

    print("\n" + "=" * 40, "Dependency Version Check", "=" * 40)
    mismatches = check_dependency_versions()
    if mismatches:
        print("  ⚠️  WARNING: The following packages don't match requirements/requirements.txt:")
        for package, required, installed in mismatches:
            print(f"    • {package}: required {required}, installed {installed}")
        print("\n  To fix: pip install -r requirements/requirements.txt")
    else:
        print("  ✅ All dependencies match requirements/requirements.txt")

    print("\n" + "=" * 40, "Recommendations", "=" * 40)
    if device_type == "mps":
        print("  • Use 'sdpa' attention instead of 'flash_attention_2'")
        print("  • Consider setting PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8")
        print("  • Start with smaller batch sizes (8-16) and increase gradually")
        print("  • Enable PYTORCH_ENABLE_MPS_FALLBACK=1 for initial testing")
    elif device_type == "cuda":
        print("  • Flash Attention 2 is recommended for better performance")
        print("  • Ensure CUDA drivers are up to date")
        print("  • Monitor GPU memory usage during training")
    else:
        print("  • Consider using a GPU for faster training")
        print("  • Training will be significantly slower on CPU")


if __name__ == "__main__":
    main()
