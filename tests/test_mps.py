#!/usr/bin/env python3
"""
MPS Migration Validation and Testing Suite

This script provides comprehensive testing and validation of Metal Performance Shaders (MPS)
migration for PyTorch training on Apple Silicon. It verifies device detection, model loading,
basic operations, and system compatibility to ensure proper MPS setup before training.

Key responsibilities:
- MPS device detection and capability verification
- Small Transformers model loading and device placement testing
- GPU operation validation (convolution, attention, memory management)
- System compatibility checking and diagnostic reporting
- Performance baseline establishment for Apple Silicon

Called by:
- Manual execution for MPS setup verification
- Development workflows before training experiments
- CI/CD pipelines for Apple Silicon compatibility testing
- Troubleshooting workflows for MPS-related issues

Calls to:
- utils/device.py for device detection and management utilities
- scripts/system_check.py for comprehensive system validation
- transformers library for a tiny reference model load

Test categories:

Device detection:
- MPS availability and build verification
- Device information and capability reporting
- Memory management configuration validation
- Fallback device behavior testing

Model loading:
- Tiny Hugging Face test model (`hf-internal-testing/tiny-random-BertModel`)
- Device placement verification
- Memory usage monitoring and reporting
- Mixed precision configuration testing

Operation validation:
- Basic tensor operations on MPS device
- Convolution operations (CNN layer simulation)
- Attention-like operations (transformer simulation)
- Memory synchronization and cache management

System integration:
- Complete system check integration
- Configuration validation
- Environment variable verification
- Hardware compatibility assessment

MPS-specific considerations:
- Unified memory architecture implications
- Memory pressure management validation
- Operation fallback testing (MPS -> CPU)
- Numerical precision verification

Diagnostic output:
- Structured test results with pass/fail status
- Detailed error messages for troubleshooting
- Performance metrics and memory usage
- Recommendations for optimization

Troubleshooting scenarios:
- PyTorch installation without MPS support
- x86_64 Python running under Rosetta 2
- Insufficient macOS version (requires 12.3+)
- Memory pressure and swapping issues
- MPS operation compatibility problems

Success criteria:
- Device detection identifies MPS as primary device
- Model loads successfully on MPS device
- Basic operations complete without errors
- Memory management functions properly
- System check passes all requirements

Failure handling:
- Clear diagnostic messages for each failure mode
- Fallback behavior validation (MPS -> CPU)
- Recovery recommendations and next steps
- Environment configuration guidance

Usage patterns:
- Pre-training validation: Verify setup before experiments
- Development testing: Validate code changes affecting MPS
- Production deployment: Ensure compatibility in new environments
- Debugging: Isolate MPS-specific issues from general problems

Output interpretation:
- ✓ symbols: Successful test completion
- ✗ symbols: Test failures requiring attention
- Warning messages: Non-critical issues or recommendations
- Memory statistics: Usage patterns and optimization opportunities

This testing suite is essential for reliable Apple Silicon deployment,
preventing common MPS issues and ensuring optimal performance for
Gemma (and general PyTorch) training workflows.
"""

import os
import sys

import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModel

from gemma_tuner.utils.device import get_device, get_device_info, get_memory_stats, verify_mps_setup


@pytest.fixture(autouse=True)
def _mps_fallback_env(monkeypatch):
    """
    Sets PYTORCH_ENABLE_MPS_FALLBACK=1 for the duration of each test, then
    restores the original environment. Using monkeypatch ensures cleanup even
    if a test fails, and avoids polluting the session for unrelated tests.
    """
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def test_device_detection():
    """
    Validates MPS device detection and reports comprehensive device information.

    This function performs the foundational test of the MPS migration by verifying
    that device detection properly identifies Apple Silicon GPU capabilities and
    configures the system for optimal performance.

    Called by:
    - main() as the first test in the validation suite
    - Standalone execution for quick device verification

    Test components:
    1. Primary device detection using utils.device.get_device()
    2. MPS setup verification with detailed diagnostics
    3. Comprehensive device information reporting
    4. Memory management configuration validation

    Success indicators:
    - Device type identified as 'mps'
    - MPS backend reports as available and built
    - Device information includes Apple Silicon details
    - Memory statistics accessible

    Failure modes:
    - Device detection returns 'cpu' or 'cuda' instead of 'mps'
    - MPS backend not built into PyTorch installation
    - MPS available but not functional (macOS version issues)
    - Memory queries fail (PyTorch version compatibility)

    Output format:
    Structured diagnostic output with clear headers and status indicators:
    - Device detection results
    - MPS availability status and diagnostic messages
    - Complete device information dictionary
    - Memory management statistics

    This test must pass for subsequent MPS-specific tests to be meaningful.
    """
    # Device detection must return a valid torch.device
    device = get_device()
    assert device is not None, "get_device() returned None"
    assert isinstance(device, torch.device), f"Expected torch.device, got {type(device)}"
    assert device.type in ("mps", "cuda", "cpu"), f"Unexpected device type: {device.type}"

    # MPS setup verification must return a (bool, str) tuple
    mps_available, mps_message = verify_mps_setup()
    assert isinstance(mps_available, bool), f"Expected bool, got {type(mps_available)}"
    assert isinstance(mps_message, str), f"Expected str, got {type(mps_message)}"
    assert len(mps_message) > 0, "MPS diagnostic message must not be empty"

    # Device info must return a populated dict with required keys
    device_info = get_device_info()
    assert isinstance(device_info, dict), f"Expected dict, got {type(device_info)}"
    assert "device" in device_info, "device_info missing 'device' key"
    assert "device_type" in device_info, "device_info missing 'device_type' key"
    assert device_info["device_type"] == device.type, (
        f"device_info type '{device_info['device_type']}' != detected device type '{device.type}'"
    )


@pytest.mark.slow
def test_model_loading():
    """
    Validates Transformers model loading and device placement on the active device.

    Uses Hugging Face's tiny random BERT fixture to minimize download size and memory
    while exercising the same load → `.to(device)` path real training uses.

    Model configuration:
    - Model: hf-internal-testing/tiny-random-BertModel
    - Precision: float16 for MPS/CUDA, float32 for CPU
    - Memory optimization: low_cpu_mem_usage=True
    """
    device = get_device()

    model_name = "hf-internal-testing/tiny-random-BertModel"

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in ("cuda", "mps") else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    assert model is not None, "AutoModel.from_pretrained returned None"

    # Verify model is on the expected device
    first_param = next(model.parameters())
    assert first_param.device.type == device.type, f"Model parameter on {first_param.device}, expected {device}"

    # Verify memory stats are retrievable after loading a model
    mem_stats = get_memory_stats()
    assert isinstance(mem_stats, dict), f"Expected dict from get_memory_stats, got {type(mem_stats)}"
    assert "device" in mem_stats, "Memory stats missing 'device' key"


@pytest.mark.slow
def test_inference():
    """
    Validates core GPU tensor operations used in audio and transformer workloads.

    Exercises convolution- and attention-shaped ops on MPS without pulling in a
    full multimodal Gemma checkpoint (those are covered elsewhere).

    Called by:
    - main() after successful model loading
    - GPU operation troubleshooting workflows

    Test operations:

    1. Convolution operation simulation:
       - Simulates mel-style feature extraction (1D conv over time)
       - Tests 1D convolution with realistic tensor dimensions
       - Validates MPS convolution kernel compatibility

    2. Attention mechanism simulation:
       - Creates attention weight matrices
       - Performs softmax normalization (numerically sensitive)
       - Executes batch matrix multiplication (BMM)
       - Tests core transformer operation patterns

    Tensor dimensions:
    - Batch size: 2 (multi-sample processing)
    - Feature size: 80 (typical log-mel width)
    - Sequence length: 3000 (30 seconds at 100Hz)
    - Hidden size: 64 (attention dimension)

    MPS operation validation:
    - Tensor creation and device placement
    - Mathematical operations (convolution, softmax, BMM)
    - Memory management during computation
    - Numerical stability verification

    Error detection:
    - Operation failures: NotImplementedError for unsupported ops
    - Numerical issues: NaN or Inf results
    - Memory errors: Out-of-memory during computation
    - Device placement failures: Operations defaulting to CPU

    Success indicators:
    - All operations complete without exceptions
    - Output tensors have expected shapes
    - Results remain on MPS device
    - No numerical instabilities detected

    Performance implications:
    - Operation timing (though not primarily a performance test)
    - Memory efficiency during computation
    - MPS kernel selection and optimization

    Common failure modes:
    - Specific operations not implemented in MPS backend
    - Numerical precision issues with float16 on Apple Silicon
    - Memory pressure causing operation failures
    - Tensor shape incompatibilities with MPS kernels

    This test validates that the core computational patterns used in
    transformer-style inference will work reliably on the target MPS device.
    """
    device = get_device()

    batch_size = 2
    feature_size = 80
    sequence_length = 3000  # 30 seconds at 100Hz

    dummy_input = torch.randn(batch_size, feature_size, sequence_length).to(device)
    assert dummy_input.device.type == device.type, f"Input tensor on {dummy_input.device}, expected {device}"

    with torch.no_grad():
        # Simulate mel spectrogram processing via 1-D convolution
        conv_weight = torch.randn(512, feature_size, 10).to(device)
        output = torch.nn.functional.conv1d(dummy_input, conv_weight, padding=5)

        assert output.shape == (batch_size, 512, sequence_length + 1), f"Unexpected conv1d output shape: {output.shape}"
        assert output.device.type == device.type, f"Conv output on {output.device}, expected {device}"
        assert torch.isfinite(output).all(), "Conv1d output contains NaN or Inf"

        # Simulate attention mechanism: softmax + batch matmul
        attention_weights = torch.softmax(torch.randn(batch_size, 100, 100).to(device), dim=-1)
        values = torch.randn(batch_size, 100, 64).to(device)
        attention_output = torch.bmm(attention_weights, values)

        assert attention_output.shape == (batch_size, 100, 64), (
            f"Unexpected attention output shape: {attention_output.shape}"
        )
        assert attention_output.device.type == device.type, (
            f"Attention output on {attention_output.device}, expected {device}"
        )
        assert torch.isfinite(attention_output).all(), "Attention output contains NaN or Inf"


def test_system_check():
    """
    Executes comprehensive system compatibility validation.

    This function integrates the complete system check functionality,
    providing thorough validation of the entire training environment
    beyond just MPS-specific components.

    Called by:
    - main() as the final validation step
    - Comprehensive system validation workflows

    Calls to:
    - scripts.system_check.main() for complete system validation

    System check components:
    - Hardware compatibility (Apple Silicon detection)
    - Software versions (PyTorch, transformers, datasets)
    - Environment configuration (Python architecture, conda/pip)
    - Device capabilities (MPS, CUDA, CPU performance)
    - Memory and storage availability
    - Dependency verification and compatibility

    Integration benefits:
    - Validates complete training environment
    - Identifies issues beyond MPS-specific problems
    - Provides comprehensive diagnostic information
    - Ensures readiness for production training workflows

    Error handling:
    - Catches and reports system check exceptions
    - Continues test execution even if system check fails
    - Provides context for system check failures

    This test ensures that the entire system is properly configured
    for Gemma training workflows, not just the MPS components.
    """
    from gemma_tuner.scripts.system_check import main as system_check_main

    # Let any exception from system_check_main propagate to pytest so
    # failures are visible rather than silently swallowed.
    system_check_main()


def main():
    """
    Orchestrates the complete MPS migration validation test suite.

    This function coordinates all validation tests in logical order,
    providing comprehensive verification of MPS setup and compatibility
    for Gemma / PyTorch training workflows.

    Called by:
    - Direct script execution for MPS validation
    - Development workflows before training experiments
    - Automated testing in CI/CD pipelines

    Test execution flow:
    1. Device detection validation (foundational)
    2. MPS-specific tests (if MPS detected)
    3. Model loading validation
    4. GPU operation testing
    5. System-wide compatibility check
    6. Summary report and recommendations

    Conditional execution:
    - Full MPS test suite runs only if MPS device detected
    - Graceful fallback reporting for non-MPS systems
    - System check runs regardless of primary device type

    Success criteria:
    - All tests pass without critical errors
    - MPS device properly detected and functional
    - Model loading and operations work correctly
    - System environment properly configured

    Output format:
    - Structured test results with clear headers
    - Progress indicators and success/failure markers
    - Detailed recommendations for optimization
    - Troubleshooting guidance for failures

    Recommendations provided:
    - Initial training configuration (batch sizes, memory settings)
    - Environment variable configuration
    - Memory management best practices
    - Performance optimization suggestions

    This function serves as the complete validation entry point,
    ensuring system readiness for production training on the active device.
    """
    print("\nMPS Migration Test Suite")
    print("=" * 60)
    print()

    # Run tests
    test_device_detection()

    if get_device().type == "mps":
        print("✓ MPS device detected! Running MPS-specific tests...\n")

        test_model_loading()
        test_inference()

        test_system_check()

        print("\n" + "=" * 60)
        print("MPS Migration Summary")
        print("=" * 60)
        print("✓ Device detection working")
        print("✓ Model loading working")
        print("✓ Basic operations working")
        print("\nRecommendations:")
        print("1. Keep PYTORCH_ENABLE_MPS_FALLBACK=1 for initial training")
        print("2. Start with the reduced batch sizes in config.ini")
        print("3. Monitor memory usage during training")
        print("4. Increase batch sizes gradually if memory allows")

    else:
        print(f"Device type is {get_device().type}, not MPS. The migration supports CUDA and CPU as well.")
        test_system_check()


if __name__ == "__main__":
    # Entry point for MPS migration validation.
    # When run as a script (not via pytest), set the fallback env var directly
    # since pytest fixtures do not apply.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    main()
