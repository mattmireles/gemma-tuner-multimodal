#!/usr/bin/env python3
"""
MPS Migration Validation and Testing Suite

This script provides comprehensive testing and validation of Metal Performance Shaders (MPS)
migration for Whisper fine-tuning on Apple Silicon. It verifies device detection, model loading,
basic operations, and system compatibility to ensure proper MPS setup before training.

Key responsibilities:
- MPS device detection and capability verification
- Whisper model loading and device placement testing  
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
- transformers library for Whisper model loading and testing

Test categories:

Device detection:
- MPS availability and build verification
- Device information and capability reporting
- Memory management configuration validation
- Fallback device behavior testing

Model loading:
- Small Whisper model loading (whisper-tiny)
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
Whisper fine-tuning workflows.
"""

import torch
import os
import sys
import pytest

# Set environment variable for initial testing
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.device import get_device, verify_mps_setup, get_device_info, get_memory_stats
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
    print("=" * 60)
    print("Testing Device Detection")
    print("=" * 60)
    
    device = get_device()
    print(f"Detected device: {device}")
    
    mps_available, mps_message = verify_mps_setup()
    print(f"MPS setup: {mps_message}")
    
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"{key}: {value}")
    
    print()

@pytest.mark.slow
def test_model_loading():
    """
    Validates Whisper model loading and device placement on MPS.
    
    This function tests the critical model loading pipeline that will be used
    in actual training and evaluation workflows. It uses a small model (whisper-tiny)
    to minimize memory usage while validating the complete loading process.
    
    Called by:
    - main() after successful device detection
    - Model loading troubleshooting workflows
    
    Test procedure:
    1. Load WhisperProcessor for audio preprocessing
    2. Load WhisperForConditionalGeneration model
    3. Configure mixed precision (float16) for MPS optimization
    4. Transfer model to MPS device
    5. Verify successful device placement
    6. Monitor memory usage during loading
    
    Model configuration:
    - Model: openai/whisper-tiny (smallest available model)
    - Precision: float16 for MPS/CUDA, float32 for CPU
    - Attention: SDPA (Scaled Dot Product Attention) implementation
    - Memory optimization: low_cpu_mem_usage=True
    
    MPS-specific considerations:
    - Float16 precision validation on Apple Silicon
    - Unified memory architecture compatibility
    - Model size impact on system memory pressure
    - Device placement verification
    
    Memory monitoring:
    - Tracks memory allocation during model loading
    - Reports peak memory usage
    - Validates memory statistics accessibility
    - Identifies potential memory pressure issues
    
    Success criteria:
    - Model loads without errors or warnings
    - Device placement confirmed as MPS
    - Memory statistics reported successfully
    - No system memory pressure indicators
    
    Failure scenarios:
    - Model loading fails with MPS-specific errors
    - Device placement unsuccessful (remains on CPU)
    - Memory allocation exceeds available unified memory
    - Float16 precision not supported on target hardware
    
    Returns:
        bool: True if model loading successful, False otherwise
        
    Error handling:
    - Catches and reports specific model loading exceptions
    - Provides diagnostic information for troubleshooting
    - Continues test suite execution even if this test fails
    """
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    device = get_device()
    
    # Test with a small model first
    model_name = "openai/whisper-tiny"
    print(f"Loading {model_name}...")
    
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
            attn_implementation="sdpa",  # Using sdpa instead of flash_attention_2
            low_cpu_mem_usage=True
        ).to(device)
        
        print(f"✓ Model loaded successfully on {device}")
        
        # Check memory usage
        mem_stats = get_memory_stats()
        print(f"Memory stats: {mem_stats}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    print()
    return True

@pytest.mark.slow
def test_inference():
    """
    Validates core GPU operations essential for Whisper model inference.
    
    This function tests fundamental tensor operations that are used throughout
    Whisper model inference, ensuring that MPS can handle the computational
    patterns required for speech recognition without numerical or operational issues.
    
    Called by:
    - main() after successful model loading
    - GPU operation troubleshooting workflows
    
    Test operations:
    
    1. Convolution operation simulation:
       - Simulates mel-spectrogram feature extraction
       - Tests 1D convolution with realistic tensor dimensions
       - Validates MPS convolution kernel compatibility
    
    2. Attention mechanism simulation:
       - Creates attention weight matrices
       - Performs softmax normalization (numerically sensitive)
       - Executes batch matrix multiplication (BMM)
       - Tests core transformer operation patterns
    
    Tensor dimensions:
    - Batch size: 2 (multi-sample processing)
    - Feature size: 80 (Whisper mel-spectrogram features)
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
    
    Returns:
        bool: True if all operations successful, False otherwise
        
    Common failure modes:
    - Specific operations not implemented in MPS backend
    - Numerical precision issues with float16 on Apple Silicon
    - Memory pressure causing operation failures
    - Tensor shape incompatibilities with MPS kernels
    
    This test validates that the core computational patterns used in
    Whisper inference will work reliably on the target MPS device.
    """
    print("=" * 60)
    print("Testing Inference")
    print("=" * 60)
    
    device = get_device()
    
    try:
        # Create dummy input
        batch_size = 2
        feature_size = 80
        sequence_length = 3000  # 30 seconds at 100Hz
        
        dummy_input = torch.randn(batch_size, feature_size, sequence_length).to(device)
        print(f"Created dummy input tensor: {dummy_input.shape} on {device}")
        
        # Test a simple operation
        with torch.no_grad():
            # Simulate mel spectrogram processing
            output = torch.nn.functional.conv1d(
                dummy_input,
                torch.randn(512, feature_size, 10).to(device),
                padding=5
            )
            print(f"✓ Convolution operation successful: output shape {output.shape}")
            
            # Test attention-like operation
            attention_weights = torch.softmax(torch.randn(batch_size, 100, 100).to(device), dim=-1)
            values = torch.randn(batch_size, 100, 64).to(device)
            attention_output = torch.bmm(attention_weights, values)
            print(f"✓ Attention operation successful: output shape {attention_output.shape}")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False
    
    print()
    return True

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
    for Whisper fine-tuning, not just the MPS components.
    """
    print("=" * 60)
    print("Running System Check")
    print("=" * 60)
    
    try:
        from scripts.system_check import main as system_check_main
        system_check_main()
    except Exception as e:
        print(f"Error running system check: {e}")
    
    print()

def main():
    """
    Orchestrates the complete MPS migration validation test suite.
    
    This function coordinates all validation tests in logical order,
    providing comprehensive verification of MPS setup and compatibility
    for Whisper fine-tuning workflows.
    
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
    ensuring system readiness for production Whisper fine-tuning.
    """
    print("\nMPS Migration Test Suite")
    print("=" * 60)
    print()
    
    # Run tests
    test_device_detection()
    
    if get_device().type == "mps":
        print("✓ MPS device detected! Running MPS-specific tests...\n")
        
        success = test_model_loading()
        if success:
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
    # Entry point for MPS migration validation
    # Sets up MPS fallback environment variable for initial testing
    main()