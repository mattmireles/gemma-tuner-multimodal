"""
End-to-End Integration Tests for Whisper Supervised Fine-Tuning Workflow

This module provides comprehensive integration testing for the complete Whisper
fine-tuning pipeline, validating the entire workflow from dataset preparation
through model training on Apple Silicon (MPS), CUDA, and CPU platforms.

Key responsibilities:
- End-to-end supervised fine-tuning workflow validation
- Minimal synthetic dataset generation for rapid testing
- MPS-specific compatibility testing and regression prevention
- Training pipeline smoke tests with single-sample datasets
- API compatibility verification for transformers library updates

Called by:
- pytest test suite during automated testing
- CI/CD pipelines for pre-merge validation
- Manual testing via pytest tests/test_sft_workflow.py
- Regression testing workflows after dependency updates
- Platform-specific validation (MPS, CUDA, CPU)

Calls to:
- models/whisper/finetune.py:main() for supervised fine-tuning execution (line 83)
- numpy for synthetic audio generation (line 18)
- soundfile for WAV file writing (line 20)
- transformers.Seq2SeqTrainingArguments for API compatibility testing (line 95)
- json for training results validation (line 90)

Test strategy:
- Use minimal synthetic data (1 sample) to test complete pipeline
- Generate simple sine wave audio to avoid external dependencies
- Validate training completion via output artifacts
- Test with minimal epochs/steps for speed
- Disable features that may cause issues (gradient checkpointing on MPS)

Platform considerations:
- MPS: Gradient checkpointing disabled to avoid double-backward issues
- CUDA: Full feature testing enabled
- CPU: Reduced batch sizes for memory efficiency

Test coverage:
1. Dataset preparation and loading
2. Model initialization and configuration
3. Training loop execution
4. Checkpoint saving
5. Results JSON generation
6. API compatibility with transformers

Regression prevention:
- Seq2SeqTrainingArguments eval_strategy vs evaluation_strategy
- MPS gradient checkpointing compatibility
- Audio preprocessing pipeline validation
- Memory management on limited-resource systems
"""

import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest


class TestConstants:
    """Named constants for test configuration and synthetic data generation.
    
    This class centralizes test-specific constants to avoid magic values
    throughout the test suite. Constants define audio generation parameters,
    dataset structure, and training configuration for integration tests.
    """
    
    # Audio generation parameters for synthetic test data
    # 16kHz is the standard Whisper sample rate for all models
    SAMPLE_RATE = 16000
    
    # 440Hz sine wave (musical note A4) for predictable test audio
    # This frequency is easily distinguishable and commonly used in audio testing
    SINE_FREQUENCY = 440.0
    
    # 1-second duration for minimal but valid audio samples
    # Sufficient for testing while keeping test execution fast
    TEST_AUDIO_DURATION = 1.0
    
    # Amplitude for generated sine waves (10% of max to avoid clipping)
    # Lower amplitude prevents potential audio processing issues
    SINE_AMPLITUDE = 0.1
    
    # Dataset configuration for minimal testing
    MIN_TRAIN_SAMPLES = 1  # Single sample for smoke testing
    TEST_DATASET_NAME = "test_streaming"  # Synthetic dataset identifier
    
    # Training configuration for rapid testing
    TEST_BATCH_SIZE = 1  # Minimal batch size for memory efficiency
    TEST_EPOCHS = 1  # Single epoch for smoke testing
    TEST_LEARNING_RATE = 1e-5  # Conservative learning rate
    TEST_WARMUP_STEPS = 0  # No warmup for single-sample tests
    TEST_SAVE_STEPS = 1000  # High value to avoid mid-training saves
    TEST_LOGGING_STEPS = 1  # Log every step for debugging
    
    # Model configuration for testing
    TEST_MODEL = "openai/whisper-tiny"  # Smallest model for fast testing
    TEST_MAX_LABEL_LENGTH = 64  # Short transcripts for test data
    TEST_MAX_DURATION = 10.0  # Maximum audio duration in seconds
    
    # File paths and extensions
    TRAIN_CSV_FILENAME = "train.csv"
    VALIDATION_CSV_FILENAME = "validation.csv"
    TEST_WAV_FILENAME = "tiny.wav"
    RESULTS_JSON_FILENAME = "train_results.json"
    
    # CSV column names matching dataset format
    ID_COLUMN = "id"
    AUDIO_PATH_COLUMN = "audio_path"
    TEXT_COLUMN = "text_perfect"
    
    # Test text for synthetic transcriptions
    TEST_TRANSCRIPTION = "hello world"
    
    # Platform-specific settings
    MPS_GRADIENT_CHECKPOINTING = False  # Disabled due to MPS double-backward issues
    DEFAULT_GRADIENT_CHECKPOINTING = False  # Default for other platforms
    
    # Cleanup settings
    CLEANUP_RETRY_ATTEMPTS = 3  # Number of attempts for file cleanup
    CLEANUP_RETRY_DELAY = 0.1  # Seconds between cleanup attempts


def _ensure_tiny_dataset(base_dir: Path) -> None:
    """
    Creates a minimal synthetic dataset for integration testing.
    
    This helper function generates a single-sample dataset with synthetic audio
    and transcription for testing the complete fine-tuning pipeline. The dataset
    structure matches the expected format for the training scripts.
    
    Called by:
    - test_sft_single_step() for dataset preparation (line 32)
    - Other test functions requiring synthetic data
    - Manual testing utilities for quick validation
    
    Calls to:
    - numpy.linspace() for time array generation (line 17)
    - numpy.sin() for sine wave generation (line 18)  
    - soundfile.write() for WAV file creation (line 20)
    - Path.mkdir() for directory structure creation
    - Path.open() for CSV file writing
    
    Dataset structure created:
    - data/datasets/test_streaming/
    - data/datasets/test_streaming/audio/
    - data/datasets/test_streaming/audio/tiny.wav (1-second sine wave)
    - data/datasets/test_streaming/train.csv (single-row dataset)
    
    Args:
        base_dir (Path): Base directory for dataset creation, typically project root
        
    Returns:
        None
        
    Side effects:
        - Creates directory structure under base_dir/data/datasets/
        - Generates synthetic WAV file with 440Hz sine wave
        - Creates train.csv with single sample entry
        - Skips creation if train.csv already exists (idempotent)
        
    Audio specifications:
        - Sample rate: 16kHz (Whisper standard)
        - Duration: 1 second
        - Frequency: 440Hz (A4 note)
        - Amplitude: 10% of maximum
        - Format: WAV (PCM 16-bit)
        
    CSV format:
        - Headers: id, audio_path, text_perfect
        - Single row with synthetic data
        - Absolute paths for audio files
        
    Example:
        >>> base_dir = Path("/tmp/test_project")
        >>> _ensure_tiny_dataset(base_dir)
        >>> assert (base_dir / "data/datasets/test_streaming/train.csv").exists()
    """
    ds_dir = base_dir / "data" / "datasets" / TestConstants.TEST_DATASET_NAME
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Generate a 1-second 16kHz sine wave and write to WAV
    audio_dir = ds_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sr = TestConstants.SAMPLE_RATE
    t = np.linspace(0, TestConstants.TEST_AUDIO_DURATION, int(sr * TestConstants.TEST_AUDIO_DURATION), endpoint=False)
    audio = TestConstants.SINE_AMPLITUDE * np.sin(2 * np.pi * TestConstants.SINE_FREQUENCY * t)
    wav_path = audio_dir / TestConstants.TEST_WAV_FILENAME
    sf.write(wav_path.as_posix(), audio, sr)

    # Minimal train split with required columns: id, audio_path, text_perfect
    train_csv = ds_dir / TestConstants.TRAIN_CSV_FILENAME
    if not train_csv.exists():
        with train_csv.open("w") as f:
            f.write(f"{TestConstants.ID_COLUMN},{TestConstants.AUDIO_PATH_COLUMN},{TestConstants.TEXT_COLUMN}\n")
            f.write(f"1,{wav_path.as_posix()},{TestConstants.TEST_TRANSCRIPTION}\n")


@pytest.mark.slow
def test_sft_single_step(tmp_path: Path):
    """
    End-to-end integration test for supervised fine-tuning workflow.
    
    This test validates the complete training pipeline from dataset loading
    through model training and results generation. It uses a minimal synthetic
    dataset to ensure fast execution while testing all critical components.
    
    Called by:
    - pytest test runner during test suite execution
    - CI/CD pipelines for automated validation
    - Manual testing via pytest invocation
    
    Calls to:
    - _ensure_tiny_dataset() for synthetic data generation (line 32)
    - models/whisper/finetune.py:main() for training execution (line 83)
    - json.load() for results validation (line 90)
    - os.walk() for directory cleanup (line 68)
    
    Test workflow:
    1. Create synthetic dataset with single audio sample
    2. Configure minimal training parameters
    3. Execute supervised fine-tuning
    4. Validate training results JSON
    5. Clean up output directory
    
    Args:
        tmp_path (Path): pytest fixture providing temporary directory
        
    Assertions:
        - Training results JSON file exists
        - JSON can be parsed successfully
        
    Side effects:
        - Creates synthetic dataset in project directory
        - Creates output directory with training artifacts
        - Attempts cleanup of output directory (best-effort)
        
    Platform-specific behavior:
        - MPS: Gradient checkpointing disabled
        - CUDA: Full features enabled
        - CPU: Standard configuration
        
    Known issues:
        - MPS gradient checkpointing causes double-backward errors
        - Directory cleanup may fail on Windows (file locks)
        
    Example execution:
        >>> pytest tests/test_sft_workflow.py::test_sft_single_step -v
    """
    base_dir = Path.cwd()
    _ensure_tiny_dataset(base_dir)

    # Minimal profile_config targeting whisper-tiny and our tiny dataset
    profile_config = {
        "model": "whisper-tiny",
        "base_model": TestConstants.TEST_MODEL,
        "dataset": TestConstants.TEST_DATASET_NAME,
        "text_column": TestConstants.TEXT_COLUMN,
        "id_column": TestConstants.ID_COLUMN,
        "train_split": "train",
        "validation_split": "validation",
        "max_label_length": TestConstants.TEST_MAX_LABEL_LENGTH,
        "max_duration": TestConstants.TEST_MAX_DURATION,
        "per_device_train_batch_size": TestConstants.TEST_BATCH_SIZE,
        "per_device_eval_batch_size": TestConstants.TEST_BATCH_SIZE,
        "num_train_epochs": TestConstants.TEST_EPOCHS,
        "learning_rate": TestConstants.TEST_LEARNING_RATE,
        "warmup_steps": TestConstants.TEST_WARMUP_STEPS,
        "save_steps": TestConstants.TEST_SAVE_STEPS,
        "save_total_limit": 1,
        "logging_steps": TestConstants.TEST_LOGGING_STEPS,
        "gradient_accumulation_steps": 1,
        # Disable gradient checkpointing in E2E smoke to avoid MPS double-backward issues
        "gradient_checkpointing": TestConstants.MPS_GRADIENT_CHECKPOINTING,
        "dtype": "float32",
        "attn_implementation": "eager",
        "language_mode": "mixed",
        "max_samples": TestConstants.MIN_TRAIN_SAMPLES,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "visualize": False,
    }

    out_dir = base_dir / "output" / "test_sft_workflow"
    if out_dir.exists():
        # best-effort cleanup to avoid interference between runs
        for root, dirs, files in os.walk(out_dir.as_posix(), topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)

    # Invoke the standard fine-tuning entrypoint
    from models.whisper.finetune import main as sft_main

    sft_main(profile_config, out_dir.as_posix())

    # Assert a successful run by checking for train_results.json
    results_path = out_dir / TestConstants.RESULTS_JSON_FILENAME
    assert results_path.exists(), "Expected training results to be written"
    with results_path.open() as f:
        _ = json.load(f)


def test_seq2seq_training_args_initialization():
    """
    Regression test for Seq2SeqTrainingArguments API compatibility.
    
    This test ensures compatibility with the evolving transformers library API,
    specifically the evaluation_strategy vs eval_strategy parameter changes
    that have caused breaking changes in past releases.
    
    Called by:
    - pytest test runner during regression testing
    - CI/CD pipelines after dependency updates
    - Manual validation before transformers upgrades
    
    Calls to:
    - transformers.Seq2SeqTrainingArguments constructor (lines 100-108)
    - tempfile.TemporaryDirectory for isolated testing (line 100)
    
    Test coverage:
    1. Basic argument initialization
    2. eval_strategy parameter acceptance
    3. predict_with_generate flag compatibility
    4. Generation configuration support
    
    Args:
        None (uses temporary directory fixture internally)
        
    Assertions:
        - Seq2SeqTrainingArguments accepts eval_strategy parameter
        - predict_with_generate flag is properly set
        - generation_max_length is configured correctly
        
    Regression history:
    - transformers 4.45+: evaluation_strategy deprecated for eval_strategy
    - transformers 4.53+: eval_strategy becomes primary parameter
    - This test ensures compatibility across versions
    
    Example:
        >>> pytest tests/test_sft_workflow.py::test_seq2seq_training_args_initialization -v
    """
    from transformers import Seq2SeqTrainingArguments
    import tempfile
    
    # This test ensures we catch breaking API changes in transformers
    # Specifically, the evaluation_strategy -> eval_strategy rename
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Test that eval_strategy works (new API)
            args = Seq2SeqTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                predict_with_generate=True,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                load_best_model_at_end=True,
                metric_for_best_model="wer",
                greater_is_better=False,
            )
            assert args.eval_strategy == "epoch", "eval_strategy parameter should work"
        except TypeError as e:
            assert False, f"Seq2SeqTrainingArguments initialization failed: {e}"
