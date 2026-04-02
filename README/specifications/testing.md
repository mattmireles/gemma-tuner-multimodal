# Testing & Quality Assurance Protocol

## Executive Summary

The Whisper Fine-Tuner implements a comprehensive three-tiered quality assurance strategy to ensure model performance, code reliability, and production readiness. This protocol covers automated testing through pytest, in-training validation for real-time quality monitoring, and standalone evaluation for detailed model assessment. The system is designed to catch issues early, prevent regressions, and provide clear metrics for model quality decisions.

### Core Testing Philosophy

Our QA approach follows three fundamental principles:

1. **Early Detection**: Issues should be caught as early as possible in the development cycle
2. **Continuous Validation**: Model quality is monitored throughout the training process
3. **Platform Parity**: Tests ensure consistent behavior across Apple Silicon (MPS), NVIDIA (CUDA), and CPU

### Quality Assurance Tiers

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Readiness                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Tier 1: In-Training Validation                            │
│  ├── Automatic evaluation at epoch boundaries              │
│  ├── Real-time WER computation                            │
│  └── Best checkpoint selection                            │
│                                                              │
│  Tier 2: Standalone Evaluation                             │
│  ├── Post-training assessment                             │
│  ├── Cross-dataset evaluation                            │
│  └── Detailed metrics export                             │
│                                                              │
│  Tier 3: Automated Test Suite                             │
│  ├── Unit tests for components                           │
│  ├── Integration tests for workflows                     │
│  └── Platform-specific validation                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Testing Infrastructure

### Directory Structure

The `tests/` directory contains our comprehensive test suite organized by testing category:

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── data/                         # Test data fixtures
│   └── datasets/
│       └── test_streaming/       # Synthetic test dataset
│           ├── audio/
│           │   └── tiny.wav      # 1-second test audio
│           └── train.csv         # Minimal training data
├── output/                       # Test output artifacts
│   └── test_sft_workflow/        # Sample test outputs
│
├── test_smoke.py                 # Basic import validation
├── test_sft_workflow.py          # Standard fine-tuning integration
├── test_lora_workflow.py         # LoRA training validation
├── test_distillation_workflow.py # Knowledge distillation testing
├── test_mps.py                   # Apple Silicon MPS validation
├── test_config.py                # Configuration system tests
├── test_dataset.py               # Dataset processing tests
├── test_dataset_utils.py         # Dataset utility tests
├── test_runs.py                  # Run management tests
├── test_inference.py             # Inference pipeline tests
├── test_language_mode.py         # Language handling tests
├── test_quick_viz.py             # Visualization tests
└── test_visualizer.py            # Advanced visualization tests
```

### Test Execution Framework

The test suite uses pytest with specific configurations for the ML environment:

```python
# conftest.py ensures project imports work correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

## Test Suite Components

### 1. Smoke Tests (`test_smoke.py`)

**Purpose**: Rapid validation that core imports and basic functionality work.

**Coverage**:
- Module import validation
- Basic dependency checks
- Minimal data loading verification

**When to Run**: 
- Before any training run
- After dependency updates
- In CI/CD pipelines (first test)

**Example Execution**:
```bash
python tests/test_smoke.py
# Output: OK: basic imports
```

### 2. Integration Tests

#### Standard Fine-Tuning Workflow (`test_sft_workflow.py`)

**Purpose**: End-to-end validation of the complete supervised fine-tuning pipeline.

**Test Strategy**:
```python
# Generate synthetic 1-second audio sample
sample_rate = 16000
frequency = 440.0  # A4 note
duration = 1.0
audio = 0.1 * np.sin(2 * np.pi * frequency * np.linspace(0, duration, int(sample_rate * duration)))

# Create minimal dataset (1 sample)
# Run complete training pipeline
# Validate outputs exist
```

**Key Validations**:
- Dataset preparation and loading
- Model initialization
- Training loop execution (1 epoch, 1 sample)
- Checkpoint saving
- Results JSON generation
- MPS gradient checkpointing compatibility

**Platform-Specific Handling**:
- MPS: Gradient checkpointing disabled (double-backward issues)
- CUDA: Full feature testing
- CPU: Reduced batch sizes

#### LoRA Workflow (`test_lora_workflow.py`)

**Purpose**: Validate Parameter-Efficient Fine-Tuning with LoRA adapters.

**Coverage**:
- LoRA configuration validation
- Adapter weight saving (10-50MB vs 1GB full model)
- Base model freezing verification
- Inference with merged adapters

#### Distillation Workflow (`test_distillation_workflow.py`)

**Purpose**: Test knowledge distillation from teacher to student models.

**Coverage**:
- Teacher model loading
- Student model initialization
- Distillation loss computation
- Temperature scaling validation

### 3. Platform-Specific Tests (`test_mps.py`)

**Purpose**: Comprehensive Apple Silicon MPS validation and diagnostics.

**Test Categories**:

```python
def test_device_detection():
    """Verify MPS is detected as primary device"""
    device = get_device()
    assert device.type == "mps"  # On Apple Silicon
    
def test_model_loading():
    """Test Whisper model loads on MPS"""
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model = model.to("mps")
    assert next(model.parameters()).device.type == "mps"
    
def test_basic_operations():
    """Validate tensor operations on MPS"""
    x = torch.randn(1, 80, 3000).to("mps")  # Mel spectrogram shape
    conv = nn.Conv1d(80, 512, 3).to("mps")
    output = conv(x)
    torch.mps.synchronize()  # Ensure completion
    
def test_memory_management():
    """Test MPS memory operations"""
    initial = torch.mps.current_allocated_memory()
    large_tensor = torch.randn(1000, 1000).to("mps")
    torch.mps.empty_cache()
    # Verify memory management works
```

**Diagnostics Output**:
- Device capability report
- Memory configuration validation
- Operation compatibility matrix
- Performance baseline metrics

### 4. Component Tests

#### Configuration Testing (`test_config.py`)

**Coverage**:
- Hierarchical configuration loading
- Profile inheritance validation
- Type coercion testing
- Validation rule enforcement

#### Dataset Testing (`test_dataset.py`, `test_dataset_utils.py`)

**Coverage**:
- Dataset loading with patches
- Audio preprocessing pipeline
- Train/validation split generation
- Blacklist and override application

#### Run Management (`test_runs.py`)

**Coverage**:
- Thread-safe ID generation
- Metadata tracking
- Directory structure creation
- Status lifecycle management

## Quality Assurance Levels

### Level 1: In-Training Validation

The validation dataset serves as the primary quality metric during training, providing real-time feedback on model performance.

#### Configuration

In `config.ini`:
```ini
[profile:my-whisper-model]
validation_split = validation
evaluation_strategy = epoch      # Evaluate at each epoch
eval_steps = 500                # Alternative: evaluate every N steps
metric_for_best_model = wer     # Track Word Error Rate
load_best_model_at_end = true   # Save best checkpoint
save_strategy = epoch            # When to save checkpoints
```

#### Validation Process During Training

```python
def compute_metrics(eval_pred):
    """Called automatically by trainer at evaluation intervals"""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Decode predictions
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Decode labels
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute WER
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}
```

#### Training Output with Validation

```
Epoch 1/3
[████████████████████] 1000/1000 | Loss: 0.45

***** Running Evaluation *****
  Num examples = 500
  Batch size = 8
{'eval_loss': 0.2831, 'eval_wer': 0.1245, 'eval_runtime': 120.4, 'epoch': 1.0}

Saving checkpoint to output/1-my-profile/checkpoint-1000
New best model saved with WER: 0.1245
```

#### Best Model Selection

The trainer automatically tracks the best checkpoint based on validation WER:

1. After each evaluation, compare current WER to best so far
2. If current is better, save this checkpoint as best
3. At training end, load best checkpoint weights
4. Final model in output directory is the best, not just the last

### Level 2: Standalone Evaluation

Post-training evaluation provides detailed analysis and cross-dataset testing.

#### Command Structure

```bash
# Evaluate trained model on its validation set
whisper-tuner evaluate <profile_name>

# Evaluate on different dataset
whisper-tuner evaluate <profile_name> --dataset <other_dataset>

# Direct model evaluation
whisper-tuner evaluate <model_path> --dataset <dataset_name>
```

#### Evaluation Process

```python
def run_evaluation(profile_config, output_dir):
    """Complete evaluation pipeline"""
    
    # 1. Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # 2. Load evaluation dataset
    eval_dataset = load_dataset_split("validation", dataset_config)
    
    # 3. Run inference on all samples
    predictions = []
    references = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            pred_ids = model.generate(batch["input_features"])
            pred_text = processor.batch_decode(pred_ids)
            predictions.extend(pred_text)
            references.extend(batch["labels"])
    
    # 4. Compute metrics
    wer = compute_wer(predictions, references)
    
    # 5. Save results
    save_predictions_csv(predictions, references, output_dir)
    save_metrics_json({"wer": wer}, output_dir)
    
    return {"wer": wer}
```

#### Output Artifacts

```
output/{run_id}-{profile}/eval/
├── metadata.json        # Evaluation configuration
├── metrics.json         # WER, CER, BLEU scores
├── predictions.csv      # Detailed predictions
│   ├── audio_path      # Source audio file
│   ├── reference       # Ground truth transcription
│   ├── prediction      # Model output
│   └── wer            # Per-sample WER
└── evaluation.log      # Detailed logs
```

#### Metrics Computed

- **WER (Word Error Rate)**: Primary metric for speech recognition
- **CER (Character Error Rate)**: Character-level accuracy
- **BLEU**: For multilingual evaluation
- **Inference Speed**: Samples per second
- **Memory Usage**: Peak GPU/MPS memory

### Level 3: Automated Test Suite

The pytest suite provides comprehensive regression testing.

#### Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Verify pytest installation
pytest --version
```

#### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_sft_workflow.py

# Run with verbose output
pytest -v tests/

# Run specific test function
pytest tests/test_mps.py::test_device_detection

# Run tests matching pattern
pytest -k "mps" tests/

# Run with coverage report
pytest --cov=. tests/

# Platform-specific test selection
pytest -m "not cuda" tests/  # Skip CUDA tests on Mac
```

#### Test Markers and Categories

```python
# Mark platform-specific tests
@pytest.mark.mps
def test_mps_operations():
    """Only runs on Apple Silicon"""
    
@pytest.mark.cuda
def test_cuda_operations():
    """Only runs on NVIDIA GPUs"""
    
@pytest.mark.slow
def test_full_training():
    """Long-running integration test"""
    
@pytest.mark.smoke
def test_basic_import():
    """Quick smoke test"""
```

## Quality Gates

### Training Quality Gates

1. **Validation WER Threshold**
   - Must achieve < 30% WER on validation set
   - Training stops early if WER plateaus for 3 epochs
   - Best checkpoint must show improvement over baseline

2. **Training Stability**
   - Loss must decrease consistently
   - No NaN/Inf values in gradients
   - Memory usage stays within limits

### Post-Training Gates

1. **Evaluation Requirements**
   - Final model WER < configured threshold
   - Consistent performance across data subsets
   - No degradation on edge cases

2. **Export Validation**
   - Model exports successfully to GGUF
   - Exported model produces identical outputs
   - File sizes within expected ranges

### Test Suite Gates

1. **Smoke Tests**: Must pass before any commit
2. **Integration Tests**: Must pass before merge
3. **Platform Tests**: Must pass on target platform
4. **Performance Tests**: No regression from baseline

## Best Practices

### 1. Validation Dataset Curation

**Requirements**:
- **Size**: Minimum 1-2 hours of audio or 500+ samples
- **Quality**: Human-verified transcriptions
- **Diversity**: Representative of production data
- **Separation**: No overlap with training data

**Structure**:
```csv
audio_path,text,duration,speaker_id,language
data/audio/sample1.wav,"Hello world",2.5,speaker_001,en
data/audio/sample2.wav,"Bonjour monde",2.3,speaker_002,fr
```

### 2. Test Data Preparation

For integration tests, use synthetic data:
```python
# Generate consistent test audio
def create_test_audio():
    """Generate 1-second sine wave for testing"""
    sample_rate = 16000
    frequency = 440.0  # A4 note
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.1 * np.sin(2 * np.pi * frequency * t)
    return audio
```

### 3. Continuous Monitoring

During training:
- Watch validation WER trends
- Monitor memory usage on MPS
- Check for gradient explosions
- Track checkpoint sizes

### 4. Failure Investigation

When tests fail:

1. **Check Platform**:
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **Verify Environment**:
   ```bash
   whisper-tuner system-check
   ```

3. **Run Minimal Test**:
   ```bash
   python tests/test_smoke.py
   ```

4. **Check Logs**:
   - Training logs in `output/{run_id}/run.log`
   - Evaluation logs in `output/{run_id}/eval/evaluation.log`
   - Test output with `pytest -v`

### 5. Performance Baselines

Maintain performance baselines for regression detection:

| Metric | MPS (M1) | CUDA (3090) | CPU |
|--------|----------|-------------|-----|
| Training Speed | 50 samples/sec | 150 samples/sec | 5 samples/sec |
| Evaluation Speed | 20 samples/sec | 60 samples/sec | 2 samples/sec |
| Memory Usage | 8GB unified | 12GB VRAM | 16GB RAM |
| Checkpoint Size | 1.2GB | 1.2GB | 1.2GB |
| LoRA Adapter | 45MB | 45MB | 45MB |

## Testing Checklist

### Pre-Training Checklist
- [ ] Run smoke tests: `python tests/test_smoke.py`
- [ ] Verify device: `python tests/test_mps.py` (on Mac)
- [ ] Check dataset: `whisper-tuner prepare <dataset>`
- [ ] Validate config: Review `config.ini` profile
- [ ] Test single batch: Run with `--max_samples 10`

### During Training Checklist
- [ ] Monitor validation WER trends
- [ ] Check memory usage doesn't exceed 80%
- [ ] Verify checkpoints are being saved
- [ ] Confirm no NaN losses
- [ ] Watch for early stopping triggers

### Post-Training Checklist
- [ ] Run standalone evaluation
- [ ] Export to GGUF format
- [ ] Compare WER to baseline
- [ ] Review predictions.csv for patterns
- [ ] Test inference speed

### Pre-Deployment Checklist
- [ ] All pytest tests pass
- [ ] WER meets production threshold
- [ ] Model exports successfully
- [ ] Performance within requirements
- [ ] Documentation updated

## Troubleshooting Guide

### Common Test Failures

#### 1. MPS Not Available
```
Error: MPS not available - check macOS 12.3+ and Apple Silicon hardware
```
**Solution**: 
- Verify ARM64 Python: `python -c "import platform; print(platform.machine())"`
- Check PyTorch MPS support: `pip show torch | grep Version`
- Reinstall with: `pip install torch torchvision torchaudio`

#### 2. Import Errors in Tests
```
ModuleNotFoundError: No module named 'main'
```
**Solution**:
- Ensure running from project root
- Check PYTHONPATH includes project directory
- Verify conftest.py is present

#### 3. Validation Dataset Not Found
```
FileNotFoundError: validation.csv not found
```
**Solution**:
- Run data preparation: `whisper-tuner prepare <dataset>`
- Check data/datasets/<dataset>/ directory
- Verify train/validation split was created

#### 4. Out of Memory During Testing
```
RuntimeError: MPS backend out of memory
```
**Solution**:
- Set memory limit: `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`
- Reduce batch size in test configuration
- Clear cache: `torch.mps.empty_cache()`

#### 5. Slow Test Execution
**Solution**:
- Run only smoke tests for quick validation
- Use pytest markers to skip slow tests
- Consider using smaller test models (whisper-tiny)

## Conclusion

This Testing & QA Protocol ensures that every model produced by the Whisper Fine-Tuner meets quality standards through multiple validation layers. The combination of automated testing, in-training validation, and standalone evaluation provides comprehensive coverage from development through deployment. Following these protocols guarantees that models are not only functional but optimized for their target platforms and use cases.