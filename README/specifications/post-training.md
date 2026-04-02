# Post-Training Model Management Product Specification

**Version**: 1.0  
**Last Updated**: January 2025  
**Status**: Production

## Executive Summary

The Whisper Fine-Tuner macOS post-training system provides a comprehensive, automated model management workflow that handles everything from the moment training completes to deployment-ready model availability. The system automatically organizes, converts, and tracks all trained models with zero manual intervention required, while still providing powerful manual export options for advanced users.

### Key Features
- **Automatic Model Organization**: Sequential run ID system with structured directories
- **Multi-Format Export**: Automatic GGUF conversion, optional CoreML/ANE optimization
- **Intelligent Model Discovery**: Find and access models by profile, status, or recency
- **Platform-Optimized Storage**: HuggingFace SafeTensors format with device-specific optimizations
- **LoRA-Aware Management**: Smart handling of adapter-only vs full model storage

### User Benefits
- **Zero Configuration**: Models are automatically saved, converted, and organized
- **Instant Access**: Clickable file:// URIs for immediate model access
- **Deployment Ready**: Automatic GGUF export for whisper.cpp compatibility
- **Space Efficient**: LoRA models store only adapters (10-50MB vs 1GB+)
- **Full Traceability**: Complete metadata tracking for every training run

## Model Storage Architecture

### Directory Structure

Every training run creates a structured directory following this pattern:

```
output/
├── next_run_id.txt              # Sequential ID counter (atomic)
├── next_run_id.txt.lock         # File lock for thread safety
├── 1-whisper-base-librispeech/  # Run directory: {id}-{profile}
│   ├── metadata.json            # Complete run metadata
│   ├── metrics.json             # Training metrics
│   ├── train_results.json       # Detailed training results
│   ├── run.log                  # Training logs
│   ├── completed                # Success marker file
│   ├── config.json              # Model configuration
│   ├── model.safetensors        # Model weights (SafeTensors format)
│   ├── tokenizer.json           # Tokenizer configuration
│   ├── processor_config.json    # Audio processor settings
│   ├── generation_config.json   # Generation parameters
│   ├── checkpoint-500/          # Intermediate checkpoints
│   │   └── [full model files]
│   ├── model-f16.gguf           # Auto-generated GGUF export
│   └── eval/                    # Evaluation results (if run)
│       ├── metadata.json
│       ├── metrics.json
│       └── predictions.csv
```

### LoRA Model Structure

LoRA models have a different structure to save space:

```
output/
└── 3-whisper-lora-custom/
    ├── metadata.json
    ├── adapter_model/           # LoRA adapter directory
    │   ├── adapter_config.json  # LoRA configuration
    │   └── adapter_model.safetensors  # Adapter weights (10-50MB)
    ├── model-f16.gguf           # Full model with merged LoRA
    └── completed
```

### Run ID System

- **Sequential IDs**: Starting from 1, incrementing for each new run
- **Thread-Safe**: FileLock ensures atomic ID generation
- **Persistent Counter**: Survives application restarts
- **Profile Integration**: ID combined with profile name for unique directory

### Metadata Tracking

Every run includes comprehensive metadata in `metadata.json`:

```json
{
    "run_id": 1,
    "run_type": "finetuning",
    "status": "completed",
    "start_time": "2024-01-15 10:30:00",
    "end_time": "2024-01-15 14:30:00",
    "profile": "whisper-base-librispeech",
    "model": "openai/whisper-base",
    "dataset": "librispeech",
    "base_model": "openai/whisper-base",
    "config": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "learning_rate": 5e-5,
        // ... full training configuration
    },
    "metrics": {
        "train_loss": 0.234,
        "train_wer": 0.089,
        "duration_sec": 14400
    }
}
```

## Model Formats and Conversions

### Primary Format: HuggingFace SafeTensors

**What Gets Saved**:
- `model.safetensors`: Model weights in SafeTensors format (secure, fast loading)
- `config.json`: Model architecture and configuration
- `tokenizer.json`: Fast tokenizer with vocabulary
- `processor_config.json`: Audio preprocessing parameters
- `generation_config.json`: Decoding parameters (beam size, temperature, etc.)

**Why SafeTensors**:
- **Security**: Prevents arbitrary code execution (unlike pickle)
- **Speed**: 2-10x faster loading than PyTorch native format
- **Compatibility**: Supported by all major ML frameworks
- **Inspection**: Metadata readable without loading tensors

### Automatic GGUF Conversion

**When It Happens**: Immediately after successful training completion

**What Is GGUF**: 
- GPT-Generated Unified Format for whisper.cpp
- Optimized for CPU/GPU inference
- 10-50x faster than PyTorch for inference
- Supports quantization (int8, int4) for further optimization

**Conversion Process**:
1. Training completes successfully
2. System auto-downloads whisper.cpp converter (first time only)
3. Detects model type (multilingual vs English-only)
4. Converts to FP16 GGUF format
5. Saves as `model-f16.gguf` in run directory
6. Generates clickable file:// URI for user

**Fallback Behavior**: If conversion fails, training still succeeds with warning message

### Optional Export: CoreML (Hybrid ANE)

**Command**: `python -m whisper_tuner.scripts.export_coreml <run_dir>`

**What It Creates**:
- `coreml/whisper-encoder.mlmodelc`: ANE-optimized encoder
- `model-f16.gguf`: CPU/GPU decoder (reuses GGUF)
- `coreml/export_manifest.json`: Export metadata

**Why Hybrid**:
- Encoder benefits from Apple Neural Engine (5-10x faster)
- Decoder better suited for CPU/GPU (sequential generation)
- Optimal resource utilization on Apple Silicon

### Standard HuggingFace Export

**Command**: `whisper-tuner export <profile_or_path>`

**Purpose**: Creates a clean HuggingFace model directory for:
- Uploading to HuggingFace Hub
- Sharing with others
- Integration with other tools

## Post-Training Workflow

### Automatic Steps (Zero User Intervention)

1. **Training Completion** (< 1 second)
   - Model saved via `trainer.save_model()`
   - Processor and tokenizer saved
   - Training results JSON generated

2. **Status Update** (< 1 second)
   - `completed` marker file created
   - `metadata.json` status set to "completed"
   - End time recorded

3. **Metrics Persistence** (< 1 second)
   - Training metrics extracted from results
   - Duration calculated and recorded
   - `metrics.json` file created

4. **User Notification** (< 1 second)
   - Success message displayed
   - Clickable file:// URI generated
   - Platform-specific URI formatting (macOS/Linux vs Windows)

5. **GGUF Auto-Export** (30 seconds - 5 minutes)
   - whisper.cpp tools acquired if needed
   - Model converted to GGUF format
   - Second clickable URI displayed for GGUF file

### Success Indicators

Users know training succeeded when they see:

```
✅ Training completed successfully!
Model saved in: file:///Users/you/whisper-fine-tuner/output/1-whisper-base-custom
📦 GGUF exported for whisper.cpp: file:///Users/you/whisper-fine-tuner/output/1-whisper-base-custom/model-f16.gguf
```

### Failure Handling

Failed runs are clearly marked:

```json
{
    "status": "failed",
    "error_message": "CUDA out of memory",
    "end_time": "2024-01-15 11:45:23"
}
```

- No `completed` marker file
- Partial checkpoints may exist
- Logs available for debugging

## Model Discovery and Access

### Finding Models by Profile

**Latest Completed Run**:
```python
from whisper_tuner.core.runs import find_latest_completed_finetuning_run
model_path = find_latest_completed_finetuning_run("output", "whisper-base")
```

**All Runs for Profile**:
```bash
ls -la output/*-whisper-base-*/
```

### Finding Models by Status

**Completed Runs**:
```bash
find output -name "completed" -type f | xargs dirname
```

**Failed Runs** (have metadata but no completed marker):
```bash
for dir in output/*/; do
    if [[ -f "$dir/metadata.json" && ! -f "$dir/completed" ]]; then
        echo "$dir"
    fi
done
```

### Accessing Models

**Direct File Access**:
- Click the file:// URI in terminal output
- Navigate to `output/{run_id}-{profile}/`
- Models are immediately usable with HuggingFace transformers

**Programmatic Access**:
```python
from transformers import WhisperForConditionalGeneration

# Load from run directory
model = WhisperForConditionalGeneration.from_pretrained(
    "output/1-whisper-base-custom"
)

# Load GGUF for whisper.cpp
import whisper_cpp
model = whisper_cpp.load_model("output/1-whisper-base-custom/model-f16.gguf")
```

## Export Operations

### Manual Export Commands

**Export to GGUF** (if auto-export was skipped):
```bash
python -m whisper_tuner.scripts.export_gguf output/1-whisper-base-custom
```

**Export to CoreML** (for iOS/macOS apps):
```bash
python -m whisper_tuner.scripts.export_coreml output/1-whisper-base-custom
```

**Export to Standard HF Format**:
```bash
whisper-tuner export output/1-whisper-base-custom
```

### Platform-Specific Optimizations

**Apple Silicon (MPS)**:
- Models saved in FP16 when possible
- CoreML export targets Apple Neural Engine
- Unified memory architecture considerations

**NVIDIA CUDA**:
- FP16 storage for GPU efficiency  
- Automatic mixed precision compatibility
- Multi-GPU checkpoint handling

**CPU**:
- FP32 storage for maximum compatibility
- GGUF quantization options (int8, int4)
- Optimized for inference libraries

## Technical Implementation

### Storage Requirements

**Standard Fine-tuned Model**:
- Tiny: ~39MB
- Base: ~145MB  
- Small: ~488MB
- Medium: ~1.5GB
- Large: ~3.1GB

**LoRA Model** (adapters only):
- All sizes: 10-50MB
- Plus base model reference

**With GGUF Export** (additional):
- Same size as PyTorch model (FP16)
- Optional quantized versions (25-75% smaller)

### Performance Characteristics

**Save Operations**:
- Model saving: 1-10 seconds
- GGUF conversion: 30 seconds - 5 minutes
- CoreML export: 1-5 minutes

**Load Operations**:
- SafeTensors: 100-500ms
- GGUF: 50-200ms
- CoreML: 200-1000ms (first load)

### Error Recovery

**Interrupted Training**:
- Checkpoint directories preserve progress
- Can resume from checkpoint-N directories
- Metadata marks as "cancelled" on SIGINT/SIGTERM

**Failed GGUF Conversion**:
- Training still marked successful
- Manual export available
- Error logged but non-fatal

**Corrupted Metadata**:
- Fallback to marker files
- Directory structure still valid
- Models remain loadable

## User Journey Examples

### Standard Training → Deploy

1. **Start Training**:
   ```bash
   whisper-tuner finetune whisper-base-meeting-audio
   ```

2. **Training Completes**:
   - See success message with file:// URI
   - See GGUF export confirmation

3. **Access Model**:
   - Click URI to open in Finder/Explorer
   - Find `model-f16.gguf` for whisper.cpp
   - Find `model.safetensors` for Python

4. **Deploy**:
   - Copy GGUF to production server
   - OR upload to HuggingFace Hub
   - OR integrate into application

### LoRA Training → Merge → Deploy

1. **Train LoRA Adapter**:
   ```bash
   whisper-tuner finetune whisper-lora-efficient
   ```

2. **Adapter Saved** (10-50MB):
   - `adapter_model/` directory created
   - Base model reference in metadata

3. **Auto-Merge for GGUF**:
   - System automatically merges LoRA + base
   - Creates full `model-f16.gguf`

4. **Optional Full Export**:
   ```bash
   whisper-tuner export output/3-whisper-lora-efficient
   ```

### Failed Run Recovery

1. **Training Fails** (OOM, error, etc.):
   - Check `output/N-profile/run.log`
   - Find last checkpoint: `checkpoint-*/`

2. **Resume from Checkpoint**:
   ```bash
   whisper-tuner finetune whisper-base-custom \
     --resume-from output/2-whisper-base-custom/checkpoint-500
   ```

3. **Or Start Fresh**:
   - New run ID assigned automatically
   - Failed run preserved for analysis

## Best Practices

### For Users

1. **Always Use Profiles**: Consistent naming helps find models later
2. **Check Completion Status**: Look for ✅ and file:// URIs
3. **Keep Output Directory Clean**: Archive old runs periodically
4. **Use GGUF for Inference**: Much faster than PyTorch models

### For Developers

1. **Respect Run Structure**: Don't modify run directories manually
2. **Use Metadata APIs**: Access via `core.runs` functions
3. **Handle Missing GGUF**: Not all runs have GGUF (export may fail)
4. **Check Model Type**: LoRA vs standard affects loading code

## Future Enhancements

### Planned Features

1. **Automatic HuggingFace Hub Upload**: Optional push to Hub on completion
2. **Model Quantization**: Automatic int8/int4 quantization options
3. **Distributed Training**: Multi-GPU checkpoint consolidation
4. **Model Versioning**: Git-like version control for models
5. **Automatic Benchmarking**: Post-training speed/quality metrics

### Under Consideration

1. **Cloud Backup**: Automatic S3/GCS backup of completed runs
2. **Model Registry**: Central database of all trained models
3. **A/B Testing**: Built-in model comparison framework
4. **Deployment Pipelines**: Direct deploy to production services
5. **Model Optimization**: Automatic pruning and optimization

## Appendix: File Formats

### SafeTensors Format
- **Extension**: `.safetensors`
- **Structure**: Header (JSON) + Tensor data (binary)
- **Advantages**: Safe, fast, framework-agnostic

### GGUF Format
- **Extension**: `.gguf`
- **Structure**: Header + Metadata + Tensor data
- **Advantages**: Optimized for inference, quantization support

### CoreML Format
- **Extension**: `.mlmodelc` (compiled) or `.mlpackage`
- **Structure**: Protobuf model + weights + metadata
- **Advantages**: Apple Neural Engine optimization

---

*This specification is maintained by the Whisper Fine-Tuner macOS team and updated with each major release.*
