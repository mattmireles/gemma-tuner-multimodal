# Whisper Fine-Tuner System Architecture

## Executive Summary

The Whisper Fine-Tuner is a comprehensive training framework designed to fine-tune OpenAI's Whisper speech recognition models with a focus on Apple Silicon optimization. The system implements three training methodologies (Standard Fine-Tuning, LoRA Parameter-Efficient Fine-Tuning, and Knowledge Distillation) while providing a unified interface for data preparation, training, evaluation, and model export.

### Core Design Principles

1. **Simplicity Over Complexity**: The system prioritizes straightforward solutions that work reliably across different hardware configurations
2. **Apple Silicon First**: Optimized for Metal Performance Shaders (MPS) while maintaining CUDA and CPU compatibility
3. **Progressive Disclosure**: Complex features are hidden behind simple interfaces (wizard mode for beginners, CLI for experts)
4. **Modular Architecture**: Clear separation between configuration, execution, and model-specific implementations
5. **Zero Configuration**: Smart defaults enable immediate usage while allowing deep customization when needed

### Key Capabilities

- **Multi-Method Training**: Standard fine-tuning, LoRA adapters, and knowledge distillation
- **Cross-Platform**: Seamless operation across Apple Silicon (MPS), NVIDIA (CUDA), and CPU
- **Comprehensive Pipeline**: End-to-end workflow from raw audio to deployable models
- **Production Ready**: Robust error handling, run management, and metadata tracking
- **Export Flexibility**: Multiple output formats including GGUF for whisper.cpp deployment

## System Architecture

```mermaid
graph TB
    %% User Entry Points
    User[User] --> Wizard[wizard.py<br/>Interactive CLI]
    User --> CLI[main.py<br/>Command Line]
    
    %% Configuration Layer
    Wizard --> |generates| TempConfig[Temporary Config]
    TempConfig --> |subprocess| CLI
    CLI --> ConfigLoader[core/config.py<br/>Config Loader]
    ConfigINI[config.ini] --> ConfigLoader
    
    %% Device Detection
    CLI --> DeviceDetect[utils/device.py<br/>Device Detection]
    DeviceDetect --> |MPS/CUDA/CPU| DeviceConfig[Device Config]
    
    %% Run Management
    CLI --> RunManager[core/runs.py<br/>Run Manager]
    RunManager --> |creates| RunDir[output/{run_id}/]
    RunManager --> |tracks| Metadata[metadata.json]
    
    %% Operation Dispatch
    ConfigLoader --> |merged config| OpDispatch[core/ops.py<br/>Operation Dispatch]
    DeviceConfig --> OpDispatch
    
    %% Operations
    OpDispatch --> |lazy load| PrepareOp[scripts/prepare_data.py]
    OpDispatch --> |lazy load| FinetuneOp[scripts/finetune.py]
    OpDispatch --> |lazy load| EvaluateOp[scripts/evaluate.py]
    OpDispatch --> |lazy load| ExportOp[scripts/export.py]
    OpDispatch --> |lazy load| BlacklistOp[scripts/blacklist.py]
    
    %% Specialized Utilities
    CLI --> |validation| GemmaPreflight[scripts/gemma_preflight.py<br/>Environment Validation]
    CLI --> |profiling| GemmaProfiler[scripts/gemma_profiler.py<br/>Performance Analysis]
    %% Model Implementations
    FinetuneOp --> |routes to| WhisperModel[models/whisper/finetune.py<br/>Standard Fine-Tuning]
    FinetuneOp --> |routes to| LoRAModel[models/whisper_lora/finetune.py<br/>LoRA Fine-Tuning]
    FinetuneOp --> |routes to| DistilModel[models/distil_whisper/finetune.py<br/>Knowledge Distillation]
    FinetuneOp --> |routes to| GemmaModel[models/gemma/finetune.py<br/>Gemma 3n Multimodal]
    
    %% Data Flow
    PrepareOp --> DataDir[data/datasets/]
    DataDir --> |loads| FinetuneOp
    
    %% Output Flow
    WhisperModel --> |saves| Checkpoints[Checkpoints]
    LoRAModel --> |saves| Adapters[LoRA Adapters]
    DistilModel --> |saves| StudentModel[Student Model]
    
    Checkpoints --> RunDir
    Adapters --> RunDir
    StudentModel --> RunDir
    
    %% Evaluation Flow
    RunDir --> |loads model| EvaluateOp
    EvaluateOp --> |metrics| EvalResults[eval/metrics.json]
    
    %% Export Flow
    RunDir --> |converts| ExportOp
    ExportOp --> |GGUF| WhisperCpp[whisper.cpp format]
    
    style User fill:#e1f5fe
    style Wizard fill:#fff3e0
    style CLI fill:#fff3e0
    style ConfigLoader fill:#f3e5f5
    style DeviceDetect fill:#e8f5e9
    style RunManager fill:#fce4ec
    style OpDispatch fill:#fff9c4
```

## Technical Architecture

### Component Deep Dive

## 1. CLI & Wizard System

The system provides two user interfaces: a direct CLI for experienced users and an interactive wizard for guided setup.

### Command Line Interface (main.py)

The CLI serves as the primary entry point, implementing a hierarchical command structure:

```
python main.py <operation> <profile_or_target> [options]
```

**Command Routing Architecture:**

1. **Early MPS Configuration** (lines 6-33)
   - Memory limits set BEFORE PyTorch import (critical for Apple Silicon)
   - Validates and clamps PYTORCH_MPS_HIGH_WATERMARK_RATIO environment variable
   - Prevents system-wide memory pressure on unified memory architecture

2. **Argument Parsing** (lines 188-216)
   - Operations: prepare, finetune, evaluate, export, export-gguf, blacklist
   - Profile-based: Uses named profiles from config.ini
   - Direct specification: model+dataset combinations
   - Optional overrides: --max_samples, --dataset, --split

3. **Operation Dispatch** (lines 226-530)
   - Each operation creates appropriate run directories
   - Registers signal handlers for graceful interruption
   - Updates run metadata throughout lifecycle
   - Automatic GGUF export after successful training

**Key Design Decision:** The CLI handles all orchestration logic, keeping operation implementations focused on their core functionality.

### Interactive Wizard (wizard.py)

The wizard provides a Steve Jobs-inspired progressive disclosure interface:

**Wizard Flow:**

1. **Hardware Detection** (uses utils/device.py)
   - Detects available compute device
   - Estimates training time based on hardware
   - Suggests optimal batch sizes

2. **Progressive Questions:**
   - Model selection (base → size → variant)
   - Training method (SFT, LoRA, Distillation)
   - Dataset selection or creation
   - Hyperparameter configuration with smart defaults

3. **Configuration Generation:**
   - Creates temporary config file in temp_configs/
   - Generates unique profile name with timestamp
   - Builds complete training command

4. **Execution:**
   - Launches main.py via subprocess
   - Streams output with beautiful formatting
   - Provides real-time progress feedback

**Why Subprocess?** The wizard generates a temporary configuration and executes main.py as a subprocess to maintain clean separation between UI and execution logic. This allows the wizard to be completely optional while leveraging all existing infrastructure.

## 2. Configuration System (core/config.py)

The configuration system implements a sophisticated hierarchical inheritance model that balances flexibility with simplicity.

### Hierarchical Configuration Loading

Configuration values cascade through six levels, with each level overriding previous values:

```
1. [DEFAULT] section (global defaults)
   ↓
2. [dataset_defaults] section (dataset processing defaults)
   ↓
3. [group:{name}] section (model family defaults, e.g., whisper, distil-whisper)
   ↓
4. [model:{name}] section (specific model configuration)
   ↓
5. [dataset:{name}] section (specific dataset configuration)
   ↓
6. [profile:{name}] section (highest precedence, training profile overrides)
```

**Example Configuration Flow:**

Profile request: "whisper-base-librispeech"

1. Load DEFAULT: `learning_rate=1e-5, epochs=3`
2. Apply dataset_defaults: `max_duration=30.0, text_column="text"`
3. Apply group:whisper: `dtype="float32", attn_implementation="sdpa"`
4. Apply model:whisper-base: `base_model="openai/whisper-base", batch_size=16`
5. Apply dataset:librispeech: `train_split="train.clean.100", validation_split="validation.clean"`
6. Apply profile:whisper-base-librispeech: `learning_rate=5e-5` (overrides DEFAULT)

Final config: Complete merged configuration with profile overrides taking precedence.

### Configuration Validation

The `_validate_profile_config()` function performs three validation phases:

1. **Type Coercion** (lines 425-465)
   - Converts INI string values to proper Python types
   - INT_COERCION_KEYS: batch_size, epochs, steps, etc.
   - FLOAT_COERCION_KEYS: learning_rate, dropout, temperature, etc.
   - BOOL_COERCION_KEYS: gradient_checkpointing, fp16, etc.
   - LIST_COERCION_MAPPING: comma-separated values for target_modules, languages

2. **Structural Validation** (lines 467-470)
   - Ensures required keys are present
   - Different requirements for profile vs model+dataset modes

3. **Semantic Validation** (lines 472-513)
   - Range checks: learning_rate ∈ [1e-7, 1.0]
   - Logical checks: max_duration > 0, batch_size ≥ 1
   - LoRA validation: rank > 0, dropout ∈ [0, 1)

**Why This Complexity?** The hierarchical system enables users to define common settings once (in group or model sections) while allowing fine-grained overrides for specific experiments. This dramatically reduces configuration duplication.

## 3. Run Management System (core/runs.py)

The run management system provides comprehensive tracking and organization for all training and evaluation operations.

### Run ID Generation

Thread-safe sequential ID generation using file locking:

```python
def get_next_run_id(output_dir: str) -> int:
    lock = FileLock("next_run_id.txt.lock", timeout=10)
    with lock:
        # Read current ID
        # Write incremented ID
        # Return current ID
```

**Why File Locking?** Multiple training processes may start simultaneously (batch training, parallel experiments). File locking ensures atomic ID generation without race conditions.

### Directory Structure

```
output/
├── next_run_id.txt                    # Sequential counter
├── next_run_id.txt.lock               # Lock file for atomic access
├── 1-whisper-base-custom/             # Training run
│   ├── metadata.json                  # Run configuration and status
│   ├── metrics.json                   # Training/evaluation metrics
│   ├── completed                      # Completion marker
│   ├── checkpoint-500/                # Model checkpoints
│   ├── eval/                         # Profile-based evaluation
│   │   ├── metadata.json
│   │   ├── metrics.json
│   │   └── predictions.csv
│   └── eval-other-dataset/           # Cross-dataset evaluation
│       └── ...
└── whisper-base+librispeech/         # Direct model+dataset evaluation
    └── eval/
        └── ...
```

### Metadata Tracking

Each run maintains comprehensive metadata in JSON format:

```json
{
    "run_id": 1,
    "run_type": "finetuning",
    "status": "running|completed|failed|cancelled",
    "start_time": "2024-01-15 10:30:00",
    "end_time": "2024-01-15 14:30:00",
    "profile": "whisper-base-custom",
    "model": "whisper-base",
    "dataset": "custom",
    "config": { /* full merged configuration */ },
    "metrics": { /* final training/eval metrics */ },
    "finetuning_run_id": null,  /* links eval to training run */
    "run_dir": "/absolute/path/to/run"
}
```

### Run Lifecycle Management

1. **Creation**: `create_run_directory()` establishes structure
2. **Updates**: `update_run_metadata()` tracks progress
3. **Completion**: `mark_run_as_completed()` sets final status
4. **Discovery**: `find_latest_finetuning_run()` locates models for evaluation

**Key Innovation:** The dual completion tracking (marker file + metadata status) ensures backward compatibility while providing rich status information.

## 4. Device Management System (utils/device.py)

The device management system handles the critical differences between Apple Silicon, NVIDIA, and CPU platforms.

### Device Detection Hierarchy

```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")     # NVIDIA
    else:
        return torch.device("cpu")      # Fallback
```

**Priority Reasoning:**
1. **MPS First**: Optimized for Apple Silicon development environment
2. **CUDA Second**: High-performance alternative for NVIDIA systems
3. **CPU Fallback**: Universal compatibility

### Platform-Specific Memory Management

#### Apple Silicon (MPS)
- **Architecture**: Unified memory (CPU and GPU share physical RAM)
- **Management**: Environment variable `PYTORCH_MPS_HIGH_WATERMARK_RATIO`
- **Strategy**: Conservative limits (0.8) prevent system-wide swapping
- **Critical**: Must be set BEFORE PyTorch import

#### NVIDIA (CUDA)
- **Architecture**: Discrete GPU memory separate from system RAM
- **Management**: `torch.cuda.set_per_process_memory_fraction()`
- **Strategy**: Aggressive limits (0.9) maximize GPU utilization
- **Flexibility**: Can be adjusted at runtime

#### CPU
- **Architecture**: System memory with OS virtual memory management
- **Management**: No explicit limits needed
- **Strategy**: Relies on OS memory management

### Device-Specific Configuration Normalization

The `apply_device_defaults()` function ensures training compatibility:

```python
if device.type == "mps":
    config["dtype"] = "float32"           # MPS doesn't support float64
    config["attn_implementation"] = "eager"  # Maximum compatibility
elif device.type == "cpu":
    config.setdefault("dtype", "float32")
    config.setdefault("attn_implementation", "eager")
# CUDA: No forced overrides (maximum flexibility)
```

**Why Force Settings?** MPS has limited operation coverage compared to CUDA. Forcing compatible settings prevents runtime failures that would occur with unsupported dtypes or attention implementations.

### Impact on Training Process

1. **Batch Size Selection**
   - MPS: Start with 32-64 (memory bandwidth limited)
   - CUDA: Maximize based on VRAM
   - CPU: Conservative (4-8) to avoid thrashing

2. **Mixed Precision**
   - MPS: Limited support, usually disabled
   - CUDA: Full support for fp16/bf16
   - CPU: Generally disabled (no benefit)

3. **Operation Coverage**
   - MPS: ~300 operations, CPU fallback for unsupported
   - CUDA: ~2000+ operations, comprehensive coverage
   - CPU: Universal support, but slow

## 5. Operation Dispatch System (core/ops.py)

The operation dispatch system implements a lazy-loading pattern that dramatically reduces startup time.

### Lazy Loading Architecture

```python
def finetune(profile_config: Dict, output_dir: str):
    # Import only when operation is called
    from scripts.finetune import main as finetune_main
    finetune_main(profile_config, output_dir)
```

**Performance Impact:**
- Without lazy loading: ~2000ms startup (loading torch, transformers, etc.)
- With lazy loading: ~50ms startup
- First operation: Includes one-time import cost
- Subsequent operations: Use Python's module cache

### Operation Categories

1. **Data Operations**
   - `prepare()`: Dataset download and preprocessing
   - `blacklist()`: Problematic sample identification

2. **Training Operations**
   - `finetune()`: Model training dispatch

3. **Evaluation Operations**
   - `evaluate()`: WER/CER metric computation

4. **Export Operations**
   - `export()`: HuggingFace format export
   - `export_gguf()`: whisper.cpp format conversion

### Model Type Routing

The finetune operation includes intelligent model routing:

```python
# In scripts/finetune.py
if has_lora_config(profile_config):
    from models.whisper_lora.finetune import main
elif has_teacher_model(profile_config):
    from models.distil_whisper.finetune import main
else:
    from models.whisper.finetune import main

main(profile_config, output_dir)
```

**Detection Logic:**
- LoRA: Presence of `lora_r`, `lora_alpha` parameters
- Distillation: Presence of `teacher_model` parameter
- Standard: Default fallback

### Error Boundaries

Each operation implements comprehensive error handling:

1. **Import Errors**: Missing dependencies
2. **Configuration Errors**: Invalid parameters
3. **Runtime Errors**: OOM, file not found
4. **Interruption**: Graceful shutdown with status update

**Design Philosophy:** Operations are isolated and atomic. Failures in one operation don't affect others, and all operations leave the system in a consistent state.

## Data Flow

### Training Pipeline

```
1. Data Preparation
   CSV → Audio Download → WAV Conversion → Train/Val Splits

2. Configuration Resolution
   Profile → Hierarchical Merge → Validation → Device Normalization

3. Training Execution
   Model Loading → Dataset Loading → Training Loop → Checkpoint Saving

4. Automatic Export
   Training Complete → GGUF Conversion → Deployment Ready
```

### Evaluation Pipeline

```
1. Model Discovery
   Profile → Latest Run → Model Path
   OR
   Model+Dataset → Direct Load

2. Evaluation Execution
   Model Loading → Dataset Loading → Inference → Metric Calculation

3. Results Persistence
   Predictions CSV → Metrics JSON → Metadata Update
```

## Enhanced System Components

### Specialized Utilities

The system includes several specialized utilities that extend core functionality for specific use cases:

#### Environment Validation (`scripts/gemma_preflight.py`)
**Purpose**: Comprehensive environment validation for Gemma 3n training on Apple Silicon.

**Key Features**:
- Python architecture validation (native ARM64 vs Rosetta emulation detection)
- PyTorch MPS backend availability and compatibility verification
- Hardware-specific dtype support testing (bfloat16 validation)
- Memory configuration guidance with performance impact analysis
- Actionable remediation steps for common issues

**Integration**: Called before Gemma training workflows to prevent environment-related failures.

#### Performance Profiling (`scripts/gemma_profiler.py`)
**Purpose**: Systematic performance analysis and resource usage measurement for Gemma 3n models.

**Key Features**:
- Model loading with memory-optimized configuration
- Synthetic workload generation for consistent benchmarking
- Device optimization testing (MPS, CUDA, CPU)
- Memory consumption tracking and performance measurement
- Structured metrics output for analysis and planning

**Integration**: Used for performance baseline establishment and hardware evaluation.

### AI-First Documentation Patterns

The system implements comprehensive AI-first documentation standards designed for AI developer maintainability:

#### Documentation Philosophy
**Core Principle**: Every comment is a prompt for the next AI developer. Documentation must provide complete context for understanding, modifying, and extending code without human intervention.

#### Implementation Standards

**1. Comprehensive File Headers**:
- Complete architecture overview for each module
- Detailed responsibility breakdown and integration points
- Cross-file connection documentation with "Called by" and "Calls to" sections
- Performance considerations and Apple Silicon optimizations

**2. Named Constants Classes**:
- All magic numbers replaced with descriptive named constants
- Comprehensive explanations for configuration values
- Grouped constants by functional area with clear naming
- Documentation of performance implications and valid ranges

**3. Function Documentation Enhancement**:
- Detailed parameter and return value documentation
- Complete error handling and fallback strategy description
- Integration examples and usage patterns
- Performance notes and optimization guidance

**4. Cross-File Integration Documentation**:
- Explicit documentation of module dependencies and interactions
- Clear data flow descriptions between components
- Integration testing strategies and validation approaches
- Error propagation and recovery mechanisms

#### Enhanced Components

The following components have been enhanced with AI-first documentation:

- **scripts/finetune.py**: Enhanced model routing and error handling patterns
- **scripts/gemma_generate.py**: Detailed inference pipeline and integration examples
- **utils/gemma_dataset_prep.py**: Comprehensive dataset preparation workflow documentation
- **scripts/gemma_preflight.py**: Environment validation logic with remediation guidance
- **scripts/gemma_profiler.py**: Performance profiling methodology and integration points
- **wizard.py**: Enhanced user experience flow documentation
- **utils/dataset_prep.py**: Core dataset utilities with cross-file integration details

#### Benefits of AI-First Documentation

1. **Autonomous Development**: AI developers can understand and modify code without human intervention
2. **Consistent Patterns**: Standardized documentation structure across all components
3. **Integration Clarity**: Clear understanding of cross-file dependencies and data flow
4. **Error Prevention**: Comprehensive error handling and fallback documentation
5. **Performance Optimization**: Detailed performance considerations and Apple Silicon optimizations

## Key Design Decisions

### 1. Early MPS Memory Configuration
**Problem**: MPS memory limits must be set before PyTorch import
**Solution**: Configuration happens at the very beginning of main.py
**Impact**: Prevents memory pressure on Apple Silicon systems

### 2. Subprocess Execution from Wizard
**Problem**: Need clean separation between UI and execution
**Solution**: Wizard generates config and calls main.py via subprocess
**Impact**: Wizard remains optional, all functionality available via CLI

### 3. Hierarchical Configuration
**Problem**: Balancing flexibility with simplicity
**Solution**: Six-level inheritance with clear precedence
**Impact**: DRY configuration with minimal duplication

### 4. Lazy Operation Loading
**Problem**: Slow startup with heavy ML dependencies
**Solution**: Defer imports until operations are called
**Impact**: 40x faster startup for help and config operations

### 5. Thread-Safe Run IDs
**Problem**: Concurrent training runs need unique IDs
**Solution**: File locking for atomic counter updates
**Impact**: Reliable ID generation for parallel workflows

## System Constraints and Limitations

### Apple Silicon (MPS) Limitations
- Limited operation coverage (~300 vs CUDA's 2000+)
- No float64 support
- Unified memory requires conservative limits
- Some operations silently fall back to CPU

### Configuration Complexity
- Six-level hierarchy can be confusing
- Profile naming restrictions (no '+' character)
- Type coercion from INI strings

### Run Management
- File-based locking may have issues on network filesystems
- Metadata files can become large with full configs
- No automatic cleanup of failed runs

### Performance Considerations
- First operation has import overhead
- MPS memory pressure affects entire system
- CPU fallback can cause unexpected slowdowns

## Future Architecture Considerations

While this document captures the current "as-is" state, understanding the architecture reveals opportunities for future enhancement:

1. **Plugin Architecture**: Model implementations could be dynamically discovered
2. **Distributed Training**: Run management could be extended for multi-node
3. **Pipeline Optimization**: Operations could be composed into DAGs
4. **Real-time Monitoring**: WebSocket-based training progress
5. **Cloud Integration**: S3/GCS support for datasets and checkpoints

## Conclusion

The Whisper Fine-Tuner architecture achieves its goals of simplicity, flexibility, and performance through careful design decisions at each layer. The system successfully abstracts complexity while providing powerful capabilities for both beginners and experts. The modular architecture ensures maintainability and extensibility as the system evolves.

The emphasis on Apple Silicon optimization, combined with cross-platform compatibility, makes this system uniquely positioned for the modern ML development environment where developers increasingly use Apple Silicon for development and various platforms for deployment.
