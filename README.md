# Whisper Fine-Tuner for macOS

A comprehensive framework for fine-tuning OpenAI's Whisper models with native Apple Silicon support via Metal Performance Shaders (MPS).

## Features

- 🚀 **Native Apple Silicon Support**: Optimized for M1/M2/M3 chips using MPS
- 🔄 **Cross-Platform**: Also supports NVIDIA GPUs (CUDA) and CPU
- 🎯 **Multiple Training Methods**: Standard fine-tuning, Knowledge Distillation, and LoRA (Parameter-Efficient Fine-Tuning)
- ⚡ **LoRA Support**: Memory-efficient training with 95%+ parameter reduction and 10-50MB checkpoints
- 🧠 **Knowledge Distillation**: Create smaller, faster models via teacher-student training
- 📊 **Comprehensive Evaluation**: WER/CER metrics with detailed analysis
- 🔍 **Outlier Detection**: Automatic blacklisting of problematic samples
- 🏷️ **Pseudo-Labeling**: Generate labels for unlabeled data
- 📦 **Export to GGML**: Convert models for whisper.cpp
- ☁️ **Cloud Storage Streaming**: Train on massive datasets without local storage

## Architecture Overview

The Whisper Fine-Tuner framework is built on a modular, platform-agnostic architecture that seamlessly adapts to Apple Silicon, NVIDIA CUDA, and CPU environments while providing sophisticated data quality management and multiple training paradigms.

### Design Principles

- **Platform Abstraction**: Unified device management layer abstracts hardware differences between MPS, CUDA, and CPU
- **Modular Training Methods**: Clean separation between standard, LoRA, and distillation training approaches
- **Data Quality First**: Hierarchical patch system for data corrections, blacklisting, and protection
- **Progressive Disclosure**: Interactive wizard for beginners, full configuration control for experts
- **Memory Efficiency**: Architecture optimized for Apple Silicon's unified memory and consumer hardware constraints

### Core Components

#### 1. Training Orchestration System

**Main Entry Point** (`main.py`):
- **Profile-Based Configuration**: Hierarchical configuration system with inheritance (DEFAULT → group → model → dataset → profile)
- **Run Management**: Sequential run ID generation with metadata tracking and failure recovery
- **Operation Routing**: Unified CLI for data preparation, training, evaluation, and export operations
- **Platform Optimization**: Early MPS memory configuration and device-specific backend setup

**Configuration System** (`config.ini`):
- **Hierarchical Profiles**: Compose training configurations from reusable components
- **Method Detection**: Automatic training method selection based on profile keys (lora_*, distil_*)
- **Override Support**: Command-line arguments override configuration values
- **Dataset Inheritance**: Share common preprocessing settings across datasets

**Run Management**:
```
output/
├── run-001-{profile}/
│   ├── metadata.json       # Run configuration and status
│   ├── checkpoint-*/        # Training checkpoints
│   ├── adapter_model/       # LoRA adapters (if applicable)
│   └── trainer_state.json   # Training metrics
```

#### 2. Device Management Layer

**Unified Device Abstraction** (`utils/device.py`):
- **Platform Detection**: Automatic selection following MPS → CUDA → CPU hierarchy
- **Memory Management**: Platform-specific strategies for unified (MPS) vs discrete (CUDA) memory
- **Synchronization**: Device-appropriate synchronization for accurate measurements
- **Diagnostics**: Comprehensive device capability reporting and MPS verification

**Apple Silicon Optimizations**:
- **Unified Memory Architecture**: Shared CPU/GPU memory pool eliminates transfer overhead
- **Memory Pressure Control**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO` prevents system-wide swapping
- **MPS Fallback Handling**: `PYTORCH_ENABLE_MPS_FALLBACK` for unsupported operations during development
- **Float32 Precision**: Consistent dtype usage avoiding MPS float64 limitations

**Platform-Specific Features**:
```python
# MPS: Unified memory, automatic operation fusion
torch.mps.synchronize()
torch.mps.current_allocated_memory()

# CUDA: Discrete memory, explicit management
torch.cuda.synchronize()
torch.cuda.memory_allocated()
```

#### 3. Model Training Modules

**Standard Fine-Tuning** (`models/whisper/finetune.py`):
- **Full Parameter Updates**: Trains all model parameters for maximum accuracy
- **HuggingFace Integration**: Leverages Seq2SeqTrainer for stable training
- **Memory Requirements**: 16-24GB for small models, scales with model size
- **Use Case**: Maximum performance when resources are available

**LoRA Training** (`models/whisper_lora/finetune.py`):
- **Parameter-Efficient**: Trains only 0.2-3% of parameters via low-rank adapters
- **Memory Efficient**: 4-8GB VRAM vs 16-24GB for standard training
- **Adapter Architecture**: Targets attention (q_proj, k_proj, v_proj) and feedforward (fc1, fc2) layers
- **Checkpoint Size**: 10-50MB adapters vs 1GB+ full models
- **8-bit Quantization**: Optional INT8 quantization for further memory reduction

**Knowledge Distillation** (`models/distil_whisper/finetune.py`):
- **Teacher-Student Architecture**: Large teacher model guides smaller student training
- **Dual Loss Function**: α × KL_divergence + (1-α) × cross_entropy
- **Temperature Scaling**: Smooths probability distributions for better knowledge transfer
- **Custom Training Loop**: Precise control over teacher-student interaction
- **Memory Intensive**: Requires loading both models simultaneously

#### 4. Dataset Management System

**Hierarchical Patch System** (`utils/dataset_utils.py`):
- **Override System**: Manual transcription corrections via CSV patches
- **Blacklist Management**: Automatic filtering of problematic samples
- **Protection Lists**: Preserve high-quality ground truth from blacklisting
- **Patch Precedence**: Override → Protection → Blacklist application order

**Patch Directory Structure**:
```
data_patches/{dataset}/
├── override_text_perfect/     # Transcription corrections
│   └── corrections.csv        # id,text_perfect columns
├── do_not_blacklist/          # Protected samples
│   └── ground_truth.csv       # id column
└── delete/                    # Blacklisted samples
    └── problematic.csv        # id column
```

**Streaming Support**:
- **Cloud Storage Integration**: Direct streaming from Google Cloud Storage
- **Memory-Efficient Loading**: Process datasets larger than available RAM
- **Patch Compatibility**: O(1) lookups maintain patch application with streaming

#### 5. Training Visualizer

**Real-Time Visualization** (`visualizer.py`):
- **Flask + SocketIO Backend**: Streams training metrics to web interface
- **PyTorch Hook Integration**: Extracts gradients, attention weights, and activations
- **WebGL Frontend**: GPU-accelerated 3D visualizations with Three.js
- **Performance Buffering**: Prevents visualization from impacting training speed

**Visualization Features**:
- **3D Neural Network**: Interactive layer visualization with gradient flow
- **Loss Landscape**: Real-time loss surface evolution
- **Attention Heatmaps**: Visualize model focus during training
- **Memory Waves**: System resource utilization patterns
- **Token Particles**: Generated token visualization

**Architecture Integration**:
```python
# Training script integration
if args.visualize:
    visualizer = TrainingVisualizer(model, device)
    visualizer.start_server()
    trainer.add_callback(visualizer.get_callback())
```

#### 6. Interactive Configuration Wizard

**Progressive Disclosure Interface** (`wizard.py`):
- **Guided Configuration**: Step-by-step training setup for beginners
- **Smart Defaults**: Intelligent parameter selection based on hardware
- **Method Recommendations**: Suggests training approach based on resources
- **Profile Generation**: Creates config.ini profiles on-the-fly

**Hardware-Aware Recommendations**:
```python
# Automatic batch size calculation
available_memory = get_available_memory()
model_memory = ModelSpecs.SIZES[model_size]["memory"]
recommended_batch = calculate_optimal_batch_size(
    available_memory, model_memory, training_method
)
```

**Integration Flow**:
1. Hardware detection and capability assessment
2. Dataset selection and validation
3. Training method recommendation
4. Parameter optimization for hardware
5. Profile generation and training execution

### Data Flow Architecture

```
Dataset CSV → Prepare Script → Patches Applied → DataLoader
                                     ↓
                              Training Module
                          (Standard/LoRA/Distill)
                                     ↓
                              Device Manager
                              (MPS/CUDA/CPU)
                                     ↓
                          Model → Checkpoints → Export
                            ↓
                      Visualizer (optional)
```

### Memory Management Strategy

**Apple Silicon (MPS)**:
- Unified memory architecture requires system-wide pressure management
- Default 80% memory allocation prevents swapping
- Gradient checkpointing for large models
- Attention slicing for memory-constrained scenarios

**NVIDIA (CUDA)**:
- Discrete GPU memory with explicit allocation control
- 90% default allocation for maximum utilization
- Automatic mixed precision for memory efficiency
- Multi-GPU support via DataParallel

**Batch Size Optimization**:
- Platform-specific recommendations based on memory bandwidth
- Gradient accumulation to simulate larger batches
- Dynamic batch sizing based on sequence length

## Model Architectures

The base Whisper models come in several sizes, each with a specific number of encoder and decoder layers. In the standard architecture, the number of layers for the encoder and decoder is symmetrical.

| Model | Encoder Layers | Decoder Layers | Parameters |
| :--- | :--- | :--- | :--- |
| Tiny | 4 | 4 | 39M |
| Base | 6 | 6 | 74M |
| Small | 12 | 12 | 244M |
| Medium | 24 | 24 | 769M |
| Large | 32 | 32 | 1.55B |

**Note on Distillation:** The knowledge distillation process in this repository often uses a "student" model with fewer *decoder* layers than the "teacher" to significantly improve inference speed. For example, a common strategy is to pair a large encoder (like the one from `large-v2`) with a much smaller decoder (e.g., from the `small` or `base` model). This retains the powerful audio understanding of the large encoder while making the text generation part much faster and more efficient.

## Training Methods

Choose the right method for your use case:

| Method | Memory Usage | Training Speed | Output Size | Best For |
|--------|--------------|----------------|-------------|----------|
| **LoRA ⭐** | Low (4-8GB) | Fast | ~10-50MB | **Recommended for most users** |
| Standard | High (16-24GB) | Moderate | ~1GB | Maximum performance needs |
| Distillation | Very High | Slow | ~1GB | Production model compression |

### 1. LoRA (Low-Rank Adaptation) ⭐ **Recommended**
- **Description**: Freezes base model, trains small adapter layers
- **Memory**: Low (4-8GB for small model vs 16-24GB standard)
- **Training Time**: Fast (2-3x faster than standard fine-tuning)
- **Output**: Lightweight adapters (~10-50MB vs ~1GB full model)
- **Trainable Parameters**: 0.2-3% of original model (99.8% reduction)
- **Use Case**: Domain adaptation, limited compute, iterative experimentation

**Why LoRA is Recommended:**
- **Memory Efficient**: Train large models on consumer hardware
- **Storage Efficient**: Multiple domain adapters take minimal space
- **Modular**: Swap adapters for different domains without retraining
- **Risk-Free**: Original model remains unchanged
- **Apple Silicon Optimized**: Designed for MPS efficiency

📖 **[Complete LoRA Guide →](README/LoRA.md)**

### 2. Standard Fine-Tuning
- **Description**: Updates all model parameters during training
- **Memory**: High (16-24GB for small model)
- **Training Time**: Moderate
- **Output**: Full model checkpoint (~1GB)
- **Use Case**: Maximum performance when compute resources are abundant

### 3. Knowledge Distillation
- **Description**: Train a smaller "student" model to mimic a larger "teacher"
- **Memory**: Very High (both models loaded simultaneously)
- **Training Time**: Long (custom training loop required)
- **Output**: Compressed student model
- **Use Case**: Production deployment requiring smaller model size

📖 **[Complete Distillation Guide →](README/Distillation.md)**

## System Requirements

### For Apple Silicon (Recommended)
- **macOS**: 12.3+ (Monterey or later)
- **Hardware**: Apple Silicon Mac (M1/M2/M3)
- **Python**: 3.8+ (ARM64 native - NOT x86_64/Rosetta)
- **RAM**: 16GB minimum, 32GB+ recommended

### For NVIDIA GPUs
- **CUDA**: 11.7+
- **GPU**: NVIDIA GPU with 8GB+ VRAM

## Installation

### 1. Verify ARM64 Python (Apple Silicon only)
```bash
python -c "import platform; print(platform.machine())"
# Should output: arm64
# If it shows x86_64, you're using Rosetta - reinstall Python/Conda for ARM64
```

### 2. Install PyTorch with MPS support
```bash
# For Apple Silicon
pip install torch torchvision torchaudio

# For CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies
```bash
# Core dependencies
pip install transformers datasets evaluate librosa soundfile accelerate
pip install packaging filelock tabulate

# For LoRA training (parameter-efficient fine-tuning)
pip install peft
```

Note: For deterministic installs, you can use a lockfile with uv or pip-tools:
```bash
# Using uv
uv pip compile requirements.txt -o requirements.lock
uv pip install -r requirements.lock

# Or using pip-tools
pip-compile --generate-hashes -o requirements.lock requirements.txt
pip install -r requirements.lock
```

### 4. Verify MPS setup
```bash
python scripts/system_check.py
```

## Cloud Storage Streaming (New!)

Train on massive datasets without downloading them locally. The framework now supports streaming audio files directly from Google Cloud Storage during training.

The preparer auto-detects `gs://` URIs and switches to streaming mode. It also validates the prepared CSV schema (requires `id` and your configured text column) before training/evaluation.

### Benefits
- **No disk space required**: Audio files stream directly from GCS
- **Massive scalability**: Train on datasets of any size
- **Cost efficient**: No need for large local storage
- **Seamless integration**: Works with all training methods (standard, LoRA, distillation)

### Setup
1. Store your audio files in Google Cloud Storage
2. Set up GCS authentication (via `gcloud auth` or service account)
3. Use GCS URIs (`gs://bucket/path/file.wav`) in your dataset CSV
4. Add `--no-download` flag when preparing data

### Example Workflow
```bash
# Your CSV contains GCS URIs like:
# audio_url,text_perfect,note_id
# gs://my-bucket/audio/file1.wav,"Hello world",1
# gs://my-bucket/audio/file2.wav,"Training example",2

# Prepare dataset without downloading
python scripts/prepare_data.py your_dataset --no-download

# Train as normal - audio streams automatically
python main.py finetune medium-lora-data3
```

The system automatically detects GCS paths and streams audio on-demand during training.

## Quick Start

### 1. Prepare your dataset
Create a CSV file with columns:
- `audio_url`: URL or path to audio files (supports GCS URLs like `gs://bucket/file.wav`)
- `text_perfect`: Target transcription
- `note_id`: Unique identifier

**Standard Mode** (downloads all audio files locally):
```bash
python scripts/prepare_data.py your_dataset
```

**Streaming Mode** (for large datasets - no local download):
```bash
python scripts/prepare_data.py your_dataset --no-download
```

With `--no-download`, audio files are streamed from Google Cloud Storage during training, eliminating disk space requirements for massive datasets.

### 2. Configure training
Edit `config.ini` to set:
- Model size (small/medium/large)
- Batch sizes (start with defaults)
- Training parameters

### 3. Run fine-tuning
```bash
# For Apple Silicon - enable fallback for initial testing
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run training
python main.py finetune medium-data3
```

### 4. Evaluate model
```bash
python scripts/evaluate.py --model_name_or_path output/run-001-medium-data3 --dataset data3
```

## LoRA Quick Start

**For most users, LoRA is the recommended starting point.** It's faster, uses less memory, and produces smaller checkpoints while maintaining excellent performance.

### 1. Choose a LoRA profile and run training
```bash
# Start with small model (recommended for testing)
python main.py finetune small-lora-data3

# Scale up to medium for better performance
python main.py finetune medium-lora-data3

# For Apple Silicon - enable fallback initially
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 2. Training produces lightweight adapters
```
output/run-001-small-lora-data3/
├── adapter_model/          # LoRA weights (~10-50MB)
├── training_args.bin      # Training configuration  
└── trainer_state.json     # Training metrics
```

### 3. Evaluate your LoRA model
```bash
python scripts/evaluate.py --model_name_or_path output/run-001-small-lora-data3 --dataset data3
```

**Need more control?** See the **[Complete LoRA Guide](README/LoRA.md)** for:
- Parameter tuning strategies
- 8-bit quantization setup
- Multiple adapter management
- Apple Silicon optimization
- Advanced troubleshooting

## Streaming Mode for Large Datasets

For datasets too large to fit in memory, enable streaming mode:

```bash
# In config.ini profile:
[profile:large-dataset-streaming]
streaming_enabled = true

# Or via command line:
python main.py finetune large-dataset-streaming
```

**Streaming Mode Features:**
- Process datasets of any size without loading into memory
- Patches (overrides, blacklists) still work via O(1) lookups
- Perfect for 100+ hour datasets or memory-constrained systems
- Training progress shows steps instead of percentage

**When to Use:**
- Dataset > 50GB or > 100 hours of audio
- Limited RAM (< dataset size)
- Production training on massive datasets

**Limitations:**
- No dataset shuffling (processes in order)
- No exact epoch boundaries
- Progress bars show steps, not percentages

## Apple Silicon Optimization

### Environment Variables
```bash
# Enable CPU fallback for unsupported operations (debugging only)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set memory limit (0.8 = 80% of system RAM)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

# Enable Flash Attention 2 (reduces memory by ~28%)
export SDPA_ALLOW_FLASH_ATTN=1
```

Additionally, to reduce memory spikes, preprocessing and dataloader workers are capped by default on MPS. Override via `preprocessing_num_workers` and `dataloader_num_workers` in `config.ini` if needed.

### Recommended Batch Sizes (Updated for PyTorch 2.3 with Flash Attention 2)

#### Standard Fine-Tuning
| Model | M1/M2 Pro (16-32GB) | M1/M2 Max (32-64GB) | M1/M2 Ultra (64-192GB) |
|-------|---------------------|---------------------|------------------------|
| Small | 8-12 | 16-20 | 24+ |
| Medium | 4-6 | 8-12 | 16+ |
| Large-v2 | 2-4 | 4-6 | 8+ |

#### LoRA Fine-Tuning (Higher batch sizes with Flash Attention 2)
| Model | M1/M2 Pro (16-32GB) | M1/M2 Max (32-64GB) | M1/M2 Ultra (64-192GB) |
|-------|---------------------|---------------------|------------------------|
| Small | 16-20 | 24-32 | 40+ |
| Medium | 8-12 | 16-20 | 24+ |
| Large-v2 | 4-6 | 8-12 | 16+ |

*Note: With Flash Attention 2 enabled (default), memory usage is ~28% lower than these estimates.*

### Performance Tips
1. **Start small**: Use conservative batch sizes initially
2. **Monitor memory**: Use Activity Monitor to check memory pressure
3. **Gradual increase**: Increase batch sizes if no swapping occurs
4. **Gradient accumulation**: Use to simulate larger batches

## CI

A macOS CI workflow performs smoke import tests, prepares a tiny streaming dataset, and runs a short evaluation with a tiny model. See `.github/workflows/ci-macos.yml`.

- Optional mini-train smoke: set repository variable `RUN_MINI_TRAIN=1` to enable a tiny guarded train step using `openai/whisper-tiny` (kept off by default to preserve CI time).

## Visualizer Controls

- Start the visualizer by setting `visualize=True` in your profile. It auto-binds to `127.0.0.1` and picks a free port starting at 8080.
- Throttle update frequency with `viz_update_steps` (defaults to your `logging_steps`).
- In the UI, use the bottom-right controls to toggle heavy elements on/off at runtime: 3D view, Attention heatmap, Token cloud, Spectrogram.
- You can also pass URL params for a lighter mode:
  - `?viz=light` disables heavy widgets by default
  - `show3D=0`, `showAttention=0`, `showTokens=0`, `showSpectrogram=0` to individually disable

## Project Structure
```
whisper-fine-tuner-macos/
├── models/
│   ├── whisper/         # Standard Whisper fine-tuning
│   ├── distil-whisper/  # Knowledge distillation training
│   └── whisper-lora/    # LoRA (Parameter-Efficient Fine-Tuning)
├── scripts/
│   ├── system_check.py  # Verify GPU/MPS setup
│   ├── evaluate.py      # Model evaluation
│   ├── blacklist.py     # Outlier detection
│   └── export.py        # GGML conversion
├── utils/
│   └── device.py        # Device selection (MPS/CUDA/CPU)
├── config.ini           # Training configurations
└── main.py             # Main entry point
```

## Common Issues

### MPS-Specific Issues
1. **"PyTorch not compiled with MPS"**: Reinstall PyTorch (ensure ARM64 Python)
2. **Memory errors**: Reduce batch size or set PYTORCH_MPS_HIGH_WATERMARK_RATIO
3. **Slow performance**: Disable PYTORCH_ENABLE_MPS_FALLBACK after testing

### General Issues
1. **Import errors**: Check all dependencies are installed
2. **OOM errors**: Reduce batch size or enable gradient checkpointing
3. **Data loading**: Ensure audio files are accessible and valid

### Distillation-Specific Issues
1. **Out of memory with dual models**: Reduce batch sizes significantly - distillation requires ~2x memory
2. **Slow distillation training**: Ensure teacher model is in eval mode and teacher inference uses torch.no_grad()
3. **Poor student model performance**: Increase temperature (2.0-4.0) for better knowledge transfer
4. **Training instability**: Lower learning rate (1e-5 or lower) and increase warmup steps
5. **KL divergence errors**: Ensure both models use same tokenizer and vocabulary

## Knowledge Distillation

Create smaller, faster Whisper models using knowledge distillation - a technique where a large "teacher" model transfers knowledge to a smaller "student" model while maintaining high accuracy.

### What is Knowledge Distillation?

Knowledge distillation trains a smaller student model to mimic a larger teacher model's behavior by:
- Learning from both ground truth labels (cross-entropy loss) and teacher predictions (KL divergence loss)
- Using temperature scaling to smooth probability distributions for better knowledge transfer
- Achieving 2-10x speed improvements with minimal accuracy loss

### Benefits
- **Smaller models**: Reduce model size by 50-80% 
- **Faster inference**: 2-10x speed improvements
- **Lower memory**: Deploy on resource-constrained devices
- **Maintained accuracy**: Minimal performance degradation

### Supported Teacher-Student Pairs

| Teacher Model | Student Model | Size Reduction | Speed Improvement |
|---------------|---------------|----------------|-------------------|
| whisper-large-v2 | whisper-medium | ~3x smaller | ~2-3x faster |
| whisper-large-v2 | whisper-small | ~6x smaller | ~4-6x faster |
| whisper-medium | whisper-small | ~2x smaller | ~1.5-2x faster |

### Custom Student Architectures (Advanced)

For ultimate control over performance and model size, you can go beyond pre-defined student models and create one with a custom architecture on the fly. This allows you to specify the exact number of decoder layers for the student model, enabling the creation of ultra-lightweight models.

The distillation script supports command-line arguments to define the student's architecture. For example, to create a student with a powerful encoder from `whisper-large-v2` but a tiny, custom 2-layer decoder, you could run:

```bash
python models/distil-whisper/finetune.py \
  --model_name_or_path openai/whisper-small \
  --teacher_model_name_or_path openai/whisper-large-v2 \
  --student_decoder_layers 2 \
  --dataset_name your-dataset
  ...
```

This works by:
1. Loading the configuration from a base model (e.g., `whisper-small`).
2. Programmatically modifying the configuration to set the desired number of decoder layers.
3. Creating a new, randomly-initialized student model from this modified configuration.
4. Proceeding with distillation to train this new model.

This technique is perfect for targeting highly resource-constrained environments where every megabyte and millisecond counts.

### How to Run Distillation

1. **Configure distillation parameters** (already set in `config.ini`):
```ini
[finetuning.distil_whisper]
temperature = 2.0      # Temperature scaling for probability smoothing
kl_weight = 0.5        # Weight balancing KL divergence vs cross-entropy loss
```

2. **Run distillation training using profiles**:
```bash
# Create a distilled whisper-small from whisper-large-v2 teacher
python main.py finetune distil-small-from-large

# Or create a distilled whisper-medium from whisper-large-v2 teacher  
python main.py finetune distil-medium-from-large
```

Alternatively, run distillation directly:
```bash
# Create a distilled whisper-small from whisper-large-v2 teacher
python models/distil-whisper/finetune.py \
  --model_name_or_path openai/whisper-small \
  --teacher_model_name_or_path openai/whisper-large-v2 \
  --dataset_name your-dataset \
  --output_dir output/distilled-whisper-small \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-5
```

3. **Memory considerations for Apple Silicon**:
```bash
# Set memory limit for dual-model training
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# Enable fallback for initial testing
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Distillation Parameters

- **temperature**: Controls probability distribution smoothing (default: 2.0)
  - Higher values (2.0-4.0): Better knowledge transfer, smoother distributions
  - Lower values (1.0-2.0): Sharper distributions, faster convergence

- **kl_weight**: Balances distillation vs data loss (default: 0.5)
  - Higher values (0.7-0.9): Emphasize teacher knowledge
  - Lower values (0.1-0.3): Emphasize ground truth data

### Memory Requirements

Distillation requires loading both teacher and student models simultaneously:

| Teacher + Student | M1/M2 (16-24GB) | M1/M2 Max (32-64GB) | M1/M2 Ultra (64-192GB) |
|-------------------|------------------|---------------------|-------------------------|
| Large + Medium    | Batch size 4     | Batch size 8        | Batch size 16+         |
| Large + Small     | Batch size 6     | Batch size 12       | Batch size 24+         |
| Medium + Small    | Batch size 8     | Batch size 16       | Batch size 32+         |

### Performance Tips

1. **Start with conservative batch sizes** due to dual-model memory requirements
2. **Monitor memory pressure** using Activity Monitor
3. **Use gradient accumulation** to simulate larger batches if memory-constrained
4. **Teacher model is frozen** - only student parameters are updated

## Advanced Usage

### Multi-GPU Training (CUDA only)
```bash
torchrun --nproc_per_node=2 main.py --profile large-v2-data3
```

### Custom Configurations
Create new profiles in `config.ini`:
```ini
[profile:custom-profile]
model = whisper-medium
dataset = your-dataset
learning_rate = 1e-5
per_device_train_batch_size = 12
```

### Export to CoreML (coming soon)
For maximum inference performance on Apple Silicon, export to CoreML after training.

## Acknowledgments

- OpenAI for Whisper
- Hugging Face for Transformers
- PyTorch team for MPS support