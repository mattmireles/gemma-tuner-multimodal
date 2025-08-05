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

### 4. Verify MPS setup
```bash
python scripts/system_check.py
```

## Cloud Storage Streaming (New!)

Train on massive datasets without downloading them locally. The framework now supports streaming audio files directly from Google Cloud Storage during training.

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
python main.py --profile medium-lora-data3
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
python main.py --profile medium-data3
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
python main.py --profile small-lora-data3

# Scale up to medium for better performance
python main.py --profile medium-lora-data3

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
python main.py --profile distil-small-from-large

# Or create a distilled whisper-medium from whisper-large-v2 teacher  
python main.py --profile distil-medium-from-large
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