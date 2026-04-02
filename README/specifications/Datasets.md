# Dataset Integration Guide

This guide covers all supported dataset types and integration methods for the Whisper Fine-Tuner framework, including local datasets, HuggingFace datasets, BigQuery imports, and large-scale datasets like NVIDIA Granary.

## Table of Contents

- [Overview](#overview)
- [Local Datasets](#local-datasets)
- [HuggingFace Datasets](#huggingface-datasets)
- [BigQuery Integration](#bigquery-integration)
- [NVIDIA Granary Dataset](#nvidia-granary-dataset)
- [Troubleshooting](#troubleshooting)

## Overview

The Whisper Fine-Tuner supports multiple dataset sources through a unified configuration system. All datasets are configured in `config.ini` using the `[dataset:name]` section format, enabling hierarchical configuration and consistent data loading patterns.

---

## NVIDIA Granary Dataset

The NVIDIA Granary dataset is one of the world's largest public speech datasets, providing ~643,000 hours of transcribed audio across 25 languages. This section provides comprehensive setup and usage instructions.

### 🌟 Why Use Granary?

- **Massive Scale**: ~643k hours of transcribed audio for robust model training
- **Multilingual**: 25 languages including English, Spanish, French, German, and more
- **High Quality**: Professional transcriptions from diverse, real-world sources
- **Diverse Content**: Parliamentary speeches, web content, audiobooks, and custom corpora
- **Production Ready**: Used by NVIDIA for state-of-the-art ASR model development

### 📋 Prerequisites

Before setting up Granary, ensure you have:

- **Storage Space**: Several terabytes of free disk space (recommend fast SSD)
- **Stable Internet**: Reliable connection for large corpus downloads
- **Python Environment**: Compatible HuggingFace datasets library
- **System Resources**: Sufficient RAM for audio file validation (16GB+ recommended)

### 🚀 Quick Setup (Recommended)

The fastest way to set up Granary is through the interactive wizard:

```bash
python wizard.py
```

1. Choose **"Setup NVIDIA Granary Dataset"** from the dataset options
2. Follow the guided workflow for language selection and corpus downloads
3. Configure audio source paths through the interactive prompts
4. Let the wizard generate and add configuration to `config.ini`
5. Optionally run dataset preparation directly from the wizard

### 📖 Manual Setup Guide

If you prefer manual configuration or need to understand the process in detail:

#### Step 1: Download Required Audio Corpora

Granary requires external audio downloads from these sources:

**Required Downloads:**

1. **VoxPopuli** - Multilingual parliamentary speeches
   - **Source**: https://github.com/facebookresearch/voxpopuli
   - **Download**: Follow repository instructions for your target language(s)
   - **Size**: ~500GB per major language
   - **Content**: High-quality parliamentary debate recordings

2. **YouTube Commons (YTC)** - Diverse web content
   - **Source**: https://research.google.com/youtube-cc/
   - **Download**: Access through Google Research datasets
   - **Size**: ~1-2TB per language subset
   - **Content**: Diverse YouTube content with Creative Commons licensing

3. **LibriLight** - Large-scale English audiobooks
   - **Source**: https://github.com/facebookresearch/libri-light
   - **Download**: Follow repository download scripts
   - **Size**: ~600GB for full English corpus
   - **Content**: Clean audiobook recordings for robust ASR training

**Important Notes:**
- YODAS corpus is included in the HuggingFace dataset download (no separate download needed)
- Consider downloading only your target language subset to save space
- Download to fast SSD storage for optimal training performance
- Ensure stable internet connection for multi-hour downloads

#### Step 2: Configure Dataset in config.ini

Add a Granary dataset section to your `config.ini`:

```ini
[dataset:granary-en]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
local_path = data/datasets/granary-en
text_column = text
train_split = train
validation_split = validation
audio_source_voxpopuli = /path/to/downloaded/voxpopuli/audio
audio_source_ytc = /path/to/downloaded/ytc/audio
audio_source_librilight = /path/to/downloaded/librilight/audio
```

**Configuration Parameters:**

- `source_type`: Must be "granary" for NVIDIA Granary datasets
- `hf_name`: HuggingFace dataset identifier (always "nvidia/Granary")
- `hf_subset`: Language code (e.g., "en", "es", "fr", "de", "it", "pt")
- `local_path`: Local directory for prepared dataset storage
- `audio_source_*`: Absolute paths to downloaded corpus directories

**Language Subset Options:**

| Language | Code | Corpus Availability | Notes |
|----------|------|-------------------|-------|
| English | `en` | All corpora | Largest subset, best for initial testing |
| Spanish | `es` | VoxPopuli, YTC | Strong parliamentary data |
| French | `fr` | VoxPopuli, YTC | High-quality European content |
| German | `de` | VoxPopuli, YTC | Rich multilingual parliamentary data |
| Italian | `it` | VoxPopuli, YTC | European parliamentary focus |
| Portuguese | `pt` | VoxPopuli, YTC | Brazilian and European variants |

#### Step 3: Prepare the Dataset

Run the Granary preparation script to validate audio files and create training manifest:

```bash
whisper-tuner prepare-granary granary-en
```

**What the preparation script does:**

1. **Downloads Metadata**: Retrieves transcription metadata from HuggingFace
2. **Validates Audio Files**: Checks existence of all audio files in configured paths
3. **Resolves Path Conflicts**: Handles varying directory structures across corpora
4. **Creates Unified Manifest**: Generates standardized CSV for training pipeline
5. **Reports Statistics**: Provides comprehensive validation and corpus distribution stats

**Expected Output:**
```
🚀 NVIDIA GRANARY DATASET PREPARATION
📋 Loading configuration...
📥 Downloading Granary metadata from HuggingFace...
✅ Downloaded metadata for 45,234 samples
🔍 Validating audio files and building manifest...
📊 Validation Results:
✅ Valid samples: 44,891
❌ Invalid samples: 343
📈 Corpus Distribution:
  voxpopuli: 18,234 samples
  ytc: 15,678 samples
  librilight: 10,979 samples
💾 Generating unified manifest...
✅ GRANARY PREPARATION COMPLETED SUCCESSFULLY
📄 Manifest created: data/datasets/granary-en/granary_en_prepared.csv
```

#### Step 4: Train with Granary Dataset

Once preparation is complete, use the dataset in your training profiles:

```ini
[profile:whisper-base-granary]
model = whisper-base
dataset = granary-en
num_train_epochs = 3
per_device_train_batch_size = 32
learning_rate = 1e-5
```

Then run training:

```bash
whisper-tuner finetune whisper-base-granary
```

### 🔧 Advanced Configuration

#### Multiple Language Subsets

Configure multiple language subsets for multilingual training:

```ini
[dataset:granary-multilingual]
source_type = granary
hf_name = nvidia/Granary
hf_subset = multilingual
local_path = data/datasets/granary-multilingual
text_column = text
train_split = train
validation_split = validation
languages = en,es,fr,de
audio_source_voxpopuli = /data/corpora/voxpopuli
audio_source_ytc = /data/corpora/ytc
audio_source_librilight = /data/corpora/librilight
```

#### Custom Corpus Paths

Handle non-standard directory structures:

```ini
[dataset:granary-custom-paths]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
local_path = data/datasets/granary-custom
text_column = text
train_split = train
validation_split = validation
# Custom paths for your specific setup
audio_source_voxpopuli = /mnt/storage/speech_data/voxpopuli_extracted
audio_source_ytc = /home/user/datasets/youtube_commons/audio_files
audio_source_librilight = /scratch/librilight/uncompressed_audio
```

#### Memory-Efficient Processing

For systems with limited RAM, use streaming preparation:

```ini
[dataset:granary-streaming]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
local_path = data/datasets/granary-streaming
text_column = text
streaming_enabled = true
max_samples = 10000  # Limit samples for development
audio_source_voxpopuli = /path/to/voxpopuli
audio_source_ytc = /path/to/ytc
audio_source_librilight = /path/to/librilight
```

#### Audio Validation Configuration

Granary preparation includes audio file validation to prevent training failures. You can configure validation behavior based on your needs:

**Full Validation (Default - Recommended for Production):**
```ini
[dataset:granary-safe]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
# skip_audio_validation = false  # Default - validates all files
# sample_validation_rate = 1.0   # Default - checks 100% of files
audio_source_voxpopuli = /path/to/voxpopuli
audio_source_ytc = /path/to/ytc
audio_source_librilight = /path/to/librilight
```

**Sample Validation (Faster, Some Risk):**
```ini
[dataset:granary-sampled]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
sample_validation_rate = 0.1  # Validate 10% of files for quick check
audio_source_voxpopuli = /path/to/voxpopuli
audio_source_ytc = /path/to/ytc
audio_source_librilight = /path/to/librilight
```

**Skip Validation (Fastest, Highest Risk):**
```ini
[dataset:granary-fast]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
skip_audio_validation = true  # Skip all validation (use with caution)
audio_source_voxpopuli = /path/to/voxpopuli
audio_source_ytc = /path/to/ytc
audio_source_librilight = /path/to/librilight
```

**Validation Options:**
- `skip_audio_validation`: `true` skips all validation, `false` enables validation (default: `false`)
- `sample_validation_rate`: Float 0.0-1.0 specifying fraction of files to validate (default: `1.0`)
- If `skip_audio_validation = true`, it overrides `sample_validation_rate`

**Trade-offs:**
- **Full validation**: Slowest but safest - prevents mid-training failures
- **Sample validation**: Good compromise - quick sanity check with reasonable confidence
- **Skip validation**: Fastest but risky - use only when you're confident all files exist

**Recommended Usage:**
- **Production training**: Use full validation (`sample_validation_rate = 1.0`)
- **Development/testing**: Use sample validation (`sample_validation_rate = 0.1` for 10%)
- **Known-good datasets**: Skip validation only if you've verified files manually

### 🛠️ Troubleshooting

#### Common Issues and Solutions

**1. "Audio file not found" errors during preparation**

```
❌ Audio file not found - Source: voxpopuli, Relative path: voxpopuli/en/audio.flac
```

**Solution**: Check audio source path configuration
- Verify the `audio_source_voxpopuli` path in config.ini
- Ensure the directory contains the expected audio files
- Try both absolute and relative path configurations

**2. "High error rate" validation failure**

```
❌ High error rate (15.2%): 6,789 of 44,623 files missing
```

**Solution**: Review corpus downloads
- Verify all required corpora are fully downloaded
- Check download integrity for corrupted files
- Consider reducing language scope to available corpora

**3. HuggingFace connection errors**

```
❌ Failed to download Granary metadata for subset 'en'
```

**Solution**: Check HuggingFace access
- Verify internet connection and HuggingFace Hub access
- Check HuggingFace account permissions
- Try accessing https://huggingface.co/datasets/nvidia/Granary directly

**4. Path resolution conflicts**

```
⚠️ Could not resolve path: voxpopuli/fr/audio.flac
```

**Solution**: Granary uses inconsistent path structures
- The preparation script handles most cases automatically
- For persistent issues, check actual directory structure
- Ensure corpus downloads preserved original directory layouts

**5. Memory issues during preparation**

```
❌ Out of memory during audio validation
```

**Solution**: Reduce memory usage
- Enable streaming mode: `streaming_enabled = true`
- Limit samples: `max_samples = 50000`
- Process language subsets separately
- Increase system virtual memory

**6. Training pipeline integration issues**

```
❌ Dataset loading failed: No prepared manifest found
```

**Solution**: Verify preparation completion
- Check that `granary_[subset]_prepared.csv` exists in local_path
- Re-run preparation if manifest is missing
- Verify dataset source configuration points to correct location

#### Advanced Debugging

**Enable detailed logging:**

```bash
export LOG_JSON=0  # Disable JSON logging for readability
whisper-tuner prepare-granary granary-en --log_file granary_debug.log
```

**Validate individual corpus paths:**

```bash
# Check each corpus directory
ls -la /path/to/voxpopuli
ls -la /path/to/ytc  
ls -la /path/to/librilight

# Verify file counts match expected numbers
find /path/to/voxpopuli -name "*.flac" | wc -l
```

**Test with smaller subset:**

```ini
[dataset:granary-test]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
local_path = data/datasets/granary-test
max_samples = 1000  # Small test set
audio_source_voxpopuli = /path/to/voxpopuli
audio_source_ytc = /path/to/ytc
audio_source_librilight = /path/to/librilight
```

### 📊 Performance Optimization

#### Training Performance Tips

**1. Storage Optimization**
- Use fast SSD storage for audio files
- Place prepared manifests on fastest available storage
- Consider audio file format (WAV vs FLAC trade-offs)

**2. Batch Size Tuning**
- Start with smaller batches (8-16) for Granary's large samples
- Monitor memory usage during training
- Scale up batch size based on available GPU memory

**3. Data Loading Optimization**
```ini
[dataset:granary-optimized]
# ... other configuration ...
dataloader_num_workers = 8  # Increase for faster loading
preprocessing_num_workers = 4  # Parallel audio processing
streaming_enabled = false  # Disable for faster training if memory allows
```

**4. Preparation Speed Optimization**
For faster dataset preparation, tune validation settings based on your confidence level:

```ini
[dataset:granary-fast-prep]
# ... other configuration ...
# Development: Quick sanity check
sample_validation_rate = 0.1  # Validate 10% for speed

# Production with known-good files: Skip validation entirely
skip_audio_validation = true  # Fastest, use only if files verified manually

# Conservative: Full validation (default)
# sample_validation_rate = 1.0  # Safest but slowest
```

**Validation Speed Comparison** (approximate times for full English dataset):
- Full validation: 2-4 hours (depends on storage speed)
- 10% sampling: 15-30 minutes  
- Skip validation: 5-10 minutes

#### Apple Silicon Specific Tips

**Memory Management:**
```bash
# Set conservative memory limits for MPS
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
```

**Optimal Training Configuration:**
```ini
[profile:whisper-base-granary-mps]
model = whisper-base
dataset = granary-en
dtype = float32  # Required for MPS compatibility
attn_implementation = eager  # Most stable for MPS
per_device_train_batch_size = 16  # Conservative for unified memory
gradient_accumulation_steps = 2  # Effective batch size 32
fp16 = false  # Use float32 for MPS stability
```

---

## Local Datasets

Local datasets stored in `data/datasets/` are automatically detected by the wizard. For manual configuration:

```ini
[dataset:my-local-dataset]
source = my-local-dataset
text_column = transcript
train_split = train
validation_split = validation
max_duration = 30.0
```

---

## HuggingFace Datasets

Popular HuggingFace datasets can be integrated directly:

```ini
[dataset:common-voice]
source = common-voice
hf_name = mozilla-foundation/common_voice_13_0
hf_subset = en
text_column = sentence
train_split = train
validation_split = validation
```

---

## BigQuery Integration

Import data directly from Google BigQuery:

1. Use the wizard's "Import from Google BigQuery" option
2. Configure GCP authentication: `gcloud auth application-default login`
3. Select project, dataset, and table through interactive prompts
4. Configure column mappings and language filtering
5. Export to local CSV format for training

---

## Troubleshooting

### General Dataset Issues

**Dataset not found:**
- Verify `[dataset:name]` section exists in config.ini
- Check `source` field matches directory name in `data/datasets/`
- Ensure CSV files exist in the dataset directory

**Column mapping errors:**
- Verify `text_column` matches actual CSV column names
- Check for required columns: `id`, `audio_path`, transcription column
- Validate CSV format and encoding (UTF-8 recommended)

**Training pipeline integration:**
- Ensure dataset preparation completed successfully
- Check for `_prepared.csv` files in dataset directory
- Verify no configuration conflicts between profile and dataset sections

For additional help, see the troubleshooting guides in individual dataset sections above.