# Data Ingestion Product Specification

## Executive Summary

The Data Ingestion system is the foundational layer of the Whisper Fine-Tuner, providing enterprise-grade data acquisition, processing, and quality management capabilities. It supports multiple data sources from cloud warehouses to local files, implements sophisticated data quality controls, and optimizes for both development iteration and production scale. The system transforms raw audio and transcription data into training-ready datasets through a robust pipeline that handles everything from BigQuery enterprise integration to single-file local datasets.

### Key Capabilities
- **Multi-Source Integration**: Seamless ingestion from BigQuery, CSV files, Google Cloud Storage, and local audio
- **Enterprise BigQuery**: Direct table export with schema introspection and intelligent column mapping
- **Quality Management**: Three-tier patch system (overrides, protections, blacklists) for data quality control
- **Streaming Support**: Process arbitrarily large datasets without memory constraints
- **Audio Processing**: Automatic resampling, format conversion, and feature extraction
- **Language Awareness**: Flexible multilingual support with mixed, strict, and override modes

### Target Users
- **Data Scientists**: Need flexible data sources and quality control for experiments
- **ML Engineers**: Require production-scale data pipelines with reliability
- **Enterprise Teams**: Must integrate with existing BigQuery data warehouses
- **Researchers**: Need fine-grained control over data quality and filtering

## Technical Architecture

### Data Flow Pipeline

```
Data Sources → Ingestion → Processing → Quality Control → Training-Ready Dataset
     ↓            ↓            ↓              ↓                    ↓
  BigQuery    CSV/Audio    Resample      Patch System         HF Dataset
  CSV Files   Validation   Tokenize      Validation          Train/Val Splits
  GCS Audio   Metadata     Features      Statistics          Cached/Streamed
```

### Core Components

#### 1. Data Source Layer

##### BigQuery Integration (`core/bigquery.py`)
```python
# Enterprise data warehouse integration
- Authentication via Application Default Credentials
- Project/Dataset/Table discovery and listing
- Schema introspection with type mapping
- Dynamic SQL generation with parameterized queries
- Automatic train/validation split generation (80/20)
- Language filtering and sampling strategies
```

**Query Generation Features**:
- Multi-table UNION ALL for consolidated datasets
- Intelligent ID column detection (id, note_id, sample_id, etc.)
- ROW_NUMBER() synthesis for tables without IDs
- CAST operations for type consistency
- WHERE clause construction for filtering
- ORDER BY RAND() for random sampling
- LIMIT constraints for development datasets

##### Local File System (`utils/dataset_utils.py`)
```python
# File-based dataset loading
- CSV file discovery and loading
- Hierarchical directory structure support
- Automatic split detection (train.csv, validation.csv)
- Fallback to prepared datasets
- Cache management for repeated loads
```

##### Google Cloud Storage (`utils/dataset_prep.py`)
```python
# Cloud storage integration
- Direct audio loading from gs:// URIs
- Retry logic with exponential backoff
- Streaming download with BytesIO buffers
- Automatic format detection
```

#### 2. Audio Processing Pipeline

##### Audio Loading (`load_audio_local_or_gcs`)
```python
def load_audio_local_or_gcs(path_or_audio, sampling_rate):
    """
    Unified audio loading supporting:
    - Local file paths (WAV, MP3, FLAC, M4A)
    - GCS URIs (gs://bucket/path)
    - In-memory arrays (for testing)
    - Dictionary format (HuggingFace Audio)
    
    Processing:
    - Automatic resampling to target rate (16kHz)
    - Mono channel conversion
    - Float32 normalization
    - Fallback to silence on errors (CI safety)
    """
```

##### Feature Extraction
```python
# Mel-spectrogram generation for Whisper
- 80/128 mel bins depending on model
- 16kHz sampling rate standard
- 25ms window, 10ms stride
- Log-mel energy normalization
- Optimized for Apple Silicon via Accelerate
```

```python
# Gemma 3n (USM audio tower) via AutoProcessor
- For Gemma 3n, feature extraction is delegated to Hugging Face's
  `AutoProcessor` (e.g., `google/gemma-3n-E2B-it`).
- The training collator builds minimal chat-style messages combining an
  `<audio:attached>` placeholder with a transcription turn, and the processor
  packs audio+text into the correct multimodal tensors.
- JSONL preparation (optional) is supported by `utils/gemma_dataset_prep.py`
  for inspection and reproducibility.
```

#### 3. Data Quality Management System

##### Patch System Architecture
```
data_patches/
└── {dataset_source}/
    ├── override_text_perfect/      # Highest priority
    │   └── corrections.csv          # Manual transcription fixes
    ├── override_text_verbatim/      
    │   └── verbatim_fixes.csv       # Verbatim text corrections
    ├── do_not_blacklist/            # Protection overrides
    │   └── ground_truth.csv         # High-quality samples
    └── delete/                      # Lowest priority
        └── problematic.csv          # Samples to filter

Precedence: Override > Protection > Blacklist
```

##### Override System
- **Purpose**: Apply manual corrections to transcriptions
- **Implementation**: O(1) lookup dictionaries for efficient application
- **Scope**: Both text_perfect and text_verbatim columns
- **Format**: CSV with id and correction columns
- **Application**: Via dataset.map() with configurable parallelism

##### Protection System
- **Purpose**: Preserve high-quality samples from filtering
- **Implementation**: Set-based membership testing
- **Priority**: Overrides blacklist decisions
- **Use Case**: Protect manually verified ground truth

##### Blacklist System
- **Purpose**: Filter problematic samples from training
- **Implementation**: Set-based filtering with protection override
- **Statistics**: Tracking of filtered vs protected samples
- **Flexibility**: Multiple blacklist files merged

### Implementation Details

#### Dataset Organization Structure

```
data/
├── datasets/
│   └── {dataset_name}/
│       ├── train.csv                 # Training split
│       ├── validation.csv             # Validation split
│       ├── {dataset_name}_prepared.csv  # Full processed dataset
│       ├── .bq_query.sql             # BigQuery source query
│       ├── metadata.json             # Dataset metadata
│       └── .cache/                   # HuggingFace cache
├── audio/
│   └── {dataset_name}/
│       └── *.wav                     # Processed audio files
└── patches/
    └── {source}/
        └── [patch directories]        # Quality control patches
```

#### CSV Schema Requirements

##### Required Columns
```csv
id,audio_path,text,language
1,/path/to/audio.wav,"transcription text",en
```

##### Extended Schema (BigQuery)
```csv
id,audio_path,text_perfect,text_verbatim,language,duration,speaker_id
1,gs://bucket/audio.wav,"cleaned text","verbatim text",en,10.5,speaker_001
```

#### Configuration System (`config.ini`)

```ini
[dataset_defaults]
text_column = text_perfect
max_label_length = 256
max_duration = 30.0
id_column = id
streaming_enabled = false
preprocessing_num_workers = 4
dataloader_num_workers = 4

[dataset:my_dataset]
source = my_dataset
text_column = text_perfect
train_split = train
validation_split = validation
languages = en,es,fr
language_mode = mixed
```

## User Journey

### 1. BigQuery Enterprise Workflow

```
User Journey:
1. Launch wizard → Select "📊 BigQuery Export"
2. Authenticate → Automatic ADC or gcloud CLI
3. Browse → List projects, datasets, tables
4. Map columns → Auto-detect audio, transcript, language
5. Configure filters → Languages, date ranges, sampling
6. Execute query → Progress bar with row count
7. Export → Generate train/validation splits
8. Register → Auto-add to config.ini
```

**Intelligent Defaults**:
- Auto-detects common column patterns (audio_url, transcript, text)
- Suggests transcript column based on content analysis
- Remembers previous selections for efficiency

### 2. Local CSV Workflow

```
User Journey:
1. Prepare CSV → Create with required columns
2. Place in directory → data/datasets/{name}/
3. Run wizard → Auto-detects available datasets
4. Configure → Set text column, language mode
5. Process → Automatic validation and splitting
```

### 3. Development Iteration Workflow

```
Quick Testing:
1. Use max_samples → Limit dataset size
2. Enable streaming → Avoid memory constraints
3. Apply patches → Test with quality controls
4. Monitor stats → View processing metrics
```

## Data Processing Pipeline

### Audio Processing Stages

#### 1. Format Conversion
- **Input**: Any ffmpeg-supported format (M4A, MP3, FLAC, OGG)
- **Output**: WAV PCM 16-bit
- **Tool**: ffmpeg with quality preservation
- **Optimization**: Parallel processing with worker pools

#### 2. Resampling
- **Target**: 16kHz (Whisper standard)
- **Method**: librosa high-quality resampling
- **Precision**: Float32 for processing
- **Fallback**: Return silence on errors (CI safety)

#### 3. Feature Extraction
```python
# Mel-spectrogram pipeline
audio → STFT → Mel-filterbank → Log-scale → Normalization → Features
16kHz    25ms     80/128 bins    Log10+ε      Zero-mean      [T, 80]
```

### Text Processing Stages

#### 1. Tokenization
- **Method**: Byte-Pair Encoding (BPE)
- **Vocabulary**: 50,000+ tokens
- **Special Tokens**: Language markers, task tokens
- **Truncation**: Configurable max_label_length
- **Padding**: Dynamic or max_length strategies

#### 2. Language Handling
```python
# Three modes of operation
"mixed"     → No language token (multilingual)
"strict"    → Per-sample language from data
"override:XX" → Force specific language
```

## Performance Optimization

### Memory Management

#### Streaming Mode
```python
# For arbitrarily large datasets
- Load patches into memory (~50MB overhead)
- Apply patches during iteration
- No dataset size limits
- Progressive statistics tracking
```

#### Batch Processing
```python
# Configurable parallelism
preprocessing_num_workers = 4  # Data loading
dataloader_num_workers = 4     # Training pipeline
batch_size = dynamic           # Platform-specific
```

#### Apple Silicon Optimizations
- Unified memory architecture leveraged
- librosa operations via Accelerate framework
- Zero-copy where possible
- MPS-friendly data layouts

### Caching Strategy

```python
# Multi-level caching
1. BigQuery results → CSV cache
2. Audio files → WAV cache
3. HuggingFace → Arrow cache
4. Patches → Memory cache
```

## Data Quality Features

### Validation Capabilities

#### Schema Validation
- Required column verification
- Data type consistency checking
- Missing value detection
- Encoding validation (UTF-8)

#### Audio Validation
```python
# Quality checks
- Duration limits (0.5-30 seconds)
- Signal integrity (non-silence)
- Sample rate verification
- Format compatibility
```

#### Text Validation
```python
# Content checks
- Non-empty transcriptions
- Character encoding validity
- Token length limits
- Language code validation
```

### Statistical Reporting

```
==== Dataset Processing Summary ====
Original dataset size:     10,000 samples
Text Override Application:
  text_perfect:           245 samples modified
  text_verbatim:          89 samples modified
Data Quality Filtering:
  Protected samples:      150 (never filtered)
  Blacklisted samples:    500 (candidates)
  Actually filtered:      350 (after protection)
Final dataset size:       9,650 samples
Data retention rate:      96.5%
```

## Platform Support

### BigQuery Integration

#### Requirements
- Google Cloud Project with BigQuery API enabled
- Application Default Credentials configured
- Read permissions on target datasets
- Python packages: google-cloud-bigquery, pandas, db-dtypes

#### Capabilities
- Multi-region support with location parameter
- Parameterized queries for security
- Batch export with progress tracking
- Automatic retry on transient failures

### File System Support

#### Local Files
- Cross-platform path handling (Windows/Unix)
- Relative and absolute path support
- Automatic directory creation
- Permission checking with fallbacks

#### Google Cloud Storage
- Direct gs:// URI support
- Authenticated access via ADC
- Streaming downloads for large files
- Retry logic for reliability

### Audio Format Support

| Format | Extension | Support Level | Notes |
|--------|-----------|--------------|-------|
| WAV | .wav | Native | Preferred format |
| MP3 | .mp3 | Full | Via ffmpeg |
| FLAC | .flac | Full | Lossless |
| M4A | .m4a | Full | Common in datasets |
| OGG | .ogg | Full | Open format |
| OPUS | .opus | Full | Modern codec |

## Configuration Reference

### Dataset Configuration

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `source` | string | Dataset source identifier | - | `"my_dataset"` |
| `text_column` | string | Column with transcriptions | `"text"` | `"text_perfect"` |
| `id_column` | string | Unique identifier column | `"id"` | `"note_id"` |
| `train_split` | string | Training split name | `"train"` | `"train"` |
| `validation_split` | string | Validation split name | `"validation"` | `"val"` |
| `max_duration` | float | Max audio duration (seconds) | 30.0 | 20.0 |
| `max_label_length` | int | Max text token length | 256 | 512 |
| `languages` | string | Language filter list | `"all"` | `"en,es,fr"` |
| `language_mode` | string | Language handling mode | `"mixed"` | `"strict"` |

### BigQuery Parameters

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|-------|
| `project_id` | string | GCP project ID | - | - |
| `location` | string | BigQuery location | None | Regional |
| `limit` | int | Row limit for query | None | 1-10M |
| `sample` | string | Sampling strategy | `"first"` | first/random |
| `languages` | list | Language filter | None | ISO codes |

### Processing Parameters

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|-------|
| `preprocessing_num_workers` | int | Parallel workers | 4 | 0-16 |
| `dataloader_num_workers` | int | DataLoader workers | 4 | 0-8 |
| `streaming_enabled` | bool | Enable streaming | False | - |
| `max_samples` | int | Sample limit | None | 1-1M |

## Error Handling

### Common Issues and Solutions

#### BigQuery Errors
```python
# Type mismatch
"Cannot read field of type STRING as INT64"
→ Solution: Use CAST in WHERE clause or quote values

# Authentication failure
"Could not authenticate with BigQuery"
→ Solution: Run 'gcloud auth application-default login'

# Empty results
"Query returned 0 rows"
→ Solution: Check filters, verify data exists
```

#### Audio Processing Errors
```python
# File not found
"Could not load audio file"
→ Fallback: Return 1 second of silence for CI

# Format unsupported
"Unknown audio format"
→ Solution: Install ffmpeg, check codecs

# Corrupted file
"Audio file corrupted"
→ Action: Add to blacklist, skip sample
```

#### Memory Errors
```python
# Out of memory
"Dataset too large for memory"
→ Solution: Enable streaming mode

# Patch files too large
"Could not load patches"
→ Solution: Split patch files, use chunking
```

## Best Practices

### 1. Data Preparation
- **Start Small**: Use limit parameter for initial testing
- **Validate Early**: Run validation before full processing
- **Clean Source**: Fix issues at source when possible
- **Document Patches**: Maintain README in patch directories

### 2. BigQuery Optimization
- **Use Projections**: Select only needed columns
- **Add Indexes**: Improve query performance
- **Partition Tables**: Use date partitioning
- **Cache Results**: Reuse exported datasets

### 3. Quality Control
- **Progressive Patches**: Start with overrides, then blacklist
- **Track Statistics**: Monitor retention rates
- **Validate Patches**: Test patches on sample data
- **Version Control**: Track patch changes in git

### 4. Performance Tuning
- **Batch Sizes**: Adjust for available memory
- **Worker Counts**: Match CPU core count
- **Streaming**: Use for 1M+ sample datasets
- **Caching**: Enable for iterative development

## Limitations and Constraints

### Current Limitations

1. **Audio Duration**
   - Maximum 30 seconds per clip
   - Minimum 0.5 seconds for quality
   - No automatic segmentation of long audio

2. **Text Processing**
   - Maximum 256 tokens default
   - UTF-8 encoding required
   - No automatic text normalization

3. **BigQuery Constraints**
   - 10GB export limit per query
   - Requires pandas/db-dtypes
   - No streaming from BigQuery

4. **Streaming Mode**
   - No random access to samples
   - Statistics are estimates
   - Incompatible with some transforms

### Workarounds

| Limitation | Workaround |
|------------|------------|
| Long audio files | Pre-segment using ffmpeg |
| Large BigQuery exports | Use sampling or date ranges |
| Memory constraints | Enable streaming mode |
| Slow patch application | Increase worker count |

## Future Roadmap

### Planned Enhancements

1. **Q1 2025**
   - HuggingFace Hub integration
   - Automatic audio segmentation
   - Real-time data augmentation
   - S3/Azure blob storage support

2. **Q2 2025**
   - Incremental dataset updates
   - Data versioning system
   - Automated quality scoring
   - Multi-modal data support

3. **Q3 2025**
   - Distributed processing
   - Stream processing from BigQuery
   - Advanced deduplication
   - Synthetic data generation

### Research Directions

- **Active Learning**: Intelligent sample selection
- **Data Augmentation**: SpecAugment, speed perturbation
- **Quality Metrics**: Automatic transcription scoring
- **Federated Datasets**: Privacy-preserving aggregation

## Integration Examples

### BigQuery Export Example
```python
from whisper_tuner.core.bigquery import build_query_and_export

dataset_dir = build_query_and_export(
    project_id="my-project",
    tables=[("dataset", "table")],
    audio_col="audio_url",
    transcript_col="transcript",
    transcript_target="text_perfect",
    language_col="language",
    languages=["en", "es"],
    limit=10000,
    sample="random",
    location="US"
)
```

### Local Dataset Loading
```python
from whisper_tuner.utils.dataset_utils import load_dataset_split

dataset, source = load_dataset_split(
    split="train",
    dataset_config={"name": "my_dataset"},
    max_samples=1000,
    patches_dir="data_patches/",
    streaming_enabled=False
)
```

### Audio Processing
```python
from whisper_tuner.utils.dataset_prep import load_audio_local_or_gcs

audio = load_audio_local_or_gcs(
    path_or_audio="gs://bucket/audio.wav",
    sampling_rate=16000,
    timeout=10,
    retries=2
)
```

## Conclusion

The Data Ingestion system provides a robust, scalable foundation for Whisper model training. Its multi-source support, sophisticated quality management, and performance optimizations make it suitable for both research experimentation and production deployments. The BigQuery integration brings enterprise-grade data warehousing to ASR model training, while the patch system ensures data quality without modifying source data.

Key strengths include seamless cloud integration, flexible data quality controls, and optimization for Apple Silicon platforms. The system's modular architecture allows for easy extension to new data sources and formats while maintaining consistency across all training methods.

For teams working with speech recognition, this data ingestion framework provides the flexibility to work with diverse data sources, the control to ensure data quality, and the scale to handle production workloads efficiently.
