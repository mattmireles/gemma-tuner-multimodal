# Blacklist/Outlier Detection System Product Specification

## Executive Summary

The Blacklist/Outlier Detection System provides intelligent, automated identification of problematic training samples that could harm model performance. By analyzing transcription quality through Word Error Rate (WER) metrics, the system generates curated blacklists while respecting manual overrides and human review decisions. This creates a powerful quality control mechanism that balances automated detection with human expertise, ensuring training datasets maintain high quality while preserving valuable edge cases.

### Key Capabilities
- **Automated Outlier Detection**: WER-based identification of mislabeled or corrupted samples
- **Manual Override Integration**: Respect human review decisions and corrections
- **Configurable Thresholds**: Different quality standards for training vs validation data
- **Detailed Diagnostics**: Comprehensive reporting with per-sample metrics and reasoning
- **Iterative Refinement**: Support for continuous dataset quality improvement
- **Human-in-the-Loop**: Seamless integration of manual review workflows

### Core Value Proposition
"Automatically identify and filter problematic training samples while preserving manually reviewed data, creating a continuously improving dataset quality management system."

### Target Users
- **ML Engineers**: Need automated quality control for training datasets
- **Data Scientists**: Require tools to identify and analyze data quality issues
- **QA Teams**: Want systematic approaches to dataset curation
- **Researchers**: Need fine-grained control over data quality experiments

## Technical Architecture

### System Overview

```
Training Dataset → Model Inference → WER Calculation → Outlier Detection → Blacklist
        ↓               ↓                 ↓                  ↓                ↓
    Audio+Text    Whisper Model      Per-Sample        Threshold         CSV Export
    Patches       Predictions         Metrics          Filtering        Integration
    Overrides     Device Optim.       WER/CER         Manual Review     Statistics
```

### Quality Control Pipeline

```python
# Blacklist generation workflow
1. Model Loading
   ├── Load fine-tuned or pre-trained model
   ├── Configure device placement
   └── Set inference parameters

2. Dataset Loading
   ├── Apply existing patches and overrides
   ├── Load audio and reference transcriptions
   └── Prepare for quality analysis

3. Quality Analysis
   ├── Generate predictions for all samples
   ├── Calculate per-sample WER/CER
   └── Track inference confidence

4. Outlier Detection
   ├── Apply configurable thresholds
   ├── Check manual override lists
   └── Identify quality issues

5. Blacklist Generation
   ├── Create detailed CSV with diagnostics
   ├── Exclude manually reviewed samples
   └── Generate quality statistics
```

### Manual Override Hierarchy

```
data_patches/{dataset}/
├── do_not_blacklist/        # Protected samples (never blacklist)
│   └── verified.csv         # Manually verified high-quality samples
├── override_text_perfect/   # Manual corrections applied
│   └── corrections.csv      # Ground truth fixes
├── override_text_verbatim/  # Verbatim corrections
│   └── verbatim.csv        # Preserve speaker disfluencies
└── delete/                  # Existing blacklist
    └── blacklist.csv       # Previously identified outliers
```

## Core Components

### 1. Blacklist Orchestration (`create_blacklist`)

The main function coordinating all aspects of quality analysis:

```python
def create_blacklist(profile_config, output_dir):
    """
    Generates comprehensive blacklist using WER-based quality analysis.
    
    Workflow:
    1. Load model and dataset with patches
    2. Run inference on evaluation split
    3. Calculate per-sample WER/CER metrics
    4. Apply quality thresholds
    5. Cross-reference manual overrides
    6. Generate blacklist CSV with diagnostics
    
    Args:
        profile_config: Configuration including model, dataset, thresholds
        output_dir: Output directory for blacklist files
        
    Returns:
        str: Path to generated blacklist CSV
    """
```

### 2. Quality Threshold Configuration

Configurable thresholds for different data splits and use cases:

```python
# Default thresholds in config.ini
[blacklist]
wer_threshold = 75.0              # Training data tolerance
validation_wer_threshold = 80.0   # Validation data tolerance
cer_threshold = 60.0              # Character-level threshold
min_audio_duration = 0.5          # Minimum valid audio length
max_audio_duration = 30.0         # Maximum valid audio length

# Per-profile overrides
[profile:whisper-medical]
wer_threshold = 50.0              # Stricter for medical domain
validation_wer_threshold = 60.0   # Higher quality required
```

**Threshold Rationale**:
- **Training Data (75% WER)**: Higher tolerance allows learning from challenging samples
- **Validation Data (80% WER)**: Stricter to ensure reliable evaluation metrics
- **Domain-Specific**: Medical, legal, or financial domains need stricter thresholds
- **Language-Specific**: Some languages may need adjusted thresholds

### 3. Manual Override Integration

The system respects human expertise through multiple override mechanisms:

```python
# Load do-not-blacklist samples
do_not_blacklist_ids = set()
for file in glob("data_patches/{dataset}/do_not_blacklist/*.csv"):
    df = pd.read_csv(file)
    do_not_blacklist_ids.update(df["id"])

# Load manually corrected samples
overridden_ids = set()
for override_dir in ["override_text_perfect", "override_text_verbatim"]:
    for file in glob(f"data_patches/{dataset}/{override_dir}/*.csv"):
        df = pd.read_csv(file)
        overridden_ids.update(df["id"])

# Apply protection during blacklist generation
for sample_id in samples:
    is_protected = sample_id in do_not_blacklist_ids
    is_overridden = sample_id in overridden_ids
    is_previously_reviewed = is_protected or is_overridden
    
    # Never blacklist manually reviewed samples
    should_blacklist = (
        wer > threshold and 
        not is_previously_reviewed
    )
```

### 4. Quality Metrics Calculation

Per-sample quality assessment with multiple metrics:

```python
def calculate_sample_quality(prediction, reference):
    """
    Calculate comprehensive quality metrics for a single sample.
    
    Metrics:
    - WER: Word Error Rate (primary metric)
    - CER: Character Error Rate (fine-grained)
    - Length ratio: Prediction/reference length
    - Repetition score: Detect repeated phrases
    - Language consistency: For multilingual datasets
    """
    
    # Normalize for fair comparison
    norm_pred = normalizer(prediction)
    norm_ref = normalizer(reference)
    
    # Calculate metrics
    wer = 100 * wer_metric.compute(
        predictions=[norm_pred],
        references=[norm_ref]
    )
    
    cer = 100 * cer_metric.compute(
        predictions=[norm_pred],
        references=[norm_ref]
    )
    
    return {
        "wer": wer,
        "cer": cer,
        "length_ratio": len(norm_pred) / max(len(norm_ref), 1),
        "is_outlier": wer > threshold
    }
```

## Blacklist Generation Workflow

### 1. Dataset Preparation

Load dataset with all existing quality patches applied:

```python
# Load dataset with patches
dataset_config = {
    "name": profile_config['dataset'],
    "text_column": profile_config["text_column"],
    "id_column": profile_config["id_column"],
    "split": profile_config["split"],
}

raw_datasets, source = load_dataset_split(
    split=profile_config["split"],
    dataset_config=dataset_config,
    patches_dir="data_patches/"
)
```

### 2. Model Inference

Generate predictions for quality assessment:

```python
# Batch inference with progress tracking
for batch in tqdm(dataloader, desc="Searching for outliers"):
    with torch.no_grad():
        # Configure language mode
        if language_mode == "strict":
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=batch_language,
                task="transcribe"
            )
        
        # Generate predictions
        generated_tokens = model.generate(
            batch["input_features"],
            max_length=max_label_length,
            num_beams=1,  # Greedy for consistency
            do_sample=False
        )
    
    # Decode predictions
    predictions = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )
```

### 3. Outlier Identification

Apply thresholds with manual override checks:

```python
outliers = []
for sample_id, wer in zip(sample_ids, wer_scores):
    # Check manual overrides
    is_protected = sample_id in do_not_blacklist_ids
    is_corrected = sample_id in overridden_ids
    
    # Determine if should blacklist
    if wer > threshold and not (is_protected or is_corrected):
        outliers.append({
            "id": sample_id,
            "blacklisted": True,
            "wer": wer,
            "reason": f"WER ({wer:.1f}%) > threshold ({threshold}%)",
            "previously_reviewed": False
        })
    elif wer > threshold and (is_protected or is_corrected):
        # Track but don't blacklist
        outliers.append({
            "id": sample_id,
            "blacklisted": False,
            "wer": wer,
            "reason": f"WER > threshold but manually reviewed",
            "previously_reviewed": True
        })
```

## Output Format

### Blacklist CSV Structure

The generated blacklist contains comprehensive diagnostic information:

```csv
id,blacklisted,ground_truth,predicted,wer,cer,model,run_id,reason,previously_reviewed
12345,true,"Hello world","Hello work",20.0,5.0,whisper-base,run-1,"WER > 75%",false
67890,false,"Complex audio","Complex auto",25.0,8.0,whisper-base,run-1,"WER > 75% but manually reviewed",true
```

**Column Descriptions**:
- `id`: Sample identifier for tracking
- `blacklisted`: Boolean flag (false for manually reviewed items)
- `ground_truth`: Original reference transcription
- `predicted`: Model-generated transcription
- `wer`: Word Error Rate percentage
- `cer`: Character Error Rate percentage
- `model`: Model used for prediction
- `run_id`: Experiment run identifier
- `reason`: Explanation for blacklisting decision
- `previously_reviewed`: Manual review status

### Statistical Summary

The system provides comprehensive quality statistics:

```python
# Generate quality report
total_samples = len(dataset)
outliers_found = len([x for x in outliers if x["blacklisted"]])
protected_samples = len([x for x in outliers if x["previously_reviewed"]])

logger.info(f"""
Blacklist Generation Summary:
  Total samples analyzed: {total_samples}
  Outliers detected: {outliers_found + protected_samples}
  Samples blacklisted: {outliers_found}
  Protected by manual review: {protected_samples}
  Average WER: {np.mean(wer_scores):.2f}%
  WER threshold: {threshold}%
""")
```

## Configuration

### Profile-Based Configuration

Configure blacklist generation per profile in `config.ini`:

```ini
[profile:whisper-base-librispeech]
# Model configuration
model_name_or_path = output/1-whisper-base/checkpoint-best
dataset = librispeech
split = validation

# Quality thresholds
wer_threshold = 75.0
validation_wer_threshold = 80.0
cer_threshold = 60.0

# Processing settings
per_device_eval_batch_size = 8
max_samples = 1000  # For testing
```

### Command-Line Usage

```bash
# Generate blacklist for profile
whisper-tuner blacklist whisper-base-profile

# With custom thresholds
whisper-tuner blacklist whisper-base-profile \
    --wer_threshold 60.0 \
    --validation_wer_threshold 70.0

# Limited samples for testing
whisper-tuner blacklist whisper-base-profile \
    --max_samples 100
```

## Integration with Training

### Automatic Blacklist Application

The training pipeline automatically excludes blacklisted samples:

```python
# In dataset loading (utils/dataset_utils.py)
def load_dataset_split(split, dataset_config, patches_dir):
    # Load raw dataset
    dataset = load_raw_dataset(dataset_config)
    
    # Apply blacklist
    blacklist_path = f"{patches_dir}/{dataset_config['name']}/delete/blacklist.csv"
    if os.path.exists(blacklist_path):
        blacklist_df = pd.read_csv(blacklist_path)
        blacklisted_ids = set(
            blacklist_df[blacklist_df["blacklisted"] == True]["id"]
        )
        dataset = dataset.filter(
            lambda x: x["id"] not in blacklisted_ids
        )
    
    return dataset
```

### Iterative Refinement Workflow

```bash
# Round 1: Initial training
whisper-tuner finetune whisper-base-profile

# Round 2: Generate blacklist
whisper-tuner blacklist whisper-base-profile

# Round 3: Retrain without outliers
whisper-tuner finetune whisper-base-profile-v2

# Round 4: Re-evaluate quality
whisper-tuner blacklist whisper-base-profile-v2
```

## Best Practices

### 1. Threshold Tuning

Start conservative and gradually tighten thresholds:

```python
# Initial exploration
thresholds = [90, 80, 70, 60, 50]
for threshold in thresholds:
    outliers = detect_outliers(dataset, threshold)
    print(f"Threshold {threshold}%: {len(outliers)} outliers")

# Choose threshold that removes 5-10% of data
optimal_threshold = select_threshold_for_percentile(dataset, 0.95)
```

### 2. Manual Review Process

Establish systematic review workflows:

```python
# 1. Generate initial blacklist
whisper-tuner blacklist profile

# 2. Export high-WER samples for review
blacklist_df = pd.read_csv("blacklist.csv")
review_candidates = blacklist_df[
    (blacklist_df["wer"] > 50) & 
    (blacklist_df["wer"] < 75)
]
review_candidates.to_csv("needs_review.csv")

# 3. After manual review, add to do_not_blacklist
verified_good = pd.read_csv("manually_verified.csv")
verified_good[["id"]].to_csv(
    "data_patches/dataset/do_not_blacklist/verified.csv"
)
```

### 3. Domain-Specific Adjustments

Different domains require different quality standards:

```ini
# Medical transcription (high accuracy required)
[profile:whisper-medical]
wer_threshold = 30.0
validation_wer_threshold = 40.0

# Conversational AI (more tolerant)
[profile:whisper-conversation]
wer_threshold = 80.0
validation_wer_threshold = 85.0

# Children's speech (very tolerant)
[profile:whisper-kids]
wer_threshold = 90.0
validation_wer_threshold = 95.0
```

### 4. Quality Monitoring

Track dataset quality over time:

```python
# Generate quality metrics
def analyze_dataset_quality(dataset, blacklist):
    metrics = {
        "total_samples": len(dataset),
        "blacklisted": len(blacklist),
        "blacklist_rate": len(blacklist) / len(dataset),
        "avg_wer": blacklist["wer"].mean(),
        "median_wer": blacklist["wer"].median(),
        "wer_std": blacklist["wer"].std(),
    }
    
    # Track trends
    save_metrics_to_history(metrics, "quality_history.json")
    plot_quality_trends("quality_history.json")
```

## Troubleshooting

### Common Issues

#### 1. Too Many False Positives
```python
# Problem: Blacklisting good samples
# Solution: Adjust thresholds
wer_threshold = 85.0  # More permissive

# Add samples to do_not_blacklist
false_positives_df[["id"]].to_csv(
    "data_patches/dataset/do_not_blacklist/false_positives.csv"
)
```

#### 2. Missing True Outliers
```python
# Problem: Not catching bad samples
# Solution: Tighten thresholds
wer_threshold = 60.0  # More strict

# Add additional quality checks
if repetition_score > 0.5:  # Repeated phrases
    blacklist.append(sample)
if silence_ratio > 0.8:  # Mostly silence
    blacklist.append(sample)
```

#### 3. Memory Issues with Large Datasets
```python
# Process in chunks
chunk_size = 1000
for i in range(0, len(dataset), chunk_size):
    chunk = dataset[i:i+chunk_size]
    chunk_blacklist = generate_blacklist(chunk)
    save_append_csv(chunk_blacklist, "blacklist.csv")
```

## Quality Patterns to Detect

### 1. Transcription Errors
- **High WER**: > 75% word error rate
- **Complete Mismatch**: WER approaching 100%
- **Wrong Language**: Transcription in different language

### 2. Audio Quality Issues
- **Silence**: Audio mostly silent
- **Noise**: Low signal-to-noise ratio
- **Truncation**: Audio cut off mid-word
- **Wrong Sample Rate**: Improperly resampled audio

### 3. Annotation Problems
- **Template Text**: Repeated boilerplate transcriptions
- **Empty Transcriptions**: Missing reference text
- **Encoding Issues**: Character encoding problems
- **Speaker Confusion**: Wrong speaker's text

### 4. Dataset Artifacts
- **Duplicate Samples**: Same audio with different IDs
- **Synthetic Audio**: Computer-generated speech
- **Test Leakage**: Validation samples in training

## Performance Considerations

### Inference Speed

| Model Size | MPS (samples/sec) | CUDA (samples/sec) | CPU (samples/sec) |
|------------|-------------------|--------------------|--------------------|
| Tiny       | 20                | 60                 | 3                  |
| Base       | 12                | 40                 | 2                  |
| Small      | 6                 | 20                 | 1                  |
| Medium     | 3                 | 10                 | 0.5                |

### Memory Requirements

| Dataset Size | Memory (Inference) | Memory (with Metrics) | Disk (Blacklist CSV) |
|--------------|--------------------|-----------------------|----------------------|
| 1K samples   | 2 GB              | 3 GB                  | 1 MB                 |
| 10K samples  | 4 GB              | 6 GB                  | 10 MB                |
| 100K samples | 8 GB              | 12 GB                 | 100 MB               |
| 1M samples   | 16 GB             | 24 GB                 | 1 GB                 |

## Integration Examples

### 1. With Training Pipeline
```python
# Automatic blacklist application during training
whisper-tuner prepare dataset_name
whisper-tuner blacklist whisper-profile
whisper-tuner finetune whisper-profile  # Automatically excludes blacklisted
```

### 2. With Evaluation
```python
# Evaluate only on high-quality samples
whisper-tuner evaluate whisper-profile \
    --exclude_blacklist \
    --min_quality_threshold 90
```

### 3. With Active Learning
```python
# Identify samples needing manual review
uncertain_samples = blacklist_df[
    (blacklist_df["wer"] > 40) & 
    (blacklist_df["wer"] < 60)
]
export_for_annotation(uncertain_samples)
```

## Future Enhancements

### Planned Features
1. **Multi-Model Consensus**: Use ensemble for robust outlier detection
2. **Active Learning Integration**: Prioritize samples for manual review
3. **Automated Correction**: Suggest corrections for common errors
4. **Quality Scoring**: Continuous quality scores instead of binary blacklist
5. **Cross-Dataset Analysis**: Identify systematic quality issues

### Research Directions
- Uncertainty-aware outlier detection
- Self-supervised quality assessment
- Domain adaptation for quality metrics
- Automated error categorization

## Conclusion

The Blacklist/Outlier Detection System provides essential quality control for training datasets, automatically identifying problematic samples while respecting human expertise. By combining WER-based detection with manual override mechanisms, it creates a powerful human-in-the-loop system for continuous dataset improvement. The configurable thresholds and detailed diagnostics enable fine-grained control over data quality, ensuring models train on clean, reliable data while preserving valuable edge cases identified through manual review.