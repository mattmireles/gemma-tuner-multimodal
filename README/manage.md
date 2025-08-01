# Manage.py Documentation

This document describes the functionality of `manage.py`, the dataset and model management utility for the Whisper Fine-Tuner.

## Overview

`manage.py` provides utilities for managing datasets, models, and training runs. It handles data preparation, model evaluation, and various data management tasks.

## Commands

### Data Preparation

```bash
python manage.py prepare <dataset_name> [--config config.ini]
```

Prepares a dataset for training by:
- Converting audio files to the correct format
- Creating train/validation splits
- Filtering by language settings
- Applying data patches and overrides

### Model Evaluation

```bash
python manage.py evaluate <profile_or_model+dataset>
```

Two evaluation modes:

1. **Profile-based**: `python manage.py evaluate medium-data3`
   - Uses the model and dataset from the specified profile
   
2. **Direct specification**: `python manage.py evaluate whisper-medium+data3`
   - Evaluates a specific model on a specific dataset

### Dataset Management

```bash
# List available datasets
python manage.py list-datasets

# Show dataset statistics
python manage.py stats <dataset_name>

# Validate dataset integrity
python manage.py validate <dataset_name>
```

## Configuration

### Dataset Configuration

Datasets are configured in `config.ini`:

```ini
[dataset:data3]
source = data3
train_split = train
validation_split = validation
languages = en,es,??
max_duration = 30.0
text_column = text_perfect
```

### Language Handling

The `languages` parameter controls which languages are included:

- `all` - Include all languages
- `en,es` - Include only English and Spanish
- `en,es,??` - Include English, Spanish, and unknown languages

The special `??` token represents unknown or unspecified languages.

### Data Patches

Override specific samples using CSV files:

```bash
# Create override file
echo "note_id,text_perfect" > data_patches/data3_overrides.csv
echo "12345,corrected transcription" >> data_patches/data3_overrides.csv

# Apply during preparation
python manage.py prepare data3 --apply-patches
```

## Adding New Datasets

1. **Create dataset directory**: `data/datasets/<dataset_name>/`
2. **Add raw CSV**: Must include columns: `note_id`, `audio_url`, `text_verbatim`, `text_perfect`
3. **Configure in config.ini**:
   ```ini
   [dataset:new_dataset]
   source = new_dataset
   languages = all
   max_duration = 30.0
   ```
4. **Prepare dataset**: `python manage.py prepare new_dataset`

## Advanced Features

### Blacklist Integration

```bash
# Generate blacklist from evaluation
python manage.py blacklist <model> <dataset> --wer-threshold 75

# Apply blacklist during training
python manage.py prepare <dataset> --apply-blacklist
```

### Data Validation

```bash
# Check for missing audio files
python manage.py validate <dataset> --check-audio

# Verify transcription quality
python manage.py validate <dataset> --min-words 3 --max-duration 30
```

### Export Functions

```bash
# Export dataset statistics
python manage.py export-stats <dataset> --format json

# Export prepared dataset
python manage.py export <dataset> --format csv --output prepared_data.csv
```

## Dataset Directory Structure

```
data/
├── datasets/
│   └── <dataset_name>/
│       ├── raw.csv           # Original data
│       ├── train.csv         # Prepared training split
│       ├── validation.csv    # Prepared validation split
│       └── metadata.json     # Dataset statistics
├── audio/
│   └── <dataset_name>/       # Converted audio files
└── patches/
    └── <dataset_name>/       # Override files
```

## Best Practices

1. **Always validate** datasets before training
2. **Use language filtering** to improve model focus
3. **Apply patches** for known transcription errors
4. **Monitor dataset statistics** for quality control
5. **Keep blacklists updated** from evaluation results

## Troubleshooting

### Common Issues

1. **Missing audio files**: Run `validate --check-audio` to identify
2. **Language mismatches**: Check `languages` setting in config
3. **Memory issues**: Use `--batch-size` parameter for large datasets
4. **Slow preparation**: Adjust `--num-workers` for parallel processing

### Debug Mode

```bash
# Enable verbose logging
python manage.py prepare <dataset> --debug

# Dry run without making changes
python manage.py prepare <dataset> --dry-run
```