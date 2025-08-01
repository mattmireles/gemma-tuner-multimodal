# Framework Architecture

This document describes the architecture of the refactored framework for finetuning and distilling Whisper models.

## Directory Structure

```
config.ini       # Main configuration file
local.ini        # Optional local overrides for config.ini
data/            # Data directory (see README_DATASETS.md)
models/          # Model-specific finetuning scripts
    whisper/
        finetune.py  # Standard Whisper fine-tuning (all parameters)
    distil-whisper/
        finetune.py  # Knowledge distillation training (teacher-student)
    whisper-lora/
        finetune.py  # LoRA fine-tuning (parameter-efficient)
scripts/         # Main scripts
    blacklist.py # Blacklist creation / outlier detection
    prepare_data.py  # Data preparation
    evaluate.py     # Model evaluation
    export.py       # Model export to GGML
    pseudo_label.py # Pseudo-label generation
    gather.py      # Evaluation results gathering
    finetune.py     # Finetuning orchestration script
    utils.py        # Utility functions (e.g., dataset loading, metadata handling)
    validate_data.py # (Deprecated) Data validation (now in prepare_data.py)
manage.py        # Utility script for managing runs
main.py          # Main entry point
output/          # Output directory for finetuning and evaluation runs
    {run_id}-{profile_name}/ # Finetuning run directory
        metadata.json       # Metadata for the finetuning run
        completed           # Indicates successful completion of the run
        eval/               # Evaluation run directory (profile-based)
            metadata.json   # Metadata for the evaluation run
            completed       # Indicates successful completion of the run
            predictions.csv # Predictions and metrics
        eval-{dataset_name}/ # Evaluation run directory (profile+dataset)
            metadata.json   # Metadata for the evaluation run
            completed       # Indicates successful completion of the run
            predictions.csv # Predictions and metrics
    {model_name}+{dataset_name}/ # Evaluation run directory (model+dataset)
        eval/               # Evaluation run directory
            metadata.json   # Metadata for the evaluation run
            completed       # Indicates successful completion of the run
            predictions.csv # Predictions and metrics
    next_run_id.txt  # Stores the next available run ID (used for finetuning and profile-based evaluations)
```

## Components

### `config.ini`

This file contains the configuration parameters for all stages of the workflow. It is divided into sections:

*   **`[DEFAULT]`:** Default values.
*   **`[dataset_defaults]`:** Default dataset configuration.
*   **`[group:...]`:**  Default configuration for model groups (e.g., `[group:whisper]`, `[group:distil-whisper]`, `[group:whisper-lora]`).
*   **`[model:...]`:** Default configuration for specific models (e.g., `[model:whisper-medium]`, `[model:distil-medium.en]`). These sections inherit from their respective group defaults.
*   **`[profile:...]`:** Defines a specific training profile (e.g., `[profile:medium-data3]`). Profiles inherit from `[DEFAULT]`, `[dataset_defaults]`, the relevant `[group:...]` section, and the relevant `[model:...]` section, and can override any of these settings. Profile names **cannot** contain the `+` character.
*   **`[dataset:...]`:** Defines a specific dataset configuration (e.g., `[dataset:data3]`).
*   **`[evaluation]`:** Evaluation settings.
*   **`[export]`:** GGML export settings.
*   **`[pseudo_label]`:** Pseudo-labeling settings.

**New Language Handling Settings:**

*   **`language_mode`:** (In `[dataset:...]` or `[profile:...]`)
    *   **`mixed`:** No language token is set during finetuning. The model handles mixed-language input.
    *   **`strict`:** The language token is set based on the `language` field of each sample.
    *   **`override:XX`:** The language token is forced to `XX` (where `XX` is a two-letter language code or `??` for unknown) for all samples.
*   **`languages`:** (In `[dataset:...]` or `[profile:...]`)
    *   **`all`:** All languages are included.
    *   **Comma-separated list of two-letter language codes (e.g., `en,es,pt`):** Only samples with these languages (or `??` if included in the list) are included during data preparation.
    *   **`??`:** Can be included in the `languages` list to include samples with unknown or unspecified language.

**LoRA Configuration Settings:**

*   **`lora_r`:** (In `[group:whisper-lora]` or `[profile:...]`) LoRA rank parameter. Controls the dimensionality of the low-rank adaptation matrices. Higher values increase model capacity but also memory usage. Default: 32.
*   **`lora_alpha`:** (In `[group:whisper-lora]` or `[profile:...]`) LoRA alpha scaling parameter. Controls the scaling of the LoRA adaptation. Typically set to 2x the rank value. Default: 64.
*   **`lora_dropout`:** (In `[group:whisper-lora]` or `[profile:...]`) Dropout rate applied to LoRA layers to prevent overfitting. Default: 0.07.
*   **`lora_target_modules`:** (In `[group:whisper-lora]` or `[profile:...]`) Comma-separated list of model modules to apply LoRA adaptation to. Default: `q_proj,v_proj,k_proj,out_proj,fc1,fc2` (attention and feedforward layers).
*   **`enable_8bit`:** (In `[group:whisper-lora]` or `[profile:...]`) Whether to use 8-bit quantization for even lower memory usage. Default: False.

A `local.ini` file can be used to override specific settings without modifying `config.ini`.

### `scripts/prepare_data.py`

This script handles data preparation, including:

1. **Downloading Audio:** Downloads `.m4a` audio files from URLs specified in the input CSV (e.g., `data3.csv`).
2. **Decoding Audio:** Decodes `.m4a` files to `.wav` format using `ffmpeg`.
3. **Creating Prepared CSV:** Generates a `_prepared.csv` file (e.g., `data3_prepared.csv`) with local paths to the decoded audio files and relevant columns.
4. **Generating Splits:** Creates train/validation split files (`train.csv`, `validation.csv`) based on a random split or a speaker-based split (if `speaker_id` is available).
5. **Path Handling:** Uses '/' as path separator in `audio_path` for cross-platform compatibility
6. **Data Validation:** Performs checks for missing values, audio file integrity, and other potential issues.
7. **Language Filtering:** Filters the dataset based on the `languages` setting in `config.ini`, including special handling for the unknown language token `??`.
8. **Applying Overrides and Blacklists:** Applies overrides and blacklists from CSV files located in `data_patches/{dataset_name}/`. See the "Data Patches (Overrides and Blacklists)" section for details.

### `utils/dataset_utils.py`

This module contains utility functions for dataset loading. The main function is:

*   **`load_dataset_split(split, dataset_config, max_train_samples=None)`:** Loads a dataset into a Hugging Face `datasets.Dataset` object. It uses the provided `dataset_config` to locate the prepared CSV and split files. It directly loads the specified split file (`train.csv` or `validation.csv`). The optional `max_train_samples` argument allows limiting the number of samples loaded.

### `models/whisper/finetune.py`

This script handles finetuning of standard Whisper models. It:

1. Loads the configuration from the specified profile in `config.ini`.
2. Loads the dataset using `load_dataset_split`.
3. Loads the pretrained Whisper model and processor.
4. Defines the data collator and training arguments (reporting to Tensorboard ALWAYS).
5. **Handles language settings:**
    *   Determines the language handling mode (`mixed`, `strict`, or `override:XX`) from the `language_mode` setting in `config.ini`.
    *   Sets the language and task for the tokenizer using `processor.get_decoder_prompt_ids` based on the determined language mode and the `language` field of each sample (if applicable).
6. Performs the finetuning process using `transformers.Seq2SeqTrainer`.
7. Logs training metrics to Tensorboard.

### `models/distil-whisper/finetune.py`

This script will handle the finetuning of Distil-Whisper models using a distillation approach (not implemented yet).

### `models/whisper-lora/finetune.py`

This script handles Parameter-Efficient Fine-Tuning (PEFT) of Whisper models using LoRA (Low-Rank Adaptation). It provides memory-efficient training by:

1. **Loading base model:** Loads the pretrained Whisper model with optional 8-bit quantization.
2. **LoRA configuration:** Sets up LoRA adapters with configurable parameters:
   - `lora_r`: LoRA rank (default: 32) - controls adapter capacity
   - `lora_alpha`: LoRA alpha scaling (default: 64) - typically 2x rank  
   - `lora_dropout`: Dropout rate for LoRA layers (default: 0.07)
   - `lora_target_modules`: Target model components for adaptation (attention and feedforward layers)
3. **Adapter training:** Freezes original model parameters and trains only LoRA adapters (0.2-3% of parameters).
4. **Checkpoint management:** Uses `SavePeftModelCallback` to save only adapter weights (~10-50MB vs ~1GB full model).
5. **Apple Silicon optimization:** Includes MPS-specific optimizations and gradient flow handling.

**LoRA Detection Logic:**
The orchestrator (`scripts/finetune.py`) detects LoRA training by checking for LoRA-specific parameters in the profile configuration (`lora_r`, `lora_alpha`, etc.) rather than relying on model name patterns.

**Key Benefits:**
- **Memory efficient:** 50-80% less VRAM usage compared to standard fine-tuning
- **Fast training:** 2-3x faster convergence due to fewer parameters
- **Storage efficient:** 95%+ smaller checkpoints (adapters only)
- **Modular:** Multiple adapters can be trained for different domains while preserving the base model

### `scripts/evaluate.py`

This script evaluates a finetuned or pre-trained model on the specified dataset and split. It calculates the Word Error Rate (WER) and Character Error Rate (CER), and optionally logs the predictions to a CSV file.

It can be used in three ways:

1. **Evaluating a finetuned model (Profile-based):**

    ```bash
    python main.py evaluate <profile_name>
    ```

    This will load the finetuned model from the latest run in the `output/{run_id}-{profile_name}` directory and evaluate it using the settings from the specified profile. The evaluation results will be saved in `output/{run_id}-{profile_name}/eval/`. The `run_id` for these evaluations will be a sequential integer.
2. **Evaluating a pre-trained model (Model+Dataset-based):**

    ```bash
    python main.py evaluate <model_name>+<dataset_name>
    ```

    This will load the pre-trained model specified by `<model_name>` (e.g., `openai/whisper-medium`) and evaluate it on the dataset specified by `<dataset_name>` (e.g., `data3`). It will use the settings from the corresponding `[model:...]` and `[dataset:...]` sections in `config.ini`. The evaluation results will be saved in the `output/<model_name>+<dataset_name>/eval/` directory. **The `run_id` for these evaluations will be `<model_name>+<dataset_name>`**.
3. **Evaluating a finetuned model on a specific dataset (Profile+Dataset-based):**

    ```bash
    python main.py evaluate <profile_name> --dataset <dataset_name>
    ```

    This will load the finetuned model from the latest run in the `output/{run_id}-{profile_name}` directory and evaluate it on the dataset specified by `<dataset_name>`. The evaluation results will be saved in `output/{run_id}-{profile_name}/eval-<dataset_name>/`. The `run_id` for these evaluations will be a sequential integer.

**Language Handling during Evaluation:**

*   The `language_mode` setting from the profile or model+dataset configuration is used to determine how to set the language during evaluation.
*   For `mixed` mode, no language token is set.
*   For `strict` mode, the language is determined from the first sample in each batch (this might be improved to handle per-sample language).
*   For `override:XX` mode, the language is forced to `XX`.

### `scripts/export.py`

This script exports a finetuned model to the GGML format using a slightly modified `convert-h5-to-ggml.py` script (originally from the `whisper.cpp` project).

### `scripts/pseudo_label.py`

This script generates pseudo-labels for a dataset using a teacher model. It's used in the distillation process for Distil-Whisper.

### `scripts/gather.py`

This script gathers evaluation results from multiple runs into a single CSV file for easier comparison. It supports gathering results from profile-based evaluations, model+dataset based evaluations, and profile+dataset based evaluations. It traverses directories in `output/` to identify the relevant runs. The output CSV includes a "Language" column extracted from the run metadata.

### `scripts/finetune.py`

This script orchestrates the finetuning process. It:

1. Parses command-line arguments to determine the profile to use.
2. Loads the configuration from `config.ini`, including defaults and profile-specific settings.
3. Dynamically imports the appropriate model-specific finetuning script (`models/whisper/finetune.py` or `models/distil-whisper/finetune.py`) based on the `model` setting in the profile.
4. Calls the `main()` function of the selected finetuning script, passing the profile configuration.

### `manage.py`

This script provides utility commands for managing and exploring finetuning and evaluation runs. It uses the information stored in the `metadata.json` files within each run directory to provide overviews and detailed information about runs. **It correctly handles all three types of evaluation runs (profile-based, model+dataset-based, and profile+dataset-based) by differentiating their `run_id` formats.** See `README_MANAGEMENT.md` for detailed documentation.

### `main.py`

This script is the main entry point for all operations. It:

1. Parses command-line arguments to determine the operation to perform (prepare, finetune, evaluate, export, pseudo_label, gather) and the profile to use (if applicable).
2. Loads the configuration from `config.ini`.
3. Calls the appropriate function or script based on the selected operation.
4. Manages the creation of output directories and updates the `metadata.json` file to track runs. **It sets the `run_id` for model+dataset evaluations to `<model_name>+<dataset_name>`**.

## Output Path Structures

The output directory structure has been redesigned to improve organization and clarity.

### Finetuning

The finetuning output path is structured as follows:

```
output/{run_id}-{profile_name}/
    metadata.json       # Metadata for the finetuning run
    completed           # Indicates successful completion of the run
```

Where:

*   **`{run_id}`:** A unique sequential integer ID for the run.
*   **`{profile_name}`:** The name of the profile used for finetuning (e.g., `medium-data3`).

### Evaluation

The evaluation output path depends on how the evaluation is performed:

*   **Profile-based evaluation:**

    ```
    output/{run_id}-{profile_name}/eval/
        metadata.json   # Metadata for the evaluation run
        completed       # Indicates successful completion of the run
        predictions.csv # Predictions and metrics
    ```

    Where:

    *   **`{run_id}-{profile_name}`:** The directory of the *finetuned* model being evaluated.
    *   The `run_id` for these evaluations is a sequential integer.

*   **`model+dataset` based evaluation:**

    ```
    output/{model_name}+{dataset_name}/eval/
        metadata.json   # Metadata for the evaluation run
        completed       # Indicates successful completion of the run
        predictions.csv # Predictions and metrics
    ```

    Where:

    *   **`{model_name}`:** The name of the pre-trained model used for evaluation (e.g., `openai/whisper-medium`).
    *   **`{dataset_name}`:** The name of the dataset used for evaluation (e.g., `data3`).
    *   **The `run_id` for these evaluations is `{model_name}+{dataset_name}`**.

*   **Profile+Dataset based evaluation:**
    ```
    output/{run_id}-{profile_name}/eval-{dataset_name}/
        metadata.json   # Metadata for the evaluation run
        completed       # Indicates successful completion of the run
        predictions.csv # Predictions and metrics
    ```
    Where:
    *   **`{run_id}-{profile_name}`:** The directory of the *finetuned* model being evaluated.
    *   **`{dataset_name}`:** The name of the dataset used for evaluation (e.g., `data123`).
    *   The `run_id` for these evaluations is a sequential integer.

### Metadata

Each run (finetuning or evaluation) has a `metadata.json` file in its directory. It contains the following fields:

*   `run_id`: A unique identifier for the run. For finetuning and profile-based evaluations, this is a sequential integer. **For model+dataset evaluations, this is set to `<model_name>+<dataset_name>`**.
*   `run_type`: "finetuning" or "evaluation".
*   `status`: "running", "completed", "failed", or "cancelled".
*   `start_time`: The start time of the run in `YYYY-MM-DD HH:MM:SS` format.
*   `end_time`: The end time of the run in `YYYY-MM-DD HH:MM:SS` format (only present for completed, failed, or cancelled runs).
*   `profile`: The name of the profile used (if applicable).
*   `model`: The name of the model used.
*   `dataset`: The name of the dataset used.
*   `finetuning_run_id`: The `run_id` of the corresponding finetuning run (for evaluations of finetuned models).
*   `config`: A dictionary containing key configuration parameters used in the run.
*   `metrics`: A dictionary containing evaluation metrics (e.g., WER, CER) (only present for evaluation runs).
*   `error_message`: An error message if the run failed (only present for failed runs).

## Data Patches (Overrides and Blacklists)

The framework supports applying overrides and blacklists to datasets during the data preparation stage. These patches are defined in CSV files located in the `data_patches/{dataset_name}/` directory, where `{dataset_name}` corresponds to the `source` key in the dataset configuration.

### Directory Structure

```
data_patches/
    {dataset_name}/
        delete/
            blacklist.csv   # Blacklist of samples to exclude
        do_not_blacklist/
            *.csv           # CSV files with IDs that should not be blacklisted
        override_text_perfect/
            *.csv           # CSV files with text_perfect overrides
        override_text_verbatim/
            *.csv           # CSV files with text_verbatim overrides
```

### Overrides

Overrides allow modifying specific fields in the dataset. Currently, only `text_perfect` and `text_verbatim` can be overridden.

*   **Override Directory:** `data_patches/{dataset_name}/override_{field_name}/`
*   **Override File Format:** CSV files (`*.csv`) with at least two columns:
    *   `id`: The ID of the sample to override (must match the `id_column` specified in `config.ini`).
    *   `{field_name}`: The new value for the field (e.g., `text_perfect`, `text_verbatim`).
*   **Multiple Override Files:** You can have multiple CSV files in the override directory.
*   **Override Application:** Overrides are applied **before** blacklisting.
*   **ID Column Flexibility:** The script will check if the `id` column exists in the override files and skip files where it's missing.
*   **Target Column Flexibility:** The script will check if the target column (e.g., `text_perfect`) exists in the override files and skip files where it's missing.

### Blacklists

Blacklists allow excluding specific samples from the dataset.

*   **Blacklist Directory:** `data_patches/{dataset_name}/delete/`
*   **Blacklist File:** `blacklist.csv`
*   **Blacklist File Format:** CSV file with at least one column:
    *   `id`: The ID of the sample to blacklist (must match the `id_column` specified in `config.ini`).
*   **Blacklist Application:** Blacklists are applied **after** overrides.
*   **ID Column Flexibility:** The script will check if the `id` column exists in the blacklist file and skip the file if it's missing.

### Do-Not-Blacklist

The `do_not_blacklist` directory allows specifying IDs that should **not** be blacklisted, even if they appear in the blacklist file.

*   **Do-Not-Blacklist Directory:** `data_patches/{dataset_name}/do_not_blacklist/`
*   **Do-Not-Blacklist File Format:** CSV files (`*.csv`) with at least one column:
    *   `id`: The ID of the sample that should not be blacklisted.
*   **Multiple Do-Not-Blacklist Files:** You can have multiple CSV files in the `do_not_blacklist` directory.
*   **Do-Not-Blacklist Application:** The `do_not_blacklist` list is used to **filter** the blacklist. Samples in the `do_not_blacklist` list will **not** be removed, even if present in the blacklist.
*   **ID Column Flexibility:** The script will check if the `id` column exists in the do-not-blacklist files and skip files where it's missing.

### Usage

1. **Create Override/Blacklist Files:** Create the necessary CSV files in the appropriate directories under `data_patches/{dataset_name}/`.
2. **Prepare Data:** Run `python main.py prepare {dataset_name} --config config.ini` to apply the overrides and blacklist during data preparation.

The `load_dataset_split` function in `utils/dataset_utils.py` automatically handles loading and applying the overrides and blacklist.

**Example:**

To override the `text_perfect` field for sample with ID `123` in the `data3` dataset, create a file named `data_patches/data3/override_text_perfect/my_overrides.csv` with the following content:

```csv
id,text_perfect
123,This is the corrected transcription.
```

To blacklist sample with ID `456`, add it to `data_patches/data3/delete/blacklist.csv`:

```csv
id
456
```

To prevent sample with ID `789` from being blacklisted, create a file named `data_patches/data3/do_not_blacklist/keep.csv` with the following content:

```csv
id
789
```

After running `prepare_data.py`, the `data3` dataset loaded by `load_dataset_split` will have the overrides applied and the blacklisted samples removed (except for `789`).

## Workflow

The typical workflow involves the following steps:

1. **Data Preparation:** `python main.py prepare <dataset_name> --config config.ini`
2. **Finetuning:** `python main.py finetune <profile_name> --config config.ini --max_train_samples <optional_num_samples>`
3. **Evaluation:**
    *   **Profile-based:** `python main.py evaluate <profile_name> --config config.ini`
    *   **`model+dataset` based:** `python main.py evaluate <model_name>+<dataset_name> --config config.ini`
    *   **Profile+Dataset based:** `python main.py evaluate <profile_name> --dataset <dataset_name> --config config.ini`
4. **Export:** `python main.py export <profile_name> --config config.ini`
5. **Gather:** `python main.py gather <profile_name_1> <profile_name_2> ... <model_name_1>+<dataset_name_1> ... --config config.ini`
6. **Management:** Use `manage.py` to list, overview, get details about runs, and clean up failed/cancelled runs (see `README_MANAGEMENT.md`).

**Example:**

To finetune the `whisper-medium` model on the `data3` dataset using the `medium-data3` profile and limit training to 100 samples:

```bash
python main.py finetune medium-data3 --max_train_samples 100
```

To evaluate the pre-trained `whisper-medium` model on the `data3` dataset:

```bash
python main.py evaluate whisper-medium+data3
```

To evaluate the finetuned model from the `medium-data3` profile on the `data123` dataset:

```bash
python main.py evaluate medium-data3 --dataset data123
```

## Future Improvements

*   **More Flexible Dataset Handling:** Implement a more generic dataset loading mechanism that can handle different dataset formats and structures.
*   **Automated Configuration:** Create a tool to automatically generate `config.ini` files based on user input or dataset properties.
*   **Hyperparameter Search:** Integrate hyperparameter search using libraries like Optuna or Ray Tune.
*   **Advanced Evaluation:** Add more comprehensive evaluation metrics and visualizations.
*   **Modular Design:** Further modularize the code to make it more reusable and maintainable.
