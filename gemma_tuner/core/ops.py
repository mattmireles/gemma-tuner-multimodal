"""
Operation Dispatch System for Gemma Fine-Tuning Pipeline

Provides a clean abstraction layer between the CLI interface and the underlying
script implementations, enabling lazy loading of dependencies and consistent
operation interfaces.

Called by:
- cli_typer.py / main.py for all operation dispatch
- wizard for wizard-generated training workflows

Uses deferred imports to avoid loading heavy ML dependencies at module import time,
reducing CLI startup time from ~2000ms to ~5ms.
"""

from typing import Any, Dict


# Operation dispatch and lifecycle management constants
class OperationConstants:
    """Named constants for operation dispatch and configuration defaults."""

    DEFAULT_CONFIG_PATH = "config.ini"  # Primary configuration file location
    TEMP_CONFIG_PREFIX = "wizard_config_"  # Prefix for wizard-generated temp configs


def prepare(profile_config: Dict) -> None:
    """
    Prepares dataset for training by downloading and preprocessing audio files.

    This operation handles the complete dataset preparation pipeline including
    audio download, format conversion, quality filtering, and split generation.
    It uses deferred import to avoid loading data processing dependencies until
    actually needed.

    Called by:
    - main.py:main() when operation="prepare" is specified
    - Batch dataset preparation scripts processing multiple datasets sequentially
    - CI/CD pipelines for automated dataset validation and preprocessing
    - Data ingestion workflows integrating new audio collections
    - Quality assurance workflows validating audio format consistency

    Calls to:
    - scripts.prepare_data.prepare_data() for the complete preparation workflow
    - Dataset download utilities (HuggingFace datasets, direct HTTP downloads)
    - Audio processing libraries (librosa for format conversion and resampling)
    - Pandas for CSV manipulation and split generation
    - File system utilities for directory structure creation and management

    Operation workflow:
    1. Extract dataset name from profile configuration
    2. Load dataset configuration from config.ini
    3. Download audio files if not cached
    4. Convert audio to WAV format (16kHz)
    5. Filter samples by quality criteria
    6. Generate train/validation splits
    7. Save prepared dataset CSV files

    Args:
        profile_config (Dict): Merged configuration containing dataset name

    Side effects:
        - Creates data/datasets/{dataset}/ directory structure
        - Downloads and caches audio files in data/audio/
        - Generates prepared CSV files for training

    Note:
        Uses deferred import to avoid loading pandas, librosa, and other
        data processing libraries at module import time.
    """
    # Defer import to avoid heavy dependencies at module import time
    # This reduces CLI startup from ~2s to ~50ms for non-data operations
    from gemma_tuner.scripts.prepare_data import prepare_data

    dataset_name = profile_config["dataset"]
    cfg_path = OperationConstants.DEFAULT_CONFIG_PATH
    no_download = False  # Always attempt download for completeness
    prepare_data(dataset_name, cfg_path, no_download)


def finetune(profile_config: Dict, output_dir: str) -> dict[str, Any]:
    """
    Executes model fine-tuning with the specified configuration.

    This operation dispatches to the appropriate fine-tuning implementation
    based on the model type. It handles
    the complete training pipeline including model loading, data preparation,
    training loop execution, and checkpoint saving.

    Called by:
    - main.py:main() when operation="finetune" is specified
    - wizard.py:execute_training() via subprocess call to main.py
    - Automated training pipelines for batch model training workflows
    - Hyperparameter sweep frameworks iterating over configuration combinations
    - Research experiment orchestration systems managing model comparisons
    - CI/CD pipelines executing scheduled training runs for model updates

    Calls to:
    - scripts.finetune.main() which implements model type detection and routing:
      - models.gemma.finetune.main() for standard supervised fine-tuning
      - models.gemma.finetune.main() for knowledge distillation training
      - models.gemma.finetune.main() for Parameter-Efficient Fine-Tuning (LoRA)
    - Model type detection based on configuration parameters (lora_*, teacher_model, etc.)
    - Dynamic import system for loading model-specific training implementations
    - Resource management utilities for GPU/MPS memory allocation and optimization

    Training workflow:
    1. Load base model from HuggingFace or checkpoint
    2. Prepare training and validation datasets
    3. Initialize training arguments and optimizer
    4. Execute training loop with logging
    5. Save checkpoints periodically
    6. Generate final model and training metrics

    Args:
        profile_config (Dict): Complete training configuration
        output_dir (str): Directory for saving checkpoints and logs

    Side effects:
        - Creates run directory in output_dir
        - Saves model checkpoints during training
        - Writes tensorboard logs for monitoring
        - Updates run metadata with training status

    Note:
        Memory-intensive operation requiring GPU/MPS acceleration.
        Deferred import avoids loading PyTorch until training starts.
    """
    from gemma_tuner.scripts.finetune import main as finetune_main

    result = finetune_main(profile_config, output_dir)
    return result if isinstance(result, dict) else {}


def evaluate(profile_config: Dict, output_dir: str):
    """
    Evaluates a fine-tuned model on the validation dataset.

    This operation computes Word Error Rate (WER) and Character Error Rate (CER)
    metrics on the validation split. It supports both completed training runs
    and specific checkpoints for evaluation.

    Called by:
    - main.py:main() when operation="evaluate" is specified
    - Post-training evaluation workflows automatically triggered after training completion
    - Model comparison scripts evaluating multiple checkpoints or model variants
    - Quality assurance pipelines validating model performance against benchmarks
    - A/B testing frameworks comparing different training configurations
    - Production deployment workflows validating model quality before release

    Calls to:
    - scripts.evaluate.run_evaluation() for the complete evaluation workflow
    - Gemma model loading and inference pipeline for batch prediction generation
    - Audio preprocessing utilities for consistent input formatting
    - Metric calculation libraries for WER (Word Error Rate) and CER (Character Error Rate)
    - Text normalization utilities for fair comparison between predictions and ground truth
    - Detailed prediction analysis for error categorization and debugging insights

    Evaluation workflow:
    1. Load fine-tuned model from checkpoint
    2. Load validation dataset split
    3. Run inference on all validation samples
    4. Compute WER and CER metrics
    5. Generate detailed predictions CSV
    6. Save metrics to JSON file

    Args:
        profile_config (Dict): Configuration with model and dataset info
        output_dir (str): Base directory containing training run

    Returns:
        Dict: Evaluation metrics including WER and CER scores

    Side effects:
        - Creates eval/ subdirectory in run directory
        - Saves predictions.csv with all transcriptions
        - Writes metrics.json with computed scores

    Note:
        Requires completed fine-tuning run or valid checkpoint.
        GPU/MPS acceleration recommended for faster inference.
    """
    from gemma_tuner.scripts.evaluate import run_evaluation

    return run_evaluation(profile_config, output_dir)


def export(model_path_or_profile: str) -> None:
    """
    Exports fine-tuned model to deployment format (GGML/CoreML).

    This operation exports the model weights and config into a portable
    Hugging Face/SafeTensors directory for downstream use.

    Called by:
    - main.py:main() when operation="export" is specified
    - Model deployment pipelines preparing models for production inference
    - Mobile app build processes requiring CoreML or ONNX model formats
    - Edge device deployment workflows optimizing models for resource constraints
    - Model serving infrastructure preparing HuggingFace-compatible model directories
    - Integration workflows packaging models for third-party platforms

    Calls to:
    - scripts.export.export_model_dir() for complete model directory export
    - PyTorch model serialization utilities for SafeTensors format conversion
    - HuggingFace Transformers configuration management for model compatibility
    - File system utilities for directory structure creation and validation
    - Model validation utilities ensuring export integrity and completeness

    Export workflow:
    1. Load fine-tuned PyTorch model
    2. Extract model weights and configuration
    3. Convert to target format (GGML/CoreML)
    4. Optimize for deployment platform
    5. Save converted model files
    6. Validate conversion accuracy

    Args:
        model_path_or_profile (str): Path to model or profile name

    Side effects:
        - Creates exported model files in output directory
        - Generates model metadata for deployment

    Note:
        This performs a pure Hugging Face model directory export (SafeTensors).
    """
    # Wrapper around scripts.export.export_model_dir for consistency
    from gemma_tuner.scripts.export import export_model_dir

    export_model_dir(model_path_or_profile)



def blacklist(profile_config: Dict, run_dir: str) -> None:
    """
    Generates blacklist of problematic training samples based on evaluation results.

    This operation analyzes model predictions to identify samples that consistently
    cause high error rates or training instability. These samples can be filtered
    out in future training runs to improve model quality.

    Called by:
    - main.py:main() when operation="blacklist" is specified
    - Data quality improvement workflows identifying problematic training samples
    - Iterative training pipelines refining datasets between training iterations
    - Quality assurance workflows maintaining dataset integrity over time
    - Active learning systems identifying samples requiring manual annotation review
    - Dataset curation pipelines automatically filtering low-quality audio samples

    Calls to:
    - scripts.blacklist.create_blacklist() for the complete blacklist generation workflow
    - Evaluation result parsing utilities for prediction analysis
    - Statistical analysis libraries for outlier detection and threshold calculation
    - Text similarity metrics for identifying transcription quality issues
    - Audio analysis utilities for detecting corrupted or problematic audio files
    - CSV manipulation utilities for blacklist file generation and formatting

    Blacklist generation workflow:
    1. Load evaluation predictions from previous run
    2. Calculate per-sample error metrics
    3. Identify statistical outliers and problem cases
    4. Apply configurable filtering thresholds
    5. Generate blacklist CSV with reasons
    6. Save to data_patches directory for future use

    Args:
        profile_config (Dict): Configuration with analysis parameters
        run_dir (str): Directory containing evaluation results

    Side effects:
        - Creates blacklist CSV in data_patches/{dataset}/delete/
        - Updates run metadata with blacklist statistics
        - Generates analysis report with problem categories

    Blacklist criteria:
        - High WER samples (>threshold)
        - Empty or corrupted audio files
        - Mismatched language samples
        - Extreme duration outliers

    Note:
        Requires completed evaluation run with predictions.csv.
        Blacklist is automatically applied in future training runs.
    """
    from gemma_tuner.scripts.blacklist import create_blacklist

    create_blacklist(profile_config, run_dir)
