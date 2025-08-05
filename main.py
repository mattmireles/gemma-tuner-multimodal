#!/usr/bin/env python3

# CRITICAL: MPS memory configuration MUST happen before ANY imports
# This includes standard library imports, as they may trigger other imports
# that could initialize PyTorch or its dependencies
import os
import platform
if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Set MPS memory limit for Apple Silicon before ANY other imports
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

"""
Whisper Fine-Tuner Main Entry Point

This is the central orchestration script for the Whisper fine-tuning system on Apple Silicon.
It provides a unified command-line interface for data preparation, model fine-tuning, evaluation,
and export operations across different Whisper model variants (whisper, distil-whisper).

Key responsibilities:
- Command-line argument parsing and operation routing
- Device detection and platform-specific optimization setup
- Run directory management and metadata tracking
- Profile configuration loading and validation
- Error handling and run status management

Supported operations:
- prepare: Dataset preprocessing and preparation
- finetune: Model fine-tuning with profile-based configuration
- evaluate: Model evaluation with metrics calculation
- export: Model export to GGML format
- pseudo_label: Pseudo-labeling for semi-supervised learning
- gather: Prediction gathering and aggregation
- validate_data: Dataset validation and integrity checking
- system_check: System compatibility and setup verification
- blacklist: Training sample blacklisting for data quality

Called by:
- Command-line invocation for all training workflows
- CI/CD pipelines for automated training and evaluation
- Batch processing scripts for multiple experiments

Calls to:
- scripts/prepare_data.py for dataset preparation
- scripts/finetune.py for model training coordination
- scripts/evaluate.py for model evaluation
- scripts/export.py for model format conversion
- utils/device.py for device detection and optimization
- All model-specific training modules via dynamic import

Profile system:
Configurations are managed through config.ini with hierarchical inheritance:
- DEFAULT: Base configuration settings
- dataset_defaults: Common dataset processing settings  
- group:*: Model group settings (e.g., whisper, distil-whisper)
- model:*: Specific model configurations
- dataset:*: Dataset-specific processing parameters
- profile:*: Complete training profiles combining model+dataset+hyperparameters

Run management:
- Each operation creates a unique run directory with metadata tracking
- Run IDs are generated sequentially with file-based locking
- Run status is tracked through metadata.json and completion markers
- Failed runs are preserved for debugging with error information

Apple Silicon optimizations:
- MPS device detection and memory fraction configuration (0.8 default)
- Unified memory architecture considerations
- CUDA fallback for non-Apple Silicon systems
- Device-specific backend optimizations (cuDNN benchmark, MPS settings)
"""

import argparse
import configparser
import importlib
# os and platform already imported above for MPS configuration
from datetime import datetime
from scripts.utils import update_metadata
import json
import traceback
from filelock import FileLock

import torch
import logging
from utils.device import get_device, set_memory_fraction
from constants import MemoryLimits

# Initialize logger before any usage
logger = logging.getLogger(__name__)

# Device Detection and Platform Optimization Setup
# This must happen early before any model loading or tensor operations
device = get_device()

# Platform-specific backend optimizations for training performance
if device.type == "cuda":
    # Enable cuDNN benchmark mode for consistent input sizes
    # Provides 10-20% training speedup with fixed input dimensions
    torch.backends.cudnn.benchmark = True
elif device.type == "mps":
    # MPS memory configuration already set before PyTorch import (line 10)
    # Log confirmation that MPS is configured
    logger.info(f"MPS device detected with memory ratio: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'not set')}")

def get_next_run_id(output_dir):
    """
    Generates the next sequential run ID with thread-safe file locking.
    
    Run IDs provide unique identification for each training/evaluation session,
    enabling organized experiment tracking and preventing conflicts in concurrent
    execution scenarios.
    
    Called by:
    - main() for finetune operations to create unique run directories
    - main() for evaluate operations to track evaluation runs  
    - main() for blacklist operations to organize blacklist generation
    
    Thread safety:
    Uses FileLock to prevent race conditions when multiple processes
    simultaneously request run IDs. This is essential for:
    - Parallel experiment execution
    - CI/CD pipeline concurrent runs
    - Multiple users on shared systems
    
    Run ID persistence:
    - Stored in {output_dir}/next_run_id.txt
    - Automatically creates file with ID 1 if none exists
    - Increments atomically within lock context
    - Lock file: {output_dir}/next_run_id.txt.lock
    
    Args:
        output_dir (str): Base output directory for run storage
        
    Returns:
        int: Next available run ID (starting from 1)
        
    Example:
        run_id = get_next_run_id("output")  # Returns 1, 2, 3, ...
        run_dir = f"output/{run_id}-{profile_name}"
    """
    lock_file = os.path.join(output_dir, "next_run_id.txt.lock")
    lock = FileLock(lock_file)

    with lock:
        try:
            with open(os.path.join(output_dir, "next_run_id.txt"), "r") as f:
                next_id = int(f.read())
        except FileNotFoundError:
            next_id = 1

        with open(os.path.join(output_dir, "next_run_id.txt"), "w") as f:
            f.write(str(next_id + 1))

    return next_id

def find_finetuning_run_dir(output_dir, profile_name):
    """
    Locates the most recent completed fine-tuning run directory for a profile.
    
    This function is essential for evaluation and blacklist operations that need
    to reference previously trained models. It only considers completed runs to
    ensure model artifacts are fully available.
    
    Called by:
    - main() blacklist operation to find the fine-tuned model for blacklist generation
    - Evaluation workflows requiring the latest trained model checkpoint
    
    Calls to:
    - os.listdir() to enumerate output directory contents
    - os.path.isdir() to filter directory entries
    - os.path.exists() to verify completion marker
    - os.path.getmtime() for temporal sorting
    
    Directory structure expected:
    output/
    ├── 1-profile_name/          # Run ID 1 for profile_name
    │   ├── completed            # Completion marker file
    │   ├── metadata.json        # Run metadata
    │   └── model_checkpoints/   # Model artifacts
    ├── 2-profile_name/          # Run ID 2 for profile_name
    └── 3-other_profile/         # Different profile
    
    Completion requirement:
    Only returns directories containing a 'completed' file, ensuring:
    - Training finished successfully
    - Model checkpoints are fully written
    - Metadata is finalized
    - Safe for downstream evaluation/blacklisting
    
    Temporal sorting:
    Uses filesystem modification time (mtime) to determine the "latest" run,
    handling cases where run IDs might not be perfectly chronological.
    
    Args:
        output_dir (str): Base directory containing all runs
        profile_name (str): Profile name to match (exact match required)
        
    Returns:
        str | None: Path to latest completed run directory, or None if no
                   completed runs exist for the profile
                   
    Example:
        latest_run = find_finetuning_run_dir("output", "whisper-base-en")
        if latest_run:
            model_path = os.path.join(latest_run, "final_model")
    """
    finetuning_runs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and f"-{profile_name}" in d and os.path.exists(os.path.join(output_dir, d, "completed"))
    ]
    if not finetuning_runs:
        return None
    finetuning_runs.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)), reverse=True)
    return os.path.join(output_dir, finetuning_runs[0])


def mark_run_as_completed(run_dir):
    """
    Creates a completion marker file to indicate successful run termination.
    
    The completion marker is a simple file-based semaphore that signals
    downstream processes that a run has finished successfully and all
    artifacts are available for use.
    
    Called by:
    - main() after successful finetune operations (line 194)
    - main() after successful evaluate operations (line 258)
    - Any operation that needs to signal completion to dependent workflows
    
    Completion marker significance:
    - Enables find_finetuning_run_dir() to locate usable trained models
    - Prevents evaluation of incomplete or failed training runs
    - Provides atomic completion signaling (file creation is atomic)
    - Used by run management and cleanup utilities
    
    File contents:
    The completion file contains the simple string "completed" for
    debugging and verification purposes, though only file existence matters.
    
    Args:
        run_dir (str): Path to the run directory requiring completion marking
        
    Creates:
        {run_dir}/completed: Empty marker file indicating successful completion
        
    Example workflow:
        try:
            # Perform training operations
            train_model(config)
            mark_run_as_completed(run_dir)
        except Exception as e:
            # Run remains unmarked as completed
            log_error(e)
    """
    with open(os.path.join(run_dir, "completed"), "w") as f:
        f.write("completed")

def create_run_directory(output_dir, profile_name, run_id, run_type, model_name=None, dataset_name=None):
    """
    Creates a structured run directory with initialized metadata for experiment tracking.
    
    This function establishes the foundational directory structure and metadata
    for each training, evaluation, or blacklisting operation. The directory
    structure and metadata schema enable systematic experiment management.
    
    Called by:
    - main() for finetune operations to create training run directories
    - main() for evaluate operations to create evaluation run directories
    - main() for blacklist operations to create blacklist generation directories
    
    Calls to:
    - os.makedirs() to create directory structure
    - find_latest_finetuning_run() to link evaluation runs to training runs
    - json.dump() to initialize metadata tracking
    
    Directory naming conventions:
    
    Finetuning runs:
    - Format: {run_id}-{profile_name}
    - Example: "1-whisper-base-en", "2-whisper-large-multilingual"
    
    Profile-based evaluation:
    - Format: {finetuning_run_dir}/eval
    - Example: "1-whisper-base-en/eval"
    
    Profile+dataset evaluation:
    - Format: {finetuning_run_dir}/eval-{dataset_name}
    - Example: "1-whisper-base-en/eval-librispeech"
    
    Model+dataset evaluation:
    - Format: {model_name}+{dataset_name}/eval
    - Example: "whisper-base+librispeech/eval"
    
    Metadata initialization:
    Creates metadata.json with comprehensive tracking information:
    - run_id: Unique identifier for this run
    - run_type: "finetuning", "evaluation", or "blacklist"
    - status: "running" (updated to "completed" or "failed" later)
    - timestamps: start_time and end_time (ISO format)
    - configuration: profile_name, model, dataset details
    - metrics: Empty dict populated during execution
    - finetuning_run_id: Links evaluation runs to their training runs
    
    Args:
        output_dir (str): Base directory for all runs
        profile_name (str | None): Profile name for profile-based operations
        run_id (str | int): Unique identifier for this run
        run_type (str): Type of operation ("finetuning", "evaluation", "blacklist")
        model_name (str | None): Model name for model+dataset evaluations
        dataset_name (str | None): Dataset name for dataset-specific operations
        
    Returns:
        str: Path to the created run directory
        
    Raises:
        ValueError: For invalid run_type or parameter combinations
        
    Example:
        run_dir = create_run_directory(
            output_dir="output",
            profile_name="whisper-base-en", 
            run_id=1,
            run_type="finetuning"
        )
        # Creates: output/1-whisper-base-en/ with metadata.json
    """
    if run_type == "finetuning":
        run_dir = os.path.join(output_dir, f"{run_id}-{profile_name}")
    elif run_type == "evaluation" or run_type == "blacklist":
        if profile_name and not dataset_name:
            # Profile-based evaluation
            finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            run_dir = os.path.join(finetuning_run_dir, "eval")
        elif model_name and dataset_name:
            # Model+dataset-based evaluation
            run_id = f"{model_name}+{dataset_name}"  # Set run_id to model+dataset
            run_dir = os.path.join(output_dir, run_id, "eval")
        elif profile_name and dataset_name:
            # profile + dataset evaluation
            finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            run_dir = os.path.join(finetuning_run_dir, f"eval-{dataset_name}")
        else:
            raise ValueError("Invalid run type of evaluation parameters")
    else:
        raise ValueError(f"Invalid run type: {run_type}")

    os.makedirs(run_dir, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "run_type": run_type,
        "status": "running",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "profile": profile_name,
        "model": model_name,
        "dataset": dataset_name,
        "config": {},
        "metrics": {},
        "finetuning_run_id": None
    }

    if run_type == "evaluation" and profile_name:
        finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
        if finetuning_run_dir:
            # Safely extract run ID from directory name (format: "{run_id}-{profile_name}")
            dir_basename = os.path.basename(finetuning_run_dir)
            if "-" in dir_basename:
                finetuning_run_id = dir_basename.split("-")[0]
            else:
                # Fallback: use the entire basename if no hyphen found
                finetuning_run_id = dir_basename
            metadata["finetuning_run_id"] = finetuning_run_id

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return run_dir

def find_latest_finetuning_run(output_dir, profile_name):
    """
    Locates the most recent fine-tuning run directory for a profile (completed or not).
    
    This function differs from find_finetuning_run_dir() by not requiring the
    'completed' marker, making it suitable for linking evaluation runs to their
    corresponding training runs even if training is still in progress.
    
    Called by:
    - create_run_directory() to establish finetuning_run_id links for evaluations
    - main() evaluate operations to find model checkpoints for evaluation
    
    Directory matching:
    - Matches directories with pattern: "{run_id}-{profile_name}"
    - Uses substring matching on profile_name (exact match recommended)
    - Sorts by filesystem modification time (most recent first)
    
    Use cases:
    - Evaluation runs need to reference their training run for metadata tracking
    - Progress monitoring of ongoing training experiments
    - Recovery and resume operations for interrupted training
    
    Args:
        output_dir (str): Base directory containing run directories
        profile_name (str): Profile name to match
        
    Returns:
        str | None: Path to latest finetuning run directory, or None if not found
        
    Example:
        latest_run = find_latest_finetuning_run("output", "whisper-base-en")
        if latest_run:
            model_checkpoint = os.path.join(latest_run, "checkpoint-best")
    """
    finetuning_runs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and f"-{profile_name}" in d
    ]
    if not finetuning_runs:
        return None

    finetuning_runs.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)), reverse=True)
    return os.path.join(output_dir, finetuning_runs[0])

def update_run_metadata(run_dir, **kwargs):
    """
    Updates the metadata.json file with new information during run execution.
    
    This function provides atomic updates to run metadata, enabling real-time
    tracking of experiment progress, configuration changes, and results.
    
    Called by:
    - main() to update configuration after profile loading (lines 186, 238)
    - main() to update status and timestamps on completion/failure (lines 195, 259)
    - main() to update metrics after evaluation completion (line 256)
    - Training scripts to update intermediate metrics and checkpoints
    
    Atomic update process:
    1. Read current metadata.json content
    2. Update with provided keyword arguments
    3. Write back to file (atomic file replacement)
    
    Common update patterns:
    
    Configuration updates:
    - After profile loading: config=profile_config, model=model_name, dataset=dataset_name
    
    Status updates:
    - On completion: status="completed", end_time=timestamp
    - On failure: status="failed", end_time=timestamp, error_message=str(exception)
    
    Metric updates:
    - After evaluation: metrics={"wer": 0.123, "bleu": 45.6}
    
    Thread safety:
    File operations are atomic at the OS level, but concurrent updates
    from multiple processes could result in race conditions. Consider
    file locking for high-concurrency scenarios.
    
    Args:
        run_dir (str): Path to run directory containing metadata.json
        **kwargs: Key-value pairs to update in metadata
        
    Example:
        update_run_metadata(
            run_dir,
            status="completed",
            end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics={"wer": 0.089}
        )
    """
    metadata_file = os.path.join(run_dir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    metadata.update(kwargs)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

def main():
    """
    Main entry point orchestrating all Whisper fine-tuning operations.
    
    This function implements the central command-line interface and operation
    routing for the entire fine-tuning system. It handles argument parsing,
    configuration loading, run management, and error handling.
    
    Operation flow:
    1. Parse command-line arguments and validate parameters
    2. Load configuration from config.ini
    3. Route to appropriate operation handler
    4. Create run directories and track metadata
    5. Execute operation with error handling
    6. Update run status and completion markers
    
    Supported operations and their workflows:
    
    prepare:
    - Calls scripts.prepare_data.prepare_data() for dataset preprocessing
    - No run directory creation (global dataset preparation)
    
    finetune:
    - Creates run directory with unique ID
    - Loads profile configuration with inheritance
    - Dynamically imports model-specific training module
    - Executes training with progress tracking
    - Marks run as completed on success
    
    evaluate:
    - Supports profile-based or model+dataset evaluation
    - Links to existing fine-tuning runs for profile evaluation
    - Creates evaluation-specific run directories
    - Executes evaluation and captures metrics
    
    export:
    - Calls scripts.export.export_ggml() for model format conversion
    - No run directory creation (model conversion utility)
    
    blacklist:
    - Creates blacklist generation run directory
    - Links to completed fine-tuning run for model access
    - Generates training sample blacklists for data quality
    
    Error handling:
    - Captures exceptions during operation execution
    - Updates run metadata with error information
    - Preserves failed runs for debugging
    - Provides detailed error context in logs
    
    Called by:
    - Command-line invocation: python main.py <operation> <args>
    - Batch processing scripts for automated experiments
    - CI/CD pipelines for continuous training/evaluation
    """
    parser = argparse.ArgumentParser(
        description="Main entry point for Whisper fine-tuning: data preparation, model training, evaluation, and export."
    )
    parser.add_argument("operation", choices=["prepare", "finetune", "evaluate", "export", "pseudo_label", "gather", "validate_data", "system_check", "blacklist"], help="Operation to perform")
    parser.add_argument("profile_or_model_dataset", nargs="?", help="Name of the profile to use (from config.ini) or model+dataset combination")
    parser.add_argument("--config", default="config.ini", help="Path to the configuration file.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training or evaluation samples to use.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use for evaluation (overrides profile dataset).")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use for evaluation/blacklisting (overrides profile split).")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    output_dir = config["DEFAULT"]["output_dir"]

    if args.operation == "gather":
        if args.profile_or_model_dataset:
            args.profiles = [p.strip() for p in args.profile_or_model_dataset.split(',')]

    if args.operation == "prepare":
        from scripts.prepare_data import prepare_data
        prepare_data(args.config)

    elif args.operation == "finetune":
        if not args.profile_or_model_dataset:
            parser.error("The 'finetune' operation requires a profile name.")
        if "+" in args.profile_or_model_dataset:
            parser.error("Profile names for 'finetune' cannot contain '+'.")

        profile_name = args.profile_or_model_dataset
        run_id = get_next_run_id(output_dir)
        run_dir = create_run_directory(output_dir, profile_name, run_id, "finetuning")

        try:
            # Load profile settings
            profile_config = load_profile_config(config, profile_name)

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            # Dynamically import the finetune module
            finetune_module = importlib.import_module("scripts.finetune")

            # Update metadata with config
            update_run_metadata(run_dir, config=profile_config, model=profile_config["model"], dataset=profile_config["dataset"])

            # fallbacks
            profile_config["force_languages"] = profile_config.get("force_languages", False)
            profile_config["languages"] = profile_config.get("languages", "all")

            # Call the main function of the finetune module, passing the profile config as a string
            finetune_module.main(profile_config, run_dir)
            mark_run_as_completed(run_dir)
            update_run_metadata(run_dir, status="completed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        except ImportError as e:
            logger.error(f"Module import error during finetuning: {e}")
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=f"ImportError: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Configuration error during finetuning: {e}")
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=f"ValueError: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during finetuning: {e}", exc_info=True)
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=str(e))
            raise

    elif args.operation == "evaluate":
        if not args.profile_or_model_dataset:
            parser.error("The 'evaluate' operation requires a profile name or a model+dataset combination.")

        run_id = get_next_run_id(output_dir)

        if "+" in args.profile_or_model_dataset:
            # Handle model+dataset combination
            model_name, dataset_name = args.profile_or_model_dataset.split("+")
            profile_config = load_model_dataset_config(config, model_name, dataset_name)
            run_dir = create_run_directory(output_dir, None, run_id, "evaluation", model_name=model_name, dataset_name=dataset_name)
            profile_config['model_name_or_path'] = profile_config["base_model"]

        else:
            # Handle profile or profile+dataset
            profile_name = args.profile_or_model_dataset
            profile_config = load_profile_config(config, profile_name)

            if args.dataset:
                # Profile + Dataset evaluation
                dataset_name = args.dataset
                run_dir = create_run_directory(output_dir, profile_name, run_id, "evaluation", dataset_name=dataset_name)
            else:
                # Profile evaluation with dataset from profile_config
                dataset_name = profile_config["dataset"]
                run_dir = create_run_directory(output_dir, profile_name, run_id, "evaluation")

            # Find latest finetuning run for the profile
            latest_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            if latest_run_dir:
                profile_config["model_name_or_path"] = latest_run_dir
            else:
                print(f"Error: No finetuning runs found for profile: {profile_name}")
                return

        try:
            # Update metadata with config
            update_run_metadata(run_dir, config=profile_config)

            from scripts.evaluate import run_evaluation
            profile_config["force_languages"] = profile_config.get("force_languages", False)
            profile_config["languages"] = profile_config.get("languages", "all")
            profile_config["dataset"] = dataset_name

            if 'language_mode' not in profile_config:
                print("Warning: Defaulting to 'strict' language mode")
                profile_config['language_mode'] = 'strict'

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            metrics = run_evaluation(profile_config, run_dir)

            if metrics:
                update_run_metadata(run_dir, metrics=metrics)

            mark_run_as_completed(run_dir)
            update_run_metadata(run_dir, status="completed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        except FileNotFoundError as e:
            logger.error(f"Model or dataset not found during evaluation: {e}")
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=f"FileNotFoundError: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Configuration error during evaluation: {e}")
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=f"ValueError: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during evaluation: {e}", exc_info=True)
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=str(e))
            raise

    elif args.operation == "export":
        from scripts.export import export_ggml
        export_ggml(config, args.profile_or_model_dataset)

    elif args.operation == "pseudo_label":
        from scripts.pseudo_label import main as pseudo_label_main
        pseudo_label_main()

    elif args.operation == "gather":
        from scripts.gather import gather_predictions
        gather_predictions(args.profiles)

    elif args.operation == "validate_data":
        from scripts.validate_data import main as validate_data_main
        validate_data_main(config)

    elif args.operation == "system_check":
        from scripts.system_check import main as system_check_main
        system_check_main()

    elif args.operation == "blacklist":
        profile_name = args.profile_or_model_dataset
        run_id = get_next_run_id(output_dir)
        run_dir = create_run_directory(output_dir, profile_name, run_id, "blacklist")
        
        finetuning_run_dir = find_finetuning_run_dir(output_dir, args.profile_or_model_dataset)

        try:
            profile_config = load_profile_config(config, profile_name)

            # Use the specified split or fallback to the profile's train split
            split = args.split if args.split else profile_config["train_split"]
            profile_config["split"] = split
            profile_config['model_name_or_path'] = finetuning_run_dir

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            from scripts.blacklist import create_blacklist
            blacklist_path = create_blacklist(profile_config, run_dir)

            print(f"Blacklist created at: {blacklist_path}")

        except FileNotFoundError as e:
            logger.error(f"Fine-tuning run not found for blacklist creation: {e}")
            raise
        except ValueError as e:
            logger.error(f"Configuration error during blacklist creation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during blacklist creation: {e}", exc_info=True)
            raise

    else:
        print(f"Invalid operation: {args.operation}")

def get_latest_run_directory(base_dir):
    """Finds the latest run directory based on timestamps in the directory names."""
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("20")]
    if not run_dirs:
        return None
    run_dirs.sort(reverse=True)
    return os.path.join(base_dir, run_dirs[0])

def load_profile_config(config, profile_name):
    """
    Loads and merges hierarchical configuration for a training profile.
    
    The configuration system uses inheritance to enable flexible experiment
    configuration while maintaining consistency. Configuration sections are
    merged in priority order (lowest to highest precedence).
    
    Called by:
    - main() for finetune operations to load training configuration (line 176)
    - main() for evaluate operations to load evaluation configuration (line 217)
    - main() for blacklist operations to load blacklist configuration (line 295)
    
    Configuration inheritance hierarchy (lowest to highest precedence):
    
    1. DEFAULT section:
       - Base settings for output directories, logging, etc.
       - Applied to all profiles universally
    
    2. dataset_defaults section:
       - Common dataset processing parameters
       - Shared across all dataset types
    
    3. group:* sections:
       - Model family settings (e.g., group:whisper, group:distil-whisper)
       - Defines architecture-specific defaults
    
    4. model:* sections:
       - Specific model configurations (e.g., model:whisper-base)
       - Inherits from group but can override specific parameters
    
    5. dataset:* sections:
       - Dataset-specific processing parameters
       - Audio preprocessing, tokenization settings
    
    6. profile:* sections:
       - Complete training profiles combining model+dataset+hyperparameters
       - Highest precedence, overrides all other settings
    
    Configuration resolution example:
    For profile:whisper-base-librispeech:
    - Starts with DEFAULT settings
    - Applies dataset_defaults
    - Applies group:whisper settings
    - Applies model:whisper-base settings  
    - Applies dataset:librispeech settings
    - Finally applies profile:whisper-base-librispeech settings
    
    Args:
        config (configparser.ConfigParser): Loaded configuration object
        profile_name (str): Name of the profile to load
        
    Returns:
        dict: Merged configuration dictionary with all inheritance applied
        
    Raises:
        ValueError: If profile section doesn't exist in config.ini
        
    Example config.ini structure:
        [DEFAULT]
        output_dir = output
        
        [dataset_defaults]
        max_duration = 30.0
        
        [group:whisper]
        model_family = whisper
        
        [model:whisper-base]
        group = whisper
        base_model = openai/whisper-base
        
        [dataset:librispeech]
        name = librispeech
        train_split = train.clean.100
        
        [profile:whisper-base-librispeech]
        model = whisper-base
        dataset = librispeech
        learning_rate = 1e-5
    """
    profile_section = f"profile:{profile_name}"
    if not config.has_section(profile_section):
        raise ValueError(f"Profile '{profile_name}' not found in config.ini.")

    profile_config = {}

    # Load defaults
    if config.has_section("DEFAULT"):
        profile_config.update(config["DEFAULT"])
    if config.has_section("dataset_defaults"):
        profile_config.update(config["dataset_defaults"])

    # Load model group and model defaults
    model_name = config.get(profile_section, "model")
    model_section = f"model:{model_name}"
    if config.has_section(model_section):
        group_name = config.get(model_section, "group")
        group_section = f"group:{group_name}"
        if config.has_section(group_section):
            profile_config.update(config[group_section])
        profile_config.update(config[model_section])

    # Load dataset defaults
    dataset_name = config.get(profile_section, "dataset")
    dataset_section = f"dataset:{dataset_name}"
    if config.has_section(dataset_section):
        profile_config.update(config[dataset_section])

    # Load profile settings (overrides defaults)
    profile_config.update(config[profile_section])

    return profile_config

def load_model_dataset_config(config, model_name, dataset_name):
    """
    Loads configuration for direct model+dataset evaluation without profiles.
    
    This function supports evaluation scenarios where users specify model and
    dataset directly rather than using predefined profiles. It merges relevant
    configuration sections while bypassing profile-specific settings.
    
    Called by:
    - main() for evaluate operations with model+dataset format (line 210)
    - Direct evaluation workflows bypassing the profile system
    
    Configuration merging (lowest to highest precedence):
    1. DEFAULT section: Base system settings
    2. dataset_defaults section: Common dataset processing
    3. group:* section: Model family defaults (derived from model section)
    4. model:* section: Specific model configuration
    5. dataset:* section: Dataset-specific settings
    
    Key differences from load_profile_config():
    - No profile:* section inheritance
    - Direct model and dataset specification
    - Suitable for ad-hoc evaluation scenarios
    - Bypasses profile-specific hyperparameter settings
    
    Use cases:
    - Evaluating pre-trained models on different datasets
    - Comparative evaluation across model variants
    - Research experiments with custom model+dataset combinations
    
    Args:
        config (configparser.ConfigParser): Loaded configuration object
        model_name (str): Name of the model (must exist as model:* section)
        dataset_name (str): Name of the dataset (must exist as dataset:* section)
        
    Returns:
        dict: Merged configuration dictionary for the model+dataset combination
        
    Raises:
        ValueError: If model or dataset sections don't exist in config.ini
        
    Example usage:
        python main.py evaluate whisper-base+librispeech
        # Loads model:whisper-base and dataset:librispeech configurations
    """
    model_section = f"model:{model_name}"
    dataset_section = f"dataset:{dataset_name}"

    if not config.has_section(model_section):
        raise ValueError(f"Model '{model_name}' not found in config.ini.")
    if not config.has_section(dataset_section):
        raise ValueError(f"Dataset '{dataset_name}' not found in config.ini.")

    config_dict = {}

    # Load defaults
    if config.has_section("DEFAULT"):
        config_dict.update(config["DEFAULT"])
    if config.has_section("dataset_defaults"):
        config_dict.update(config["dataset_defaults"])

    # Load model group defaults
    group_name = config.get(model_section, "group")
    group_section = f"group:{group_name}"
    if config.has_section(group_section):
        config_dict.update(config[group_section])

    # Load model and dataset settings
    config_dict.update(config[model_section])
    config_dict.update(config[dataset_section])

    return config_dict

if __name__ == "__main__":
    # Entry point for command-line execution
    # Device detection and optimization setup occurs at module import time
    main()
