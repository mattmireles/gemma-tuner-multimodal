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
Whisper Fine-Tuner CLI

Thin orchestration layer that delegates to core modules:
- core.config: configuration/profile resolution
- core.runs: run directories and metadata
- core.ops: operation dispatch to scripts/*
- core.logging: consistent logging init
"""

import argparse
import configparser
from datetime import datetime
import time
import logging
import signal
import torch

from core.logging import init_logging, add_file_handler
from core.config import load_profile_config, load_model_dataset_config
from core.runs import (
    get_next_run_id,
    create_run_directory,
    update_run_metadata,
    mark_run_as_completed,
    find_latest_finetuning_run,
    find_latest_completed_finetuning_run,
)
from core import ops
from utils.device import get_device

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

def find_finetuning_run_dir(output_dir, profile_name):
    # Keep an alias for existing behavior used by blacklist op
    from core.runs import find_latest_finetuning_run as _find
    latest = _find(output_dir, profile_name)
    if not latest:
        return None
    # require completed marker
    return latest if os.path.exists(os.path.join(latest, "completed")) else None


# mark_run_as_completed imported from core.runs

"""create_run_directory is imported from core.runs"""

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

"""update_run_metadata is imported from core.runs"""

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
    # Logging options via environment for simplicity
    log_json = os.environ.get("LOG_JSON", "0") == "1"
    init_logging("INFO", json_format=log_json)
    parser = argparse.ArgumentParser(
        description="Whisper Fine-Tuner: prepare data, finetune models, evaluate, export, and more.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "operation",
        choices=[
            "prepare",
            "finetune",
            "evaluate",
            "export",
            "pseudo_label",
            "gather",
            "validate_data",
            "system_check",
            "blacklist",
        ],
        help="Subcommand to run",
    )
    parser.add_argument("profile_or_model_dataset", nargs="?", help="Name of the profile to use (from config.ini) or model+dataset combination")
    parser.add_argument("--config", default="config.ini", help="Path to the configuration file.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training or evaluation samples to use.")
    parser.add_argument("--json_logging", action="store_true", help="Enable JSON logs (overrides LOG_JSON env)")
    parser.add_argument("--log_file", type=str, default=None, help="Optional path to also write logs to a file")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use for evaluation (overrides profile dataset).")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use for evaluation/blacklisting (overrides profile split).")

    args = parser.parse_args()

    if args.json_logging:
        init_logging("INFO", json_format=True)

    config = configparser.ConfigParser()
    config.read(args.config)

    output_dir = config["DEFAULT"]["output_dir"]

    if args.operation == "gather":
        if args.profile_or_model_dataset:
            args.profiles = [p.strip() for p in args.profile_or_model_dataset.split(',')]

    if args.operation == "prepare":
        if not args.profile_or_model_dataset:
            parser.error("The 'prepare' operation requires a dataset name (as defined in config.ini).")
        # Build a minimal config dict mirroring profile_config expectation for prepare()
        profile_config = {"dataset": args.profile_or_model_dataset}
        ops.prepare(profile_config)

    elif args.operation == "finetune":
        if not args.profile_or_model_dataset:
            parser.error("The 'finetune' operation requires a profile name.")
        if "+" in args.profile_or_model_dataset:
            parser.error("Profile names for 'finetune' cannot contain '+'.")

        profile_name = args.profile_or_model_dataset
        run_id = get_next_run_id(output_dir)
        run_dir = create_run_directory(output_dir, profile_name, run_id, "finetuning")

        # Attach file logging inside the run directory if requested
        if args.log_file:
            add_file_handler(args.log_file, json_format=args.json_logging)
        else:
            add_file_handler(os.path.join(run_dir, "run.log"), json_format=args.json_logging)

        # Register signal handlers to mark run as cancelled on interruption
        def _register_signal_handlers(current_run_dir: str):
            def _handle_signal(signum, frame):
                logger.warning(f"Received signal {signum}. Marking run as cancelled and exiting...")
                try:
                    update_run_metadata(
                        current_run_dir,
                        status="cancelled",
                        end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                finally:
                    raise SystemExit(130)
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)

        _register_signal_handlers(run_dir)

        try:
            profile_config = load_profile_config(config, profile_name)

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            # Update metadata with config
            update_run_metadata(run_dir, config=profile_config, model=profile_config["model"], dataset=profile_config["dataset"])

            # fallbacks
            profile_config["force_languages"] = profile_config.get("force_languages", False)
            profile_config["languages"] = profile_config.get("languages", "all")

            # Device-safe config normalization for MPS and CPU defaults
            if device.type == "mps":
                if profile_config.get("dtype") != "float32":
                    logger.warning("Overriding dtype to float32 for MPS compatibility")
                profile_config["dtype"] = "float32"
                if profile_config.get("attn_implementation") != "eager":
                    logger.warning("Overriding attn_implementation to 'eager' for MPS compatibility")
                profile_config["attn_implementation"] = "eager"
            elif device.type == "cpu":
                # Prefer float32/eager by default on CPU unless explicitly set
                profile_config.setdefault("dtype", "float32")
                profile_config.setdefault("attn_implementation", "eager")

            # Dispatch to fine-tuning orchestrator
            _t0 = time.time()
            ops.finetune(profile_config, run_dir)
            # Try to capture and persist training metrics
            try:
                results_path = os.path.join(run_dir, "train_results.json")
                if os.path.exists(results_path):
                    import json as _json
                    with open(results_path, "r") as rf:
                        train_metrics = _json.load(rf)
                    from core.runs import write_metrics
                    train_metrics["duration_sec"] = round(time.time() - _t0, 3)
                    write_metrics(run_dir, {"train": train_metrics})
            except Exception:
                pass
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
            # Prefer completed run via metadata; fall back to latest any
            latest_run_dir = find_latest_completed_finetuning_run(output_dir, profile_name) or find_latest_finetuning_run(output_dir, profile_name)
            if latest_run_dir:
                profile_config["model_name_or_path"] = latest_run_dir
            else:
                raise FileNotFoundError(f"No fine-tuning runs found for profile '{profile_name}'. Train a model before evaluating.")

        try:
            # Update metadata with config
            update_run_metadata(run_dir, config=profile_config)

            profile_config["force_languages"] = profile_config.get("force_languages", False)
            profile_config["languages"] = profile_config.get("languages", "all")
            profile_config["dataset"] = dataset_name

            if 'language_mode' not in profile_config:
                logger.warning("Defaulting to 'strict' language mode")
                profile_config['language_mode'] = 'strict'

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            # Device-safe config normalization for MPS and CPU defaults
            if device.type == "mps":
                if profile_config.get("dtype") != "float32":
                    logger.warning("Overriding dtype to float32 for MPS compatibility")
                profile_config["dtype"] = "float32"
                if profile_config.get("attn_implementation") != "eager":
                    logger.warning("Overriding attn_implementation to 'eager' for MPS compatibility")
                profile_config["attn_implementation"] = "eager"
            elif device.type == "cpu":
                profile_config.setdefault("dtype", "float32")
                profile_config.setdefault("attn_implementation", "eager")

            metrics = ops.evaluate(profile_config, run_dir)

            if metrics:
                update_run_metadata(run_dir, metrics=metrics)
                try:
                    from core.runs import write_metrics
                    write_metrics(run_dir, metrics)
                except Exception:
                    pass

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
        if not args.profile_or_model_dataset:
            parser.error("The 'export' operation requires a model path or HF id.")
        ops.export(args.profile_or_model_dataset)

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

        # Register signal handlers to mark run as cancelled on interruption
        def _register_signal_handlers(current_run_dir: str):
            def _handle_signal(signum, frame):
                logger.warning(f"Received signal {signum}. Marking run as cancelled and exiting...")
                try:
                    update_run_metadata(
                        current_run_dir,
                        status="cancelled",
                        end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                finally:
                    raise SystemExit(130)
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)

        _register_signal_handlers(run_dir)
        
        finetuning_run_dir = find_finetuning_run_dir(output_dir, args.profile_or_model_dataset)

        try:
            profile_config = load_profile_config(config, profile_name)

            # Use the specified split or fallback to the profile's train split
            split = args.split if args.split else profile_config["train_split"]
            profile_config["split"] = split
            if not finetuning_run_dir:
                raise FileNotFoundError(f"No completed fine-tuning run found for profile '{profile_name}'.")
            profile_config['model_name_or_path'] = finetuning_run_dir

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            from scripts.blacklist import create_blacklist
            # Device-safe config normalization for MPS and CPU defaults
            if device.type == "mps":
                if profile_config.get("dtype") != "float32":
                    logger.warning("Overriding dtype to float32 for MPS compatibility")
                profile_config["dtype"] = "float32"
                if profile_config.get("attn_implementation") != "eager":
                    logger.warning("Overriding attn_implementation to 'eager' for MPS compatibility")
                profile_config["attn_implementation"] = "eager"
            elif device.type == "cpu":
                profile_config.setdefault("dtype", "float32")
                profile_config.setdefault("attn_implementation", "eager")

            blacklist_path = create_blacklist(profile_config, run_dir)

            logger.info(f"Blacklist created at: {blacklist_path}")

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
        logger.error(f"Invalid operation: {args.operation}")

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
    # The "DEFAULT" section is a special mapping always present in ConfigParser
    # and not returned by has_section(). Merge it unconditionally.
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
    # Always merge DEFAULT pseudo-section (present even if has_section returns False)
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
