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
from utils.device import get_device, apply_device_defaults, get_env_info

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
    - Calls scripts.export.export_model_dir() for HF/SafeTensors export
    - No run directory creation (model export utility)
    
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

            # Device-safe config normalization
            apply_device_defaults(profile_config)

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

        # Attach file logging inside the run directory
        if args.log_file:
            add_file_handler(args.log_file, json_format=args.json_logging)
        else:
            add_file_handler(os.path.join(run_dir, "evaluation.log"), json_format=args.json_logging)

        try:
            # Update metadata with config and environment info
            update_run_metadata(run_dir, config=profile_config, env=get_env_info())

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
            apply_device_defaults(profile_config)

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

        # Attach file logging inside the run directory
        if args.log_file:
            add_file_handler(args.log_file, json_format=args.json_logging)
        else:
            add_file_handler(os.path.join(run_dir, "blacklist.log"), json_format=args.json_logging)

        finetuning_run_dir = find_latest_completed_finetuning_run(output_dir, args.profile_or_model_dataset)

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

if __name__ == "__main__":
    # Entry point for command-line execution
    # Device detection and optimization setup occurs at module import time
    main()
