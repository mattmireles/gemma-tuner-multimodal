#!/usr/bin/env python3
"""
Modern Typer-based CLI for Whisper Fine-Tuner with Enhanced Developer Experience

This module provides the primary command-line interface for the Whisper fine-tuning
system using Typer's modern CLI framework. It offers a more ergonomic and type-safe
alternative to the legacy argparse implementation while maintaining full compatibility
with the underlying core operations.

Key responsibilities:
- Command routing for all fine-tuning operations (prepare, finetune, evaluate, export)
- Type-safe argument parsing with rich help documentation
- Automatic signal handling for graceful interruption support
- Unified logging configuration across all commands
- Run directory management and metadata tracking
- Device configuration normalization for Apple Silicon optimization

Called by:
- Direct shell invocation: `python cli_typer.py <command> <args>`
- Shell alias: `whisper-tuner <command> <args>` (if configured)
- CI/CD pipelines using modern CLI patterns
- Automated testing frameworks via Typer's testing utilities
- Docker containers for reproducible experiments

Calls to:
- core/config.py for profile and dataset configuration loading
- core/runs.py for experiment tracking and run management
- core/ops.py for delegated operation execution
- utils/device.py for MPS/CUDA/CPU device selection
- core/logging.py for structured logging initialization
- manage.py for legacy run management operations (via SimpleNamespace mapping)
- wizard.py for interactive configuration interface

Command structure:
- prepare: Dataset preparation and preprocessing
- finetune: Model training with profile-based configuration
- evaluate: Model evaluation on test datasets
- export: Model conversion to deployment formats
- blacklist: Quality-based sample filtering
- streaming: Real-time ASR inference
- runs: Run management and analysis (list, overview, details, cleanup)
- wizard: Interactive fine-tuning configuration wizard

Signal handling:
Implements POSIX signal handlers for graceful interruption:
- SIGINT (Ctrl-C): Marks run as cancelled, saves metadata, exits cleanly
- SIGTERM: Handles container/process manager termination requests

Device handling:
Automatically detects and configures optimal device settings:
- Apple Silicon (MPS): Optimized batch sizes and memory settings
- NVIDIA CUDA: GPU acceleration with appropriate dtype selection
- CPU fallback: Universal compatibility mode

Run management:
Every command creates a structured run directory containing:
- metadata.json: Complete experiment configuration and results
- run.log: Detailed execution logs for debugging
- Model checkpoints and evaluation outputs
- Performance metrics and timing information

Design principles:
- Type safety: Leverages Python type hints for validation
- Ergonomics: Intuitive command structure with rich help
- Compatibility: Maintains parity with legacy argparse CLI
- Extensibility: Easy addition of new commands via decorators
- Testability: Typer's testing utilities enable comprehensive CLI testing
"""

from __future__ import annotations

# Early Apple Silicon (MPS) bootstrap MUST run before any torch imports.
# It needs to be after the future import (per Python rules) but before
# any third-party imports that could pull in torch transitively.
#
# Called by: This import happens at module load time for cli_typer.py
# Calls: core.bootstrap module which sets up MPS environment variables
# Side effects: Sets PYTORCH_MPS_HIGH_WATERMARK_RATIO and related env vars
# Critical for: Preventing MPS memory pressure issues during training
import core.bootstrap  # noqa: F401  (early side-effects; deliberately unused)

import os
import json
from datetime import datetime
import signal
from typing import Optional

import typer

from wft_constants import FileSystem, LoggingDefaults
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
from core.runs import update_experiments_csv  # lightweight CSV experiment index
from core.runs import update_experiments_sqlite  # optional SQLite experiment index
from core import ops
from utils.device import get_device, apply_device_defaults, get_env_info


# Constants for AI-first documentation clarity
class ExitCodes:
    """Standard POSIX exit codes used throughout the CLI for consistent error reporting."""
    SUCCESS = 0
    GENERAL_ERROR = 1
    INTERRUPTED_BY_CTRL_C = 130  # Standard POSIX code for SIGINT termination

class OutputFormats:
    """Output formatting constants for consistent user experience across commands."""
    WER_DECIMAL_PRECISION = 3  # WER values displayed with 3 decimal places for readability
    
class DefaultPaths:
    """Default file paths used across multiple commands for consistency."""
    DEFAULT_OUTPUT_DIR = FileSystem.OUTPUT_DIR_DEFAULT
    DEFAULT_CONFIG_FILE = "config.ini"

# CLI Application Instance with Enhanced Help
# Creates the main Typer application that coordinates all subcommands
# Provides automatic help generation and command discovery
app = typer.Typer(
    help="Whisper Fine-Tuner (Typer): prepare, finetune, evaluate, export, blacklist",
    add_completion=False,  # Disable shell completion for cleaner interface
    rich_markup_mode="rich",  # Enable rich terminal formatting
)


def _normalize_device_defaults(profile_config: dict) -> None:
    """Apply device-specific configuration defaults for optimal performance.
    
    Called by:
    - finetune() before training execution
    - evaluate() before model evaluation
    - blacklist() before quality analysis
    
    Calls to:
    - utils/device.apply_device_defaults() for MPS/CUDA/CPU optimization
    
    Side effects:
    - Modifies profile_config in-place with device-optimal settings
    - Sets batch_size, gradient_accumulation, mixed_precision defaults
    """
    apply_device_defaults(profile_config)


@app.command()
def prepare(
    dataset: str = typer.Argument(..., help="Dataset name as defined in config.ini [dataset:*]"),
    config: str = typer.Option(DefaultPaths.DEFAULT_CONFIG_FILE, help="Path to configuration file"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
):
    """Prepare dataset for training by downloading and preprocessing audio files.
    
    This command handles the complete dataset preparation pipeline including
    audio download, format conversion, quality filtering, and split generation.
    
    Called by:
    - Direct CLI invocation: `whisper-tuner prepare <dataset>`
    - Batch dataset preparation scripts processing multiple datasets
    - CI/CD pipelines for automated dataset validation and preprocessing
    - Data ingestion workflows integrating new audio collections
    
    Calls to:
    - core.logging.init_logging() for logging configuration
    - core.ops.prepare() for dataset preparation workflow dispatch
    
    Args:
        dataset: Dataset name as defined in config.ini [dataset:*] sections
        config: Path to INI configuration file (default: config.ini)
        json_logging: Enable structured JSON logging format
        
    Side effects:
        - Creates data/datasets/{dataset}/ directory structure
        - Downloads and caches audio files in data/audio/
        - Generates prepared CSV files for training
    """
    init_logging(LoggingDefaults.DEFAULT_LEVEL, json_format=json_logging)
    ops.prepare({"dataset": dataset})


@app.command(name="prepare-granary")
def prepare_granary(
    profile: str = typer.Argument(..., help="Dataset profile name for Granary dataset (e.g., granary-en)"),
    config: str = typer.Option(DefaultPaths.DEFAULT_CONFIG_FILE, help="Path to configuration file"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
):
    """Prepare NVIDIA Granary dataset for training with optimized validation and streaming support.
    
    This command handles NVIDIA Granary dataset preparation with specialized audio source
    management and optimized validation workflows for multi-source datasets.
    
    Called by:
    - Direct CLI invocation: `whisper-tuner prepare-granary <profile>`
    - Specialized dataset preparation workflows requiring Granary-specific handling
    - Research workflows using NVIDIA Granary multi-domain datasets
    
    Calls to:
    - core.logging.init_logging() for logging configuration
    - scripts.prepare_granary.prepare_granary() for Granary-specific preparation
    
    Args:
        profile: Dataset profile name (e.g., granary-en) as defined in config.ini
        config: Path to INI configuration file (default: config.ini)
        json_logging: Enable structured JSON logging format
        
    Side effects:
        - Creates data/datasets/granary-*/ directory structures
        - Downloads and processes multi-source audio collections
        - Generates Granary-specific manifest and CSV files
    """
    init_logging(LoggingDefaults.DEFAULT_LEVEL, json_format=json_logging)
    
    # Import here to avoid circular dependency
    from scripts.prepare_granary import prepare_granary as _prepare_granary
    
    try:
        manifest_path = _prepare_granary(profile)
        typer.echo(f"✅ Granary dataset prepared successfully!")
        typer.echo(f"📄 Manifest: {manifest_path}")
        typer.echo(f"🎯 Ready for training with profile: {profile}")
    except Exception as e:
        typer.echo(f"❌ Granary preparation failed: {e}", err=True)
        raise typer.Exit(ExitCodes.GENERAL_ERROR)


@app.command()
def finetune(
    profile: str = typer.Argument(..., help="Profile name as defined under [profile:*]"),
    config: str = typer.Option(DefaultPaths.DEFAULT_CONFIG_FILE, help="Path to configuration file"),
    max_samples: Optional[int] = typer.Option(None, help="Max samples for quick runs"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
    log_file: Optional[str] = typer.Option(None, help="Optional file path for logs"),
):
    """Execute model fine-tuning with comprehensive device optimization and run management.
    
    This command handles the complete fine-tuning pipeline including model loading,
    dataset preparation, training execution, and checkpoint management with full
    Apple Silicon MPS optimization and device-agnostic configuration.
    
    Called by:
    - Direct CLI invocation: `whisper-tuner finetune <profile>`
    - wizard.py:execute_training() for interactive training workflows
    - Batch training automation scripts for systematic model comparison
    - Hyperparameter sweep frameworks iterating over configurations
    - CI/CD pipelines executing scheduled training runs
    
    Calls to:
    - core.logging.init_logging() and add_file_handler() for logging setup
    - _load_config() for configuration file parsing
    - core.runs.get_next_run_id(), create_run_directory() for run management
    - load_profile_config() for profile configuration resolution
    - _normalize_device_defaults() for device-specific optimization
    - core.ops.finetune() for training execution dispatch
    - update_run_metadata(), mark_run_as_completed() for run tracking
    
    Args:
        profile: Training profile name as defined in config.ini [profile:*] sections
        config: Path to INI configuration file (default: config.ini)
        max_samples: Optional sample limit for quick training runs
        json_logging: Enable structured JSON logging format
        log_file: Optional custom log file path (default: run_dir/run.log)
        
    Side effects:
        - Creates structured run directory with metadata and logs
        - Saves model checkpoints and training artifacts
        - Updates experiments CSV and SQLite databases
        - Modifies profile configuration in-place with device defaults
        
    Signal handling:
        - SIGINT/SIGTERM: Graceful cancellation with metadata preservation
        - Run marked as cancelled with proper cleanup and exit codes
    """
    init_logging(LoggingDefaults.DEFAULT_LEVEL, json_format=json_logging)
    cfg = _load_config(config)
    output_dir = cfg["DEFAULT"]["output_dir"]
    run_id = get_next_run_id(output_dir)
    run_dir = create_run_directory(output_dir, profile, run_id, "finetuning")
    add_file_handler(log_file or os.path.join(run_dir, LoggingDefaults.RUN_LOG_FILENAME), json_format=json_logging)

    def _handle_signal(signum, frame):
        """Signal handler that marks the run as cancelled and exits.

        Called by:
        - POSIX signal delivery (SIGINT/SIGTERM) while a Typer command is running

        Calls to:
        - update_run_metadata() to persist cancellation status and end time

        Side effects:
        - Updates `metadata.json` in the run directory with status="cancelled"
        - Exits the process with POSIX code 130 (terminated by Ctrl-C)
        """
        try:
            update_run_metadata(run_dir, status="cancelled", end_time=_now())
        finally:
            raise SystemExit(ExitCodes.INTERRUPTED_BY_CTRL_C)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        profile_config = load_profile_config(cfg, profile)
        if max_samples is not None:
            profile_config["max_samples"] = max_samples
        _normalize_device_defaults(profile_config)
        update_run_metadata(
            run_dir,
            config=profile_config,
            model=profile_config["model"],
            dataset=profile_config["dataset"],
            env=get_env_info(),
        )
        ops.finetune(profile_config, run_dir)
        _maybe_merge_train_metrics(run_dir)
        mark_run_as_completed(run_dir)
        update_run_metadata(run_dir, status="completed", end_time=_now())
        try:
            update_experiments_csv(output_dir, run_dir)
            update_experiments_sqlite(output_dir, run_dir)
        except Exception:
            pass
    except Exception as e:
        update_run_metadata(run_dir, status="failed", end_time=_now(), error_message=str(e))
        try:
            update_experiments_csv(output_dir, run_dir)
            update_experiments_sqlite(output_dir, run_dir)
        except Exception:
            pass
        raise


@app.command()
def evaluate(
    target: str = typer.Argument(..., help="Profile name or model+dataset (e.g., whisper-tiny+test_streaming)"),
    config: str = typer.Option(DefaultPaths.DEFAULT_CONFIG_FILE, help="Path to configuration file"),
    dataset: Optional[str] = typer.Option(None, help="Dataset override when using a profile"),
    max_samples: Optional[int] = typer.Option(None, help="Max samples for quick runs"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
    log_file: Optional[str] = typer.Option(None, help="Optional file path for logs"),
):
    """Evaluate fine-tuned Whisper models with comprehensive metrics and device optimization.
    
    This command provides comprehensive evaluation capabilities supporting multiple evaluation
    modes, metrics calculation, and detailed analysis reporting with optimizations for
    Apple Silicon, CUDA, and CPU platforms.

    Called by:
    - Direct CLI invocation: `whisper-tuner evaluate <target>`
    - Post-training evaluation workflows triggered after training completion
    - Model comparison and benchmarking workflows evaluating multiple checkpoints
    - Quality assurance pipelines validating performance against thresholds
    - CI/CD pipelines executing automated model validation and regression testing

    Calls to:
    - core.logging.init_logging() and add_file_handler() for logging setup
    - _load_config() for configuration file parsing
    - core.runs.get_next_run_id(), create_run_directory() for run management
    - load_model_dataset_config() or load_profile_config() for configuration resolution
    - core.runs.find_latest_completed_finetuning_run() for checkpoint location
    - _normalize_device_defaults() for device-specific optimization
    - core.ops.evaluate() for evaluation execution dispatch
    - update_run_metadata(), mark_run_as_completed() for run tracking

    Evaluation modes:
    - Profile-based: Evaluate latest fine-tuned model from profile training
    - Model+dataset: Direct evaluation using model+dataset syntax (e.g., whisper-base+librispeech)
    - Cross-dataset: Evaluate trained model on alternative datasets

    Args:
        target: Profile name or model+dataset combination for evaluation
        config: Path to INI configuration file (default: config.ini)
        dataset: Dataset override when evaluating a profile on different data
        max_samples: Optional sample limit for quick evaluation runs
        json_logging: Enable structured JSON logging format
        log_file: Optional custom log file path (default: run_dir/run.log)
        
    Side effects:
        - Creates structured evaluation run directory with metrics and predictions
        - Generates detailed CSV files with prediction analysis
        - Updates experiments database with evaluation results
        - Applies device-specific optimizations for evaluation platform
    """
    init_logging(LoggingDefaults.DEFAULT_LEVEL, json_format=json_logging)
    cfg = _load_config(config)
    output_dir = cfg["DEFAULT"]["output_dir"]
    run_id = get_next_run_id(output_dir)

    if "+" in target:
        model_name, dataset_name = target.split("+")
        profile_config = load_model_dataset_config(cfg, model_name, dataset_name)
        run_dir = create_run_directory(output_dir, None, run_id, "evaluation", model_name=model_name, dataset_name=dataset_name)
        profile_config["model_name_or_path"] = profile_config["base_model"]
    else:
        profile_name = target
        profile_config = load_profile_config(cfg, profile_name)
        if dataset:
            run_dir = create_run_directory(output_dir, profile_name, run_id, "evaluation", dataset_name=dataset)
            dataset_name = dataset
        else:
            run_dir = create_run_directory(output_dir, profile_name, run_id, "evaluation")
            dataset_name = profile_config["dataset"]
        latest = find_latest_completed_finetuning_run(output_dir, profile_name) or find_latest_finetuning_run(output_dir, profile_name)
        if not latest:
            raise FileNotFoundError(f"No fine-tuning runs found for profile '{profile_name}'. Train before evaluating.")
        profile_config["model_name_or_path"] = latest

    add_file_handler(log_file or os.path.join(run_dir, LoggingDefaults.RUN_LOG_FILENAME), json_format=json_logging)
    update_run_metadata(run_dir, config=profile_config, env=get_env_info())
    profile_config["force_languages"] = profile_config.get("force_languages", False)
    profile_config["languages"] = profile_config.get("languages", "all")
    profile_config["dataset"] = dataset_name
    if max_samples is not None:
        profile_config["max_samples"] = max_samples
    _normalize_device_defaults(profile_config)

    metrics = ops.evaluate(profile_config, run_dir)
    if metrics:
        update_run_metadata(run_dir, metrics=metrics)
        _write_metrics(run_dir, metrics)
    mark_run_as_completed(run_dir)
    update_run_metadata(run_dir, status="completed", end_time=_now())
    try:
        update_experiments_csv(output_dir, run_dir)
        update_experiments_sqlite(output_dir, run_dir)
    except Exception:
        pass


@app.command()
def export(model_path_or_profile: str = typer.Argument(..., help="Model path or HF id to export")):
    """Export a model to a portable HF/SafeTensors directory.

    Called by:
    - `whisper-tuner export <model_path_or_hub_id>`

    Calls to:
    - core.ops.export() which defers to scripts.export.export_model_dir()

    Args:
        model_path_or_profile: Local model directory or Hugging Face model id
    """
    ops.export(model_path_or_profile)


@app.command()
def blacklist(
    profile: str = typer.Argument(..., help="Profile to generate blacklist for"),
    config: str = typer.Option(DefaultPaths.DEFAULT_CONFIG_FILE, help="Path to configuration file"),
    split: Optional[str] = typer.Option(None, help="Split override for blacklist generation"),
    max_samples: Optional[int] = typer.Option(None, help="Max samples for quick runs"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
    log_file: Optional[str] = typer.Option(None, help="Optional file path for logs"),
):
    """Generate intelligent WER-based blacklist for training dataset quality improvement.
    
    This command implements automated outlier detection to identify mislabeled or
    problematic audio samples in training datasets. It generates blacklists while
    respecting manual overrides and providing detailed diagnostic information.

    Called by:
    - Direct CLI invocation: `whisper-tuner blacklist <profile>`
    - Training pipelines requiring iterative dataset quality improvement
    - Quality assurance workflows maintaining dataset integrity
    - Active learning systems identifying samples requiring manual review
    - Dataset maintenance pipelines for automated quality filtering

    Calls to:
    - core.logging.init_logging() for logging configuration  
    - _load_config() for configuration file parsing
    - core.runs.get_next_run_id(), create_run_directory() for run management
    - core.runs.find_latest_finetuning_run() for model checkpoint location
    - load_profile_config() for profile configuration resolution
    - _normalize_device_defaults() for device-specific optimization
    - scripts.blacklist.create_blacklist() for blacklist generation analysis

    Args:
        profile: Training profile name used to locate the fine-tuned model
        config: Path to INI configuration file (default: config.ini)
        split: Dataset split to analyze (defaults to profile's train split)
        max_samples: Optional sample cap for quick blacklist analysis
        json_logging: Enable structured JSON logging format
        log_file: Optional custom log file path (default: run_dir/run.log)
        
    Side effects:
        - Creates blacklist CSV files with detailed diagnostic information
        - Updates run metadata with blacklist generation results
        - Integrates with data patch system for manual override handling
        - Provides statistical summary of dataset quality assessment
    """
    init_logging(LoggingDefaults.DEFAULT_LEVEL, json_format=json_logging)
    cfg = _load_config(config)
    output_dir = cfg["DEFAULT"]["output_dir"]
    run_id = get_next_run_id(output_dir)
    run_dir = create_run_directory(output_dir, profile, run_id, "blacklist")
    add_file_handler(log_file or os.path.join(run_dir, LoggingDefaults.RUN_LOG_FILENAME), json_format=json_logging)

    finetuning_run_dir = find_latest_finetuning_run(output_dir, profile)
    if not finetuning_run_dir:
        raise FileNotFoundError(f"No completed fine-tuning run found for profile '{profile}'.")

    profile_config = load_profile_config(cfg, profile)
    profile_config["split"] = split if split else profile_config["train_split"]
    profile_config["model_name_or_path"] = finetuning_run_dir
    if max_samples is not None:
        profile_config["max_samples"] = max_samples
    _normalize_device_defaults(profile_config)

    from scripts.blacklist import create_blacklist
    path = create_blacklist(profile_config, run_dir)
    typer.echo(f"Blacklist created at: {path}")


# -------- Runs management (unified Typer wrapper for manage.py) --------
runs_app = typer.Typer(help="Manage, list, and inspect training/evaluation runs")


@runs_app.command("list")
def runs_list(
    type: Optional[str] = typer.Option(None, "--type", help="Filter by run type"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Exact profile name filter"),
    model: Optional[str] = typer.Option(None, "--model", help="Substring model filter"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Substring dataset filter"),
    finetuning_run_id: Optional[str] = typer.Option(None, "--finetuning-run-id", help="Link evaluations to a train run"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date YYYY-MM-DD"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date YYYY-MM-DD"),
    min_wer: Optional[float] = typer.Option(None, "--min-wer", help="Minimum WER"),
    max_wer: Optional[float] = typer.Option(None, "--max-wer", help="Maximum WER"),
    include_failed: bool = typer.Option(False, "--include-failed", help="Include failed runs"),
    output_dir: str = typer.Option(DefaultPaths.DEFAULT_OUTPUT_DIR, "--output-dir", help="Root runs directory"),
):
    """List training and evaluation runs with filtering in a formatted table.
    
    This command provides a unified CLI interface to the run management system,
    wrapping the legacy manage.py functionality with modern Typer argument handling.
    
    Called by:
    - Users investigating training results and comparing runs
    - CI/CD pipelines generating run reports
    - `whisper-tuner runs overview` for aggregated statistics
    
    Calls:
    - `manage.list_runs()` via SimpleNamespace parameter mapping
    - Internally uses run metadata from {output_dir}/*/metadata.json files
    - Filters runs based on provided criteria using pandas operations
    
    Data flow:
    1. Converts Typer arguments to SimpleNamespace for manage.py compatibility
    2. manage.list_runs() scans output_dir for run directories
    3. Loads and filters metadata.json files per criteria
    4. Outputs formatted table via rich/tabulate
    
    Args:
        type: Filter by run type ('train', 'eval', 'export'). Case-insensitive matching.
        profile: Exact match filter for training profile name from config files
        model: Substring filter for model names (e.g., 'whisper-base', 'distil')
        dataset: Substring filter for dataset names used in training/eval
        finetuning_run_id: Show only evaluations linked to specific training run
        from_date: Include runs from this date onwards (YYYY-MM-DD format)
        to_date: Include runs up to this date (YYYY-MM-DD format)  
        min_wer: Filter runs with WER >= this value (for evaluation runs)
        max_wer: Filter runs with WER <= this value (for evaluation runs)
        include_failed: Include runs with status='failed' or incomplete metadata
        output_dir: Root directory containing run subdirectories (default: ./output)
    
    Output format:
    - Tabulated list showing: run_id, type, model, dataset, WER, status, date
    - Color-coded status indicators (green=success, red=failed, yellow=running)
    - WER values displayed with {OutputFormats.WER_DECIMAL_PRECISION} decimal precision for evaluation runs
    """
    from types import SimpleNamespace
    from manage import list_runs
    
    # SimpleNamespace Bridge Pattern for Legacy Integration:
    # The manage.py module expects argparse Namespace objects with specific attribute names.
    # We convert Typer's typed parameters to a SimpleNamespace that manage.py can consume.
    # This pattern maintains backward compatibility while providing modern CLI ergonomics.
    ns = SimpleNamespace(
        type=type,
        profile=profile,
        model=model,
        dataset=dataset,
        finetuning_run_id=finetuning_run_id,
        from_date=from_date,
        to_date=to_date,
        min_wer=min_wer,
        max_wer=max_wer,
        include_failed=include_failed,
    )
    # Let manage.py do its own date parsing and print formatting
    list_runs(ns, output_dir=output_dir)


@runs_app.command()
def overview(
    type: Optional[str] = typer.Option(None, "--type", help="Filter by run type"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Exact profile name filter"),
    model: Optional[str] = typer.Option(None, "--model", help="Substring model filter"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Substring dataset filter"),
    finetuning_run_id: Optional[str] = typer.Option(None, "--finetuning-run-id", help="Filter by train run ID"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date YYYY-MM-DD"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date YYYY-MM-DD"),
    min_wer: Optional[float] = typer.Option(None, "--min-wer", help="Minimum WER"),
    max_wer: Optional[float] = typer.Option(None, "--max-wer", help="Maximum WER"),
    include_failed: bool = typer.Option(False, "--include-failed", help="Include failed runs"),
    output_dir: str = typer.Option(DefaultPaths.DEFAULT_OUTPUT_DIR, "--output-dir", help="Root runs directory"),
):
    """Generate statistical overview and identify best-performing runs.
    
    This command provides aggregated analytics across all runs matching the filter criteria,
    showing performance trends, success rates, and highlighting top performers.
    
    Called by:
    - Users analyzing training effectiveness across multiple experiments
    - CI/CD pipelines generating performance reports
    - Research workflows comparing different model configurations
    
    Calls:
    - `manage.overview()` via SimpleNamespace parameter mapping
    - Internally aggregates metadata.json files using pandas groupby operations
    - Calculates statistics like mean WER, success rates, training duration trends
    
    Data flow:
    1. Applies same filtering logic as runs_list()
    2. Aggregates metrics by model, dataset, and profile combinations
    3. Identifies best runs by lowest WER within each category
    4. Outputs formatted statistical summaries and recommendations
    
    Args:
        type: Filter by run type ('train', 'eval', 'export') for targeted analysis
        profile: Exact match filter for specific training configuration profiles
        model: Focus analysis on specific model variants (substring matching)
        dataset: Analyze performance on specific datasets (substring matching)
        finetuning_run_id: Show evaluation overview for specific training run
        from_date: Include runs from this date onwards (YYYY-MM-DD format)
        to_date: Include runs up to this date (YYYY-MM-DD format)
        min_wer: Focus on runs with WER >= threshold (identify problem areas)
        max_wer: Focus on runs with WER <= threshold (identify successes)
        include_failed: Include failed runs in statistical analysis
        output_dir: Root directory containing run subdirectories
    
    Output sections:
    - Summary statistics: total runs, success rate, mean/median WER
    - Best performers: lowest WER runs by category
    - Trend analysis: performance over time, if date range specified
    - Recommendations: suggested next experiments based on results
    """
    from types import SimpleNamespace
    from manage import overview as _overview
    ns = SimpleNamespace(
        type=type,
        profile=profile,
        model=model,
        dataset=dataset,
        finetuning_run_id=finetuning_run_id,
        from_date=from_date,
        to_date=to_date,
        min_wer=min_wer,
        max_wer=max_wer,
        include_failed=include_failed,
    )
    _overview(ns, output_dir=output_dir)


@runs_app.command()
def details(
    run_id: str = typer.Argument(..., help="Run ID to display"),
    output_dir: str = typer.Option(DefaultPaths.DEFAULT_OUTPUT_DIR, "--output-dir", help="Root runs directory"),
):
    """Display detailed JSON metadata and logs for a specific training/evaluation run.
    
    This command provides deep inspection into a single run's configuration, metrics,
    and execution details for debugging and analysis purposes.
    
    Called by:
    - Users debugging failed runs or investigating specific results
    - CI/CD pipelines extracting metrics for reporting
    - Research workflows analyzing hyperparameter effects
    
    Calls:
    - `manage.details()` via SimpleNamespace parameter mapping
    - Loads {output_dir}/{run_id}/metadata.json for structured data
    - Optionally displays logs from {output_dir}/{run_id}/training.log
    
    Data flow:
    1. Validates run_id exists in output_dir
    2. Loads and pretty-prints metadata.json with syntax highlighting
    3. Shows key metrics summary (WER, loss, duration) if available
    4. Displays recent log entries for context
    
    Args:
        run_id: Unique identifier for the run (e.g., 'train_20240119_143022')
        output_dir: Root directory containing run subdirectories
        
    Output format:
    - JSON metadata with syntax highlighting and collapsible sections
    - Key metrics summary table for quick reference
    - Recent log entries showing training progress or error details
    - File size information for model artifacts and datasets
    """
    from types import SimpleNamespace
    from manage import details as _details
    ns = SimpleNamespace(run_id=run_id)
    _details(ns, output_dir=output_dir)


@runs_app.command()
def cleanup(
    output_dir: str = typer.Option(DefaultPaths.DEFAULT_OUTPUT_DIR, "--output-dir", help="Root runs directory"),
):
    """Delete failed, cancelled, or incomplete runs to free disk space.
    
    This command performs safe cleanup of runs that failed during execution,
    were manually cancelled, or left incomplete due to system issues.
    
    Called by:
    - Users managing disk space after multiple training experiments
    - CI/CD pipelines cleaning up after test runs
    - Automated maintenance scripts on training servers
    
    Calls:
    - `manage.cleanup()` via SimpleNamespace parameter mapping
    - Scans {output_dir}/*/metadata.json for status='failed' or incomplete runs
    - Uses safe deletion patterns to avoid removing valid partial results
    
    Data flow:
    1. Scans all run directories for status indicators
    2. Identifies failed runs based on metadata.json status field
    3. Identifies incomplete runs missing required artifacts
    4. Prompts user for confirmation before deletion
    5. Removes run directories and reports disk space freed
    
    Args:
        output_dir: Root directory containing run subdirectories to clean
        
    Safety measures:
    - Only deletes runs with explicit 'failed' status or missing critical files
    - Preserves runs with 'running' status that might resume
    - Shows detailed list of runs to be deleted before confirmation
    - Creates backup of metadata.json files before deletion
    
    Output:
    - List of runs identified for cleanup with reasons
    - Confirmation prompt with total disk space to be freed
    - Summary of cleanup results and remaining run count
    """
    from types import SimpleNamespace
    from manage import cleanup as _cleanup
    ns = SimpleNamespace()  # no args needed currently
    _cleanup(ns, output_dir=output_dir)


app.add_typer(runs_app, name="runs")


@app.command(name="wizard")
def run_wizard() -> None:
    """Launch the interactive fine-tuning wizard for guided model training setup.
    
    This command starts a user-friendly, step-by-step interface that guides users
    through the complete process of configuring and launching fine-tuning jobs.
    
    Called by:
    - New users learning the fine-tuning workflow
    - Users preferring interactive configuration over command-line arguments
    - Automated scripts that need guided parameter selection
    
    Calls:
    - `wizard.wizard_main()` as the primary interactive interface
    - Internally chains through configuration, dataset, and model selection
    - May spawn training processes via core.finetune modules
    
    Data flow:
    1. Presents interactive prompts for configuration selection
    2. Validates user choices against available datasets and models
    3. Generates configuration files in standard format
    4. Optionally launches training immediately or saves for later execution
    
    Features:
    - Interactive model selection with compatibility checking
    - Dataset validation and preprocessing recommendations  
    - Hardware detection and optimization suggestions (MPS, CUDA, CPU)
    - Configuration file generation for reproducible runs
    - Integration with existing run management system
    
    Error handling:
    - ImportError: wizard module not available (missing dependencies)
    - User interruption (Ctrl+C): graceful exit with partial configuration saved
    - Invalid selections: retry prompts with helpful guidance
    
    Note: This is a legacy interface maintained for backward compatibility.
    Consider using direct CLI commands for automation and CI/CD workflows.
    """
    try:
        from wizard import wizard_main
        wizard_main()
    except Exception as e:
        typer.echo(f"❌ Wizard failed: {e}", err=True)
        typer.echo("Tip: For the modern CLI, use 'whisper-tuner <command>'.", err=True)
        raise typer.Exit(ExitCodes.GENERAL_ERROR)


# -------- Legacy commands wrapper --------
legacy_app = typer.Typer(help="Legacy interfaces kept for backward compatibility")


@legacy_app.command("manage")
def legacy_manage():
    """Run legacy manage.py main() if available."""
    try:
        import manage as _manage
        if hasattr(_manage, "main"):
            _manage.main()
        else:
            typer.echo("manage.py has no main(); use 'runs' commands instead.")
    except Exception as e:
        typer.echo(f"❌ legacy manage failed: {e}", err=True)
        raise typer.Exit(ExitCodes.GENERAL_ERROR)


@legacy_app.command("main")
def legacy_main():
    """Run legacy main.py entrypoint if available (deprecated)."""
    try:
        import main as _main
        if hasattr(_main, "main"):
            _main.main()
        else:
            typer.echo("main.py has no main(); use modern CLI commands instead.")
    except Exception as e:
        typer.echo(f"❌ legacy main failed: {e}", err=True)
        raise typer.Exit(ExitCodes.GENERAL_ERROR)


app.add_typer(legacy_app, name="legacy")


@app.command(name="system-check")
def system_check():
    """Generate comprehensive system compatibility and diagnostics report.

    This command provides thorough system validation for Whisper fine-tuning environments,
    verifying hardware compatibility, software versions, and configuration settings across
    Apple Silicon, NVIDIA CUDA, and CPU platforms.

    Called by:
    - Direct CLI invocation: `whisper-tuner system-check`
    - CI/CD pipelines for automated compatibility testing and validation
    - Troubleshooting workflows for deployment and environment setup issues
    - Pre-training validation to ensure optimal system configuration
    - Development environment setup verification and optimization

    Calls to:
    - utils.device.get_env_info() for comprehensive environment information
    - utils.device.get_device_info() for device capability assessment
    - utils.device.verify_mps_setup() for Apple Silicon MPS validation
    - Platform detection libraries for hardware identification

    Validation categories:
    - Hardware: Device type detection (MPS/CUDA/CPU) and capability reporting
    - Software: PyTorch, Transformers, Datasets library version verification
    - Environment: OS version, Python architecture, conda/pip configuration
    - Performance: Memory availability, MPS compatibility, optimization status

    Output format:
    - Concise human-readable summary suitable for bug reports and CI logs
    - Clear pass/fail indicators for critical system components
    - MPS-specific diagnostics for Apple Silicon compatibility
    - Optimization recommendations and troubleshooting guidance
    """
    try:
        import platform as _platform
        import torch as _torch  # type: ignore
    except Exception:
        _torch = None  # type: ignore
        _platform = None  # type: ignore

    from utils.device import get_env_info, get_device_info, verify_mps_setup

    info = get_env_info()
    dev = get_device_info()

    typer.echo("🧰 System Check")
    typer.echo(f"Python: {info.get('python')}")
    typer.echo(f"OS: {info.get('os')}")
    typer.echo(f"PyTorch: {info.get('torch')}")
    typer.echo(f"Transformers: {info.get('transformers')}")
    typer.echo(f"Datasets: {info.get('datasets')}")
    typer.echo(f"NumPy: {info.get('numpy')}")
    typer.echo(f"Device: {dev.get('device')} ({dev.get('description', dev.get('device_type'))})")

    # MPS details if relevant
    if dev.get("device_type") == "mps":
        ok, msg = verify_mps_setup()
        status = "OK" if ok else "ISSUE"
        typer.echo(f"MPS: {status} - {msg}")

    typer.echo("✅ system-check completed")

def _load_config(path: str):
    """Load an INI configuration into ConfigParser with no side effects.

    Called by:
    - Typer commands before delegating to core/main logic

    Returns:
        ConfigParser: Parsed configuration object
    """
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def _write_metrics(run_dir: str, metrics: dict) -> None:
    """Persist metrics into `metrics.json` inside a run directory.

    Called by:
    - evaluate() after successful evaluation

    Args:
        run_dir: Target run directory path
        metrics: Metrics dictionary to write
    """
    try:
        from core.runs import write_metrics
        write_metrics(run_dir, metrics)
    except Exception:
        pass


def _maybe_merge_train_metrics(run_dir: str) -> None:
    """Merge `train_results.json` into metrics for consolidated tracking.

    Called by:
    - finetune() after training completes, when train_results.json exists
    """
    try:
        results_path = os.path.join(run_dir, "train_results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as rf:
                train_metrics = json.load(rf)
            from core.runs import write_metrics
            write_metrics(run_dir, {"train": train_metrics})
    except Exception:
        pass


def _now() -> str:
    """Return current timestamp in the standard metadata format (UTC-local)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    app()
