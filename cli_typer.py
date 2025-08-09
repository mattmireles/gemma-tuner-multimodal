#!/usr/bin/env python3
"""
Optional Typer-based CLI wrapper for Whisper Fine-Tuner.

This is an opt-in interface that mirrors the argparse CLI in main.py but with
a more ergonomic developer experience. It delegates to the same core modules
(core/config, core/runs, core/ops, utils/device) to avoid logic drift.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
import signal
from typing import Optional

import typer

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


app = typer.Typer(help="Whisper Fine-Tuner (Typer): prepare, finetune, evaluate, export, blacklist")


def _normalize_device_defaults(profile_config: dict) -> None:
    apply_device_defaults(profile_config)


@app.command()
def prepare(
    dataset: str = typer.Argument(..., help="Dataset name as defined in config.ini [dataset:*]"),
    config: str = typer.Option("config.ini", help="Path to configuration file"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
):
    init_logging("INFO", json_format=json_logging)
    ops.prepare({"dataset": dataset})


@app.command()
def finetune(
    profile: str = typer.Argument(..., help="Profile name as defined under [profile:*]"),
    config: str = typer.Option("config.ini", help="Path to configuration file"),
    max_samples: Optional[int] = typer.Option(None, help="Max samples for quick runs"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
    log_file: Optional[str] = typer.Option(None, help="Optional file path for logs"),
):
    init_logging("INFO", json_format=json_logging)
    cfg = _load_config(config)
    output_dir = cfg["DEFAULT"]["output_dir"]
    run_id = get_next_run_id(output_dir)
    run_dir = create_run_directory(output_dir, profile, run_id, "finetuning")
    add_file_handler(log_file or os.path.join(run_dir, "run.log"), json_format=json_logging)

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
            raise SystemExit(130)

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
    except Exception as e:
        update_run_metadata(run_dir, status="failed", end_time=_now(), error_message=str(e))
        raise


@app.command()
def evaluate(
    target: str = typer.Argument(..., help="Profile name or model+dataset (e.g., whisper-tiny+test_streaming)"),
    config: str = typer.Option("config.ini", help="Path to configuration file"),
    dataset: Optional[str] = typer.Option(None, help="Dataset override when using a profile"),
    max_samples: Optional[int] = typer.Option(None, help="Max samples for quick runs"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
    log_file: Optional[str] = typer.Option(None, help="Optional file path for logs"),
):
    """Run model evaluation from the Typer CLI with LLM-first clarity.

    Called by:
    - `whisper-tuner evaluate <profile>` for profile-based evaluation
    - `whisper-tuner evaluate <model+dataset>` for direct model/dataset evaluation

    Calls to:
    - load_profile_config()/load_model_dataset_config() to resolve configuration
    - create_run_directory() to establish evaluation run structure
    - core.ops.evaluate() to execute the evaluation pipeline

    Args:
        target: Profile name or "model+dataset" (e.g., "whisper-small+data3")
        config: Path to INI file used for configuration resolution
        dataset: Optional dataset override when a profile name is supplied
        max_samples: Optional sample cap for quick/CI runs
        json_logging: Enable JSON log output for aggregation systems
        log_file: Optional additional file sink for logs
    """
    init_logging("INFO", json_format=json_logging)
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

    add_file_handler(log_file or os.path.join(run_dir, "run.log"), json_format=json_logging)
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
    config: str = typer.Option("config.ini", help="Path to configuration file"),
    split: Optional[str] = typer.Option(None, help="Split override for blacklist generation"),
    max_samples: Optional[int] = typer.Option(None, help="Max samples for quick runs"),
    json_logging: bool = typer.Option(False, "--json-logging", help="Enable JSON logs"),
    log_file: Optional[str] = typer.Option(None, help="Optional file path for logs"),
):
    """Generate a WER-based blacklist for a given training profile.

    Called by:
    - `whisper-tuner blacklist <profile>`

    Calls to:
    - load_profile_config() to resolve profile parameters
    - core.runs helpers to create the run directory
    - scripts.blacklist.create_blacklist() to perform analysis and write CSV

    Args:
        profile: Training profile name used to locate the finetuned model
        config: Path to INI configuration
        split: Dataset split to analyze (defaults to profile's train split)
        max_samples: Optional sample cap for quick analysis
        json_logging: Enable JSON logs
        log_file: Optional file sink path for logs
    """
    init_logging("INFO", json_format=json_logging)
    cfg = _load_config(config)
    output_dir = cfg["DEFAULT"]["output_dir"]
    run_id = get_next_run_id(output_dir)
    run_dir = create_run_directory(output_dir, profile, run_id, "blacklist")
    add_file_handler(log_file or os.path.join(run_dir, "run.log"), json_format=json_logging)

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


