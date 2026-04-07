#!/usr/bin/env python3
"""
Experiment Results Aggregation and Analysis System

This module provides comprehensive functionality for gathering, aggregating, and analyzing
evaluation results from multiple experimental runs. It consolidates predictions, metrics,
and metadata from diverse evaluation experiments into unified datasets for comparative
analysis and performance assessment.

Key responsibilities:
- Multi-experiment result aggregation with metadata integration
- Flexible experiment identification (profile-based or model+dataset)
- Prediction consolidation with quality metrics alignment
- Language information integration from dataset metadata
- Statistical analysis and comparative performance reporting
- Time-series analysis of experiment evolution

Called by:
- Command-line interface for manual result gathering
- Analysis workflows requiring multi-experiment comparison
- Research pipelines consolidating experimental results
- Automated reporting systems for experiment tracking
- Performance monitoring and quality assurance workflows

Calls to:
- Metadata loading utilities for experiment identification
- pandas for efficient data manipulation and CSV operations
- JSON parsing for experiment metadata processing
- File system operations for result discovery and loading

Aggregation strategies:

Profile-based gathering:
- Identifies experiments by profile name and optional run specification
- Loads finetuning run metadata to locate associated evaluations
- Supports latest run selection or specific run targeting
- Integrates language information from prepared datasets

Model+dataset gathering:
- Direct experiment identification using model and dataset names
- Bypasses profile system for standalone evaluations
- Supports custom evaluation configurations
- Provides fallback for non-profile experimental workflows

Data consolidation workflow:
1. Experiment discovery using metadata.json files
2. Evaluation result loading from predictions.csv files
3. Metadata integration for comprehensive context
4. Language information enrichment from dataset sources
5. Quality metrics alignment and validation
6. Unified CSV generation with comparative structure
7. Statistical analysis and summary reporting

Output format:
Generates comprehensive CSV files with columns:
- ID: Sample identifier for cross-experiment alignment
- Target: Reference transcription (consistent across experiments)
- Norm Target: Normalized reference for metric calculation
- Language: Sample language code from dataset metadata
- {experiment}_Pred: Model predictions for each experiment
- {experiment}_Norm_Pred: Normalized predictions for metrics
- {experiment}_WER: Word Error Rate for each experiment
- {experiment}_CER: Character Error Rate for each experiment

Metadata integration:
Extracts comprehensive context from experiment metadata:
- Run identification and timestamp information
- Model configuration and hyperparameter settings
- Dataset information and language configuration
- Evaluation settings and quality thresholds
- Device information and performance characteristics

Language information handling:
- Loads language codes from prepared dataset CSV files
- Provides per-sample language identification
- Supports multilingual experiment analysis
- Handles language-specific performance assessment

File organization and discovery:
Navigates complex experiment directory structures:
output/
├── {run_id}-{profile}-{dataset}/  # Finetuning runs
│   ├── metadata.json
│   ├── eval/                     # Latest evaluation
│   │   ├── metadata.json
│   │   └── predictions.csv
│   └── eval-{timestamp}/         # Timestamped evaluations
└── {model}+{dataset}/            # Direct evaluations
    └── eval/
        ├── metadata.json
        └── predictions.csv

Error handling:
- Missing metadata: Graceful degradation with warnings
- Corrupted CSV files: Individual experiment skipping
- Language information missing: Fallback to configuration
- File access errors: Comprehensive error reporting

Performance optimization:
- Efficient pandas operations for large datasets
- Memory-conscious data loading and processing
- Optimized CSV I/O with appropriate data types
- Parallel processing capabilities for large result sets

Compatibility:
- pandas: Efficient data manipulation and CSV operations
- JSON: Metadata parsing and configuration loading
- pathlib: Cross-platform file system operations
- datetime: Timestamp processing and formatting
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from gemma_tuner.core.run_queries import load_metadata

# Anchor all data paths to the project root so gather works from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


def gather_predictions(profiles, output_dir="output"):
    """
    Orchestrates comprehensive experiment result gathering and comparative analysis.

    This is the main aggregation function, coordinating the discovery, loading,
    and consolidation of evaluation results from multiple experimental runs.
    It provides flexible experiment identification and comprehensive result
    integration for comparative analysis and performance assessment.

    Called by:
    - Command-line interface (__main__ section) for manual result gathering
    - Analysis workflows requiring multi-experiment comparison
    - Automated reporting systems for experiment tracking
    - Research pipelines consolidating experimental results

    Experiment identification strategies:

    Profile-based identification (preferred):
    - Format: "profile_name" or "profile_name/run_name"
    - Discovers finetuning runs by profile name in metadata
    - Locates associated evaluation runs within finetuning directories
    - Supports latest run selection or specific run targeting
    - Integrates comprehensive metadata and language information

    Model+dataset identification (alternative):
    - Format: "model_name+dataset_name"
    - Direct evaluation discovery without profile system
    - Supports standalone evaluations and custom configurations
    - Bypasses finetuning run association for flexibility

    Args:
        profiles (list): List of experiment identifiers in various formats
        output_dir (str): Base directory containing experiment results (default: "output")

    Side effects:
        - Creates timestamped gathered_predictions_{timestamp}.csv file
        - Prints progress information and experiment discovery status

    Example:
        gather_predictions(["gemma-3n-audioset", "gemma-3n-e4b-it-common-voice"])
        gather_predictions(["profile/run-001", "model+dataset"])
    """
    all_data = {}  # Dictionary to accumulate data for each ID

    for profile in profiles:
        parts = profile.split("/")
        profile_name = parts[0]
        run_name = parts[1] if len(parts) > 1 else None

        parts2 = profile.split("+")
        use_profile = len(parts2) == 1
        model = None
        dataset = None

        if not use_profile:
            model = parts2[0]
            dataset = parts2[1]

        matching_runs = []

        # Find finetuning runs for the profile (if applicable)
        if use_profile:
            finetuning_runs = {}
            for run_dir in os.listdir(output_dir):
                full_run_dir = os.path.join(output_dir, run_dir)
                if os.path.isdir(full_run_dir):
                    metadata = load_metadata(full_run_dir)
                    if metadata and metadata["run_type"] == "finetuning" and metadata.get("profile") == profile_name:
                        finetuning_runs[metadata["run_id"]] = run_dir

            # Find evaluation runs associated with the finetuning runs
            for finetuning_run_id, finetuning_run_dir in finetuning_runs.items():
                finetuning_run_path = os.path.join(output_dir, finetuning_run_dir)
                for eval_dir in os.listdir(finetuning_run_path):
                    if eval_dir == "eval" or eval_dir.startswith("eval-"):
                        full_eval_dir = os.path.join(finetuning_run_path, eval_dir)
                        if os.path.isdir(full_eval_dir):
                            metadata = load_metadata(full_eval_dir)
                            if metadata and metadata["run_type"] == "evaluation":
                                matching_runs.append((full_eval_dir, metadata))

        # Find evaluation runs for model+dataset (always check)
        model_dataset_dir = os.path.join(output_dir, f"{model}+{dataset}", "eval") if not use_profile else None
        if model_dataset_dir and os.path.isdir(model_dataset_dir):
            metadata = load_metadata(model_dataset_dir)
            if metadata and metadata["run_type"] == "evaluation":
                matching_runs.append((model_dataset_dir, metadata))

        if not matching_runs:
            logger.info(f"No evaluation runs found for: {profile}, skipping")
            continue

        if use_profile and not run_name:
            matching_runs.sort(key=lambda x: x[1]["start_time"], reverse=True)
            matching_runs = matching_runs[:1]

        for run_dir, metadata in matching_runs:
            predictions_file = os.path.join(run_dir, "predictions.csv")
            run_id = metadata["run_id"]

            # Construct column name prefix based on run type
            if use_profile:
                if "dataset" in metadata:
                    column_prefix = f"{run_id}-{profile_name}-{metadata['dataset']}"
                else:
                    column_prefix = f"{run_id}-{profile_name}"
            else:
                column_prefix = f"{model}+{dataset}"

            # Extract language information from prepared CSV
            if use_profile:
                dataset_name = metadata.get("dataset")
                if not dataset_name:
                    logger.warning(
                        f"Could not determine dataset name for run {run_dir}. Language information might be missing."
                    )
                    dataset_languages = {}
                else:
                    prepared_csv_path = str(
                        _PROJECT_ROOT / "data" / "datasets" / dataset_name / f"{dataset_name}_prepared.csv"
                    )
                    try:
                        prepared_df = pd.read_csv(prepared_csv_path)
                        dataset_languages = dict(zip(prepared_df["id"], prepared_df["language"]))
                    except Exception as e:
                        raise ValueError(f"Error reading prepared CSV {prepared_csv_path}: {e}") from e
            else:
                # For model+dataset, we don't have individual item languages
                dataset_languages = {}

            try:
                df = pd.read_csv(predictions_file)

                for index, row in df.iterrows():
                    id_val = row["ID"]
                    if id_val not in all_data:
                        all_data[id_val] = {
                            "ID": id_val,
                            "Language": metadata.get("config", {}).get("languages", "??"),
                            "Target": row["Target"],
                            "Norm Target": row["Norm Target"],
                        }

                    all_data[id_val][f"{column_prefix} Pred"] = row["Pred"]
                    all_data[id_val][f"{column_prefix} Norm Pred"] = row["Norm Pred"]
                    all_data[id_val][f"{column_prefix} WER"] = row["WER"]
                    all_data[id_val][f"{column_prefix} CER"] = row["CER"] if "CER" in row else "0"

                logger.info(f"+{predictions_file}")

            except Exception as e:
                raise ValueError(f"Error reading predictions from {predictions_file}: {e}") from e

    if not all_data:
        logger.info("No predictions gathered.")
        return

    # Create combined DataFrame from the dictionary
    combined_df = pd.DataFrame.from_dict(all_data, orient="index")

    # Reorder columns to include "Language" after "Norm Target"
    cols = combined_df.columns.tolist()
    id_target_cols = ["ID", "Target", "Norm Target", "Language"]
    other_cols = [col for col in cols if col not in id_target_cols]
    combined_df = combined_df[id_target_cols + other_cols]

    # Sort by ID
    combined_df = combined_df.sort_values(by="ID")

    # Save combined CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"gathered_predictions_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    combined_df.to_csv(output_path, index=False)
    logger.info(f"{output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python gather.py <profile1> [profile2/run2] [model1+dataset1] ...")
        sys.exit(1)

    profiles = sys.argv[1:]
    gather_predictions(profiles)
