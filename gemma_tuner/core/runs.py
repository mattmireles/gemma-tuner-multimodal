"""
Run Management System for Gemma Fine-Tuning Pipeline

This module provides comprehensive run directory management, metadata tracking,
and experiment organization for the Gemma fine-tuning system. It implements
a structured approach to organizing training runs, evaluations, and exports
with full metadata tracking and atomic operations.

Key responsibilities:
- Sequential run ID generation with thread-safe locking
- Hierarchical run directory creation and organization
- Metadata tracking for all pipeline operations
- Run status management and completion marking
- Metrics persistence and aggregation
- Run discovery and latest run resolution

Called by:
- main.py for all run directory operations (lines 217-263)
- scripts/finetune.py for training run management
- scripts/evaluate.py for evaluation run tracking
- scripts/blacklist.py for blacklist generation tracking

Run directory structure:
output/
├── next_run_id.txt           # Sequential ID counter
├── next_run_id.txt.lock      # File lock for atomic ID generation
├── 1-gemma-3n-e4b-it-custom/    # Training run directory
│   ├── metadata.json         # Run metadata and configuration
│   ├── metrics.json          # Training metrics
│   ├── completed             # Completion marker file
│   ├── checkpoint-500/       # Training checkpoints
│   └── eval/                 # Evaluation subdirectory
│       ├── metadata.json     # Evaluation metadata
│       ├── metrics.json      # WER/CER metrics
│       └── predictions.csv   # Detailed predictions
└── 2-distil-gemma-3n/   # Another training run

Metadata schema:
{
    "run_id": 1,
    "run_type": "finetuning",
    "status": "running|completed|failed",
    "start_time": "2024-01-15 10:30:00",
    "end_time": "2024-01-15 14:30:00",
    "profile": "gemma-3n-e4b-it-custom",
    "model": "gemma-3n-e4b-it",
    "dataset": "custom",
    "config": {...},  # Full merged configuration
    "metrics": {...},  # Final metrics
    "finetuning_run_id": null,  # For evaluation runs
    "run_dir": "/path/to/run"
}

Thread safety:
Uses FileLock for atomic run ID generation to prevent race conditions
when multiple training processes start simultaneously.
"""

import csv
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

from filelock import FileLock

from gemma_tuner.constants import Timing
from gemma_tuner.utils.device import to_bool

logger = logging.getLogger(__name__)


# Run management constants
class RunConstants:
    """Named constants for run management configuration."""

    # File names
    RUN_ID_FILE = "next_run_id.txt"
    RUN_ID_LOCK_FILE = "next_run_id.txt.lock"
    METADATA_FILE = "metadata.json"
    METRICS_FILE = "metrics.json"
    COMPLETION_MARKER = "completed"

    # Run types
    RUN_TYPE_FINETUNING = "finetuning"
    RUN_TYPE_EVALUATION = "evaluation"
    RUN_TYPE_BLACKLIST = "blacklist"

    # Run status values
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    # Directory names
    EVAL_SUBDIR = "eval"

    # Default values
    INITIAL_RUN_ID = 1


def _open_private_file(path: str):
    """Open path for writing with 0o600 permissions, closing the fd on failure.

    Returns an open file object. Guarantees the raw file descriptor is not
    leaked if os.fdopen raises (e.g. when the process hits its FD limit).
    """
    _fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        return os.fdopen(_fd, "w")
    except Exception:
        os.close(_fd)
        raise


def get_next_run_id(output_dir: str) -> int:
    """
    Generates the next sequential run ID with thread-safe file locking.

    This function implements atomic run ID generation using file-based locking
    to ensure unique IDs even when multiple processes start simultaneously.
    The ID counter persists across sessions for consistent run numbering.

    Called by:
    - main.py:main() when creating new fine-tuning runs (line 217)
    - Automated training pipelines requiring unique run identification

    Thread safety mechanism:
    1. Acquire exclusive lock on next_run_id.txt.lock
    2. Read current ID from next_run_id.txt (or start at 1)
    3. Write incremented ID back to file
    4. Release lock and return current ID

    Args:
        output_dir (str): Base output directory for all runs

    Returns:
        int: Next available sequential run ID

    Side effects:
        - Creates output_dir if it doesn't exist
        - Creates/updates next_run_id.txt with incremented value
        - Creates next_run_id.txt.lock for synchronization

    Example:
        >>> run_id = get_next_run_id("output")
        >>> print(run_id)  # e.g., 42
        >>> # Next call returns 43
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use file lock for atomic ID generation across processes
    lock_path = os.path.join(output_dir, RunConstants.RUN_ID_LOCK_FILE)
    lock = FileLock(lock_path, timeout=Timing.FILE_LOCK_TIMEOUT)

    with lock:
        id_file_path = os.path.join(output_dir, RunConstants.RUN_ID_FILE)
        try:
            with open(id_file_path, "r") as f:
                next_id = int(f.read())
        except (FileNotFoundError, ValueError, OSError):
            # First run or corrupt file — start at initial ID
            next_id = RunConstants.INITIAL_RUN_ID

        # Write incremented ID for next use
        with open(id_file_path, "w") as f:
            f.write(str(next_id + 1))

    return next_id


def find_latest_finetuning_run(output_dir: str, profile_name: str) -> Optional[str]:
    """
    Finds the most recent fine-tuning run directory for a profile.

    This function locates the latest training run for a given profile,
    regardless of completion status. Used for linking evaluation runs
    to their corresponding training runs and for run resumption.

    Called by:
    - main.py:find_latest_finetuning_run() alias function (line 73)
    - create_run_directory() for evaluation run linking (lines 89, 96, 121)
    - Evaluation workflows needing latest model checkpoint

    Run directory matching:
    - Pattern: "{run_id}-{profile_name}" (e.g., "5-gemma-3n-e4b-it-custom")
    - Uses substring matching on profile name after hyphen
    - Sorts by filesystem modification time (most recent first)

    Args:
        output_dir (str): Base output directory containing runs
        profile_name (str): Profile name to search for

    Returns:
        Optional[str]: Full path to latest run directory, None if not found

    Example:
        >>> path = find_latest_finetuning_run("output", "gemma-3n-e4b-it-custom")
        >>> print(path)  # "output/5-gemma-3n-e4b-it-custom"

    Note:
        Returns runs regardless of completion status. Use
        find_latest_completed_finetuning_run() if you need only completed runs.
    """
    if not os.path.isdir(output_dir):
        return None

    # Find all directories matching the profile pattern
    # Use exact suffix match to avoid substring collisions
    # Directory format is "{number}-{profile_name}", so split on first "-"
    runs = [
        d
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.split("-", 1)[-1] == profile_name
    ]

    if not runs:
        return None

    # Sort by modification time (most recent first)
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)), reverse=True)
    return os.path.join(output_dir, runs[0])


def find_latest_completed_finetuning_run(output_dir: str, profile_name: str) -> Optional[str]:
    """
    Finds the most recent successfully completed fine-tuning run for a profile.

    This function specifically locates completed training runs, checking both
    metadata status and completion marker files. Essential for evaluation
    operations that require fully trained models.

    Called by:
    - create_run_directory() as fallback for evaluation runs (line 121)
    - Evaluation scripts requiring completed models
    - Model export workflows needing final checkpoints

    Completion detection hierarchy:
    1. Check metadata.json for status="completed" (preferred)
    2. Fall back to presence of "completed" marker file
    3. Both methods ensure backward compatibility

    Completion verification workflow:
    1. Scan all directories matching profile pattern
    2. Check metadata.json status field if present
    3. Check for "completed" marker file as fallback
    4. Track modification times for latest selection
    5. Return most recently modified completed run

    Args:
        output_dir (str): Base output directory containing runs
        profile_name (str): Profile name to search for

    Returns:
        Optional[str]: Path to latest completed run, None if no completed runs

    Example:
        >>> path = find_latest_completed_finetuning_run("output", "gemma-3n-e4b-it")
        >>> if path:
        >>>     print(f"Found completed run: {path}")
        >>> else:
        >>>     print("No completed runs found")

    Note:
        Prefers metadata.json status for accuracy but maintains backward
        compatibility with older runs using only marker files.
    """
    latest_dir = None
    latest_mtime = -1.0

    if not os.path.isdir(output_dir):
        return None

    for d in os.listdir(output_dir):
        run_path = os.path.join(output_dir, d)

        # Skip non-directories and non-matching profiles
        if not os.path.isdir(run_path) or d.split("-", 1)[-1] != profile_name:
            continue

        # Check completion status via metadata or marker
        meta_path = os.path.join(run_path, RunConstants.METADATA_FILE)
        status_completed = False

        # Priority 1: Check metadata.json status field
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                status_completed = meta.get("status") == RunConstants.STATUS_COMPLETED
            except Exception:
                # Corrupted metadata, check marker file instead
                status_completed = False

        # Priority 2: Fall back to completion marker file
        if not status_completed:
            marker_path = os.path.join(run_path, RunConstants.COMPLETION_MARKER)
            if os.path.exists(marker_path):
                status_completed = True

        # Track most recent completed run
        if status_completed:
            mtime = os.path.getmtime(run_path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_dir = run_path

    return latest_dir


def mark_run_as_completed(run_dir: str) -> None:
    """
    Marks a run as successfully completed with a marker file.

    This function creates a completion marker file and updates metadata
    to indicate successful run completion. Used by training and evaluation
    scripts to signal successful termination.

    Called by:
    - main.py:main() after successful fine-tuning (line 254)
    - scripts/evaluate.py after successful evaluation
    - Training scripts on successful completion

    Completion marking workflow:
    1. Create "completed" marker file in run directory
    2. Update metadata.json status to "completed"
    3. Record completion timestamp

    Args:
        run_dir (str): Path to run directory to mark as completed

    Side effects:
        - Creates "completed" file in run_dir
        - Updates metadata.json with completed status
        - Sets end_time in metadata

    Note:
        Both marker file and metadata update ensure backward compatibility
        and redundancy in completion detection.
    """
    # Create marker file for backward compatibility.
    # Use os.open() with 0o600 so the marker is owner-readable only — it may
    # contain model identifiers that should not be world-readable on shared systems.
    marker_path = os.path.join(run_dir, RunConstants.COMPLETION_MARKER)
    with _open_private_file(marker_path) as f:
        f.write("completed")

    # Also update metadata if present
    update_run_metadata(
        run_dir, status=RunConstants.STATUS_COMPLETED, end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def create_run_directory(
    output_dir: str,
    profile_name: Optional[str],
    run_id,
    run_type: str,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> str:
    """
    Creates a run directory with appropriate structure and initializes metadata.

    This function implements the directory creation logic for all run types,
    establishing the hierarchical structure that organizes training, evaluation,
    and blacklist generation runs. It handles both profile-based and direct
    model+dataset configurations.

    Called by:
    - main.py:main() for all run types (lines 220, 231, 238)
    - Automated training pipelines creating run structures

    Calls to:
    - find_latest_finetuning_run() for evaluation directory placement
    - os.makedirs() for directory creation
    - json.dump() for metadata initialization

    Directory structure by run type:

    Fine-tuning runs:
    - Pattern: "{run_id}-{profile_name}/"
    - Example: "5-gemma-3n-e4b-it-custom/"
    - Contains checkpoints, logs, and training artifacts

    Evaluation runs (three modes):
    1. Profile evaluation (evaluates latest fine-tuning):
       - Location: "{finetuning_run}/eval/"
       - Links to parent training run via metadata

    2. Direct model+dataset evaluation:
       - Pattern: "{model}+{dataset}/eval/"
       - For evaluating pre-trained or external models

    3. Profile with alternative dataset:
       - Location: "{finetuning_run}/eval-{dataset}/"
       - For cross-dataset evaluation

    Blacklist runs:
    - Same structure as evaluation runs
    - Contains analysis results and generated blacklists

    Args:
        output_dir (str): Base output directory for runs
        profile_name (Optional[str]): Profile name for profile-based runs
        run_id: Sequential run ID or composite ID for direct runs
        run_type (str): Type of run (finetuning, evaluation, blacklist)
        model_name (Optional[str]): Model name for direct evaluation
        dataset_name (Optional[str]): Dataset name for direct evaluation

    Returns:
        str: Path to created run directory

    Raises:
        ValueError: If run type invalid or required training run not found

    Side effects:
        - Creates run directory structure
        - Initializes metadata.json with run information
    """
    if run_type == RunConstants.RUN_TYPE_FINETUNING:
        # Fine-tuning runs use sequential ID and profile name
        run_dir = os.path.join(output_dir, f"{run_id}-{profile_name}")

    elif run_type in (RunConstants.RUN_TYPE_EVALUATION, RunConstants.RUN_TYPE_BLACKLIST):
        # Evaluation/blacklist runs nest under training runs or standalone

        if profile_name and not dataset_name:
            # Mode 1: Evaluate latest fine-tuning for profile
            finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            if not finetuning_run_dir:
                raise ValueError(f"No fine-tuning runs found for profile '{profile_name}'.")
            run_dir = os.path.join(finetuning_run_dir, RunConstants.EVAL_SUBDIR)

        elif model_name and dataset_name:
            # Mode 2: Direct model+dataset evaluation (no training run)
            run_id = f"{model_name}+{dataset_name}"
            run_dir = os.path.join(output_dir, run_id, RunConstants.EVAL_SUBDIR)

        elif profile_name and dataset_name:
            # Mode 3: Evaluate training run on different dataset
            finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            if finetuning_run_dir is None:
                raise ValueError(
                    f"No completed finetuning run found for profile '{profile_name}' in '{output_dir}'. "
                    "Run finetune first before evaluating on a cross-dataset."
                )
            run_dir = os.path.join(finetuning_run_dir, f"eval-{dataset_name}")

        else:
            raise ValueError("Invalid evaluation parameters")
    else:
        raise ValueError(f"Invalid run type: {run_type}")

    os.makedirs(run_dir, exist_ok=True)

    # Initialize comprehensive metadata for run tracking
    metadata = {
        "run_id": run_id,
        "run_type": run_type,
        "status": RunConstants.STATUS_RUNNING,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "profile": profile_name,
        "model": model_name,
        "dataset": dataset_name,
        "config": {},  # Will be populated with full configuration
        "metrics": {},  # Will be populated with final metrics
        "finetuning_run_id": None,  # Link to parent training run
        "run_dir": run_dir,
    }

    # Link evaluation runs to their parent training runs
    if run_type == RunConstants.RUN_TYPE_EVALUATION and profile_name:
        # Try completed runs first, fall back to any run
        finetuning_run_dir = find_latest_completed_finetuning_run(
            output_dir, profile_name
        ) or find_latest_finetuning_run(output_dir, profile_name)
        if finetuning_run_dir:
            # Extract run ID from directory name (e.g., "5-profile" -> "5")
            base = os.path.basename(finetuning_run_dir)
            finetuning_run_id = base.split("-")[0] if "-" in base else base
            metadata["finetuning_run_id"] = finetuning_run_id

    # Write initial metadata with owner-only permissions (run dirs may sit in a
    # shared output directory; metadata includes model paths and hyperparameters).
    metadata_path = os.path.join(run_dir, RunConstants.METADATA_FILE)
    with _open_private_file(metadata_path) as f:
        json.dump(metadata, f, indent=4)

    return run_dir


def update_run_metadata(run_dir: str, **kwargs) -> None:
    """
    Updates run metadata fields in an existing metadata.json file.

    This function provides atomic metadata updates for tracking run progress,
    status changes, and metric updates. It preserves existing metadata while
    merging new values, ensuring comprehensive run tracking.

    Called by:
    - mark_run_as_completed() to update status and end time
    - Training scripts to record configuration and hyperparameters
    - Evaluation scripts to store computed metrics
    - Error handlers to mark failed runs

    Update patterns:
    - Status updates: update_run_metadata(dir, status="completed")
    - Metric recording: update_run_metadata(dir, metrics={"wer": 0.15})
    - Configuration: update_run_metadata(dir, config=training_args)
    - Timestamps: update_run_metadata(dir, end_time="2024-01-15 14:30:00")

    Args:
        run_dir (str): Path to run directory containing metadata.json
        **kwargs: Key-value pairs to update in metadata

    Side effects:
        - Reads existing metadata.json (creates if missing)
        - Merges new values with existing metadata
        - Writes updated metadata back to file

    Example:
        >>> update_run_metadata(
        ...     "output/5-gemma-3n-e4b-it",
        ...     status="completed",
        ...     metrics={"final_wer": 0.12},
        ...     end_time="2024-01-15 14:30:00"
        ... )

    Atomic update mechanism:
    - Uses temporary file write + atomic rename to prevent corruption
    - Preserves existing metadata fields not being updated
    - Atomic against partial writes (temp file + os.replace) but NOT safe for concurrent writers

    Note:
        Updates are merged at the top level only. Nested dictionaries
        are replaced entirely, not merged recursively. For complex nested
        updates, read existing metadata, modify in memory, then write complete object.
    """
    meta_path = os.path.join(run_dir, RunConstants.METADATA_FILE)

    # Load existing metadata or create minimal structure
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        # Metadata missing, create minimal structure
        meta = {"run_dir": run_dir}
    except (json.JSONDecodeError, ValueError):
        # Metadata file exists but is corrupt (e.g. partial write from power loss).
        # Log a warning and start fresh rather than propagating a crash to callers
        # like mark_run_as_completed() that must not be interrupted.
        logger.warning("Corrupt metadata file at %s — resetting to defaults. Original content lost.", meta_path)
        meta = {"run_dir": run_dir}

    # Merge updates with existing metadata.
    # Convert any ProfileConfig values to plain dicts for JSON serialization.
    # Build a new dict rather than mutating kwargs in-place while iterating —
    # mutating a dict's values during iteration is CPython-implementation-defined.
    serializable = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in kwargs.items()}
    meta.update(serializable)

    # Write updated metadata atomically via tmp+replace to prevent partial writes.
    # Create tmp file with 0o600 so the replaced target inherits restrictive perms.
    tmp_path = meta_path + ".tmp"
    with _open_private_file(tmp_path) as f:
        json.dump(meta, f, indent=4)
    os.replace(tmp_path, meta_path)


def write_metrics(run_dir: str, metrics: Dict[str, Any]) -> None:
    """
    Writes or updates metrics to a dedicated metrics.json file.

    This function maintains a separate metrics file for detailed training
    and evaluation metrics, supporting incremental updates throughout the
    run lifecycle. Metrics are merged with existing values for comprehensive
    tracking.

    Called by:
    - Training loops to record loss, learning rate, and progress
    - Evaluation scripts to save WER/CER scores
    - Checkpoint callbacks to persist intermediate metrics
    - Final metric aggregation at run completion

    Metrics organization:
    {
        "train_loss": [0.5, 0.4, 0.3, ...],  # Per-step losses
        "eval_wer": 0.15,                    # Final evaluation metric
        "best_checkpoint": "checkpoint-500",  # Best model checkpoint
        "training_time_hours": 4.5,          # Total training time
        "final_metrics": {                   # Aggregated final results
            "wer": 0.15,
            "cer": 0.08
        }
    }

    Args:
        run_dir (str): Run directory to write metrics to
        metrics (Dict[str, Any]): Metrics to write or update

    Side effects:
        - Creates run_dir if it doesn't exist
        - Creates or updates metrics.json in run_dir
        - Merges new metrics with existing values

    Example:
        >>> write_metrics(
        ...     "output/5-gemma-3n-e4b-it",
        ...     {"epoch_1_loss": 0.45, "learning_rate": 5e-5}
        ... )

    Note:
        Like update_run_metadata(), merges at top level only.
        Arrays and nested dicts are replaced, not merged.
    """
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, RunConstants.METRICS_FILE)

    # Load existing metrics for merging
    try:
        existing: Dict[str, Any]
        with open(metrics_path, "r") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}
    except (json.JSONDecodeError, ValueError):
        # Corrupt metrics.json (e.g. partial write from a previous crash).
        # Reset to empty rather than propagating to finalize_training_run.
        logger.warning("Corrupt metrics file at %s — resetting to defaults.", metrics_path)
        existing = {}

    # Merge new metrics with existing
    existing.update(metrics)

    # Write updated metrics atomically (tmp + os.replace) to prevent truncated JSON on crash.
    # Create tmp file with 0o600 — metrics include WER/CER and may expose dataset info.
    tmp_path = str(metrics_path) + ".tmp"
    with _open_private_file(tmp_path) as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp_path, metrics_path)


def summarize_run_for_csv(run_dir: str) -> Dict[str, Any]:
    """Create a flat summary row for a run suitable for CSV indexing.

    Fields include identifiers, status, timing, key hyperparameters, and core metrics
    that make result comparison easy without loading individual directories.

    Returns a dictionary keyed by stable column names.
    """
    meta_path = os.path.join(run_dir, RunConstants.METADATA_FILE)
    metrics_path = os.path.join(run_dir, RunConstants.METRICS_FILE)
    meta = {}
    metrics = {}
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        logger.warning("Could not read metadata from %s — using defaults for CSV summary", run_dir, exc_info=True)
        meta = {"run_dir": run_dir}
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        logger.warning("Could not read metrics from %s — using empty defaults for CSV summary", run_dir, exc_info=True)
        metrics = {}

    cfg = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    row = {
        "run_dir": run_dir,
        "run_id": meta.get("run_id"),
        "run_type": meta.get("run_type"),
        "status": meta.get("status"),
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
        "profile": meta.get("profile"),
        "model": meta.get("model") or cfg.get("model"),
        "dataset": meta.get("dataset") or cfg.get("dataset"),
        # Core metrics (optional)
        "wer": (metrics.get("wer") if isinstance(metrics, dict) else None)
        or (metrics.get("final_metrics", {}).get("wer") if isinstance(metrics, dict) else None),
        "cer": (metrics.get("cer") if isinstance(metrics, dict) else None)
        or (metrics.get("final_metrics", {}).get("cer") if isinstance(metrics, dict) else None),
        # Key hyperparameters
        "learning_rate": cfg.get("learning_rate"),
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
        "gradient_checkpointing": to_bool(cfg.get("gradient_checkpointing")),
        "attn_implementation": cfg.get("attn_implementation"),
        "dtype": cfg.get("dtype"),
    }
    return row


def update_experiments_csv(output_dir: str, run_dir: str) -> str:
    """Append or update a row for this run in output/experiments.csv (idempotent).

    - Creates the CSV with header if missing
    - Uses run_dir as a unique key to update existing rows
    - Returns the path to the CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiments.csv")
    row = summarize_run_for_csv(run_dir)

    # Load existing rows (if any)
    rows: list[Dict[str, Any]] = []
    fieldnames = list(row.keys())
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Preserve existing header columns union
                for name in reader.fieldnames or []:
                    if name not in fieldnames:
                        fieldnames.append(name)
        except Exception:
            # Corrupted CSV; fall back to recreating
            rows = []

    # Upsert by run_dir
    updated = False
    for i, r in enumerate(rows):
        if r.get("run_dir") == run_dir:
            rows[i] = {**r, **row}
            updated = True
            break
    if not updated:
        rows.append(row)

    # Atomic write
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp_path, csv_path)
    return csv_path


def _ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    """Create experiments table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            run_dir TEXT PRIMARY KEY,
            run_id TEXT,
            run_type TEXT,
            status TEXT,
            start_time TEXT,
            end_time TEXT,
            profile TEXT,
            model TEXT,
            dataset TEXT,
            wer REAL,
            cer REAL,
            learning_rate REAL,
            per_device_train_batch_size INTEGER,
            gradient_accumulation_steps INTEGER,
            gradient_checkpointing INTEGER,
            attn_implementation TEXT,
            dtype TEXT
        )
        """
    )
    conn.commit()


def update_experiments_sqlite(output_dir: str, run_dir: str) -> str:
    """Upsert a row into output/experiments.db for the given run.

    Returns the path to the SQLite database file.
    """
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "experiments.db")
    row = summarize_run_for_csv(run_dir)

    conn = sqlite3.connect(db_path)
    try:
        _ensure_sqlite_schema(conn)
        # Normalize booleans to 0/1 for SQLite
        row_sql = dict(row)
        row_sql["gradient_checkpointing"] = 1 if to_bool(row_sql.get("gradient_checkpointing")) else 0

        placeholders = ", ".join([f"{k} = :{k}" for k in row_sql.keys()])
        # Try update-first, then insert if no row changed
        cur = conn.execute(
            f"UPDATE experiments SET {placeholders} WHERE run_dir = :run_dir",
            row_sql,
        )
        if cur.rowcount == 0:
            columns = ", ".join(row_sql.keys())
            bind = ", ".join([f":{k}" for k in row_sql.keys()])
            conn.execute(
                f"INSERT INTO experiments ({columns}) VALUES ({bind})",
                row_sql,
            )
        conn.commit()
    finally:
        conn.close()

    return db_path
