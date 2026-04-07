#!/usr/bin/env python3
"""
Dataset Loading and Patch Management System

This module provides sophisticated dataset loading capabilities with comprehensive
data quality management through override and blacklist systems. It enables
fine-grained control over training data quality by applying manual corrections,
filtering problematic samples, and protecting high-quality ground truth data.

Key responsibilities:
- Dataset split loading with configurable sample limits
- Hierarchical patch application (overrides, blacklists, protections)
- Data quality management and validation
- Training sample filtering and modification
- Statistical reporting of data modifications

Called by:
- models.gemma.finetune.py:main() for standard Gemma training data loading and preprocessing
- models.gemma.finetune.py:main() for knowledge distillation training data preparation
- models.gemma.finetune.py:main() for LoRA training data loading with patch system
- scripts/evaluate.py for evaluation data loading with consistent patch application
- scripts/blacklist.py for blacklist generation and problematic sample analysis
- scripts/prepare_data.py for dataset validation and quality assessment
- core/ops.py functions indirectly through model-specific training implementations
- Data preparation workflows requiring quality-controlled dataset loading
- Validation and testing pipelines ensuring data consistency across train/eval splits

Patch system architecture:
data_patches/{dataset_source}/
├── override_text_perfect/      # Manual transcription corrections
│   ├── correction1.csv
│   └── correction2.csv
├── override_text_verbatim/     # Verbatim transcription overrides
│   └── verbatim_fixes.csv
├── do_not_blacklist/           # Protected samples (never filtered)
│   └── ground_truth.csv
└── delete/                     # Blacklisted samples (filtered out)
    ├── problematic.csv
    └── low_quality.csv

Patch application order (precedence):
1. Override application: Text corrections applied first
2. Protection marking: Do-not-blacklist samples identified
3. Blacklist filtering: Problematic samples removed (respects protections)

Data quality workflow:
1. Load base dataset from CSV files
2. Apply manual text corrections (overrides)
3. Identify protected samples (do_not_blacklist)
4. Filter out blacklisted samples (except protected ones)
5. Report modification statistics

Flexible column handling:
Supports datasets with varying column structures by checking for required
columns ('id', target columns) before applying patches. Missing columns
are gracefully handled with warning messages.

Sample limiting:
Supports max_samples parameter for:
- Development and testing with smaller datasets
- Memory-constrained training environments
- Quick experimentation and prototyping

Statistical reporting:
Provides comprehensive statistics on all data modifications:
- Override counts by column type
- Blacklist filtering statistics
- Protected sample counts
- Final dataset size and composition

Error handling:
- Corrupted patch files: Detailed error messages with file paths
- Missing columns: Warnings with graceful degradation
- IO errors: Exception propagation with context

Use cases:
- Production training: Apply all patches for maximum data quality
- Evaluation: Use same patches for consistency with training
- Development: Use max_samples for faster iteration
- Data analysis: Understand impact of data quality measures
"""

import configparser
import logging
import math
import os
import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Iterable

import pandas as pd

from gemma_tuner.utils.dataset_sources import DatasetLoadContext, resolve_dataset_source_adapter

logger = logging.getLogger(__name__)


def _ensure_modality_local_csv_only(
    context: DatasetLoadContext, adapter, streaming_enabled: bool, modality: str
) -> None:
    """Text/image fine-tuning (v1) is limited to non-streaming local CSV loads."""
    if context.modality != modality:
        return
    if streaming_enabled:
        raise ValueError(f"modality={modality} requires non-streaming dataset loading in v1.")
    if adapter.name != "local-csv":
        raise ValueError(
            f"modality={modality} is only supported for local CSV datasets in v1; "
            f"got adapter {adapter.name!r} (source_type={context.source_type!r}). "
            "Use a local [dataset:…] section with a CSV source, not Granary/BigQuery/GCS streaming."
        )


# Anchored config path — resolves relative to the project root regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_INI = _REPO_ROOT / "config" / "config.ini"


def _has_cwd_config() -> bool:
    """True if a config file is present in cwd (legacy root or under config/)."""
    return Path("config.ini").exists() or Path("config/config.ini").exists()


def resolve_data_datasets_dir(dataset_name: str) -> str:
    """Absolute path to ``data/datasets/<dataset_name>``.

    If ``config.ini`` exists in the current working directory, the path is taken
    relative to cwd (matches :func:`_get_config` and supports tests that
    ``chdir`` into a temp tree).

    If there is no ``config.ini`` in cwd (e.g. ``gemma-macos-tuner finetune`` was
    started from outside the repo while passing ``--config /path/to/config.ini``),
    the path is anchored to the repository root (parent of the ``gemma_tuner``
    package), so CSVs and image files resolve consistently regardless of cwd.
    """
    name = str(dataset_name).strip()
    if _has_cwd_config():
        return str((Path("data") / "datasets" / name).resolve())
    return str((_REPO_ROOT / "data" / "datasets" / name).resolve())


def resolve_patches_base_dir(patches_dir: str) -> str:
    """Absolute base for patch trees (``<base>/<dataset_source>/override_*/``).

    Uses the same rule as :func:`resolve_data_datasets_dir`: cwd-relative when
    ``config.ini`` is in cwd, otherwise anchored to the repository root so
    overrides/blacklists resolve when the CLI runs from outside the repo.
    """
    raw = (patches_dir or "data_patches").strip() or "data_patches"
    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())
    if _has_cwd_config():
        return str((Path.cwd() / p).resolve())
    return str((_REPO_ROOT / p).resolve())


# Lazy-loaded config singleton. Deferred so tests can os.chdir() or monkeypatch
# _CONFIG_INI before the first call to load_dataset_split(). Thread-safe for
# single-threaded training pipelines (all current callers).
_config: configparser.ConfigParser | None = None


def _get_config() -> configparser.ConfigParser:
    """Return the module-level ConfigParser, reading config.ini on first access.

    Checks CWD for config.ini first (so tests that chdir work), then falls back
    to the anchored _CONFIG_INI path computed at import time.
    """
    global _config
    if _config is None:
        _config = configparser.ConfigParser()
        cwd_ini = Path("config.ini")
        cwd_nested = Path("config/config.ini")
        if cwd_ini.exists():
            _config.read(cwd_ini)
        elif cwd_nested.exists():
            _config.read(cwd_nested)
        else:
            _config.read(_CONFIG_INI)
    return _config


@dataclass
class PatchBundle:
    overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    protected_ids: set[str] = field(default_factory=set)
    blacklist_ids: set[str] = field(default_factory=set)
    patches_path: str = ""


def load_dataset_split(split, dataset_config, max_samples=None, patches_dir="data_patches/", streaming_enabled=False):
    """
    Loads a dataset split with comprehensive data quality management through patches.

    This function is the cornerstone of the data loading system, providing sophisticated
    data quality control through a hierarchical patch system. It enables fine-grained
    control over training data quality while maintaining reproducibility and traceability.

    Called by:
    - models.gemma.finetune.py for training data loading
    - models/distil/finetune.py for training data loading
    - scripts/evaluate.py for evaluation data loading
    - scripts/blacklist.py for blacklist generation analysis

    Calls to:
    - datasets.load_dataset() for base dataset loading
    - pandas.read_csv() for patch file processing
    - configparser for dataset source resolution

    Dataset loading strategy:
    1. Source resolution: Determine dataset source from config
    2. Path construction: Build dataset and cache directory paths
    3. Base loading: Load specified split with optional sample limiting
    4. Patch application: Apply overrides, protections, and blacklists
    5. Statistics reporting: Report all modifications applied

    Patch system hierarchy:

    Override application (highest precedence):
    - override_text_perfect/: Manual transcription corrections
    - override_text_verbatim/: Verbatim transcription fixes
    - Applied first, modify transcript text directly

    Protection system:
    - do_not_blacklist/: Samples protected from filtering
    - Identified before blacklist application
    - Overrides blacklist decisions for high-quality samples

    Blacklist filtering (lowest precedence):
    - delete/: Problematic samples to remove from training
    - Applied last, respects protection system
    - Reduces dataset size but improves quality

    Sample limiting:
    When max_samples is specified, uses HuggingFace datasets slicing:
    - Format: split="{split}[:{max_samples}]"
    - Applied before patch processing for efficiency
    - Useful for development, testing, memory constraints

    Flexible column handling:
    Patch files may have different column structures. Function checks for:
    - Required 'id' column for sample identification
    - Target columns ('text_perfect', 'text_verbatim') for overrides
    - Graceful handling of missing columns with warnings

    Error handling:
    - Missing dataset files: Propagated to caller
    - Corrupted patch files: Detailed error with file path
    - Missing required columns: Warning with skip behavior
    - IO errors: Exception propagation with context

    Performance considerations:
    - Override dictionaries: O(1) lookup for efficient batch processing
    - Set operations: O(1) blacklist/protection checks
    - In-memory processing: Suitable for datasets up to several GB

    Args:
        split (str): Dataset split name ("train", "validation", "test")
        dataset_config (dict): Dataset configuration containing:
            - name: Dataset identifier for path construction
            - Additional dataset-specific parameters
        max_samples (int | None): Maximum samples to load (None = all samples)
        patches_dir (str): Base directory for patch files (default: "data_patches/")
        streaming_enabled (bool): Enable streaming mode for large datasets (default: False)

    Returns:
        tuple[datasets.Dataset, str]:
            - Processed dataset with all patches applied
            - Dataset source identifier for further processing

    Raises:
        ValueError: If dataset source is invalid or patch files are corrupted
        FileNotFoundError: If dataset files don't exist

    Example:
        dataset_config = {"name": "librispeech"}
        dataset, source = load_dataset_split(
            split="train",
            dataset_config=dataset_config,
            max_samples=1000,
            patches_dir="data_patches/",
        )
        print(f"Loaded {len(dataset)} samples from {source}")

    Statistical output:
        ==== Override and Blacklist Summary ====
          Overridden 'text_perfect' in 45 samples.
          Overridden 'text_verbatim' in 12 samples.
          Blacklisted 23 samples (excluding do-not-blacklist IDs).
          Do-not-blacklist IDs: 8 (these were not blacklisted)
        Final dataset size: 1034 samples
    """
    patches_dir = resolve_patches_base_dir(patches_dir)

    context, adapter = _resolve_load_context(
        split=split,
        dataset_config=dataset_config,
        max_samples=max_samples,
        patches_dir=patches_dir,
        streaming_enabled=streaming_enabled,
        config=_get_config(),
    )
    _ensure_modality_local_csv_only(context, adapter, streaming_enabled, "text")
    _ensure_modality_local_csv_only(context, adapter, streaming_enabled, "image")
    source = adapter.patch_source(context)

    # Debug information for troubleshooting data loading issues
    logger.info("==== Dataset Loading Configuration ====")
    logger.info(f"Dataset name:   {context.dataset_name}")
    logger.info(f"Source type:    {context.source_type or 'default'}")
    logger.info(f"Adapter:        {adapter.name}")
    logger.info(f"Split path:     {context.split_path}")
    logger.info(f"Cache dir:      {context.cache_dir}")
    logger.info(f"Patches dir:    {patches_dir}")
    logger.info(f"Max samples:    {max_samples}")
    logger.info(f"Source:         {source}")
    logger.info(f"Streaming:      {streaming_enabled}")

    dataset = adapter.load_base_dataset(context)

    if not streaming_enabled:
        original_count = len(dataset)
        logger.info(f"Base dataset loaded: {original_count} samples")
    else:
        original_count = None  # Length not available in streaming mode
        logger.info("Base dataset loaded in streaming mode")

    # === BASIC SCHEMA VALIDATION ===
    # Ensure core columns are present before applying patches
    # Required: 'id' plus at least one text column configured for this dataset
    # Optional but recommended: 'audio_path' and 'language'
    required_columns = {"id"}
    configured_text_col = dataset_config.get("text_column")
    if configured_text_col:
        required_columns.add(configured_text_col)
    if context.modality == "text" and context.text_sub_mode == "instruction":
        if not context.prompt_column:
            raise ValueError(
                "modality=text with text_sub_mode=instruction requires prompt_column in dataset_config (profile)."
            )
        required_columns.add(context.prompt_column)
    if context.modality == "image":
        ipc = context.image_path_column.strip() if context.image_path_column else "image_path"
        required_columns.add(ipc)
        if context.image_sub_mode == "vqa":
            if not context.prompt_column:
                raise ValueError(
                    "modality=image with image_sub_mode=vqa requires prompt_column in dataset_config (profile)."
                )
            required_columns.add(context.prompt_column)

    def _validate_columns(columns: Iterable[str]):
        missing = [c for c in required_columns if c not in columns]
        if missing:
            raise ValueError("Dataset schema validation failed: missing required column(s): " + ", ".join(missing))

    if not streaming_enabled:
        _validate_columns(dataset.features.keys())
    else:
        # For streaming, best effort validation by peeking a single sample
        try:
            first_example = next(iter(dataset))
            _validate_columns(first_example.keys())
        except StopIteration:
            raise ValueError("Empty dataset in streaming mode; no samples to validate.")

    # === Patch Application System ===
    # Patches are organized by dataset source for flexibility across datasets
    patches_path = os.path.join(patches_dir, source)
    logger.info(f"Applying patches from: {patches_path}")
    patch_bundle = _load_patch_bundle(patches_path)

    # For streaming mode, we need to load all patches into memory first
    # These are small files (few MB total) so this is efficient
    if streaming_enabled:
        return _apply_patches_streaming(dataset, patch_bundle, source, max_samples, configured_text_col)

    # Non-streaming mode: apply patches directly to loaded dataset.
    # Use configurable parallelism for patch application; falls back to None (single-process).
    #
    # IMPORTANT: HuggingFace datasets treats num_proc=0 as invalid and raises ValueError.
    # config.ini defaults preprocessing_num_workers=0 for Apple Silicon MPS to avoid
    # multiprocessing overhead. We convert 0 (and any value < 1) to None, which is the
    # correct HuggingFace sentinel for single-process (in-process) execution.
    try:
        num_workers_for_patches = int(_config["dataset_defaults"].get("preprocessing_num_workers", 1))
        if num_workers_for_patches < 1:
            # None = single-process mode in HuggingFace datasets (correct for MPS / default config)
            num_workers_for_patches = None
    except Exception:
        num_workers_for_patches = None
    dataset, override_counts, filtered_count = _apply_patch_bundle(
        dataset,
        patch_bundle,
        configured_text_col=configured_text_col,
        num_workers=num_workers_for_patches,
        original_count=original_count,
    )

    # === FINAL STATISTICS AND REPORTING ===
    final_count = len(dataset)

    logger.info("\n" + "=" * 50)
    logger.info("DATASET PROCESSING SUMMARY")
    logger.info("=" * 50)
    if original_count is not None:
        logger.info(f"Original dataset size:     {original_count:,} samples")
    else:
        logger.info("Original dataset size:     (streaming — unknown)")

    # Override statistics
    total_overrides = sum(override_counts.values())
    if total_overrides > 0:
        logger.info("\nText Override Application:")
        for column, count in override_counts.items():
            if count > 0:
                logger.info(f"  {column:20}: {count:,} samples modified")
        logger.info(f"  Total overrides:       {total_overrides:,} modifications")

    # Protection and blacklist statistics
    if patch_bundle.protected_ids or patch_bundle.blacklist_ids:
        logger.info("\nData Quality Filtering:")
        logger.info(f"  Protected samples:     {len(patch_bundle.protected_ids):,} (never filtered)")
        logger.info(f"  Blacklisted samples:   {len(patch_bundle.blacklist_ids):,} (candidates for removal)")
        logger.info(f"  Actually filtered:     {filtered_count:,} (after protection override)")

        # Protection effectiveness
        if patch_bundle.protected_ids and patch_bundle.blacklist_ids:
            protected_from_blacklist = len(patch_bundle.protected_ids & patch_bundle.blacklist_ids)
            if protected_from_blacklist > 0:
                logger.info(f"  Protection rescues:    {protected_from_blacklist:,} (blacklisted but protected)")

    if final_count is not None:
        logger.info(f"\nFinal dataset size:        {final_count:,} samples")

        # Data modification impact
        if original_count is not None and original_count > 0:
            retention_rate = (final_count / original_count) * 100
            logger.info(f"Data retention rate:       {retention_rate:.1f}%")
    else:
        logger.info("\nStreaming mode active - final size determined during iteration")

    logger.info("=" * 50)

    return dataset, source


def _resolve_load_context(split, dataset_config, max_samples, patches_dir, streaming_enabled, config):
    dataset_name = dataset_config["name"]
    section_key = f"dataset:{dataset_name}"
    if not config.has_section(section_key):
        raise ValueError(
            f"No [{section_key}] section found in config.ini. "
            f"Verify config.ini exists at {_CONFIG_INI} and contains this section."
        )
    dataset_section = config[section_key]
    source_type = (dataset_section.get("source_type") or "").strip().lower()
    source = (dataset_section.get("source") or "").strip()

    if not source and source_type in {"granary", "bigquery", "bigquery-prepared"}:
        source = dataset_name

    if not source:
        raise ValueError(
            f"Dataset '{dataset_name}' has invalid or missing source: {source}. "
            f"Check config.ini for dataset:{dataset_name} section with 'source' field "
            f"or use source_type=granary for Granary datasets."
        )

    dataset_dir = resolve_data_datasets_dir(dataset_name)
    modality = (dataset_config.get("modality") or "audio").strip().lower()
    text_sub_mode = (dataset_config.get("text_sub_mode") or "instruction").strip().lower()
    _pc = dataset_config.get("prompt_column")
    if _pc is None or (isinstance(_pc, str) and not str(_pc).strip()):
        prompt_column = None
    else:
        prompt_column = str(_pc).strip()

    _ism = (dataset_config.get("image_sub_mode") or "caption").strip().lower()
    _ipc = dataset_config.get("image_path_column")
    if _ipc is None or (isinstance(_ipc, str) and not str(_ipc).strip()):
        image_path_column = "image_path"
    else:
        image_path_column = str(_ipc).strip()

    context = DatasetLoadContext(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        max_samples=max_samples,
        patches_dir=patches_dir,
        streaming_enabled=streaming_enabled,
        dataset_section=dataset_section,
        source=source,
        source_type=source_type,
        dataset_dir=dataset_dir,
        split_path=os.path.join(dataset_dir, f"{split}.csv").replace("\\", "/"),
        prepared_fallback_path=os.path.join(dataset_dir, f"{dataset_name}_prepared.csv").replace("\\", "/"),
        cache_dir=os.path.join(dataset_dir, ".cache"),
        modality=modality,
        text_sub_mode=text_sub_mode,
        prompt_column=prompt_column,
        image_sub_mode=_ism,
        image_path_column=image_path_column,
    )
    return context, resolve_dataset_source_adapter(context)


def _load_patch_bundle(patches_path: str) -> PatchBundle:
    logger.info("\n--- Loading patch bundle ---")
    bundle = PatchBundle(
        overrides={
            "text_perfect": {},
            "text_verbatim": {},
        },
        patches_path=patches_path,
    )

    override_dirs = {
        "text_perfect": os.path.join(patches_path, "override_text_perfect"),
        "text_verbatim": os.path.join(patches_path, "override_text_verbatim"),
    }
    for column, override_dir in override_dirs.items():
        if not os.path.exists(override_dir):
            logger.info(f"No {column} override directory found: {override_dir}")
            continue
        logger.info(f"Processing {column} overrides from {override_dir}")
        for override_file in glob(os.path.join(override_dir, "*.csv")):
            override_df = _read_patch_csv(override_file)
            if "id" not in override_df.columns or column not in override_df.columns:
                logger.warning(f"Override file {override_file} missing required columns ('id', '{column}'). Skipping.")
                continue
            override_df = override_df.dropna(subset=["id"])
            for _, row in override_df.iterrows():
                bundle.overrides[column][_normalize_sample_id(row["id"])] = (
                    "" if pd.isna(row[column]) else str(row[column])
                )

    bundle.protected_ids = _load_patch_id_set(os.path.join(patches_path, "do_not_blacklist"), "protected")
    bundle.blacklist_ids = _load_patch_id_set(os.path.join(patches_path, "delete"), "blacklisted")
    return bundle


def _load_patch_id_set(directory: str, label: str) -> set[str]:
    identifiers: set[str] = set()
    if not os.path.exists(directory):
        logger.info(f"No {label} directory found: {directory}")
        return identifiers
    logger.info(f"Loading {label} sample IDs from {directory}")
    for patch_file in glob(os.path.join(directory, "*.csv")):
        patch_df = _read_patch_csv(patch_file)
        if "id" not in patch_df.columns:
            logger.warning(f"Patch file {patch_file} missing 'id' column. Skipping.")
            continue
        loaded_ids = {_normalize_sample_id(value) for value in patch_df["id"].dropna()}
        identifiers.update(loaded_ids)
        logger.info(f"  Added {len(loaded_ids)} {label} IDs from {os.path.basename(patch_file)}")
    return identifiers


def _apply_patch_bundle(dataset, patch_bundle, configured_text_col, num_workers, original_count):
    logger.info("\n--- Phase 1: Override Application ---")
    override_counts = {}
    for column, override_dict in patch_bundle.overrides.items():
        override_counts[column] = 0
        if not override_dict:
            continue

        # Mutable list used as a closure counter so the inner function can increment it.
        # A plain int would not be mutable from inside a nested function in Python 2-style
        # closures; a list cell avoids the nonlocal keyword and works across Python versions.
        _override_count = [0]

        def apply_overrides(
            example,
            _overrides=override_dict,
            _column=column,
            _text_col=configured_text_col,
            _counter=_override_count,
        ):
            # Start from a full copy so all original columns are preserved.
            new_example = example.copy()
            sample_id = _normalize_sample_id(example["id"])
            if sample_id in _overrides:
                # Sample has a manual correction: apply it and mirror to the base text col.
                value = _overrides[sample_id]
                # Only write the override column if it already exists in the row to
                # avoid introducing a new column that not every row has (which breaks
                # the Arrow schema). The base text column is always written.
                if _column in example:
                    new_example[_column] = value
                if _text_col:
                    new_example[_text_col] = value
                _counter[0] += 1
            else:
                # Sample has no override: preserve the existing column value if present.
                # Do NOT create the column with an empty-string fallback — that would
                # inject corrupt empty transcripts into every row that lacks the column.
                if _column in example:
                    new_example[_column] = example[_column]
                # else: column doesn't exist for this sample — leave it absent
            return new_example

        # The closure counter only works correctly in single-process mode. In multiprocessing,
        # each worker receives a forked copy of _override_count and increments its own copy;
        # the parent copy stays at 0. Force num_proc=None for the override phase to guarantee
        # accurate statistics. The dataset is typically small at this point (patch phase runs
        # before any large-scale filtering), so the single-process overhead is acceptable.
        override_num_proc = None
        if num_workers is not None and num_workers > 1:
            logger.warning(
                "Override closure counter requires single-process map; forcing num_proc=None "
                "for the override phase (num_workers=%s will be used for subsequent steps).",
                num_workers,
            )
        dataset = dataset.map(apply_overrides, num_proc=override_num_proc)
        # Use the closure counter for accurate modification count instead of computing
        # a set intersection between override keys and dataset IDs (which only measures
        # overlap, not actual rows touched by the map).
        override_counts[column] = _override_count[0]

    logger.info("\n--- Phase 2: Blacklist Filtering ---")
    filtered_count = 0
    if patch_bundle.blacklist_ids:
        logger.info(f"Applying blacklist filtering (respecting {len(patch_bundle.protected_ids)} protected samples)")

        def should_keep_sample(example, _blacklist=patch_bundle.blacklist_ids, _protected=patch_bundle.protected_ids):
            sample_id = _normalize_sample_id(example["id"])
            return sample_id not in _blacklist or sample_id in _protected

        dataset = dataset.filter(should_keep_sample, num_proc=num_workers)
        filtered_count = (original_count - len(dataset)) if original_count is not None else 0
        logger.info(f"Filtered out {filtered_count} samples after applying protections")
    else:
        logger.info("No blacklist filtering applied (no blacklist files found)")

    return dataset, override_counts, filtered_count


def _apply_patches_streaming(dataset, patch_bundle, source, max_samples=None, configured_text_col=None):
    """
    Applies patches to a streaming dataset on-the-fly during iteration.

    This function loads all patch files into memory (small footprint, few MB)
    and applies them during dataset iteration. This enables processing of
    arbitrarily large datasets without loading them fully into memory.

    Streaming mode differences from non-streaming:
    - Patches applied per-sample during iteration, not via dataset.map()
    - No dataset length available for progress tracking
    - Blacklisting happens during iteration, not via dataset.filter()
    - Statistics are estimates based on samples seen

    Memory footprint:
    - Override dictionaries: ~1-10MB for thousands of corrections
    - Blacklist/protection sets: ~1MB for thousands of IDs
    - Total overhead: <50MB even for large patch sets

    Args:
        dataset: HuggingFace IterableDataset in streaming mode
        patches_path (str): Directory containing patch files
        source (str): Dataset source identifier
        max_samples (int | None): Limit samples (applied during iteration)

    Returns:
        tuple[IterableDataset, str]: Wrapped dataset with patches, source ID
    """
    logger.info("\n--- Streaming patch application ---")
    override_dicts = patch_bundle.overrides
    logger.info(f"  Total protected IDs: {len(patch_bundle.protected_ids)}")
    logger.info(f"  Total blacklisted IDs: {len(patch_bundle.blacklist_ids)}")

    # Create generator that applies patches on-the-fly
    def stream_with_patches():
        """
        Generator that yields patched samples from the streaming dataset.

        Processing order per sample:
        1. Check blacklist (skip if blacklisted and not protected)
        2. Apply text overrides if ID matches
        3. Yield processed sample
        """
        samples_seen = 0
        samples_yielded = 0
        samples_blacklisted = 0
        samples_overridden = {col: 0 for col in override_dicts}

        for example in dataset:
            samples_seen += 1

            # Check sample limit
            if max_samples is not None and samples_yielded >= max_samples:
                break

            sample_id = _normalize_sample_id(example.get("id"))

            # Apply blacklist filtering
            if sample_id in patch_bundle.blacklist_ids and sample_id not in patch_bundle.protected_ids:
                samples_blacklisted += 1
                continue  # Skip this sample

            # Apply text overrides
            for column, overrides in override_dicts.items():
                if sample_id in overrides:
                    example[column] = overrides[sample_id]
                    if configured_text_col:
                        example[configured_text_col] = overrides[sample_id]
                    samples_overridden[column] += 1

            samples_yielded += 1
            yield example

        # Print final statistics
        logger.info("\n--- Streaming Processing Statistics ---")
        logger.info(f"Samples seen:        {samples_seen}")
        logger.info(f"Samples yielded:     {samples_yielded}")
        logger.info(f"Samples blacklisted: {samples_blacklisted}")

        for column, count in samples_overridden.items():
            if count > 0:
                logger.info(f"Overrides ({column}): {count}")

    # Wrap the generator as an IterableDataset
    from datasets import IterableDataset

    # Create the wrapped dataset
    patched_dataset = IterableDataset.from_generator(stream_with_patches)

    logger.info("\nStreaming dataset with patches prepared")
    logger.info("=" * 50)

    return patched_dataset, source


def _read_patch_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Error processing patch file '{path}': {exc}") from exc


def _normalize_sample_id(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text
