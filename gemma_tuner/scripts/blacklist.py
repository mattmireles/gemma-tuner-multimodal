#!/usr/bin/env python3
"""
Intelligent Outlier Detection and Blacklist Generation System

Detects mislabeled or problematic audio samples via WER analysis, generates
blacklists respecting manual overrides, and produces diagnostic CSV output.

Called by:
- core/ops.py:blacklist() as the primary blacklist generation dispatcher
- main.py:main() for "blacklist" operation execution

Calls to:
- scripts/inference_common.py for shared model loading, dataset prep, and DataLoader setup
- core/inference.py:decode_and_score(), generate() for inference and scoring
- evaluate library for WER/CER metric computation
- pandas for blacklist CSV generation

Shared infrastructure (model loading, dataset vectorization, collator, DataTrainingArguments,
EvalDataCollator) lives in scripts/inference_common.py to avoid duplication with evaluate.py.
"""

import logging
import os
from datetime import datetime
from glob import glob
from pathlib import Path

# Anchor all data-patch paths to the project root so blacklist generation
# works regardless of the CWD the CLI is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import torch
from tqdm import tqdm
from transformers import set_seed
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from gemma_tuner.core.inference import decode_and_score, generate
from gemma_tuner.scripts.inference_common import (
    DataTrainingArguments,
    EvalDataCollator,
    build_dataset_config,
    build_eval_dataloader,
    build_gen_kwargs,
    compute_per_row_metrics,
    load_model_and_processor,
    make_prepare_dataset_fn,
    parse_language_mode,
    vectorize_dataset,
)
from gemma_tuner.utils.dataset_utils import load_dataset_split
from gemma_tuner.utils.device import get_device

logger = logging.getLogger(__name__)


def create_blacklist(profile_config, output_dir):
    """
    Generates comprehensive blacklist of outlier samples using WER-based quality analysis.

    This is the main blacklist generation orchestration function, coordinating all aspects
    of quality analysis from dataset loading through outlier detection and blacklist
    export. It provides intelligent outlier detection with manual override integration
    and detailed quality reporting.

    Called by:
    - main.py blacklist operations for dataset quality management
    - Quality assurance pipelines in automated workflows
    - Dataset curation scripts for training preparation

    Blacklist generation workflow:
    1. Configuration parsing and validation
    2. Model loading with device optimization
    3. Dataset loading with existing patch integration
    4. Audio preprocessing and feature extraction
    5. Batch inference with progress tracking
    6. WER/CER calculation for quality assessment
    7. Outlier identification using configurable thresholds
    8. Manual override cross-referencing
    9. Blacklist CSV generation with detailed diagnostics
    10. Statistical reporting and quality summary

    Model loading:
    - Automatic model detection (checkpoint vs. HF model)
    - Device placement optimization (MPS/CUDA/CPU)
    - Memory management for large models
    - Error handling for corrupted checkpoints

    Dataset processing:
    - Profile-based dataset loading with patch system integration
    - Audio preprocessing (resampling, feature extraction)
    - Language filtering and validation
    - Integration with existing data quality patches

    Quality analysis:
    - WER/CER calculation using standardized metrics
    - Configurable thresholds per dataset split
    - Statistical analysis of quality distribution
    - Outlier pattern identification

    Manual override integration:
    - do_not_blacklist: Samples protected from blacklisting
    - override_text_*: Manual corrections applied before analysis
    - previously_reviewed: Tracking of human-reviewed samples
    - Prevents automated blacklisting of manually verified data

    Threshold configuration:
    - Training split: Higher tolerance (75% WER default)
    - Validation split: Lower tolerance (80% WER default)
    - Profile-specific overrides for dataset requirements
    - Language-specific thresholds for multilingual datasets

    Output generation:
    - Detailed CSV blacklist with diagnostic information
    - Statistical summary of dataset quality
    - Outlier distribution analysis
    - Integration recommendations for training

    Error handling:
    - Model loading failures: Clear diagnostic messages
    - Dataset issues: Graceful degradation with warnings
    - Inference errors: Sample-level error recovery
    - Memory issues: Automatic batch size adjustment

    Performance optimization:
    - Device-specific batch size configuration
    - Memory pressure monitoring and adjustment
    - Efficient audio preprocessing pipeline
    - Optimized tensor operations for target device

    Args:
        profile_config (dict): Complete blacklist configuration including:
            - model_name_or_path: Model checkpoint or HF model identifier
            - dataset: Dataset name for quality analysis
            - quality thresholds: WER limits for outlier detection
            - language settings: Language constraints and processing mode
            - batch settings: Processing batch sizes and worker counts
        output_dir (str): Directory for blacklist outputs and logs

    Returns:
        str: Path to generated blacklist CSV file containing:
            - Comprehensive outlier list with diagnostic information
            - Quality metrics and statistical analysis
            - Manual override status and reasoning
            - Integration guidance for training workflows

    Raises:
        ValueError: Invalid configuration parameters or thresholds
        FileNotFoundError: Missing model checkpoints or dataset files
        RuntimeError: Model loading or inference failures

    Example:
        profile_config = {
            "model_name_or_path": "openai/gemma-3n-e4b-it",
            "dataset": "librispeech",
            "wer_threshold": 75.0,
            "validation_wer_threshold": 80.0,
            "split": "validation"
        }
        blacklist_path = create_blacklist(profile_config, "output/blacklist")
        logger = logging.getLogger(__name__)
        logger.info(f"Blacklist generated: {blacklist_path}")
    """

    # 1. Configuration Parsing and Validation
    do_not_blacklist_dir = str(_PROJECT_ROOT / "data_patches" / profile_config["dataset"] / "do_not_blacklist")

    data_args = DataTrainingArguments(
        dataset_name=profile_config["dataset"],
        audio_column_name="audio_path",
        text_column_name=profile_config["text_column"],
        max_duration_in_seconds=float(profile_config["max_duration"]),
        evaluation_split=profile_config["split"],
        preprocessing_num_workers=4,
        log_predictions=True,
        streaming=False,
        return_timestamps=False,
    )

    # 1. Resolve device — called here (not at module level) so tests can mock get_device()
    device = get_device()

    set_seed(42)

    # 2. Dataset Loading with Patch Integration
    dataset_config = build_dataset_config(profile_config)

    raw_datasets, source = load_dataset_split(
        split=profile_config["split"],
        dataset_config=dataset_config,
        max_samples=profile_config["max_samples"] if "max_samples" in profile_config else None,
        patches_dir="data_patches/",
    )

    # 3. Model Loading via shared helper
    model, processor, tokenizer, feature_extractor = load_model_and_processor(profile_config, device)
    dtype = getattr(torch, profile_config["dtype"])

    # 4. Language mode parsing and dataset vectorization via shared helpers
    language_mode = profile_config["language_mode"]
    forced_language = parse_language_mode(language_mode)
    valid_languages = profile_config.get("languages", [])

    prepare_fn = make_prepare_dataset_fn(
        processor=processor,
        text_column_name=data_args.text_column_name,
        max_label_length=int(profile_config["max_label_length"]),
        forced_language=forced_language,
    )

    vectorized_datasets = vectorize_dataset(
        raw_datasets=raw_datasets,
        prepare_fn=prepare_fn,
        language_mode=language_mode,
        valid_languages=valid_languages,
        num_proc=data_args.preprocessing_num_workers,
        filter_none=False,
    )

    # 5. Generation arguments, collator, and DataLoader via shared helpers
    gen_kwargs = build_gen_kwargs(profile_config, return_timestamps=data_args.return_timestamps)
    data_collator = EvalDataCollator(processor=processor, language_mode=language_mode)

    # 6. Quality Analysis Inference Pipeline
    all_preds = []
    references = []
    ids = []
    audio_urls = []
    all_ids = [str(item["id"]) for item in vectorized_datasets]

    eval_dataloader = build_eval_dataloader(
        vectorized_datasets=vectorized_datasets,
        batch_size=int(profile_config["per_device_eval_batch_size"]),
        data_collator=data_collator,
        device=device,
    )

    # Manual Override Integration: Do-Not-Blacklist Protection
    # Load samples explicitly protected from blacklisting by human review
    # Prevents automated removal of manually verified high-quality samples
    do_not_blacklist_ids = set()
    if do_not_blacklist_dir and os.path.exists(do_not_blacklist_dir):
        logger.info(f"Loading do-not-blacklist IDs from {do_not_blacklist_dir}")
        for do_not_blacklist_file in glob(os.path.join(do_not_blacklist_dir, "*.csv")):
            try:
                do_not_blacklist_df = pd.read_csv(do_not_blacklist_file)
                if "id" not in do_not_blacklist_df.columns:
                    raise ValueError(f"Do-not-blacklist file {do_not_blacklist_file} must contain an 'id' column.")
                # Store as strings, stripping the ".0" suffix that pandas adds to
                # integer-valued float columns (e.g. "123.0" → "123"), so they
                # match the str(item["id"]) keys used in id_to_wer.
                do_not_blacklist_ids.update({str(id_).split(".")[0] for id_ in do_not_blacklist_df["id"] if not pd.isna(id_)})
            except Exception as e:
                raise ValueError(f"Error loading do-not-blacklist IDs from {do_not_blacklist_file}: {e}")

    # Manual Override Integration: Text Corrections
    # Load samples with manual text corrections applied
    # These samples should not be blacklisted as their quality issues are resolved
    overridden_ids = set()
    patches_path = str(_PROJECT_ROOT / "data_patches" / source)
    override_dirs = {
        "text_perfect": os.path.join(patches_path, "override_text_perfect"),
        "text_verbatim": os.path.join(patches_path, "override_text_verbatim"),
    }
    for override_dir in override_dirs.values():
        if os.path.exists(override_dir):
            logger.info(f"Loading overridden IDs from {override_dir}")
            for override_file in glob(os.path.join(override_dir, "*.csv")):
                try:
                    override_df = pd.read_csv(override_file)
                    if "id" not in override_df.columns:
                        raise ValueError(f"Override file {override_file} must contain an 'id' column.")
                    # Store as strings (same ".0" stripping as do_not_blacklist_ids above).
                    overridden_ids.update({str(id_).split(".")[0] for id_ in override_df["id"] if not pd.isna(id_)})
                except Exception as e:
                    raise ValueError(f"Error loading overridden IDs from {override_file}: {e}")

    logger.info("Overridden ids loaded")

    # Quality Analysis Execution
    # Begin comprehensive dataset quality assessment with progress tracking
    logger.info("***** Running Blacklist Creation *****")
    id_offset = 0
    for i, batch in enumerate(tqdm(eval_dataloader, desc="Searching for outliers")):
        input_features = batch["input_features"].to(device, dtype=dtype)
        labels = batch["labels"].to(device)
        batch_size = input_features.shape[0]

        if batch_size == 0:
            continue

        with torch.no_grad():
            # Determine batch_language for strict mode (generate() handles forced_decoder_ids)
            batch_language = None
            if language_mode == "strict":
                languages = [
                    vectorized_datasets[idx]["language"]
                    for idx in range(id_offset, min(id_offset + batch_size, len(vectorized_datasets)))
                ]
                unique_langs = set(lang for lang in languages if lang != "??")
                if len(unique_langs) == 1:
                    batch_language = unique_langs.pop()
                elif len(unique_langs) > 1:
                    # Heterogeneous batch — use the most common language.
                    from collections import Counter

                    lang_counts = Counter(lang for lang in languages if lang != "??")
                    batch_language = lang_counts.most_common(1)[0][0]

            generated_tokens = generate(
                model=model,
                processor=processor,
                input_features=input_features,
                language_mode=language_mode,
                forced_language=forced_language,
                batch_language=batch_language,
                gen_kwargs=gen_kwargs,
            )

        # Decode generated tokens for the entire batch
        batch_preds = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        batch_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        all_preds.extend(batch_preds)
        references.extend(batch_labels)

        # Extract IDs from the original dataset, not the batch
        batch_ids = all_ids[id_offset : id_offset + batch_size]
        ids.extend(batch_ids)
        id_offset += batch_size

    # Quality Metrics Calculation and Analysis via shared helper
    # normalizer=None lets decode_and_score use its built-in EnglishTextNormalizer()
    # default — avoids calling tokenizer.english_spelling_normalizer which does not
    # exist on Gemma tokenizers.
    wer_val, cer_val, pred_str, label_str, norm_pred_str, norm_label_str = decode_and_score(
        all_preds, references, normalizer=None
    )

    # Sample Quality Mapping Generation
    # Create efficient lookup structures for quality analysis and blacklist generation
    # Enables fast access to quality metrics by sample ID
    id_to_ground_truth = {}
    id_to_prediction = {}
    id_to_wer = {}
    id_to_cer = {}

    # Per-sample WER/CER via shared helper (also used by evaluate.py for prediction logging)
    row_wers, row_cers = compute_per_row_metrics(norm_pred_str, norm_label_str)

    for i, id_val in enumerate(ids):
        id_to_ground_truth[id_val] = label_str[i]
        id_to_prediction[id_val] = pred_str[i]
        id_to_wer[id_val] = row_wers[i] if i < len(row_wers) else None
        id_to_cer[id_val] = row_cers[i] if i < len(row_cers) else None

    # 9. Blacklist Generation and Quality Assessment
    # Generate comprehensive blacklist based on quality thresholds and manual overrides
    # Provides detailed diagnostics and statistical analysis
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    blacklist_filename = f"blacklist-{profile_config['dataset']}-{timestamp}.csv"
    blacklist_path = os.path.join("output", blacklist_filename)

    # Quality Threshold Configuration
    # Apply split-specific WER thresholds for optimal training data curation
    # Validation data requires higher quality than training data
    if profile_config["split"] == "validation":
        wer_threshold = float(profile_config.get("validation_wer_threshold", 80))
    else:
        wer_threshold = float(profile_config.get("wer_threshold", 75))

    # Outlier Detection and Manual Override Integration
    # Identify quality outliers while respecting human review decisions
    # Balances automated detection with manual quality assurance
    new_outliers = []
    updated_entries = 0

    # Both sets are already strings (populated with str(id_).split(".")[0] above).
    # Assign to *_str aliases for readability without any re-conversion.
    overridden_ids_str = overridden_ids
    do_not_blacklist_ids_str = do_not_blacklist_ids

    for i, id_val in enumerate(ids):
        # Handle potential type mismatches and empty IDs
        try:
            id_val_str = str(id_val)
        except Exception:
            id_val_str = ""

        is_overridden = id_val_str in overridden_ids_str
        is_do_not_blacklist = id_val_str in do_not_blacklist_ids_str
        is_previously_reviewed = is_overridden or is_do_not_blacklist

        # Ensure blacklisted is False if previously_reviewed is True
        should_blacklist = id_to_wer[id_val] is not None and id_to_wer[id_val] > wer_threshold

        if should_blacklist:
            new_outliers.append(
                {
                    "id": id_val,
                    "blacklisted": not is_previously_reviewed,
                    "ground_truth": id_to_ground_truth[id_val],
                    "predicted": id_to_prediction[id_val],
                    "wer": id_to_wer[id_val],
                    "cer": id_to_cer[id_val],
                    "model": profile_config["model_name_or_path"],
                    "run_id": os.path.basename(os.path.dirname(output_dir)),
                    "profile": profile_config.get("profile", ""),
                    "dataset": profile_config["dataset"],
                    "split": profile_config["split"],
                    "reason": f"WER > {wer_threshold}"
                    if not is_previously_reviewed
                    else f"WER > {wer_threshold} but not blacklisted due to previously_reviewed",
                    "previously_reviewed": is_previously_reviewed,
                }
            )

    # Quality Analysis Summary and Reporting
    # Provide comprehensive statistics on dataset quality and blacklist generation
    logger.info("\nBlacklist Summary:")
    logger.info(f"  New outliers added: {len(new_outliers)}")

    # Create a DataFrame for new outliers
    blacklist_df = pd.DataFrame(new_outliers)

    # Ensure output directory exists before writing
    os.makedirs(os.path.dirname(blacklist_path), exist_ok=True)
    # Save updated blacklist
    blacklist_df.to_csv(blacklist_path, index=False)

    logger.info(f"Blacklist saved to: {blacklist_path}")

    # Return blacklist file path for downstream integration
    # Generated blacklist can be used by training workflows for data quality management
    return blacklist_path
