#!/usr/bin/env python3
"""
Model Evaluation System for Gemma Fine-Tuning

Evaluates fine-tuned Gemma models with WER/CER metrics, multi-language support,
and detailed CSV prediction export.

Called by:
- core/ops.py:evaluate() as the primary evaluation dispatcher
- main.py:main() for "evaluate" operation execution

Calls to:
- scripts/inference_common.py for shared model loading, dataset prep, and DataLoader setup
- core/inference.py:decode_and_score(), generate() for inference and scoring
- evaluate library for WER/CER metric computation

Shared infrastructure (model loading, dataset vectorization, collator, DataTrainingArguments,
EvalDataCollator) lives in scripts/inference_common.py to avoid duplication with blacklist.py.
"""

import csv
import logging
import os

import torch
from tabulate import tabulate
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

# Evaluation Constants for Consistent Processing
DEFAULT_MAX_DURATION = 30.0  # Maximum audio duration for processing
DEFAULT_BATCH_SIZE = 16  # Default batch size for evaluation

logger = logging.getLogger(__name__)


def run_evaluation(profile_config, output_dir):
    """
    Executes comprehensive model evaluation with metrics calculation and reporting.

    This is the main evaluation orchestration function, coordinating all aspects
    of model evaluation from dataset loading through metrics calculation and
    result export. It provides a complete evaluation pipeline with extensive
    configuration options and error handling.

    Called by:
    - main.py evaluate operations for all evaluation workflows
    - Automated evaluation pipelines in CI/CD systems
    - Research scripts for model performance analysis

    Evaluation workflow:
    1. Configuration parsing and validation
    2. Model loading with device optimization
    3. Dataset loading with patch application
    4. Audio preprocessing and feature extraction
    5. Batch inference with progress tracking
    6. Metrics calculation (WER, BLEU, etc.)
    7. Results export (CSV predictions, JSON metrics)
    8. Statistical reporting and logging

    Model loading:
    - Automatic model detection (checkpoint vs. HF model)
    - Device placement optimization (MPS/CUDA/CPU)
    - Memory management for large models
    - Error handling for corrupted checkpoints

    Dataset processing:
    - Profile-based dataset loading with patch system
    - Audio preprocessing (resampling, feature extraction)
    - Language filtering and validation
    - Sample limiting for development/testing

    Inference optimization:
    - Batch processing for GPU efficiency
    - Memory management to prevent OOM errors
    - Progress tracking for long evaluations
    - Error recovery for individual sample failures

    Language handling:
    - Language mode configuration (strict/flexible)
    - Forced language processing for multilingual models
    - Language detection and validation
    - Language-specific metric calculation

    Metrics calculation:
    - WER (Word Error Rate): Primary speech recognition metric
    - Additional metrics via evaluate library integration
    - Per-sample and aggregate statistics
    - Language-specific performance analysis

    Output generation:
    - Detailed CSV predictions for error analysis
    - JSON metrics for automated processing
    - Human-readable evaluation reports
    - Debug logs for troubleshooting

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
        profile_config (dict): Complete evaluation configuration including:
            - model_name_or_path: Model checkpoint or HF model identifier
            - dataset: Dataset name for evaluation
            - language settings: Language constraints and processing mode
            - evaluation parameters: Batch size, metrics, output options
            - device settings: Hardware optimization parameters
        output_dir (str): Directory for evaluation outputs and logs

    Returns:
        dict | None: Evaluation metrics dictionary containing:
            - wer: Word Error Rate (primary metric)
            - sample_count: Number of samples evaluated
            - language_distribution: Language breakdown (multilingual)
            - evaluation_duration: Time taken for evaluation
            - device_info: Hardware used for evaluation
            Returns None if evaluation fails

    Raises:
        ValueError: Invalid configuration parameters
        FileNotFoundError: Missing model checkpoints or dataset files
        RuntimeError: Model loading or inference failures

    Example:
        profile_config = {
            "model_name_or_path": "output/1-gemma-3n-e4b-it",
            "dataset": "librispeech",
            "language_mode": "strict",
            "max_samples": 1000
        }
        metrics = run_evaluation(profile_config, "output/1-gemma-3n-e4b-it/eval")
        if metrics:
            logger.info(f"WER: {metrics['wer']:.3f}")
    """
    # 1. Resolve device — called here (not at module level) so tests can mock get_device()
    device = get_device()

    # 2. Parse input arguments
    # Cap preprocessing workers on MPS by default (config override respected via profile)
    configured_workers = profile_config.get("preprocessing_num_workers", None)
    if configured_workers is not None:
        try:
            configured_workers = int(configured_workers)
        except Exception:
            configured_workers = None
    default_workers = 1 if device.type == "mps" else 8
    effective_workers = configured_workers if configured_workers is not None else default_workers

    data_args = DataTrainingArguments(
        dataset_name=profile_config["dataset"],
        audio_column_name="audio_path",
        text_column_name=profile_config["text_column"],
        max_duration_in_seconds=float(profile_config["max_duration"]),
        evaluation_split=profile_config["validation_split"],
        preprocessing_num_workers=effective_workers,
        log_predictions=True,
        streaming=False,
        return_timestamps=False,
    )

    predictions_filename = "predictions.csv"

    set_seed(42)

    # 2. Load dataset via shared helper
    dataset_config = build_dataset_config(profile_config)

    raw_datasets, source = load_dataset_split(
        split=profile_config["validation_split"], dataset_config=dataset_config, max_samples=data_args.max_samples
    )

    # 3. Load pretrained model, tokenizer, and feature extractor via shared helper
    model, processor, tokenizer, feature_extractor = load_model_and_processor(profile_config, device)
    dtype_str = profile_config.get("dtype", "float32")
    if not hasattr(torch, dtype_str):
        raise ValueError(f"Invalid dtype in profile config: {dtype_str!r}. Expected e.g. 'float32', 'bfloat16'.")
    dtype = getattr(torch, dtype_str)

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
        num_proc=1 if data_args.streaming else data_args.preprocessing_num_workers,
        filter_none=True,
    )

    # 5. Generation arguments and data collator via shared helpers
    gen_kwargs = build_gen_kwargs(profile_config, return_timestamps=data_args.return_timestamps)
    data_collator = EvalDataCollator(processor=processor, language_mode=language_mode)

    # 6. Run evaluation pipeline
    os.makedirs(output_dir, exist_ok=True)

    predictions_filepath = os.path.join(output_dir, predictions_filename)
    with open(predictions_filepath, "w", newline="", encoding="utf-8") as predictions_file:
        predictions_writer = csv.writer(predictions_file)
        predictions_writer.writerow(["ID", "Target", "Pred", "Norm Target", "Norm Pred", "WER", "CER"])

        all_preds = []
        references = []
        ids = []
        # Build all_ids AFTER the .filter() call above so this list aligns 1-to-1
        # with what the DataLoader will yield. Building from a pre-filter snapshot
        # produces a longer list and causes ID-to-prediction misalignment in the CSV.
        all_ids = [item["id"] for item in vectorized_datasets]

        # Cap dataloader workers on MPS by default unless overridden via profile
        configured_dl_workers = profile_config.get("dataloader_num_workers", None)
        if configured_dl_workers is not None:
            try:
                configured_dl_workers = int(configured_dl_workers)
            except Exception:
                configured_dl_workers = None

        eval_dataloader = build_eval_dataloader(
            vectorized_datasets=vectorized_datasets,
            batch_size=int(profile_config["per_device_eval_batch_size"]),
            data_collator=data_collator,
            device=device,
            num_workers_override=configured_dl_workers,
        )

        logger.info("***** Running Evaluation *****")
        current_batch_size = int(profile_config["per_device_eval_batch_size"])
        tried_smaller_batch = False
        id_offset = 0
        i = 0
        while True:
            try:
                for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                    input_features = batch["input_features"].to(device, dtype=dtype)
                    labels = batch["labels"].to(device)
                    batch_size = input_features.shape[0]
                    batch_ids = all_ids[id_offset : id_offset + batch_size]

                    with torch.no_grad():
                        # Set the language and task for generation.
                        # In strict mode a batch may contain samples from
                        # multiple languages.  Detect that case and generate
                        # per-language groups separately so each sample gets
                        # the correct forced decoder prompt.
                        if language_mode == "strict":
                            batch_languages = batch.get("language", [None])
                            unique_langs = set(batch_languages)

                            if len(unique_langs) == 1:
                                # Homogeneous batch -- fast path
                                lang = batch_languages[0]
                                generated_tokens = generate(
                                    model=model,
                                    processor=processor,
                                    input_features=input_features,
                                    language_mode=language_mode,
                                    forced_language=forced_language,
                                    batch_language=lang,
                                    gen_kwargs=gen_kwargs,
                                )
                            else:
                                # Heterogeneous batch -- generate per-language
                                # groups and reassemble in original sample order.
                                import numpy as np

                                generated_parts = [None] * input_features.shape[0]
                                for lang in unique_langs:
                                    indices = [j for j, l in enumerate(batch_languages) if l == lang]
                                    lang_features = input_features[indices]
                                    lang_tokens = generate(
                                        model=model,
                                        processor=processor,
                                        input_features=lang_features,
                                        language_mode=language_mode,
                                        forced_language=forced_language,
                                        batch_language=lang,
                                        gen_kwargs=gen_kwargs,
                                    )
                                    for k, idx in enumerate(indices):
                                        generated_parts[idx] = lang_tokens[k]

                                # Validate all parts were filled
                                unfilled = [i for i, p in enumerate(generated_parts) if p is None]
                                if unfilled:
                                    raise RuntimeError(
                                        f"Heterogeneous batch assembly failed: indices {unfilled} were not filled. "
                                        f"Languages in batch: {batch_languages}"
                                    )

                                # Pad ragged per-language arrays to the same seq length before stacking
                                max_seq_len = max(p.shape[-1] for p in generated_parts)
                                padded_parts = []
                                for p in generated_parts:
                                    if p.shape[-1] < max_seq_len:
                                        pad_width = max_seq_len - p.shape[-1]
                                        p = np.pad(
                                            p, ((0, pad_width),), constant_values=processor.tokenizer.pad_token_id
                                        )
                                    padded_parts.append(p)
                                generated_tokens = np.stack(padded_parts)

                        elif language_mode == "mixed":
                            generated_tokens = generate(
                                model=model,
                                processor=processor,
                                input_features=input_features,
                                language_mode=language_mode,
                                forced_language=forced_language,
                                batch_language=None,
                                gen_kwargs=gen_kwargs,
                            )
                        elif language_mode.startswith("override:"):
                            generated_tokens = generate(
                                model=model,
                                processor=processor,
                                input_features=input_features,
                                language_mode=language_mode,
                                forced_language=forced_language,
                                batch_language=None,
                                gen_kwargs=gen_kwargs,
                            )
                        else:
                            raise ValueError(f"Invalid language mode: {language_mode}")

                    # Decode generated tokens for the entire batch
                    batch_preds = processor.batch_decode(generated_tokens, skip_special_tokens=True)
                    batch_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

                    all_preds.extend(batch_preds)
                    references.extend(batch_labels)
                    ids.extend(batch_ids)
                    id_offset += batch_size
                break
            except RuntimeError as e:
                msg = str(e).lower()
                oom = (
                    ("out of memory" in msg)
                    or ("cuda" in msg and "memory" in msg)
                    or (device.type == "mps" and "failed to allocate" in msg)
                )
                if not oom or tried_smaller_batch or current_batch_size <= 1:
                    raise
                tried_smaller_batch = True
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    f"OOM detected during evaluation. Retrying with smaller batch size: {current_batch_size}"
                )
                # Clear accumulated results to avoid duplicates on retry
                all_preds.clear()
                references.clear()
                ids.clear()
                id_offset = 0
                eval_dataloader = build_eval_dataloader(
                    vectorized_datasets=vectorized_datasets,
                    batch_size=current_batch_size,
                    data_collator=data_collator,
                    device=device,
                    num_workers_override=configured_dl_workers,
                )

        # Compute metrics via shared helper
        # normalizer=None lets decode_and_score use its built-in EnglishTextNormalizer()
        # default — avoids calling tokenizer.english_spelling_normalizer which does not
        # exist on Gemma tokenizers.
        wer, cer, pred_str, label_str, norm_pred_str, norm_label_str = decode_and_score(
            all_preds, references, normalizer=None
        )

        # Prepare the metrics dictionary to be returned
        metrics_dict = {
            "wer": wer,
            "cer": cer,
        }
        if metrics_dict["wer"] is not None:
            logger.info(f"WER: {wer}")
            logger.info(f"CER: {cer}")

            # Log predictions
            if data_args.log_predictions:
                str_data = []
                # Compute per-row WER/CER via shared helper (also used by blacklist.py)
                row_wers, row_cers = compute_per_row_metrics(norm_pred_str, norm_label_str)
                for i in range(len(pred_str)):
                    row_wer = row_wers[i] if i < len(row_wers) else None
                    row_cer = row_cers[i] if i < len(row_cers) else None
                    str_data.append(
                        [
                            ids[i] if i < len(ids) else "",
                            label_str[i],
                            pred_str[i],
                            norm_label_str[i],
                            norm_pred_str[i],
                            row_wer,
                            row_cer,
                        ]
                    )
                predictions_writer.writerows(str_data)
                try:
                    predictions_file.flush()
                except Exception:
                    pass

    # Context manager automatically closes the file here
    # Calculate relative path
    project_root = os.getcwd()
    relative_path = os.path.relpath(predictions_filepath, project_root)

    logger.info(f"Evaluation results saved to: {relative_path}")

    # Tabular output of metrics
    if wer is not None:
        table = [["WER", metrics_dict["wer"]], ["CER", metrics_dict["cer"]]]
        logger.info("\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

    return metrics_dict
