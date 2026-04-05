#!/usr/bin/env python3
"""
Shared Inference Infrastructure for Gemma Evaluation and Blacklist Scripts

This module extracts the common boilerplate shared by evaluate.py and blacklist.py:
model/processor loading, dataset preparation, feature extraction, language-mode
parsing, and DataLoader construction.  Both scripts import from here instead of
duplicating ~150 lines of identical setup code.

Called by:
- scripts/evaluate.py:run_evaluation() for model loading, dataset prep, and DataLoader setup
- scripts/blacklist.py:create_blacklist() for the same

Calls to:
- utils/device.py:get_device() for platform-specific device selection
- utils/dataset_utils.py:load_dataset_split() for dataset loading with patch system
- utils/dataset_prep.py:encode_labels(), load_audio_local_or_gcs(), resolve_language()
- transformers.AutoModelForCausalLM.from_pretrained() for model loading
- transformers.AutoProcessor.from_pretrained() for processor loading
- transformers.set_seed() for reproducibility

Design rationale:
- Functions return plain data (model, processor, dataset, dataloader) -- no workflow logic.
- Callers own their own control flow (evaluation loop, blacklist loop, OOM retry, etc.).
- Only truly duplicated code lives here; script-specific logic stays in each script.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from gemma_tuner.models.gemma.constants import AudioProcessingConstants
from gemma_tuner.utils.dataset_prep import (
    encode_labels,
    load_audio_local_or_gcs,
    resolve_language,
)

# Version Compatibility Enforcement
# Gemma inference requires specific transformers and datasets versions for:
# - Model loading compatibility and consistent inference behavior
# - Feature extraction consistency across runs
# - SafeTensors support for secure model loading
check_min_version("4.53.0")
require_version("datasets>=4.0.0", "To fix: pip install -U datasets")

logger = logging.getLogger(__name__)

# Inference Constants for Consistent Processing
GEMMA_SAMPLING_RATE = AudioProcessingConstants.DEFAULT_SAMPLING_RATE  # Required sampling rate for Gemma models
EVALUATION_SEED = 42  # Fixed seed for reproducible inference results


@dataclass
class DataTrainingArguments:
    """
    Configuration parameters for inference dataset processing and model input.

    Shared between evaluate.py and blacklist.py.  Both scripts construct this
    from their profile_config dict, optionally adding script-specific fields
    (e.g. blacklist's ``do_not_blacklist_dir``) as plain local variables.

    Configuration categories:

    Dataset identification:
    - dataset_name: HuggingFace dataset identifier or custom dataset name

    Caching and performance:
    - preprocessing_num_workers: Parallel processing workers

    Audio processing:
    - audio_column_name: Dataset column containing audio file paths
    - max_duration_in_seconds: Audio length filtering threshold

    Text processing:
    - text_column_name: Dataset column containing reference transcriptions

    Constraints:
    - max_samples: Limit size for testing/debugging
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Override dataset from profile configuration."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    audio_column_name: str = field(
        default="audio_path",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio_path'"},
    )
    text_column_name: str = field(
        default="text_verbatim",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text_verbatim'."},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of eval examples to this value if set."},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return the timestamps with the text. This enables the `FlaxWhisperTimestampsLogitsProcessor`."
        },
    )
    log_predictions: bool = field(
        default=True,
        metadata={"help": "Whether or not to log the ground truths / pred text to the logging platform."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use Datasets' streaming mode to load and the data."},
    )
    evaluation_split: str = field(
        default="validation",
        metadata={"help": "Which split of the dataset to use for evaluation. Defaults to 'validation'."},
    )


@dataclass
class EvalDataCollator:
    """
    Data collator for batching audio samples during inference.

    Handles padding of variable-length audio features and label tensors for
    efficient batch processing.  Used identically by both evaluate.py and
    blacklist.py DataLoaders.

    Called by:
    - DataLoader during evaluation/blacklist batch processing

    Responsibilities:
    - Audio feature tensor batching with padding
    - Label tensor preparation with -100 masking for ignored positions
    - Language metadata preservation (strict mode groups by language)

    Args:
        processor: AutoProcessor for feature extraction and tokenization
        language_mode: Language processing mode ("mixed", "strict", "override:lang")
    """

    processor: Any
    language_mode: str = "mixed"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self.language_mode == "strict":
            # Group features by language
            features_by_language = {}
            for feature in features:
                lang = feature["language"]
                if lang not in features_by_language:
                    features_by_language[lang] = []
                features_by_language[lang].append(feature)

            # Process each language group separately
            batches = []
            for lang, lang_features in features_by_language.items():
                input_features = [{"input_features": feature["input_features"]} for feature in lang_features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

                label_features = [{"input_ids": feature["labels"]} for feature in lang_features]
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
                batch["labels"] = labels.long()
                batch["language"] = [lang for _ in lang_features]
                batches.append(batch)

            # Pad each batch to max length across language groups, then concatenate.
            # input_features shape is [batch, mel_bins, time_frames] — use shape[2] for the
            # variable time dimension, NOT shape[1] (mel_bins is fixed, e.g. 80 or 128).
            max_len_input_features = max(batch["input_features"].shape[2] for batch in batches)
            max_len_labels = max(batch["labels"].shape[1] for batch in batches)

            padded_batches = []
            for batch in batches:
                padded_batch = {}
                padding_size_input_features = max_len_input_features - batch["input_features"].shape[2]
                padding_size_labels = max_len_labels - batch["labels"].shape[1]

                # F.pad tuple is applied right-to-left on dimensions:
                # (0, padding_size) pads only the last dim (time), leaving mel_bins untouched.
                padded_batch["input_features"] = torch.nn.functional.pad(
                    batch["input_features"], (0, padding_size_input_features)
                )
                padded_batch["labels"] = torch.nn.functional.pad(batch["labels"], (0, padding_size_labels), value=-100)
                padded_batch["language"] = batch["language"]
                padded_batches.append(padded_batch)

            final_batch = {}
            final_batch["input_features"] = torch.cat([batch["input_features"] for batch in padded_batches], dim=0)
            final_batch["labels"] = torch.cat([batch["labels"] for batch in padded_batches], dim=0)
            final_batch["language"] = [lang for batch in padded_batches for lang in batch["language"]]

            return final_batch
        else:  # mixed or override
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels.long()
            batch["language"] = [feature["language"] for feature in features]

            return batch


def parse_language_mode(language_mode):
    """
    Parses the language_mode string into (forced_language, valid_languages_ignored_here).

    Called by:
    - evaluate.py:run_evaluation() and blacklist.py:create_blacklist()

    Returns:
        forced_language (str | None): The language code if mode is "override:<lang>", else None.

    Raises:
        ValueError: If "override:" prefix is present but no language follows.
    """
    if language_mode.startswith("override:"):
        parts = language_mode.split(":", 1)
        if len(parts) > 1:
            return parts[1]
        else:
            raise ValueError(f"Invalid override language mode format: {language_mode}. Expected 'override:language'")
    return None


def load_model_and_processor(profile_config, device):
    """
    Loads AutoModelForCausalLM + AutoProcessor from profile config.

    Handles both local checkpoint directories and HuggingFace Hub model identifiers.
    Applies dtype and attention implementation settings from the profile.

    Called by:
    - scripts/evaluate.py:run_evaluation()
    - scripts/blacklist.py:create_blacklist()

    Args:
        profile_config (dict): Must contain keys:
            - model_name_or_path: Local path or HF model id
            - dtype: Torch dtype string (e.g. "float32")
            - attn_implementation: Attention implementation (e.g. "eager", "sdpa")
        device (torch.device): Target device for model placement

    Returns:
        tuple: (model, processor, tokenizer, feature_extractor)
            - model: AutoModelForCausalLM on target device
            - processor: AutoProcessor
            - tokenizer: processor.tokenizer (convenience alias)
            - feature_extractor: processor.feature_extractor (convenience alias, may be None)
    """
    model_name_or_path = profile_config["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    feature_extractor = getattr(processor, "feature_extractor", None)

    dtype_str = profile_config.get("dtype", "float32")
    if not hasattr(torch, dtype_str):
        raise ValueError(f"Invalid dtype in profile config: {dtype_str!r}. Expected e.g. 'float32', 'bfloat16'.")
    dtype = getattr(torch, dtype_str)

    if os.path.isdir(model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=profile_config["attn_implementation"],
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=profile_config["attn_implementation"],
            low_cpu_mem_usage=True,
        ).to(device)

    return model, processor, tokenizer, feature_extractor


def build_dataset_config(profile_config):
    """
    Builds the dataset_config dict expected by load_dataset_split().

    Called by:
    - scripts/evaluate.py:run_evaluation()
    - scripts/blacklist.py:create_blacklist()

    Args:
        profile_config (dict): Profile configuration with dataset keys.

    Returns:
        dict: Dataset configuration for load_dataset_split().
    """
    return {
        "name": profile_config["dataset"],
        "text_column": profile_config["text_column"],
        "max_label_length": int(profile_config["max_label_length"]),
        "max_duration": float(profile_config["max_duration"]),
        "id_column": profile_config.get("id_column", "id"),
        "speaker_id_column": None,
        "train_split": profile_config["train_split"],
    }


def make_prepare_dataset_fn(processor, text_column_name, max_label_length, forced_language):
    """
    Returns a prepare_dataset closure suitable for Dataset.map().

    The closure loads audio, extracts features, resolves language, and encodes
    labels -- the identical logic previously duplicated in evaluate.py and blacklist.py.

    Called by:
    - scripts/evaluate.py:run_evaluation() -- passed to raw_datasets.map()
    - scripts/blacklist.py:create_blacklist() -- passed to raw_datasets.map()

    Args:
        processor: AutoProcessor instance
        text_column_name (str): Column name for reference text
        max_label_length (int): Maximum label token length
        forced_language (str | None): Language override, or None for auto-detect

    Returns:
        callable: prepare_dataset(batch, language_mode, valid_languages) -> batch
    """
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def prepare_dataset(batch, language_mode, valid_languages):
        # Load audio via shared helper (local, GCS, or dict), CI-safe fallback
        audio = load_audio_local_or_gcs(
            batch.get("audio_path", batch.get("audio")),
            sampling_rate=feature_extractor.sampling_rate,
        )
        batch["audio"] = {"array": audio, "sampling_rate": feature_extractor.sampling_rate}

        # Compute input features
        batch["input_features"] = feature_extractor(
            audio, sampling_rate=feature_extractor.sampling_rate
        ).input_features[0]

        # Compute input length
        batch["input_length"] = len(audio) / feature_extractor.sampling_rate

        # Determine language and task
        language, task = resolve_language(
            language_mode=language_mode,
            sample_language=batch.get("language"),
            forced_language=forced_language,
        )

        batch["language"] = language

        # Encode target text to label ids (decoder prompt ids handled at generation)
        labels = encode_labels(
            tokenizer=tokenizer,
            text=batch[text_column_name],
            max_len=max_label_length,
        )

        batch["labels"] = labels

        return batch

    return prepare_dataset


def vectorize_dataset(raw_datasets, prepare_fn, language_mode, valid_languages, num_proc, filter_none=True):
    """
    Applies prepare_fn via Dataset.map() and optionally filters None results.

    Called by:
    - scripts/evaluate.py:run_evaluation()
    - scripts/blacklist.py:create_blacklist()

    Args:
        raw_datasets: HuggingFace Dataset to process
        prepare_fn: Callable returned by make_prepare_dataset_fn()
        language_mode (str): Language mode string
        valid_languages (list): Acceptable language codes
        num_proc (int): Number of parallel workers
        filter_none (bool): Whether to filter out None samples (default True)

    Returns:
        Dataset: Vectorized dataset with columns [input_features, labels, language, id]
    """
    dataset_columns_to_keep = ["input_features", "labels", "language", "id"]
    columns_to_remove = [col for col in list(raw_datasets.features.keys()) if col not in dataset_columns_to_keep]

    vectorized = raw_datasets.map(
        prepare_fn,
        remove_columns=columns_to_remove,
        fn_kwargs={
            "language_mode": language_mode,
            "valid_languages": valid_languages,
        },
        num_proc=num_proc,
    )

    if filter_none:
        vectorized = vectorized.filter(lambda example: example is not None)

    return vectorized


def build_gen_kwargs(profile_config, return_timestamps=False):
    """
    Builds the generation kwargs dict from profile config.

    Called by:
    - scripts/evaluate.py:run_evaluation()
    - scripts/blacklist.py:create_blacklist()

    Args:
        profile_config (dict): Profile configuration.
        return_timestamps (bool): Whether to return timestamps.

    Returns:
        dict: Generation keyword arguments for model.generate().
    """
    return {
        "max_length": int(profile_config["max_label_length"]),
        "return_timestamps": return_timestamps,
        "num_beams": int(profile_config.get("num_beams", 1)),
        "do_sample": False,
        "top_k": 0,
    }


def build_eval_dataloader(vectorized_datasets, batch_size, data_collator, device, num_workers_override=None):
    """
    Constructs a DataLoader with MPS-safe worker count defaults.

    MPS contexts don't fork cleanly, so multiprocessing DataLoader workers
    are disabled on MPS by default to prevent hangs.

    Called by:
    - scripts/evaluate.py:run_evaluation()
    - scripts/blacklist.py:create_blacklist()

    Args:
        vectorized_datasets: Processed HuggingFace Dataset
        batch_size (int): Batch size for the DataLoader
        data_collator: EvalDataCollator instance
        device (torch.device): Target device (used to determine worker defaults)
        num_workers_override (int | None): Explicit worker count; if None uses
            0 for MPS, 4 for other devices.

    Returns:
        DataLoader: Ready-to-iterate DataLoader
    """
    if num_workers_override is not None:
        effective_workers = num_workers_override
    else:
        effective_workers = 0 if device.type == "mps" else 4

    return DataLoader(
        vectorized_datasets,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=effective_workers,
    )


# Module-level cache for per-row metric objects. evaluate.load() performs
# filesystem stat and module import verification on every call; caching avoids
# that overhead across repeated invocations within the same process.
_wer_metric_cache = None
_cer_metric_cache = None


def _get_row_metrics():
    """Return (wer_metric, cer_metric), loading once and caching for the process lifetime."""
    global _wer_metric_cache, _cer_metric_cache
    if _wer_metric_cache is None or _cer_metric_cache is None:
        import evaluate as _evaluate
        _wer_metric_cache = _evaluate.load("wer")
        _cer_metric_cache = _evaluate.load("cer")
    return _wer_metric_cache, _cer_metric_cache


def compute_per_row_metrics(norm_pred_str: list, norm_label_str: list) -> tuple:
    """Compute per-row WER and CER for parallel normalized prediction/label sequences.

    This helper deduplicates the identical per-row metric computation loops that
    exist in both evaluate.py and blacklist.py. Both scripts need per-sample scores
    (evaluate.py for prediction logging, blacklist.py for quality-based filtering).

    Called by:
    - scripts/evaluate.py:run_evaluation() when log_predictions is enabled
    - scripts/blacklist.py:create_blacklist() for per-sample WER/CER scoring

    Calls to:
    - evaluate library (via _get_row_metrics cache) for WER and CER computation

    Args:
        norm_pred_str (list): Normalized prediction strings.
        norm_label_str (list): Normalized reference/label strings.

    Returns:
        tuple[list, list]: (wer_scores, cer_scores) where each entry is a float
            on the 0-100 scale or None when computation fails for that row.
    """
    wer_metric, cer_metric = _get_row_metrics()
    wer_scores: list = []
    cer_scores: list = []
    for i, (pred, ref) in enumerate(zip(norm_pred_str, norm_label_str)):
        try:
            wer_scores.append(100 * wer_metric.compute(predictions=[pred], references=[ref]))
            cer_scores.append(100 * cer_metric.compute(predictions=[pred], references=[ref]))
        except Exception as e:
            logger.debug("compute_per_row_metrics: row %d failed (pred=%r, ref=%r): %s", i, pred, ref, e)
            wer_scores.append(None)
            cer_scores.append(None)
    return wer_scores, cer_scores
