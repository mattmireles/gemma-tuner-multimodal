"""Shared dataset helpers: audio loading, label encoding, language resolution.

Audio: ``load_audio_local_or_gcs`` resamples to the requested rate, clips float
waveforms to [-1, 1], and raises :class:`AudioLoadError` on load failure (no
silent synthetic silence).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import librosa
import numpy as np

from gemma_tuner.models.gemma.constants import AudioProcessingConstants
from gemma_tuner.utils.safe_io import validate_safe_path

logger = logging.getLogger(__name__)


class AudioLoadError(Exception):
    """Raised when audio loading fails from a path or GCS."""


class DatasetPrepConstants:
    """Named constants for dataset preprocessing configuration and optimization."""

    # Audio Processing Configuration
    # Single source of truth lives in AudioProcessingConstants; aliased here for
    # backwards-compatibility so callers that use DatasetPrepConstants.DEFAULT_SAMPLING_RATE
    # continue to work without change.
    DEFAULT_SAMPLING_RATE = AudioProcessingConstants.DEFAULT_SAMPLING_RATE

    # Audio Loading Configuration
    DEFAULT_TIMEOUT_SECONDS = 10  # Network timeout for GCS audio loading
    DEFAULT_RETRY_COUNT = 2  # Number of retries for failed audio loading

    # Google Cloud Storage Configuration
    GCS_URI_PREFIX = "gs://"  # GCS URI prefix for bucket detection
    GCS_BLOB_DELIMITER = "/"  # Path delimiter for GCS blob names

    # Audio Format Detection
    # Supported input data structures for universal audio loading
    AUDIO_DICT_ARRAY_KEY = "array"  # Key for audio data in dataset dictionary
    AUDIO_DICT_RATE_KEY = "sampling_rate"  # Key for sampling rate in dataset dictionary

    # Memory Optimization
    NUMPY_FLOAT32_DTYPE = np.float32  # Optimal dtype for Apple Silicon processing

    # Label Encoding Configuration
    # Gemma tokenizer integration constants
    MAX_TOKEN_LENGTH_DEFAULT = 448  # Standard Gemma sequence length
    LABEL_PADDING_VALUE = -100  # Standard padding value for loss calculation

    # Language Resolution
    # Supported language modes for multilingual processing
    LANGUAGE_MODE_AUTO = "auto"  # Automatic language detection
    LANGUAGE_MODE_MIXED = "mixed"  # Mixed-language dataset support
    LANGUAGE_MODE_STRICT = "strict"  # Single language enforcement

    # Performance Optimization
    # Librosa configuration for Apple Silicon
    LIBROSA_RESAMPLE_TYPE = "kaiser_best"  # High-quality resampling algorithm


def _clip_audio_float32(waveform: np.ndarray) -> np.ndarray:
    """Clamp float samples to [-1.0, 1.0] before the audio feature extractor (see gemma4-guide.md)."""
    a = np.asarray(waveform, dtype=np.float32)
    return np.clip(a, -1.0, 1.0)


def load_audio_local_or_gcs(
    path_or_audio: Any,
    sampling_rate: int | None,
    timeout: int = DatasetPrepConstants.DEFAULT_TIMEOUT_SECONDS,
    retries: int = DatasetPrepConstants.DEFAULT_RETRY_COUNT,
) -> np.ndarray:
    """Load mono float32 audio; resample to ``sampling_rate`` (default 16 kHz); clip to [-1, 1].

    Accepts: HuggingFace ``datasets.Audio`` dict (``array`` + ``sampling_rate``), local path,
    ``gs://`` URI (retries on failure), or a raw 1-D array (already at target rate; no resample).

    Raises:
        AudioLoadError: Failed to load from a path or GCS after retries.
        ValueError: Raw input is a scalar instead of a 1-D array.
    """
    # Normalize sampling rate to a safe default using named constants
    if sampling_rate is None:  # use `is None`, not falsy check — 0 is a valid explicit value
        sampling_rate = DatasetPrepConstants.DEFAULT_SAMPLING_RATE

    # Already datasets.Audio-style dict
    if isinstance(path_or_audio, dict) and DatasetPrepConstants.AUDIO_DICT_ARRAY_KEY in path_or_audio:
        audio = np.asarray(
            path_or_audio[DatasetPrepConstants.AUDIO_DICT_ARRAY_KEY], dtype=DatasetPrepConstants.NUMPY_FLOAT32_DTYPE
        )
        src_sr = int(path_or_audio.get(DatasetPrepConstants.AUDIO_DICT_RATE_KEY, sampling_rate))
        if src_sr != sampling_rate:
            audio = librosa.resample(
                audio, orig_sr=src_sr, target_sr=sampling_rate, res_type=DatasetPrepConstants.LIBROSA_RESAMPLE_TYPE
            )
        return _clip_audio_float32(audio)

    # Raw array-like — assumes audio is already at the target sampling_rate.
    # Callers passing raw arrays must ensure the rate matches; no resampling
    # is possible without knowing the source rate.
    if not isinstance(path_or_audio, str):
        audio = np.asarray(path_or_audio, dtype=np.float32)
        if audio.ndim == 0:
            raise ValueError(
                "load_audio_local_or_gcs received a scalar (possibly None) as raw audio. "
                "Expected a 1-D array or a string path."
            )
        # Warn callers that the sampling_rate parameter is silently ignored for raw arrays.
        # There is no source-rate metadata, so resampling cannot be applied.  The caller
        # is responsible for pre-resampling the array to the correct rate before passing it.
        logger.warning(
            "load_audio_local_or_gcs received a raw audio array; sampling rate "
            "conversion is not applied. Ensure the array is already at %d Hz.",
            sampling_rate,
        )
        return _clip_audio_float32(audio)

    # String path or GCS URI
    if path_or_audio.startswith(DatasetPrepConstants.GCS_URI_PREFIX):
        import time
        from io import BytesIO

        from google.cloud import storage  # type: ignore

        # Instantiate the GCS client once before the retry loop.
        # Creating storage.Client() performs credential negotiation; doing it
        # inside the loop would pay that cost on every retry, which is wasteful
        # and can cause thundering-herd problems against the metadata server.
        gcs_client = storage.Client()

        gcs_path = path_or_audio[len(DatasetPrepConstants.GCS_URI_PREFIX) :]
        bucket_name, blob_name = gcs_path.split(DatasetPrepConstants.GCS_BLOB_DELIMITER, 1)

        last_err: Optional[Exception] = None
        for attempt in range(max(1, retries + 1)):
            # Exponential backoff between retries (skip sleep on the first attempt).
            # Sleeping 2**attempt seconds: attempt 1 → 2 s, attempt 2 → 4 s, etc.
            if attempt > 0:
                time.sleep(2**attempt)
            try:
                bucket = gcs_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                buffer = BytesIO()
                # No direct timeout in SDK; rely on short blobs + retries
                blob.download_to_file(buffer)
                buffer.seek(0)
                audio = librosa.load(buffer, sr=sampling_rate)[0]
                return _clip_audio_float32(audio)
            except Exception as e:  # noqa: BLE001 - best-effort fallback
                last_err = e
                continue
        raise AudioLoadError(
            f"Failed to load audio from GCS '{path_or_audio}' after {retries} retries. Last error: {last_err}"
        ) from last_err

    # Local file path
    try:
        # Relative paths are confined to the current working directory, which
        # preserves the repo's prepared-CSV convention while still rejecting
        # traversal out of that tree. Absolute paths remain supported for
        # documented workflows such as Granary manifests.
        safe_path = validate_safe_path(
            path_or_audio,
            base_dir=str(Path.cwd()) if not Path(path_or_audio).is_absolute() else None,
            allow_symlinks=False,
        )
        audio = librosa.load(str(safe_path), sr=sampling_rate)[0]
        return _clip_audio_float32(audio)
    except ValueError as e:
        raise AudioLoadError(f"Invalid audio path '{path_or_audio}': {e}") from e
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio from local path '{path_or_audio}': {e}") from e


def encode_labels(tokenizer, text: str, max_len: int = DatasetPrepConstants.MAX_TOKEN_LENGTH_DEFAULT):
    """
    Standard tokenization and padding for Gemma sequence-to-sequence training.

    This function provides consistent label encoding used across all Gemma training
    workflows. It handles tokenization, length management, and padding using the
    standard Gemma tokenizer interface with proper special token handling.

    Called by:
    - models/*/finetune.py data collators during training batch preparation
    - Evaluation scripts for consistent label preprocessing
    - Dataset validation utilities for tokenization testing

    Calls to:
    - tokenizer.encode() for text-to-token conversion
    - PyTorch tensor operations for padding and truncation

    Tokenization Process:
    1. Text tokenization using Gemma tokenizer with special tokens
    2. Length validation and truncation to maximum sequence length
    3. Padding to fixed length for efficient batch processing
    4. Conversion to PyTorch tensor with appropriate loss masking

    Args:
        tokenizer: Gemma tokenizer instance with encode() method
        text (str): Input transcription text for tokenization
        max_len (int): Maximum sequence length for truncation and padding

    Returns:
        torch.LongTensor: Encoded and padded token sequence
            - Length: max_len (padded or truncated as needed)
            - Padding value: Tokenizer-specific (usually -100 for loss masking)
            - dtype: long (required for embedding lookup)

    Example:
        tokens = encode_labels(tokenizer, "Hello world", 448)
        # Returns: tensor([50258, 23280, 1002, -100, -100, ...])  # length 448
    """
    return tokenizer(
        text,
        max_length=int(max_len),
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).input_ids[0]


def resolve_language(
    language_mode: str,
    sample_language: Optional[str],
    forced_language: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Resolves language and task configuration for multilingual Gemma training.

    This function implements the language resolution logic used across all
    training workflows to handle multilingual datasets, language-specific training,
    and mixed-language scenarios with consistent behavior.

    Called by:
    - models/*/finetune.py data collators for per-sample language resolution
    - Evaluation scripts for consistent language handling
    - Multilingual training workflows requiring language specification

    Language Resolution Modes:

    1. Mixed Mode ("mixed"):
       - Purpose: Mixed-language datasets with automatic language detection
       - Behavior: Returns None for language (auto-detect) with transcribe task
       - Use case: Large multilingual datasets where language varies per sample
       - Model behavior: Gemma automatically detects language during inference

    2. Strict Mode ("strict"):
       - Purpose: Single-language datasets with explicit language specification
       - Behavior: Uses sample-level language metadata for training
       - Use case: Language-specific fine-tuning or evaluation
       - Model behavior: Forces specific language token in training

    3. Override Mode ("override:<lang>" or forced_language):
       - Purpose: Force specific language regardless of sample metadata
       - Behavior: Overrides all sample languages with specified language
       - Use case: Converting multilingual data to single-language training
       - Model behavior: Trains model to expect specific language only

    Args:
        language_mode (str): Language resolution strategy:
            - "mixed": Auto-detect language per sample
            - "strict": Use sample-provided language metadata
            - "override:<lang>": Force specific language (e.g., "override:en")
        sample_language (Optional[str]): Language code from sample metadata
        forced_language (Optional[str]): Override language (takes precedence)

    Returns:
        Tuple[Optional[str], str]: (language_code, task)
            - language_code: None for auto-detect, string for specific language
            - task: Always "transcribe" (Gemma task specification)

    Raises:
        ValueError: If language_mode is not recognized

    Examples:
        # Mixed-language dataset (auto-detect)
        lang, task = resolve_language("mixed", "en")
        # Returns: (None, "transcribe")

        # Single-language dataset
        lang, task = resolve_language("strict", "en")
        # Returns: ("en", "transcribe")

        # Force English regardless of sample language
        lang, task = resolve_language("override:en", "fr")
        # Returns: ("en", "transcribe")

        # Force language via parameter
        lang, task = resolve_language("mixed", "en", forced_language="es")
        # Returns: ("es", "transcribe")

    Integration Notes:
    - Language codes follow ISO 639-1 standard (e.g., "en", "fr", "es")
    - Task is always "transcribe" (Gemma doesn't support other tasks in training)
    - None language enables automatic language detection
    - Consistent behavior across all training and evaluation workflows
    """
    # Explicit forced_language always takes priority regardless of mode
    if forced_language:
        return forced_language, "transcribe"
    if language_mode in (DatasetPrepConstants.LANGUAGE_MODE_AUTO, DatasetPrepConstants.LANGUAGE_MODE_MIXED):
        return None, "transcribe"
    if language_mode == DatasetPrepConstants.LANGUAGE_MODE_STRICT:
        return sample_language, "transcribe"
    if language_mode.startswith("override:"):
        lang = language_mode.split(":", 1)[1]
        return lang, "transcribe"
    raise ValueError(
        f"Invalid language mode: {language_mode}. "
        f"Supported modes: {DatasetPrepConstants.LANGUAGE_MODE_MIXED}, "
        f"{DatasetPrepConstants.LANGUAGE_MODE_STRICT}, override:<lang>"
    )
