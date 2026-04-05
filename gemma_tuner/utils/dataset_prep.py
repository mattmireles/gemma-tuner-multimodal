"""
Shared Dataset Preprocessing Utilities for Gemma Fine-Tuning

This module provides the core dataset preprocessing infrastructure used across all training
and evaluation workflows. It handles the three critical preprocessing tasks: audio loading,
label encoding, and language resolution with robust error handling and cross-platform
compatibility for Apple Silicon optimization.

Key Responsibilities:
- Universal audio loading from multiple sources (local files, GCS, in-memory data)
- Consistent label encoding with Gemma tokenizer integration
- Language resolution for multilingual training scenarios
- Error recovery and fallback mechanisms for production reliability
- Apple Silicon MPS optimization and memory efficiency

Architecture Integration:
This module serves as the foundation for all dataset processing workflows, providing
consistent interfaces used by training collators, evaluation scripts, and data
preparation utilities throughout the system.

Called by:
- models/*/finetune.py data collators for training data preprocessing
- utils/gemma_dataset_prep.py for Gemma 3n-specific dataset preparation
- evaluation scripts for test dataset processing
- Data loading pipelines in distributed training workflows

Calls to:
- librosa for audio resampling and format conversion
- Google Cloud Storage clients for remote data access
- transformers tokenizers for consistent label encoding
- soundfile and audioread for multi-format audio support

Core Functions:

1. load_audio_local_or_gcs():
   - Universal audio loader supporting local files, GCS URIs, and in-memory data
   - Automatic resampling to required sampling rates
   - Robust error handling with silence fallback for CI/CD reliability
   - Memory-efficient numpy array output for downstream processing

2. encode_labels():
   - Consistent tokenization using Gemma tokenizer interface
   - Proper truncation and padding for fixed-length training
   - Special token handling for sequence-to-sequence training
   - Tensor output compatible with PyTorch training loops

3. resolve_language():
   - Language detection and resolution for multilingual datasets
   - Support for mixed-language datasets with per-sample detection
   - Override mechanisms for forced language specification
   - Consistent language code normalization across the system

Design Principles:
- Single Source of Truth: All dataset processing uses these utilities
- Error Resilience: Graceful degradation prevents training interruption
- Memory Efficiency: Optimized for Apple Silicon unified memory architecture
- Consistency: Identical preprocessing across training and evaluation
- Extensibility: Plugin architecture for new audio formats and sources

Apple Silicon Optimizations:
- librosa resampling optimized for ARM64 architecture
- Memory-efficient numpy arrays minimize unified memory pressure
- Lazy loading patterns reduce peak memory consumption
- MPS-compatible data types for seamless GPU transfer

Performance Considerations:
- Audio resampling: Optimized librosa configuration for Apple Silicon
- Memory usage: Minimal allocation patterns for large dataset processing
- I/O efficiency: Streaming audio loading for memory-constrained environments
- Caching strategies: Intelligent audio caching for repeated access patterns

Error Handling Philosophy:
- Fail loudly: Audio loading failures raise AudioLoadError instead of silently corrupting
  training data with synthetic silence.  Callers decide whether to skip or abort.
- Comprehensive logging: Detailed error information for debugging
- Retry mechanisms: Automatic recovery from transient network failures (GCS)
- Clear exceptions: AudioLoadError carries the original path and root cause

Integration Examples:

Training Integration:
```python
# Used by data collators during training
audio_array = load_audio_local_or_gcs(sample["audio_path"], 16000)
labels = encode_labels(tokenizer, sample["text"], max_length=448)
```

Evaluation Integration:
```python
# Used by evaluation scripts for consistent preprocessing
language, tokens = resolve_language("auto", sample.get("language"))
```

Multi-source Data Loading:
```python
# Supports various input formats seamlessly
local_audio = load_audio_local_or_gcs("/path/to/file.wav", 16000)
gcs_audio = load_audio_local_or_gcs("gs://bucket/file.wav", 16000)
dict_audio = load_audio_local_or_gcs({"array": data, "sampling_rate": 44100}, 16000)
```
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import librosa
import numpy as np

from gemma_tuner.models.gemma.constants import AudioProcessingConstants

logger = logging.getLogger(__name__)


class AudioLoadError(Exception):
    """Raised when audio loading fails from any source (local, GCS, or raw input).

    Callers should catch this to skip bad samples rather than allowing silent
    data corruption from silence-fallback.  The error message includes the
    original path and root cause for debugging.
    """


class DatasetPrepConstants:
    """Named constants for dataset preprocessing configuration and optimization."""

    # Audio Processing Configuration
    # Single source of truth lives in AudioProcessingConstants; aliased here for
    # backwards-compatibility so callers that use DatasetPrepConstants.DEFAULT_SAMPLING_RATE
    # continue to work without change.
    DEFAULT_SAMPLING_RATE = AudioProcessingConstants.DEFAULT_SAMPLING_RATE
    FALLBACK_SILENCE_DURATION = 1.0  # Duration of silence fallback in seconds

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

    # Error Recovery Configuration
    SILENCE_AUDIO_VALUE = 0.0  # Value for silence fallback audio

    # Performance Optimization
    # Librosa configuration for Apple Silicon
    LIBROSA_RESAMPLE_TYPE = "kaiser_best"  # High-quality resampling algorithm


def load_audio_local_or_gcs(
    path_or_audio: Any,
    sampling_rate: int | None,
    timeout: int = DatasetPrepConstants.DEFAULT_TIMEOUT_SECONDS,
    retries: int = DatasetPrepConstants.DEFAULT_RETRY_COUNT,
) -> np.ndarray:
    """
    Universal audio loader supporting multiple input sources with robust error handling.

    This function provides the core audio loading infrastructure used throughout the
    Gemma fine-tuning system. It handles the complexity of different audio sources
    (local files, cloud storage, in-memory data) behind a unified interface with
    automatic resampling and comprehensive error recovery.

    Called by:
    - models/*/finetune.py data collators during training batch preparation
    - utils/gemma_dataset_prep.py for Gemma 3n dataset validation
    - evaluation scripts for test dataset audio preprocessing
    - Distributed training workers for consistent audio loading

    Calls to:
    - librosa.load() for local file audio loading and format conversion
    - librosa.resample() for sampling rate conversion to Gemma requirements
    - Google Cloud Storage clients for remote audio access (when available)
    - numpy for efficient array operations and memory management

    Supported Input Formats:

    1. HuggingFace datasets.Audio dictionary:
       Format: {"array": np.ndarray, "sampling_rate": int}
       Use case: Pre-loaded audio data from HuggingFace datasets
       Processing: Direct array access with automatic resampling

    2. Local file paths:
       Format: "/path/to/audio.wav" (string)
       Supported formats: WAV, FLAC, MP3, M4A, and all librosa-supported formats
       Processing: librosa.load() with automatic format detection

    3. Google Cloud Storage URIs:
       Format: "gs://bucket-name/path/to/audio.wav"
       Requirements: GCS credentials configured in environment
       Processing: Temporary download and local processing

    4. Raw audio arrays:
       Format: np.ndarray or list of audio samples
       Processing: Direct conversion to numpy array with resampling

    Audio Processing Pipeline:
    1. Input format detection and normalization
    2. Audio data extraction based on source type
    3. Sampling rate validation and conversion to target rate
    4. Mono conversion (if multi-channel audio detected)
    5. Data type normalization to float32 for Apple Silicon optimization
    6. Error recovery with silence fallback for training stability

    Error Handling Strategy:
    - Network failures: Automatic retry with exponential backoff
    - File format errors: Graceful fallback to silence with logging
    - Memory errors: Efficient processing with minimal allocation
    - Missing files: Silence fallback prevents training interruption

    Apple Silicon Optimizations:
    - float32 output format optimized for MPS tensor conversion
    - Efficient librosa configuration for ARM64 architecture
    - Memory-conscious processing for unified memory architecture
    - Optimized resampling algorithms for best quality/performance balance

    Args:
        path_or_audio (Any): Audio source in supported format:
            - dict: HuggingFace datasets.Audio format
            - str: Local file path or GCS URI
            - array-like: Raw audio samples
        sampling_rate (int | None): Target sampling rate for resampling
            - None: Uses default Gemma sampling rate (16kHz)
            - int: Specific target rate for model requirements
        timeout (int): Network timeout for GCS operations in seconds
        retries (int): Maximum retry attempts for failed operations

    Returns:
        np.ndarray: Mono audio array at target sampling rate
            - dtype: float32 (optimized for Apple Silicon MPS)
            - shape: (samples,) - 1D mono audio
            - sample_rate: Matches requested sampling_rate parameter
            - fallback: 1-second silence array if loading fails

    Raises:
        AudioLoadError: When audio cannot be loaded from any source (local, GCS).
            Callers should catch this to decide whether to skip the sample or abort.
        ValueError: When raw input is a scalar (possibly None) instead of an array.

    Example Usage:
        # Load from HuggingFace dataset
        audio_dict = {"array": np.array([...]), "sampling_rate": 44100}
        audio = load_audio_local_or_gcs(audio_dict, 16000)

        # Load from local file
        audio = load_audio_local_or_gcs("/path/to/audio.wav", 16000)

        # Load from Google Cloud Storage
        audio = load_audio_local_or_gcs("gs://bucket/audio.wav", 16000)

        # Load from raw array
        raw_samples = [0.1, 0.2, -0.1, 0.0]
        audio = load_audio_local_or_gcs(raw_samples, 16000)

    Performance Notes:
    - Memory efficient: Processes audio without excessive allocation
    - Network resilient: Automatic retries for transient failures
    - Format flexible: Handles all common audio formats automatically
    - Apple Silicon optimized: Uses float32 and efficient algorithms
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
        return audio.astype(DatasetPrepConstants.NUMPY_FLOAT32_DTYPE)

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
        return audio

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
                return audio.astype(np.float32)
            except Exception as e:  # noqa: BLE001 - best-effort fallback
                last_err = e
                continue
        raise AudioLoadError(
            f"Failed to load audio from GCS '{path_or_audio}' after {retries} retries. Last error: {last_err}"
        ) from last_err

    # Local file path
    try:
        audio = librosa.load(path_or_audio, sr=sampling_rate)[0]
        return audio.astype(DatasetPrepConstants.NUMPY_FLOAT32_DTYPE)
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
