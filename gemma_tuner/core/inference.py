"""
Unified inference helpers used by evaluation and blacklist flows.

Functions:
- prepare_features(audio_or_path, feature_extractor) -> input_features (1D mel features tensor)
- generate(model, processor, input_features, language_mode, forced_language=None, batch_language=None, gen_kwargs=None) -> token_ids (np.ndarray)
- decode_and_score(tokenizer, predictions, references, normalizer=None) -> (wer, cer, pred_str, label_str, norm_pred_str, norm_label_str)

Notes:
- Keeps language handling consistent across scripts: "mixed" (no forced ids), "strict" (use batch_language), "override:<lang>" (force specific lang).
- Keeps evaluate library usage encapsulated.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import evaluate
import librosa
import numpy as np


def prepare_features(audio_or_path: Any, feature_extractor) -> Any:
    """
    Prepare Gemma input features from an audio path or preloaded array/dict.

    This function handles multiple input formats and prepares audio data for Gemma
    model inference by extracting mel-spectrogram features. It provides a unified
    interface for processing audio from various sources including local files,
    Google Cloud Storage, and in-memory arrays.

    Called by:
    - scripts/evaluate.py during batch evaluation of model predictions
    - models/*/finetune.py for validation during training
    - Testing scripts requiring audio feature extraction

    Calls to:
    - librosa.load() for audio file loading and resampling
    - feature_extractor.__call__() for audio feature extraction
    - google.cloud.storage (optional) for GCS file access

    Input format handling:
    1. Dict with "array" key: Direct from datasets.Audio feature
    2. String path: Local file or GCS URL (gs://)
    3. NumPy array: Raw audio samples

    GCS handling strategy:
    - Best-effort loading with silent fallback for CI/CD environments
    - Falls back to 1 second of silence if GCS access fails
    - Prevents test failures when GCS credentials unavailable

    Args:
        audio_or_path: Audio input in one of three formats:
            - str: Local file path or GCS URL (gs://bucket/path)
            - dict: {"array": audio_samples, "sampling_rate": sr}
            - array-like: 1D audio samples (np.array or list)
        feature_extractor: Audio feature extractor (e.g. from AutoProcessor)
            - Expects sampling_rate attribute (typically 16000 Hz)
            - Returns dict with "input_features" key

    Returns:
        input_features: Model-ready mel-spectrogram features
            - Shape: [n_mel_bins, n_frames] (typically [80, 3000])
            - Single example, suitable for batching later
            - dtype matches feature_extractor output (typically float32)
    """
    sampling_rate = feature_extractor.sampling_rate

    # Already a dict-like from datasets.Audio
    if isinstance(audio_or_path, dict) and "array" in audio_or_path:
        audio = audio_or_path["array"]
    elif isinstance(audio_or_path, str):
        if audio_or_path.startswith("gs://"):
            # Best-effort GCS load with silence fallback to keep CI light
            try:
                from io import BytesIO

                from google.cloud import storage  # type: ignore

                gcs_path = audio_or_path[5:]
                bucket_name, blob_name = gcs_path.split("/", 1)
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                buffer = BytesIO()
                blob.download_to_file(buffer)
                buffer.seek(0)
                audio = librosa.load(buffer, sr=sampling_rate)[0]
            except Exception as e:
                import logging as _logging

                _logging.getLogger(__name__).error(
                    "GCS audio load failed for %s: %s — skipping sample to avoid silent WER corruption",
                    audio_or_path,
                    e,
                )
                raise ValueError(f"GCS audio load failed for {audio_or_path}: {e}") from e
        else:
            audio = librosa.load(audio_or_path, sr=sampling_rate)[0]
    else:
        # Assume ndarray or list-like
        audio = np.asarray(audio_or_path)

    return feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]


def generate(
    model,
    processor,
    input_features,
    language_mode: str,
    forced_language: Optional[str] = None,
    batch_language: Optional[str] = None,
    gen_kwargs: Optional[dict] = None,
):
    """
    Run Gemma generation with consistent language handling across training and evaluation.

    This function provides a unified interface for Gemma model inference with proper
    language handling strategies. It manages the complex interaction between language
    detection, forced language modes, and decoder prompt configuration to ensure
    consistent behavior across different use cases.

    Called by:
    - scripts/evaluate.py for model evaluation with language control
    - scripts/blacklist.py for identifying problematic audio samples
    - models/*/finetune.py during validation steps
    - Testing scripts requiring controlled inference

    Calls to:
    - processor.get_decoder_prompt_ids() for language token generation
    - model.generate() for actual transcription generation (forced_decoder_ids
      passed as kwarg, never mutating model.generation_config)

    Language Mode Strategies:

    1. "mixed" mode:
       - Allows model to auto-detect language per sample
       - No forced decoder tokens
       - Best for multilingual datasets with unknown languages
       - May have higher latency due to language detection

    2. "strict" mode:
       - Uses batch_language parameter for entire batch
       - Enforces language consistency within batch
       - Optimal for homogeneous language batches
       - Falls back to mixed if batch_language is None or "??"

    3. "override:<lang>" mode:
       - Forces specific language for all samples
       - Ignores dataset language metadata
       - Useful for testing or known monolingual scenarios
       - Language specified as ISO 639-1 code (e.g., "en", "es")

    Apple Silicon Considerations:
    - CPU tensor movement after generation for MPS compatibility
    - NumPy conversion prevents MPS memory retention issues
    - Efficient batch processing with unified memory architecture

    Args:
        model: Gemma CausalLM model instance
            - Must be on appropriate device (cuda/mps/cpu)
            - Can be LoRA-adapted or full model
        processor: AutoProcessor with tokenizer and feature extractor
            - Used for decoder prompt generation
            - Handles language token mapping
        input_features: Tensor of mel-spectrogram features
            - Shape: [batch_size, n_mel_bins, n_frames]
            - Must be on same device as model
        language_mode: Language handling strategy
            - "mixed": Auto-detect per sample
            - "strict": Use batch_language parameter
            - "override:<lang>": Force specific language
        forced_language: Optional language for override mode
            - Takes precedence over language_mode parsing
            - ISO 639-1 language code
        batch_language: Language for strict mode (per-batch)
            - Used only when language_mode="strict"
            - None or "??" triggers fallback to mixed
        gen_kwargs: Additional generation parameters
            - max_new_tokens, num_beams, temperature, etc.
            - Forwarded directly to model.generate()

    Returns:
        np.ndarray: Generated token IDs
            - Shape: [batch_size, seq_length]
            - CPU NumPy array for cross-device compatibility
            - Ready for tokenizer decoding
    """
    if gen_kwargs is None:
        gen_kwargs = {}

    # Determine forced_decoder_ids per language mode and pass as a kwarg
    # instead of mutating model.generation_config (avoids shared-state side effects).
    if language_mode == "mixed":
        forced_ids = None
    elif language_mode == "strict":
        if batch_language and batch_language != "??":
            forced_ids = processor.get_decoder_prompt_ids(language=batch_language, task="transcribe")
        else:
            forced_ids = None
    elif language_mode.startswith("override:") or forced_language:
        lang = forced_language or language_mode.split(":", 1)[1]
        forced_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
    else:
        raise ValueError(f"Invalid language mode: {language_mode}")

    outputs = model.generate(
        input_features=input_features,
        forced_decoder_ids=forced_ids,
        **gen_kwargs,
    )
    return outputs.cpu().numpy()


def decode_and_score(
    predictions: Sequence[Any],
    references: Sequence[Any],
    normalizer: Optional[Any] = None,
):
    """
    Decode predictions and references, normalize text, and compute WER/CER metrics.

    This function implements the complete evaluation pipeline for ASR model outputs,
    handling text normalization and metric computation. It provides consistent
    scoring across different evaluation contexts with proper handling of edge cases
    and empty sequences.

    Called by:
    - scripts/evaluate.py for batch evaluation scoring
    - scripts/blacklist.py for identifying high-error samples
    - models/*/finetune.py compute_metrics() during training validation
    - Testing scripts for accuracy verification

    Calls to:
    - EnglishTextNormalizer for text standardization
    - evaluate.load("wer") for Word Error Rate computation
    - evaluate.load("cer") for Character Error Rate computation

    Metric Computation:

    WER (Word Error Rate):
    - Measures word-level transcription accuracy
    - Formula: (Substitutions + Deletions + Insertions) / Total_Words * 100
    - Industry standard for ASR evaluation
    - Lower is better (0% = perfect, >100% possible)

    CER (Character Error Rate):
    - Measures character-level transcription accuracy
    - More granular than WER, useful for morphologically rich languages
    - Formula: (Char_Substitutions + Char_Deletions + Char_Insertions) / Total_Chars * 100
    - Generally higher than WER for English

    Text Normalization Strategy:
    - Removes punctuation and special characters
    - Converts to lowercase for case-insensitive comparison
    - Standardizes whitespace and word boundaries
    - Handles common transcription variations (e.g., "don't" vs "dont")
    - Reduces false penalties for stylistic differences

    Edge Case Handling:
    - Filters empty predictions/references to avoid division errors
    - Returns None metrics if no valid pairs exist
    - Preserves original string lists for debugging
    - Maintains index alignment for downstream processing

    Args:
        predictions: Sequence of prediction strings
            - Raw model outputs after decoding
            - Can be list, tuple, or any sequence type
        references: Sequence of reference strings
            - Ground truth transcriptions
            - Must align index-wise with predictions
        normalizer: Optional text normalization callable
            - Defaults to EnglishTextNormalizer if None
            - Should accept string and return normalized string
            - Custom normalizers for non-English languages supported

    Returns:
        Tuple[Optional[float], Optional[float], List[str], List[str], List[str], List[str]]:
            - wer: Word Error Rate percentage (0-100) or None if no valid pairs
            - cer: Character Error Rate percentage (0-100) or None if no valid pairs
            - pred_str: Original prediction strings (preserved)
            - label_str: Original reference strings (preserved)
            - norm_pred_str: Normalized predictions (index-aligned; "" for empty pairs)
            - norm_label_str: Normalized references (index-aligned; "" for empty pairs)

    Performance Considerations:
    - Late import of normalizer reduces startup time
    - Batch normalization for efficiency
    - Metric libraries cached after first load
    - Filtering step prevents unnecessary computation
    """
    pred_str = [str(p) for p in predictions]
    label_str = [str(l) for l in references]

    if normalizer is None:
        # Late import to avoid heavy deps at import time
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer  # type: ignore

        normalizer = EnglishTextNormalizer()

    # Build index-aligned normalized lists: empty string placeholders for
    # filtered-out pairs so callers can index norm_*_str with the same `i`
    # they use for pred_str / label_str without IndexError.
    norm_pred_str = []
    norm_label_str = []
    for p, l in zip(pred_str, label_str):
        if p and l:
            norm_pred_str.append(normalizer(p))
            norm_label_str.append(normalizer(l))
        else:
            norm_pred_str.append("")
            norm_label_str.append("")

    # Collect only the non-empty pairs for WER/CER computation
    score_preds = [np_ for np_, nl_ in zip(norm_pred_str, norm_label_str) if np_ and nl_]
    score_refs = [nl_ for np_, nl_ in zip(norm_pred_str, norm_label_str) if np_ and nl_]
    if not score_preds:
        return None, None, pred_str, label_str, norm_pred_str, norm_label_str

    # Prefer local files only to keep tests offline; fall back gracefully
    def _levenshtein(a: Sequence, b: Sequence) -> int:
        """Generic Levenshtein distance (works on strings or token lists)."""
        la, lb = len(a), len(b)
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[la][lb]

    def _cer_fallback(preds: list[str], refs: list[str]) -> float:
        if not preds:
            return 0.0
        total_dist = sum(_levenshtein(p, r) for p, r in zip(preds, refs))
        total_chars = sum(max(1, len(r)) for r in refs)
        return total_dist / total_chars

    def _wer_fallback(preds: list[str], refs: list[str]) -> float:
        if not preds:
            return 0.0
        total_dist = sum(_levenshtein(p.split(), r.split()) for p, r in zip(preds, refs))
        total_words = sum(max(1, len(r.split())) for r in refs)
        return total_dist / total_words

    try:
        # Try local-only first (offline/CI), then allow network download
        try:
            wer_metric = evaluate.load("wer", download_config=evaluate.DownloadConfig(local_files_only=True))
            cer_metric = evaluate.load("cer", download_config=evaluate.DownloadConfig(local_files_only=True))
        except Exception:
            wer_metric = evaluate.load("wer")
            cer_metric = evaluate.load("cer")
        wer = 100 * wer_metric.compute(predictions=score_preds, references=score_refs)
        cer = 100 * cer_metric.compute(predictions=score_preds, references=score_refs)
    except Exception:
        # Offline or HF unavailable: use internal fallbacks sufficient for tests
        wer = 100 * _wer_fallback(score_preds, score_refs)
        cer = 100 * _cer_fallback(score_preds, score_refs)

    return wer, cer, list(pred_str), list(label_str), norm_pred_str, norm_label_str
