"""
Metrics Computation System for Gemma Model Evaluation

This module provides the single, canonical metrics computation used by ALL
Gemma training backends (standard, LoRA, and distil). Every backend
imports ``build_wer_metrics`` from here instead of defining its own inline
``compute_metrics`` closure.

Key responsibilities:
- WER (Word Error Rate) computation for word-level accuracy
- CER (Character Error Rate) computation for character-level accuracy (optional)
- Text normalization for consistent evaluation (optional)
- Safe handling of 3-D logit tensors and tuple predictions
- Padding token handling for proper decoding
- Offline-safe metric loading with stub fallback

Called by:
- models.gemma.finetune_core/trainer.py:build_trainer() for standard training
- models.gemma.finetune.py:finetune() for distillation evaluation
- models.gemma.finetune.py:main() for LoRA evaluation

Metric definitions:
- WER: (Substitutions + Deletions + Insertions) / Total Words * 100
- CER: (Substitutions + Deletions + Insertions) / Total Characters * 100

Lower scores are better (0% = perfect, 100% = completely wrong).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class MetricConstants:
    """Named constants for metric computation."""

    # Padding token value used in labels (-100 is the HuggingFace convention,
    # ignored by CrossEntropyLoss and must be replaced before decoding).
    LABEL_PADDING_TOKEN = -100

    # Metric scaling factor (convert to percentage)
    PERCENTAGE_SCALE = 100

    # Metric names for evaluation
    WER_METRIC_NAME = "wer"
    CER_METRIC_NAME = "cer"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_wer_metrics(
    tokenizer: Any,
    normalizer: Optional[Any] = None,
    decoder: Optional[Any] = None,
    include_cer: bool = False,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    """
    Build WER/CER metric objects and a ``compute_fn`` closure for HuggingFace Trainer.

    This is the **single source of truth** for evaluation-metric wiring across
    all three Gemma training backends (standard, LoRA, distil).

    Args:
        tokenizer:
            A tokenizer (or processor) whose ``pad_token_id`` is used to
            replace -100 padding tokens before decoding.  Required.
        normalizer:
            Optional text normalizer (e.g. ``BasicTextNormalizer()``).  When
            provided, the returned ``compute_fn`` will return *normalized* WER
            (and, if ``include_cer``, normalized CER).  When ``None``, raw
            (un-normalized) WER is returned.
        decoder:
            The object whose ``.batch_decode()`` is called.  Defaults to
            ``tokenizer`` itself.  Pass a ``processor`` here when the backend
            uses ``processor.batch_decode`` (standard and distil
            backends do this).
        include_cer:
            If ``True``, CER is computed alongside WER.  Default ``False``.
        local_files_only:
            If ``True``, ``evaluate.load`` will not attempt network downloads.
            Useful for offline / CI environments.

    Returns:
        Dict with keys:
            - ``wer_metric``: The WER metric object (or a stub if loading failed).
            - ``cer_metric``: The CER metric object (only present when ``include_cer``; may be ``None``).
            - ``compute_fn``: A callable with signature ``(EvalPrediction) -> dict``
              suitable for ``Seq2SeqTrainer(compute_metrics=...)``.

    The compute_fn handles:
        - Tuple predictions (extracts first element)
        - 3-D logit tensors (argmax to get token IDs)
        - Label padding replacement (-100 -> pad_token_id)
        - Labels are copied before mutation (safe for repeated evaluation)
        - Optional text normalization
        - WER always returned; CER returned when ``include_cer=True``

    Example (standard backend)::

        from gemma_tuner.models.common.metrics import build_wer_metrics
        metrics = build_wer_metrics(tokenizer=tokenizer, decoder=processor)
        trainer = Seq2SeqTrainer(compute_metrics=metrics["compute_fn"], ...)

    Example (LoRA backend with normalization)::

        from gemma_tuner.models.common.metrics import build_wer_metrics
        normalizer = BasicTextNormalizer()
        metrics = build_wer_metrics(
            tokenizer=tokenizer, normalizer=normalizer, include_cer=False,
        )
        trainer = Seq2SeqTrainer(compute_metrics=metrics["compute_fn"], ...)
    """

    # --- load metrics (offline-safe) ---
    wer_metric = _load_metric(
        MetricConstants.WER_METRIC_NAME, local_files_only=local_files_only
    )
    cer_metric: Any = None
    if include_cer:
        cer_metric = _load_metric(
            MetricConstants.CER_METRIC_NAME, local_files_only=local_files_only
        )

    # Decoder defaults to the tokenizer when not explicitly provided.
    _decoder = decoder if decoder is not None else tokenizer

    def compute_fn(pred) -> Dict[str, float]:
        """
        Compute WER (and optionally CER) from an ``EvalPrediction``.

        Called automatically by HuggingFace ``Trainer`` at the end of each
        evaluation loop.  Handles tuple outputs, 3-D logit tensors, and
        -100 label padding.

        Args:
            pred: ``EvalPrediction`` with ``.predictions`` and ``.label_ids``.

        Returns:
            Dict mapping metric names to float values (percentages for
            normalized metrics, raw ratios otherwise).
        """
        predictions = pred.predictions
        label_ids = pred.label_ids

        # --- Unpack tuple outputs (some HF models return (logits, ...)) ---
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # --- Convert 3-D logit tensor [B, T, V] to token IDs [B, T] ---
        if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
            pred_ids = np.argmax(predictions, axis=-1)
        else:
            pred_ids = predictions

        # --- Replace -100 padding with pad_token_id for decoding ---
        # Copy labels so the original array is not mutated across eval calls.
        label_ids = label_ids.copy()
        label_ids[label_ids == MetricConstants.LABEL_PADDING_TOKEN] = (
            tokenizer.pad_token_id
        )

        # --- Decode to strings ---
        pred_str = _decoder.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = _decoder.batch_decode(label_ids, skip_special_tokens=True)

        # --- Optional normalisation ---
        if normalizer is not None:
            pred_str = [normalizer(p).strip() for p in pred_str]
            label_str = [normalizer(r).strip() for r in label_str]

        # --- Compute WER ---
        wer_value = wer_metric.compute(
            predictions=pred_str, references=label_str
        )
        results: Dict[str, float] = {
            MetricConstants.WER_METRIC_NAME: float(
                MetricConstants.PERCENTAGE_SCALE * wer_value
            )
        }

        # --- Optionally compute CER ---
        if cer_metric is not None:
            cer_value = cer_metric.compute(
                predictions=pred_str, references=label_str
            )
            results[MetricConstants.CER_METRIC_NAME] = float(
                MetricConstants.PERCENTAGE_SCALE * cer_value
            )

        return results

    return {
        "wer_metric": wer_metric,
        "cer_metric": cer_metric,
        "compute_fn": compute_fn,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_metric(name: str, local_files_only: bool = False) -> Any:
    """
    Load an ``evaluate`` metric, falling back to a zero-stub on failure.

    The stub ensures training can proceed in offline / CI environments where
    the metric script is unavailable.

    Args:
        name: Metric name (e.g. ``"wer"`` or ``"cer"``).
        local_files_only: If ``True``, disallow network downloads.

    Returns:
        A metric object with a ``.compute(predictions=, references=)`` method.
    """
    import evaluate

    try:
        download_cfg = evaluate.DownloadConfig(local_files_only=local_files_only)
        return evaluate.load(name, download_config=download_cfg)
    except Exception:

        class _StubMetric:
            """Returns 0.0 when the real metric is unavailable."""

            def compute(self, predictions, references):
                return 0.0

        return _StubMetric()
