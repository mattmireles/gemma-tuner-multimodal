from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from gemma_tuner.models.gemma.constants import (
    AudioProcessingConstants,
    GemmaTrainingConstants,
    GemmaValidationConstants,
)

logger = logging.getLogger(__name__)


def validate_bos_tokens_present(encoded: Dict[str, torch.Tensor], tokenizer) -> None:
    """Confirm each sequence contains ``bos_token_id`` somewhere (multimodal-safe).

    For multimodal inputs, tokens before the text BOS are legal; we only error if
    BOS is missing entirely. Shared by audio and text collators.
    """
    if "input_ids" not in encoded or tokenizer is None or not hasattr(tokenizer, "bos_token_id"):
        return

    if tokenizer.bos_token_id is None:
        return

    input_ids = encoded["input_ids"]
    bos_missing_samples = []

    for batch_index, sample_token_ids in enumerate(input_ids):
        bos_present = torch.any(sample_token_ids == tokenizer.bos_token_id).item()
        if not bos_present:
            bos_missing_samples.append(batch_index)

    if bos_missing_samples:
        max_display = GemmaValidationConstants.MAX_DISPLAYED_ERROR_SAMPLES
        displayed = bos_missing_samples[:max_display]
        ellipsis = "..." if len(bos_missing_samples) > max_display else ""
        raise RuntimeError(
            f"CRITICAL: <bos> token absent entirely from {len(bos_missing_samples)} samples "
            f"(batch indices: {displayed}{ellipsis}). "
            f"Gemma 3n requires a <bos> token somewhere in each sequence for stable training. "
            f"For multimodal inputs, audio/image tokens may precede <bos> — that is fine. "
            f"Expected token ID: {tokenizer.bos_token_id}."
        )


def mask_gemma_prompt_tokens(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    warned_prompt_masking: List[bool],
) -> None:
    """Mask prompt tokens in labels so loss is computed only on the assistant response."""
    start_of_turn_id = getattr(tokenizer, "start_of_turn_token_id", None)
    if start_of_turn_id is None:
        start_of_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        if start_of_turn_id == getattr(tokenizer, "unk_token_id", None):
            start_of_turn_id = None

    if start_of_turn_id is None:
        if not warned_prompt_masking[0]:
            logger.warning(
                "mask_gemma_prompt_tokens: could not resolve <start_of_turn> token ID. "
                "Prompt tokens will NOT be masked — this degrades fine-tuning quality."
            )
            warned_prompt_masking[0] = True
        return

    model_header_ids = tokenizer.encode("model\n", add_special_tokens=False)
    header_len = len(model_header_ids)

    ignore_id = GemmaTrainingConstants.IGNORE_TOKEN_ID
    for i in range(labels.size(0)):
        sot_positions = (input_ids[i] == start_of_turn_id).nonzero(as_tuple=True)[0]
        if len(sot_positions) >= 2:
            response_start = sot_positions[-1].item() + 1 + header_len
            labels[i, :response_start] = ignore_id
        elif len(sot_positions) == 1:
            logger.warning(
                "mask_gemma_prompt_tokens: sample %d has only one "
                "<start_of_turn> token (position %d). Cannot reliably determine prompt/"
                "response boundary — skipping prompt masking for this sample. "
                "Check that your dataset produces proper two-turn chat templates.",
                i,
                sot_positions[0].item(),
            )


class DataCollatorGemmaAudio:
    """Data collator that packs audio+text into Gemma inputs via AutoProcessor.

    Cross-file connections:
    - Consumes rows loaded by `utils.dataset_utils.load_dataset_split()` which must
      include: `id`, `audio_path`, and a text column configured by profile.
    - Delegates audio feature extraction and text tokenization to AutoProcessor to
      ensure exact replication of Gemma 3n preprocessing (USM audio tower).

    Returns dicts compatible with Gemma 3n CausalLM forward(). Exact key names are
    determined by the model processor (e.g., `input_ids`, `attention_mask`, and one
    of `audio_values`/`input_features` plus any multimodal masks).
    """

    def __init__(self, processor, text_column: str, sampling_rate_hint: Optional[int] = None):
        self.processor = processor
        self.text_column = text_column
        self.sampling_rate_hint = sampling_rate_hint
        self._warned_prompt_masking: List[bool] = [False]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        from gemma_tuner.utils.dataset_prep import load_audio_local_or_gcs

        audios: List[List[float]] = []
        texts: List[str] = []

        sampling_rate = self._get_sampling_rate()

        for ex in features:
            audio_path = ex.get("audio_path", ex.get("audio"))
            if audio_path is None:
                raise KeyError(
                    f"DataCollatorGemmaAudio: no audio path found in sample. "
                    f"Expected 'audio_path' or 'audio' key. Available keys: {list(ex.keys())}"
                )
            audio = load_audio_local_or_gcs(audio_path, sampling_rate=sampling_rate)
            text = ex.get(self.text_column)
            if text is None:
                raise KeyError(
                    f"DataCollatorGemmaAudio: text column '{self.text_column}' missing from sample. "
                    f"Available keys: {list(ex.keys())}"
                )
            audios.append(audio)
            texts.append(text)

        messages_batch = []
        for t in texts:
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "<audio:attached>"},
                            {"type": "text", "text": "Please transcribe this audio."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": t}]},
                ]
            )

        try:
            encoded = self.processor(
                messages=messages_batch,
                audios=audios,
                return_tensors="pt",
                padding=True,
            )
        except TypeError as e:
            raise RuntimeError(
                f"Gemma 3n processor does not support messages interface: {e}. "
                f"This is required for proper chat templating with <bos>, <start_of_turn>, <end_of_turn> tokens. "
                f"Ensure you're using a compatible transformers version (>=4.38.2) and processor."
            ) from e

        if hasattr(self.processor, "tokenizer"):
            validate_bos_tokens_present(encoded, self.processor.tokenizer)

        if "labels" not in encoded:
            labels = encoded.get("input_ids").clone()
            if "attention_mask" in encoded and hasattr(self.processor, "tokenizer"):
                pad_id = self.processor.tokenizer.pad_token_id
                labels[labels == pad_id] = GemmaTrainingConstants.IGNORE_TOKEN_ID

            mask_gemma_prompt_tokens(labels, encoded["input_ids"], self.processor.tokenizer, self._warned_prompt_masking)
            encoded["labels"] = labels

        return encoded

    def _get_sampling_rate(self) -> int:
        """Return sampling rate using hint > processor.sampling_rate > feature_extractor > 16kHz default."""
        if self.sampling_rate_hint is not None:
            return self.sampling_rate_hint
        if hasattr(self.processor, "sampling_rate") and self.processor.sampling_rate is not None:
            return self.processor.sampling_rate
        if (
            hasattr(self.processor, "feature_extractor")
            and hasattr(self.processor.feature_extractor, "sampling_rate")
            and self.processor.feature_extractor.sampling_rate is not None
        ):
            return self.processor.feature_extractor.sampling_rate
        return AudioProcessingConstants.DEFAULT_SAMPLING_RATE
