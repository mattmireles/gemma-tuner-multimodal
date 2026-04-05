from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


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
        self._warned_prompt_masking: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        from gemma_tuner.models.gemma.constants import GemmaTrainingConstants
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

        self._validate_bos_tokens_present(encoded)

        if "labels" not in encoded:
            labels = encoded.get("input_ids").clone()
            if "attention_mask" in encoded and hasattr(self.processor, "tokenizer"):
                pad_id = self.processor.tokenizer.pad_token_id
                labels[labels == pad_id] = GemmaTrainingConstants.IGNORE_TOKEN_ID

            self._mask_prompt_tokens(labels, encoded["input_ids"])
            encoded["labels"] = labels

        return encoded

    def _mask_prompt_tokens(self, labels: torch.Tensor, input_ids: torch.Tensor) -> None:
        """Mask prompt tokens in labels so loss is computed only on the assistant response."""
        from gemma_tuner.models.gemma.constants import GemmaTrainingConstants

        tokenizer = self.processor.tokenizer

        start_of_turn_id = getattr(tokenizer, "start_of_turn_token_id", None)
        if start_of_turn_id is None:
            start_of_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
            if start_of_turn_id == getattr(tokenizer, "unk_token_id", None):
                start_of_turn_id = None

        if start_of_turn_id is None:
            if not getattr(self, "_warned_prompt_masking", False):
                logger.warning(
                    "DataCollatorGemmaAudio: could not resolve <start_of_turn> token ID. "
                    "Prompt tokens will NOT be masked — this degrades fine-tuning quality."
                )
                self._warned_prompt_masking = True
            return

        model_header_ids = tokenizer.encode("model\n", add_special_tokens=False)
        header_len = len(model_header_ids)

        ignore_id = GemmaTrainingConstants.IGNORE_TOKEN_ID
        for i in range(labels.size(0)):
            sot_positions = (input_ids[i] == start_of_turn_id).nonzero(as_tuple=True)[0]
            if len(sot_positions) >= 2:
                # Normal multi-turn case: mask everything up to (and including the header of)
                # the last <start_of_turn>, so loss is computed only on the assistant response.
                response_start = sot_positions[-1].item() + 1 + header_len
                labels[i, :response_start] = ignore_id
            elif len(sot_positions) == 1:
                # Only one <start_of_turn> found — the prompt/response boundary is ambiguous.
                # Masking up to response_start would likely zero out the only real response
                # content, producing zero loss for this sample.
                # Safer: leave labels unchanged so the sample contributes signal, and warn.
                logger.warning(
                    "DataCollatorGemmaAudio._mask_prompt_tokens: sample %d has only one "
                    "<start_of_turn> token (position %d). Cannot reliably determine prompt/"
                    "response boundary — skipping prompt masking for this sample. "
                    "Check that your dataset produces proper two-turn chat templates.",
                    i,
                    sot_positions[0].item(),
                )
                # Do not mask: leave labels[i] untouched.

    def _validate_bos_tokens_present(self, encoded: Dict[str, torch.Tensor]) -> None:
        """Validate that all sequences contain a <bos> token somewhere for stable Gemma 3n training.

        For multimodal inputs (e.g. Gemma with audio), audio feature tokens legally precede
        the text BOS token, so the first attended token is NOT a BOS token. Checking only
        position 0 would raise a false RuntimeError on every multimodal batch.

        Strategy: use torch.any() to confirm bos_token_id appears anywhere in the sequence.
        If it is absent entirely, that is a real error worth raising. If it exists but not at
        position 0, that is legal for multimodal inputs and is silently accepted.
        """
        from gemma_tuner.models.gemma.constants import GemmaValidationConstants

        if "input_ids" not in encoded or not hasattr(self.processor, "tokenizer"):
            return

        tokenizer = self.processor.tokenizer
        if not hasattr(tokenizer, "bos_token_id") or tokenizer.bos_token_id is None:
            return

        input_ids = encoded["input_ids"]
        bos_missing_samples = []

        for batch_index, sample_token_ids in enumerate(input_ids):
            # Check whether bos_token_id appears anywhere in this sample's token IDs.
            # We intentionally do NOT require it to be at position 0 because multimodal
            # inputs (audio tokens, image tokens, etc.) may legally precede the text BOS.
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

    def _get_sampling_rate(self) -> int:
        """Return sampling rate using hint > processor.sampling_rate > feature_extractor > 16kHz default."""
        from gemma_tuner.models.gemma.constants import AudioProcessingConstants

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
