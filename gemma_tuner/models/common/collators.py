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


def _find_subsequence_ids(haystack: torch.Tensor, needle: List[int]) -> int:
    """Index of first position in ``haystack`` where ``needle`` matches; ``-1`` if absent."""
    if not needle:
        return 0
    h = haystack.tolist()
    n, m = len(h), len(needle)
    if m > n:
        return -1
    for start in range(0, n - m + 1):
        if h[start : start + m] == needle:
            return start
    return -1


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
    """Mask prompt tokens in labels so loss is computed only on the assistant response.

    Prefer ``return_assistant_tokens_mask=True`` on ``apply_chat_template`` when the
    tokenizer supports it (see :class:`DataCollatorGemmaText`). This fallback locates the
    assistant span after the **last** ``<start_of_turn>`` by searching for the tokenized
    ``model`` role header as a **subsequence** (not a fixed offset), so extra turn markers
    inside the user turn do not break masking.
    """
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

    ignore_id = GemmaTrainingConstants.IGNORE_TOKEN_ID
    for i in range(labels.size(0)):
        sot_positions = (input_ids[i] == start_of_turn_id).nonzero(as_tuple=True)[0]
        if len(sot_positions) == 0:
            if not warned_prompt_masking[0]:
                logger.warning(
                    "mask_gemma_prompt_tokens: at least one sample has no <start_of_turn> tokens; "
                    "skipping prompt masking for those rows."
                )
                warned_prompt_masking[0] = True
            continue

        last_sot = int(sot_positions[-1].item())
        after = input_ids[i, last_sot + 1 :]
        response_start: Optional[int] = None
        for header_text in ("model\n", "model"):
            try:
                needle = tokenizer.encode(header_text, add_special_tokens=False)
            except Exception:
                continue
            if not needle:
                continue
            rel = _find_subsequence_ids(after, needle)
            if rel >= 0:
                response_start = last_sot + 1 + rel + len(needle)
                break

        if response_start is None:
            if not warned_prompt_masking[0]:
                logger.warning(
                    "mask_gemma_prompt_tokens: could not find tokenized 'model' role header "
                    "after last <start_of_turn> for at least one sample (e.g. index %d, last_sot=%d); "
                    "skipping prompt masking for those rows.",
                    i,
                    last_sot,
                )
                warned_prompt_masking[0] = True
            continue

        if response_start > labels.size(1):
            continue
        labels[i, :response_start] = ignore_id


def ensure_gemma_mm_token_type_ids(encoded: Dict[str, Any]) -> None:
    """Ensure ``token_type_ids`` and ``mm_token_type_ids`` exist for Gemma multimodal forward.

    Some processors omit these; Gemma 4 may error or misroute modality tokens without them.
    When missing, inject zeros shaped like ``input_ids`` (see ``README/guides/apple-silicon/gemma4-guide.md``
    and https://github.com/huggingface/transformers/issues/45200).
    """
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        return
    ref = input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)
    if "token_type_ids" not in encoded or encoded.get("token_type_ids") is None:
        encoded["token_type_ids"] = torch.zeros_like(ref)
    if "mm_token_type_ids" not in encoded or encoded.get("mm_token_type_ids") is None:
        encoded["mm_token_type_ids"] = torch.zeros_like(ref)


class DataCollatorGemmaText:
    """Batch text-only examples for Gemma CausalLM (no audio forward).

    Instruction mode: user (prompt) + assistant (response) via ``apply_chat_template``.
    When supported, uses ``return_assistant_tokens_mask=True`` for label boundaries;
    otherwise falls back to :func:`mask_gemma_prompt_tokens`. Padding positions are
    masked in ``labels`` using ``attention_mask == 0`` (not ``pad_token_id``), so EOS
    is not dropped when ``pad_token == eos_token``.

    Completion mode: single text column, full sequence trained (no prompt mask).
    """

    def __init__(
        self,
        tokenizer,
        text_column: str,
        prompt_column: Optional[str] = None,
        max_length: int = 2048,
        sub_mode: str = "instruction",
    ):
        if sub_mode not in ("instruction", "completion"):
            raise ValueError(f"DataCollatorGemmaText: sub_mode must be 'instruction' or 'completion', got {sub_mode!r}")
        if sub_mode == "instruction" and not prompt_column:
            raise ValueError("DataCollatorGemmaText: prompt_column is required when sub_mode is 'instruction'")
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.max_length = max_length
        self.sub_mode = sub_mode
        self._warned_prompt_masking: List[bool] = [False]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.sub_mode == "instruction":
            return self._collate_instruction(features)
        return self._collate_completion(features)

    def _collate_instruction(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        assert self.prompt_column is not None
        messages_batch = []
        for ex in features:
            prompt = ex.get(self.prompt_column)
            response = ex.get(self.text_column)
            if prompt is None:
                raise KeyError(
                    f"DataCollatorGemmaText: prompt column {self.prompt_column!r} missing. Keys: {list(ex.keys())}"
                )
            if response is None:
                raise KeyError(
                    f"DataCollatorGemmaText: text column {self.text_column!r} missing. Keys: {list(ex.keys())}"
                )
            messages_batch.append(
                [
                    {"role": "user", "content": str(prompt)},
                    {"role": "assistant", "content": str(response)},
                ]
            )

        tmpl_kwargs = dict(
            tokenize=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_length,
        )
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages_batch,
                return_assistant_tokens_mask=True,
                **tmpl_kwargs,
            )
        except ValueError as e:
            if "assistant" not in str(e).lower() and "return_assistant_tokens_mask" not in str(e):
                raise
            encoded = self.tokenizer.apply_chat_template(messages_batch, **tmpl_kwargs)

        ensure_gemma_mm_token_type_ids(encoded)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        am = encoded.get("assistant_masks")
        if isinstance(am, torch.Tensor) and am.shape == input_ids.shape:
            labels[am == 0] = GemmaTrainingConstants.IGNORE_TOKEN_ID
        else:
            mask_gemma_prompt_tokens(labels, input_ids, self.tokenizer, self._warned_prompt_masking)
        labels[attention_mask == 0] = GemmaTrainingConstants.IGNORE_TOKEN_ID
        encoded["labels"] = labels
        return encoded

    def _collate_completion(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        for ex in features:
            t = ex.get(self.text_column)
            if t is None:
                raise KeyError(
                    f"DataCollatorGemmaText: text column {self.text_column!r} missing. Keys: {list(ex.keys())}"
                )
            texts.append(str(t))

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ensure_gemma_mm_token_type_ids(encoded)
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = GemmaTrainingConstants.IGNORE_TOKEN_ID
        encoded["labels"] = labels
        return encoded


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

        # Real Gemma multimodal processors do not accept processor(messages=...).
        # apply_chat_template(tokenize=True) would try load_audio() on placeholder paths;
        # we render prompts first, then attach pre-loaded waveforms via processor(text=..., audio=...).
        prompts = self.processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.processor(
            text=prompts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=sampling_rate,
        )

        ensure_gemma_mm_token_type_ids(encoded)

        if hasattr(self.processor, "tokenizer"):
            validate_bos_tokens_present(encoded, self.processor.tokenizer)

        if "labels" not in encoded:
            labels = encoded.get("input_ids").clone()
            if "attention_mask" in encoded and hasattr(self.processor, "tokenizer"):
                pad_id = self.processor.tokenizer.pad_token_id
                labels[labels == pad_id] = GemmaTrainingConstants.IGNORE_TOKEN_ID

            mask_gemma_prompt_tokens(
                labels, encoded["input_ids"], self.processor.tokenizer, self._warned_prompt_masking
            )
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
