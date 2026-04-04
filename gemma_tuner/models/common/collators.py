from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorWhisperStrict:
    """
    Batch collator for standard Gemma fine-tuning with language-mode handling.

    Supports:
    - mixed: single batch, standard padding
    - strict: group by per-sample language, pad within group, then concatenate
    - override: same as mixed but language is forced upstream
    """

    processor: Any
    tokenizer: Any
    feature_extractor: Any
    language_mode: str = "mixed"
    ignore_token_id: int = -100

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Assembles a batch from list of sample features with language-aware processing.

        Called by:
        - PyTorch DataLoader during training/evaluation batch assembly
        - Seq2SeqTrainer.training_step() for each training iteration

        Processing flow:
        1. Check language_mode to determine batching strategy
        2. For strict mode: group samples by language attribute
        3. Pad audio features to maximum length in batch/group
        4. Pad text labels and mask padding positions
        5. For strict mode: align and concatenate language groups
        6. Return batch dictionary for model forward pass

        Args:
            features: List of preprocessed samples, each containing:
                - input_features: Mel-spectrogram tensor [seq_len, 80]
                - labels: Tokenized text tensor [text_len]
                - language: Language code (for strict mode)

        Returns:
            Dictionary containing:
                - input_features: Padded audio batch [batch_size, max_seq_len, 80]
                - labels: Padded and masked text batch [batch_size, max_text_len]
        """
        if self.language_mode == "strict":
            features_by_language: Dict[str, List[Dict[str, Union[List[int], torch.Tensor]]]] = {}
            for feature in features:
                lang = feature["language"]
                features_by_language.setdefault(lang, []).append(feature)

            batches: List[Dict[str, torch.Tensor]] = []
            for _, lang_features in features_by_language.items():
                input_features = [{"input_features": f["input_features"]} for f in lang_features]
                batch = self.feature_extractor.pad(input_features, return_tensors="pt")

                label_features = [{"input_ids": f["labels"]} for f in lang_features]
                labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.ignore_token_id)
                batch["labels"] = labels.long()
                batches.append(batch)

            # input_features shape is [B, n_mels, T] — time is dim 2
            max_len_feats = max(b["input_features"].shape[2] for b in batches)
            max_len_labels = max(b["labels"].shape[1] for b in batches)

            padded_batches: List[Dict[str, torch.Tensor]] = []
            for b in batches:
                pb: Dict[str, torch.Tensor] = {}
                pf = max_len_feats - b["input_features"].shape[2]
                pl = max_len_labels - b["labels"].shape[1]
                # Pad the last (time) dimension: F.pad tuple is (last_dim_left, last_dim_right)
                pb["input_features"] = torch.nn.functional.pad(b["input_features"], (0, pf))
                pb["labels"] = torch.nn.functional.pad(b["labels"], (0, pl), value=self.ignore_token_id)
                padded_batches.append(pb)

            final_batch: Dict[str, torch.Tensor] = {}
            final_batch["input_features"] = torch.cat([b["input_features"] for b in padded_batches], dim=0)
            final_batch["labels"] = torch.cat([b["labels"] for b in padded_batches], dim=0)
            return final_batch

        # mixed or override
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.ignore_token_id)
        batch["labels"] = labels.long()
        return batch


@dataclass
class DataCollatorWhisperLoRA:
    """
    Batch collator for LoRA fine-tuning.
    - Pads audio features and labels per batch
    - Replaces padding with -100 for label loss masking
    - Removes BOS if present
    """

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        feats_padded = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Strip BOS per-sample rather than with a batch-level .all() check.
        # The batch-level approach skips stripping entirely if even one sample lacks BOS,
        # creating inconsistent label targets. Instead we strip per-sample and warn on
        # mixed batches (some samples have BOS, some don't).
        bos_id = self.processor.tokenizer.bos_token_id
        has_bos = labels[:, 0] == bos_id  # shape: [batch_size], dtype bool
        if has_bos.any().item() and not has_bos.all().item():
            logger.warning(
                "DataCollatorWhisperLoRA: mixed BOS batch — only %d/%d samples have BOS at position 0. "
                "Stripping BOS per-sample to avoid inconsistent label targets.",
                has_bos.sum().item(),
                labels.size(0),
            )
        if has_bos.any().item():
            # Build output row-by-row: strip column 0 only for samples that have BOS there.
            # For samples WITHOUT BOS, keep them at the original length and re-pad the stripped
            # samples on the right with -100 so all rows remain the same length. Using
            # labels[i, :-1] would be wrong here because the last token is not guaranteed to be
            # padding — the longest sample in the batch has no trailing pad at all.
            stripped_rows = []
            for i in range(labels.size(0)):
                if has_bos[i].item():
                    # Drop BOS (col 0) and append a -100 pad on the right to keep original length.
                    stripped_rows.append(torch.cat([labels[i, 1:], labels.new_full((1,), -100)]))
                else:
                    stripped_rows.append(labels[i])
            labels = torch.stack(stripped_rows, dim=0)

        # Return the clean keys Gemma uses during training
        batch = {
            "input_features": feats_padded["input_features"],
            "labels": labels,
        }
        return batch


@dataclass
class DataCollatorWhisperDistill:
    """
    Batch collator for distillation training with decoder-start handling.
    - Pads audio features and labels
    - Masks padding positions with -100
    - Removes decoder_start_token_id if present at position 0
    """

    processor: Any
    decoder_start_token_id: int
    ignore_token_id: int = -100

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: f[model_input_name]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.ignore_token_id)

        # Strip decoder_start_token per-sample rather than with a batch-level .all() check.
        # The batch-level approach skips stripping entirely if even one sample lacks the token,
        # creating inconsistent label targets. Warn when the batch is mixed.
        has_dst = labels[:, 0] == self.decoder_start_token_id  # shape: [batch_size], dtype bool
        if has_dst.any().item() and not has_dst.all().item():
            logger.warning(
                "DataCollatorWhisperDistill: mixed decoder_start_token batch — only %d/%d samples have "
                "decoder_start_token_id (%d) at position 0. Stripping per-sample.",
                has_dst.sum().item(),
                labels.size(0),
                self.decoder_start_token_id,
            )
        if has_dst.any().item():
            # Build output row-by-row: strip column 0 only for samples that have the token there.
            # For samples WITHOUT the token, keep at original length; re-pad stripped samples on
            # the right with ignore_token_id. Using labels[i, :-1] would corrupt real content for
            # the longest (unpadded) samples in the batch.
            stripped_rows = []
            for i in range(labels.size(0)):
                if has_dst[i].item():
                    stripped_rows.append(torch.cat([labels[i, 1:], labels.new_full((1,), self.ignore_token_id)]))
                else:
                    stripped_rows.append(labels[i])
            labels = torch.stack(stripped_rows, dim=0)

        batch["labels"] = labels
        return batch


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
