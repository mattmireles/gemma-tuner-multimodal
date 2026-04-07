"""Tests for DataCollatorGemmaText (instruction + completion)."""

from __future__ import annotations

import numpy as np
import torch

from gemma_tuner.models.common.collators import (
    DataCollatorGemmaText,
    _find_subsequence_ids,
    ensure_gemma_mm_token_type_ids,
)
from gemma_tuner.models.gemma.constants import GemmaTrainingConstants


class _FakeInstructionTokenizer:
    """Minimal tokenizer mimicking Gemma layout for offline tests (no HF download)."""

    bos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3
    start_of_turn_token_id = 7

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "model\n":
            return [20, 21]
        # First word of assistant response "Hello" -> single id
        if text.strip() == "Hello":
            return [100]
        return [99]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<start_of_turn>":
            return 7
        return self.unk_token_id

    def apply_chat_template(
        self,
        messages_batch,
        *,
        tokenize: bool = True,
        return_tensors: str = "pt",
        padding: bool = True,
        return_dict: bool = True,
        add_generation_prompt: bool = False,
        truncation: bool = True,
        max_length: int = 2048,
        return_assistant_tokens_mask: bool = False,
        **kwargs,
    ):
        del tokenize, return_tensors, padding, return_dict, add_generation_prompt, truncation, max_length, kwargs
        batch = len(messages_batch)
        input_ids = torch.zeros(batch, 12, dtype=torch.long)
        attention_mask = torch.ones(batch, 12, dtype=torch.long)
        for i in range(batch):
            input_ids[i, 0] = self.bos_token_id
            input_ids[i, 2] = 7
            input_ids[i, 5] = 7
            input_ids[i, 6:8] = torch.tensor([20, 21])
            # First supervised token of assistant (matches encode("Hello")[0])
            input_ids[i, 8] = 100
            input_ids[i, 9] = 42
            attention_mask[i, -2:] = 0
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_assistant_tokens_mask:
            am = torch.zeros_like(input_ids)
            am[:, 8:10] = 1
            out["assistant_masks"] = am
        return out


class _FakeTokenizerGapBeforeModelHeader(_FakeInstructionTokenizer):
    """Extra token between last <start_of_turn> and model\\n ids — fixed offset would fail."""

    def apply_chat_template(self, messages_batch, **kwargs):
        kwargs.pop("return_assistant_tokens_mask", None)
        batch = len(messages_batch)
        input_ids = torch.zeros(batch, 13, dtype=torch.long)
        attention_mask = torch.ones(batch, 13, dtype=torch.long)
        for i in range(batch):
            input_ids[i, 0] = self.bos_token_id
            input_ids[i, 2] = 7
            input_ids[i, 5] = 7
            input_ids[i, 6] = 99
            input_ids[i, 7:9] = torch.tensor([20, 21])
            input_ids[i, 9] = 100
            input_ids[i, 10] = 42
            attention_mask[i, -2:] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _SlowTokenizerRejected(_FakeInstructionTokenizer):
    """Simulates slow tokenizer: assistant mask unsupported, fallback path runs."""

    def apply_chat_template(self, messages_batch, **kwargs):
        if kwargs.pop("return_assistant_tokens_mask", False):
            raise ValueError(
                "`return_assistant_tokens_mask` is not possible with slow tokenizers. "
                "Make sure you have `tokenizers` installed."
            )
        return super().apply_chat_template(messages_batch, **kwargs)


class _FakeCompletionTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        del padding, truncation, max_length, return_tensors
        input_ids = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_find_subsequence_ids():
    h = torch.tensor([1, 2, 20, 21, 5])
    assert _find_subsequence_ids(h, [20, 21]) == 2
    assert _find_subsequence_ids(h, [99]) == -1


def test_instruction_submode_masks_prompt_and_padding():
    tok = _FakeInstructionTokenizer()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        prompt_column="prompt",
        max_length=128,
        sub_mode="instruction",
    )
    batch = [
        {"prompt": "p1", "response": "Hello"},
        {"prompt": "p2", "response": "Hello"},
    ]
    out = collator(batch)
    assert "labels" in out and "input_ids" in out
    assert "token_type_ids" in out and "mm_token_type_ids" in out
    assert torch.equal(out["token_type_ids"], torch.zeros_like(out["input_ids"]))
    assert torch.equal(out["mm_token_type_ids"], torch.zeros_like(out["input_ids"]))
    labels, input_ids, am = out["labels"], out["input_ids"], out["attention_mask"]
    assert labels.shape == input_ids.shape
    # Padding masked
    assert (labels[am == 0] == GemmaTrainingConstants.IGNORE_TOKEN_ID).all()
    # First trainable token matches response first word encoding (position 8 in fake layout)
    first_word_id = tok.encode("Hello", add_special_tokens=False)[0]
    for i in range(labels.size(0)):
        row = labels[i]
        non_ignore = (row != GemmaTrainingConstants.IGNORE_TOKEN_ID).nonzero(as_tuple=True)[0]
        assert non_ignore.numel() > 0
        first_pos = non_ignore[0].item()
        assert input_ids[i, first_pos].item() == first_word_id


def test_instruction_fallback_subsequence_when_extra_tokens_before_model_header():
    tok = _FakeTokenizerGapBeforeModelHeader()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        prompt_column="prompt",
        max_length=128,
        sub_mode="instruction",
    )
    out = collator([{"prompt": "p", "response": "Hello"}])
    labels, input_ids = out["labels"], out["input_ids"]
    first_word_id = tok.encode("Hello", add_special_tokens=False)[0]
    non_ignore = (labels[0] != GemmaTrainingConstants.IGNORE_TOKEN_ID).nonzero(as_tuple=True)[0]
    assert non_ignore[0].item() == 9
    assert input_ids[0, 9].item() == first_word_id


def test_instruction_slow_tokenizer_falls_back_to_subsequence_mask():
    tok = _SlowTokenizerRejected()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        prompt_column="prompt",
        max_length=128,
        sub_mode="instruction",
    )
    out = collator([{"prompt": "p", "response": "Hello"}])
    labels, input_ids = out["labels"], out["input_ids"]
    first_word_id = tok.encode("Hello", add_special_tokens=False)[0]
    non_ignore = (labels[0] != GemmaTrainingConstants.IGNORE_TOKEN_ID).nonzero(as_tuple=True)[0]
    assert non_ignore[0].item() == 8
    assert input_ids[0, 8].item() == first_word_id


def test_completion_submode_masks_only_padding_not_eos_when_pad_equals_eos():
    tok = _FakeCompletionTokenizer()
    collator = DataCollatorGemmaText(
        tok,
        text_column="text",
        prompt_column=None,
        max_length=128,
        sub_mode="completion",
    )
    out = collator([{"text": "a"}, {"text": "bcd"}])
    assert "token_type_ids" in out and "mm_token_type_ids" in out
    assert torch.equal(out["token_type_ids"], torch.zeros_like(out["input_ids"]))
    assert torch.equal(out["mm_token_type_ids"], torch.zeros_like(out["input_ids"]))
    labels, input_ids = out["labels"], out["input_ids"]
    assert tok.pad_token_id == tok.eos_token_id
    # Row 0: valid tokens 5,6 — 6 is eos; must not be wiped by a naive pad_id mask
    assert labels[0, 1].item() == input_ids[0, 1].item() == 6
    assert (labels[0, 2:] == GemmaTrainingConstants.IGNORE_TOKEN_ID).all()


def test_instruction_requires_prompt_column():
    tok = _FakeInstructionTokenizer()
    try:
        DataCollatorGemmaText(tok, text_column="t", prompt_column=None, sub_mode="instruction")
    except ValueError as e:
        assert "prompt_column" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_ensure_gemma_mm_token_type_ids_fills_missing_and_none():
    batch = {"input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)}
    ensure_gemma_mm_token_type_ids(batch)
    assert torch.equal(batch["token_type_ids"], torch.zeros_like(batch["input_ids"]))
    assert torch.equal(batch["mm_token_type_ids"], torch.zeros_like(batch["input_ids"]))

    batch2 = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "token_type_ids": None,
    }
    ensure_gemma_mm_token_type_ids(batch2)
    assert "mm_token_type_ids" in batch2
    assert torch.equal(batch2["token_type_ids"], torch.zeros(1, 2, dtype=torch.long))
    assert torch.equal(batch2["mm_token_type_ids"], torch.zeros(1, 2, dtype=torch.long))


def test_invalid_sub_mode():
    tok = _FakeInstructionTokenizer()
    try:
        DataCollatorGemmaText(tok, text_column="t", prompt_column="p", sub_mode="other")
    except ValueError as e:
        assert "sub_mode" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
