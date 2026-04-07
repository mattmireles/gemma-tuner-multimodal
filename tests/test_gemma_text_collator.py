"""Tests for DataCollatorGemmaText (instruction + completion)."""

from __future__ import annotations

import torch

from gemma_tuner.models.common.collators import DataCollatorGemmaText
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
    ):
        del tokenize, return_tensors, padding, return_dict, add_generation_prompt, truncation, max_length
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
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeCompletionTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        del padding, truncation, max_length, return_tensors
        input_ids = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


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


def test_invalid_sub_mode():
    tok = _FakeInstructionTokenizer()
    try:
        DataCollatorGemmaText(tok, text_column="t", prompt_column="p", sub_mode="other")
    except ValueError as e:
        assert "sub_mode" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
