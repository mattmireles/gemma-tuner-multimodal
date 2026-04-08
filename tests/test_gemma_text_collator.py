"""Tests for DataCollatorGemmaText (instruction + completion)."""

from __future__ import annotations

import numpy as np
import torch

from gemma_tuner.models.common.collators import (
    DataCollatorGemmaText,
    _find_subsequence_ids,
    inject_mm_token_type_ids,
    mask_gemma_prompt_tokens,
)
from gemma_tuner.models.gemma.constants import GemmaTrainingConstants
from gemma_tuner.models.gemma.family import GemmaFamily


class _FakeInstructionTokenizer:
    """Minimal tokenizer mimicking Gemma layout for offline tests (no HF download)."""

    bos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3
    start_of_turn_token_id = 7

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "<start_of_turn>":
            return [7]
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


class _FakeLongPromptTokenizer:
    """Token count grows with user length; truncation=True keeps first max_length tokens (drops tail / assistant)."""

    bos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages_batch, **kwargs):
        return_assistant_tokens_mask = kwargs.pop("return_assistant_tokens_mask", False)
        truncation = kwargs.pop("truncation", True)
        max_length = kwargs.pop("max_length", 2048)
        for k in (
            "tokenize",
            "return_tensors",
            "padding",
            "return_dict",
            "add_generation_prompt",
        ):
            kwargs.pop(k, None)
        kwargs.clear()

        user = str(messages_batch[0][0]["content"])
        assistant = str(messages_batch[0][1]["content"])
        n_tokens = 4 + (len(user) // 3) + (len(assistant) // 2)
        lost_assistant = bool(truncation and n_tokens > max_length)
        if lost_assistant:
            n_tokens = max_length
        input_ids = torch.zeros(1, n_tokens, dtype=torch.long)
        input_ids[0, 0] = self.bos_token_id
        if not lost_assistant:
            input_ids[0, -2:] = torch.tensor([200, 201])
        else:
            input_ids[0, -2:] = torch.tensor([3, 3])
        attention_mask = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_assistant_tokens_mask:
            am = torch.zeros_like(input_ids)
            if not lost_assistant:
                am[0, -2:] = 1
            out["assistant_masks"] = am
        return out


class _FakeGemma4AllZeroAssistantMask:
    """HF can return assistant_masks with correct shape but all zeros (no {% generation %} in template).

    Also mimics the real Gemma 4 tokenizer behavior: ``<|turn>`` (the actual start-of-turn
    marker) is a single token (id 50), while ``<|turn|>`` (an older incorrect guess) tokenizes
    into a 4-token subword sequence that matches nothing in real inputs. Tests that pass the
    wrong marker will silently fall back to no-masking and can be caught by asserting warnings.
    """

    bos_token_id = 1

    def apply_chat_template(self, messages_batch, **kwargs):
        kwargs.pop("return_assistant_tokens_mask", None)
        for k in (
            "tokenize",
            "return_tensors",
            "padding",
            "return_dict",
            "add_generation_prompt",
            "truncation",
            "max_length",
        ):
            kwargs.pop(k, None)
        # Same layout as test_mask_gemma_prompt_tokens_gemma4_uses_turn_marker_then_model_header
        input_ids = torch.tensor([[1, 50, 7, 8, 50, 20, 21, 100, 101]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        out["assistant_masks"] = torch.zeros_like(input_ids)
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "<|turn>":
            return [50]
        if text == "<|turn|>":
            # Matches real Gemma 4 tokenizer behavior — this string is NOT a special
            # token and tokenizes to a 4-subword sequence that never matches anything
            # in the rendered chat. Kept in the fake so regression tests can catch a
            # return to the old (wrong) marker.
            return [900, 901, 902, 903]
        if text == "model\n":
            return [20, 21]
        if text == "model":
            return [20]
        if text.strip() == "Hello":
            return [100]
        return [99]


class _FakeCompletionTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        del padding, truncation, max_length, return_tensors
        input_ids = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_instruction_gemma4_injects_mm_token_type_ids():
    tok = _FakeInstructionTokenizer()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        family=GemmaFamily.GEMMA_4,
        prompt_column="prompt",
        max_length=128,
        sub_mode="instruction",
    )
    out = collator(
        [
            {"prompt": "p1", "response": "Hello"},
        ]
    )
    assert "token_type_ids" in out and "mm_token_type_ids" in out
    assert torch.equal(out["token_type_ids"], torch.zeros_like(out["input_ids"]))
    assert torch.equal(out["mm_token_type_ids"], torch.zeros_like(out["input_ids"]))


def test_instruction_gemma4_degenerate_all_zero_assistant_masks_uses_fallback():
    """Belt-and-suspenders: even if a (hypothetical) HF version returns all-zero assistant_masks
    for Gemma 4, the collator must fall through to :func:`mask_gemma_prompt_tokens` and produce
    correct labels.

    Under normal operation the ``supports_assistant_mask=False`` capability causes the collator
    to never ask for ``return_assistant_tokens_mask=True`` in the first place, so ``assistant_masks``
    is absent from the encoded output. This test exercises the defensive branch where the key
    *is* present but degenerate — guarding against a future tokenizer release that volunteers
    an all-zero mask even without the flag.
    """
    tok = _FakeGemma4AllZeroAssistantMask()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        family=GemmaFamily.GEMMA_4,
        prompt_column="prompt",
        max_length=128,
        sub_mode="instruction",
    )
    out = collator([{"prompt": "p", "response": "Hello"}])
    labels = out["labels"]
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    assert (labels != ignore).any(), "expected some supervised tokens, not all -100"
    assert labels[0, 7].item() == 100
    assert labels[0, 8].item() == 101
    assert (labels[0, :7] == ignore).all()


def test_instruction_gemma4_skips_primary_assistant_mask_path():
    """Gemma 4's chat template lacks ``{% generation %}``, so the collator must not ask HF
    for ``return_assistant_tokens_mask=True`` at all — doing so prints a noisy per-batch warning
    and returns all-zeros anyway. Verify the collator calls ``apply_chat_template`` without that
    kwarg for Gemma 4.
    """

    captured_kwargs: list[dict] = []

    class _SpyTokenizer:
        bos_token_id = 1

        def apply_chat_template(self, messages_batch, **kwargs):
            captured_kwargs.append(dict(kwargs))
            # Return a minimal encoded dict sufficient for the collator to proceed.
            input_ids = torch.tensor([[1, 50, 7, 50, 20, 21, 100, 101]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if text == "<|turn>":
                return [50]
            if text == "model\n":
                return [20, 21]
            if text == "model":
                return [20]
            return [99]

    tok = _SpyTokenizer()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        family=GemmaFamily.GEMMA_4,
        prompt_column="prompt",
        max_length=128,
        sub_mode="instruction",
        # Disable pre-measurement so the spy only captures the final encoding call.
        instruction_truncation="none",
    )
    collator([{"prompt": "p", "response": "Hello"}])
    assert len(captured_kwargs) == 1
    assert "return_assistant_tokens_mask" not in captured_kwargs[0], (
        "Gemma 4 must NOT request return_assistant_tokens_mask (chat template lacks the "
        "`{% generation %}` block; HF would return all-zeros + warn every batch)."
    )


def test_instruction_gemma3n_still_uses_primary_assistant_mask_path():
    """Gemma 3n's template *does* have ``{% generation %}``, so the collator should keep
    using ``return_assistant_tokens_mask=True`` for this family.
    """

    captured_kwargs: list[dict] = []

    class _SpyTokenizer:
        bos_token_id = 1

        def apply_chat_template(self, messages_batch, **kwargs):
            captured_kwargs.append(dict(kwargs))
            input_ids = torch.tensor([[1, 7, 9, 7, 20, 21, 100, 101]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            out = {"input_ids": input_ids, "attention_mask": attention_mask}
            if kwargs.get("return_assistant_tokens_mask"):
                # Mark the last two positions as supervised (Gemma 3n's HF path works).
                am = torch.zeros_like(input_ids)
                am[0, -2:] = 1
                out["assistant_masks"] = am
            return out

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if text == "<start_of_turn>":
                return [7]
            if text == "model\n":
                return [20, 21]
            if text == "model":
                return [20]
            return [99]

    tok = _SpyTokenizer()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        family=GemmaFamily.GEMMA_3N,
        prompt_column="prompt",
        max_length=128,
        sub_mode="instruction",
        instruction_truncation="none",
    )
    collator([{"prompt": "p", "response": "Hello"}])
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("return_assistant_tokens_mask") is True, (
        "Gemma 3n must continue to use the primary assistant_masks path."
    )


def test_mask_gemma_prompt_tokens_gemma4_uses_turn_marker_then_model_header():
    """Gemma 4: boundary is ``<|turn>`` (single token); assistant content follows tokenized model role header."""

    class _Tok:
        bos_token_id = 1

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Real Gemma 4 tokenizer: <|turn> is a single token id; <|turn|> is not a
            # special token and tokenizes to a 4-subword sequence. We mirror that split
            # so a regression to the wrong marker string would break this test instead
            # of silently passing.
            if text == "<|turn>":
                return [50]
            if text == "<|turn|>":
                return [900, 901, 902, 903]
            if text == "model\n":
                return [20, 21]
            if text == "model":
                return [20]
            return [99]

    tok = _Tok()
    # Two <|turn> markers; last begins assistant turn; model\n then response tokens.
    input_ids = torch.tensor([[1, 50, 7, 8, 50, 20, 21, 100, 101]], dtype=torch.long)
    labels = input_ids.clone()
    warned: list[bool] = [False]
    mask_gemma_prompt_tokens(
        labels,
        input_ids,
        tok,
        warned,
        control_token="<|turn>",
    )
    assert warned[0] is False
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    assert (labels[0, :7] == ignore).all()
    assert labels[0, 7].item() == 100
    assert labels[0, 8].item() == 101


def test_mask_gemma_prompt_tokens_gemma4_wrong_marker_warns_and_leaves_labels_unmasked():
    """Regression guard: passing the incorrect ``<|turn|>`` marker must NOT silently match.

    If someone reverts `family.py` to the old (wrong) ``<|turn|>`` string, the real Gemma 4
    tokenizer encodes it to a 4-subword sequence that never appears in the rendered chat,
    so the subsequence search finds nothing, a warning fires, and masking is skipped.
    """

    class _Tok:
        bos_token_id = 1

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if text == "<|turn>":
                return [50]
            if text == "<|turn|>":
                return [900, 901, 902, 903]
            if text == "model\n":
                return [20, 21]
            if text == "model":
                return [20]
            return [99]

        def convert_tokens_to_ids(self, token: str) -> int:
            # Mirror HF behavior for non-special tokens: return unk_token_id (here 3).
            return 3

        unk_token_id = 3

    tok = _Tok()
    input_ids = torch.tensor([[1, 50, 7, 8, 50, 20, 21, 100, 101]], dtype=torch.long)
    labels = input_ids.clone()
    warned: list[bool] = [False]
    mask_gemma_prompt_tokens(
        labels,
        input_ids,
        tok,
        warned,
        control_token="<|turn|>",  # WRONG marker on purpose
    )
    assert warned[0] is True, "wrong marker should trigger the skip-masking warning"
    # Labels should be unchanged (no tokens masked) because the marker never matched.
    assert torch.equal(labels, input_ids)


def test_find_subsequence_ids():
    h = torch.tensor([1, 2, 20, 21, 5])
    assert _find_subsequence_ids(h, [20, 21]) == 2
    assert _find_subsequence_ids(h, [99]) == -1


def test_instruction_submode_masks_prompt_and_padding():
    tok = _FakeInstructionTokenizer()
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        family=GemmaFamily.GEMMA_3N,
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
        family=GemmaFamily.GEMMA_3N,
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
        family=GemmaFamily.GEMMA_3N,
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
        family=GemmaFamily.GEMMA_3N,
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
        DataCollatorGemmaText(
            tok, text_column="t", family=GemmaFamily.GEMMA_3N, prompt_column=None, sub_mode="instruction"
        )
    except ValueError as e:
        assert "prompt_column" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_inject_mm_token_type_ids_fills_missing_and_none():
    batch = {"input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)}
    inject_mm_token_type_ids(batch)
    assert torch.equal(batch["token_type_ids"], torch.zeros_like(batch["input_ids"]))
    assert torch.equal(batch["mm_token_type_ids"], torch.zeros_like(batch["input_ids"]))

    batch2 = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "token_type_ids": None,
    }
    inject_mm_token_type_ids(batch2)
    assert "mm_token_type_ids" in batch2
    assert torch.equal(batch2["token_type_ids"], torch.zeros(1, 2, dtype=torch.long))
    assert torch.equal(batch2["mm_token_type_ids"], torch.zeros(1, 2, dtype=torch.long))


def test_invalid_sub_mode():
    tok = _FakeInstructionTokenizer()
    try:
        DataCollatorGemmaText(tok, text_column="t", family=GemmaFamily.GEMMA_3N, prompt_column="p", sub_mode="other")
    except ValueError as e:
        assert "sub_mode" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_instruction_left_user_truncates_prompt_keeps_assistant_tail():
    """Long user text is left-trimmed so HF truncation does not drop the assistant span."""
    tok = _FakeLongPromptTokenizer()
    long_prompt = "x" * 3000
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        family=GemmaFamily.GEMMA_3N,
        prompt_column="prompt",
        max_length=20,
        sub_mode="instruction",
        instruction_truncation="left_user",
    )
    out = collator([{"prompt": long_prompt, "response": "yy"}])
    assert out["input_ids"][0, -2].item() == 200
    assert out["input_ids"][0, -1].item() == 201


def test_instruction_truncation_none_drops_assistant_tail():
    """Without pre-truncation, tokenizer-only truncation can remove the assistant from the window."""
    tok = _FakeLongPromptTokenizer()
    long_prompt = "x" * 3000
    collator = DataCollatorGemmaText(
        tok,
        text_column="response",
        family=GemmaFamily.GEMMA_3N,
        prompt_column="prompt",
        max_length=20,
        sub_mode="instruction",
        instruction_truncation="none",
    )
    out = collator([{"prompt": long_prompt, "response": "yy"}])
    assert out["input_ids"][0, -2].item() == 3


def test_invalid_instruction_truncation():
    tok = _FakeInstructionTokenizer()
    try:
        DataCollatorGemmaText(
            tok,
            text_column="t",
            family=GemmaFamily.GEMMA_3N,
            prompt_column="p",
            sub_mode="instruction",
            instruction_truncation="bad",
        )
    except ValueError as e:
        assert "instruction_truncation" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
