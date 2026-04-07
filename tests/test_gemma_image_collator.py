"""Tests for DataCollatorGemmaImage (caption + VQA) and image loading."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
import torch
from PIL import Image as PILImage

from gemma_tuner.models.common.collators import (
    DataCollatorGemmaImage,
    _load_image_as_rgb,
    apply_image_token_budget_to_processor,
    mask_gemma_prompt_tokens,
)
from gemma_tuner.models.gemma.constants import GemmaTrainingConstants
from gemma_tuner.models.gemma.family import GemmaFamily
from tests._fakes import FakeImageProcessor


class _TokenizerMaskingGemma3n:
    """Minimal tokenizer for :func:`mask_gemma_prompt_tokens` unit tests (Gemma 3n control token)."""

    bos_token_id = 1
    start_of_turn_token_id = 7
    unk_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "<start_of_turn>":
            return [7]
        if text == "model\n":
            return [20, 21]
        if text == "model":
            return [20]
        return [99]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<start_of_turn>":
            return 7
        return self.unk_token_id


class _TokenizerMaskingGemma4:
    """Minimal tokenizer for mask tests with ``<|turn|>`` (Gemma 4)."""

    bos_token_id = 1
    unk_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "<|turn|>":
            return [50]
        if text == "model\n":
            return [20, 21]
        return [99]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.unk_token_id


def _make_png_bytes(mode: str, size: tuple[int, int] = (32, 32)) -> bytes:
    buf = io.BytesIO()
    if mode == "CMYK":
        im = PILImage.new("CMYK", size, color=(40, 20, 10, 0))
        im.save(buf, format="TIFF")
    elif mode == "RGBA":
        im = PILImage.new("RGBA", size, color=(255, 0, 0, 128))
        im.save(buf, format="PNG")
    else:
        im = PILImage.new("RGB", size, color=(10, 200, 30))
        im.save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_rgba_cmyk_same_size_after_rgb(tmp_path: Path):
    paths = []
    for mode in ("RGB", "RGBA", "CMYK"):
        p = tmp_path / f"t_{mode}.png"
        p.write_bytes(_make_png_bytes(mode))
        paths.append(p)

    rgb_shape = _load_image_as_rgb(paths[0]).size
    assert _load_image_as_rgb(paths[1]).size == rgb_shape
    assert _load_image_as_rgb(paths[2]).size == rgb_shape


def test_load_image_as_rgb_missing_file_raises_file_not_found(tmp_path: Path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        _load_image_as_rgb(missing)


def test_rgba_cmyk_collator_same_output_shape(tmp_path: Path):
    proc = FakeImageProcessor()
    apply_image_token_budget_to_processor(proc, 280)
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="caption",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        image_token_budget=280,
        sub_mode="caption",
    )
    shapes = []
    for mode in ("RGB", "RGBA", "CMYK"):
        p = tmp_path / f"x_{mode}.png"
        p.write_bytes(_make_png_bytes(mode))
        out = collator([{"id": "1", "image_path": str(p), "caption": "Paris"}])
        shapes.append(out["input_ids"].shape)
    assert shapes[0] == shapes[1] == shapes[2]


def test_mask_gemma_prompt_tokens_masks_prompt_and_keeps_assistant_answer_gemma3n():
    """Boundary comes from ``mask_gemma_prompt_tokens`` (last SOT + ``model\\n`` subsequence), not the fake processor."""
    tok = _TokenizerMaskingGemma3n()
    # Simulated layout: BOS, several <start_of_turn> spans (user + noise), then model header, then answer ids.
    # Last SOT at index 7; after it [20,21]=model\\n, then supervised tokens 200,201.
    input_ids = torch.tensor([[1, 7, 10, 11, 7, 12, 7, 7, 20, 21, 200, 201]])
    labels = input_ids.clone()
    warned = [False]
    mask_gemma_prompt_tokens(
        labels,
        input_ids,
        tok,
        warned,
        control_token="<start_of_turn>",
    )
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    assert (labels[0, :10] == ignore).all()
    assert (labels[0, 10:] == input_ids[0, 10:]).all()


def test_mask_gemma_prompt_tokens_uses_last_control_span_not_first_gemma3n():
    """Extra ``<start_of_turn>`` inside the simulated user region: masking still keys off the last SOT before ``model\\n``."""
    tok = _TokenizerMaskingGemma3n()
    input_ids = torch.tensor([[1, 7, 9, 9, 7, 8, 7, 20, 21, 30, 31]])
    labels = input_ids.clone()
    warned = [False]
    mask_gemma_prompt_tokens(labels, input_ids, tok, warned, control_token="<start_of_turn>")
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    # Last SOT at index 6; response after model\\n starts at 6+1+0+2 = 9
    assert (labels[0, :9] == ignore).all()
    assert (labels[0, 9:] == input_ids[0, 9:]).all()


def test_mask_gemma_prompt_tokens_gemma4_turn_marker():
    tok = _TokenizerMaskingGemma4()
    input_ids = torch.tensor([[1, 50, 10, 50, 20, 21, 200, 201]])
    labels = input_ids.clone()
    warned = [False]
    mask_gemma_prompt_tokens(labels, input_ids, tok, warned, control_token="<|turn|>")
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    # Last <|turn|> at index 3; response_start = 3+1+0+2 = 6
    assert (labels[0, :6] == ignore).all()
    assert (labels[0, 6:] == input_ids[0, 6:]).all()


def test_mask_gemma_prompt_tokens_leaves_row_unchanged_when_model_header_missing():
    tok = _TokenizerMaskingGemma3n()
    input_ids = torch.tensor([[1, 7, 10, 11, 200, 201]])
    labels = input_ids.clone()
    warned = [False]
    mask_gemma_prompt_tokens(labels, input_ids, tok, warned, control_token="<start_of_turn>")
    assert (labels == input_ids).all()


def test_vqa_first_supervised_token_matches_answer(tmp_path: Path):
    """Smoke: collator + fake processor; does not assert real chat-template tokenization (see mask_gemma_prompt_tokens tests)."""
    proc = FakeImageProcessor()
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="answer",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        prompt_column="question",
        image_token_budget=280,
        sub_mode="vqa",
    )
    p = tmp_path / "q.png"
    p.write_bytes(_make_png_bytes("RGB"))
    out = collator(
        [
            {
                "id": "a",
                "image_path": str(p),
                "question": "Capital of France?",
                "answer": "Paris",
            }
        ]
    )
    labels = out["labels"][0]
    input_ids = out["input_ids"][0]
    first_supervised = (labels != GemmaTrainingConstants.IGNORE_TOKEN_ID).nonzero(as_tuple=True)[0]
    assert first_supervised.numel() >= 1
    idx = int(first_supervised[0].item())
    assert idx < input_ids.numel()
    first_word_id = proc.tokenizer.encode("Paris", add_special_tokens=False)[0]
    assert int(input_ids[idx].item()) == first_word_id


def test_apply_image_token_budget_rebuilds_sequence():
    proc = FakeImageProcessor()
    proc.image_seq_length = 100
    proc.full_image_sequence = "old"
    apply_image_token_budget_to_processor(proc, 280)
    assert proc.image_seq_length == 280
    assert "old" not in proc.full_image_sequence
    assert proc.full_image_sequence.count("<img>") == 280


def test_apply_image_token_budget_warns_without_image_seq_length(caplog):
    class _NoImageSeq:
        pass

    caplog.set_level(logging.WARNING)
    apply_image_token_budget_to_processor(_NoImageSeq(), 280)
    assert "image_seq_length" in caplog.text
    assert "image_token_budget" in caplog.text


def test_image_collator_masks_padding_via_attention_mask_only(tmp_path: Path):
    """Padding is masked with attention_mask == 0 (not pad_id equality)."""
    proc = FakeImageProcessor()
    collator = DataCollatorGemmaImage(
        proc,
        text_column="caption",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        sub_mode="caption",
    )
    p = tmp_path / "pad.png"
    p.write_bytes(_make_png_bytes("RGB"))
    out = collator([{"id": "1", "image_path": str(p), "caption": "Paris"}])
    am = out["attention_mask"]
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    assert (out["labels"][am == 0] == ignore).all()
    assert (out["labels"][am == 1] != ignore).any()
