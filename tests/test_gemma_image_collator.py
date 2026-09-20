"""Tests for DataCollatorGemmaImage (caption + VQA) and image loading."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
import torch
from PIL import Image as PILImage

from gemma_tuner.models.common import collators as collators_mod
from gemma_tuner.models.common.collators import (
    DataCollatorGemmaImage,
    _load_image_as_rgb,
    apply_image_token_budget_to_processor,
    build_image_views,
    completion_logits_to_keep,
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
    """Minimal tokenizer for mask tests with ``<|turn>`` (Gemma 4 start-of-turn, id 105 in real tokenizer)."""

    bos_token_id = 1
    unk_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Match real Gemma 4 tokenizer behavior: <|turn> is a single special token;
        # <|turn|> (with bars on both sides) is NOT a special token and tokenizes to
        # a multi-subword sequence that never appears in the rendered chat.
        if text == "<|turn>":
            return [50]
        if text == "<|turn|>":
            return [900, 901, 902, 903]
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
    mask_gemma_prompt_tokens(labels, input_ids, tok, warned, control_token="<|turn>")
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    # Last <|turn> at index 3; response_start = 3+1+0+2 = 6
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


def test_completion_logits_to_keep_covers_shifted_supervised_suffix() -> None:
    ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
    labels = torch.tensor([[ignore, ignore, ignore, 10, 11, 12]])
    assert completion_logits_to_keep(labels) == 4
    labels[0, 4] = ignore
    with pytest.raises(ValueError, match="contiguous supervised suffix"):
        completion_logits_to_keep(labels)


def test_apply_image_token_budget_warns_without_image_seq_length(caplog):
    collators_mod.reset_apply_image_budget_warning_dedupe()

    class _NoImageSeq:
        pass

    caplog.set_level(logging.WARNING)
    apply_image_token_budget_to_processor(_NoImageSeq(), 280)
    assert "image_seq_length" in caplog.text
    assert "image_token_budget" in caplog.text


def test_apply_image_token_budget_warns_once_per_processor_type(caplog):
    collators_mod.reset_apply_image_budget_warning_dedupe()

    class _NoImageSeq:
        pass

    caplog.set_level(logging.WARNING)
    apply_image_token_budget_to_processor(_NoImageSeq(), 280)
    apply_image_token_budget_to_processor(_NoImageSeq(), 128)
    records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(records) == 1


class _CapturingImageProcessor(FakeImageProcessor):
    """Records the ``images`` kwarg shape passed to ``__call__`` for assertion."""

    def __init__(self):
        super().__init__()
        self.last_images = None
        self.last_messages = None
        self.last_kwargs = None

    def apply_chat_template(self, messages_batch, tokenize=False, add_generation_prompt=False, **kwargs):
        self.last_messages = messages_batch
        return super().apply_chat_template(
            messages_batch,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def __call__(self, text=None, images=None, return_tensors=None, padding=None, **kwargs):
        self.last_images = images
        self.last_kwargs = kwargs
        return super().__call__(text=text, images=images, return_tensors=return_tensors, padding=padding, **kwargs)


class _StrictTelepathicProcessor(FakeImageProcessor):
    def __init__(self):
        super().__init__()
        self.tokenizer.eos_token_id = 1
        original_encode = self.tokenizer.encode

        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            if text == "<end_of_turn>":
                return [1]
            return original_encode(text, add_special_tokens=add_special_tokens)

        self.tokenizer.encode = encode

    def __call__(self, text=None, images=None, return_tensors=None, padding=None, **kwargs):
        encoded = super().__call__(
            text=text,
            images=images,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )
        encoded["input_ids"][:, 7] = self.tokenizer.eos_token_id
        return encoded


def test_image_collator_passes_list_per_sample_to_processor(tmp_path: Path):
    """Gemma 4 multimodal processor expects ``images`` as one list per text sample.

    Regression for PR #22: collator wraps each PIL image as ``[img]`` so batched calls
    keep the per-sample boundary intact.
    """
    proc = _CapturingImageProcessor()
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="caption",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        sub_mode="caption",
    )
    paths = []
    for i in range(3):
        p = tmp_path / f"b_{i}.png"
        p.write_bytes(_make_png_bytes("RGB"))
        paths.append(str(p))
    collator([{"id": str(i), "image_path": paths[i], "caption": "x"} for i in range(3)])
    assert isinstance(proc.last_images, list) and len(proc.last_images) == 3
    for inner in proc.last_images:
        assert isinstance(inner, list) and len(inner) == 1


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


def test_global_plus_quadrants_odd_dimensions_cover_source_exactly() -> None:
    image = PILImage.new("RGB", (5, 3))
    for y in range(3):
        for x in range(5):
            image.putpixel((x, y), (x, y, x + y))
    global_view, top_left, top_right, bottom_left, bottom_right = build_image_views(
        image, "global_plus_four_nonoverlapping_quadrants"
    )
    assert [view.size for view in (global_view, top_left, top_right, bottom_left, bottom_right)] == [
        (5, 3),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
    ]
    reconstructed = PILImage.new("RGB", image.size)
    reconstructed.paste(top_left, (0, 0))
    reconstructed.paste(top_right, (2, 0))
    reconstructed.paste(bottom_left, (0, 1))
    reconstructed.paste(bottom_right, (2, 1))
    assert reconstructed.tobytes() == image.tobytes()


@pytest.mark.parametrize("size", [(1, 5), (5, 1), (1, 1)])
def test_global_plus_quadrants_rejects_one_pixel_dimension(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="width and height >= 2"):
        build_image_views(
            PILImage.new("RGB", size),
            "global_plus_four_nonoverlapping_quadrants",
        )


def test_five_view_conditioned_collator_preserves_order_and_message_bytes(tmp_path: Path) -> None:
    proc = _CapturingImageProcessor()
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="answer",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        prompt_column="question",
        image_view_policy="global_plus_four_nonoverlapping_quadrants",
        system_prompt_column="system_prompt",
        max_length=16,
        sub_mode="vqa",
    )
    path = tmp_path / "odd.png"
    PILImage.new("RGB", (5, 3), color=(10, 20, 30)).save(path)
    collator(
        [
            {
                "id": "x",
                "image_path": str(path),
                "question": "exact context and OCR bytes",
                "answer": '{"context_analysis":{}}',
                "system_prompt": "exact rendered intent bytes",
            }
        ]
    )
    assert [image.size for image in proc.last_images[0]] == [
        (5, 3),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
    ]
    messages = proc.last_messages[0]
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"][0]["text"] == "exact rendered intent bytes"
    assert [item["type"] for item in messages[1]["content"]] == [
        "image",
        "image",
        "image",
        "image",
        "image",
        "text",
    ]
    assert messages[1]["content"][-1]["text"] == "exact context and OCR bytes"
    assert messages[2]["content"][0]["text"] == '{"context_analysis":{}}'
    assert proc.last_kwargs["truncation"] is False


def test_compact_collator_rejects_any_system_prompt_field(tmp_path: Path) -> None:
    proc = FakeImageProcessor()
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="answer",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        prompt_column="question",
        sub_mode="vqa",
    )
    path = tmp_path / "image.png"
    PILImage.new("RGB", (4, 4)).save(path)
    with pytest.raises(ValueError, match="forbidden system_prompt"):
        collator(
            [
                {
                    "image_path": str(path),
                    "question": "q",
                    "answer": "a",
                    "system_prompt": "",
                }
            ]
        )


def test_image_collator_fails_closed_above_max_length(tmp_path: Path) -> None:
    proc = FakeImageProcessor()
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="caption",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        max_length=8,
        sub_mode="caption",
    )
    path = tmp_path / "image.png"
    PILImage.new("RGB", (4, 4)).save(path)
    with pytest.raises(ValueError, match="untruncated batch exceeds"):
        collator([{"image_path": str(path), "caption": "a"}])


def test_strict_telepathic_contract_requires_complete_ocr_target_and_eos(
    tmp_path: Path,
) -> None:
    proc = _StrictTelepathicProcessor()
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="answer",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        prompt_column="question",
        require_telepathic_contract=True,
        completion_only_logits=True,
        sub_mode="vqa",
    )
    path = tmp_path / "image.png"
    PILImage.new("RGB", (4, 4)).save(path)
    prompt = (
        "Use the screenshot, context, and OCR. Return valid JSON with exactly one "
        "top-level key: context_analysis.\n\n"
        "<context>\n{}\n</context>\n\n"
        "<first_pass_screenshot_ocr>\n{}\n</first_pass_screenshot_ocr>\n"
    )
    result = collator(
        [
            {
                "image_path": str(path),
                "question": prompt,
                "answer": '{"context_analysis":{}}',
            }
        ]
    )
    assert int(result["labels"][0, 7]) == proc.tokenizer.eos_token_id
    assert result["logits_to_keep"] == 3

    with pytest.raises(ValueError, match="first_pass_screenshot_ocr"):
        collator(
            [
                {
                    "image_path": str(path),
                    "question": prompt.replace("</first_pass_screenshot_ocr>", ""),
                    "answer": '{"context_analysis":{}}',
                }
            ]
        )
    with pytest.raises(ValueError, match="exactly one context_analysis"):
        collator(
            [
                {
                    "image_path": str(path),
                    "question": prompt,
                    "answer": '{"wrong":{}}',
                }
            ]
        )
