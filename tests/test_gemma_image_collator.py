"""Tests for DataCollatorGemmaImage (caption + VQA) and image loading."""

from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image as PILImage

from gemma_tuner.models.common.collators import (
    DataCollatorGemmaImage,
    _load_image_as_rgb,
    apply_image_token_budget_to_processor,
)
from gemma_tuner.models.gemma.constants import GemmaTrainingConstants
from gemma_tuner.models.gemma.family import GemmaFamily


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


class _FakeImageProcessor:
    """Minimal processor: records image modes and returns fixed tensor shapes."""

    def __init__(self):
        class Tok:
            pad_token_id = 0
            bos_token_id = 1
            unk_token_id = 3
            start_of_turn_token_id = 7

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                if text == "<start_of_turn>":
                    return [7]
                if text == "model\n":
                    return [20, 21]
                if text.strip() == "Paris":
                    return [200]
                return [99]

            def convert_tokens_to_ids(self, token: str) -> int:
                if token == "<start_of_turn>":
                    return 7
                return self.unk_token_id

        self.tokenizer = Tok()
        self.image_seq_length = 256
        self.boi_token = "<boi>"
        self.eoi_token = "<eoi>"
        self.image_token = "<img>"
        self.full_image_sequence = ""

    def apply_chat_template(self, messages_batch, tokenize=False, add_generation_prompt=False, **kwargs):
        del kwargs
        assert tokenize is False
        batch = len(messages_batch)
        return [f"prompt{i}" for i in range(batch)]

    def __call__(self, text=None, images=None, return_tensors=None, padding=None, **kwargs):
        del kwargs
        assert text is not None and images is not None
        batch = len(text)
        # [bos, …, sot, …, sot, model\\n ids, answer, pad] — answer stays in supervised region
        input_ids = torch.zeros((batch, 9), dtype=torch.long)
        attention_mask = torch.ones((batch, 9), dtype=torch.long)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids[:, 2] = 7
        input_ids[:, 4] = 7
        input_ids[:, 5:7] = torch.tensor([20, 21])
        input_ids[:, 7] = 200
        attention_mask[:, 8] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_load_image_rgba_cmyk_same_size_after_rgb(tmp_path: Path):
    paths = []
    for mode in ("RGB", "RGBA", "CMYK"):
        p = tmp_path / f"t_{mode}.png"
        p.write_bytes(_make_png_bytes(mode))
        paths.append(p)

    rgb_shape = _load_image_as_rgb(paths[0]).size
    assert _load_image_as_rgb(paths[1]).size == rgb_shape
    assert _load_image_as_rgb(paths[2]).size == rgb_shape


def test_rgba_cmyk_collator_same_output_shape(tmp_path: Path):
    proc = _FakeImageProcessor()
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


def test_vqa_first_supervised_token_matches_answer(tmp_path: Path):
    proc = _FakeImageProcessor()
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
    proc = _FakeImageProcessor()
    proc.image_seq_length = 100
    proc.full_image_sequence = "old"
    apply_image_token_budget_to_processor(proc, 280)
    assert proc.image_seq_length == 280
    assert "old" not in proc.full_image_sequence
    assert proc.full_image_sequence.count("<img>") == 280


def test_image_collator_masks_padding_via_attention_mask_only(tmp_path: Path):
    """Padding is masked with attention_mask == 0 (not pad_id equality)."""
    proc = _FakeImageProcessor()
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
