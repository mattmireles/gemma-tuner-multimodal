"""The actual training collator must consume Plan 31's per-row mode."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from gemma_tuner.models.common.collators import DataCollatorGemmaImage
from gemma_tuner.models.gemma.family import GemmaFamily
from tests._fakes import FakeImageProcessor


PROMPT = (
    '{"context":{"application":"Chat"}}\n\nReturn valid JSON only.'
    '\n\n<first_pass_screenshot_ocr>\n{"spans":["OCR secret"]}'
    '\n</first_pass_screenshot_ocr>\n'
)
TARGET = '{"context_analysis":{}}'


class CapturingStrictProcessor(FakeImageProcessor):
    def __init__(self):
        super().__init__()
        self.last_messages = None
        self.last_images = None
        self.tokenizer.eos_token_id = 1
        original_encode = self.tokenizer.encode

        def encode(text, add_special_tokens=False):
            if text == "<end_of_turn>":
                return [1]
            return original_encode(text, add_special_tokens=add_special_tokens)

        self.tokenizer.encode = encode

    def apply_chat_template(self, messages_batch, **kwargs):
        self.last_messages = messages_batch
        return super().apply_chat_template(messages_batch, **kwargs)

    def __call__(self, *args, images=None, **kwargs):
        self.last_images = images
        encoded = super().__call__(*args, images=images, **kwargs)
        encoded["input_ids"][:, 7] = self.tokenizer.eos_token_id
        return encoded


@pytest.mark.parametrize(
    "mode,roles,views,text",
    [
        ("full", ["system", "user", "assistant"], 5, PROMPT),
        ("no_ocr", ["system", "user", "assistant"], 5, PROMPT.split("\n\n<first_pass_screenshot_ocr>", 1)[0]),
        ("no_quadrants", ["system", "user", "assistant"], 1, PROMPT),
        ("no_system", ["user", "assistant"], 5, PROMPT),
        ("image_instruction", ["user", "assistant"], 1, "Return valid JSON only."),
        ("image_only", ["user", "assistant"], 1, None),
    ],
)
def test_training_collator_exact_plan31_mode(tmp_path: Path, mode, roles, views, text):
    path = tmp_path / "screenshot.png"
    Image.new("RGB", (5, 3), "white").save(path)
    proc = CapturingStrictProcessor()
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="answer",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        prompt_column="question",
        system_prompt_column="system_prompt",
        input_mode_column="input_mode",
        image_view_policy="global_plus_four_nonoverlapping_quadrants",
        completion_only_logits=True,
        sub_mode="vqa",
    )
    encoded = collator(
        [{
            "id": "row-1", "image_path": str(path), "question": PROMPT,
            "system_prompt": "system text", "input_mode": mode, "answer": TARGET,
        }]
    )
    assert len(proc.last_images[0]) == views
    messages = proc.last_messages[0]
    assert [message["role"] for message in messages] == roles
    user = messages[-2]["content"]
    assert len([item for item in user if item["type"] == "image"]) == views
    actual_text = [item["text"] for item in user if item["type"] == "text"]
    assert actual_text == ([] if text is None else [text])
    assert messages[-1]["content"][0]["text"] == TARGET
    assert torch.equal(encoded["labels"][:, -2:], encoded["input_ids"][:, -2:]) is False
    assert (encoded["labels"] != -100).any()


def test_plan31_collator_rejects_missing_mode(tmp_path: Path):
    path = tmp_path / "screenshot.png"
    Image.new("RGB", (5, 3), "white").save(path)
    collator = DataCollatorGemmaImage(
        processor=CapturingStrictProcessor(), text_column="answer", family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path", prompt_column="question", system_prompt_column="system_prompt",
        input_mode_column="input_mode", image_view_policy="global_plus_four_nonoverlapping_quadrants",
        sub_mode="vqa",
    )
    with pytest.raises(ValueError, match="missing input fields"):
        collator([{"id": "row-1", "image_path": str(path), "question": PROMPT,
                   "system_prompt": "system", "answer": TARGET}])


def test_same_assistant_target_and_loss_mask_across_all_modes(tmp_path: Path):
    path = tmp_path / "screenshot.png"
    Image.new("RGB", (5, 3), "white").save(path)
    labels = []
    for mode in ("full", "no_ocr", "no_quadrants", "no_system", "image_instruction", "image_only"):
        collator = DataCollatorGemmaImage(
            processor=CapturingStrictProcessor(), text_column="answer", family=GemmaFamily.GEMMA_3N,
            image_path_column="image_path", prompt_column="question", system_prompt_column="system_prompt",
            input_mode_column="input_mode", image_view_policy="global_plus_four_nonoverlapping_quadrants",
            completion_only_logits=True, sub_mode="vqa",
        )
        encoded = collator([{"id": "row-1", "image_path": str(path), "question": PROMPT,
                             "system_prompt": "system", "input_mode": mode, "answer": TARGET}])
        labels.append(encoded["labels"])
    assert all(torch.equal(labels[0], other) for other in labels[1:])
