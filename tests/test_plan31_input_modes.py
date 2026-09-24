"""Plan 31 input omission must remove tokens, not add sentinels."""

import pytest

from gemma_tuner.models.common.plan31_input_modes import (
    INSTRUCTION,
    MODES,
    render_plan31_input,
)


PROMPT = (
    '{"context":{"application":"Chat"}}\n\n'
    + INSTRUCTION
    + '\n\n<first_pass_screenshot_ocr>\n{"spans":["private OCR"]}'
    + '\n</first_pass_screenshot_ocr>\n'
)
VIEWS = [object() for _ in range(5)]


@pytest.mark.parametrize("mode", MODES)
def test_exact_mode_projection(mode):
    views, messages = render_plan31_input(
        mode=mode, full_prompt=PROMPT, system_prompt="full system", full_views=VIEWS
    )
    assert views[0] is VIEWS[0]
    assert len(views) == (1 if mode in {"no_quadrants", "image_instruction", "image_only"} else 5)
    assert [m["role"] for m in messages] == (
        ["user"] if mode in {"no_system", "image_instruction", "image_only"} else ["system", "user"]
    )
    content = messages[-1]["content"]
    assert [item["type"] for item in content] == ["image"] * len(views) + (
        [] if mode == "image_only" else ["text"]
    )
    if mode == "image_only":
        assert len(messages) == 1
        assert len(content) == 1
    else:
        text = content[-1]["text"]
        if mode == "image_instruction":
            assert text == INSTRUCTION
        elif mode == "no_ocr":
            assert text == PROMPT.split("\n\n<first_pass_screenshot_ocr>", 1)[0]
        else:
            assert text == PROMPT
        assert ("private OCR" in text) == (mode in {"full", "no_quadrants", "no_system"})


def test_full_mode_is_original_content_without_rewriting():
    views, messages = render_plan31_input(
        mode="full", full_prompt=PROMPT, system_prompt="full system", full_views=VIEWS
    )
    assert views == VIEWS
    assert messages[0]["content"][0]["text"] == "full system"
    assert messages[1]["content"][-1]["text"] == PROMPT


@pytest.mark.parametrize("prompt", [PROMPT.replace("</first_pass_screenshot_ocr>", ""), PROMPT + "extra"])
def test_bad_ocr_boundary_fails_closed(prompt):
    with pytest.raises(ValueError):
        render_plan31_input(mode="no_ocr", full_prompt=prompt, system_prompt="system", full_views=VIEWS)


def test_missing_original_or_unknown_mode_fails_closed():
    with pytest.raises(ValueError):
        render_plan31_input(mode="image_only", full_prompt=PROMPT, system_prompt="system", full_views=[])
    with pytest.raises(ValueError):
        render_plan31_input(mode="wrong", full_prompt=PROMPT, system_prompt="system", full_views=VIEWS)
