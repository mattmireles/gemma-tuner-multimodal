"""Exact Plan 31 projections of the frozen full screenshot input.

The accepted answer is intentionally not handled here: all modes supervise
the same response bytes. This module only changes the input messages/views.
"""

from __future__ import annotations

from typing import Any


MODES = (
    "full",
    "no_ocr",
    "no_quadrants",
    "no_system",
    "image_instruction",
    "image_only",
)
INSTRUCTION = "Return valid JSON only."
OCR_OPEN = "\n\n<first_pass_screenshot_ocr>\n"
OCR_CLOSE = "\n</first_pass_screenshot_ocr>"


def without_ocr(full_prompt: str) -> str:
    """Remove the sole trailing structured OCR block, failing on ABI drift."""
    if full_prompt.count(OCR_OPEN) != 1 or full_prompt.count(OCR_CLOSE) != 1:
        raise ValueError("Plan 31 requires exactly one canonical OCR block")
    before, after_open = full_prompt.split(OCR_OPEN, 1)
    _, after_close = after_open.split(OCR_CLOSE, 1)
    if after_close not in ("", "\n") or not before.rstrip().endswith(INSTRUCTION):
        raise ValueError("Plan 31 OCR block or user instruction moved")
    return before


def render_plan31_input(
    *,
    mode: str,
    full_prompt: str,
    system_prompt: str,
    full_views: list[Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return selected views and prompt messages, excluding the assistant target."""
    if mode not in MODES:
        raise ValueError(f"unknown Plan 31 input mode: {mode!r}")
    if len(full_views) != 5:
        raise ValueError("Plan 31 requires original screenshot plus four frozen views")
    if not system_prompt:
        raise ValueError("Plan 31 source system prompt is missing")
    base_prompt = without_ocr(full_prompt)
    if base_prompt.count(INSTRUCTION) != 1:
        raise ValueError("Plan 31 user instruction is not unique")

    views = full_views[:1] if mode in {"no_quadrants", "image_instruction", "image_only"} else full_views
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": view} for view in views
    ]
    if mode == "image_only":
        user_text = None
    elif mode == "image_instruction":
        user_text = INSTRUCTION
    elif mode in {"no_ocr"}:
        user_text = base_prompt
    else:
        user_text = full_prompt
    if user_text is not None:
        user_content.append({"type": "text", "text": user_text})

    messages: list[dict[str, Any]] = []
    if mode not in {"no_system", "image_instruction", "image_only"}:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_content})
    return views, messages
