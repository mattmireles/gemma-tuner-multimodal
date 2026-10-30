"""Contracts for the generic MLX screenshot-judge runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "eval_screenshot_judge_mlx.py"
SPEC = importlib.util.spec_from_file_location("eval_screenshot_judge_mlx", MODULE_PATH)
assert SPEC and SPEC.loader
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


def item(index: int) -> dict[str, str]:
    return {
        "example_id": f"id-{index}",
        "image_ref": "/tmp/not-used.png",
        "image_sha256": "image-hash",
        "rendered_prompt": f"prompt-{index}",
        "rendered_prompt_sha256": judge.sha256_text(f"prompt-{index}"),
        "work_item_hash": f"hash-{index}",
    }


def generated(text: str) -> dict[str, object]:
    return {
        "text": text,
        "elapsed_seconds": 1.0,
        "prompt_tokens": 10,
        "generation_tokens": 2,
        "peak_memory_gb": 3.0,
        "finish_reason": "stop",
    }


def test_prompt_has_one_user_turn_and_no_system_role() -> None:
    assert judge.build_messages(item(0)) == [{"role": "user", "content": "prompt-0"}]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Excellent", "Excellent"),
        ("VeryGood\n", "VeryGood"),
        ("Good", "Good"),
        ("NeedsImprovement", "NeedsImprovement"),
        ("Fail - omitted the active message", "Fail"),
        ("Grade: Excellent", None),
        ("```Excellent```", None),
        ("Fail", None),
    ],
)
def test_grade_parser_is_strict(text: str, expected: str | None) -> None:
    assert judge.parse_grade(text) == expected


def test_thought_is_hashed_and_only_final_is_gradeable() -> None:
    final, thought_hash = judge.split_thinking_output(
        "<|channel>thought\nprivate reasoning<channel|>Excellent"
    )
    assert final == "Excellent"
    assert thought_hash == judge.sha256_text("private reasoning")
    missing_final, thought_hash = judge.split_thinking_output(
        "<|channel>thought\nprivate reasoning"
    )
    assert missing_final == ""
    assert thought_hash == judge.sha256_text("private reasoning")


def test_append_only_resume_skips_exact_existing_items(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    calls: list[str] = []

    def generator(work: dict[str, str]) -> dict[str, object]:
        calls.append(work["work_item_hash"])
        return generated("Excellent")

    items = [item(0), item(1)]
    first = judge.evaluate_items(items, ledger, "6bit", "settings", generator)
    second = judge.evaluate_items(items, ledger, "6bit", "settings", generator)
    assert len(first) == len(second) == 2
    assert calls == ["hash-0", "hash-1"]
    assert len(ledger.read_text().splitlines()) == 2


def test_resume_rejects_settings_mismatch(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "work_item_hash": "hash-0",
                "arm": "bf16",
                "settings_sha256": "old",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="arm/settings mismatch"):
        judge.evaluate_items([item(0)], ledger, "6bit", "new", lambda _: generated("Good"))


def test_resume_rejects_unexpected_work_item(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "work_item_hash": "extra",
                "arm": "6bit",
                "settings_sha256": "settings",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="unexpected work items"):
        judge.evaluate_items([item(0)], ledger, "6bit", "settings", lambda _: generated("Good"))
