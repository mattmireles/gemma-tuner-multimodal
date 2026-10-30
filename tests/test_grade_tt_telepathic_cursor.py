from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "grade_tt_telepathic_cursor.py"
SPEC = importlib.util.spec_from_file_location("grade_tt_telepathic_cursor", MODULE)
assert SPEC and SPEC.loader
grading = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grading)


def test_pinned_cursor_selectors() -> None:
    assert grading.JUDGES == {
        "grok": "cursor-grok-4.6-medium",
        "composer": "composer-2.5[fast=false]",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Excellent", ("Excellent", None)),
        ("VeryGood\n", ("VeryGood", None)),
        ("Fail", ("Fail", None)),
        ("Fail - speaker attribution is wrong", ("Fail", "speaker attribution is wrong")),
    ],
)
def test_parse_grade(raw: str, expected: tuple[str, str | None]) -> None:
    assert grading.parse_grade(raw) == expected


def test_parse_grade_rejects_explanation_for_non_fail() -> None:
    with pytest.raises(ValueError, match="malformed"):
        grading.parse_grade("Excellent - looks good")


def test_dispatch_is_resumable_and_binds_selector(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    prompt = "grade this"
    item = {
        "work_item_hash": "a" * 64,
        "example_id": "blind-1",
        "image_ref": str(image),
        "rendered_prompt": prompt,
        "rendered_prompt_sha256": grading.sha256_text(prompt),
    }
    calls = []

    def runner(row, model, agent):  # noqa: ANN001
        calls.append((row, model, agent))
        return {"response": "Good", "latency_seconds": 1.25}

    ledger = tmp_path / "grades.jsonl"
    first = grading.dispatch(
        [item], ledger, judge="grok", workers=1, agent=tmp_path / "agent", runner=runner
    )
    second = grading.dispatch(
        [item], ledger, judge="grok", workers=1, agent=tmp_path / "agent", runner=runner
    )
    assert first["new"] == 1
    assert second["new"] == 0
    assert len(calls) == 1
    row = json.loads(ledger.read_text())
    assert row["model_selector"] == "cursor-grok-4.6-medium"
    assert row["grade"] == "Good"


def test_read_items_rejects_identity_leak(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps({"work_item_hash": "x", "arm": "stock"}) + "\n")
    with pytest.raises(ValueError, match="leaked"):
        grading.read_items(path)
