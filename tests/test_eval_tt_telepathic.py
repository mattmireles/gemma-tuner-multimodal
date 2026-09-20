"""Contracts for matched telepathic-context generation and blind work items."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

MODULE_PATH = Path(__file__).parents[1] / "tools" / "eval_tt_telepathic.py"
SPEC = importlib.util.spec_from_file_location("eval_tt_telepathic", MODULE_PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def source_row(example_id: str, image: Path, conditioned: bool = False) -> dict[str, str]:
    row = {
        "id": example_id,
        "image_path": str(image),
        "prompt": "user prompt",
        "response": '{"context_analysis":{}}',
        "image_view_policy": evaluation.VIEW_POLICY,
    }
    if conditioned:
        row["system_prompt"] = "cut-down intent prompt"
    return row


def test_image_views_are_global_then_exact_nonoverlapping_quadrants() -> None:
    image = Image.new("L", (5, 3))
    views = evaluation.build_image_views(image)
    assert [view.mode for view in views] == ["RGB"] * 5
    assert [view.size for view in views] == [(5, 3), (2, 1), (3, 1), (2, 2), (3, 2)]
    assert sum(width * height for width, height in [view.size for view in views[1:]]) == 15


def test_messages_differ_only_by_conditioned_system_role(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    compact = source_row("1", image)
    conditioned = source_row("1", image, conditioned=True)
    assert evaluation.build_messages(compact, "compact") == [
        {"role": "user", "content": "user prompt"}
    ]
    assert evaluation.build_messages(conditioned, "conditioned") == [
        {"role": "system", "content": "cut-down intent prompt"},
        {"role": "user", "content": "user prompt"},
    ]


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ('{"context_analysis":{}}', True),
        ('{"context_analysis":{},"extra":1}', False),
        ('```json\n{"context_analysis":{}}\n```', False),
        ('[]', False),
    ],
)
def test_candidate_json_contract_is_strict(text: str, valid: bool) -> None:
    assert evaluation.parse_candidate_json(text)[0] is valid


def test_generation_is_append_only_and_resumable(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    rows = [source_row("1", image), source_row("2", image)]
    ledger = tmp_path / "generation.jsonl"
    calls: list[str] = []

    def generator(row, messages):
        calls.append(row["id"])
        return {
            "text": '{"context_analysis":{}}', "elapsed_seconds": 1.0,
            "prompt_tokens": 10, "generation_tokens": 5, "peak_memory_gb": 2.0,
            "finish_reason": "stop",
        }

    first = evaluation.generate_rows(
        rows, arm="compact", ledger=ledger, settings_sha256="settings", generator=generator
    )
    second = evaluation.generate_rows(
        rows, arm="compact", ledger=ledger, settings_sha256="settings", generator=generator
    )
    assert len(first) == len(second) == 2
    assert calls == ["1", "2"]
    assert len(ledger.read_text().splitlines()) == 2


def test_blind_work_items_have_no_arm_model_target_or_reference_response(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    template = "S=${system_instruction}\nU=${user_message}\nO=${output}"
    generated = [{
        "example_id": "1", "image_ref": str(image),
        "image_sha256": evaluation.sha256_file(image),
        "system_instruction": "system", "user_message": "user",
        "candidate_output": '{"context_analysis":{}}',
    }]
    item = evaluation.build_blind_work_items(generated, template)[0]
    assert not evaluation.PROHIBITED_WORK_ITEM_FIELDS.intersection(item)
    assert "response" not in item
    assert item["rendered_prompt"] == 'S=system\nU=user\nO={"context_analysis":{}}'


def test_frozen_rows_follow_manifest_order_and_reject_compact_system_column(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "compact").mkdir(parents=True)
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    csv_path = staging / "compact" / "validation.csv"
    csv_path.write_text(
        "id,image_path,prompt,response,image_view_policy\n"
        f'2,{image},p,r,{evaluation.VIEW_POLICY}\n'
        f'1,{image},p,r,{evaluation.VIEW_POLICY}\n', encoding="utf-8"
    )
    manifest = {
        "validation_first_20_ids": ["1", "2"],
        "file_sha256": {"compact/validation.csv": evaluation.sha256_file(csv_path)},
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = evaluation.read_frozen_rows(
        arm="compact", staging=staging, manifest_path=manifest_path, count=2
    )
    assert [row["id"] for row in rows] == ["1", "2"]
