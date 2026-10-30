"""Plan 31 exposures bind exact source order, mode, screenshot, and target."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gemma_tuner.utils.plan31_exposure_ledger import Plan31ExposureLedger


PROMPT = ('{"context":{}}\n\nReturn valid JSON only.'
          '\n\n<first_pass_screenshot_ocr>\n{"spans":[]}\n</first_pass_screenshot_ocr>\n')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path: Path):
    modes = ("full", "image_only", "no_ocr")
    features = []
    schedule = []
    for index, mode in enumerate(modes, start=1):
        image = tmp_path / f"image-{index}.png"
        image.write_bytes(f"image-{index}".encode())
        feature = {"id": f"row-{index}", "image_path": str(image),
                   "prompt": PROMPT, "system_prompt": "system", "response": '{"context_analysis":{}}',
                   "input_mode": mode}
        features.append(feature)
        schedule.append({"epoch": 1, "source_ordinal": index, "id": feature["id"], "mode": mode,
                         "image_sha256": sha(image.read_bytes()),
                         "target_sha256": sha((feature["response"] + "\n").encode())})
    schedule_path = tmp_path / "schedule.jsonl"
    schedule_path.write_text("".join(json.dumps(row) + "\n" for row in schedule))
    return features, schedule_path


def ledger(tmp_path: Path, features, schedule_path):
    return Plan31ExposureLedger(tmp_path / "exposures.jsonl", schedule_path=schedule_path,
                                train_rows=features, start_ordinal=1, end_ordinal=3)


def test_plan31_exact_segment_resume_and_content_hashes(tmp_path: Path):
    features, schedule_path = fixture(tmp_path)
    first = ledger(tmp_path, features, schedule_path)
    first.stage([features[0]])
    assert not first.path.exists()  # forward/backward has not succeeded yet
    first.commit()
    resumed = ledger(tmp_path, features, schedule_path)
    assert len(resumed.rows) == 1
    resumed.stage([features[1]])
    resumed.commit()
    resumed.stage([features[2]])
    resumed.commit()
    receipt = resumed.verify_complete()
    assert receipt["exposures"] == 3
    assert len({row["row_key_sha256"] for row in resumed.rows}) == 3
    assert all(len(row["rendered_messages_sha256"]) == 64 for row in resumed.rows)
    assert [row["views"] for row in resumed.rows] == [5, 1, 5]


def test_plan31_rejects_duplicate_reorder_conflict_and_missing(tmp_path: Path):
    features, schedule_path = fixture(tmp_path)
    current = ledger(tmp_path, features, schedule_path)
    with pytest.raises(RuntimeError, match="duplicated or reordered"):
        current.stage([features[1]])
    with pytest.raises(RuntimeError, match="prompt changed"):
        current.stage([{**features[0], "prompt": PROMPT + "changed"}])
    with pytest.raises(RuntimeError, match="target changed"):
        current.stage([{**features[0], "response": "changed"}])
    current.stage([features[0]])
    with pytest.raises(RuntimeError, match="duplicated or reordered"):
        current.stage([features[0]])
    with pytest.raises(RuntimeError, match="pending, missing"):
        current.verify_complete()
    current.commit()
    with pytest.raises(RuntimeError, match="pending, missing"):
        current.verify_complete()
    modified = list(features)
    modified[0] = {**modified[0], "input_mode": "no_ocr"}
    with pytest.raises(ValueError, match="mode differs"):
        ledger(tmp_path, modified, schedule_path)
    with pytest.raises(ValueError, match="conflicts with frozen"):
        Plan31ExposureLedger(tmp_path / "exposures.jsonl", schedule_path=schedule_path,
                             train_rows=features, start_ordinal=2, end_ordinal=3)


def test_plan32_epoch_two_schedule_is_bound_and_recorded(tmp_path: Path):
    features, schedule_path = fixture(tmp_path)
    schedule = [json.loads(line) for line in schedule_path.read_text().splitlines()]
    for row in schedule:
        row["epoch"] = 2
    schedule_path.write_text("".join(json.dumps(row) + "\n" for row in schedule))

    current = Plan31ExposureLedger(
        tmp_path / "epoch-two.jsonl", schedule_path=schedule_path,
        train_rows=features, start_ordinal=1, end_ordinal=3,
        expected_epoch=2, schema_version="plan32_training_exposure_v1",
    )
    current.stage(features)
    current.commit()
    assert [row["epoch"] for row in current.rows] == [2, 2, 2]
    assert all(row["schema_version"] == "plan32_training_exposure_v1" for row in current.rows)
    assert current.verify_complete()["schema_version"] == "plan32_exposure_verification_v1"
