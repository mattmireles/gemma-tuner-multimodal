"""Plan 31 projection is content-bound, deterministic, and rejects conflicts."""

import csv
import json
from pathlib import Path

import pytest

from tools.prepare_plan31_mixed import FIELDS, build, sha, sha_file
from gemma_tuner.utils.plan31_projection import verify_projection


PROMPT = (
    '{"context":{"application":"Chat"}}\n\nReturn valid JSON only.'
    '\n\n<first_pass_screenshot_ocr>\n{"spans":[]}'
    '\n</first_pass_screenshot_ocr>\n'
)
TARGET = '{"context_analysis":{}}'
COUNTS = {"full": 281, "no_ocr": 421, "no_quadrants": 422,
          "no_system": 844, "image_instruction": 422, "image_only": 422}


def fixture(tmp_path: Path):
    source = tmp_path / "source/full"
    source.mkdir(parents=True)
    modes = [mode for mode, count in COUNTS.items() for _ in range(count)]
    train = []
    schedule = []
    for i, mode in enumerate(modes):
        row = {"id": f"train-{i}", "owner_id": f"owner-{i}",
               "image_path": f"/private/train-{i}.png", "prompt": PROMPT,
               "response": TARGET, "image_view_policy": "global_plus_four_nonoverlapping_quadrants",
               "system_prompt": "system"}
        train.append(row)
        schedule.append({"source_ordinal": i + 1, "id": row["id"], "mode": mode,
                         "image_sha256": "image", "target_sha256": sha((TARGET + "\n").encode())})
    validation = [
        {**train[0], "id": f"validation-{i}", "owner_id": f"validation-owner-{i}",
         "image_path": f"/private/validation-{i}.png"}
        for i in range(252)
    ]
    for split, rows in (("train", train), ("validation", validation)):
        with (source / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS[:-1])
            writer.writeheader()
            writer.writerows(rows)
    manifest = source.parent / "manifest.json"
    manifest.write_text(json.dumps({"file_sha256": {
        f"full/{split}.csv": sha_file(source / f"{split}.csv")
        for split in ("train", "validation")
    }}))
    schedule_path = tmp_path / "schedule.jsonl"
    schedule_path.write_text("".join(json.dumps(row) + "\n" for row in schedule))
    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text("".join(json.dumps({"id": row["id"], "image_sha256": "image"}) + "\n"
                                  for row in validation[:60]))
    return source, manifest, schedule_path, panel_path, tmp_path / "output"


def test_build_is_idempotent_and_keeps_targets_unchanged(tmp_path):
    source, manifest, schedule, panel, output = fixture(tmp_path)
    args = dict(source=source, source_manifest=manifest, schedule=schedule,
                panel=panel, output=output, verify_images=False)
    first = build(**args)
    assert build(**args) == first
    assert first["mode_counts"] == COUNTS
    with (output / "train.csv").open(newline="") as handle:
        projected = list(csv.DictReader(handle))
    assert len(projected) == 2812
    assert [row["response"] for row in projected] == [TARGET] * 2812
    assert [row["input_mode"] for row in projected] == [
        mode for mode, count in COUNTS.items() for _ in range(count)
    ]


def test_reordered_schedule_fails_closed(tmp_path):
    source, manifest, schedule, panel, output = fixture(tmp_path)
    lines = schedule.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    schedule.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="reordered"):
        build(source=source, source_manifest=manifest, schedule=schedule,
              panel=panel, output=output, verify_images=False)


def test_changed_target_fails_closed(tmp_path):
    source, manifest, schedule, panel, output = fixture(tmp_path)
    lines = schedule.read_text().splitlines()
    first = json.loads(lines[0])
    first["target_sha256"] = "0" * 64
    lines[0] = json.dumps(first)
    schedule.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="target hash mismatch"):
        build(source=source, source_manifest=manifest, schedule=schedule,
              panel=panel, output=output, verify_images=False)


def test_projection_receipt_rejects_changed_panel_pixels_and_csv(tmp_path):
    source, manifest, schedule, panel, output = fixture(tmp_path)
    validation_path = source / "validation.csv"
    with validation_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    frozen = []
    for index, row in enumerate(rows[:60]):
        image = tmp_path / f"panel-{index}.png"
        image.write_bytes(f"pixels-{index}".encode())
        row["image_path"] = str(image)
        frozen.append({"id": row["id"], "image_sha256": sha_file(image), "split": "validation"})
    with validation_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS[:-1])
        writer.writeheader()
        writer.writerows(rows)
    manifest_json = json.loads(manifest.read_text())
    manifest_json["file_sha256"]["full/validation.csv"] = sha_file(validation_path)
    manifest.write_text(json.dumps(manifest_json))
    panel.write_text("".join(json.dumps(row) + "\n" for row in frozen))
    build(source=source, source_manifest=manifest, schedule=schedule,
          panel=panel, output=output, verify_images=False)
    receipt_hash = sha_file(output / "projection-v3.receipt.json")
    assert verify_projection(output, receipt_hash)["panel_rows_per_mode"] == 60
    (tmp_path / "panel-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="panel screenshot changed"):
        verify_projection(output, receipt_hash)
    (tmp_path / "panel-0.png").write_bytes(b"pixels-0")
    with (output / "validation-panel-full.csv").open("a") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="projection file changed"):
        verify_projection(output, receipt_hash)
