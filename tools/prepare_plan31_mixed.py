"""Project sealed Plan 28 rows into the one-mode-per-row Plan 31 trainer CSV.

Only the new ignored output tree is written. Source prompts, targets, images,
and Plan 30 artifacts are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from gemma_tuner.models.common.plan31_input_modes import MODES, render_plan31_input


ROOT = Path(__file__).resolve().parents[1]
PD = ROOT.parent / "perfect-dictator"
SOURCE = ROOT / "data/datasets/tt-screenshot-plan28-sft-v1/full"
SOURCE_MANIFEST = SOURCE.parent / "manifest.json"
SCHEDULE = PD / "data/datasets/tt_screenshot_plan28_sft_v1/private/plan31-freeze-v4/epoch-1-modes.jsonl"
PANEL = PD / "data/datasets/tt_screenshot_plan28_sft_v1/private/plan31-freeze-v1/validation-60.jsonl"
OUTPUT = ROOT / "data/datasets/tt-screenshot-plan31-mixed-v4/full"
FIELDS = ("id", "owner_id", "image_path", "prompt", "response", "image_view_policy", "system_prompt", "input_mode")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"incomplete JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_once(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    candidate = temporary.read_bytes()
    if path.exists() and path.read_bytes() != candidate:
        temporary.unlink()
        raise ValueError(f"conflicting existing Plan 31 projection: {path}")
    temporary.replace(path)
    return sha(candidate)


def build(*, source: Path = SOURCE, source_manifest: Path = SOURCE_MANIFEST,
          schedule: Path = SCHEDULE, panel: Path = PANEL, output: Path = OUTPUT,
          verify_images: bool = True) -> dict:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    train_path, validation_path = source / "train.csv", source / "validation.csv"
    for split, path in (("train", train_path), ("validation", validation_path)):
        if sha_file(path) != manifest["file_sha256"][f"full/{split}.csv"]:
            raise ValueError(f"sealed Plan 29 {split} CSV hash mismatch")
    train, validation = csv_rows(train_path), csv_rows(validation_path)
    assigned, panel_rows = load_jsonl(schedule), load_jsonl(panel)
    if (len(train), len(validation), len(assigned), len(panel_rows)) != (2812, 252, 2812, 60):
        raise ValueError("Plan 31 source/schedule/panel count mismatch")
    if len({row["id"] for row in train + validation}) != 3064:
        raise ValueError("source row IDs are not disjoint")
    modes = Counter()
    projected_train = []
    for ordinal, (row, assignment) in enumerate(zip(train, assigned), start=1):
        if (assignment["source_ordinal"] != ordinal or row["id"] != assignment["id"]
                or assignment["mode"] not in MODES):
            raise ValueError("Plan 31 mode schedule is reordered or invalid")
        if row["image_view_policy"] != "global_plus_four_nonoverlapping_quadrants":
            raise ValueError("source view policy changed")
        if verify_images and sha_file(Path(row["image_path"])) != assignment["image_sha256"]:
            raise ValueError("Plan 31 screenshot hash mismatch")
        if sha((row["response"] + "\n").encode()) != assignment["target_sha256"]:
            raise ValueError("Plan 31 target hash mismatch")
        render_plan31_input(
            mode=assignment["mode"], full_prompt=row["prompt"],
            system_prompt=row["system_prompt"], full_views=[None] * 5,
        )
        modes[assignment["mode"]] += 1
        projected_train.append({**row, "input_mode": assignment["mode"]})
    if modes != {"full": 281, "no_ocr": 421, "no_quadrants": 422,
                 "no_system": 844, "image_instruction": 422, "image_only": 422}:
        raise ValueError("Plan 31 mode counts changed")

    projected_validation = []
    by_id = {row["id"]: row for row in validation}
    for row in validation:
        render_plan31_input(mode="full", full_prompt=row["prompt"],
                            system_prompt=row["system_prompt"], full_views=[None] * 5)
        projected_validation.append({**row, "input_mode": "full"})
    selected = []
    for record in panel_rows:
        row = by_id.get(record["id"])
        if row is None or (verify_images and sha_file(Path(row["image_path"])) != record["image_sha256"]):
            raise ValueError("Plan 31 panel row/image mismatch")
        selected.append(row)
    if len({row["id"] for row in selected}) != 60:
        raise ValueError("Plan 31 panel contains duplicate rows")

    output_hashes = {}
    schedule_copy = output / "epoch-1-modes.jsonl"
    schedule_copy.parent.mkdir(parents=True, exist_ok=True)
    schedule_bytes = schedule.read_bytes()
    if schedule_copy.exists() and schedule_copy.read_bytes() != schedule_bytes:
        raise ValueError("conflicting existing Plan 31 schedule copy")
    schedule_copy.write_bytes(schedule_bytes)
    output_hashes["epoch-1-modes.jsonl"] = sha(schedule_bytes)
    panel_copy = output / "validation-60.jsonl"
    panel_bytes = panel.read_bytes()
    if panel_copy.exists() and panel_copy.read_bytes() != panel_bytes:
        raise ValueError("conflicting existing Plan 31 panel copy")
    panel_copy.write_bytes(panel_bytes)
    output_hashes["validation-60.jsonl"] = sha(panel_bytes)
    output_hashes["train.csv"] = write_csv_once(output / "train.csv", projected_train)
    output_hashes["validation.csv"] = write_csv_once(output / "validation.csv", projected_validation)
    for mode in MODES:
        output_hashes[f"validation-panel-{mode}.csv"] = write_csv_once(
            output / f"validation-panel-{mode}.csv",
            [{**row, "input_mode": mode} for row in selected],
        )
    receipt = {
        "schema_version": "plan31_mixed_projection_v3",
        "train_rows": 2812,
        "validation_rows": 252,
        "panel_rows_per_mode": 60,
        "mode_counts": dict(sorted(modes.items())),
        "source_manifest_sha256": sha_file(source_manifest),
        "schedule_sha256": sha_file(schedule),
        "panel_sha256": sha_file(panel),
        "outputs_sha256": output_hashes,
        "test_rows_read": 0,
    }
    receipt_path = output / "projection-v3.receipt.json"
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists() and receipt_path.read_bytes() != receipt_bytes:
        raise ValueError("conflicting Plan 31 projection receipt")
    receipt_path.write_bytes(receipt_bytes)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(source=args.source, source_manifest=args.source_manifest,
                           schedule=args.schedule, panel=args.panel, output=args.output),
                     sort_keys=True))
