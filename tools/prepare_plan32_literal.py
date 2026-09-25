"""Build the ignored Plan 32 six-mode trainer bundle from frozen private inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from gemma_tuner.models.common.plan31_input_modes import MODES


ROOT = Path(__file__).resolve().parents[1]
PD = ROOT.parent / "perfect-dictator"
LITERAL = PD / "data/datasets/tt_screenshot_literal_sft_v1/private/v1"
FREEZE = PD / "data/datasets/tt_screenshot_literal_sft_v1/private/plan32-freeze-v1"
PLAN31 = ROOT / "data/datasets/tt-screenshot-plan31-mixed-v4/full"
PANEL = PD / "data/datasets/tt_screenshot_plan28_sft_v1/private/plan31-freeze-v1/validation-60.jsonl"
OUTPUT = ROOT / "data/datasets/tt-screenshot-plan32-literal-v1-deploy"
FIELDS = ("id", "owner_id", "source_ordinal", "image_path", "prompt", "response",
          "image_view_policy", "system_prompt", "input_mode")
VIEW_POLICY = "global_plus_four_nonoverlapping_quadrants"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[bytes, list[dict]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"incomplete JSONL: {path}")
    return raw, [json.loads(line) for line in raw.splitlines()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_once(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    raw = temporary.read_bytes()
    if path.exists() and path.read_bytes() != raw:
        temporary.unlink()
        raise ValueError(f"conflicting existing Plan 32 file: {path}")
    temporary.replace(path)
    return sha(raw)


def write_jsonl_once(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"conflicting existing Plan 32 file: {path}")
    if not path.exists():
        path.write_bytes(raw)
    return sha(raw)


def _project_rows(rows: list[dict], source_rows: dict[str, dict[str, str]], schedule: list[dict], *, check_image: bool) -> list[dict[str, str]]:
    if len(rows) != len(schedule):
        raise ValueError("Plan 32 row/schedule counts differ")
    projected = []
    for ordinal, (row, mode_record) in enumerate(zip(rows, schedule), start=1):
        identifier = str(row["id"])
        source = source_rows.get(identifier)
        if (source is None or mode_record["id"] != identifier
                or int(mode_record["source_ordinal"]) != ordinal
                or mode_record["epoch"] not in (1, 2)):
            raise ValueError("Plan 32 source rows or frozen schedule are reordered")
        if (str(row["owner_hash"]) != source["owner_id"]
                or str(row["image_sha256"]) != mode_record["image_sha256"]
                or sha((str(row["response"]) + "\n").encode()) != mode_record["target_sha256"]
                or sha(str(row["user_prompt"]).encode()) != mode_record["prompt_sha256"]
                or sha(str(row["system_prompt"]).encode()) != mode_record["system_sha256"]):
            raise ValueError("Plan 32 projected content differs from frozen source/schedule")
        source_image = Path(source["image_path"]).resolve(strict=True)
        image_path = ROOT / "data/datasets/tt-screenshot-plan31-mixed-v4-deploy/images" / (
            identifier + source_image.suffix.lower()
        )
        image_path = image_path.resolve(strict=True)
        if check_image and (sha_file(source_image) != str(row["image_sha256"])
                            or sha_file(image_path) != str(row["image_sha256"])):
            raise ValueError("Plan 32 image bytes differ from frozen image hash")
        mode = str(mode_record["mode"])
        if mode not in MODES:
            raise ValueError("unknown Plan 32 input mode")
        projected.append({
            "id": identifier,
            "owner_id": str(row["owner_hash"]),
            "source_ordinal": str(ordinal),
            "image_path": (
                f"../../tt-screenshot-plan31-mixed-v4-deploy/images/{identifier}{source_image.suffix.lower()}"
            ),
            "prompt": str(row["user_prompt"]),
            "response": str(row["response"]),
            "image_view_policy": VIEW_POLICY,
            "system_prompt": str(row["system_prompt"]),
            "input_mode": mode,
        })
    return projected


def build(*, check_images: bool = True) -> dict:
    manifest_path = LITERAL / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_bytes, train = read_jsonl(LITERAL / "train.jsonl")
    validation_bytes, validation = read_jsonl(LITERAL / "validation.jsonl")
    if sha(train_bytes) != manifest["files"]["train.jsonl"]["sha256"]:
        raise ValueError("literal train JSONL differs from manifest")
    if sha(validation_bytes) != manifest["files"]["validation.jsonl"]["sha256"]:
        raise ValueError("literal validation JSONL differs from manifest")
    if (len(train), len(validation)) != (2812, 252):
        raise ValueError("literal split counts changed")
    if any(row.get("split") != "train" for row in train) or any(row.get("split") != "validation" for row in validation):
        raise ValueError("literal split assignment changed")
    base_train = read_csv(PLAN31 / "train.csv")
    base_validation = read_csv(PLAN31 / "validation.csv")
    if (len(base_train), len(base_validation)) != (2812, 252):
        raise ValueError("Plan 31 image/path inventory changed")
    train_paths = {row["id"]: row for row in base_train}
    validation_paths = {row["id"]: row for row in base_validation}
    if len(train_paths) != 2812 or len(validation_paths) != 252:
        raise ValueError("Plan 31 source IDs are not unique")

    panel_bytes, panel = read_jsonl(PANEL)
    if len(panel) != 60:
        raise ValueError("frozen fresh panel must contain 60 rows")
    panel_ids = [str(row["id"]) for row in panel]
    if len(set(panel_ids)) != 60:
        raise ValueError("frozen fresh panel contains duplicate IDs")
    validation_by_id = {str(row["id"]): row for row in validation}
    if any(identifier not in validation_by_id for identifier in panel_ids):
        raise ValueError("frozen fresh panel is outside the literal validation split")

    outputs = {}
    for epoch in (1, 2):
        schedule_path = FREEZE / f"epoch-{epoch}-modes.jsonl"
        schedule_bytes, schedule = read_jsonl(schedule_path)
        receipt_path = FREEZE / f"epoch-{epoch}-modes.receipt.json"
        schedule_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if sha(schedule_bytes) != schedule_receipt["schedule_sha256"]:
            raise ValueError("frozen Plan 32 schedule hash changed")
        projected_train = _project_rows(train, train_paths, schedule, check_image=check_images)
        counts = Counter(row["input_mode"] for row in projected_train)
        expected_counts = {"full": 281, "no_system": 844, "no_ocr": 421,
                           "no_quadrants": 422, "image_instruction": 422, "image_only": 422}
        if dict(counts) != expected_counts:
            raise ValueError("Plan 32 mode mixture changed")

        val_schedule = [{"epoch": epoch, "source_ordinal": i + 1, "id": str(row["id"]),
                         "mode": "full", "image_sha256": str(row["image_sha256"]),
                         "target_sha256": sha((str(row["response"]) + "\n").encode()),
                         "prompt_sha256": sha(str(row["user_prompt"]).encode()),
                         "system_sha256": sha(str(row["system_prompt"]).encode())}
                        for i, row in enumerate(validation)]
        projected_validation = _project_rows(validation, validation_paths, val_schedule, check_image=check_images)
        by_validation_id = {row["id"]: row for row in projected_validation}
        selected = [by_validation_id[identifier] for identifier in panel_ids]
        output_dir = OUTPUT / f"epoch-{epoch}"
        hashes = {
            "train.csv": write_csv_once(output_dir / "train.csv", projected_train),
            "validation.csv": write_csv_once(output_dir / "validation.csv", projected_validation),
            "validation-image-sha256.jsonl": write_jsonl_once(
                output_dir / "validation-image-sha256.jsonl",
                [{"id": str(row["id"]), "image_sha256": str(row["image_sha256"])}
                 for row in validation],
            ),
        }
        for mode in MODES:
            hashes[f"validation-panel-{mode}.csv"] = write_csv_once(
                output_dir / f"validation-panel-{mode}.csv",
                [{**row, "input_mode": mode} for row in selected],
            )
        for name, data in (("epoch-modes.jsonl", schedule_bytes), ("validation-60.jsonl", panel_bytes)):
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"conflicting existing Plan 32 asset: {path}")
            if not path.exists():
                path.write_bytes(data)
            hashes[name] = sha(data)
        receipt = {
            "schema_version": "plan32_literal_projection_v1",
            "epoch": epoch,
            "train_rows": len(projected_train),
            "validation_rows": len(projected_validation),
            "fresh_panel_rows": len(selected),
            "mode_counts": dict(sorted(counts.items())),
            "literal_manifest_sha256": sha_file(manifest_path),
            "literal_train_sha256": sha(train_bytes),
            "literal_validation_sha256": sha(validation_bytes),
            "schedule_sha256": sha(schedule_bytes),
            "fresh_panel_sha256": sha(panel_bytes),
            "outputs_sha256": hashes,
            "test_rows_read": 0,
        }
        receipt_path = output_dir / "projection.receipt.json"
        receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
        if receipt_path.exists() and receipt_path.read_bytes() != receipt_bytes:
            raise ValueError(f"conflicting Plan 32 projection receipt: epoch {epoch}")
        if not receipt_path.exists():
            receipt_path.write_bytes(receipt_bytes)
        outputs[f"epoch-{epoch}"] = receipt
    return {"schema_version": "plan32_trainer_projection_v1", "epochs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-image-hash-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(check_images=not args.skip_image_hash_check), sort_keys=True))
