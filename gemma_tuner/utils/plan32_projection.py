"""Fail-closed verification of the private two-epoch Plan 32 projection."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

MODES = ("full", "no_ocr", "no_quadrants", "no_system", "image_instruction", "image_only")
COUNTS = {"full": 281, "no_system": 844, "no_ocr": 421,
          "no_quadrants": 422, "image_instruction": 422, "image_only": 422}
# Plan 32 (E4B, two epochs) and Plan 33 (E2B, three epochs) share this projection ABI.
PROJECTION_SCHEMAS = {"plan32_literal_projection_v1": 2, "plan33_literal_projection_v1": 3}
FIELDS = ("id", "owner_id", "source_ordinal", "image_path", "prompt", "response",
          "image_view_policy", "system_prompt", "input_mode")


def literal_projection_schema(profile_config) -> str:
    """Map a profile's frozen literal epoch count to its projection receipt schema."""
    epochs = int(profile_config.get("literal_epochs", 2))
    for schema, count in PROJECTION_SCHEMAS.items():
        if count == epochs:
            return schema
    raise ValueError("literal_epochs must be 2 (Plan 32) or 3 (Plan 33)")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"incomplete Plan 32 JSONL: {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Plan 32 CSV fields changed: {path.name}")
        return list(reader)


def _image(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def _target_contract(response: str) -> None:
    value = json.loads(response)
    if not isinstance(value, dict) or list(value) != ["context_analysis"]:
        raise ValueError("Plan 32 target must be a context_analysis JSON object")
    context = value["context_analysis"]
    if not isinstance(context, dict) or set(context) != {
        "type", "active_conversation", "technical_context"
    }:
        raise ValueError("Plan 32 target has unexpected top-level context fields")
    active = context["active_conversation"]
    technical = context["technical_context"]
    if not isinstance(active, dict) or set(active) != {"participants", "current_topic"}:
        raise ValueError("Plan 32 active_conversation fields changed")
    if not isinstance(technical, dict) or set(technical) != {"language", "unusual_terms"}:
        raise ValueError("Plan 32 technical_context fields changed")
    if not isinstance(context["type"], str) or not isinstance(technical["language"], str):
        raise ValueError("Plan 32 literal type/language fields must be strings")
    participants = active["participants"]
    topic = active["current_topic"]
    if not isinstance(participants, list) or not isinstance(topic, dict) or set(topic) != {"summary", "existing_text"}:
        raise ValueError("Plan 32 conversation fields changed")
    if not isinstance(topic["summary"], str) or not isinstance(topic["existing_text"], str):
        raise ValueError("Plan 32 summary/existing_text must be strings")
    for participant in participants:
        if (not isinstance(participant, dict)
                or set(participant) != {"name", "role", "recent_messages"}
                or not all(isinstance(participant[key], str) for key in ("name", "role"))
                or not isinstance(participant["recent_messages"], list)
                or not all(isinstance(message, str) for message in participant["recent_messages"])):
            raise ValueError("Plan 32 participant literal fields changed")
    if not isinstance(technical["unusual_terms"], list) or not all(
            isinstance(term, str) for term in technical["unusual_terms"]):
        raise ValueError("Plan 32 unusual_terms must be an array of strings")


def _verify_validation_image_hashes(
    root: Path, validation: list[dict[str, str]], image_hash_by_id: dict[str, str]
) -> None:
    for row in validation:
        image = _image(root, row["image_path"])
        if sha_file(image) != image_hash_by_id[row["id"]]:
            raise ValueError("Plan 32 validation screenshot differs from frozen source hash")


def verify_projection(
    directory: str | Path, *, expected_receipt_sha256: str, epoch: int,
    schema_version: str = "plan32_literal_projection_v1",
) -> dict:
    root = Path(directory).resolve(strict=True)
    receipt_path = root / "projection.receipt.json"
    if sha_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("Plan 32 projection receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if schema_version not in PROJECTION_SCHEMAS:
        raise ValueError("unknown literal projection schema")
    if (receipt.get("schema_version") != schema_version
            or int(receipt.get("epoch", 0)) != epoch
            or receipt.get("train_rows") != 2812
            or receipt.get("validation_rows") != 252
            or receipt.get("fresh_panel_rows") != 60
            or receipt.get("test_rows_read") != 0):
        raise ValueError("Plan 32 projection receipt contract changed")
    if not 1 <= epoch <= PROJECTION_SCHEMAS[schema_version]:
        raise ValueError("literal projection epoch is outside its frozen lineage")
    outputs = receipt.get("outputs_sha256", {})
    expected_names = {
        "train.csv", "validation.csv", "validation-image-sha256.jsonl",
        "epoch-modes.jsonl", "validation-60.jsonl",
    }
    expected_names.update(f"validation-panel-{mode}.csv" for mode in MODES)
    if set(outputs) != expected_names:
        raise ValueError("Plan 32 projection file set changed")
    for name, digest in outputs.items():
        if sha_file(root / name) != digest:
            raise ValueError(f"Plan 32 projection file changed: {name}")
    if (outputs["epoch-modes.jsonl"] != receipt["schedule_sha256"]
            or outputs["validation-60.jsonl"] != receipt["fresh_panel_sha256"]):
        raise ValueError("Plan 32 schedule or fresh panel identity changed")

    schedule = read_jsonl(root / "epoch-modes.jsonl")
    schedule_hash = sha_file(root / "epoch-modes.jsonl")
    train = read_csv(root / "train.csv")
    if len(schedule) != 2812 or len(train) != 2812 or Counter(row["input_mode"] for row in train) != COUNTS:
        raise ValueError("Plan 32 training schedule or mode counts changed")
    train_owners: set[str] = set()
    train_images: set[str] = set()
    for ordinal, (frozen, row) in enumerate(zip(schedule, train), start=1):
        image = _image(root, row["image_path"])
        if (int(frozen["epoch"]) != epoch or int(frozen["source_ordinal"]) != ordinal
                or frozen["id"] != row["id"] or int(row["source_ordinal"]) != ordinal
                or frozen["mode"] != row["input_mode"]
                or sha(str(row["prompt"]).encode()) != frozen["prompt_sha256"]
                or sha(str(row["system_prompt"]).encode()) != frozen["system_sha256"]
                or sha((row["response"] + "\n").encode()) != frozen["target_sha256"]
                or sha_file(image) != frozen["image_sha256"]):
            raise ValueError("Plan 32 training row differs from its frozen schedule")
        train_owners.add(row["owner_id"])
        train_images.add(frozen["image_sha256"])
        _target_contract(row["response"])

    validation = read_csv(root / "validation.csv")
    validation_images = read_jsonl(root / "validation-image-sha256.jsonl")
    panel = read_jsonl(root / "validation-60.jsonl")
    if (len(validation) != 252 or len(validation_images) != 252
            or len({row["id"] for row in validation_images}) != 252
            or len(panel) != 60 or len({row["id"] for row in panel}) != 60):
        raise ValueError("Plan 32 validation or frozen panel count changed")
    validation_by_id = {row["id"]: row for row in validation}
    if len(validation_by_id) != 252:
        raise ValueError("Plan 32 validation IDs are not unique")
    image_hash_by_id = {row["id"]: row["image_sha256"] for row in validation_images}
    if set(image_hash_by_id) != set(validation_by_id):
        raise ValueError("Plan 32 validation image identities differ from validation rows")
    panel_ids = [str(row["id"]) for row in panel]
    for identifier in panel_ids:
        if identifier not in validation_by_id:
            raise ValueError("Plan 32 fresh panel is outside validation split")
    _verify_validation_image_hashes(root, validation, image_hash_by_id)
    for row in validation:
        _target_contract(row["response"])
    if set(train_images) & {sha_file(_image(root, row["image_path"])) for row in validation}:
        raise ValueError("Plan 32 train/validation images overlap")
    validation_owners = {row["owner_id"] for row in validation}
    if train_owners & validation_owners:
        raise ValueError("Plan 32 train/validation owners overlap")

    for mode in MODES:
        panel_rows = read_csv(root / f"validation-panel-{mode}.csv")
        if len(panel_rows) != 60 or [row["id"] for row in panel_rows] != panel_ids:
            raise ValueError(f"Plan 32 {mode} validation panel changed")
        for row, identifier in zip(panel_rows, panel_ids):
            base = validation_by_id[identifier]
            if (row["input_mode"] != mode or row["prompt"] != base["prompt"]
                    or row["system_prompt"] != base["system_prompt"]
                    or row["response"] != base["response"]
                    or sha_file(_image(root, row["image_path"])) != image_hash_by_id[identifier]):
                raise ValueError(f"Plan 32 {mode} panel changed retained input or target")
    return receipt
