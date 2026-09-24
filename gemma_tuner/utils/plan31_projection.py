"""Fail-closed verification of the private Plan 31 dataset projection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_projection(directory: str | Path, expected_receipt_sha256: str) -> dict:
    root = Path(directory)
    receipt_path = root / "projection-v3.receipt.json"
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("Plan 31 projection receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema_version") != "plan31_mixed_projection_v3"
            or receipt.get("train_rows") != 2812
            or receipt.get("validation_rows") != 252
            or receipt.get("panel_rows_per_mode") != 60
            or receipt.get("test_rows_read") != 0):
        raise ValueError("Plan 31 projection receipt contract changed")
    outputs = receipt.get("outputs_sha256", {})
    expected_names = {"train.csv", "validation.csv", "epoch-1-modes.jsonl", "validation-60.jsonl"}
    expected_names.update(f"validation-panel-{mode}.csv" for mode in
                          ("full", "no_ocr", "no_quadrants", "no_system", "image_instruction", "image_only"))
    if set(outputs) != expected_names:
        raise ValueError("Plan 31 projection file set changed")
    for name, digest in outputs.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Plan 31 projection file changed: {name}")
    if (outputs["epoch-1-modes.jsonl"] != receipt["schedule_sha256"]
            or outputs["validation-60.jsonl"] != receipt["panel_sha256"]):
        raise ValueError("Plan 31 schedule or panel identity changed")
    panel = [json.loads(line) for line in (root / "validation-60.jsonl").read_text().splitlines()]
    with (root / "validation-panel-full.csv").open(newline="", encoding="utf-8") as handle:
        panel_csv = list(csv.DictReader(handle))
    if len(panel) != len(panel_csv) or len({row["id"] for row in panel}) != 60:
        raise ValueError("Plan 31 frozen validation panel is incomplete or duplicated")
    for frozen, projected in zip(panel, panel_csv):
        if frozen["id"] != projected["id"] or frozen["split"] != "validation":
            raise ValueError("Plan 31 panel row order or split changed")
        image_path = Path(projected["image_path"])
        if not image_path.is_absolute():
            image_path = root / image_path
        if sha256_file(image_path) != frozen["image_sha256"]:
            raise ValueError("Plan 31 panel screenshot changed")
    return receipt
