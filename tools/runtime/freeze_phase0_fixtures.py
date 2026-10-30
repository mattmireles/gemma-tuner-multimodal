#!/usr/bin/env python3
"""Freeze redacted Phase 0 manifests and a local ignored private fixture root.

The source datasets are explicitly supplied by the operator.  Tracked output
contains only hashes, opaque fixture IDs, generation settings, and quality
labels.  Prompt bodies, expected text, and media are copied into the private
root and must remain ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from reference_common import (
    MANIFEST_SCHEMA,
    PRIVATE_SCHEMA,
    canonical_json,
    sha256_file,
    sha256_text,
)

AUDIO_PROMPT = "Please transcribe this audio."


def _rank(value: str) -> str:
    return sha256_text(f"gemma4-e4b-runtime-phase0:{value}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _source_path(value: str, source: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _save_image_views(source: Path, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        mid_x, mid_y = width // 2, height // 2
        views = {
            "image_0": image,
            "image_1": image.crop((0, 0, mid_x, mid_y)),
            "image_2": image.crop((mid_x, 0, width, mid_y)),
            "image_3": image.crop((0, mid_y, mid_x, height)),
            "image_4": image.crop((mid_x, mid_y, width, height)),
        }
        paths: dict[str, Path] = {}
        for name, view in views.items():
            path = destination / f"{name}.png"
            view.save(path, format="PNG", optimize=False)
            paths[name] = path
        return paths


def _image_source_id(row: dict[str, Any]) -> str | None:
    value = row.get("source_hash") or row.get("example_id")
    return str(value) if value else None


def _image_expected_output(row: dict[str, Any]) -> str | None:
    value = row.get("response") if row.get("response") is not None else row.get("gold_response")
    return str(value) if value else None


def freeze_image(source: Path, count: int, private_root: Path, manifest_root: Path) -> list[dict[str, Any]]:
    candidates = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"image_path", "system_prompt", "user_prompt"}
    candidates = [
        row
        for row in candidates
        if required <= set(row)
        and _image_source_id(row)
        and _image_expected_output(row)
        and _source_path(row["image_path"], source).is_file()
    ]
    selected = sorted(candidates, key=lambda row: _rank(_image_source_id(row) or ""))[:count]
    if len(selected) != count:
        raise ValueError(f"requested {count} image fixtures but found {len(selected)} eligible rows")
    rows: list[dict[str, Any]] = []
    for source_row in selected:
        source_id = _image_source_id(source_row)
        expected_output = _image_expected_output(source_row)
        assert source_id is not None and expected_output is not None
        fixture_id = sha256_text(f"image_to_text:{source_id}")
        directory = private_root / "image" / fixture_id
        media_paths = _save_image_views(_source_path(source_row["image_path"], source), directory)
        request = {
            "messages": [
                {"role": "system", "content": source_row["system_prompt"]},
                {"role": "user", "content": source_row["user_prompt"]},
            ]
        }
        private_ref = Path("image") / fixture_id / "fixture.json"
        _write_json(
            private_root / private_ref,
            {
                "schema_version": PRIVATE_SCHEMA,
                "fixture_id": fixture_id,
                "mode": "image_to_text",
                "request": request,
                "media": {name: path.name for name, path in media_paths.items()},
                "expected_output": expected_output,
            },
        )
        rows.append(
            {
                "schema_version": MANIFEST_SCHEMA,
                "fixture_id": fixture_id,
                "mode": "image_to_text",
                "private_ref": str(private_ref),
                "source": {
                    "dataset": "tt_screenshot_plan29_sft_v1_validation",
                    "record_sha256": sha256_text(canonical_json(source_row)),
                },
                "request_sha256": sha256_text(canonical_json(request)),
                "input_sha256": {name: sha256_file(path) for name, path in media_paths.items()},
                "expected": {
                    "output_sha256": sha256_text(expected_output),
                    "quality_floor": "dual_judge_gold",
                },
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "thinking": False,
                    "max_new_tokens": 8192,
                    "views": 5,
                },
            }
        )
    _write_manifest(manifest_root / "image" / "manifest.jsonl", rows)
    return rows


def freeze_audio(source: Path, count: int, private_root: Path, manifest_root: Path) -> list[dict[str, Any]]:
    with source.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    candidates = [
        row
        for row in candidates
        if row.get("id")
        and row.get("audio_path")
        and row.get("text_perfect")
        and _source_path(row["audio_path"], source).is_file()
    ]
    selected = sorted(candidates, key=lambda row: _rank(str(row["id"])))[:count]
    if len(selected) != count:
        raise ValueError(f"requested {count} audio fixtures but found {len(selected)} eligible rows")
    rows: list[dict[str, Any]] = []
    for source_row in selected:
        source_id = str(source_row["id"])
        fixture_id = sha256_text(f"audio_to_text:{source_id}")
        directory = private_root / "audio" / fixture_id
        directory.mkdir(parents=True, exist_ok=True)
        source_audio = _source_path(source_row["audio_path"], source)
        destination_audio = directory / f"audio{source_audio.suffix.lower()}"
        shutil.copy2(source_audio, destination_audio)
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": AUDIO_PROMPT},
                    ],
                }
            ],
            "sampling_rate": 16000,
        }
        private_ref = Path("audio") / fixture_id / "fixture.json"
        _write_json(
            private_root / private_ref,
            {
                "schema_version": PRIVATE_SCHEMA,
                "fixture_id": fixture_id,
                "mode": "audio_to_text",
                "request": request,
                "media": {"audio": destination_audio.name},
                "expected_output": source_row["text_perfect"],
            },
        )
        rows.append(
            {
                "schema_version": MANIFEST_SCHEMA,
                "fixture_id": fixture_id,
                "mode": "audio_to_text",
                "private_ref": str(private_ref),
                "source": {
                    "dataset": "tt_whisper_gold_smoke_prompt_v3_nodraft",
                    "record_sha256": sha256_text(canonical_json(source_row)),
                },
                "request_sha256": sha256_text(canonical_json(request)),
                "input_sha256": {"audio": sha256_file(destination_audio)},
                "expected": {
                    "output_sha256": sha256_text(str(source_row["text_perfect"])),
                    "quality_floor": "human_approved_transcript",
                },
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "thinking": False,
                    "max_new_tokens": 256,
                },
            }
        )
    _write_manifest(manifest_root / "audio" / "manifest.jsonl", rows)
    return rows


def freeze_text(source: Path, count: int, private_root: Path, manifest_root: Path) -> list[dict[str, Any]]:
    required = {"id", "request_json", "expected_output"}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"text fixture source is missing columns: {missing}")
        candidates = [row for row in reader if all(row.get(field) for field in required)]
    selected = sorted(candidates, key=lambda row: _rank(str(row["id"])))[:count]
    if len(selected) != count:
        raise ValueError(f"requested {count} text fixtures but found {len(selected)} eligible rows")
    rows: list[dict[str, Any]] = []
    for source_row in selected:
        fixture_id = sha256_text(f"text_to_text:{source_row['id']}")
        try:
            request = json.loads(source_row["request_json"])
        except json.JSONDecodeError as error:
            raise ValueError(f"text fixture {source_row['id']} has invalid request_json") from error
        if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
            raise ValueError(f"text fixture {source_row['id']} request_json must contain a messages array")
        if not request["messages"]:
            raise ValueError(f"text fixture {source_row['id']} request_json messages must not be empty")
        for message in request["messages"]:
            if not isinstance(message, dict) or not {"role", "content"} <= set(message):
                raise ValueError(f"text fixture {source_row['id']} request_json messages require role and content")
            if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                raise ValueError(f"text fixture {source_row['id']} request_json message role/content must be strings")
        expected_output = source_row["expected_output"]
        private_ref = Path("text") / fixture_id / "fixture.json"
        _write_json(
            private_root / private_ref,
            {
                "schema_version": PRIVATE_SCHEMA,
                "fixture_id": fixture_id,
                "mode": "text_to_text",
                "request": request,
                "media": {},
                "expected_output": expected_output,
            },
        )
        rows.append(
            {
                "schema_version": MANIFEST_SCHEMA,
                "fixture_id": fixture_id,
                "mode": "text_to_text",
                "private_ref": str(private_ref),
                "source": {
                    "dataset": "product_rewriter_frozen",
                    "record_sha256": sha256_text(canonical_json(source_row)),
                },
                "request_sha256": sha256_text(canonical_json(request)),
                "input_sha256": {"request": sha256_text(canonical_json(request))},
                "expected": {
                    "output_sha256": sha256_text(expected_output),
                    "quality_floor": "human_approved_rewrite",
                },
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "thinking": False,
                    "max_new_tokens": 512,
                },
            }
        )
    _write_manifest(manifest_root / "text" / "manifest.jsonl", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-jsonl", type=Path)
    parser.add_argument("--audio-csv", type=Path)
    parser.add_argument("--text-csv", type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--private-root", type=Path, default=Path("tests/runtime/private"))
    parser.add_argument("--manifest-root", type=Path, default=Path("tests/runtime/fixtures"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not any((args.image_jsonl, args.audio_csv, args.text_csv)):
        raise ValueError("provide at least one source")
    counts: dict[str, int] = {}
    if args.image_jsonl:
        counts["image_to_text"] = len(freeze_image(args.image_jsonl, args.count, args.private_root, args.manifest_root))
    if args.audio_csv:
        counts["audio_to_text"] = len(freeze_audio(args.audio_csv, args.count, args.private_root, args.manifest_root))
    if args.text_csv:
        counts["text_to_text"] = len(freeze_text(args.text_csv, args.count, args.private_root, args.manifest_root))
    print(canonical_json({"frozen": counts, "private_root": str(args.private_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
