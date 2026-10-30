"""Shared contracts for Gemma 4 reference capture and benchmarking.

Tracked fixture manifests contain hashes and opaque references only.  The
corresponding request bodies and media stay below a caller-provided private
root so product screenshots, audio, prompts, and outputs never enter Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA = "gemma4-e4b-runtime-fixture-v1"
PRIVATE_SCHEMA = "gemma4-e4b-runtime-private-fixture-v1"
MODES = {"image_to_text", "audio_to_text", "text_to_text"}
STAGES = (
    "preprocess",
    "encoder",
    "handoff",
    "prefill",
    "first_token",
    "decode",
    "postprocess",
    "end_to_end",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_manifest_hash(rows: Iterable[dict[str, Any]]) -> str:
    return sha256_text("\n".join(canonical_json(row) for row in rows) + "\n")


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def validate_manifest_row(row: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "fixture_id",
        "mode",
        "private_ref",
        "request_sha256",
        "input_sha256",
        "expected",
        "generation",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"fixture missing fields: {missing}")
    if row["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported fixture schema: {row['schema_version']}")
    if row["mode"] not in MODES:
        raise ValueError(f"unsupported fixture mode: {row['mode']}")
    private_ref = Path(str(row["private_ref"]))
    if private_ref.is_absolute() or ".." in private_ref.parts:
        raise ValueError("private_ref must be a safe path relative to the private fixture root")
    _require_sha256(row["fixture_id"], "fixture_id")
    _require_sha256(row["request_sha256"], "request_sha256")
    input_hashes = row["input_sha256"]
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError("input_sha256 must be a non-empty object")
    for name, digest in input_hashes.items():
        _require_sha256(digest, f"input_sha256.{name}")
    expected = row["expected"]
    if not isinstance(expected, dict) or "output_sha256" not in expected:
        raise ValueError("expected.output_sha256 is required")
    _require_sha256(expected["output_sha256"], "expected.output_sha256")
    generation = row["generation"]
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be positive")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"fixture manifest is empty: {path}")
    fixture_ids: set[str] = set()
    for row in rows:
        validate_manifest_row(row)
        fixture_id = str(row["fixture_id"])
        if fixture_id in fixture_ids:
            raise ValueError(f"duplicate fixture_id: {fixture_id}")
        fixture_ids.add(fixture_id)
    return rows


def select_fixture(rows: list[dict[str, Any]], fixture_id: str | None) -> dict[str, Any]:
    if fixture_id is None:
        if len(rows) != 1:
            raise ValueError("--fixture-id is required when a manifest contains multiple fixtures")
        return rows[0]
    matches = [row for row in rows if row["fixture_id"] == fixture_id]
    if len(matches) != 1:
        raise ValueError(f"fixture_id not found exactly once: {fixture_id}")
    return matches[0]


def private_root_from(value: str | None) -> Path:
    raw = value or os.environ.get("GEMMA_E4B_FIXTURE_ROOT")
    if not raw:
        raise ValueError("set --private-root or GEMMA_E4B_FIXTURE_ROOT")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def load_private_fixture(row: dict[str, Any], private_root: Path) -> dict[str, Any]:
    path = (private_root / str(row["private_ref"])).resolve()
    try:
        path.relative_to(private_root)
    except ValueError as error:
        raise ValueError("private fixture escaped the configured root") from error
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PRIVATE_SCHEMA:
        raise ValueError(f"unsupported private fixture schema in {path}")
    if payload.get("fixture_id") != row["fixture_id"] or payload.get("mode") != row["mode"]:
        raise ValueError(f"private fixture identity mismatch in {path}")
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"private fixture request must be an object in {path}")
    if sha256_text(canonical_json(request)) != row["request_sha256"]:
        raise ValueError(f"private fixture request hash mismatch in {path}")
    media = payload.get("media", {})
    if not isinstance(media, dict):
        raise ValueError(f"private fixture media must be an object in {path}")
    expected_hashes = row["input_sha256"]
    if "request" in expected_hashes and expected_hashes["request"] != row["request_sha256"]:
        raise ValueError(f"input request hash mismatch in {path}")
    expected_media_hashes = {name: digest for name, digest in expected_hashes.items() if name != "request"}
    if set(media) != set(expected_media_hashes):
        raise ValueError(f"private fixture media inventory mismatch in {path}")
    resolved_media: dict[str, Path] = {}
    for name, relative in media.items():
        media_path = (path.parent / str(relative)).resolve()
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        if sha256_file(media_path) != expected_media_hashes[name]:
            raise ValueError(f"private fixture media hash mismatch for {name} in {path}")
        resolved_media[name] = media_path
    payload["_path"] = path
    payload["_resolved_media"] = resolved_media
    return payload


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty collection")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_stage_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for stage in STAGES:
        values = [float(row[stage]) for row in rows]
        summary[stage] = {
            "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95),
            "min_ms": min(values),
            "max_ms": max(values),
        }
    return summary


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_item in enumerate(reference, 1):
        current = [row_index]
        for column_index, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def normalized_error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    word_pattern = re.compile(r"[\w']+", re.UNICODE)
    reference_words = word_pattern.findall(reference.casefold())
    hypothesis_words = word_pattern.findall(hypothesis.casefold())
    reference_chars = list(" ".join(reference_words))
    hypothesis_chars = list(" ".join(hypothesis_words))
    return {
        "wer": _edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words)),
        "cer": _edit_distance(reference_chars, hypothesis_chars) / max(1, len(reference_chars)),
    }
