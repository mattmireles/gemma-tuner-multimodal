#!/usr/bin/env python3
"""Build deterministic blinded Cursor work items from matched HF generations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PROMPT_SHA256 = "afb58a57173811636a44aa37b18a445c7790d0b338a291a887ae9887a6416cbb"
BLIND_SEED = "gemma4-e4b-full-r64-one-epoch-cursor-v1"
FORBIDDEN_FIELDS = frozenset(
    {"arm", "model", "checkpoint", "split", "target", "target_json", "competing_output"}
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def load_template(path: Path) -> str:
    if sha256_file(path) != PROMPT_SHA256:
        raise ValueError("evaluation prompt hash mismatch")
    return path.read_text(encoding="utf-8")


def render(template: str, row: dict[str, str], output: str) -> str:
    values = {
        "${system_instruction}": row["system_prompt"],
        "${user_message}": row["prompt"],
        "${output}": output,
    }
    remainder = template
    for placeholder, value in values.items():
        if template.count(placeholder) != 1:
            raise ValueError(f"expected exactly one {placeholder}")
        remainder = remainder.replace(placeholder, "")
        template = template.replace(placeholder, value)
    if "${" in remainder:
        raise ValueError("unresolved evaluation prompt placeholder")
    return template


def parse_generation(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("generation must be NAME=PATH")
    return name, Path(path).resolve()


def build(
    *,
    csv_path: Path,
    generations: list[tuple[str, Path]],
    template: str,
    count: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))[:count]
    if len(source) != count:
        raise ValueError("validation CSV has fewer rows than requested")
    ids = [row["id"] for row in source]
    by_id = {row["id"]: row for row in source}
    items: list[dict[str, str]] = []
    mapping: list[dict[str, str]] = []
    for candidate, ledger_path in generations:
        rows = read_jsonl(ledger_path)
        indexed = {str(row["example_id"]): row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(ids):
            raise ValueError(f"{candidate} generation IDs differ from frozen validation IDs")
        for example_id in ids:
            source_row = by_id[example_id]
            generated = indexed[example_id]
            image = Path(source_row["image_path"])
            if not image.is_absolute():
                image = (csv_path.parent / image).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            rendered = render(template, source_row, str(generated["candidate_output"]))
            rendered_sha = sha256_text(rendered)
            blind_id = sha256_text(f"{BLIND_SEED}:{candidate}:{example_id}:{rendered_sha}")[:24]
            identity = canonical({
                "blind_id": blind_id,
                "image_sha256": sha256_file(image),
                "prompt_sha256": PROMPT_SHA256,
                "rendered_prompt_sha256": rendered_sha,
            })
            item = {
                "work_item_hash": sha256_text(identity),
                "example_id": blind_id,
                "image_ref": str(image),
                "image_sha256": sha256_file(image),
                "rendered_prompt": rendered,
                "rendered_prompt_sha256": rendered_sha,
            }
            if FORBIDDEN_FIELDS.intersection(item):
                raise AssertionError("blind work item leaked candidate identity")
            items.append(item)
            mapping.append({
                "work_item_hash": item["work_item_hash"],
                "blind_id": blind_id,
                "candidate": candidate,
                "example_id": example_id,
                "candidate_sha256": str(generated["candidate_sha256"]),
            })
    order = {item["work_item_hash"]: sha256_text(f"{BLIND_SEED}:{item['work_item_hash']}") for item in items}
    items.sort(key=lambda item: order[item["work_item_hash"]])
    mapping.sort(key=lambda row: order[row["work_item_hash"]])
    if len({item["work_item_hash"] for item in items}) != len(items):
        raise ValueError("duplicate blinded work item")
    return items, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--generation", action="append", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--work-items", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    generations = [parse_generation(value) for value in args.generation]
    if len({name for name, _ in generations}) != len(generations):
        raise ValueError("duplicate candidate name")
    items, mapping = build(
        csv_path=args.csv.resolve(),
        generations=generations,
        template=load_template(args.prompt.resolve()),
        count=args.count,
    )
    write_jsonl(args.work_items.resolve(), items)
    write_jsonl(args.mapping.resolve(), mapping)
    print(canonical({
        "work_items": len(items),
        "work_items_sha256": sha256_file(args.work_items.resolve()),
        "mapping_sha256": sha256_file(args.mapping.resolve()),
    }))


if __name__ == "__main__":
    main()
