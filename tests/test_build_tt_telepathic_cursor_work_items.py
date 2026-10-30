from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).parents[1] / "tools" / "build_tt_telepathic_cursor_work_items.py"
SPEC = importlib.util.spec_from_file_location("build_tt_cursor", MODULE)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_build_randomizes_and_blinds_candidates(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    csv_path = tmp_path / "validation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "image_path", "system_prompt", "prompt"]
        )
        writer.writeheader()
        writer.writerow({
            "id": "source-private-id",
            "image_path": image.name,
            "system_prompt": "system",
            "prompt": "user",
        })
    generations = []
    for candidate in ("stock", "step78", "step156"):
        path = tmp_path / f"{candidate}.jsonl"
        output = f'{{"context_analysis":{{"candidate":"{candidate}"}}}}'
        path.write_text(json.dumps({
            "example_id": "source-private-id",
            "candidate_output": output,
            "candidate_sha256": builder.sha256_text(output),
        }) + "\n")
        generations.append((candidate, path))
    template = "${system_instruction}\n${user_message}\n${output}"
    items, mapping = builder.build(
        csv_path=csv_path, generations=generations, template=template, count=1
    )
    assert len(items) == len(mapping) == 3
    assert {row["candidate"] for row in mapping} == {"stock", "step78", "step156"}
    assert all(not builder.FORBIDDEN_FIELDS.intersection(item) for item in items)
    assert all(item["example_id"] != "source-private-id" for item in items)
    assert all("candidate" not in item for item in items)
