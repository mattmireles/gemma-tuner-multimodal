#!/usr/bin/env python3
"""Grade blinded telepathic-context work items with pinned Cursor CLI judges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

GRADES = ("Excellent", "VeryGood", "Good", "NeedsImprovement", "Fail")
JUDGES = {
    "grok": "cursor-grok-4.6-medium",
    "composer": "composer-2.5[fast=false]",
}
FORBIDDEN_FIELDS = frozenset(
    {"arm", "model", "checkpoint", "split", "target", "target_json", "competing_output"}
)
Runner = Callable[[dict[str, str], str, Path], dict[str, Any]]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def parse_grade(response: str) -> tuple[str, str | None]:
    first = response.strip().splitlines()[0] if response.strip() else ""
    if first in GRADES[:-1]:
        return first, None
    if first == "Fail":
        return "Fail", None
    if first.startswith("Fail - ") and len(first.removeprefix("Fail - ").strip()) >= 3:
        return "Fail", first.removeprefix("Fail - ").strip()
    raise ValueError("malformed judge response")


def read_items(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hashes = [str(row["work_item_hash"]) for row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate work item hash")
    for row in rows:
        leaked = FORBIDDEN_FIELDS.intersection(row)
        if leaked:
            raise ValueError(f"blind work item leaked fields: {sorted(leaked)}")
        if sha256_text(str(row["rendered_prompt"])) != row["rendered_prompt_sha256"]:
            raise ValueError("rendered prompt hash mismatch")
        image = Path(row["image_ref"])
        if not image.is_file():
            raise FileNotFoundError(image)
    return rows


def read_ledger(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["work_item_hash"])
        if key in result and result[key] != row:
            raise ValueError("conflicting grade ledger entry")
        result[key] = row
    return result


def run_cursor(item: dict[str, str], model: str, agent: Path) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gemma-blind-judge-") as directory:
        workspace = Path(directory)
        source = Path(item["image_ref"])
        image = workspace / source.name
        shutil.copy2(source, image)
        prompt = (
            f"{item['rendered_prompt']}\n\n"
            f"Read only the attached screenshot at {image}. Do not use shell, MCP, search, "
            "or inspect any other file. Return only the grade format requested above."
        )
        completed = subprocess.run(
            [
                str(agent),
                "-p",
                prompt,
                "--model",
                model,
                "--mode",
                "ask",
                "--workspace",
                str(workspace),
                "--print",
                "--output-format",
                "text",
                "--trust",
                "--sandbox",
                "enabled",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            raise RuntimeError(f"Cursor judge failed with exit {completed.returncode}")
        return {
            "response": completed.stdout.strip(),
            "latency_seconds": time.monotonic() - started,
        }


def dispatch(
    items: list[dict[str, str]],
    ledger: Path,
    *,
    judge: str,
    workers: int,
    agent: Path,
    runner: Runner = run_cursor,
) -> dict[str, Any]:
    model = JUDGES[judge]
    existing = read_ledger(ledger)
    for row in existing.values():
        if row.get("judge") != judge or row.get("model_selector") != model:
            raise ValueError("existing ledger judge mismatch")
    pending = [item for item in items if item["work_item_hash"] not in existing]
    completed: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(runner, item, model, agent): item for item in pending}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            result = future.result()
            response = str(result["response"])
            grade, reason = parse_grade(response)
            row: dict[str, Any] = {
                "schema_version": "gemma_cursor_blind_grade_v1",
                "work_item_hash": item["work_item_hash"],
                "example_id": item["example_id"],
                "judge": judge,
                "model_selector": model,
                "grade": grade,
                "response_sha256": sha256_text(response),
                "latency_seconds": float(result["latency_seconds"]),
            }
            if reason is not None:
                row["fail_reason"] = reason
            completed[item["work_item_hash"]] = row
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        for item in pending:
            handle.write(canonical(completed[item["work_item_hash"]]) + "\n")
            handle.flush()
    return {"judge": judge, "model_selector": model, "total": len(items), "new": len(pending)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-items", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--judge", choices=tuple(JUDGES), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--agent", type=Path, default=Path.home() / ".local/bin/agent")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be in [1, 4]")
    items = read_items(args.work_items.resolve())
    if args.limit is not None:
        items = items[: args.limit]
    print(canonical(dispatch(
        items,
        args.ledger.resolve(),
        judge=args.judge,
        workers=args.workers,
        agent=args.agent.resolve(),
    )))


if __name__ == "__main__":
    main()
