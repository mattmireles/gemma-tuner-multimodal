"""Compare the same frozen 20 full-input rows across Plan 32 Mac runtimes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

RANK = {"REJECTED": 0, "NEEDS_REVIEW": 1, "APPROVED": 2}


def read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"partial ledger: {path}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_arm(path: Path) -> tuple[list[dict], dict]:
    rows = [row for row in read_jsonl(path / "generations.jsonl") if row["mode"] == "full"]
    if len(rows) != 20 or len({row["id"] for row in rows}) != 20:
        raise ValueError(f"expected 20 distinct full-input generations: {path}")
    vote_rows = [row for row in read_jsonl(path / "votes.jsonl") if row["mode"] == "full"]
    votes = {(row["id"], row["judge"]): row for row in vote_rows}
    if len(vote_rows) != 40 or len(votes) != 40:
        raise ValueError(f"expected 40 unique full-input votes: {path}")
    return rows, votes


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(rows: list[dict], votes: dict) -> dict:
    latencies = [row["elapsed_seconds"] for row in rows]
    natural = [row for row in rows if row["finish_reason"] == "stop"]
    statuses = {}
    for judge in ("grok", "composer"):
        statuses[judge] = dict(Counter(
            vote["verdict"]["status"] if vote["state"] == "valid" else f"technical:{vote['state']}"
            for (row_id, name), vote in votes.items() if name == judge
        ))
    return {
        "natural_stops": len(natural),
        "token_limit": sum(row["finish_reason"] == "length" for row in rows),
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "median_natural_seconds": statistics.median(row["elapsed_seconds"] for row in natural)
        if natural else None,
        "mean_tokens": statistics.mean(row["generation_tokens"] for row in rows),
        "median_tokens": statistics.median(row["generation_tokens"] for row in rows),
        "judge_statuses": statuses,
    }


def paired(left: dict, right: dict, ids: list[str]) -> dict:
    result = {}
    for judge in ("grok", "composer"):
        counts = Counter()
        for row_id in ids:
            a, b = left[(row_id, judge)], right[(row_id, judge)]
            if a["state"] != "valid" or b["state"] != "valid":
                counts["technical_excluded"] += 1
                continue
            delta = RANK[b["verdict"]["status"]] - RANK[a["verdict"]["status"]]
            counts["right_win" if delta > 0 else "left_win" if delta < 0 else "tie"] += 1
        result[judge] = dict(counts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--ane-w6", type=Path, required=True)
    parser.add_argument("--ane-w4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {
        name: load_arm(path)
        for name, path in (("bf16", args.bf16), ("ane_w6", args.ane_w6), ("ane_w4", args.ane_w4))
    }
    ids = [row["id"] for row in arms["bf16"][0]]
    hashes = [row["image_sha256"] for row in arms["bf16"][0]]
    prompts = [row["prompt_sha256"] for row in arms["bf16"][0]]
    for name, (rows, _) in arms.items():
        if [row["id"] for row in rows] != ids or [row["image_sha256"] for row in rows] != hashes:
            raise ValueError(f"{name} is not the same 20 screenshots in the same order")
        if [row["prompt_sha256"] for row in rows] != prompts:
            raise ValueError(f"{name} does not use the same full-input prompts")
    result = {
        "panel_ids": ids,
        "arms": {name: summarize(*data) for name, data in arms.items()},
        "paired_vs_bf16": {
            name: paired(arms["bf16"][1], arms[name][1], ids)
            for name in ("ane_w6", "ane_w4")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": result["arms"], "paired_vs_bf16": result["paired_vs_bf16"]}, indent=2))


if __name__ == "__main__":
    main()
