"""Aggregate Plan 33 exposed-20 screen ledgers (no private content) and apply the frozen screen gate."""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

from tools.run_plan33_eval import MODES, parse_plan33_vote

ROOT = Path(__file__).resolve().parents[1] / "output/plan33-eval"


def status(vote: dict) -> str:
    try:
        return parse_plan33_vote(vote["raw_output"])["status"]
    except Exception:  # noqa: BLE001 - any unparseable judge output is a technical invalid
        return "TECHNICAL_INVALID"


def summarize(arm: str) -> dict:
    directory = ROOT / f"exposed20-{arm}"
    generations = [json.loads(line) for line in (directory / "generations.jsonl").read_text().splitlines()]
    votes = [json.loads(line) for line in (directory / "votes.jsonl").read_text().splitlines()]
    totals = {judge: collections.Counter(status(v) for v in votes if v["judge"] == judge) for judge in ("grok", "composer")}
    per_mode = {mode: {
        "runaways": sum(g["finish_reason"] != "stop" for g in generations if g["mode"] == mode),
        **{judge: dict(collections.Counter(status(v) for v in votes if v["mode"] == mode and v["judge"] == judge))
           for judge in ("grok", "composer")}} for mode in MODES}
    natural = [g for g in generations if g["finish_reason"] == "stop"]
    return {
        "arm": arm, "generations": len(generations), "votes": len(votes),
        "runaways": sum(g["finish_reason"] != "stop" for g in generations),
        "totals": {judge: dict(counts) for judge, counts in totals.items()},
        "balance": {judge: counts["APPROVED"] - counts["REJECTED"] for judge, counts in totals.items()},
        "median_tokens": statistics.median(g["generation_tokens"] for g in generations),
        "median_latency_s": round(statistics.median(g["elapsed_seconds"] for g in generations), 2),
        "median_natural_latency_s": round(statistics.median(g["elapsed_seconds"] for g in natural), 2) if natural else None,
        "peak_mlx_gib": round(max(g["peak_mlx_bytes"] for g in generations) / 2**30, 2),
        "per_mode": per_mode,
    }


def screen_gate(candidate: dict, stock: dict) -> dict:
    """Frozen rule: eligible unless runaways exceed stock or balance < stock - 6 for either judge."""
    reasons = []
    if candidate["runaways"] > stock["runaways"]:
        reasons.append(f"runaways {candidate['runaways']} > stock {stock['runaways']}")
    for judge in ("grok", "composer"):
        if candidate["balance"][judge] < stock["balance"][judge] - 6:
            reasons.append(f"{judge} balance {candidate['balance'][judge]} < stock {stock['balance'][judge]} - 6")
    return {"eligible": not reasons, "reasons": reasons}


if __name__ == "__main__":
    stock = summarize("stock-e2b-bf16")
    for arm in sys.argv[1:]:
        result = summarize(arm)
        result["screen_gate"] = screen_gate(result, stock)
        print(json.dumps(result, sort_keys=True))
