#!/usr/bin/env python3
"""Validate and aggregate a completed private image-review export."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from reference_common import canonical_json, sha256_file

GRADES = ("Excellent", "VeryGood", "Good", "NeedsImprovement", "Fail")
GOOD_OR_BETTER = frozenset(GRADES[:3])


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def score_review(review: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    if review.get("schema_version") != "gemma4-e4b-image-review-v1":
        raise ValueError("unsupported review schema")
    if mapping.get("schema_version") != "gemma4-e4b-image-review-map-v1":
        raise ValueError("unsupported review mapping schema")
    if review.get("review_id") != mapping.get("review_id"):
        raise ValueError("review and mapping IDs do not match")

    map_by_pair = {str(item["pair_id"]): item for item in mapping.get("items", [])}
    results = review.get("results")
    if not isinstance(results, list) or len(results) != len(map_by_pair):
        raise ValueError("review result count does not match mapping")
    result_by_pair = {str(item.get("pair_id", "")): item for item in results}
    if len(result_by_pair) != len(results) or set(result_by_pair) != set(map_by_pair):
        raise ValueError("review pair inventory does not match mapping")

    grade_counts = {arm: collections.Counter() for arm in ("reference", "candidate")}
    omission_counts = collections.Counter()
    preferences = collections.Counter()
    for pair_id, arm_map in map_by_pair.items():
        result = result_by_pair[pair_id]
        outputs = result.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {"A", "B"}:
            raise ValueError(f"review outputs are incomplete: {pair_id}")
        slots = arm_map.get("slots")
        if (
            not isinstance(slots, dict)
            or set(slots) != {"A", "B"}
            or set(slots.values())
            != {
                "reference",
                "candidate",
            }
        ):
            raise ValueError(f"review mapping slots are invalid: {pair_id}")
        for slot in ("A", "B"):
            score = outputs[slot]
            if not isinstance(score, dict):
                raise ValueError(f"review output score is invalid: {pair_id}/{slot}")
            grade = score.get("grade")
            if grade not in GRADES:
                raise ValueError(f"review grade is missing or invalid: {pair_id}/{slot}")
            omission = score.get("material_omission")
            if not isinstance(omission, bool):
                raise ValueError(f"material_omission must be boolean: {pair_id}/{slot}")
            arm = slots[slot]
            grade_counts[arm][grade] += 1
            omission_counts[arm] += int(omission)
        preference = result.get("preference")
        if preference == "Tie":
            preferences["tie"] += 1
        elif preference in {"A", "B"}:
            preferences[slots[preference]] += 1
        else:
            raise ValueError(f"review preference is missing or invalid: {pair_id}")

    arms: dict[str, Any] = {}
    for arm in ("reference", "candidate"):
        counts = grade_counts[arm]
        arms[arm] = {
            "grades": {grade: counts[grade] for grade in GRADES},
            "good_or_better": sum(counts[grade] for grade in GOOD_OR_BETTER),
            "fail": counts["Fail"],
            "material_omissions": omission_counts[arm],
        }
    checks = {
        "good_or_better_not_lower": arms["candidate"]["good_or_better"] >= arms["reference"]["good_or_better"],
        "fail_not_higher": arms["candidate"]["fail"] <= arms["reference"]["fail"],
        "material_omissions_not_higher": arms["candidate"]["material_omissions"]
        <= arms["reference"]["material_omissions"],
    }
    return {
        "schema_version": "gemma4-e4b-image-review-summary-v1",
        "review_id": review["review_id"],
        "cases": len(results),
        "arms": arms,
        "preferences": {
            "candidate": preferences["candidate"],
            "reference": preferences["reference"],
            "tie": preferences["tie"],
        },
        "matched_product_noninferiority": {
            "pass": all(checks.values()),
            "checks": checks,
        },
        "scope": "Plan 29 product corpus only; legacy frozen-20 model-selection grades are separate evidence.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review = _load(args.review)
    mapping = _load(args.mapping)
    summary = score_review(review, mapping)
    summary["review_sha256"] = sha256_file(args.review)
    summary["mapping_sha256"] = sha256_file(args.mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 0 if summary["matched_product_noninferiority"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
