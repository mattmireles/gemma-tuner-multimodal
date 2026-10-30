"""Verify the immutable Plan 33 literal v3 corpus before any E2B training.

Read-only over private inputs. Emits a redacted receipt containing only hashes,
counts, and pass/fail facts; no prompts, OCR, targets, IDs, or paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT.parent / "perfect-dictator"
V3 = PD / "data/datasets/tt_screenshot_literal_sft_v3/private/v3"
V1 = PD / "data/datasets/tt_screenshot_literal_sft_v1/private/v1"
PARENT = PD / "data/datasets/tt_screenshot_plan28_sft_v1/private/plan29-sft-v1"
SOURCE = PD / "data/datasets/tt_screenshot_json_v2"
SOURCE_MANIFESTS = ("manifest.jsonl", "manifest_ext1.jsonl", "manifest_ext2.jsonl", "manifest_ext3.jsonl")

SCHEMA = "tt_screenshot_literal_sft_v3"
PARENT_SCHEMA = "tt_screenshot_plan28_sft_v1"
EXPECTED = {
    "manifest.json": "004f932f2e6d14799e69a40c1779930dc144d297269271993d191f5869d7f442",
    "train.jsonl": "3017a07aa52e623d5a953f6b30003cc3f12855d4e8eb6e85a2ff9010bbb8789c",
    "validation.jsonl": "8b235bf82ca8c9f5dba595228d23273473b1d2ee6aeb58ddd064b514c169813e",
}
PARENT_EXPECTED = {
    "train.jsonl": "bb680f18fc6a609dd1e31648f3070e17077af4a82e177551d9fdb33486a13fd6",
    "validation.jsonl": "b9109cc2e6f7f830a82bef6b13502cd53bbb76947433c1b900f738e582cc90db",
}
ROWS = {"train.jsonl": 2812, "validation.jsonl": 252}
SEALED_TEST_ROWS = 232
TELEPATHIC_PREFIX = "you are a telepathic context analyst"
JSON_ONLY = "Return valid JSON only."
OCR_CLOSE = "</first_pass_screenshot_ocr>"
# Literal target allowlist: every leaf path permitted in the concise v3 target.
ALLOWED_PATHS = {
    ("context_analysis", "type"),
    ("context_analysis", "active_conversation", "participants", "[]", "name"),
    ("context_analysis", "active_conversation", "participants", "[]", "role"),
    ("context_analysis", "active_conversation", "participants", "[]", "recent_messages", "[]"),
    ("context_analysis", "active_conversation", "current_topic", "summary"),
    ("context_analysis", "active_conversation", "current_topic", "existing_text"),
    ("context_analysis", "technical_context", "language"),
    ("context_analysis", "technical_context", "unusual_terms", "[]"),
}
ALLOWED_PREFIXES = {path[:i] for path in ALLOWED_PATHS for i in range(1, len(path) + 1)}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"incomplete JSONL: {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def target_paths(value, prefix: tuple = ()) -> set[tuple]:
    """Return every key path in a target; empty containers count as their own path."""
    if isinstance(value, dict):
        paths = {prefix} if not value else set()
        for key, child in value.items():
            paths |= target_paths(child, prefix + (key,))
        return paths
    if isinstance(value, list):
        paths = {prefix + ("[]",)} if not value else set()
        for child in value:
            paths |= target_paths(child, prefix + ("[]",))
        return paths
    return {prefix}


def check_prompt(row: dict) -> list[str]:
    errors = []
    system = row.get("system_prompt") or ""
    user = row.get("user_prompt") or ""
    if "confidence scoring" in system.lower() or "# confidence" in system.lower():
        errors.append("system_confidence")
    if user.count(JSON_ONLY) != 1:
        errors.append("json_only_count")
    if not user.rstrip().endswith(JSON_ONLY):
        errors.append("json_only_not_last")
    if OCR_CLOSE in user and user.rfind(OCR_CLOSE) > user.rfind(JSON_ONLY):
        errors.append("json_only_before_ocr")
    return errors


def check_target(row: dict) -> list[str]:
    try:
        target = json.loads(row["response"])
    except json.JSONDecodeError:
        return ["target_unparseable"]
    extra = {path for path in target_paths(target) if path not in ALLOWED_PREFIXES}
    return ["target_extra_field"] if extra else []


def verify(check_images: bool) -> dict:
    facts: dict = {"schema_version": "plan33_data_verification_v1", "dataset_schema": SCHEMA}
    for name, expected in EXPECTED.items():
        actual = sha_file(V3 / name)
        if actual != expected:
            raise ValueError(f"v3 hash mismatch: {name}")
    for name, expected in PARENT_EXPECTED.items():
        if sha_file(PARENT / name) != expected:
            raise ValueError(f"parent hash mismatch: {name}")
    manifest = json.loads((V3 / "manifest.json").read_bytes())
    if manifest["schema_version"] != SCHEMA or manifest["parent_schema_version"] != PARENT_SCHEMA:
        raise ValueError("manifest schema lineage mismatch")
    if manifest["source_files"] != PARENT_EXPECTED:
        raise ValueError("manifest parent binding mismatch")
    facts["file_sha256"] = dict(EXPECTED)
    facts["parent_file_sha256"] = dict(PARENT_EXPECTED)

    splits: dict[str, list[dict]] = {}
    for name, count in ROWS.items():
        rows = load_jsonl(V3 / name)
        if len(rows) != count:
            raise ValueError(f"row count mismatch: {name}")
        split = name.removesuffix(".jsonl")
        if Counter(row["split"] for row in rows) != {split: count}:
            raise ValueError(f"split label mismatch: {name}")
        if {row["schema_version"] for row in rows} != {SCHEMA}:
            raise ValueError(f"row schema mismatch: {name}")
        splits[split] = rows
    facts["rows"] = {split: len(rows) for split, rows in splits.items()}

    # Lineage: v3 rows are exactly the parent rows, and targets are byte-identical to literal v1.
    for split, rows in splits.items():
        parent = load_jsonl(PARENT / f"{split}.jsonl")
        if [row["source_hash"] for row in rows] != [row["source_hash"] for row in parent]:
            raise ValueError(f"parent row identity/order mismatch: {split}")
        if [row["image_sha256"] for row in rows] != [row["image_sha256"] for row in parent]:
            raise ValueError(f"parent image binding mismatch: {split}")
        if [row["owner_hash"] for row in rows] != [row["owner_hash"] for row in parent]:
            raise ValueError(f"parent owner binding mismatch: {split}")
        v1 = {row["id"]: row["response"] for row in load_jsonl(V1 / f"{split}.jsonl")}
        if any(v1.get(row["id"]) != row["response"] for row in rows):
            raise ValueError(f"target drift from literal v1: {split}")
    facts["lineage"] = {"parent_row_order_identical": True, "targets_byte_identical_to_literal_v1": True}

    everything = splits["train"] + splits["validation"]
    if len({row["id"] for row in everything}) != len(everything):
        raise ValueError("duplicate id")
    if len({row["image_sha256"] for row in everything}) != len(everything):
        raise ValueError("duplicate image")
    train_owners = {row["owner_hash"] for row in splits["train"]}
    val_owners = {row["owner_hash"] for row in splits["validation"]}
    if train_owners & val_owners:
        raise ValueError("owner crossing train/validation")
    facts["split_disjointness"] = {
        "train_owners": len(train_owners),
        "validation_owners": len(val_owners),
        "owner_overlap": 0,
        "image_overlap": 0,
    }

    # Sealed-test closure: count and key test rows only by owner hash and R2 URI; never read test content.
    test_owners: set[str] = set()
    test_uris: set[str] = set()
    for name in SOURCE_MANIFESTS:
        for line in (SOURCE / name).read_bytes().splitlines():
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            if not str(row.get("system_prompt") or "").strip().lower().startswith(TELEPATHIC_PREFIX):
                continue
            test_owners.add(sha_bytes(str(row["owner_id"]).encode("utf-8")))
            test_uris.add(str(row["screenshot_r2_uri"]))
    test_count = len(test_uris)
    if test_count != SEALED_TEST_ROWS:
        raise ValueError(f"unexpected sealed test count {test_count}")
    all_owners = train_owners | val_owners
    all_uris = {row["image_r2_uri"] for row in everything}
    if all_owners & test_owners or all_uris & test_uris:
        raise ValueError("sealed test crossing")
    facts["sealed_test"] = {
        "rows": test_count,
        "owners": len(test_owners),
        "owner_overlap": 0,
        "r2_uri_overlap": 0,
        "test_content_read": False,
    }

    prompt_errors = Counter(error for row in everything for error in check_prompt(row))
    target_errors = Counter(error for row in everything for error in check_target(row))
    if prompt_errors or target_errors:
        raise ValueError(f"contract violations: {dict(prompt_errors | target_errors)}")
    facts["prompt_contract"] = {
        "system_confidence_rows": 0,
        "json_only_exactly_once_and_last_after_ocr": len(everything),
        "user_prompt_trailing_newline_rows": sum(row["user_prompt"].endswith("\n") for row in everything),
    }
    facts["target_contract"] = {"allowlisted_rows": len(everything), "unparseable": 0, "extra_fields": 0}

    if check_images:
        mismatched = sum(sha_file(Path(row["image_path"])) != row["image_sha256"] for row in everything)
        if mismatched:
            raise ValueError(f"{mismatched} image bytes mismatched")
        facts["image_bytes"] = {"verified": len(everything), "mismatched": 0}

    facts["passed"] = True
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-images", action="store_true", help="skip image byte rehash")
    parser.add_argument("--receipt", type=Path, help="write the redacted receipt here")
    args = parser.parse_args()
    facts = verify(check_images=not args.skip_images)
    raw = json.dumps(facts, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(raw, encoding="utf-8")
    print(raw, end="")


if __name__ == "__main__":
    main()
