#!/usr/bin/env python3
"""Upload only sealed immutable checkpoints and append a verified ledger entry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from gemma_tuner.utils.checkpoints import COMPLETE_MARKER, verify_complete_checkpoint
from tools.gcp_tt_sft_contract import DEFAULT_CONTRACT, load_contract

Runner = Callable[..., subprocess.CompletedProcess[str]]


def read_ledger(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        key = str(entry["checkpoint"])
        if key in entries and entries[key] != entry:
            raise ValueError("conflicting checkpoint ledger entry")
        entries[key] = entry
    return entries


def upload_checkpoint(
    checkpoint: Path,
    destination_root: str,
    contract: dict[str, Any],
    ledger: Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    manifest = verify_complete_checkpoint(checkpoint)
    existing = read_ledger(ledger).get(checkpoint.name)
    destination = f"{destination_root.rstrip('/')}/{checkpoint.name}"
    expected = {
        "schema_version": "gemma_checkpoint_courier_ledger_v1",
        "checkpoint": checkpoint.name,
        "destination": destination,
        "tree_sha256": manifest["tree_sha256"],
    }
    if existing is not None:
        if existing != expected:
            raise ValueError("checkpoint already has a conflicting ledger entry")
        return existing
    gcp = contract["gcp"]
    common = [f"--project={gcp['project']}", f"--account={gcp['account']}"]
    runner(
        [
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            *common,
            str(checkpoint),
            destination_root.rstrip("/") + "/",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    remote_marker = runner(
        [
            "gcloud",
            "storage",
            "cat",
            *common,
            f"{destination}/{COMPLETE_MARKER}",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if json.loads(remote_marker) != manifest:
        raise RuntimeError("uploaded checkpoint marker does not match local manifest")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    print(
        json.dumps(
            upload_checkpoint(
                args.checkpoint.resolve(),
                args.destination,
                load_contract(args.contract.resolve()),
                args.ledger.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
