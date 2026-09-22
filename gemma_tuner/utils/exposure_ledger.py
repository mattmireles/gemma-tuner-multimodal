"""Append-only, content-redacted training exposure evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from transformers import TrainerCallback


def _row_key(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class ExposureLedger:
    """Stage a micro-batch, then commit it only after backward succeeds."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        self.seen: set[str] = set()
        self.pending: list[list[dict[str, Any]]] = []
        if self.path.exists():
            for expected, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != "gemma_training_exposure_v1":
                    raise ValueError("unexpected exposure ledger schema")
                if int(row.get("exposure", -1)) != expected:
                    raise ValueError("exposure ledger is not a contiguous append-only sequence")
                key = str(row.get("row_key_sha256", ""))
                if len(key) != 64 or key in self.seen:
                    raise ValueError("exposure ledger contains an invalid or duplicate row key")
                self.rows.append(row)
                self.seen.add(key)

    def stage(self, features: Iterable[dict[str, Any]]) -> None:
        staged: list[dict[str, Any]] = []
        for feature in features:
            if "id" not in feature:
                raise ValueError("exposure tracking requires an id column")
            key = _row_key(feature["id"])
            pending_keys = {
                row["row_key_sha256"] for batch in self.pending for row in batch
            }
            if key in self.seen or key in pending_keys or any(row["row_key_sha256"] == key for row in staged):
                raise RuntimeError("training row would be exposed more than once")
            staged.append({"row_key_sha256": key})
        if not staged:
            raise ValueError("cannot stage an empty exposure batch")
        self.pending.append(staged)

    def commit(self) -> None:
        if not self.pending:
            raise RuntimeError("no exposure batch is pending")
        staged_batch = self.pending.pop(0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for staged in staged_batch:
                row = {
                    "schema_version": "gemma_training_exposure_v1",
                    "exposure": len(self.rows) + 1,
                    **staged,
                }
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                self.rows.append(row)
                self.seen.add(row["row_key_sha256"])
            handle.flush()
            os.fsync(handle.fileno())

    def verify_complete(self, row_ids: Iterable[Any]) -> dict[str, Any]:
        expected = [_row_key(value) for value in row_ids]
        if self.pending:
            raise RuntimeError("an exposure batch is still pending")
        actual = [row["row_key_sha256"] for row in self.rows]
        if len(expected) != len(set(expected)):
            raise ValueError("expected training row IDs are not unique")
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError("exposure ledger does not cover the expected training rows exactly once")
        return {
            "schema_version": "gemma_training_exposure_verification_v1",
            "exposures": len(actual),
            "unique_rows": len(set(actual)),
            "complete": True,
        }


class ExposureTrackingCollator:
    """Stage the identities of the exact features passed to the real collator."""

    def __init__(self, delegate: Any, ledger: ExposureLedger) -> None:
        self.delegate = delegate
        self.ledger = ledger

    def __call__(self, features: list[dict[str, Any]]) -> Any:
        batch = self.delegate(features)
        self.ledger.stage(features)
        return batch


class ExposureCommitCallback(TrainerCallback):
    """Commit after a successful backward substep or optimizer-step boundary."""

    def __init__(self, ledger: ExposureLedger) -> None:
        self.ledger = ledger

    def on_substep_end(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        self.ledger.commit()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        self.ledger.commit()
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        if self.ledger.pending:
            raise RuntimeError("training ended with an uncommitted exposure batch")
        return control
