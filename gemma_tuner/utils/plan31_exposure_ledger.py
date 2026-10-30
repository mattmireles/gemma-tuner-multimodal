"""Content-bound, ordered exposure ledger for one Plan 31 training segment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from gemma_tuner.models.common.plan31_input_modes import render_plan31_input


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Plan31ExposureLedger:
    """Accept only the next frozen row and exact mode/content in source order."""

    def __init__(
        self,
        path: str | Path,
        *,
        schedule_path: str | Path,
        train_rows: Iterable[dict[str, Any]],
        start_ordinal: int,
        end_ordinal: int,
        expected_epoch: int = 1,
        schema_version: str = "plan31_training_exposure_v1",
    ) -> None:
        self.path = Path(path)
        schedule_path = Path(schedule_path)
        schedule_bytes = schedule_path.read_bytes()
        if not schedule_bytes.endswith(b"\n"):
            raise ValueError("Plan 31 mode schedule is incomplete")
        schedule = [json.loads(line) for line in schedule_bytes.splitlines()]
        features = list(train_rows)
        if len(schedule) != len(features) or not 1 <= start_ordinal <= end_ordinal <= len(features):
            raise ValueError("Plan 31 schedule/segment row counts are inconsistent")
        self.schedule_sha256 = _sha(schedule_bytes)
        self.start_ordinal = start_ordinal
        self.end_ordinal = end_ordinal
        self.expected_epoch = int(expected_epoch)
        self.schema_version = str(schema_version)
        if self.expected_epoch < 1 or not self.schema_version:
            raise ValueError("exposure epoch and schema version must be valid")
        self.expected: list[dict[str, Any]] = []
        ids = set()
        for ordinal, (schedule_row, feature) in enumerate(zip(schedule, features), start=1):
            identifier = str(feature["id"])
            if (identifier in ids or schedule_row["id"] != identifier
                    or schedule_row["source_ordinal"] != ordinal
                    or int(schedule_row["epoch"]) != self.expected_epoch):
                raise ValueError("Plan 31 source order or identity conflicts with schedule")
            ids.add(identifier)
            mode = str(feature["input_mode"])
            if schedule_row["mode"] != mode:
                raise ValueError("Plan 31 mode differs from frozen schedule")
            for feature_key, schedule_key in (
                ("prompt", "prompt_sha256"), ("system_prompt", "system_sha256")
            ):
                if schedule_key in schedule_row and _sha(str(feature[feature_key]).encode("utf-8")) != schedule_row[schedule_key]:
                    raise ValueError(f"source {feature_key} differs from frozen schedule")
            image_path = Path(str(feature["image_path"]))
            if not image_path.is_file() or _file_sha(image_path) != schedule_row["image_sha256"]:
                raise ValueError("Plan 31 screenshot differs from frozen schedule")
            target_sha = _sha((str(feature["response"]) + "\n").encode("utf-8"))
            if target_sha != schedule_row["target_sha256"]:
                raise ValueError("Plan 31 target differs from frozen schedule")
            views, messages = render_plan31_input(
                mode=mode, full_prompt=str(feature["prompt"]),
                system_prompt=str(feature["system_prompt"]),
                full_views=[f"view:{index}" for index in range(5)],
            )
            self.expected.append({
                "schema_version": self.schema_version,
                "schedule_sha256": self.schedule_sha256,
                "source_ordinal": ordinal,
                "epoch": int(schedule_row["epoch"]),
                "row_key_sha256": _sha(identifier.encode("utf-8")),
                "mode": mode,
                "image_sha256": schedule_row["image_sha256"],
                "prompt_sha256": _sha(str(feature["prompt"]).encode("utf-8")),
                "system_sha256": _sha(str(feature["system_prompt"]).encode("utf-8")),
                "target_sha256": target_sha,
                "rendered_messages_sha256": _sha(_canonical(messages)),
                "image_order_sha256": _sha(_canonical(views)),
                "views": len(views),
            })
        self.rows: list[dict[str, Any]] = []
        self.pending: list[list[dict[str, Any]]] = []
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise ValueError("Plan 31 exposure ledger ends with a partial record")
            for line in raw.splitlines():
                row = json.loads(line)
                index = start_ordinal - 1 + len(self.rows)
                if index >= end_ordinal or row != {"exposure": len(self.rows) + 1, **self.expected[index]}:
                    raise ValueError("Plan 31 exposure ledger conflicts with frozen row order or content")
                self.rows.append(row)

    def stage(self, features: Iterable[dict[str, Any]]) -> None:
        staged = []
        for feature in features:
            index = self.start_ordinal - 1 + len(self.rows) + sum(map(len, self.pending)) + len(staged)
            if index >= len(self.expected):
                raise RuntimeError("Plan 31 exposure would exceed the frozen epoch")
            if _sha(str(feature["id"]).encode("utf-8")) != self.expected[index]["row_key_sha256"]:
                raise RuntimeError("Plan 31 exposure is duplicated or reordered")
            if str(feature["input_mode"]) != self.expected[index]["mode"]:
                raise RuntimeError("Plan 31 exposure mode changed")
            for field, key in (("prompt", "prompt_sha256"), ("system_prompt", "system_sha256")):
                if _sha(str(feature[field]).encode("utf-8")) != self.expected[index][key]:
                    raise RuntimeError(f"Plan 31 exposure {field} changed")
            if _sha((str(feature["response"]) + "\n").encode("utf-8")) != self.expected[index]["target_sha256"]:
                raise RuntimeError("Plan 31 exposure target changed")
            if _file_sha(Path(str(feature["image_path"]))) != self.expected[index]["image_sha256"]:
                raise RuntimeError("Plan 31 exposure image changed")
            staged.append(self.expected[index])
        if not staged:
            raise ValueError("cannot stage an empty Plan 31 batch")
        self.pending.append(staged)

    def commit(self) -> None:
        if not self.pending:
            raise RuntimeError("no Plan 31 exposure batch is pending")
        batch = self.pending[0]
        if len(self.rows) + len(batch) > self.end_ordinal - self.start_ordinal + 1:
            raise RuntimeError("Plan 31 segment would commit beyond its sealed stop")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for item in batch:
                row = {"exposure": len(self.rows) + 1, **item}
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                self.rows.append(row)
            handle.flush()
            os.fsync(handle.fileno())
        self.pending.pop(0)

    def verify_complete(self) -> dict[str, Any]:
        count = self.end_ordinal - self.start_ordinal + 1
        if self.pending or len(self.rows) != count:
            raise RuntimeError("Plan 31 segment has pending, missing, or extra exposures")
        return {"schema_version": f"{self.schema_version.removesuffix('_training_exposure_v1')}_exposure_verification_v1",
                "exposures": count,
                "start_ordinal": self.start_ordinal, "end_ordinal": self.end_ordinal,
                "schedule_sha256": self.schedule_sha256, "ledger_sha256": _file_sha(self.path)}
