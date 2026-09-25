"""Verify one immutable 88-step segment in the two-epoch Plan 32 lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from gemma_tuner.models.common.plan31_input_modes import MODES
from gemma_tuner.utils.checkpoints import verify_complete_checkpoint


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"incomplete JSONL: {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def verify(output: Path, schedule_path: Path, *, epoch: int, start: int, stop: int) -> dict:
    epoch_start, epoch_end = ((0, 352) if epoch == 1 else (352, 704) if epoch == 2 else (-1, -1))
    if (epoch_start < 0 or start < epoch_start or start >= epoch_end
            or stop - start != 88 or (start - epoch_start) % 88 or stop > epoch_end):
        raise ValueError("Plan 32 segment is not one aligned 88-step interval in its epoch")

    checkpoint = verify_complete_checkpoint(output / f"checkpoint-{stop}")
    if int(checkpoint["global_step"]) != stop:
        raise ValueError("checkpoint global step differs from segment stop")
    base_ordinal = (start - epoch_start) * 8
    end_ordinal = min((stop - epoch_start) * 8, 2812)
    expected_count = end_ordinal - base_ordinal
    schedule = read_jsonl(schedule_path)
    schedule_hash = sha(schedule_path)
    if len(schedule) != 2812:
        raise ValueError("Plan 32 schedule must contain 2,812 rows")
    exposures = read_jsonl(output / "exposures.jsonl")
    if len(exposures) != expected_count:
        raise ValueError("Plan 32 successful exposure count changed")
    for local, record in enumerate(exposures, start=1):
        ordinal = base_ordinal + local
        frozen = schedule[ordinal - 1]
        if (record["exposure"] != local or record["source_ordinal"] != ordinal
                or record["epoch"] != epoch or record["mode"] != frozen["mode"]
                or record["schedule_sha256"] != schedule_hash
                or record["row_key_sha256"] != hashlib.sha256(str(frozen["id"]).encode()).hexdigest()):
            raise ValueError("Plan 32 exposure ledger differs from its frozen epoch schedule")
    exposure_receipt = json.loads((output / "plan32_exposure_verification.json").read_text())
    if (exposure_receipt["exposures"] != expected_count
            or exposure_receipt["start_ordinal"] != base_ordinal + 1
            or exposure_receipt["end_ordinal"] != end_ordinal
            or exposure_receipt["schedule_sha256"] != schedule_hash
            or exposure_receipt["ledger_sha256"] != sha(output / "exposures.jsonl")):
        raise ValueError("Plan 32 exposure receipt does not bind the segment ledger")

    telemetry_path = output / "validation_telemetry.jsonl"
    telemetry = read_jsonl(telemetry_path)
    events = {}
    for row in telemetry:
        key = (row["event"], int(row["step"]))
        if key in events:
            raise ValueError(f"duplicate Plan 32 telemetry event: {key}")
        if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in row.values()):
            raise ValueError("Plan 32 telemetry contains a non-finite metric")
        if row["event"].endswith("_eval") or row["event"].startswith("mode_validation_eval:"):
            if row.get("scored_tokens", 0) <= 0 or not math.isfinite(row["loss"]):
                raise ValueError("Plan 32 evaluation loss is missing a finite token denominator")
        events[key] = row

    for step in range(start + 1, stop + 1):
        row = events.get(("optimization", step))
        if row is None or not all(key in row for key in ("stochastic_loss", "grad_norm", "learning_rate")):
            raise ValueError(f"missing finite optimization event at step {step}")
    fixed_steps = {start, stop, *range(start + 22, stop + 1, 22)}
    for step in fixed_steps:
        for name in ("fixed_train_eval", "fixed_validation_eval"):
            if (name, step) not in events:
                raise ValueError(f"missing {name} at step {step}")
    for step in (start, stop):
        for mode in MODES:
            row = events.get((f"mode_validation_eval:{mode}", step))
            if row is None or int(row.get("examples", 0)) != 60:
                raise ValueError(f"missing 60-row {mode} validation at step {step}")
    full_event = events.get(("full_validation_eval", stop))
    if stop in (352, 704) and (full_event is None or int(full_event.get("examples", 0)) != 252):
        raise ValueError("missing full-252 validation loss at epoch end")
    checkpoint_event = events.get(("checkpoint", stop))
    if checkpoint_event is None or checkpoint_event["tree_sha256"] != checkpoint["tree_sha256"]:
        raise ValueError("checkpoint telemetry tree hash mismatch")

    return {
        "schema_version": "plan32_segment_verification_v1",
        "epoch": epoch,
        "start_step": start,
        "stop_step": stop,
        "successful_exposures": expected_count,
        "source_ordinal_range": [base_ordinal + 1, end_ordinal],
        "final_partial_accumulation": 4 if stop in (352, 704) else 0,
        "optimization_events": 88,
        "fixed_loss_steps": sorted(fixed_steps),
        "mode_loss_steps": [start, stop],
        "full_validation_loss": None if full_event is None else full_event["loss"],
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "schedule_sha256": schedule_hash,
        "exposures_sha256": sha(output / "exposures.jsonl"),
        "telemetry_sha256": sha(telemetry_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int, choices=(1, 2))
    parser.add_argument("--start-step", required=True, type=int)
    parser.add_argument("--stop-step", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir, args.schedule, epoch=args.epoch,
                            start=args.start_step, stop=args.stop_step), sort_keys=True))


if __name__ == "__main__":
    main()
