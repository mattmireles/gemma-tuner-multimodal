"""Verify a sealed Plan 31 training segment without exporting private rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from gemma_tuner.utils.checkpoints import verify_complete_checkpoint


MODES = ("full", "no_ocr", "no_quadrants", "no_system", "image_instruction", "image_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError(f"incomplete JSONL file: {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def verify(output: Path, schedule_path: Path, start: int, stop: int) -> dict:
    if start < 352 or stop - start != 88 or stop > 704:
        raise ValueError("Plan 31 segments must advance exactly 88 steps after 352")
    checkpoint = verify_complete_checkpoint(output / f"checkpoint-{stop}")
    source_ordinal = (start - 352) * 8
    end_ordinal = min((stop - 352) * 8, 2812)
    expected_exposures = end_ordinal - source_ordinal

    schedule = read_jsonl(schedule_path)
    schedule_hash = sha256(schedule_path)
    if len(schedule) != 2812:
        raise ValueError("frozen schedule row count changed")
    exposures_path = output / "exposures.jsonl"
    exposures = read_jsonl(exposures_path)
    if len(exposures) != expected_exposures:
        raise ValueError("segment exposure count changed")
    for index, row in enumerate(exposures, start=1):
        ordinal = source_ordinal + index
        frozen = schedule[ordinal - 1]
        if (row["exposure"] != index or row["source_ordinal"] != ordinal
                or row["mode"] != frozen["mode"] or row["schedule_sha256"] != schedule_hash):
            raise ValueError("exposure order, mode, or schedule changed")
    exposure_receipt = json.loads((output / "plan31_exposure_verification.json").read_text())
    if (exposure_receipt["exposures"] != expected_exposures
            or exposure_receipt["start_ordinal"] != source_ordinal + 1
            or exposure_receipt["end_ordinal"] != end_ordinal
            or exposure_receipt["ledger_sha256"] != sha256(exposures_path)):
        raise ValueError("exposure verification receipt changed")

    telemetry_path = output / "validation_telemetry.jsonl"
    telemetry = read_jsonl(telemetry_path)
    events: dict[tuple[str, int], dict] = {}
    for row in telemetry:
        key = (row["event"], int(row["step"]))
        if key in events:
            raise ValueError("duplicate telemetry event")
        for value in row.values():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError("non-finite telemetry value")
        if row["event"].endswith("_eval") or row["event"].startswith("mode_validation_eval:"):
            if row.get("scored_tokens", 0) <= 0 or not math.isfinite(row["loss"]):
                raise ValueError("missing or non-finite validation loss denominator")
        events[key] = row
    for step in range(start + 1, stop + 1):
        row = events.get(("optimization", step))
        if row is None or not all(key in row for key in ("stochastic_loss", "grad_norm", "learning_rate")):
            raise ValueError(f"missing training metrics at step {step}")
    fixed_steps = {start, stop, *range(start + 22, stop + 1, 22)}
    for step in fixed_steps:
        for event in ("fixed_train_eval", "fixed_validation_eval"):
            if (event, step) not in events:
                raise ValueError(f"missing {event} at step {step}")
    for step in (start, stop):
        for mode in MODES:
            row = events.get((f"mode_validation_eval:{mode}", step))
            if row is None or row.get("examples") != 60:
                raise ValueError(f"missing 60-row mode validation at step {step}: {mode}")
    full = events.get(("full_validation_eval", stop))
    if full is None or full.get("scored_tokens", 0) <= 0:
        raise ValueError("missing full-252 validation loss at segment stop")
    saved = events.get(("checkpoint", stop))
    if saved is None or saved.get("tree_sha256") != checkpoint["tree_sha256"]:
        raise ValueError("checkpoint telemetry does not match sealed checkpoint")

    return {
        "schema_version": "plan31_segment_verification_v1",
        "start_step": start,
        "stop_step": stop,
        "successful_exposures": expected_exposures,
        "final_partial_accumulation": 4 if stop == 704 else 0,
        "optimization_events": 88,
        "fixed_loss_steps": sorted(fixed_steps),
        "mode_loss_steps": [start, stop],
        "full_validation_loss": full["loss"],
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "schedule_sha256": schedule_hash,
        "exposures_sha256": sha256(exposures_path),
        "telemetry_sha256": sha256(telemetry_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--start-step", required=True, type=int)
    parser.add_argument("--stop-step", required=True, type=int)
    args = parser.parse_args()
    receipt = verify(args.output_dir, args.schedule, args.start_step, args.stop_step)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
