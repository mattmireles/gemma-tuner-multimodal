"""Verify one immutable segment of the three-epoch Plan 33 E2B lineage.

Sealed segments are 88 aligned steps; the disposable smoke is one step at
0->1 or 1->2. Every optimizer step must carry finite loss, grad norm, LR,
exposure accounting, mode counts, scored tokens, wall time, and peak memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from gemma_tuner.models.common.plan31_input_modes import MODES
from gemma_tuner.utils.checkpoints import verify_complete_checkpoint

STEPS_PER_EPOCH = 352
ROWS_PER_EPOCH = 2812
ACCUMULATION = 8
EPOCH_ENDS = (352, 704, 1056)
STEP_FIELDS = (
    "stochastic_loss", "grad_norm", "learning_rate", "step_exposures", "source_ordinal_range",
    "mode_counts", "scored_tokens", "wall_time_unix", "step_seconds", "peak_accelerator_bytes",
)


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


def segment_bounds(epoch: int, start: int, stop: int) -> tuple[int, int, bool]:
    """Return (base ordinal, end ordinal, smoke) or raise on a misaligned segment."""
    if epoch not in (1, 2, 3):
        raise ValueError("Plan 33 has exactly three epochs")
    epoch_start, epoch_end = STEPS_PER_EPOCH * (epoch - 1), STEPS_PER_EPOCH * epoch
    smoke = stop <= 2
    if smoke:
        if epoch != 1 or (start, stop) not in ((0, 1), (1, 2), (0, 2)):
            raise ValueError("Plan 33 smoke is limited to epoch-one steps 0->1, 1->2, and 0->2")
    elif (start < epoch_start or start >= epoch_end or stop - start != 88
            or (start - epoch_start) % 88 or stop > epoch_end):
        raise ValueError("Plan 33 segment is not one aligned 88-step interval in its epoch")
    base = (start - epoch_start) * ACCUMULATION
    end = min((stop - epoch_start) * ACCUMULATION, ROWS_PER_EPOCH)
    return base, end, smoke


def verify(output: Path, schedule_path: Path, *, epoch: int, start: int, stop: int) -> dict:
    base_ordinal, end_ordinal, smoke = segment_bounds(epoch, start, stop)
    checkpoint = verify_complete_checkpoint(output / f"checkpoint-{stop}")
    if int(checkpoint["global_step"]) != stop:
        raise ValueError("checkpoint global step differs from segment stop")
    expected_count = end_ordinal - base_ordinal
    schedule = read_jsonl(schedule_path)
    schedule_hash = sha(schedule_path)
    if len(schedule) != ROWS_PER_EPOCH:
        raise ValueError("Plan 33 schedule must contain 2,812 rows")
    exposures = read_jsonl(output / "exposures.jsonl")
    if len(exposures) != expected_count:
        raise ValueError("Plan 33 successful exposure count changed")
    for local, record in enumerate(exposures, start=1):
        ordinal = base_ordinal + local
        frozen = schedule[ordinal - 1]
        if (record["exposure"] != local or record["source_ordinal"] != ordinal
                or record["epoch"] != epoch or record["mode"] != frozen["mode"]
                or record["schedule_sha256"] != schedule_hash
                or record["row_key_sha256"] != hashlib.sha256(str(frozen["id"]).encode()).hexdigest()):
            raise ValueError("Plan 33 exposure ledger differs from its frozen epoch schedule")
    exposure_receipt = json.loads((output / "plan32_exposure_verification.json").read_text())
    if (exposure_receipt["exposures"] != expected_count
            or exposure_receipt["start_ordinal"] != base_ordinal + 1
            or exposure_receipt["end_ordinal"] != end_ordinal
            or exposure_receipt["schedule_sha256"] != schedule_hash
            or exposure_receipt["ledger_sha256"] != sha(output / "exposures.jsonl")):
        raise ValueError("Plan 33 exposure receipt does not bind the segment ledger")

    telemetry_path = output / "validation_telemetry.jsonl"
    events = {}
    for row in read_jsonl(telemetry_path):
        key = (row["event"], int(row["step"]))
        if key in events:
            raise ValueError(f"duplicate Plan 33 telemetry event: {key}")
        if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            raise ValueError("Plan 33 telemetry contains a non-finite metric")
        if row["event"].endswith("_eval") or row["event"].startswith("mode_validation_eval:"):
            if row.get("scored_tokens", 0) <= 0 or not math.isfinite(row["loss"]):
                raise ValueError("Plan 33 evaluation loss is missing a finite token denominator")
        events[key] = row

    consumed = 0
    step_seconds, peaks = [], []
    for step in range(start + 1, stop + 1):
        row = events.get(("optimization", step))
        if row is None or any(key not in row for key in STEP_FIELDS):
            raise ValueError(f"missing complete optimization event at step {step}")
        want = ACCUMULATION if step not in EPOCH_ENDS else ROWS_PER_EPOCH - (STEPS_PER_EPOCH - 1) * ACCUMULATION
        first = base_ordinal + consumed + 1
        if (row["step_exposures"] != want or row["source_ordinal_range"] != [first, first + want - 1]
                or sum(row["mode_counts"].values()) != want or row["scored_tokens"] <= 0):
            raise ValueError(f"optimization accounting mismatch at step {step}")
        frozen_modes = Counter(schedule[i - 1]["mode"] for i in range(first, first + want))
        if row["mode_counts"] != dict(sorted(frozen_modes.items())):
            raise ValueError(f"optimization mode counts differ from schedule at step {step}")
        consumed += want
        step_seconds.append(row["step_seconds"])
        peaks.append(row["peak_accelerator_bytes"])
    if consumed != expected_count:
        raise ValueError("optimization events do not account for every exposure")

    fixed_steps = {start, stop} if smoke else {start, stop, *range(start + 22, stop + 1, 22)}
    for step in fixed_steps:
        for name in ("fixed_train_eval", "fixed_validation_eval"):
            if (name, step) not in events:
                raise ValueError(f"missing {name} at step {step}")
    mode_steps = [step for step in (start, stop) if (f"mode_validation_eval:{MODES[0]}", step) in events]
    if not smoke and mode_steps != [start, stop]:
        raise ValueError("sealed segments require six-mode losses at start and stop")
    for step in mode_steps:
        for mode in MODES:
            row = events.get((f"mode_validation_eval:{mode}", step))
            if row is None or int(row.get("examples", 0)) != 60:
                raise ValueError(f"missing 60-row {mode} validation at step {step}")
    full_event = events.get(("full_validation_eval", stop))
    if stop in EPOCH_ENDS and (full_event is None or int(full_event.get("examples", 0)) != 252):
        raise ValueError("missing full-252 validation loss at epoch end")
    checkpoint_event = events.get(("checkpoint", stop))
    if checkpoint_event is None or checkpoint_event["tree_sha256"] != checkpoint["tree_sha256"]:
        raise ValueError("checkpoint telemetry tree hash mismatch")

    return {
        "schema_version": "plan33_segment_verification_v1",
        "smoke": smoke,
        "epoch": epoch,
        "start_step": start,
        "stop_step": stop,
        "cumulative_exposures": ROWS_PER_EPOCH * (epoch - 1) + end_ordinal,
        "successful_exposures": expected_count,
        "source_ordinal_range": [base_ordinal + 1, end_ordinal],
        "final_partial_accumulation": 4 if stop in EPOCH_ENDS else 0,
        "optimization_events": stop - start,
        "step_metrics": {step: {key: events[("optimization", step)][key]
                                for key in ("stochastic_loss", "grad_norm", "scored_tokens")}
                         for step in range(start + 1, stop + 1)},
        "median_step_seconds": sorted(step_seconds)[len(step_seconds) // 2],
        "max_peak_accelerator_bytes": max(peaks),
        "fixed_loss_steps": sorted(fixed_steps),
        "mode_loss_steps": mode_steps,
        "full_validation_loss": None if full_event is None else full_event["loss"],
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "schedule_sha256": schedule_hash,
        "exposures_sha256": sha(output / "exposures.jsonl"),
        "telemetry_sha256": sha(telemetry_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--start-step", required=True, type=int)
    parser.add_argument("--stop-step", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir, args.schedule, epoch=args.epoch,
                            start=args.start_step, stop=args.stop_step), sort_keys=True))


if __name__ == "__main__":
    main()
