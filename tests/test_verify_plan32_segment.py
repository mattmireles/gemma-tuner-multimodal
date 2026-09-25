"""Plan 32 segment seals bind exposures, mode losses, and checkpoint identity."""

import hashlib
import json

import pytest

from tools import verify_plan32_segment as segment


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path, monkeypatch, *, epoch=1, start=0, stop=88):
    output = tmp_path / "run"
    output.mkdir()
    schedule_path = tmp_path / "schedule.jsonl"
    schedule = [
        {"epoch": epoch, "source_ordinal": i, "id": f"row-{i}",
         "mode": "full", "image_sha256": "image"}
        for i in range(1, 2813)
    ]
    _write_jsonl(schedule_path, schedule)
    schedule_hash = segment.sha(schedule_path)
    epoch_start = 0 if epoch == 1 else 352
    start_ordinal = (start - epoch_start) * 8 + 1
    end_ordinal = min((stop - epoch_start) * 8, 2812)
    exposures = [
        {"exposure": i, "source_ordinal": start_ordinal + i - 1,
         "epoch": epoch, "mode": "full", "schedule_sha256": schedule_hash,
         "row_key_sha256": hashlib.sha256(f"row-{start_ordinal + i - 1}".encode()).hexdigest()}
        for i in range(1, end_ordinal - start_ordinal + 2)
    ]
    exposure_path = output / "exposures.jsonl"
    _write_jsonl(exposure_path, exposures)
    (output / "plan32_exposure_verification.json").write_text(json.dumps({
        "exposures": end_ordinal - start_ordinal + 1,
        "start_ordinal": start_ordinal, "end_ordinal": end_ordinal,
        "schedule_sha256": schedule_hash,
        "ledger_sha256": segment.sha(exposure_path),
    }), encoding="utf-8")

    events = [
        {"event": "optimization", "step": step, "stochastic_loss": 0.5,
         "grad_norm": 1.0, "learning_rate": 0.0001}
        for step in range(start + 1, stop + 1)
    ]
    fixed_steps = {start, stop, *range(start + 22, stop + 1, 22)}
    for step in fixed_steps:
        for name in ("fixed_train_eval", "fixed_validation_eval"):
            events.append({"event": name, "step": step, "loss": 0.5, "scored_tokens": 100})
    for step in (start, stop):
        for mode in segment.MODES:
            events.append({"event": f"mode_validation_eval:{mode}", "step": step,
                           "examples": 60, "loss": 0.5, "scored_tokens": 100})
    if stop in (352, 704):
        events.append({"event": "full_validation_eval", "step": stop,
                       "examples": 252, "loss": 0.5, "scored_tokens": 1000})
    events.append({"event": "checkpoint", "step": stop, "tree_sha256": "tree"})
    _write_jsonl(output / "validation_telemetry.jsonl", events)
    monkeypatch.setattr(segment, "verify_complete_checkpoint", lambda _: {"global_step": stop, "tree_sha256": "tree"})
    return output, schedule_path, events


def test_plan32_segment_verifier_accepts_complete_segment(tmp_path, monkeypatch):
    output, schedule, _ = _fixture(tmp_path, monkeypatch)
    receipt = segment.verify(output, schedule, epoch=1, start=0, stop=88)
    assert receipt["successful_exposures"] == 704
    assert receipt["optimization_events"] == 88
    assert receipt["final_partial_accumulation"] == 0


def test_plan32_epoch_two_starts_at_source_ordinal_one(tmp_path, monkeypatch):
    output, schedule, _ = _fixture(tmp_path, monkeypatch, epoch=2, start=352, stop=440)
    receipt = segment.verify(output, schedule, epoch=2, start=352, stop=440)
    assert receipt["source_ordinal_range"] == [1, 704]


def test_plan32_segment_verifier_rejects_nonfinite_training_metric(tmp_path, monkeypatch):
    output, schedule, events = _fixture(tmp_path, monkeypatch)
    events[0]["grad_norm"] = float("nan")
    _write_jsonl(output / "validation_telemetry.jsonl", events)
    with pytest.raises(ValueError, match="non-finite"):
        segment.verify(output, schedule, epoch=1, start=0, stop=88)


def test_plan32_segment_verifier_requires_explicit_full_validation_count(tmp_path, monkeypatch):
    output, schedule, events = _fixture(tmp_path, monkeypatch, start=264, stop=352)
    for row in events:
        if row["event"] == "full_validation_eval":
            del row["examples"]
    _write_jsonl(output / "validation_telemetry.jsonl", events)
    with pytest.raises(ValueError, match="full-252"):
        segment.verify(output, schedule, epoch=1, start=264, stop=352)
