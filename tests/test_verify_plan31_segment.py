"""Sealed Plan 31 segments must retain every training and validation metric."""

import hashlib
import json

import pytest

from tools import verify_plan31_segment as segment


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    schedule_path = tmp_path / "schedule.jsonl"
    _write_jsonl(schedule_path, [{"mode": "image_only"}] * 2812)
    schedule_hash = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    exposures = [
        {"exposure": index, "source_ordinal": index, "mode": "image_only",
         "schedule_sha256": schedule_hash}
        for index in range(1, 705)
    ]
    exposure_path = output / "exposures.jsonl"
    _write_jsonl(exposure_path, exposures)
    (output / "plan31_exposure_verification.json").write_text(json.dumps({
        "exposures": 704, "start_ordinal": 1, "end_ordinal": 704,
        "ledger_sha256": segment.sha256(exposure_path),
    }), encoding="utf-8")
    events = [
        {"event": "optimization", "step": step, "stochastic_loss": 0.5,
         "grad_norm": 1.0, "learning_rate": 0.0001}
        for step in range(353, 441)
    ]
    for step in (352, 374, 396, 418, 440):
        for event in ("fixed_train_eval", "fixed_validation_eval"):
            events.append({"event": event, "step": step, "loss": 0.5, "scored_tokens": 100})
    for step in (352, 440):
        for mode in segment.MODES:
            events.append({"event": f"mode_validation_eval:{mode}", "step": step,
                           "examples": 60, "loss": 0.5, "scored_tokens": 100})
    events.extend((
        {"event": "full_validation_eval", "step": 440, "loss": 0.5, "scored_tokens": 1000},
        {"event": "checkpoint", "step": 440, "tree_sha256": "tree"},
    ))
    _write_jsonl(output / "validation_telemetry.jsonl", events)
    monkeypatch.setattr(segment, "verify_complete_checkpoint", lambda _: {"tree_sha256": "tree"})
    return output, schedule_path, events


def test_segment_verifier_accepts_complete_metrics_and_checkpoint(tmp_path, monkeypatch):
    output, schedule, _ = _fixture(tmp_path, monkeypatch)
    receipt = segment.verify(output, schedule, 352, 440)
    assert receipt["successful_exposures"] == 704
    assert receipt["optimization_events"] == 88
    assert receipt["mode_loss_steps"] == [352, 440]


def test_segment_verifier_rejects_missing_grad_norm(tmp_path, monkeypatch):
    output, schedule, events = _fixture(tmp_path, monkeypatch)
    del events[0]["grad_norm"]
    _write_jsonl(output / "validation_telemetry.jsonl", events)
    with pytest.raises(ValueError, match="missing training metrics"):
        segment.verify(output, schedule, 352, 440)


def test_segment_verifier_rejects_missing_mode_loss(tmp_path, monkeypatch):
    output, schedule, events = _fixture(tmp_path, monkeypatch)
    events = [row for row in events if not (
        row["event"] == "mode_validation_eval:image_only" and row["step"] == 440
    )]
    _write_jsonl(output / "validation_telemetry.jsonl", events)
    with pytest.raises(ValueError, match="missing 60-row mode validation"):
        segment.verify(output, schedule, 352, 440)
