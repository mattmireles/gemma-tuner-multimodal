"""Plan 33 segment seals bind exposures, per-step accounting, and resume identity."""

import hashlib
import json
from collections import Counter

import pytest
import torch

from gemma_tuner.utils.exposure_ledger import ExposureTrackingCollator
from tools import run_plan33_segment as runner
from tools import verify_plan33_segment as segment

MODES = ("full", "no_system", "no_ocr", "no_quadrants", "image_instruction", "image_only")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path, monkeypatch, *, epoch=1, start=0, stop=88, tamper=None):
    output = tmp_path / "run"
    output.mkdir()
    schedule_path = tmp_path / "schedule.jsonl"
    schedule = [{"epoch": epoch, "source_ordinal": i, "id": f"row-{i}", "mode": MODES[i % 6]}
                for i in range(1, 2813)]
    _write_jsonl(schedule_path, schedule)
    schedule_hash = segment.sha(schedule_path)
    base, end, _ = segment.segment_bounds(epoch, start, stop)
    exposures = [{"exposure": i, "source_ordinal": base + i, "epoch": epoch, "mode": schedule[base + i - 1]["mode"],
                  "schedule_sha256": schedule_hash,
                  "row_key_sha256": hashlib.sha256(f"row-{base + i}".encode()).hexdigest()}
                 for i in range(1, end - base + 1)]
    _write_jsonl(output / "exposures.jsonl", exposures)
    (output / "plan32_exposure_verification.json").write_text(json.dumps({
        "exposures": end - base, "start_ordinal": base + 1, "end_ordinal": end,
        "schedule_sha256": schedule_hash, "ledger_sha256": segment.sha(output / "exposures.jsonl")}))
    events, cursor = [], base
    for step in range(start + 1, stop + 1):
        want = 4 if step in (352, 704, 1056) else 8
        modes = Counter(schedule[i]["mode"] for i in range(cursor, cursor + want))
        events.append({"event": "optimization", "step": step, "stochastic_loss": 1.0, "grad_norm": 1.0,
                       "learning_rate": 1e-4, "step_exposures": want, "source_ordinal_range": [cursor + 1, cursor + want],
                       "mode_counts": dict(sorted(modes.items())), "scored_tokens": 300 * want,
                       "wall_time_unix": 1.0 + step, "step_seconds": 30.0, "peak_accelerator_bytes": 10})
        cursor += want
    fixed = {start, stop} if stop - start == 1 else {start, stop, *range(start + 22, stop + 1, 22)}
    for step in fixed:
        for name in ("fixed_train_eval", "fixed_validation_eval"):
            events.append({"event": name, "step": step, "loss": 1.0, "scored_tokens": 10, "examples": 20})
    if stop - start == 88:
        for step in (start, stop):
            for mode in MODES:
                events.append({"event": f"mode_validation_eval:{mode}", "step": step, "loss": 1.0,
                               "scored_tokens": 10, "examples": 60})
    if stop in (352, 704, 1056):
        events.append({"event": "full_validation_eval", "step": stop, "loss": 0.5, "scored_tokens": 10, "examples": 252})
    events.append({"event": "checkpoint", "step": stop, "tree_sha256": "tree"})
    if tamper:
        tamper(events)
    _write_jsonl(output / "validation_telemetry.jsonl", events)
    monkeypatch.setattr(segment, "verify_complete_checkpoint",
                        lambda path: {"global_step": stop, "tree_sha256": "tree"})
    return output, schedule_path


@pytest.mark.parametrize("epoch,start,stop,cumulative", [
    (1, 0, 88, 704), (1, 264, 352, 2812), (2, 352, 440, 3516), (2, 616, 704, 5624),
    (3, 704, 792, 6328), (3, 968, 1056, 8436),
])
def test_sealed_segments_report_exact_cumulative_exposures(tmp_path, monkeypatch, epoch, start, stop, cumulative):
    output, schedule = _fixture(tmp_path, monkeypatch, epoch=epoch, start=start, stop=stop)
    result = segment.verify(output, schedule, epoch=epoch, start=start, stop=stop)
    assert result["cumulative_exposures"] == cumulative
    assert result["final_partial_accumulation"] == (4 if stop % 352 == 0 else 0)


def test_smoke_segments_are_one_step_and_epoch_one_only(tmp_path, monkeypatch):
    output, schedule = _fixture(tmp_path, monkeypatch, start=1, stop=2)
    result = segment.verify(output, schedule, epoch=1, start=1, stop=2)
    assert result["smoke"] and result["source_ordinal_range"] == [9, 16]
    for bad in ((2, 352, 353), (1, 2, 3), (1, 0, 44), (1, 1, 3)):
        with pytest.raises(ValueError):
            segment.segment_bounds(*bad)


@pytest.mark.parametrize("tamper", [
    lambda events: events[0].pop("scored_tokens"),
    lambda events: events[0].update(step_exposures=7),
    lambda events: events[0].update(mode_counts={"full": 8}),
    lambda events: events[0].update(source_ordinal_range=[2, 9]),
    lambda events: events[0].update(stochastic_loss=float("nan")),
    lambda events: events.append(dict(events[0])),
])
def test_incomplete_or_inconsistent_step_accounting_fails(tmp_path, monkeypatch, tamper):
    output, schedule = _fixture(tmp_path, monkeypatch, tamper=tamper)
    with pytest.raises(ValueError):
        segment.verify(output, schedule, epoch=1, start=0, stop=88)


def test_tracking_collator_counts_supervised_tokens_per_row():
    class Ledger:
        def stage(self, features):
            self.staged = features

    labels = torch.tensor([[-100, -100, 5, 6, 7]])
    collator = ExposureTrackingCollator(lambda rows: {"labels": labels}, Ledger())
    collator([{"id": "a"}])
    assert collator.scored_tokens == [3]


def test_runner_refuses_non_disposable_smoke_output_and_crossed_lineages(tmp_path):
    config = runner.ROOT / "config/plan33-e2b-literal-three-epochs.ini"
    with pytest.raises(ValueError, match="disposable"):
        runner.preflight(config, "plan33-smoke-step1", tmp_path / "plan33-literal-epoch1-step88")
