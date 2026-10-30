from __future__ import annotations

from types import SimpleNamespace

import pytest

from gemma_tuner.utils.exposure_ledger import (
    ExposureCommitCallback,
    ExposureLedger,
    ExposureTrackingCollator,
)


def test_exposure_commits_only_after_successful_substep_and_resumes(tmp_path) -> None:
    path = tmp_path / "exposures.jsonl"
    ledger = ExposureLedger(path)
    collator = ExposureTrackingCollator(lambda rows: {"count": len(rows)}, ledger)
    callback = ExposureCommitCallback(ledger)

    assert collator([{"id": "a"}]) == {"count": 1}
    assert not path.exists()
    callback.on_substep_end(None, None, SimpleNamespace())
    assert len(path.read_text().splitlines()) == 1

    resumed = ExposureLedger(path)
    with pytest.raises(RuntimeError, match="more than once"):
        ExposureTrackingCollator(lambda rows: rows, resumed)([{"id": "a"}])
    ExposureTrackingCollator(lambda rows: rows, resumed)([{"id": "b"}])
    callback = ExposureCommitCallback(resumed)
    callback.on_step_end(None, None, SimpleNamespace())
    assert resumed.verify_complete(["a", "b"])["exposures"] == 2


def test_failed_batch_is_not_committed(tmp_path) -> None:
    ledger = ExposureLedger(tmp_path / "exposures.jsonl")

    def fail(_rows):
        raise RuntimeError("collation failed")

    with pytest.raises(RuntimeError, match="collation failed"):
        ExposureTrackingCollator(fail, ledger)([{"id": "a"}])
    assert ledger.pending == []
    assert not ledger.path.exists()


def test_corrupt_or_incomplete_ledgers_fail_closed(tmp_path) -> None:
    path = tmp_path / "exposures.jsonl"
    path.write_text('{"schema_version":"gemma_training_exposure_v1","exposure":2,"row_key_sha256":"' + "0" * 64 + '"}\n')
    with pytest.raises(ValueError, match="contiguous"):
        ExposureLedger(path)

    ledger = ExposureLedger(tmp_path / "clean.jsonl")
    ExposureTrackingCollator(lambda rows: rows, ledger)([{"id": "a"}])
    ExposureCommitCallback(ledger).on_step_end(None, None, SimpleNamespace())
    with pytest.raises(ValueError, match="exactly once"):
        ledger.verify_complete(["a", "b"])


def test_accumulation_prefetch_commits_fifo(tmp_path) -> None:
    ledger = ExposureLedger(tmp_path / "exposures.jsonl")
    collator = ExposureTrackingCollator(lambda rows: rows, ledger)
    callback = ExposureCommitCallback(ledger)
    for value in range(8):
        collator([{"id": value}])
    assert len(ledger.pending) == 8
    for _ in range(7):
        callback.on_substep_end(None, None, SimpleNamespace())
    callback.on_step_end(None, None, SimpleNamespace())
    assert ledger.pending == []
    assert ledger.verify_complete(range(8))["exposures"] == 8


def test_train_end_discards_only_max_step_prefetch(tmp_path) -> None:
    ledger = ExposureLedger(tmp_path / "exposures.jsonl")
    ExposureTrackingCollator(lambda rows: rows, ledger)([{"id": "prefetched"}])
    callback = ExposureCommitCallback(ledger)

    with pytest.raises(RuntimeError, match="uncommitted exposure"):
        callback.on_train_end(
            None, SimpleNamespace(global_step=7, max_steps=8), SimpleNamespace()
        )
    assert len(ledger.pending) == 1

    callback.on_train_end(
        None, SimpleNamespace(global_step=8, max_steps=8), SimpleNamespace()
    )
    assert ledger.pending == []
    assert not ledger.path.exists()


def test_train_end_uses_active_segment_max_steps_not_resumed_state(tmp_path) -> None:
    ledger = ExposureLedger(tmp_path / "exposures.jsonl")
    ExposureTrackingCollator(lambda rows: rows, ledger)([{"id": "prefetched"}])
    callback = ExposureCommitCallback(ledger)

    callback.on_train_end(
        SimpleNamespace(max_steps=264),
        SimpleNamespace(global_step=264, max_steps=352),
        SimpleNamespace(),
    )
    assert ledger.pending == []
