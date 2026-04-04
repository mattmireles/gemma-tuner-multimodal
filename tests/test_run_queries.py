import json
import os

from whisper_tuner.core.run_queries import RunQuery, build_overview, cleanup_runs, get_run_details, list_runs


def _write_metadata(path, metadata):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metadata.json"), "w") as handle:
        json.dump(metadata, handle)


def test_list_runs_discovers_nested_and_model_dataset_runs(tmp_path):
    output_dir = tmp_path / "output"

    train_dir = output_dir / "1-profile-a"
    _write_metadata(
        train_dir,
        {
            "run_id": "1",
            "run_type": "finetuning",
            "status": "completed",
            "profile": "profile-a",
            "model": "whisper-tiny",
            "dataset": "dataset-a",
            "start_time": "2026-04-01 10:00:00",
        },
    )
    _write_metadata(
        train_dir / "eval",
        {
            "run_id": "2",
            "run_type": "evaluation",
            "status": "completed",
            "profile": "profile-a",
            "model": "whisper-tiny",
            "dataset": "dataset-a",
            "finetuning_run_id": "1",
            "start_time": "2026-04-01 11:00:00",
            "metrics": {"wer": 0.12},
        },
    )
    _write_metadata(
        output_dir / "whisper-base+dataset-b" / "eval",
        {
            "run_id": "3",
            "run_type": "evaluation",
            "status": "completed",
            "profile": "",
            "start_time": "2026-04-02 11:00:00",
            "metrics": {"wer": 0.08},
        },
    )

    rows = list_runs(str(output_dir), RunQuery.from_filters(type="evaluation"))
    assert {row.run_id for row in rows} == {"2", "3"}
    model_dataset_row = next(row for row in rows if row.run_id == "3")
    assert model_dataset_row.model == "whisper-base"
    assert model_dataset_row.dataset == "dataset-b"


def test_run_overview_and_details_use_typed_queries(tmp_path):
    output_dir = tmp_path / "output"

    _write_metadata(
        output_dir / "1-profile-a",
        {
            "run_id": "1",
            "run_type": "finetuning",
            "status": "completed",
            "profile": "profile-a",
            "model": "whisper-tiny",
            "dataset": "toy",
            "start_time": "2026-04-01 10:00:00",
        },
    )
    _write_metadata(
        output_dir / "1-profile-a" / "eval",
        {
            "run_id": "2",
            "run_type": "evaluation",
            "status": "completed",
            "profile": "profile-a",
            "model": "whisper-tiny",
            "dataset": "toy",
            "finetuning_run_id": "1",
            "start_time": "2026-04-01 11:00:00",
            "metrics": {"wer": 0.15},
        },
    )
    _write_metadata(
        output_dir / "2-profile-b" / "eval",
        {
            "run_id": "3",
            "run_type": "evaluation",
            "status": "completed",
            "profile": "profile-b",
            "model": "whisper-base",
            "dataset": "toy",
            "finetuning_run_id": "2",
            "start_time": "2026-04-02 11:00:00",
            "metrics": {"wer": 0.05},
        },
    )

    overview = build_overview(str(output_dir), RunQuery.from_filters(dataset="toy"))
    assert overview.total_runs == 3
    assert overview.finetuning_runs == 1
    assert overview.evaluation_runs == 2
    assert overview.average_wer == 0.1
    assert {best.run_id for best in overview.best_runs} == {"2", "3"}

    details = get_run_details(str(output_dir), "2")
    assert details is not None
    assert details.metadata["finetuning_run_id"] == "1"


def test_cleanup_runs_deletes_failed_and_cancelled(tmp_path):
    output_dir = tmp_path / "output"
    failed_dir = output_dir / "1-failed"
    cancelled_dir = output_dir / "2-cancelled"
    completed_dir = output_dir / "3-completed"

    _write_metadata(failed_dir, {"run_id": "1", "status": "failed"})
    _write_metadata(cancelled_dir, {"run_id": "2", "status": "cancelled"})
    _write_metadata(completed_dir, {"run_id": "3", "status": "completed"})
    with open(failed_dir / "artifact.bin", "wb") as handle:
        handle.write(b"123456")

    result = cleanup_runs(str(output_dir))

    assert sorted(item.status for item in result.deleted_runs) == ["cancelled", "failed"]
    assert result.total_bytes_freed >= 6
    assert not failed_dir.exists()
    assert not cancelled_dir.exists()
    assert completed_dir.exists()
