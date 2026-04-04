import json
import os

from gemma_tuner.core.finalization import finalize_evaluation_run, finalize_training_run
from gemma_tuner.core.runs import RunConstants, create_run_directory


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def test_finalize_training_run_merges_metrics(tmp_path):
    output_dir = tmp_path / "output"
    run_dir = create_run_directory(str(output_dir), "profile-a", 1, RunConstants.RUN_TYPE_FINETUNING)

    result = finalize_training_run(
        run_dir,
        str(output_dir),
        training_result={"train_metrics": {"loss": 0.42}},
        duration_sec=12.345,
    )

    assert result.train_metrics["loss"] == 0.42
    assert result.train_metrics["duration_sec"] == 12.345

    metrics = _read_json(os.path.join(run_dir, "metrics.json"))
    assert metrics["train"]["loss"] == 0.42

    metadata = _read_json(os.path.join(run_dir, "metadata.json"))
    assert metadata["status"] == "completed"


def test_finalize_evaluation_run_persists_metrics(tmp_path):
    output_dir = tmp_path / "output"
    create_run_directory(str(output_dir), "profile-c", 2, RunConstants.RUN_TYPE_FINETUNING)
    run_dir = create_run_directory(str(output_dir), "profile-c", 3, RunConstants.RUN_TYPE_EVALUATION)

    finalize_evaluation_run(run_dir, str(output_dir), {"wer": 0.07, "cer": 0.03})

    metrics = _read_json(os.path.join(run_dir, "metrics.json"))
    assert metrics["wer"] == 0.07
    metadata = _read_json(os.path.join(run_dir, "metadata.json"))
    assert metadata["metrics"]["wer"] == 0.07
    assert metadata["status"] == "completed"
