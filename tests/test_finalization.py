import json
import os

from whisper_tuner.core.finalization import finalize_evaluation_run, finalize_training_run
from whisper_tuner.core.runs import RunConstants, create_run_directory


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def test_finalize_training_run_merges_metrics_and_exports_once(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    run_dir = create_run_directory(str(output_dir), "profile-a", 1, RunConstants.RUN_TYPE_FINETUNING)

    export_calls = []

    def fake_gguf(path):
        export_calls.append(("gguf", path))
        return os.path.join(path, "model.gguf")

    def fake_coreml(path):
        export_calls.append(("coreml", path))
        return os.path.join(path, "coreml", "encoder.mlpackage")

    monkeypatch.setattr("whisper_tuner.core.finalization.ops.export_gguf", fake_gguf)
    monkeypatch.setattr("whisper_tuner.core.finalization.ops.export_coreml", fake_coreml)

    result = finalize_training_run(
        run_dir,
        str(output_dir),
        profile_config={"export_gguf": True, "export_coreml": True},
        training_result={"train_metrics": {"loss": 0.42}},
        duration_sec=12.345,
    )

    assert export_calls == [("gguf", run_dir), ("coreml", run_dir)]
    assert result.train_metrics["loss"] == 0.42
    assert result.train_metrics["duration_sec"] == 12.345

    metrics = _read_json(os.path.join(run_dir, "metrics.json"))
    assert metrics["train"]["loss"] == 0.42

    metadata = _read_json(os.path.join(run_dir, "metadata.json"))
    assert metadata["status"] == "completed"
    assert metadata["export_results"]["gguf"]["path"].endswith("model.gguf")
    assert metadata["export_results"]["coreml"]["path"].endswith("encoder.mlpackage")


def test_finalize_training_run_respects_disabled_exports(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    run_dir = create_run_directory(str(output_dir), "profile-b", 2, RunConstants.RUN_TYPE_FINETUNING)

    monkeypatch.setattr(
        "whisper_tuner.core.finalization.ops.export_gguf",
        lambda path: (_ for _ in ()).throw(AssertionError("GGUF export should not run")),
    )
    monkeypatch.setattr(
        "whisper_tuner.core.finalization.ops.export_coreml",
        lambda path: (_ for _ in ()).throw(AssertionError("CoreML export should not run")),
    )

    result = finalize_training_run(
        run_dir,
        str(output_dir),
        profile_config={"export_gguf": False, "export_coreml": False},
        training_result={"train_metrics": {"loss": 0.11}},
    )

    assert result.export_results["gguf"].attempted is False
    assert result.export_results["coreml"].attempted is False


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
