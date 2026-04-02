import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torchcodec", reason="Slow audio tests require torchcodec for dataset audio decoding.")


def _ensure_tiny_dataset(base_dir: Path) -> None:
    ds_dir = base_dir / "data" / "datasets" / "test_streaming"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Generate a 1-second 16kHz sine wave and write to WAV
    audio_dir = ds_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.linspace(0, 1.0, int(sr), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    wav_path = audio_dir / "tiny.wav"
    sf.write(wav_path.as_posix(), audio, sr)

    # Minimal train split with required columns: id, audio_path, text_perfect
    train_csv = ds_dir / "train.csv"
    if not train_csv.exists():
        with train_csv.open("w") as f:
            f.write("id,audio_path,text_perfect\n")
            f.write(f"1,{wav_path.as_posix()},hello world\n")
    validation_csv = ds_dir / "validation.csv"
    if not validation_csv.exists():
        with validation_csv.open("w") as f:
            f.write("id,audio_path,text_perfect\n")
            f.write(f"1,{wav_path.as_posix()},hello world\n")


@pytest.mark.slow
def test_distillation_single_step(tmp_path: Path):
    # Use tmp_path (provided by pytest) instead of cwd so test artifacts don't
    # accumulate in the repo between runs and cleanup is automatic.
    base_dir = tmp_path
    _ensure_tiny_dataset(base_dir)

    # Minimal profile_config for distillation: small student, large teacher, one step
    profile_config = {
        "model": "distil-tiny-from-medium",  # name not used directly by module
        "base_model": "openai/whisper-tiny",  # student
        "teacher_model": "openai/whisper-medium",  # full HuggingFace ID required; config.ini not used here
        "dataset": "test_streaming",
        "text_column": "text_perfect",
        "train_split": "train",
        "validation_split": "validation",
        "max_label_length": 64,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 1,
        "learning_rate": 1e-5,
        "warmup_steps": 0,
        "logging_steps": 1,
        "save_steps": 1000,
        "save_total_limit": 1,
        # keep defaults to float32/eager for MPS safety
    }

    # tmp_path is a fresh per-test directory; no manual cleanup needed
    out_dir = base_dir / "output" / "test_distill_workflow"
    out_dir.mkdir(parents=True, exist_ok=True)

    from whisper_tuner.models.distil_whisper.finetune import main as distil_main

    distil_main(profile_config, out_dir.as_posix())

    # Assert a successful run by checking for train_results.json
    results_path = out_dir / "train_results.json"
    assert results_path.exists(), "Expected training results to be written"
    with results_path.open() as f:
        _ = json.load(f)
