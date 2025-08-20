import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest


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


@pytest.mark.slow
def test_distillation_single_step(tmp_path: Path):
    base_dir = Path.cwd()
    _ensure_tiny_dataset(base_dir)

    # Minimal profile_config for distillation: small student, large teacher, one step
    profile_config = {
        "model": "distil-tiny-from-medium",  # name not used directly by module
        "base_model": "openai/whisper-tiny",  # student
        "teacher_model": "whisper-medium",    # will resolve via config.ini to openai/whisper-medium
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

    out_dir = base_dir / "output" / "test_distill_workflow"
    if out_dir.exists():
        for root, dirs, files in os.walk(out_dir.as_posix(), topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)

    from models.distil_whisper.finetune import main as distil_main

    distil_main(profile_config, out_dir.as_posix())

    # Assert a successful run by checking for train_results.json
    results_path = out_dir / "train_results.json"
    assert results_path.exists(), "Expected training results to be written"
    with results_path.open() as f:
        _ = json.load(f)
