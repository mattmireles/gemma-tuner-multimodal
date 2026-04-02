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
    validation_csv = ds_dir / "validation.csv"
    if not validation_csv.exists():
        with validation_csv.open("w") as f:
            f.write("id,audio_path,text_perfect\n")
            f.write(f"1,{wav_path.as_posix()},hello world\n")


@pytest.mark.slow
def test_lora_single_step(tmp_path: Path):
    base_dir = Path.cwd()
    _ensure_tiny_dataset(base_dir)

    # Minimal LoRA profile_config targeting whisper-tiny
    profile_config = {
        "model": "whisper-tiny-lora",
        "base_model": "openai/whisper-tiny",
        "dataset": "test_streaming",
        "text_column": "text_perfect",
        "id_column": "id",
        "train_split": "train",
        "validation_split": "validation",
        "max_label_length": 64,
        "max_duration": 10.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "num_train_epochs": 1,
        "learning_rate": 1e-4,
        "warmup_steps": 0,
        "save_steps": 1000,
        "save_total_limit": 1,
        "logging_steps": 1,
        "gradient_accumulation_steps": 1,
        # Disable gradient checkpointing in E2E smoke to avoid MPS double-backward issues
        "gradient_checkpointing": False,
        "dtype": "float32",
        "attn_implementation": "eager",
        "language_mode": "mixed",
        "max_samples": 1,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        # LoRA-specific
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
        "enable_8bit": False,
        "visualize": False,
    }

    out_dir = base_dir / "output" / "test_lora_workflow"
    if out_dir.exists():
        # best-effort cleanup
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

    from whisper_tuner.models.whisper_lora.finetune import main as lora_main

    lora_main(profile_config, out_dir.as_posix())

    # Assert a successful run by checking for train_results.json
    results_path = out_dir / "train_results.json"
    assert results_path.exists(), "Expected training results to be written"
    with results_path.open() as f:
        _ = json.load(f)
