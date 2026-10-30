from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file

from gemma_tuner.utils.checkpoints import seal_checkpoint
from tools.courier_checkpoint import upload_checkpoint


def write_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"peft_type":"LORA"}')
    save_file({"adapter": torch.ones(2)}, checkpoint / "adapter_model.safetensors")
    for name in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"):
        torch.save({"step": 2}, checkpoint / name)
    (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
    seal_checkpoint(checkpoint)
    return checkpoint


def test_courier_uploads_only_complete_checkpoint_and_verifies_remote_marker(
    tmp_path: Path,
) -> None:
    checkpoint = write_checkpoint(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        stdout = (
            (checkpoint / ".complete.json").read_text() if command[2:4] == ["cat", "--project=gist-is-backend"] else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    contract = {
        "gcp": {
            "project": "gist-is-backend",
            "account": "whisper-gcp-sft-runner@gist-is-backend.iam.gserviceaccount.com",
        }
    }
    entry = upload_checkpoint(
        checkpoint,
        "gs://bucket/experiment/checkpoints",
        contract,
        ledger,
        runner=runner,
    )
    assert entry["checkpoint"] == "checkpoint-2"
    assert len(commands) == 2
    for command in commands:
        assert "--project=gist-is-backend" in command
        assert "--account=whisper-gcp-sft-runner@gist-is-backend.iam.gserviceaccount.com" in command
    assert json.loads(ledger.read_text())["tree_sha256"] == entry["tree_sha256"]
    assert (
        upload_checkpoint(
            checkpoint,
            "gs://bucket/experiment/checkpoints",
            contract,
            ledger,
            runner=runner,
        )
        == entry
    )
    assert len(commands) == 2
