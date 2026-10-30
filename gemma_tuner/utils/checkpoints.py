"""Atomic closure and verification for immutable Hugging Face checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from safetensors import safe_open
from transformers import TrainerCallback

COMPLETE_MARKER = ".complete.json"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
REQUIRED_FILES = frozenset(
    {
        "adapter_config.json",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
)
MODEL_FILES = frozenset({"adapter_model.safetensors", "adapter_model.bin"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError("checkpoint directory must be named checkpoint-N")
    return int(match.group(1))


def build_checkpoint_manifest(path: Path) -> dict[str, Any]:
    step = _checkpoint_step(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {
        item.relative_to(path).as_posix(): item
        for item in path.rglob("*")
        if item.is_file() and item.name != COMPLETE_MARKER and not item.name.endswith(".tmp")
    }
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise ValueError(f"checkpoint is missing required files: {sorted(missing)}")
    if not MODEL_FILES.intersection(files):
        raise ValueError("checkpoint has no adapter model file")
    adapter_safetensors = files.get("adapter_model.safetensors")
    if adapter_safetensors is not None:
        try:
            with safe_open(adapter_safetensors, framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    raise ValueError("adapter safetensors contains no tensors")
        except Exception as exc:
            raise ValueError("adapter safetensors is unreadable") from exc
    for name in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"):
        if not zipfile.is_zipfile(files[name]):
            raise ValueError(f"checkpoint torch artifact is unreadable: {name}")
    trainer_state = json.loads(files["trainer_state.json"].read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != step:
        raise ValueError("trainer_state global_step does not match checkpoint name")
    entries = {
        name: {"bytes": file.stat().st_size, "sha256": sha256_file(file)} for name, file in sorted(files.items())
    }
    if any(entry["bytes"] <= 0 for entry in entries.values()):
        raise ValueError("checkpoint contains an empty required artifact")
    identity = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": "gemma_immutable_checkpoint_v1",
        "checkpoint": path.name,
        "global_step": step,
        "files": entries,
        "tree_sha256": hashlib.sha256(identity).hexdigest(),
    }


def seal_checkpoint(path: Path) -> dict[str, Any]:
    manifest = build_checkpoint_manifest(path)
    marker = path / COMPLETE_MARKER
    temporary = path / f"{COMPLETE_MARKER}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(marker)
    return manifest


def verify_complete_checkpoint(path: Path) -> dict[str, Any]:
    marker = path / COMPLETE_MARKER
    if not marker.is_file():
        raise ValueError("checkpoint has no atomic completion marker")
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    actual = build_checkpoint_manifest(path)
    if recorded != actual:
        raise ValueError("checkpoint contents differ from completion marker")
    return actual


class ImmutableCheckpointCallback(TrainerCallback):
    """Seal a checkpoint only after Trainer has completed its save callback."""

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        seal_checkpoint(checkpoint)
        return control


class StopAfterStepCallback(TrainerCallback):
    """Create a normal Trainer checkpoint, then end at a planned resume boundary."""

    def __init__(self, step: int) -> None:
        self.step = int(step)
        if self.step < 1:
            raise ValueError("stop-after step must be positive")

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        if state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control
