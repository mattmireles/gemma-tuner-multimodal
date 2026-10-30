"""Preflight and run exactly one Plan 32 88-step segment."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(config_path: Path, profile_name: str, output_dir: Path) -> dict:
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.utils.plan32_projection import verify_projection
    from gemma_tuner.utils.checkpoints import verify_complete_checkpoint

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    profile = load_profile_config(config, profile_name)
    if int(profile.get("plan32_epoch", 0)) not in (1, 2):
        raise ValueError("profile is not a Plan 32 epoch segment")
    epoch = int(profile["plan32_epoch"])
    start, stop = int(profile["telemetry_start_step"]), int(profile["stop_after_step"])
    expected_dataset = f"tt-screenshot-plan32-literal-v1-deploy/epoch-{epoch}"
    if str(profile.get("source")) != expected_dataset:
        raise ValueError("Plan 32 schedule epoch and training projection differ")
    if int(profile.get("max_steps", -1)) != stop:
        raise ValueError("max_steps must equal the sealed absolute stop step")

    schedule = Path(str(profile["plan32_schedule_path"]))
    if not schedule.is_absolute():
        schedule = ROOT / schedule
    schedule = schedule.resolve(strict=True)
    if sha(schedule) != str(profile["plan32_schedule_sha256"]):
        raise ValueError("Plan 32 frozen schedule hash mismatch")
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2812 or any(int(row["epoch"]) != epoch for row in rows):
        raise ValueError("Plan 32 schedule has wrong row count or epoch")

    dataset_dir = ROOT / "data/datasets" / expected_dataset
    receipt_path = Path(str(profile["plan32_projection_receipt_path"]))
    if not receipt_path.is_absolute():
        receipt_path = ROOT / receipt_path
    receipt_path = receipt_path.resolve(strict=True)
    if receipt_path != (dataset_dir / "projection.receipt.json").resolve(strict=True):
        raise ValueError("Plan 32 projection receipt path is outside the epoch dataset")
    if sha(receipt_path) != str(profile["plan32_projection_receipt_sha256"]):
        raise ValueError("Plan 32 projection receipt hash mismatch")
    receipt = verify_projection(
        dataset_dir,
        expected_receipt_sha256=str(profile["plan32_projection_receipt_sha256"]),
        epoch=epoch,
    )
    if receipt["schedule_sha256"] != sha(schedule):
        raise ValueError("Plan 32 projection receipt refers to another schedule")
    for name in ("train.csv", "validation.csv", *(f"validation-panel-{mode}.csv" for mode in (
            "full", "no_ocr", "no_quadrants", "no_system", "image_instruction", "image_only"))):
        if sha(dataset_dir / name) != receipt["outputs_sha256"][name]:
            raise ValueError(f"Plan 32 projected dataset bytes changed: {name}")

    resume = profile.get("resume_from_checkpoint")
    if resume:
        checkpoint_path = Path(str(resume))
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint = verify_complete_checkpoint(checkpoint_path.resolve(strict=True))
        state = json.loads((checkpoint_path / "trainer_state.json").read_text(encoding="utf-8"))
        if int(checkpoint["global_step"]) != start or int(state["global_step"]) != start:
            raise ValueError("Plan 32 resume checkpoint does not match segment start")
    elif not (epoch == 1 and start == 0 and profile.get("initial_adapter_path")):
        raise ValueError("only the first epoch segment may start without an optimizer checkpoint")
    else:
        adapter = Path(str(profile["initial_adapter_path"]))
        if not adapter.is_absolute():
            adapter = ROOT / adapter
        verify_complete_checkpoint(adapter.resolve(strict=True))

    import transformers

    if transformers.__version__ != str(profile["required_transformers_version"]):
        raise RuntimeError("Plan 32 requires the pinned Transformers version")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Plan 32 output directory is not empty: {output_dir}")
    return {
        "epoch": epoch,
        "start_step": start,
        "stop_step": stop,
        "schedule_sha256": sha(schedule),
        "projection_receipt_sha256": sha(receipt_path),
        "resume_checkpoint": None if not resume else str(resume),
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir = output_dir.resolve()
    result = preflight(config_path, args.profile, output_dir)
    print(json.dumps({"preflight_ok": True, **result}, sort_keys=True))
    if args.preflight_only:
        return
    import os

    os.environ["GEMMA_TUNER_CONFIG"] = str(config_path)
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.models.gemma.finetune import main as train

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(config_path)
    train(load_profile_config(config, args.profile), str(output_dir))


if __name__ == "__main__":
    run()
