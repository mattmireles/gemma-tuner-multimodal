"""Preflight and run exactly one Plan 33 stock-E2B literal v3 segment.

A segment is either one sealed 88-step quarter of the three-epoch lineage or
one step of the disposable epoch-one smoke. Preflight fails closed on any
schedule, projection, resume, runtime, allocator, GPU-exclusivity, or disk
contract breach before model weights load.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
PROJECTION_SCHEMA = "plan33_literal_projection_v1"
DEFAULT_MIN_FREE_GIB = 40


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def accelerator_gates(min_free_gib: float = DEFAULT_MIN_FREE_GIB) -> dict:
    """Require the expandable CUDA allocator and an otherwise idle GPU."""
    import torch

    facts: dict = {"cuda": torch.cuda.is_available()}
    usage = shutil.disk_usage(ROOT)
    facts["disk_free_bytes"] = usage.free
    if usage.free < min_free_gib * 1024**3:
        raise RuntimeError(f"Plan 33 requires at least {min_free_gib} GiB free disk before a segment")
    if not facts["cuda"]:
        return facts
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError("Plan 33 requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    if processes:
        raise RuntimeError(f"another GPU compute process is running: {processes}")
    free, total = torch.cuda.mem_get_info()
    facts.update({
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_bytes": int(free),
        "gpu_total_bytes": int(total),
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
        "allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
    })
    return facts


def preflight(
    config_path: Path, profile_name: str, output_dir: Path, *, min_free_gib: float = DEFAULT_MIN_FREE_GIB
) -> dict:
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.utils.checkpoints import verify_complete_checkpoint
    from gemma_tuner.utils.plan32_projection import verify_projection

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    profile = load_profile_config(config, profile_name)
    if int(profile.get("literal_epochs", 0)) != 3 or int(profile.get("plan32_epoch", 0)) not in (1, 2, 3):
        raise ValueError("profile is not a Plan 33 three-epoch literal segment")
    if str(profile.get("model_revision")) != MODEL_REVISION or profile.get("initial_adapter_path"):
        raise ValueError("Plan 33 trains pinned stock E2B with no warm-start adapter")
    epoch = int(profile["plan32_epoch"])
    start, stop = int(profile["telemetry_start_step"]), int(profile["stop_after_step"])
    smoke = int(profile.get("segment_steps", 88)) in (1, 2)
    if smoke != profile_name.startswith("plan33-smoke-"):
        raise ValueError("only plan33-smoke-* profiles may run one-step segments")
    if smoke and "smoke" not in output_dir.name:
        raise ValueError("smoke output must be a disposable smoke directory")
    expected_dataset = f"tt-screenshot-plan33-literal-v3-deploy/epoch-{epoch}"
    if str(profile.get("source")) != expected_dataset:
        raise ValueError("Plan 33 schedule epoch and training projection differ")
    if int(profile.get("max_steps", -1)) != stop:
        raise ValueError("max_steps must equal the sealed absolute stop step")

    schedule = _resolve(str(profile["plan32_schedule_path"]))
    if sha(schedule) != str(profile["plan32_schedule_sha256"]):
        raise ValueError("Plan 33 frozen schedule hash mismatch")
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2812 or any(int(row["epoch"]) != epoch for row in rows):
        raise ValueError("Plan 33 schedule has wrong row count or epoch")

    dataset_dir = ROOT / "data/datasets" / expected_dataset
    receipt_path = _resolve(str(profile["plan32_projection_receipt_path"]))
    if receipt_path != (dataset_dir / "projection.receipt.json").resolve(strict=True):
        raise ValueError("Plan 33 projection receipt path is outside the epoch dataset")
    if sha(receipt_path) != str(profile["plan32_projection_receipt_sha256"]):
        raise ValueError("Plan 33 projection receipt hash mismatch")
    receipt = verify_projection(
        dataset_dir,
        expected_receipt_sha256=str(profile["plan32_projection_receipt_sha256"]),
        epoch=epoch,
        schema_version=PROJECTION_SCHEMA,
    )
    if receipt["schedule_sha256"] != sha(schedule):
        raise ValueError("Plan 33 projection receipt refers to another schedule")

    resume = profile.get("resume_from_checkpoint")
    if resume:
        checkpoint_path = _resolve(str(resume))
        checkpoint = verify_complete_checkpoint(checkpoint_path)
        state = json.loads((checkpoint_path / "trainer_state.json").read_text(encoding="utf-8"))
        if int(checkpoint["global_step"]) != start or int(state["global_step"]) != start:
            raise ValueError("Plan 33 resume checkpoint does not match segment start")
        if ("smoke" in checkpoint_path.parent.name) != smoke:
            raise ValueError("smoke and real lineages must never resume each other")
    elif not (epoch == 1 and start == 0):
        raise ValueError("only the first epoch-one segment may start from stock weights")

    import transformers

    if transformers.__version__ != str(profile["required_transformers_version"]):
        raise RuntimeError("Plan 33 requires the pinned Transformers version")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Plan 33 output directory is not empty: {output_dir}")
    return {
        "epoch": epoch,
        "start_step": start,
        "stop_step": stop,
        "smoke": smoke,
        "schedule_sha256": sha(schedule),
        "projection_receipt_sha256": sha(receipt_path),
        "resume_checkpoint": None if not resume else str(resume),
        "transformers": transformers.__version__,
        "accelerator": accelerator_gates(min_free_gib),
    }


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir = output_dir.resolve()
    if args.min_free_gib < DEFAULT_MIN_FREE_GIB and not args.profile.startswith("plan33-smoke-"):
        raise ValueError("only disposable smoke segments may lower the disk headroom gate")
    result = preflight(config_path, args.profile, output_dir, min_free_gib=args.min_free_gib)
    print(json.dumps({"preflight_ok": True, **result}, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    os.environ["GEMMA_TUNER_CONFIG"] = str(config_path)
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.models.gemma.finetune import main as train

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    train(load_profile_config(config, args.profile), str(output_dir))


if __name__ == "__main__":
    run()
