"""Run and verify one non-lineage warm-start optimizer step for Plan 32."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate_smoke_profile(profile: dict) -> None:
    from gemma_tuner.utils.device import to_bool

    if (
        to_bool(profile.get("require_validation_telemetry", False)) is not True
        or to_bool(profile.get("record_exposures", False)) is not True
        or to_bool(profile.get("load_validation", False)) is not True
        or profile.get("input_mode_column")
        or int(profile.get("max_samples", 0)) != 1
        or int(profile.get("max_steps", 0)) != 1
        or int(profile.get("stop_after_step", 0)) != 1
        or int(profile.get("telemetry_train_rows", 0)) != 1
        or int(profile.get("telemetry_validation_rows", 0)) != 1
    ):
        raise ValueError("Plan 32 smoke must be one row/step, strict-loss-enabled, and outside successful epoch lineage")
    if int(profile.get("plan32_epoch", 0)):
        raise ValueError("Plan 32 smoke must not claim a successful epoch")
    if profile.get("required_transformers_version") != "5.5.2":
        raise ValueError("Plan 32 smoke lost the corrected Transformers pin")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path, output_dir: Path) -> dict:
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.utils.checkpoints import verify_complete_checkpoint
    from gemma_tuner.utils.plan32_projection import verify_projection

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    profile = load_profile_config(config, "plan32-warmstart-smoke")
    validate_smoke_profile(profile)
    import transformers

    if transformers.__version__ != "5.5.2":
        raise RuntimeError(f"Plan 32 smoke requires Transformers 5.5.2, found {transformers.__version__}")
    adapter = Path(str(profile["initial_adapter_path"]))
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    adapter_receipt = verify_complete_checkpoint(adapter.resolve(strict=True))
    dataset_dir = ROOT / "data/datasets" / str(profile["source"])
    receipt_path = dataset_dir / "projection.receipt.json"
    verify_projection(dataset_dir, expected_receipt_sha256=sha(receipt_path), epoch=1)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"smoke output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GEMMA_TUNER_CONFIG"] = str(config_path)
    from gemma_tuner.models.gemma.finetune import main as train

    train(profile, str(output_dir))
    metrics = [json.loads(line) for line in (output_dir / "redacted_metrics.jsonl").read_text().splitlines()]
    if not any(int(row["step"]) == 1 and math.isfinite(float(row["loss"])) for row in metrics):
        raise RuntimeError("Plan 32 one-step smoke did not record finite training loss")
    telemetry = [json.loads(line) for line in (output_dir / "validation_telemetry.jsonl").read_text().splitlines()]
    telemetry_by_key = {(row["event"], int(row["step"])): row for row in telemetry}
    expected = {
        (event, step)
        for event in ("fixed_train_eval", "fixed_validation_eval")
        for step in (0, 1)
    }
    if not expected.issubset(telemetry_by_key):
        raise RuntimeError("Plan 32 smoke lacks step-zero and step-one train/validation loss")
    for key in expected:
        row = telemetry_by_key[key]
        if not math.isfinite(float(row["loss"])) or int(row["scored_tokens"]) <= 0:
            raise RuntimeError(f"Plan 32 smoke has invalid loss/denominator at {key}")
    exposure_rows = [json.loads(line) for line in (output_dir / "exposures.jsonl").read_text().splitlines()]
    if len(exposure_rows) != 1:
        raise RuntimeError("Plan 32 smoke must record exactly one diagnostic exposure")
    gradients = json.loads((output_dir / "gradient_subsystems.json").read_text())
    if gradients["observations"] != 1 or any(
        gradients["by_subsystem"][name]["maximum_l2_norm"] <= 0
        for name in ("vision", "projector", "decoder")
    ):
        raise RuntimeError("Plan 32 smoke did not update all required LoRA subsystems")
    adapter_files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name.startswith("adapter_"))
    if not adapter_files:
        adapter_files = sorted(path for path in output_dir.iterdir() if path.name.startswith("adapter_model."))
    if not adapter_files:
        raise RuntimeError("Plan 32 smoke did not save a warm-start adapter")
    result = {
        "schema_version": "plan32_warmstart_smoke_v1",
        "successful_epoch_exposures": 0,
        "smoke_steps": 1,
        "finite_loss": True,
        "step_zero_and_step_one_train_validation_loss": True,
        "telemetry_events": len(telemetry),
        "telemetry_scored_token_denominators": {f"{event}@{step}": telemetry_by_key[(event, step)]["scored_tokens"]
                                                  for event, step in sorted(expected)},
        "diagnostic_exposures_not_epoch_lineage": 1,
        "gradient_subsystems": gradients["required"],
        "adapter_source_tree_sha256": adapter_receipt["tree_sha256"],
        "smoke_adapter_files": {path.name: sha(path) for path in adapter_files},
        "projected_dataset_receipt_sha256": sha(receipt_path),
        "test_rows_read": 0,
    }
    receipt_output = output_dir / "smoke-verification.json"
    receipt_output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/plan32-literal-two-epochs.ini")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/plan32-warmstart-smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(strict=True), args.output_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
