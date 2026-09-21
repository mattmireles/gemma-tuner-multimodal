#!/usr/bin/env python3
"""Build a private, portable training tree for bounded GCP experiments."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = ROOT / "data" / "datasets" / "tt-screenshot-telepathic-v3-sft"
DEFAULT_OUTPUT = ROOT / "data" / "bundles" / "tt-screenshot-telepathic-v3-sft-gcp"
DEFAULT_CONTRACT = ROOT / "config" / "telepathic_context_sft_phase0.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def portable_rows(
    rows: list[dict[str, str]], *, selected_ids: set[str] | None, images: Path
) -> list[dict[str, str]]:
    selected = [row.copy() for row in rows if selected_ids is None or row["id"] in selected_ids]
    if selected_ids is not None and {row["id"] for row in selected} != selected_ids:
        raise ValueError("selected training IDs are missing from canonical staging")
    for row in selected:
        source = Path(row["image_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower() or ".png"
        destination = images / f"{row['id']}{suffix}"
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ValueError(f"image identity collision for {row['id']}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        row["image_path"] = f"../images/{destination.name}"
    return selected


def profile_for_subset(
    profiles: configparser.ConfigParser,
    *,
    arm: str,
    subset: str,
    rank: int,
    learning_rate: float,
    smoke: bool,
) -> None:
    base = dict(profiles[f"profile:telepathic-{arm}"])
    name = f"telepathic-{arm}-{subset}-r{rank}-lr{learning_rate:g}"
    base.update({
        "dataset": f"tt-screenshot-telepathic-v3-sft/{arm}-{subset}",
        "lora_r": str(rank),
        "lora_alpha": str(rank * 2),
        "learning_rate": str(learning_rate),
        "weight_decay": "0.01",
        "warmup_steps": "0",
        "warmup_ratio": "0.03",
        "lr_scheduler_type": "cosine",
        "logging_steps": "1",
        "save_strategy": "no" if smoke or subset == "overfit" else "steps",
        "eval_strategy": "no",
        "load_validation": "false",
        "require_gradient_subsystems": "vision,projector,decoder",
        "completion_only_logits": "true",
    })
    if smoke:
        base.update({"max_steps": "1", "num_train_epochs": "1", "gradient_accumulation_steps": "1"})
    elif subset == "overfit":
        # Overfit proof only: update after every one of the 32 frozen examples.
        # Eight-example accumulation made one observable step take >14 minutes
        # while hiding whether learning was occurring. The one-epoch probe took
        # 284.5 seconds and reduced loss by ~68%, so five passes are the bounded
        # horizon for the preregistered 90% reduction gate. Paired pilot/full
        # profiles retain their frozen batch geometry.
        base.update({"num_train_epochs": "5", "gradient_accumulation_steps": "1"})
    profiles[f"profile:{name}"] = base


def profile_for_rank64_followup(profiles: configparser.ConfigParser) -> None:
    profile_for_subset(
        profiles, arm="conditioned", subset="overfit", rank=64,
        learning_rate=1e-4, smoke=False,
    )
    followup = profiles["profile:telepathic-conditioned-overfit-r64-lr0.0001"]
    followup.update({
        "num_train_epochs": "20",
        "gradient_accumulation_steps": "1",
        "lr_scheduler_type": "constant",
        "warmup_steps": "0",
        "warmup_ratio": "0",
        "save_strategy": "steps",
        "save_steps": "160",
        "save_total_limit": "3",
    })


def profile_for_full_prompt_epoch(profiles: configparser.ConfigParser) -> None:
    """One production-proxy epoch with exact half/final optimizer checkpoints."""
    base = dict(profiles["profile:telepathic-full"])
    base.update({
        "dataset": "tt-screenshot-telepathic-v3-sft/full",
        "lora_r": "64",
        "lora_alpha": "128",
        "lora_dropout": "0.05",
        "learning_rate": "0.0001",
        "weight_decay": "0.01",
        "num_train_epochs": "1",
        "gradient_accumulation_steps": "8",
        "lr_scheduler_type": "constant",
        "warmup_steps": "0",
        "warmup_ratio": "0",
        "logging_steps": "1",
        "save_strategy": "steps",
        "save_steps": "78",
        "save_total_limit": "2",
        "eval_strategy": "no",
        "load_validation": "false",
        "require_gradient_subsystems": "vision,projector,decoder",
        "completion_only_logits": "true",
    })
    profiles["profile:telepathic-full-r64-one-epoch"] = base
    half = dict(base)
    half["stop_after_step"] = "78"
    profiles["profile:telepathic-full-r64-half-epoch"] = half


def prune_to_full_prompt_epoch(profiles: configparser.ConfigParser) -> None:
    keep = {
        "dataset_defaults",
        "group:gemma",
        "model:gemma-4-e4b-it-pinned",
        "dataset:tt-screenshot-telepathic-v3-sft/full",
        "profile:telepathic-full-r64-one-epoch",
        "profile:telepathic-full-r64-half-epoch",
    }
    for section in list(profiles.sections()):
        if section not in keep:
            profiles.remove_section(section)


def build(
    staging: Path,
    output: Path,
    contract_path: Path,
    *,
    smoke_id: str,
    full_prompt_only: bool = False,
) -> dict[str, Any]:
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    overfit_ids = set(str(value) for value in manifest["pilot_256_ids"][:32])
    pilot_ids = set(str(value) for value in manifest["pilot_256_ids"])
    smoke_ids = {str(smoke_id)}
    written: dict[str, dict[str, int]] = {}
    file_hashes: dict[str, str] = {}

    arms = ("full",) if full_prompt_only else ("compact", "conditioned", "full")
    for arm in arms:
        train_path = staging / arm / "train.csv"
        validation_path = staging / arm / "validation.csv"
        if sha256_file(train_path) != manifest["file_sha256"][f"{arm}/train.csv"]:
            raise ValueError(f"canonical {arm} train CSV hash mismatch")
        if sha256_file(validation_path) != manifest["file_sha256"][f"{arm}/validation.csv"]:
            raise ValueError(f"canonical {arm} validation CSV hash mismatch")
        fields, train_rows = read_csv(train_path)
        validation_fields, validation_rows = read_csv(validation_path)
        written[arm] = {}
        subsets = {arm: (train_rows, None)}
        if not full_prompt_only:
            subsets.update({
                f"{arm}-pilot": (train_rows, pilot_ids),
                f"{arm}-overfit": (train_rows, overfit_ids),
                f"{arm}-smoke": (train_rows, smoke_ids),
            })
        for name, (rows, selected_ids) in subsets.items():
            destination = output / name / "train.csv"
            projected = portable_rows(rows, selected_ids=selected_ids, images=images)
            write_csv(destination, fields, projected)
            written[arm][name] = len(projected)
            file_hashes[destination.relative_to(output).as_posix()] = sha256_file(destination)
        validation_destination = output / arm / "validation.csv"
        projected_validation = portable_rows(validation_rows, selected_ids=None, images=images)
        write_csv(validation_destination, validation_fields, projected_validation)
        file_hashes[validation_destination.relative_to(output).as_posix()] = sha256_file(validation_destination)

    profiles = configparser.ConfigParser(interpolation=None)
    profiles.read(staging / "profiles.ini")
    if not full_prompt_only:
        prompt_source = (ROOT / contract["messages"]["conditioned_template"]).resolve()
        if sha256_file(prompt_source) != contract["messages"]["conditioned_template_sha256"]:
            raise ValueError("conditioned prompt hash mismatch")
        prompt_destination = output / "intent_system_prompt.txt"
        shutil.copy2(prompt_source, prompt_destination)
        file_hashes[prompt_destination.relative_to(output).as_posix()] = sha256_file(prompt_destination)
        remote_root = "data/datasets/tt-screenshot-telepathic-v3-sft"
        profiles["profile:telepathic-conditioned"]["conditioned_system_prompt_template"] = (
            f"{remote_root}/intent_system_prompt.txt"
        )
    for arm in arms:
        if full_prompt_only:
            continue
        for subset in ("pilot", "overfit", "smoke"):
            dataset = f"tt-screenshot-telepathic-v3-sft/{arm}-{subset}"
            profiles[f"dataset:{dataset}"] = {
                "source": dataset,
                "train_split": "train",
                "validation_split": "validation",
            }
        profile_for_subset(
            profiles, arm=arm, subset="smoke", rank=8, learning_rate=5e-5, smoke=True
        )
        for rank in (8, 16):
            for learning_rate in (5e-5, 1e-4):
                profile_for_subset(
                    profiles, arm=arm, subset="overfit", rank=rank,
                    learning_rate=learning_rate, smoke=False,
                )
        # Follow-up capacity falsification after the matched rank-8 KILL.
        # This is intentionally conditioned-only and changes one scientific
        # axis: adapter capacity. A constant learning rate and longer horizon
        # test whether the first run was capacity/step limited before full SFT.
        if arm == "conditioned":
            profile_for_rank64_followup(profiles)
    profile_for_full_prompt_epoch(profiles)
    if full_prompt_only:
        prune_to_full_prompt_epoch(profiles)
    profiles_path = output / "profiles.ini"
    with profiles_path.open("w", encoding="utf-8") as handle:
        profiles.write(handle)
    file_hashes["profiles.ini"] = sha256_file(profiles_path)

    receipt = {
        "schema_version": "gemma4_e4b_telepathic_gcp_bundle_v1",
        "source_manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": sha256_file(contract_path),
        "overfit_32_sha256": hashlib.sha256(
            ("\n".join(str(value) for value in manifest["pilot_256_ids"][:32]) + "\n").encode()
        ).hexdigest(),
        "pilot_256_sha256": hashlib.sha256(
            ("\n".join(str(value) for value in manifest["pilot_256_ids"]) + "\n").encode()
        ).hexdigest(),
        "smoke_id_sha256": hashlib.sha256((str(smoke_id) + "\n").encode()).hexdigest(),
        "images": len(list(images.iterdir())),
        "image_bytes": sum(path.stat().st_size for path in images.iterdir()),
        "written": written,
        "full_prompt_only": full_prompt_only,
        "file_sha256": file_hashes,
    }
    receipt_path = output / "bundle-manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--smoke-id", required=True)
    parser.add_argument("--full-prompt-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(
        args.staging.resolve(), args.output.resolve(), args.contract.resolve(),
        smoke_id=args.smoke_id, full_prompt_only=args.full_prompt_only,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
