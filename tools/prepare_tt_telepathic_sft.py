#!/usr/bin/env python3
"""Create private, content-bound CSV projections for the telepathic SFT experiment."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "config" / "telepathic_context_sft_phase0.json"
DEFAULT_OUTPUT = ROOT / "data" / "datasets" / "tt-screenshot-telepathic-v3-sft"
DEFAULT_PLAN29_PROJECTION = ROOT.parent / "perfect-dictator" / "data" / "datasets" / "tt_screenshot_plan28_sft_v1" / "private" / "plan29-sft-v1"
DEFAULT_PLAN29_OUTPUT = ROOT / "data" / "datasets" / "tt-screenshot-plan28-sft-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_contract(model: dict[str, Any]) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(model["repo_id"], revision=model["revision"], files_metadata=True)
    if info.sha != model["revision"]:
        raise ValueError("model revision did not resolve to the frozen commit")
    license_name = getattr(info.card_data, "license", None)
    if license_name != model["license"]:
        raise ValueError("model license changed")
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    for name, expected in model["files"].items():
        sibling = siblings.get(name)
        if sibling is None:
            raise ValueError(f"model file is missing: {name}")
        upstream_sha = sibling.lfs.sha256 if sibling.lfs else None
        if upstream_sha is not None:
            actual = upstream_sha
        else:
            actual = sha256_file(Path(hf_hub_download(model["repo_id"], name, revision=model["revision"])))
        if actual != expected:
            raise ValueError(f"model file hash changed: {name}")


def stable_key(seed: str, value: Any) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def parse_context(prompt: str) -> dict[str, Any]:
    start, end = "<context>\n", "\n</context>"
    if prompt.count(start) != 1 or prompt.count(end) != 1:
        raise ValueError("user prompt must contain exactly one canonical context block")
    payload = prompt.split(start, 1)[1].split(end, 1)[0]
    parsed = json.loads(payload)
    if list(parsed) != ["context"] or not isinstance(parsed["context"], dict):
        raise ValueError("canonical context wrapper is malformed")
    return parsed["context"]


def render_conditioned(template: str, row: dict[str, Any], *, unresolved_user_name_fallback: str = "the user") -> str:
    context = parse_context(row["user_prompt"])
    user_name = str(context["user"])
    if user_name == "{USER_FULL_NAME}":
        user_name = unresolved_user_name_fallback
    values = {
        "{USER_FULL_NAME}": user_name,
        "{APPLICATION_NAME}": str(context["application"]),
    }
    rendered = template
    for placeholder, value in values.items():
        if rendered.count(placeholder) < 1:
            raise ValueError(f"conditioned template must contain {placeholder}")
        rendered = rendered.replace(placeholder, value)
    if "{USER_FULL_NAME}" in rendered or "{APPLICATION_NAME}" in rendered:
        raise ValueError("conditioned template contains unresolved placeholders")
    return rendered


def owner_diverse_subset(rows: list[dict[str, Any]], seed: str, count: int) -> list[str]:
    by_owner: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda r: stable_key(seed, r["context_id"])):
        by_owner[str(row["owner_id"])].append(row)
    owners = sorted(by_owner, key=lambda owner: stable_key(seed, owner))
    selected: list[str] = []
    while len(selected) < count:
        made_progress = False
        for owner in owners:
            if by_owner[owner] and len(selected) < count:
                selected.append(str(by_owner[owner].popleft()["context_id"]))
                made_progress = True
        if not made_progress:
            raise ValueError("pilot subset is larger than available training rows")
    return selected


def build_plan29_projection(source: Path, output: Path) -> dict[str, Any]:
    """Project the sealed Plan 29 JSONL into the existing full-prompt trainer ABI."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tt_screenshot_plan28_sft_v1":
        raise ValueError("unexpected Plan 29 projection schema")
    output.mkdir(parents=True, exist_ok=True)
    full = output / "full"
    full.mkdir(parents=True, exist_ok=True)
    fields = ["id", "owner_id", "image_path", "prompt", "response", "image_view_policy", "system_prompt"]
    file_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, expected in (("train", 2812), ("validation", 252)):
        source_path = source / f"{split}.jsonl"
        expected_file = manifest["files"][f"{split}.jsonl"]
        if sha256_file(source_path) != expected_file["sha256"]:
            raise ValueError(f"sealed Plan 29 {split} hash mismatch")
        rows = [json.loads(line) for line in source_path.open(encoding="utf-8")]
        if len(rows) != expected or len({row["id"] for row in rows}) != expected:
            raise ValueError(f"unexpected Plan 29 {split} rows")
        if any(row["split"] != split for row in rows):
            raise ValueError("split mutation in Plan 29 projection")
        destination = full / f"{split}.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                image = Path(row["image_path"])
                if not image.is_file() or sha256_file(image) != row["image_sha256"]:
                    raise ValueError("Plan 29 image binding failed")
                if row["image_view_policy"] != "original-plus-four-quadrants":
                    raise ValueError("Plan 29 view policy changed")
                writer.writerow({
                    "id": row["id"],
                    "owner_id": row["owner_hash"],
                    "image_path": str(image.resolve()),
                    "prompt": row["user_prompt"],
                    "response": row["response"],
                    "image_view_policy": "global_plus_four_nonoverlapping_quadrants",
                    "system_prompt": row["system_prompt"],
                })
        temporary.replace(destination)
        counts[split] = len(rows)
        file_hashes[f"full/{split}.csv"] = sha256_file(destination)
    profiles = configparser.ConfigParser(interpolation=None)
    profiles["DEFAULT"] = {
        "num_train_epochs": "1", "logging_steps": "1", "save_steps": "88",
        "save_total_limit": "4", "gradient_accumulation_steps": "8",
        "learning_rate": "0.0001", "warmup_steps": "0", "output_dir": "output",
    }
    profiles["dataset_defaults"] = {
        "text_column": "response", "max_label_length": "1024", "max_duration": "1.0",
        "id_column": "id", "streaming_enabled": "false", "preprocessing_num_workers": "0",
        "dataloader_num_workers": "0",
    }
    profiles["group:gemma"] = {"dtype": "bfloat16", "attn_implementation": "sdpa", "optim": "adamw_torch"}
    profiles["model:gemma-4-e4b-it-pinned"] = {
        "base_model": "google/gemma-4-E4B-it",
        "model_revision": "fee6332c1abaafb77f6f9624236c63aa2f1d0187",
        "group": "gemma", "per_device_train_batch_size": "1", "per_device_eval_batch_size": "1",
    }
    dataset_name = "tt-screenshot-plan28-sft-v1/full"
    profiles[f"dataset:{dataset_name}"] = {"source": dataset_name, "train_split": "train", "validation_split": "validation"}
    profiles["profile:telepathic-plan29-r64-one-epoch"] = {
        "model": "gemma-4-e4b-it-pinned", "dataset": dataset_name, "modality": "image",
        "image_sub_mode": "vqa", "image_path_column": "image_path", "prompt_column": "prompt",
        "system_prompt_column": "system_prompt", "text_column": "response",
        "image_token_budget": "280", "image_view_policy": "global_plus_four_nonoverlapping_quadrants",
        "require_telepathic_contract": "false", "completion_only_logits": "true",
        "max_seq_length": "24576", "gradient_checkpointing": "true", "full_determinism": "true",
        "seed": "42", "load_validation": "false", "save_strategy": "steps", "save_steps": "88",
        "save_total_limit": "4", "eval_strategy": "no", "lora_r": "64", "lora_alpha": "128",
        "lora_dropout": "0.05", "learning_rate": "0.0001", "weight_decay": "0.01",
        "num_train_epochs": "1", "gradient_accumulation_steps": "8", "lr_scheduler_type": "constant",
        "warmup_steps": "0", "warmup_ratio": "0", "logging_steps": "1",
        "require_gradient_subsystems": "vision,projector,decoder",
        "record_exposures": "true",
        "lora_target_modules_regex": "^(?:model\\.language_model\\..*\\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|model\\.vision_tower\\.encoder\\.layers\\.\\d+\\.(?:self_attn\\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\\.(?:gate_proj|up_proj|down_proj))|model\\.embed_vision\\.embedding_projection)$",
    }
    profiles_path = output / "profiles.ini"
    with profiles_path.open("w", encoding="utf-8") as handle:
        profiles.write(handle)
    file_hashes["profiles.ini"] = sha256_file(profiles_path)
    output_manifest = {
        "schema_version": "gemma4_e4b_plan29_private_staging_v1",
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "file_sha256": file_hashes,
        "written": {"full": counts},
        "validation_order_ids": [row["id"] for row in [json.loads(line) for line in (source / "validation.jsonl").open()]],
        "sealed_test": {"count": 232, "staged": False},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**output_manifest, "manifest_sha256": sha256_file(manifest_path)}


def full_prompt_rows(dataset: dict[str, Any], authority_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load v2 prompt bytes and prove all non-prompt identity against v3."""
    source = (ROOT / dataset["full_prompt_path"]).resolve()
    if sha256_file(source) != dataset["full_prompt_sha256"]:
        raise ValueError("full-prompt dataset hash mismatch")
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    by_id = {str(row["context_id"]): row for row in rows}
    authority_by_id = {str(row["context_id"]): row for row in authority_rows}
    if len(by_id) != len(rows) or set(by_id) != set(authority_by_id):
        raise ValueError("full-prompt and v3 context IDs differ")
    for context_id, authority in authority_by_id.items():
        full = by_id[context_id]
        for field in ("owner_id", "split", "target_json", "screenshot_local_path"):
            if full.get(field) != authority.get(field):
                raise ValueError(f"full-prompt row differs from v3 authority: {field}")
        if not str(full.get("system_prompt", "")).strip():
            raise ValueError("full-prompt row has no system prompt")
        prompt = str(full.get("user_prompt", ""))
        for tag in ("<first_pass_screenshot_ocr>", "</first_pass_screenshot_ocr>"):
            if prompt.count(tag) != 1:
                raise ValueError(f"full-prompt row requires exactly one {tag}")
    return by_id


def build(contract_path: Path, output: Path, *, verify_model: bool = False) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if verify_model:
        verify_model_contract(contract["model"])
    dataset = contract["dataset"]
    source = (ROOT / dataset["path"]).resolve()
    if sha256_file(source) != dataset["sha256"]:
        raise ValueError("v3 dataset hash mismatch")
    prompt_path = (ROOT / contract["messages"]["conditioned_template"]).resolve()
    if sha256_file(prompt_path) != contract["messages"]["conditioned_template_sha256"]:
        raise ValueError("conditioned intent template hash mismatch")
    template = prompt_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    if any("system_prompt" in row for row in rows):
        raise ValueError("v3 row contains forbidden system_prompt")
    actual_counts = Counter(str(row["split"]) for row in rows)
    for split in ("train", "validation", "test"):
        if actual_counts[split] != dataset["counts"][split]:
            raise ValueError(f"unexpected {split} count")
    owners: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        owners[str(row["owner_id"])].add(str(row["split"]))
    if len(owners) != dataset["counts"]["owners"] or any(len(v) != 1 for v in owners.values()):
        raise ValueError("owner count or split isolation mismatch")

    full_by_id = full_prompt_rows(dataset, rows)

    output.mkdir(parents=True, exist_ok=True)
    common_fields = ["id", "owner_id", "image_path", "prompt", "response", "image_view_policy"]
    written: dict[str, dict[str, int]] = {}
    file_hashes: dict[str, str] = {}
    pending_files: list[tuple[Path, Path]] = []
    for arm in ("compact", "conditioned", "full"):
        written[arm] = {}
        fields = common_fields if arm == "compact" else [*common_fields, "system_prompt"]
        arm_root = output / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation"):
            destination = arm_root / f"{split}.csv"
            temporary = destination.with_suffix(".csv.tmp")
            split_rows = [row for row in rows if row["split"] == split]
            if split == "validation":
                seed = dataset["validation_order"]["seed"]
                split_rows.sort(key=lambda row: stable_key(seed, row["context_id"]))
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in split_rows:
                    image = (source.parent / row["screenshot_local_path"]).resolve()
                    if not image.is_file():
                        raise FileNotFoundError("a source screenshot is missing")
                    target = json.loads(row["target_json"])
                    if list(target) != ["context_analysis"]:
                        raise ValueError("target violates context_analysis contract")
                    source_row = full_by_id[str(row["context_id"])] if arm == "full" else row
                    projected = {
                        "id": row["context_id"],
                        "owner_id": row["owner_id"],
                        "image_path": str(image),
                        "prompt": source_row["user_prompt"],
                        "response": row["target_json"],
                        "image_view_policy": contract["visual_abi"]["policy"],
                    }
                    if arm == "conditioned":
                        projected["system_prompt"] = render_conditioned(
                            template,
                            row,
                            unresolved_user_name_fallback=contract["messages"]["unresolved_user_name_fallback"],
                        )
                    elif arm == "full":
                        projected["system_prompt"] = source_row["system_prompt"]
                    writer.writerow(projected)
            written[arm][split] = len(split_rows)
            file_hashes[destination.relative_to(output).as_posix()] = sha256_file(temporary)
            pending_files.append((temporary, destination))

    for temporary, destination in pending_files:
        temporary.replace(destination)
    for arm in ("compact", "conditioned", "full"):
        for split in ("train", "validation"):
            (output / f"{arm}-{split}.csv").unlink(missing_ok=True)

    profiles = configparser.ConfigParser(interpolation=None)
    profiles["DEFAULT"] = {
        "num_train_epochs": "2",
        "logging_steps": "1",
        "save_steps": "32",
        "save_total_limit": "8",
        "gradient_accumulation_steps": str(contract["paired_training"]["gradient_accumulation_steps"]),
        "learning_rate": "5e-5",
        "warmup_steps": "2",
        "output_dir": "output",
    }
    profiles["dataset_defaults"] = {
        "text_column": "response",
        "max_label_length": "1024",
        "max_duration": "1.0",
        "id_column": "id",
        "streaming_enabled": "false",
        "preprocessing_num_workers": "0",
        "dataloader_num_workers": "0",
    }
    profiles["group:gemma"] = {
        "dtype": "bfloat16",
        "attn_implementation": contract["paired_training"]["attn_implementation"],
        "optim": contract["paired_training"]["optimizer"],
    }
    profiles["model:gemma-4-e4b-it-pinned"] = {
        "base_model": contract["model"]["repo_id"],
        "model_revision": contract["model"]["revision"],
        "group": "gemma",
        "per_device_train_batch_size": "1",
        "per_device_eval_batch_size": "1",
    }
    for arm in ("compact", "conditioned", "full"):
        dataset_name = f"tt-screenshot-telepathic-v3-sft/{arm}"
        profiles[f"dataset:{dataset_name}"] = {
            "source": dataset_name,
            "train_split": "train",
            "validation_split": "validation",
        }
        profile = {
            "model": "gemma-4-e4b-it-pinned",
            "dataset": dataset_name,
            "modality": "image",
            "image_sub_mode": "vqa",
            "image_path_column": "image_path",
            "prompt_column": "prompt",
            "text_column": "response",
            "image_token_budget": str(contract["visual_abi"]["image_token_budget_per_view"]),
            "image_view_policy": contract["visual_abi"]["policy"],
            "require_telepathic_contract": "false" if arm == "full" else "true",
            "completion_only_logits": "true",
            "max_seq_length": str(contract["paired_training"]["max_seq_length"]),
            "gradient_checkpointing": "true",
            "full_determinism": "true",
            "seed": str(contract["paired_training"]["seed"]),
            "load_validation": "false",
            "save_strategy": "steps",
            "eval_strategy": "no",
            "lora_r": "8",
            "lora_alpha": "16",
            "lora_dropout": str(contract["paired_training"]["lora_dropout"]),
            "lora_target_modules_regex": contract["paired_training"]["lora_target_regex"],
        }
        if arm == "conditioned":
            profile.update(
                {
                    "system_prompt_column": "system_prompt",
                    "conditioned_system_prompt_template": str(prompt_path),
                    "conditioned_system_prompt_sha256": contract["messages"]["conditioned_template_sha256"],
                }
            )
        elif arm == "full":
            profile["system_prompt_column"] = "system_prompt"
        profiles[f"profile:telepathic-{arm}"] = profile
    profiles_path = output / "profiles.ini"
    temporary_profiles = output / "profiles.ini.tmp"
    with temporary_profiles.open("w", encoding="utf-8") as handle:
        profiles.write(handle)
    temporary_profiles.replace(profiles_path)
    file_hashes[profiles_path.relative_to(output).as_posix()] = sha256_file(profiles_path)

    validation = sorted(
        (row for row in rows if row["split"] == "validation"),
        key=lambda row: stable_key(dataset["validation_order"]["seed"], row["context_id"]),
    )
    train = [row for row in rows if row["split"] == "train"]
    manifest = {
        "schema_version": "gemma4_e4b_telepathic_private_staging_v1",
        "contract_sha256": sha256_file(contract_path),
        "source_sha256": dataset["sha256"],
        "written": written,
        "file_sha256": file_hashes,
        "validation_order_ids": [str(row["context_id"]) for row in validation],
        "validation_first_20_ids": [str(row["context_id"]) for row in validation[:20]],
        "pilot_256_ids": owner_diverse_subset(train, dataset["pilot_subset"]["seed"], 256),
        "sealed_test": {"count": actual_counts["test"], "staged": False},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema_version": "gemma4_e4b_telepathic_phase0_receipt_v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": manifest["contract_sha256"],
        "source_sha256": dataset["sha256"],
        "model": contract["model"],
        "model_metadata_verified": verify_model,
        "messages": contract["messages"],
        "visual_abi": contract["visual_abi"],
        "target_abi": contract["target_abi"],
        "paired_training": contract["paired_training"],
        "evaluation": contract["evaluation"],
        "gcp": contract["gcp"],
        "budget_usd": contract["budget_usd"],
        "advance_rules": contract["advance_rules"],
        "counts": written,
        "validation_order_sha256": hashlib.sha256(
            ("\n".join(manifest["validation_order_ids"]) + "\n").encode()
        ).hexdigest(),
        "validation_first_20_sha256": hashlib.sha256(
            ("\n".join(manifest["validation_first_20_ids"]) + "\n").encode()
        ).hexdigest(),
        "pilot_256_sha256": hashlib.sha256(("\n".join(manifest["pilot_256_ids"]) + "\n").encode()).hexdigest(),
        "private_manifest_sha256": sha256_file(manifest_path),
        "sealed_test_staged": False,
        "conditioned_user_name_fallback_rows": sum(
            parse_context(row["user_prompt"])["user"] == "{USER_FULL_NAME}" for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--verify-model", action="store_true")
    parser.add_argument("--plan29-projection", type=Path)
    parser.add_argument("--plan29-output", type=Path, default=DEFAULT_PLAN29_OUTPUT)
    args = parser.parse_args()
    if args.plan29_projection:
        print(json.dumps(build_plan29_projection(args.plan29_projection.resolve(), args.plan29_output.resolve()), indent=2, sort_keys=True))
        return
    receipt = build(args.contract.resolve(), args.output.resolve(), verify_model=args.verify_model)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
