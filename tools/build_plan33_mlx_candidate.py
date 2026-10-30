"""Build a Plan 33 MLX BF16 candidate from a sealed E2B adapter.

Decoder (205) and projector (1) LoRA targets carry no clamps, so they are
merged exactly (FP32 math, stored BF16) into the verified stock MLX snapshot.
The 112 clamped vision targets stay unmerged in ``plan33-vision-lora.safetensors``
and are applied at load time outside the clamps (tools/plan33_mlx_vision_lora.py),
reproducing the PEFT training form. Untouched shards are hard links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx

from tools.plan33_mlx_vision_lora import EXPECTED_VISION_TARGETS, VISION_LORA_FILE

EXPECTED = {"language": 205, "vision": EXPECTED_VISION_TARGETS, "projection": 1}
PREFIX = "base_model.model.model."


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(module: str) -> tuple[str, str]:
    """Return (kind, MLX module path) for a PEFT target module name."""
    core = module.removeprefix(PREFIX)
    if core.startswith("language_model."):
        return "language", core.replace("language_model.", "language_model.model.", 1)
    if core.startswith("vision_tower."):
        return "vision", core
    if core == "embed_vision.embedding_projection":
        return "projection", core
    raise ValueError(f"unexpected LoRA target: {module}")


def build(stock: Path, adapter: Path, output: Path, arm: str) -> dict:
    stock_receipt = json.loads((stock / "plan33-mlx-receipt.json").read_text())
    for name, expected in stock_receipt["files_sha256"].items():
        if sha_file(stock / name) != expected:
            raise ValueError(f"stock MLX file changed: {name}")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if (config.get("r"), config.get("lora_alpha"), config.get("use_dora"), config.get("use_rslora")) != (64, 128, False, False):
        raise ValueError("adapter is not the frozen rank-64 alpha-128 LoRA")
    scale = config["lora_alpha"] / config["r"]
    seal = json.loads((adapter / ".complete.json").read_text())
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if seal["files"][name]["sha256"] != sha_file(adapter / name):
            raise ValueError(f"adapter file differs from its seal: {name}")

    lora = mx.load(str(adapter / "adapter_model.safetensors"))
    modules = sorted({key[: -len(".lora_A.weight")] for key in lora if key.endswith(".lora_A.weight")})
    if len(lora) != 2 * len(modules):
        raise ValueError("adapter must hold exactly one A/B pair per target")
    kinds = {module: classify(module) for module in modules}
    counts = {kind: sum(1 for k, _ in kinds.values() if k == kind) for kind in EXPECTED}
    if counts != EXPECTED:
        raise ValueError(f"LoRA target coverage mismatch: {counts}")

    index = json.loads((stock / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    merges: dict[str, list[tuple[str, str]]] = {}
    for module, (kind, mlx_module) in kinds.items():
        if kind == "vision":
            continue
        key = f"{mlx_module}.weight"
        if key not in weight_map:
            raise ValueError(f"MLX base lacks target weight {key}")
        merges.setdefault(weight_map[key], []).append((module, key))

    output.mkdir(parents=True, exist_ok=False)
    for path in stock.iterdir():
        if path.name == "plan33-mlx-receipt.json" or path.name in merges:
            continue
        os.link(path.resolve(), output / path.name)
    max_delta = 0.0
    for shard, targets in merges.items():
        tensors = mx.load(str(stock / shard))
        for module, key in targets:
            base = tensors[key]
            delta = scale * (lora[f"{module}.lora_B.weight"].astype(mx.float32)
                             @ lora[f"{module}.lora_A.weight"].astype(mx.float32))
            if delta.shape != base.shape:
                raise ValueError(f"shape mismatch for {key}: {delta.shape} vs {base.shape}")
            tensors[key] = (base.astype(mx.float32) + delta).astype(base.dtype)
            max_delta = max(max_delta, float(mx.abs(delta).max().item()))
        mx.save_safetensors(str(output / shard), tensors, metadata={"format": "mlx"})

    vision = {"__scale__": mx.array(scale, dtype=mx.float32)}
    for module, (kind, mlx_module) in kinds.items():
        if kind == "vision":
            vision[f"{mlx_module}.lora_a"] = lora[f"{module}.lora_A.weight"]
            vision[f"{mlx_module}.lora_b"] = lora[f"{module}.lora_B.weight"]
    mx.save_safetensors(str(output / VISION_LORA_FILE), vision)

    files = {path.name: sha_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    receipt = {
        "schema_version": "plan33_mlx_model_v1",
        "arm": arm,
        "precision": "bfloat16",
        "quantization": None,
        "source_stock_receipt_sha256": sha_file(stock / "plan33-mlx-receipt.json"),
        "adapter": {"weights_sha256": sha_file(adapter / "adapter_model.safetensors"),
                    "complete_sha256": sha_file(adapter / ".complete.json"),
                    "global_step": seal["global_step"], "tree_sha256": seal["tree_sha256"], "scale": scale},
        "coverage": {"merged_language": counts["language"], "merged_projection": counts["projection"],
                     "unmerged_vision": counts["vision"], "rewritten_shards": sorted(merges),
                     "max_abs_merged_delta": max_delta},
        "vision_lora": "unmerged side path added outside the clamps at load time (PEFT training form)",
        "files_sha256": files,
    }
    (output / "plan33-mlx-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    receipt = build(args.stock.resolve(), args.adapter.resolve(), args.output.resolve(), args.arm)
    print(json.dumps({k: receipt[k] for k in ("arm", "coverage")}, sort_keys=True))


if __name__ == "__main__":
    main()
