"""Apply Plan 33 vision LoRA adapters in MLX in the exact PEFT training form.

Gemma 4 vision linears clamp their input and output. PEFT adds the LoRA path
outside those clamps: ``y = clip_out(W @ clip_in(x)) + scale * B @ (A @ x)``.
Merging the delta into ``W`` would move it inside the clamps and change the
model (Plan 33 step-704 canary: ~5% image-embedding shift). This module keeps
the 112 vision adapters as an unmerged side path, matching training.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

VISION_LORA_FILE = "plan33-vision-lora.safetensors"
EXPECTED_VISION_TARGETS = 112


class ClippedBaseWithLoRA(nn.Module):
    """Wrap a clamped linear and add an unclamped low-rank delta, like PEFT."""

    def __init__(self, base: nn.Module, lora_a: mx.array, lora_b: mx.array, scale: float) -> None:
        super().__init__()
        self.base = base
        self.lora_a = lora_a  # (rank, in_features), PEFT lora_A.weight layout
        self.lora_b = lora_b  # (out_features, rank), PEFT lora_B.weight layout
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        delta = (x @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + (self.scale * delta).astype(x.dtype)


def apply_vision_lora(model: nn.Module, model_dir: Path) -> int:
    """Replace every adapted vision linear with its training-form wrapper."""
    weights = mx.load(str(Path(model_dir) / VISION_LORA_FILE))
    scale = float(weights.pop("__scale__").item())
    modules = sorted({key.rsplit(".", 1)[0] for key in weights})
    if len(modules) != EXPECTED_VISION_TARGETS or len(weights) != 2 * EXPECTED_VISION_TARGETS:
        raise ValueError(f"expected {EXPECTED_VISION_TARGETS} vision adapters, found {len(modules)}")
    for module in modules:
        parent_name, attribute = module.rsplit(".", 1)
        parent = model
        for part in parent_name.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        base = getattr(parent, attribute)
        if not hasattr(base, "linear") or not getattr(base, "use_clipping", False):
            raise ValueError(f"{module} is not a clamped vision linear")
        setattr(parent, attribute, ClippedBaseWithLoRA(
            base, weights[f"{module}.lora_a"], weights[f"{module}.lora_b"], scale))
    mx.eval(model.parameters())
    return len(modules)
