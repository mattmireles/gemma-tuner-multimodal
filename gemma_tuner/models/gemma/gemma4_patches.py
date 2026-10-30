"""PEFT compatibility conversion for loaded Gemma 4 clippable linears."""

from __future__ import annotations

import torch
import torch.nn as nn


class PeftCompatibleGemma4ClippableLinear(nn.Linear):
    """An ``nn.Linear`` view of an already-loaded Gemma 4 wrapper.

    Conversion happens *after* ``from_pretrained`` so published
    ``*.linear.weight`` keys load through the native Transformers architecture.
    The same Parameter object is then moved onto this PEFT-compatible module;
    no pretrained value is copied, translated, or reinitialized.
    """

    def __init__(self, source: nn.Module) -> None:
        inner = getattr(source, "linear", None)
        if not isinstance(inner, nn.Linear) or inner.bias is not None:
            raise TypeError("expected a bias-free native Gemma4ClippableLinear wrapper")
        super().__init__(
            inner.in_features,
            inner.out_features,
            bias=False,
            device=inner.weight.device,
            dtype=inner.weight.dtype,
        )
        self.weight = inner.weight
        self.use_clipped_linears = bool(getattr(source, "use_clipped_linears"))
        if self.use_clipped_linears:
            for name in ("input_min", "input_max", "output_min", "output_max"):
                value = getattr(source, name, None)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"native Gemma4ClippableLinear lacks buffer {name}")
                self.register_buffer(name, value)

    @property
    def linear(self) -> "PeftCompatibleGemma4ClippableLinear":
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = super().forward(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


def convert_loaded_clippable_linears(model: nn.Module) -> int:
    """Replace native loaded wrappers in-place while preserving Parameters exactly."""
    converted = 0
    for name, child in list(model.named_children()):
        if child.__class__.__name__ == "Gemma4ClippableLinear":
            setattr(model, name, PeftCompatibleGemma4ClippableLinear(child))
            converted += 1
        else:
            converted += convert_loaded_clippable_linears(child)
    return converted


def apply_clippable_linear_patch() -> None:
    """Reject the obsolete pre-load monkey patch."""
    raise RuntimeError(
        "pre-load Gemma4ClippableLinear patching is unsafe; load native weights "
        "then call convert_loaded_clippable_linears(model)"
    )
