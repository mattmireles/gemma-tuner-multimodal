"""Optional runtime patches for Gemma 4 + PEFT on stacks with ``transformers>=5.5``.

Apply **before** ``from_pretrained`` on Gemma 4 multimodal checkpoints. See
``README/guides/apple-silicon/gemma4-guide.md`` and PEFT `#3129`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def apply_clippable_linear_patch() -> None:
    """Replace ``Gemma4ClippableLinear`` with an ``nn.Linear`` subclass (PEFT-friendly).

    Upstream wraps an ``nn.Linear`` in ``nn.Module``; PEFT often requires
    ``isinstance(module, nn.Linear)``. This patch inlines the same forward
    (clamp → linear → clamp) while inheriting ``nn.Linear``. Idempotent.

    No-op if ``transformers`` has no Gemma 4 modeling module or patch already applied.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4 as m
    except ImportError:
        return

    if getattr(m, "_GEMMA4_CLIPPABLE_LINEAR_PATCH_APPLIED", False):
        return

    if issubclass(m.Gemma4ClippableLinear, nn.Linear):
        m._GEMMA4_CLIPPABLE_LINEAR_PATCH_APPLIED = True
        return

    class PatchedGemma4ClippableLinear(nn.Linear):
        # Signature matches transformers 5.5+ modeling_gemma4.Gemma4ClippableLinear
        # (config, in_features, out_features); inner linear is always bias=False.
        def __init__(self, config, in_features: int, out_features: int, **kwargs) -> None:
            kwargs.pop("bias", None)
            super().__init__(in_features, out_features, bias=False, **kwargs)
            self.use_clipped_linears = config.use_clipped_linears
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        @property
        def linear(self) -> "PatchedGemma4ClippableLinear":
            # Compatibility shim: upstream ``transformers.models.gemma4.modeling_gemma4``
            # reads ``self.ffw_layer_1.linear.weight.dtype`` (and similar patterns
            # for ``linear_start`` / ``post`` in the audio / vision tower blocks)
            # on the assumption that ``Gemma4ClippableLinear`` still wraps an
            # inner ``self.linear = nn.Linear(...)`` submodule. After this patch
            # the class IS an ``nn.Linear``, so that attribute access would
            # ``AttributeError`` without this shim, aborting the forward pass:
            #
            #   File "transformers/models/gemma4/modeling_gemma4.py", line 392,
            #     in forward
            #     gradient_clipping = min(
            #         self.gradient_clipping,
            #         torch.finfo(self.ffw_layer_1.linear.weight.dtype).max,
            #     )
            #   AttributeError: 'PatchedGemma4ClippableLinear' object has no
            #     attribute 'linear'
            #
            # Returning ``self`` makes ``self.linear.weight`` resolve to
            # ``self.weight`` — same tensor, same dtype. Defined as a
            # ``@property`` (not a submodule) so it is not registered in
            # ``self._modules`` and does not create a self-referential entry in
            # ``named_modules`` / state-dict iteration.
            return self

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if self.use_clipped_linears:
                hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
            hidden_states = super().forward(hidden_states)
            if self.use_clipped_linears:
                hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
            return hidden_states

    m.Gemma4ClippableLinear = PatchedGemma4ClippableLinear
    m._GEMMA4_CLIPPABLE_LINEAR_PATCH_APPLIED = True
