"""Shared model utilities for all fine-tuning backends.

Called by:
- models.gemma.finetune.py: install_kw_filter for PEFT/Gemma kwarg mismatch
"""

from __future__ import annotations

# Keys that Gemma models do not accept but some Trainer utilities may inject.
# Removing them at the .forward() boundary prevents TypeErrors during LoRA training.
_WHISPER_UNEXPECTED_KWARGS = frozenset(
    {"input_ids", "inputs_ids", "inputs_embeds", "decoder_inputs_embeds", "num_items_in_batch"}
)


def install_kw_filter(module) -> None:
    """Patch module.forward to drop kwargs that Gemma does not accept.

    Context: When fine-tuning Gemma with PEFT/LoRA, some HuggingFace Trainer
    utilities inject text-model kwargs (e.g. input_ids, num_items_in_batch) that
    Whisper's encoder-decoder architecture does not accept. This function wraps
    .forward() to silently drop those keys.

    Apply at multiple PEFT nesting levels:
        install_kw_filter(model)
        if hasattr(model, "base_model"):
            install_kw_filter(model.base_model)
            if hasattr(model.base_model, "model"):
                install_kw_filter(model.base_model.model)

    Note: Patching .forward() directly is fragile on model re-wraps. If the model
    is moved to a new device or re-wrapped by PEFT, the closure captures a stale
    reference. Re-apply this function after any such operation.
    """
    try:
        _orig = module.forward
    except Exception:
        return

    def _filtered(*f_args, **f_kwargs):
        for k in _WHISPER_UNEXPECTED_KWARGS:
            f_kwargs.pop(k, None)
        return _orig(*f_args, **f_kwargs)

    try:
        module.forward = _filtered  # type: ignore[assignment]
    except Exception:
        pass
