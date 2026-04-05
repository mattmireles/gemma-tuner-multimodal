"""Shared model utilities for all fine-tuning backends.

Called by:
- models.gemma.finetune.py: install_kw_filter for PEFT/Gemma kwarg mismatch
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Keys that Gemma LoRA models do not accept but some Trainer utilities may inject.
# Removing them at the .forward() boundary prevents TypeErrors during LoRA training.
#
# Context: The Gemma 3n multimodal forward() signature accepts audio/vision tensors
# and attention masks, but NOT the text-encoder-only kwargs that HuggingFace Trainer
# occasionally injects (e.g. num_items_in_batch for loss scaling, or legacy
# inputs_embeds variants). This constant is the single source of truth for which
# kwargs to strip — updated here, enforced by install_kw_filter below.
_GEMMA_UNEXPECTED_KWARGS = frozenset(
    {
        # "inputs_ids" — common typo variant that some PEFT layers inject
        "inputs_ids",
        # Encoder-decoder kwargs that Gemma's causal-LM forward() does not accept
        "inputs_embeds",
        "decoder_inputs_embeds",
        # HuggingFace Trainer >= 4.38 injects this for per-sample loss scaling
        "num_items_in_batch",
        # NOTE: canonical "input_ids" is intentionally NOT in this set — it is a primary
        # input to AutoModelForCausalLM.forward() and must never be stripped.
    }
)


def install_kw_filter(module) -> None:
    """Patch module.forward to drop kwargs that Gemma does not accept.

    Context: When fine-tuning Gemma with PEFT/LoRA, some HuggingFace Trainer
    utilities inject kwargs (e.g. num_items_in_batch for loss scaling,
    decoder_inputs_embeds for encoder-decoder models) that Gemma's causal-LM
    architecture does not accept. This function wraps .forward() to silently
    drop those keys before they reach the model, preventing TypeErrors.

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
        logger.debug("Cannot read .forward on %s — skipping kwarg filter", type(module).__name__)
        return

    def _filtered(*f_args, **f_kwargs):
        for k in _GEMMA_UNEXPECTED_KWARGS:
            f_kwargs.pop(k, None)
        return _orig(*f_args, **f_kwargs)

    try:
        module.forward = _filtered  # type: ignore[assignment]
    except Exception:
        logger.warning(
            "Cannot patch .forward on %s — kwarg filter not installed; "
            "unexpected kwargs may cause TypeErrors during training",
            type(module).__name__,
        )
