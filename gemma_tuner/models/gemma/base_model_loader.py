"""Shared Gemma base model loading for training, export, and other entrypoints.

Uses ``AutoConfig`` + architecture hints to load multimodal checkpoints with
``AutoModelForMultimodalLM`` / ``AutoModelForImageTextToText`` when appropriate,
so vision/audio towers stay attached. If those loaders fail, we **raise** rather
than fall back to ``AutoModelForCausalLM`` (which would drop towers; see gemma4-guide.md).

Called by:
- ``gemma_tuner.models.gemma.finetune`` — training
- ``gemma_tuner.scripts.export`` — LoRA merge / full-model export
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from gemma_tuner.models.gemma.family import GemmaFamily

logger = logging.getLogger(__name__)


def config_is_multimodal_gemma_like(config: Any) -> bool:
    """True when the checkpoint is multimodal Gemma (*ForConditionalGeneration*), not text-only CausalLM."""
    arch = getattr(config, "architectures", None) or []
    if isinstance(arch, str):
        arch = [arch]
    for name in arch:
        if isinstance(name, str) and "ForConditionalGeneration" in name:
            return True
    mt = getattr(config, "model_type", None)
    if mt in ("gemma3n", "gemma4"):
        return True
    return False


def load_base_model_for_gemma(
    model_id: str,
    *,
    family: GemmaFamily,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    revision: str | None = None,
) -> Any:
    """Load base weights using the Auto class that matches ``config.architectures``."""
    if family == GemmaFamily.GEMMA_4:
        from gemma_tuner.models.gemma.gemma4_patches import apply_clippable_linear_patch

        apply_clippable_linear_patch()

    try:
        config_kwargs = {"trust_remote_code": True}
        if revision:
            config_kwargs["revision"] = revision
        config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    except Exception as e:
        logger.warning(
            "Could not load AutoConfig for %s (%s); using AutoModelForCausalLM.",
            model_id,
            e,
        )
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if revision:
            load_kwargs["revision"] = revision
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if not config_is_multimodal_gemma_like(config):
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if revision:
            load_kwargs["revision"] = revision
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    multimodal_lm: Any = None
    try:
        from transformers import AutoModelForMultimodalLM

        multimodal_lm = AutoModelForMultimodalLM
    except ImportError:
        pass

    loaders: List[Any] = []
    if multimodal_lm is not None:
        loaders.append(multimodal_lm)
    loaders.append(AutoModelForImageTextToText)

    last_err: Optional[Exception] = None
    for loader_cls in loaders:
        try:
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "attn_implementation": attn_implementation,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            if revision:
                load_kwargs["revision"] = revision
            model = loader_cls.from_pretrained(model_id, **load_kwargs)
            logger.info("Loaded multimodal base model %s via %s", model_id, loader_cls.__name__)
            return model
        except Exception as e:
            last_err = e
            logger.debug("Loader %s failed for %s: %s", loader_cls.__name__, model_id, e)

    logger.error(
        "Multimodal AutoModel loaders failed for %s (%s); refusing AutoModelForCausalLM fallback — "
        "vision/audio towers would be dropped and training would not match a real multimodal checkpoint. "
        "Upgrade transformers, fix the checkpoint path, or use a compatible model revision.",
        model_id,
        last_err,
    )
    raise RuntimeError(
        f"Failed to load multimodal base model {model_id!r} with AutoModelForMultimodalLM / "
        f"AutoModelForImageTextToText (last error: {last_err!r}). "
        "Loading as CausalLM would omit encoder towers; aborting."
    ) from last_err
