"""Gemma 3n vs Gemma 4 family detection and environment gates.

Detection is by ``model_id`` substring only. Support (installed ``transformers`` version)
is validated separately via :func:`assert_family_supported`.
"""

from __future__ import annotations

from enum import Enum
from importlib import metadata
from typing import Any, Dict, FrozenSet

from packaging.version import Version

MIN_TRANSFORMERS_GEMMA3N = "4.46.0"
MIN_TRANSFORMERS_GEMMA4 = "5.5.0"


class GemmaFamily(str, Enum):
    GEMMA_3N = "gemma3n"
    GEMMA_4 = "gemma4"


def detect_family(model_id: str) -> GemmaFamily:
    """Return ``GEMMA_3N`` or ``GEMMA_4`` based on ``model_id`` (case-insensitive)."""
    mid = model_id.lower()
    if "gemma-3n" in mid:
        return GemmaFamily.GEMMA_3N
    if "gemma-4" in mid:
        return GemmaFamily.GEMMA_4
    # Tiny HF stubs used in tests (e.g. fxmarty/tiny-random-GemmaForCausalLM) lack version tokens.
    if "tiny-random" in mid and "gemma" in mid:
        return GemmaFamily.GEMMA_3N
    raise ValueError(
        f"Unsupported Gemma model_id for family detection: {model_id!r}. "
        "Expected a Hugging Face id or local path containing 'gemma-3n' or 'gemma-4'."
    )


def family_capabilities(family: GemmaFamily) -> Dict[str, Any]:
    """Behavior flags shared by collators, loader, and docs (single source of truth)."""
    if family == GemmaFamily.GEMMA_3N:
        return {
            "control_token": "<start_of_turn>",
            "needs_clippable_patch": False,
            "needs_mm_token_type_ids_injection": False,
            "min_transformers_version": MIN_TRANSFORMERS_GEMMA3N,
        }
    return {
        # Role boundary token only. Collators find the tokenized model-role header after the *last*
        # <|turn|> (same idea as Gemma 3n: <start_of_turn> + header). Do not use "<|turn|>model" — that
        # string is not one contiguous id span in real chat tokenization.
        "control_token": "<|turn|>",
        "needs_clippable_patch": True,
        "needs_mm_token_type_ids_injection": True,
        "min_transformers_version": MIN_TRANSFORMERS_GEMMA4,
    }


def _installed_transformers_version() -> str:
    return metadata.version("transformers")


def assert_family_supported(family: GemmaFamily) -> None:
    """Raise if the installed ``transformers`` is too old for ``family``."""
    caps = family_capabilities(family)
    need = Version(str(caps["min_transformers_version"]))
    got = Version(_installed_transformers_version())
    if got >= need:
        return
    if family == GemmaFamily.GEMMA_4:
        raise RuntimeError(
            f"Gemma 4 requires transformers>={MIN_TRANSFORMERS_GEMMA4}; you have {got}. "
            "Install the Gemma 4 stack: pip install -r requirements-gemma4.txt "
            "(see README.md), or use a Gemma 3n model id."
        )
    raise RuntimeError(
        f"transformers {got} is below the minimum for Gemma 3n in this repo ({need})."
    )


# Entrypoints that still use ``AutoModelForCausalLM``-only paths; Gemma 4 is rejected until upgraded.
GEMMA4_UNSUPPORTED_ENTRYPOINTS: FrozenSet[str] = frozenset(
    {
        "gemma_generate",
        "gemma_profiler",
        "gemma_dataset_prep",
        "export",
        "inference_common",
        "eval_gemma_asr",
    }
)


def assert_entrypoint_support(entrypoint: str, family: GemmaFamily) -> None:
    """If ``family`` is Gemma 4 and ``entrypoint`` is not upgraded yet, fail fast."""
    if family != GemmaFamily.GEMMA_4:
        return
    if entrypoint not in GEMMA4_UNSUPPORTED_ENTRYPOINTS:
        return
    raise RuntimeError(
        f"Gemma 4 is not implemented in {entrypoint!r} yet. "
        "Use `gemma-macos-tuner finetune` with a Gemma 4 model after "
        "`pip install -r requirements-gemma4.txt`, or switch to a Gemma 3n model id."
    )


def gate_gemma_model(model_id: str, *, entrypoint: str) -> GemmaFamily:
    """Detect family, enforce transformers version, and enforce entrypoint capability."""
    family = detect_family(model_id)
    assert_family_supported(family)
    assert_entrypoint_support(entrypoint, family)
    return family
