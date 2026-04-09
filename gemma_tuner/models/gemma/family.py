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
    """Return ``GEMMA_3N`` or ``GEMMA_4`` based on ``model_id`` (case-insensitive).

    Order matters: explicit ``gemma-3n`` / ``gemma-4`` (and class-name ``gemma4``) win before the
    ``tiny-random`` stub heuristic so e.g. ``tiny-random-Gemma4ForCausalLM`` is Gemma 4, not 3n.

    Raises:
        RuntimeError: If ``model_id`` does not match a supported Gemma 3n/4 pattern (same family
        as :func:`assert_family_supported` / :func:`gate_gemma_model` — prefer catching ``RuntimeError``
        for all family gate failures).
    """
    mid = model_id.lower()
    if "gemma-3n" in mid:
        return GemmaFamily.GEMMA_3N
    if "gemma-4" in mid:
        return GemmaFamily.GEMMA_4
    # Class-style ids (e.g. ...Gemma4ForCausalLM) use "gemma4" without a hyphen; not matched by "gemma-4".
    if "gemma4" in mid and "gemma-3n" not in mid and "gemma3n" not in mid:
        return GemmaFamily.GEMMA_4
    # Tiny HF stubs (e.g. fxmarty/tiny-random-GemmaForCausalLM) lack version tokens in the id.
    if "tiny-random" in mid and "gemma" in mid:
        return GemmaFamily.GEMMA_3N
    raise RuntimeError(
        f"Unsupported Gemma model_id for family detection: {model_id!r}. "
        "Expected a Hugging Face id or local path containing 'gemma-3n' or 'gemma-4'."
    )


def family_capabilities(family: GemmaFamily) -> Dict[str, Any]:
    """Behavior flags shared by collators, loader, and docs (single source of truth).

    Returns a fresh dict each call — **call once** in collator ``__init__`` (store on ``self._caps``),
    not per batch.

    Capability keys:

    - ``control_token`` (str): the **start-of-turn** marker used by the collator's fallback
      masking path (:func:`gemma_tuner.models.common.collators.mask_gemma_prompt_tokens`).
      Collators find the tokenized ``model`` role header as a subsequence after the *last*
      occurrence of this token in the input. For Gemma 3n the marker is ``<start_of_turn>``;
      for Gemma 4 the tokenizer renders an **asymmetric** pair — ``<|turn>`` (single token,
      id 105) opens a turn and ``<turn|>`` (id 106) closes it — so the opener is what the
      masking path keys off. **Do not** use ``<|turn|>`` (bars on both sides): that string
      does not exist in the Gemma 4 tokenizer at all and encodes to a 4-token subword
      sequence that never matches anything in the rendered chat.

    - ``supports_assistant_mask`` (bool): whether ``tokenizer.apply_chat_template(
      return_assistant_tokens_mask=True)`` produces a usable mask. Hugging Face requires
      the Jinja chat template to wrap the assistant reply in a ``{% generation %}`` block;
      otherwise it silently returns an all-zero mask *and* prints a noisy warning on every
      batch. Gemma 3n's template has the block; Gemma 4's (as of transformers 5.5.0) does
      not, so collators skip the primary path entirely for Gemma 4 and go straight to
      :func:`mask_gemma_prompt_tokens`.
    """
    if family == GemmaFamily.GEMMA_3N:
        return {
            "control_token": "<start_of_turn>",
            "supports_assistant_mask": True,
            "needs_clippable_patch": False,
            # Multimodal Gemma paths can omit these keys on some transformers versions; zeros match
            # input_ids shape (see transformers#45200). Same injection as Gemma 4 — harmless when unused.
            "needs_mm_token_type_ids_injection": True,
            "min_transformers_version": MIN_TRANSFORMERS_GEMMA3N,
        }
    return {
        # Gemma 4 start-of-turn token: single id 105 in the real tokenizer, analog of
        # 3n's <start_of_turn>. The matching close token is <turn|> (id 106) — do not
        # confuse them. See ``supports_assistant_mask`` below for why the primary HF
        # assistant-mask path is disabled for this family.
        "control_token": "<|turn>",
        "supports_assistant_mask": False,
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
            "Install the Gemma 4 stack: pip install -r requirements/requirements-gemma4.txt "
            "(see README.md), or use a Gemma 3n model id."
        )
    raise RuntimeError(f"transformers {got} is below the minimum for Gemma 3n in this repo ({need}).")


# Entrypoints that still use ``AutoModelForCausalLM``-only paths; Gemma 4 is rejected until upgraded.
# ``export`` and ``eval_gemma_asr`` now use :func:`gemma_tuner.models.gemma.base_model_loader.load_base_model_for_gemma`.
GEMMA4_UNSUPPORTED_ENTRYPOINTS: FrozenSet[str] = frozenset(
    {
        "gemma_generate",
        "gemma_profiler",
        "gemma_dataset_prep",
        "inference_common",
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
        "`pip install -r requirements/requirements-gemma4.txt`, or switch to a Gemma 3n model id."
    )


def gate_gemma_model(model_id: str, *, entrypoint: str) -> GemmaFamily:
    """Detect family, enforce transformers version, and enforce entrypoint capability."""
    family = detect_family(model_id)
    assert_family_supported(family)
    assert_entrypoint_support(entrypoint, family)
    return family
