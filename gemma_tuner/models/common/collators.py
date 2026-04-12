from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover — training env always has Pillow
    Image = None  # type: ignore[misc, assignment]

from gemma_tuner.models.gemma.constants import (
    GemmaTrainingConstants,
    GemmaValidationConstants,
    resolve_processor_sampling_rate,
)
from gemma_tuner.models.gemma.family import GemmaFamily, family_capabilities

logger = logging.getLogger(__name__)


def _is_null(value: Any) -> bool:
    """Return True if *value* is None or a pandas-style NaN (float('nan'))."""
    if value is None:
        return True
    try:
        return isinstance(value, float) and value != value  # NaN != NaN
    except (TypeError, ValueError):
        return False


def _find_subsequence_ids(haystack: torch.Tensor, needle: List[int]) -> int:
    """Index of first position in ``haystack`` where ``needle`` matches; ``-1`` if absent."""
    if not needle:
        return 0
    h = haystack.tolist()
    n, m = len(h), len(needle)
    if m > n:
        return -1
    for start in range(0, n - m + 1):
        if h[start : start + m] == needle:
            return start
    return -1


def _instruction_seq_len(tokenizer, user: str, assistant: str) -> int:
    """Token length for one user + assistant chat (no truncation). Used to left-trim long prompts."""
    conv = [[{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]]
    enc = tokenizer.apply_chat_template(
        conv,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        return_dict=True,
        add_generation_prompt=False,
        truncation=False,
    )
    input_ids = enc["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[1])
    if isinstance(input_ids, (list, tuple)) and input_ids:
        row = input_ids[0]
        if isinstance(row, torch.Tensor):
            return int(row.numel())
        return len(row)
    raise TypeError(f"_instruction_seq_len: unexpected input_ids type {type(input_ids)!r}")


def _left_trim_user_for_instruction_budget(tokenizer, user: str, assistant: str, max_length: int) -> str:
    """Drop characters from the **start** of ``user`` until the chat fits ``max_length`` tokens."""
    if _instruction_seq_len(tokenizer, user, assistant) <= max_length:
        return user
    lo, hi = 0, len(user)
    while lo < hi:
        mid = (lo + hi) // 2
        if _instruction_seq_len(tokenizer, user[mid:], assistant) <= max_length:
            hi = mid
        else:
            lo = mid + 1
    return user[lo:]


def _shrink_assistant_prefix_for_instruction_budget(tokenizer, user: str, assistant: str, max_length: int) -> str:
    """Keep the longest prefix of ``assistant`` that still fits with ``user`` (after user trim)."""
    if _instruction_seq_len(tokenizer, user, assistant) <= max_length:
        return assistant
    if not assistant:
        return assistant
    lo, hi = 0, len(assistant)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _instruction_seq_len(tokenizer, user, assistant[:mid]) <= max_length:
            lo = mid
        else:
            hi = mid - 1
    return assistant[:lo]


def _fit_instruction_pair_to_max_length(
    tokenizer,
    user: str,
    assistant: str,
    max_length: int,
    *,
    warned_user_trim: List[bool],
    warned_assistant_trim: List[bool],
    warned_still_overflow: List[bool],
) -> tuple[str, str]:
    """Prefer left-trimming the user prompt so the assistant turn is not dropped by end truncation."""
    orig_u, orig_a = user, assistant
    user = _left_trim_user_for_instruction_budget(tokenizer, user, assistant, max_length)
    if user != orig_u and not warned_user_trim[0]:
        logger.warning(
            "DataCollatorGemmaText: user prompt left-truncated to fit max_length=%d "
            "(tokens removed from the start of the prompt so the assistant reply stays in-window).",
            max_length,
        )
        warned_user_trim[0] = True
    if _instruction_seq_len(tokenizer, user, assistant) <= max_length:
        return user, assistant
    assistant = _shrink_assistant_prefix_for_instruction_budget(tokenizer, user, assistant, max_length)
    if assistant != orig_a and not warned_assistant_trim[0]:
        logger.warning(
            "DataCollatorGemmaText: assistant response truncated (prefix kept) to fit max_length=%d.",
            max_length,
        )
        warned_assistant_trim[0] = True
    if _instruction_seq_len(tokenizer, user, assistant) > max_length and not warned_still_overflow[0]:
        logger.warning(
            "DataCollatorGemmaText: sequence still exceeds max_length=%d after prompt/response fitting; "
            "HF truncation may remove the assistant tail — consider raising max_seq_length.",
            max_length,
        )
        warned_still_overflow[0] = True
    return user, assistant


def validate_bos_tokens_present(encoded: Dict[str, torch.Tensor], tokenizer) -> None:
    """Confirm each sequence contains ``bos_token_id`` somewhere (multimodal-safe).

    For multimodal inputs, tokens before the text BOS are legal; we only error if
    BOS is missing entirely. Shared by audio and text collators.
    """
    if "input_ids" not in encoded or tokenizer is None or not hasattr(tokenizer, "bos_token_id"):
        return

    if tokenizer.bos_token_id is None:
        return

    input_ids = encoded["input_ids"]
    bos_missing_samples = []

    for batch_index, sample_token_ids in enumerate(input_ids):
        bos_present = torch.any(sample_token_ids == tokenizer.bos_token_id).item()
        if not bos_present:
            bos_missing_samples.append(batch_index)

    if bos_missing_samples:
        max_display = GemmaValidationConstants.MAX_DISPLAYED_ERROR_SAMPLES
        displayed = bos_missing_samples[:max_display]
        ellipsis = "..." if len(bos_missing_samples) > max_display else ""
        raise RuntimeError(
            f"CRITICAL: <bos> token absent entirely from {len(bos_missing_samples)} samples "
            f"(batch indices: {displayed}{ellipsis}). "
            f"Gemma 3n requires a <bos> token somewhere in each sequence for stable training. "
            f"For multimodal inputs, audio/image tokens may precede <bos> — that is fine. "
            f"Expected token ID: {tokenizer.bos_token_id}."
        )


def _control_token_subsequence_ids(tokenizer, control_token: str) -> List[int]:
    """Token ids for the start-of-turn marker (e.g. ``<start_of_turn>`` or ``<|turn>``).

    Gemma 4 uses ``<|turn>`` (id 105 in the real tokenizer) as the start-of-turn token.
    The assistant span is then found separately by searching for the tokenized ``model``
    role header after the **last** boundary (see :func:`mask_gemma_prompt_tokens`).
    """
    if control_token == "<start_of_turn>" and getattr(tokenizer, "start_of_turn_token_id", None) is not None:
        return [int(tokenizer.start_of_turn_token_id)]
    # Optional fast path if a future tokenizer exposes a dedicated id (encode still works otherwise).
    if control_token == "<|turn>" and getattr(tokenizer, "turn_token_id", None) is not None:
        return [int(tokenizer.turn_token_id)]
    try:
        ids = tokenizer.encode(control_token, add_special_tokens=False)
        if ids:
            return ids
    except Exception:
        pass
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        tid = tokenizer.convert_tokens_to_ids(control_token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is not None and tid != unk:
            return [tid]
    return []


def mask_gemma_prompt_tokens(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    warned_prompt_masking: List[bool],
    *,
    control_token: str,
) -> None:
    """Mask prompt tokens in labels so loss is computed only on the assistant response.

    Prefer ``return_assistant_tokens_mask=True`` on ``apply_chat_template`` when the
    tokenizer supports it (see :class:`DataCollatorGemmaText`). This fallback locates the
    assistant span after the **last** control-token span (``control_token`` from
    :func:`~gemma_tuner.models.gemma.family.family_capabilities`) by searching for the
    tokenized ``model`` role header as a **subsequence** (not a fixed offset), so extra
    turn markers inside the user turn do not break masking.
    """
    ctl_ids = _control_token_subsequence_ids(tokenizer, control_token)
    if not ctl_ids:
        if not warned_prompt_masking[0]:
            logger.warning(
                "mask_gemma_prompt_tokens: could not resolve control token %r to ids. "
                "Prompt tokens will NOT be masked — this degrades fine-tuning quality.",
                control_token,
            )
            warned_prompt_masking[0] = True
        return

    ignore_id = GemmaTrainingConstants.IGNORE_TOKEN_ID
    ctl_len = len(ctl_ids)
    for i in range(labels.size(0)):
        row = input_ids[i]
        if ctl_len == 1:
            sot_positions = (row == ctl_ids[0]).nonzero(as_tuple=True)[0]
        else:
            h = row.tolist()
            n, m = len(h), ctl_len
            starts = [s for s in range(0, n - m + 1) if h[s : s + m] == ctl_ids]
            if starts:
                sot_positions = torch.tensor(starts, dtype=torch.long, device=row.device)
            else:
                sot_positions = row.new_zeros(0, dtype=torch.long)

        if len(sot_positions) == 0:
            if not warned_prompt_masking[0]:
                logger.warning(
                    "mask_gemma_prompt_tokens: at least one sample has no control-token span %r; "
                    "skipping prompt masking for those rows.",
                    control_token,
                )
                warned_prompt_masking[0] = True
            continue

        last_ctl_start = int(sot_positions[-1].item())
        after = row[last_ctl_start + ctl_len :]
        response_start: Optional[int] = None
        for header_text in ("model\n", "model"):
            try:
                needle = tokenizer.encode(header_text, add_special_tokens=False)
            except Exception:
                continue
            if not needle:
                continue
            rel = _find_subsequence_ids(after, needle)
            if rel >= 0:
                response_start = last_ctl_start + ctl_len + rel + len(needle)
                break

        if response_start is None:
            if not warned_prompt_masking[0]:
                logger.warning(
                    "mask_gemma_prompt_tokens: could not find tokenized 'model' role header "
                    "after last control span for at least one sample (e.g. index %d, last_ctl_start=%d); "
                    "skipping prompt masking for those rows.",
                    i,
                    last_ctl_start,
                )
                warned_prompt_masking[0] = True
            continue

        if response_start > labels.size(1):
            continue
        labels[i, :response_start] = ignore_id


def inject_mm_token_type_ids(encoded: Dict[str, Any]) -> None:
    """Inject ``token_type_ids`` and ``mm_token_type_ids`` when the processor omitted them.

    Applies to both Gemma 3n and Gemma 4 when :func:`~gemma_tuner.models.gemma.family.family_capabilities`
    sets ``needs_mm_token_type_ids_injection`` (multimodal forwards may omit these keys on some
    transformers versions). When missing, add tensors of zeros shaped like ``input_ids``. See
    https://github.com/huggingface/transformers/issues/45200 and
    ``README/guides/apple-silicon/gemma4-guide.md``.
    """
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        return
    ref = input_ids if isinstance(input_ids, torch.Tensor) else torch.as_tensor(input_ids)
    if "token_type_ids" not in encoded or encoded.get("token_type_ids") is None:
        encoded["token_type_ids"] = torch.zeros_like(ref)
    if "mm_token_type_ids" not in encoded or encoded.get("mm_token_type_ids") is None:
        encoded["mm_token_type_ids"] = torch.zeros_like(ref)


# Backwards compatibility for older imports.
ensure_gemma_mm_token_type_ids = inject_mm_token_type_ids


class _DedupeApplyImageBudgetWarning(logging.Filter):
    """Emit at most one missing-``image_seq_length`` warning per processor class name (no module-level set)."""

    def __init__(self) -> None:
        super().__init__()
        self._seen_types: Set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.msg, str) or not record.msg.startswith("apply_image_token_budget_to_processor:"):
            return True
        if not record.args:
            return True
        tname = record.args[0]
        if not isinstance(tname, str):
            return True
        if tname in self._seen_types:
            return False
        self._seen_types.add(tname)
        return True


_dedupe_apply_image_budget_warning = _DedupeApplyImageBudgetWarning()
logger.addFilter(_dedupe_apply_image_budget_warning)


def reset_apply_image_budget_warning_dedupe() -> None:
    """Clear per-type dedupe state (tests only — avoids order-dependent assertions)."""
    _dedupe_apply_image_budget_warning._seen_types.clear()


def apply_image_token_budget_to_processor(processor: Any, budget: int) -> None:
    """Set ``image_seq_length`` / expanded image placeholder sequence on Gemma multimodal processors.

    Hugging Face Gemma3 processors bake ``full_image_sequence`` from ``image_seq_length`` at init;
    mutating ``image_seq_length`` alone is insufficient — rebuild the expanded sequence when possible.
    """
    b = int(budget)
    if not hasattr(processor, "image_seq_length"):
        tname = type(processor).__name__
        logger.warning(
            "apply_image_token_budget_to_processor: processor %r has no image_seq_length; "
            "cannot apply image_token_budget=%s (train/serve mismatch risk). "
            "Use a Gemma multimodal processor or a compatible transformers revision.",
            tname,
            b,
        )
        return
    if int(getattr(processor, "image_seq_length", 0)) == b:
        return
    processor.image_seq_length = b
    img_tok = getattr(processor, "image_token", None)
    boi = getattr(processor, "boi_token", None)
    eoi = getattr(processor, "eoi_token", None)
    if img_tok is not None and boi is not None and eoi is not None:
        expanded = "".join([str(img_tok)] * b)
        processor.full_image_sequence = f"\n\n{boi}{expanded}{eoi}\n\n"


def _load_image_as_rgb(path: Any) -> Any:
    """Open an image from path and return a PIL Image in RGB (handles CMYK / RGBA)."""
    if Image is None:
        raise RuntimeError("PIL is required for image fine-tuning; install Pillow.")
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load image {path!r}: {e}") from e


class DataCollatorGemmaMultimodal:
    """Shared pipeline for Gemma multimodal collators that use ``processor(text=..., …)``.

    Subclasses set ``processor``, ``_caps`` (from :func:`~gemma_tuner.models.gemma.family.family_capabilities`),
    and ``_warned_prompt_masking``. Centralizes chat-template rendering, optional ``token_type_ids`` injection,
    BOS validation, and prompt masking so fixes apply to every modality that follows this pattern.
    """

    processor: Any
    _caps: Dict[str, Any]
    _warned_prompt_masking: List[bool]

    def _apply_chat_template_and_processor(
        self,
        messages_batch: List[Any],
        **processor_kwargs: Any,
    ) -> Dict[str, Any]:
        prompts = self.processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=False,
        )
        return self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            **processor_kwargs,
        )

    def _inject_mm_token_types_and_validate_bos(self, encoded: Dict[str, Any]) -> None:
        if self._caps["needs_mm_token_type_ids_injection"]:
            inject_mm_token_type_ids(encoded)
        if hasattr(self.processor, "tokenizer"):
            validate_bos_tokens_present(encoded, self.processor.tokenizer)

    def _labels_with_prompt_mask_and_attention_padding(self, encoded: Dict[str, Any]) -> None:
        """Clone ``input_ids`` → ``labels``, mask prompt tokens, then mask padding via ``attention_mask``."""
        labels = encoded["input_ids"].clone()
        mask_gemma_prompt_tokens(
            labels,
            encoded["input_ids"],
            self.processor.tokenizer,
            self._warned_prompt_masking,
            control_token=self._caps["control_token"],
        )
        am = encoded.get("attention_mask")
        if am is not None:
            labels[am == 0] = GemmaTrainingConstants.IGNORE_TOKEN_ID
        encoded["labels"] = labels

    def _labels_with_pad_and_prompt_mask_if_missing(self, encoded: Dict[str, Any]) -> None:
        """Audio path: build ``labels`` only if the processor did not set them; mask pad ids then prompt."""
        if "labels" in encoded:
            return
        labels = encoded["input_ids"].clone()
        if "attention_mask" in encoded and hasattr(self.processor, "tokenizer"):
            pad_id = self.processor.tokenizer.pad_token_id
            labels[labels == pad_id] = GemmaTrainingConstants.IGNORE_TOKEN_ID

        mask_gemma_prompt_tokens(
            labels,
            encoded["input_ids"],
            self.processor.tokenizer,
            self._warned_prompt_masking,
            control_token=self._caps["control_token"],
        )
        encoded["labels"] = labels


class DataCollatorGemmaImage(DataCollatorGemmaMultimodal):
    """Batch image+text examples for Gemma multimodal (captioning or VQA).

    Uses ``apply_chat_template(..., tokenize=False)`` then ``processor(text=..., images=...)``.
    RGB conversion is applied here so dataset loaders stay format-agnostic.
    """

    _CAPTION_INSTRUCTION = "Describe this image."

    def __init__(
        self,
        processor,
        text_column: str,
        *,
        family: GemmaFamily,
        image_path_column: str = "image_path",
        prompt_column: Optional[str] = None,
        image_token_budget: int = 280,
        sub_mode: str = "caption",
    ):
        if sub_mode not in ("caption", "vqa"):
            raise ValueError(f"DataCollatorGemmaImage: sub_mode must be 'caption' or 'vqa', got {sub_mode!r}")
        if sub_mode == "vqa" and not prompt_column:
            raise ValueError("DataCollatorGemmaImage: prompt_column is required when sub_mode is 'vqa'")
        self.processor = processor
        self.text_column = text_column
        self.image_path_column = image_path_column
        self.prompt_column = prompt_column
        self.image_token_budget = int(image_token_budget)
        self.sub_mode = sub_mode
        self._family = family
        self._caps = family_capabilities(family)
        self._warned_prompt_masking: List[bool] = [False]
        apply_image_token_budget_to_processor(self.processor, self.image_token_budget)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images: List[Any] = []
        messages_batch: List[Any] = []

        for ex in features:
            row_id = ex.get("id", ex.get("note_id", "<unknown>"))
            path = ex.get(self.image_path_column)
            if _is_null(path):
                raise KeyError(
                    f"DataCollatorGemmaImage: image path column {self.image_path_column!r} missing or null. "
                    f"Keys: {list(ex.keys())}"
                )
            try:
                img = _load_image_as_rgb(path)
            except Exception as e:
                raise RuntimeError(f"DataCollatorGemmaImage: row id={row_id!r}: {e}") from e
            images.append(img)

            text_val = ex.get(self.text_column)
            if self.text_column not in ex:
                raise KeyError(
                    f"DataCollatorGemmaImage: text column {self.text_column!r} missing. Keys: {list(ex.keys())}"
                )
            if _is_null(text_val):
                raise ValueError(
                    f"DataCollatorGemmaImage: text column {self.text_column!r} has a null value (row id={row_id!r})"
                )

            if self.sub_mode == "caption":
                user_content: List[Dict[str, Any]] = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self._CAPTION_INSTRUCTION},
                ]
            else:
                assert self.prompt_column is not None
                q = ex.get(self.prompt_column)
                if self.prompt_column not in ex:
                    raise KeyError(
                        f"DataCollatorGemmaImage: prompt column {self.prompt_column!r} missing. Keys: {list(ex.keys())}"
                    )
                if _is_null(q):
                    raise ValueError(
                        f"DataCollatorGemmaImage: prompt column {self.prompt_column!r} has a null value (row id={row_id!r})"
                    )
                user_content = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": str(q)},
                ]

            messages_batch.append(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": str(text_val)}]},
                ]
            )

        encoded = self._apply_chat_template_and_processor(messages_batch, images=images)
        self._inject_mm_token_types_and_validate_bos(encoded)
        self._labels_with_prompt_mask_and_attention_padding(encoded)
        return encoded


class DataCollatorGemmaText:
    """Batch text-only examples for Gemma CausalLM (no audio forward).

    Instruction mode: user (prompt) + assistant (response) via ``apply_chat_template``.
    When supported, uses ``return_assistant_tokens_mask=True`` for label boundaries.
    If ``assistant_masks`` is present but **all zeros** (common when the chat template
    lacks the Jinja ``generation`` keyword), the collator ignores that mask and falls
    back to :func:`mask_gemma_prompt_tokens`. Padding positions are
    masked in ``labels`` using ``attention_mask == 0`` (not ``pad_token_id``), so EOS
    is not dropped when ``pad_token == eos_token``.

    Long prompts: with ``instruction_truncation="left_user"`` (default), the collator
    measures token length and **left-truncates the user string** so the rendered chat
    fits ``max_length`` before Hugging Face end-truncation, preserving the assistant
    span. Use ``instruction_truncation="none"`` for the old behavior (truncate the
    full sequence from the right only).

    Completion mode: single text column, full sequence trained (no prompt mask).
    """

    def __init__(
        self,
        tokenizer,
        text_column: str,
        *,
        family: GemmaFamily,
        prompt_column: Optional[str] = None,
        max_length: int = 2048,
        sub_mode: str = "instruction",
        instruction_truncation: str = "left_user",
    ):
        if sub_mode not in ("instruction", "completion"):
            raise ValueError(f"DataCollatorGemmaText: sub_mode must be 'instruction' or 'completion', got {sub_mode!r}")
        if sub_mode == "instruction" and not prompt_column:
            raise ValueError("DataCollatorGemmaText: prompt_column is required when sub_mode is 'instruction'")
        if instruction_truncation not in ("left_user", "none"):
            raise ValueError(
                f"DataCollatorGemmaText: instruction_truncation must be 'left_user' or 'none', "
                f"got {instruction_truncation!r}"
            )
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.max_length = max_length
        self.sub_mode = sub_mode
        self.instruction_truncation = instruction_truncation
        self._family = family
        # Cache capability dict once; family_capabilities() returns a fresh dict each call.
        self._caps = family_capabilities(family)
        self._warned_prompt_masking: List[bool] = [False]
        self._warned_degenerate_assistant_mask: bool = False
        self._warned_instruction_user_trim: List[bool] = [False]
        self._warned_instruction_assistant_trim: List[bool] = [False]
        self._warned_instruction_still_overflow: List[bool] = [False]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.sub_mode == "instruction":
            return self._collate_instruction(features)
        return self._collate_completion(features)

    def _collate_instruction(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        assert self.prompt_column is not None
        messages_batch = []
        for ex in features:
            prompt = ex.get(self.prompt_column)
            response = ex.get(self.text_column)
            if self.prompt_column not in ex:
                raise KeyError(
                    f"DataCollatorGemmaText: prompt column {self.prompt_column!r} missing. Keys: {list(ex.keys())}"
                )
            if _is_null(prompt):
                raise ValueError(
                    f"DataCollatorGemmaText: prompt column {self.prompt_column!r} has a null value"
                )
            if self.text_column not in ex:
                raise KeyError(
                    f"DataCollatorGemmaText: text column {self.text_column!r} missing. Keys: {list(ex.keys())}"
                )
            if _is_null(response):
                raise ValueError(
                    f"DataCollatorGemmaText: text column {self.text_column!r} has a null value"
                )
            sp, sr = str(prompt), str(response)
            if self.instruction_truncation == "left_user":
                sp, sr = _fit_instruction_pair_to_max_length(
                    self.tokenizer,
                    sp,
                    sr,
                    self.max_length,
                    warned_user_trim=self._warned_instruction_user_trim,
                    warned_assistant_trim=self._warned_instruction_assistant_trim,
                    warned_still_overflow=self._warned_instruction_still_overflow,
                )
            messages_batch.append(
                [
                    {"role": "user", "content": sp},
                    {"role": "assistant", "content": sr},
                ]
            )

        tmpl_kwargs = dict(
            tokenize=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_length,
        )
        # Gemma 4's shipped chat template does not wrap assistant replies in the Jinja
        # ``{% generation %}`` block that Hugging Face needs to build assistant spans.
        # Asking for ``return_assistant_tokens_mask=True`` still "succeeds" but returns
        # an all-zero mask *and* prints a noisy per-batch HF warning. Skip the request
        # entirely for families flagged as unsupported so the collator goes straight to
        # the :func:`mask_gemma_prompt_tokens` fallback without any warning noise.
        use_primary_mask = bool(self._caps.get("supports_assistant_mask", True))
        if use_primary_mask:
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages_batch,
                    return_assistant_tokens_mask=True,
                    **tmpl_kwargs,
                )
            except ValueError as e:
                if "assistant" not in str(e).lower() and "return_assistant_tokens_mask" not in str(e):
                    raise
                encoded = self.tokenizer.apply_chat_template(messages_batch, **tmpl_kwargs)
        else:
            encoded = self.tokenizer.apply_chat_template(messages_batch, **tmpl_kwargs)

        if self._caps["needs_mm_token_type_ids_injection"]:
            inject_mm_token_type_ids(encoded)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        ignore_id = GemmaTrainingConstants.IGNORE_TOKEN_ID
        am = encoded.get("assistant_masks")
        # Hugging Face can still return assistant_masks with shape matching input_ids but
        # all zeros even when we asked for it (chat template missing ``{% generation %}``).
        # Using ``labels[am == 0]`` would then mask every position. Per-row: use
        # assistant_masks only when it marks at least one supervised token; otherwise
        # fall back to mask_gemma_prompt_tokens. (Paranoia belt-and-suspenders — when
        # ``supports_assistant_mask`` is False we never asked, so ``am`` is None here.)
        if isinstance(am, torch.Tensor) and am.shape == input_ids.shape:
            degenerate_rows: List[int] = []
            for i in range(input_ids.size(0)):
                if int(am[i].sum().item()) > 0:
                    labels[i][am[i] == 0] = ignore_id
                else:
                    degenerate_rows.append(i)
            if degenerate_rows:
                if not self._warned_degenerate_assistant_mask:
                    logger.warning(
                        "DataCollatorGemmaText: assistant_masks has no supervised tokens for %d/%d row(s); "
                        "using mask_gemma_prompt_tokens fallback (often: chat template missing the Jinja "
                        "`generation` keyword so HF cannot build assistant spans).",
                        len(degenerate_rows),
                        input_ids.size(0),
                    )
                    self._warned_degenerate_assistant_mask = True
                for i in degenerate_rows:
                    mask_gemma_prompt_tokens(
                        labels[i : i + 1],
                        input_ids[i : i + 1],
                        self.tokenizer,
                        self._warned_prompt_masking,
                        control_token=self._caps["control_token"],
                    )
        else:
            mask_gemma_prompt_tokens(
                labels,
                input_ids,
                self.tokenizer,
                self._warned_prompt_masking,
                control_token=self._caps["control_token"],
            )
        labels[attention_mask == 0] = ignore_id
        encoded["labels"] = labels
        return encoded

    def _collate_completion(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        for ex in features:
            t = ex.get(self.text_column)
            if self.text_column not in ex:
                raise KeyError(
                    f"DataCollatorGemmaText: text column {self.text_column!r} missing. Keys: {list(ex.keys())}"
                )
            if _is_null(t):
                raise ValueError(
                    f"DataCollatorGemmaText: text column {self.text_column!r} has a null value"
                )
            texts.append(str(t))

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self._caps["needs_mm_token_type_ids_injection"]:
            inject_mm_token_type_ids(encoded)
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = GemmaTrainingConstants.IGNORE_TOKEN_ID
        encoded["labels"] = labels
        return encoded


class DataCollatorGemmaAudio(DataCollatorGemmaMultimodal):
    """Data collator that packs audio+text into Gemma inputs via AutoProcessor.

    Cross-file connections:
    - Consumes rows loaded by `utils.dataset_utils.load_dataset_split()` which must
      include: `id`, `audio_path`, and a text column configured by profile.
    - Delegates audio feature extraction and text tokenization to AutoProcessor to
      ensure exact replication of Gemma 3n preprocessing (USM audio tower).

    Returns dicts compatible with Gemma 3n CausalLM forward(). Exact key names are
    determined by the model processor (e.g., `input_ids`, `attention_mask`, and one
    of `audio_values`/`input_features` plus any multimodal masks).
    """

    def __init__(
        self,
        processor,
        text_column: str,
        *,
        family: GemmaFamily,
        sampling_rate_hint: Optional[int] = None,
    ):
        self.processor = processor
        self.text_column = text_column
        self.sampling_rate_hint = sampling_rate_hint
        self._family = family
        # Cache capability dict once; family_capabilities() returns a fresh dict each call.
        self._caps = family_capabilities(family)
        self._warned_prompt_masking: List[bool] = [False]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        from gemma_tuner.utils.dataset_prep import load_audio_local_or_gcs

        audios: List[List[float]] = []
        texts: List[str] = []

        sampling_rate = resolve_processor_sampling_rate(self.processor, hint=self.sampling_rate_hint)

        for ex in features:
            audio_path = ex.get("audio_path", ex.get("audio"))
            if _is_null(audio_path):
                raise KeyError(
                    f"DataCollatorGemmaAudio: no audio path found in sample. "
                    f"Expected 'audio_path' or 'audio' key. Available keys: {list(ex.keys())}"
                )
            audio = load_audio_local_or_gcs(audio_path, sampling_rate=sampling_rate)
            text = ex.get(self.text_column)
            if self.text_column not in ex:
                raise KeyError(
                    f"DataCollatorGemmaAudio: text column '{self.text_column}' missing from sample. "
                    f"Available keys: {list(ex.keys())}"
                )
            if _is_null(text):
                raise ValueError(
                    f"DataCollatorGemmaAudio: text column '{self.text_column}' has a null value"
                )
            audios.append(audio)
            texts.append(text)

        messages_batch = []
        for t in texts:
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "<audio:attached>"},
                            {"type": "text", "text": "Please transcribe this audio."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": t}]},
                ]
            )

        # Real Gemma multimodal processors do not accept processor(messages=...).
        # apply_chat_template(tokenize=True) would try load_audio() on placeholder paths;
        # we render prompts first, then attach pre-loaded waveforms via processor(text=..., audio=...).
        encoded = self._apply_chat_template_and_processor(
            messages_batch,
            audio=audios,
            sampling_rate=sampling_rate,
        )
        self._inject_mm_token_types_and_validate_bos(encoded)
        self._labels_with_pad_and_prompt_mask_if_missing(encoded)
        return encoded
