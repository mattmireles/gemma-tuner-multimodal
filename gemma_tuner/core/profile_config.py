"""
Typed ProfileConfig Dataclass for Gemma Fine-Tuning Pipeline

This module defines the canonical typed container for all profile configuration
values. It replaces the untyped Dict[str, Any] that previously flowed through
the entire pipeline, providing:

- **Type safety**: Every known key has a declared Python type.
- **Discoverability**: IDE autocomplete shows all valid fields.
- **Backwards compatibility**: Supports dict-style access ([], .get(), .update(),
  `in`, .pop(), .setdefault(), etc.) so existing consumer code works unchanged.
- **Extensibility**: Unknown keys (from INI sections, wizard, etc.) are stored in
  an _extras dict and accessible via the same dict interface.

Called by:
- core/config.py:load_profile_config() constructs ProfileConfig after validation
- core/config.py:load_model_dataset_config() constructs ProfileConfig after validation

Consumed by:
- cli_typer.py for CLI-layer mutations (max_samples, model_name_or_path, etc.)
- utils/device.py:apply_device_defaults() for MPS/CUDA/CPU optimization
- scripts/finetune.py → models/gemma/finetune.py for training
- scripts/evaluate.py for model evaluation
- scripts/blacklist.py for quality-based filtering
- scripts/inference_common.py for shared inference setup
- models/common/args.py for worker count determination
- wizard/config.py for wizard-generated configs
- scripts/prepare_granary.py for Granary dataset preparation

Design decisions:
- Uses dataclass with field(default=...) for optional keys, NOT __init__ params,
  so construction from a validated dict is clean via from_dict().
- All known fields default to the module-level _UNSET sentinel.  from_dict()
  only sets fields that appear in the source dict; fields absent from the source
  remain _UNSET.  __contains__ (and therefore .get(), .keys(), .items()) treats
  a field with value _UNSET as "not present", matching plain-dict semantics.
- Dict-like methods delegate to dataclasses.fields() for known keys and _extras
  for unknown keys. This means profile_config["model"] and profile_config.model
  both work.
- Mutation is allowed (not frozen) because the CLI layer, device layer, and
  training modules all mutate the config after construction.
- No __setattr__ override is needed: setting any known field to a real value
  (not _UNSET) automatically makes it visible via __contains__ because that
  check is always live (getattr(...) is not _UNSET).  pop() resets a field back
  to _UNSET to hide it again.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional, Union

# Sentinel for "field not set" — distinct from None (which is a valid value
# for some optional fields like max_samples).  All known ProfileConfig fields
# default to this value; __contains__ returns False for any field still holding
# _UNSET, matching plain-dict "key not present" semantics.
_UNSET = object()


@dataclass
class ProfileConfig:
    """Typed configuration container for the Gemma fine-tuning pipeline.

    All known configuration keys are declared as typed fields with _UNSET as
    their default.  Unknown keys (from INI passthrough, wizard additions, etc.)
    are stored in _extras and accessible via the same dict-style interface.

    Construction:
        Use ProfileConfig.from_dict(validated_dict) after _validate_profile_config()
        has coerced types and applied defaults.

    Access patterns (all supported):
        config.model                        # attribute access (preferred for new code)
        config["model"]                     # dict-style read
        config["model"] = "new-model"       # dict-style write
        config.get("model", "default")      # dict-style get with default
        "model" in config                   # membership test
        config.update({"key": "value"})     # bulk update
        config.pop("key", None)             # remove and return
        config.setdefault("key", "value")   # set if missing

    Iteration / serialization:
        dict(config)                        # convert to plain dict
        for k, v in config.items(): ...     # iterate key-value pairs
        json.dumps(dict(config))            # JSON serialization
    """

    # ── Required identity keys (always present after profile/model+dataset load) ──

    model: str = _UNSET  # type: ignore[assignment]
    dataset: str = _UNSET  # type: ignore[assignment]
    base_model: str = _UNSET  # type: ignore[assignment]

    # ── Required training keys (enforced by REQUIRED_PROFILE_KEYS) ──

    train_split: str = _UNSET  # type: ignore[assignment]
    validation_split: str = _UNSET  # type: ignore[assignment]
    text_column: str = _UNSET  # type: ignore[assignment]
    max_label_length: int = _UNSET  # type: ignore[assignment]
    max_duration: float = _UNSET  # type: ignore[assignment]
    per_device_train_batch_size: int = _UNSET  # type: ignore[assignment]
    num_train_epochs: int = _UNSET  # type: ignore[assignment]
    logging_steps: int = _UNSET  # type: ignore[assignment]
    save_steps: int = _UNSET  # type: ignore[assignment]
    save_total_limit: int = _UNSET  # type: ignore[assignment]
    gradient_accumulation_steps: int = _UNSET  # type: ignore[assignment]

    # ── Optional training hyperparameters ──

    per_device_eval_batch_size: Optional[int] = _UNSET  # type: ignore[assignment]
    warmup_steps: Optional[int] = _UNSET  # type: ignore[assignment]
    learning_rate: Optional[float] = _UNSET  # type: ignore[assignment]
    weight_decay: Optional[float] = _UNSET  # type: ignore[assignment]
    lora_r: Optional[int] = _UNSET  # type: ignore[assignment]
    lora_alpha: Optional[int] = _UNSET  # type: ignore[assignment]
    lora_dropout: Optional[float] = _UNSET  # type: ignore[assignment]
    lora_target_modules: Optional[List[str]] = _UNSET  # type: ignore[assignment]
    temperature: Optional[float] = _UNSET  # type: ignore[assignment]
    distillation_temperature: Optional[float] = _UNSET  # type: ignore[assignment]
    distillation_alpha: Optional[float] = _UNSET  # type: ignore[assignment]
    kl_weight: Optional[float] = _UNSET  # type: ignore[assignment]
    num_beams: Optional[int] = _UNSET  # type: ignore[assignment]

    # ── Optional dataset/evaluation keys ──

    max_samples: Optional[int] = _UNSET  # type: ignore[assignment]
    id_column: Optional[str] = _UNSET  # type: ignore[assignment]

    # ── Modality: audio (speech) vs text-only vs image fine-tuning ──

    modality: str = _UNSET  # type: ignore[assignment]
    text_sub_mode: str = _UNSET  # type: ignore[assignment]
    prompt_column: Optional[str] = _UNSET  # type: ignore[assignment]
    max_seq_length: int = _UNSET  # type: ignore[assignment]
    image_sub_mode: str = _UNSET  # type: ignore[assignment]
    image_path_column: str = _UNSET  # type: ignore[assignment]
    image_token_budget: int = _UNSET  # type: ignore[assignment]
    validation_wer_threshold: Optional[float] = _UNSET  # type: ignore[assignment]
    wer_threshold: Optional[float] = _UNSET  # type: ignore[assignment]
    sample_validation_rate: float = _UNSET  # type: ignore[assignment]

    # ── Device and precision ──

    dtype: Optional[str] = _UNSET  # type: ignore[assignment]
    attn_implementation: Optional[str] = _UNSET  # type: ignore[assignment]
    fp16: bool = _UNSET  # type: ignore[assignment]
    bf16: bool = _UNSET  # type: ignore[assignment]

    # ── Feature flags with defaults (from FALLBACK_DEFAULTS) ──

    language_mode: str = _UNSET  # type: ignore[assignment]
    languages: Union[str, List[str]] = _UNSET  # type: ignore[assignment]
    force_languages: bool = _UNSET  # type: ignore[assignment]
    streaming_enabled: bool = _UNSET  # type: ignore[assignment]
    gradient_checkpointing: bool = _UNSET  # type: ignore[assignment]
    skip_audio_validation: bool = _UNSET  # type: ignore[assignment]
    preprocessing_num_workers: int = _UNSET  # type: ignore[assignment]
    dataloader_num_workers: int = _UNSET  # type: ignore[assignment]

    # ── Evaluation / save strategy ──

    eval_strategy: Optional[str] = _UNSET  # type: ignore[assignment]
    save_strategy: Optional[str] = _UNSET  # type: ignore[assignment]
    load_validation: Optional[bool] = _UNSET  # type: ignore[assignment]
    visualize: Optional[bool] = _UNSET  # type: ignore[assignment]
    use_peft: Optional[bool] = _UNSET  # type: ignore[assignment]

    # ── CLI-injected keys (set after config load) ──

    model_name_or_path: Optional[str] = _UNSET  # type: ignore[assignment]
    split: Optional[str] = _UNSET  # type: ignore[assignment]

    # ── Wizard-only keys ──

    peft_method: Optional[str] = _UNSET  # type: ignore[assignment]
    dataset_name: Optional[str] = _UNSET  # type: ignore[assignment]
    dataset_config: Optional[str] = _UNSET  # type: ignore[assignment]
    eval_split: Optional[str] = _UNSET  # type: ignore[assignment]
    dataset_source_type: Optional[str] = _UNSET  # type: ignore[assignment]
    train_dataset_path: Optional[str] = _UNSET  # type: ignore[assignment]
    eval_dataset_path: Optional[str] = _UNSET  # type: ignore[assignment]

    # ── Granary-only keys ──

    hf_subset: Optional[str] = _UNSET  # type: ignore[assignment]
    hf_name: Optional[str] = _UNSET  # type: ignore[assignment]
    local_path: Optional[str] = _UNSET  # type: ignore[assignment]
    audio_sources: Optional[Dict[str, str]] = _UNSET  # type: ignore[assignment]

    # ── Misc keys that flow through from INI ──

    profile: Optional[str] = _UNSET  # type: ignore[assignment]
    group: Optional[str] = _UNSET  # type: ignore[assignment]
    source: Optional[str] = _UNSET  # type: ignore[assignment]
    concatenate_audio: Optional[bool] = _UNSET  # type: ignore[assignment]
    enable_8bit: Optional[bool] = _UNSET  # type: ignore[assignment]
    streaming: Optional[bool] = _UNSET  # type: ignore[assignment]

    # ── Overflow for unknown keys ──

    _extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ─────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProfileConfig:
        """Construct a ProfileConfig from a validated configuration dict.

        Known keys are assigned to their typed fields (leaving all others at
        the _UNSET sentinel).  Unknown keys go to _extras.

        This is the canonical construction path — called by load_profile_config()
        and load_model_dataset_config() after _validate_profile_config() has
        already coerced types and applied defaults.

        Args:
            d: Validated configuration dict with coerced types.

        Returns:
            ProfileConfig instance with all values assigned.
        """
        known_names = {f.name for f in fields(cls) if f.name not in _INTERNAL_FIELDS}
        known = {}
        extras = {}
        for k, v in d.items():
            if k in known_names:
                known[k] = v
            else:
                extras[k] = v
        instance = cls(**known)
        instance._extras = extras
        return instance

    # ─────────────────────────────────────────────────────────────────────
    # Dict-compatible interface (backwards compatibility)
    # ─────────────────────────────────────────────────────────────────────

    # Internal fields that should never appear in the dict-like interface.
    # Note: this is a class variable, not a dataclass field.

    def _known_field_names(self) -> set:
        """Return the set of known (non-internal) field names."""
        return {f.name for f in fields(self) if f.name not in _INTERNAL_FIELDS}

    def __getitem__(self, key: str) -> Any:
        if key in self._known_field_names():
            value = getattr(self, key)
            if value is _UNSET:
                raise KeyError(key)
            return value
        if key in self._extras:
            return self._extras[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._known_field_names():
            object.__setattr__(self, key, value)
        else:
            self._extras[key] = value

    def __contains__(self, key: object) -> bool:
        """Dict-like membership test.

        Returns True only if the key was explicitly set (field value is not
        _UNSET, or key is present in _extras).  Known fields that were never
        assigned report as "not in", matching plain-dict semantics.
        """
        if not isinstance(key, str):
            return False
        if key in self._known_field_names():
            return getattr(self, key, _UNSET) is not _UNSET
        return key in self._extras

    def __delitem__(self, key: str) -> None:
        if key in self._extras:
            del self._extras[key]
        elif key in self._known_field_names():
            raise KeyError(f"Cannot delete known field '{key}' — set it to None instead")
        else:
            raise KeyError(key)

    def _is_set(self, key: str) -> bool:
        """Return True if key was explicitly set (known field not _UNSET, or in _extras)."""
        return key in self

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible .get() with default value support.

        For known fields that were never explicitly set (_UNSET), returns the
        caller's default (matching plain-dict semantics). For set known fields
        and extras keys, returns the stored value.
        """
        if key in self:
            return self[key]
        return default

    def pop(self, key: str, *args: Any) -> Any:
        """Dict-compatible .pop() — removes key from dict view and returns value.

        For known fields: resets the field back to _UNSET so it no longer
        appears in __contains__, get(), keys(), or items(). The underlying
        dataclass attribute is set to _UNSET (the field cannot be deleted).

        For extras keys: removes from _extras entirely.
        """
        if key in self._known_field_names():
            value = getattr(self, key, _UNSET)
            if value is not _UNSET:
                object.__setattr__(self, key, _UNSET)
                return value
        if key in self._extras:
            return self._extras.pop(key)
        if args:
            return args[0]
        raise KeyError(key)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Dict-compatible .setdefault() — sets value only if key not in config."""
        if key in self:
            return self[key]
        self[key] = default
        return default

    def update(self, other: Union[Dict[str, Any], "ProfileConfig", None] = None, **kwargs: Any) -> None:
        """Dict-compatible .update()."""
        if other is not None:
            items = other.items() if hasattr(other, "items") else other
            for k, v in items:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def keys(self) -> List[str]:
        """Return all explicitly-set keys (known fields not _UNSET + extras).

        Only includes known fields whose value is not _UNSET (i.e., were
        explicitly provided via from_dict() or set via __setitem__). This
        matches plain-dict iteration semantics.
        """
        known = [
            f.name
            for f in fields(self)
            if f.name not in _INTERNAL_FIELDS and getattr(self, f.name, _UNSET) is not _UNSET
        ]
        return known + list(self._extras.keys())

    def values(self) -> List[Any]:
        """Return all values in key order."""
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple]:
        """Return all (key, value) pairs."""
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self) -> Iterator[str]:
        """Iterate over all keys (enables dict(config))."""
        return iter(self.keys())

    def __len__(self) -> int:
        set_count = sum(
            1 for f in fields(self) if f.name not in _INTERNAL_FIELDS and getattr(self, f.name, _UNSET) is not _UNSET
        )
        return set_count + len(self._extras)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (for JSON serialization, metadata, etc.)."""
        return dict(self.items())


# Module-level constant (not a dataclass field) — keeps _INTERNAL_FIELDS out of
# fields(ProfileConfig) so it doesn't appear in any dict-interface operations.
_INTERNAL_FIELDS: frozenset = frozenset({"_extras"})
