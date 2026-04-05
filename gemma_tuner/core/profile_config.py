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
- Dict-like methods delegate to dataclasses.fields() for known keys and _extras
  for unknown keys. This means profile_config["model"] and profile_config.model
  both work.
- Mutation is allowed (not frozen) because the CLI layer, device layer, and
  training modules all mutate the config after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional, Union


# Sentinel for "field not set" — distinct from None (which is a valid value
# for some optional fields like max_samples).
_UNSET = object()


@dataclass
class ProfileConfig:
    """Typed configuration container for the Gemma fine-tuning pipeline.

    All known configuration keys are declared as typed fields. Unknown keys
    (from INI passthrough, wizard additions, etc.) are stored in _extras and
    accessible via the dict-style interface.

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

    model: str = ""
    dataset: str = ""
    base_model: str = ""

    # ── Required training keys (enforced by REQUIRED_PROFILE_KEYS) ──

    train_split: str = ""
    validation_split: str = ""
    text_column: str = ""
    max_label_length: int = 0
    max_duration: float = 0.0
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 1
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 2
    gradient_accumulation_steps: int = 1

    # ── Optional training hyperparameters ──

    per_device_eval_batch_size: Optional[int] = None
    warmup_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None
    temperature: Optional[float] = None
    distillation_temperature: Optional[float] = None
    distillation_alpha: Optional[float] = None
    kl_weight: Optional[float] = None
    num_beams: Optional[int] = None

    # ── Optional dataset/evaluation keys ──

    max_samples: Optional[int] = None
    id_column: Optional[str] = None
    validation_wer_threshold: Optional[float] = None
    wer_threshold: Optional[float] = None
    sample_validation_rate: float = 1.0

    # ── Device and precision ──

    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    fp16: bool = False
    bf16: bool = False

    # ── Feature flags with defaults (from FALLBACK_DEFAULTS) ──

    language_mode: str = "strict"
    languages: Union[str, List[str]] = "all"
    force_languages: bool = False
    streaming_enabled: bool = False
    gradient_checkpointing: bool = False
    skip_audio_validation: bool = False
    preprocessing_num_workers: int = 0
    dataloader_num_workers: int = 4

    # ── Evaluation / save strategy ──

    eval_strategy: Optional[str] = None
    save_strategy: Optional[str] = None
    load_validation: Optional[bool] = None
    visualize: Optional[bool] = None
    use_peft: Optional[bool] = None

    # ── CLI-injected keys (set after config load) ──

    model_name_or_path: Optional[str] = None
    split: Optional[str] = None

    # ── Wizard-only keys ──

    peft_method: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    eval_split: Optional[str] = None
    dataset_source_type: Optional[str] = None
    train_dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None

    # ── Granary-only keys ──

    hf_subset: Optional[str] = None
    hf_name: Optional[str] = None
    local_path: Optional[str] = None
    audio_sources: Optional[Dict[str, str]] = None

    # ── Misc keys that flow through from INI ──

    profile: Optional[str] = None
    group: Optional[str] = None
    source: Optional[str] = None
    concatenate_audio: Optional[bool] = None
    enable_8bit: Optional[bool] = None
    streaming: Optional[bool] = None

    # ── Overflow for unknown keys ──

    _extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Tracks which known fields were explicitly set via from_dict().
    # Used by __contains__ to match dict-like "key in config" semantics:
    # a known field that was never in the source dict reports as "not in".
    _set_fields: set = field(default_factory=set, repr=False)

    # ─────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProfileConfig:
        """Construct a ProfileConfig from a validated configuration dict.

        Known keys are assigned to their typed fields; unknown keys go to _extras.
        Tracks which known fields were explicitly present in d so that __contains__
        matches dict-like "key in config" semantics.

        This is the canonical construction path — called by load_profile_config()
        and load_model_dataset_config() after _validate_profile_config() has
        already coerced types and applied defaults.

        Args:
            d: Validated configuration dict with coerced types.

        Returns:
            ProfileConfig instance with all values assigned.
        """
        internal = {"_extras", "_set_fields"}
        known_names = {f.name for f in fields(cls) if f.name not in internal}
        known = {}
        extras = {}
        set_fields: set = set()
        for k, v in d.items():
            if k in known_names:
                known[k] = v
                set_fields.add(k)
            else:
                extras[k] = v
        instance = cls(**known)
        instance._extras = extras
        instance._set_fields = set_fields
        return instance

    # ─────────────────────────────────────────────────────────────────────
    # Dict-compatible interface (backwards compatibility)
    # ─────────────────────────────────────────────────────────────────────

    # Internal fields that should never appear in the dict-like interface.
    _INTERNAL_FIELDS = frozenset({"_extras", "_set_fields"})

    def __setattr__(self, key: str, value: Any) -> None:
        """Override attribute assignment to keep _set_fields in sync.

        When caller writes `config.model = "x"` directly (attribute style),
        this ensures "model" is added to _set_fields so that `"model" in config`
        and `config.get("model")` return the expected values — matching the
        behavior of `config["model"] = "x"` (which goes through __setitem__).

        During dataclass __init__, _set_fields is not yet initialized (it is
        declared last in the field list), so we guard with a hasattr-style check
        via __dict__ lookup to avoid infinite recursion.
        """
        object.__setattr__(self, key, value)
        # Only track known domain fields — not internals, not extras.
        if key not in self._INTERNAL_FIELDS:
            # __dict__ lookup avoids calling __getattribute__ which could recurse.
            sf = self.__dict__.get("_set_fields")
            if sf is not None:
                sf.add(key)

    def _known_field_names(self) -> set[str]:
        """Return the set of known (non-internal) field names."""
        return {f.name for f in fields(self) if f.name not in self._INTERNAL_FIELDS}

    def __getitem__(self, key: str) -> Any:
        known = self._known_field_names()
        if key in known:
            return getattr(self, key)
        if key in self._extras:
            return self._extras[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        known = self._known_field_names()
        if key in known:
            object.__setattr__(self, key, value)
            self._set_fields.add(key)
        else:
            self._extras[key] = value

    def __contains__(self, key: object) -> bool:
        """Dict-like membership test.

        For known fields, only returns True if the field was explicitly set
        (via from_dict() or __setitem__). This matches plain-dict semantics
        where a key is "in" the dict only if it was actually inserted.
        Unknown keys check _extras as usual.
        """
        if not isinstance(key, str):
            return False
        if key in self._set_fields:
            return True
        return key in self._extras

    def __delitem__(self, key: str) -> None:
        if key in self._extras:
            del self._extras[key]
        elif key in self._known_field_names():
            raise KeyError(f"Cannot delete known field '{key}' — set it to None instead")
        else:
            raise KeyError(key)

    def _is_set(self, key: str) -> bool:
        """Return True if key was explicitly set (known field in _set_fields or in _extras)."""
        return key in self._set_fields or key in self._extras

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible .get() with default value support.

        For known fields that were never explicitly set, returns the caller's
        default (matching plain-dict semantics). For set known fields and
        extras keys, returns the stored value.
        """
        if key in self:
            return self[key]
        return default

    def pop(self, key: str, *args: Any) -> Any:
        """Dict-compatible .pop() — removes key from dict view and returns value.

        For known fields: removes from _set_fields so the field no longer
        appears in __contains__, get(), keys(), or items(). The underlying
        attribute value is preserved (dataclass fields can't be deleted) but
        is no longer reachable via the dict-style interface.

        For extras keys: removes from _extras entirely.
        """
        if key in self._set_fields:
            value = getattr(self, key)
            self._set_fields.discard(key)
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
        """Return all explicitly-set keys (known fields that were set + extras).

        Only includes known fields that were explicitly provided via from_dict()
        or set via __setitem__. This matches plain-dict iteration semantics.
        """
        known = [f.name for f in fields(self)
                 if f.name not in self._INTERNAL_FIELDS and f.name in self._set_fields]
        return known + list(self._extras.keys())

    def values(self) -> List[Any]:
        """Return all values in key order."""
        return [self[k] for k in self.keys()]

    def items(self) -> List[tuple[str, Any]]:
        """Return all (key, value) pairs."""
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self) -> Iterator[str]:
        """Iterate over all keys (enables dict(config))."""
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._set_fields) + len(self._extras)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (for JSON serialization, metadata, etc.)."""
        return dict(self.items())
