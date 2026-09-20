#!/usr/bin/env python
# coding=utf-8
"""
Gemma 3n LoRA Fine-Tuning for Apple Silicon (MPS) - Enhanced with Critical Bug Fixes

This module integrates Google's Gemma 3n multimodal (audio+text) model into the
gemma-macos-tuner framework. It implements a PEFT (LoRA) training path
optimized for Apple Silicon using the PyTorch MPS backend.

CRITICAL BUG FIXES IMPLEMENTED:
This enhanced version addresses 8 critical stability, performance, and data integrity
issues identified during comprehensive Gemma 3n integration audit:

🚨 CRITICAL FIXES:
1. Processor Interface Fallback Bug - ELIMINATED dangerous text rendering fallback
2. Missing <bos> Token Validation - IMPLEMENTED comprehensive sequence validation

🔧 HIGH PRIORITY FIXES:
3. Enhanced bfloat16 Testing - Comprehensive operation testing beyond tensor creation
4. Silent Audio Loading Failures - Added explicit error logging for data integrity
5. MPS Memory Management Logic - Fixed None comparison in watermark validation

⚠️ MEDIUM PRIORITY FIXES:
6. Config/Code Mismatch for LoRA - Standardized comma-separated string handling
7. Complex Sampling Rate Logic - Simplified with centralized detection hierarchy

📊 MONITORING ENHANCEMENTS:
8. MPS Memory Pressure Monitoring - Proactive swapping detection and warnings

Called by:
- scripts/finetune.py:main() dynamic router when model name contains "gemma"
- core/ops.py:finetune() via scripts.finetune dispatch

Calls to:
- utils/dataset_utils.load_dataset_split() for dataset loading (CSV-based)
- utils.device.get_device() for enhanced MPS detection and memory monitoring
- utils.device.check_memory_pressure() for unified memory pressure tracking
- utils/dataset_prep.py:load_audio_local_or_gcs() with enhanced error logging
- transformers AutoModelForCausalLM/AutoProcessor with strict validation
- peft.get_peft_model for LoRA adapter injection with target module validation

Design notes:
- Uses eager attention and comprehensive bfloat16 testing on MPS with robust fallbacks
- Collation loads audio with error detection and delegates multimodal packing to
  validated AutoProcessor, ensuring proper chat templating and <bos> token presence
- Implements AI-first documentation with extensive cross-file connections and context
- Named constants replace all magic numbers for AI-readable code maintenance
- Comprehensive validation prevents silent failures and provides actionable errors

References:
- README/guides/Gemma3n-fine-tune-apple-silicon-condensed.md for validation rules
- Bug fix implementation follows Apple Silicon MPS optimization guidelines
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Union

if TYPE_CHECKING:
    from gemma_tuner.core.profile_config import ProfileConfig

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging

from gemma_tuner.models.common.collators import (
    DataCollatorGemmaAudio,
    DataCollatorGemmaAudioVisual,
    DataCollatorGemmaImage,
    DataCollatorGemmaText,
    apply_image_token_budget_to_processor,
)
from gemma_tuner.models.common.metrics import build_wer_metrics
from gemma_tuner.models.common.results import persist_training_results
from gemma_tuner.models.common.utils import install_kw_filter
from gemma_tuner.models.common.visualizer import VisualizerTrainerCallback
from gemma_tuner.models.common.viz_trainer import GemmaVizTrainer
from gemma_tuner.models.gemma.base_model_loader import load_base_model_for_gemma
from gemma_tuner.models.gemma.constants import (
    GemmaTrainingConstants,
    GemmaValidationConstants,
)
from gemma_tuner.models.gemma.family import (
    assert_entrypoint_support,
    assert_family_supported,
    detect_family,
)
from gemma_tuner.utils.checkpoints import (
    ImmutableCheckpointCallback,
    StopAfterStepCallback,
    verify_complete_checkpoint,
)
from gemma_tuner.utils.dataset_utils import load_dataset_split, resolve_data_datasets_dir
from gemma_tuner.utils.device import empty_cache, get_device, to_bool
from gemma_tuner.utils.gradient_receipt import (
    GradientSubsystemReceiptCallback,
    RedactedTrainingMetricsCallback,
)
from gemma_tuner.utils.integrity import create_integrity_manifest

# Re-export DataCollatorGemmaAudio so existing imports from this module still work.
# The canonical class lives in models/common/collators.py; the local duplicate was
# removed because it called load_audio_local_or_gcs without importing it (NameError).
__all__ = ["DataCollatorGemmaAudio"]

logger = logging.getLogger(__name__)


def completion_only_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute causal CE when Gemma returns only the supervised suffix logits."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("completion-only loss expects [B,K,V] logits and [B,L] labels")
    kept = int(logits.shape[1])
    if kept < 2 or kept > labels.shape[1]:
        raise ValueError("completion-only logits length is incompatible with labels")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, -(kept - 1) :]
    if attention_mask is not None:
        shift_attention = attention_mask[:, -(kept - 1) :].to(dtype=torch.bool)
        shift_logits = shift_logits[shift_attention]
        shift_labels = shift_labels[shift_attention]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1).to(shift_logits.device),
        ignore_index=GemmaTrainingConstants.IGNORE_TOKEN_ID,
    )


class CompletionOnlyTrainer(Trainer):
    """Avoid full-sequence 256k-vocabulary logits for suffix-only SFT loss."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prepared = dict(inputs)
        labels = prepared.pop("labels")
        attention_mask = prepared.get("attention_mask")
        outputs = model(**prepared)
        loss = completion_only_causal_loss(outputs.logits, labels, attention_mask)
        return (loss, outputs) if return_outputs else loss


def _test_mps_bfloat16_support(device: torch.device) -> bool:
    """
    Comprehensively tests MPS bfloat16 support including operations and gradients.

    This function performs thorough bfloat16 compatibility testing that goes beyond
    simple tensor creation. Many MPS implementations support tensor creation but
    fail during actual mathematical operations or gradient computation.

    Called by:
    - models/gemma/finetune.py:main() during device and dtype initialization
    - Invoked once per training session to determine optimal precision

    Test methodology:
    1. Create bfloat16 tensors with gradient tracking enabled
    2. Perform matrix multiplication (core training operation)
    3. Execute reduction operations (loss computation)
    4. Compute gradients (backward pass simulation)
    5. Verify dtype preservation and numerical validity

    Why comprehensive testing is needed:
    - MPS bfloat16 support varies across Apple Silicon generations
    - Some operations work while others fail silently or produce NaN
    - Training failures from dtype issues are difficult to debug
    - Early detection enables graceful fallback to float32

    Reference:
    - README/guides/Gemma3n-fine-tune-apple-silicon-condensed.md:246-247
    - "The MPS backend may default to float16, which has a more limited numerical range"

    Args:
        device (torch.device): Device to test (should be MPS device)

    Returns:
        bool: True if bfloat16 is fully supported, False if fallback to float32 needed

    Note:
        Returns False for non-MPS devices. Callers must use device-specific checks
        (e.g. torch.cuda.is_bf16_supported()) for CUDA — this function only validates
        MPS bfloat16 support and is not a general-purpose dtype probe.

        Uses GemmaValidationConstants.BFLOAT16_TEST_TENSOR_SIZE for test tensor dimensions
        to balance thorough testing with memory efficiency.
    """
    if device.type != "mps":
        # Only test MPS devices - other devices have different bfloat16 characteristics
        return False

    input_tensor = weight_tensor = output_tensor = loss_value = None
    try:
        # Create test tensors with gradient tracking (mimics training conditions)
        # Use randn (not zeros) to actually stress the dtype with non-trivial values
        test_tensor_size = GemmaValidationConstants.BFLOAT16_TEST_TENSOR_SIZE
        input_tensor = torch.randn(
            test_tensor_size, test_tensor_size, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        weight_tensor = torch.randn(test_tensor_size, test_tensor_size, device=device, dtype=torch.bfloat16)

        # Test core operations that will be used during training
        # Matrix multiplication (fundamental for transformer attention and FFN)
        output_tensor = torch.matmul(input_tensor, weight_tensor)

        # Reduction operation (loss computation)
        loss_value = output_tensor.sum()

        # Gradient computation (backward pass)
        loss_value.backward()

        # Comprehensive validation of bfloat16 operation success
        return (
            input_tensor.dtype == torch.bfloat16  # Input dtype preserved
            and output_tensor.dtype == torch.bfloat16  # Output dtype preserved
            and torch.isfinite(loss_value).item()  # No NaN/inf in results
            and input_tensor.grad is not None  # Gradients computed
            and input_tensor.grad.dtype == torch.bfloat16  # Gradient dtype preserved
        )

    except Exception as exception:
        # Any exception indicates bfloat16 is not fully supported
        logger.debug(f"MPS bfloat16 comprehensive test failed: {exception}")
        return False
    finally:
        # Clean up test tensors to avoid memory leaks on any exit path
        del input_tensor, weight_tensor, output_tensor, loss_value


def resolve_training_torch_compile(device: torch.device, profile_config: dict[str, Any]) -> bool:
    """``TrainingArguments.torch_compile``: honor profile on CUDA/CPU; force False on MPS (gemma4-guide.md)."""
    requested = bool(profile_config.get("torch_compile", False))
    if device.type == "mps":
        if requested:
            logger.warning(
                "torch_compile=True is not supported for Gemma training on MPS; forcing torch_compile=False "
                "(see README/guides/apple-silicon/gemma4-guide.md)."
            )
        return False
    return requested


def _finalize_training_artifacts(
    output_dir: str,
    *,
    trainer: Trainer,
    train_result: Any,
    modality: str,
    integrity_metadata: dict[str, Any],
) -> None:
    """Write post-training artifacts in the order integrity verification expects."""
    persist_training_results(output_dir, trainer=trainer, train_result=train_result, modality=modality)

    try:
        create_integrity_manifest(output_dir, metadata=integrity_metadata)
    except Exception as e:
        logger.warning("Failed to create integrity manifest (non-fatal): %s", e)


def _discover_candidate_target_modules(model) -> List[str]:
    """Scan model modules to find which default LoRA target heads exist in the model.

    Used when no lora_target_modules is specified in the profile config — it
    discovers which of the default target names (from GemmaTrainingConstants) are
    actually present in the model's named modules.

    Called by:
    - main() during LoRA configuration to validate discovered defaults.
    - NOT responsible for validating user-specified targets (that happens in main()
      via a ValueError if none of the requested targets exist).

    Returns:
        Sorted list of module short-names (e.g. ["k_proj", "q_proj", "v_proj"])
        that exist in the model. Empty list if none of the defaults are found.
    """
    candidates = set(GemmaTrainingConstants.LORA_TARGET_MODULES)
    present = set()
    for name, _ in model.named_modules():
        for c in list(candidates):
            if name.endswith(c):
                present.add(c)
    return sorted(present)


def _raise_if_lora_targets_use_peft_incompatible_linears(model: torch.nn.Module, target_suffixes: List[str]) -> None:
    """Fail fast before ``get_peft_model`` when LoRA names match non-``nn.Linear`` wrappers.

    Name-based discovery (see :func:`_discover_candidate_target_modules`) can list
    suffixes such as ``q_proj`` even when the implementation is
    ``Gemma4ClippableLinear`` or similar. PEFT then fails ``isinstance(..., nn.Linear)``
    with an opaque error. See ``README/guides/apple-silicon/gemma4-guide.md`` and
    https://github.com/huggingface/peft/issues/3129 (monkey-patch or narrow targets).
    """
    if not target_suffixes:
        return
    bad: List[str] = []
    for name, module in model.named_modules():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        if isinstance(module, torch.nn.Linear):
            continue
        cls_name = module.__class__.__name__
        if cls_name == "Gemma4ClippableLinear" or cls_name.endswith("ClippableLinear"):
            bad.append(f"{name} ({cls_name})")
    if not bad:
        return
    shown = bad[:6]
    suffix = f" … (+{len(bad) - 6} more)" if len(bad) > 6 else ""
    raise RuntimeError(
        "LoRA target_modules match layer(s) that are not plain torch.nn.Linear — PEFT cannot "
        f"attach adapters here: {', '.join(shown)}{suffix}. "
        "Mitigation: monkey-patch Gemma4ClippableLinear (or the relevant *ClippableLinear class) "
        "to inherit nn.Linear before model load, or set lora_target_modules to suffixes that "
        "still resolve to nn.Linear (inspect model.named_modules()). "
        "Docs: README/guides/apple-silicon/gemma4-guide.md (PEFT / Gemma4ClippableLinear). "
        "Reference: https://github.com/huggingface/peft/issues/3129"
    )


def _validate_lora_target_regex(model: torch.nn.Module, pattern: str) -> List[str]:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid lora_target_modules_regex: {exc}") from exc
    matched: List[str] = []
    incompatible: List[str] = []
    for name, module in model.named_modules():
        if not compiled.fullmatch(name):
            continue
        if not isinstance(module, torch.nn.Linear):
            incompatible.append(f"{name} ({module.__class__.__name__})")
        else:
            matched.append(name)
    if incompatible:
        raise RuntimeError("lora_target_modules_regex matched non-linear modules: " + ", ".join(incompatible[:8]))
    if not matched:
        raise ValueError("lora_target_modules_regex matched no torch.nn.Linear modules")
    return matched


def _trainable_parameter_receipt(model: torch.nn.Module) -> dict[str, Any]:
    by_subsystem = {
        "vision": 0,
        "projector": 0,
        "decoder": 0,
        "output_head": 0,
        "audio": 0,
        "other": 0,
    }
    names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lm_head." in f".{name}." or name.startswith("lm_head."):
            subsystem = "output_head"
        elif "vision_tower" in name:
            subsystem = "vision"
        elif "embed_vision" in name:
            subsystem = "projector"
        elif "language_model" in name:
            subsystem = "decoder"
        elif "audio_tower" in name:
            subsystem = "audio"
        else:
            subsystem = "other"
        by_subsystem[subsystem] += parameter.numel()
        names.append(name)
    return {
        "schema_version": "gemma_trainable_parameters_v1",
        "by_subsystem": by_subsystem,
        "total": sum(by_subsystem.values()),
        "parameter_name_sha256": hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest(),
        "parameter_tensors": len(names),
    }


def _validate_conditioned_prompt_file(profile_config: "ProfileConfig") -> None:
    column = profile_config.get("system_prompt_column")
    template = profile_config.get("conditioned_system_prompt_template")
    expected = profile_config.get("conditioned_system_prompt_sha256")
    provenance = profile_config.get("system_prompt_provenance_path")
    provenance_sha = profile_config.get("system_prompt_provenance_sha256")
    if not column:
        if template or expected or provenance or provenance_sha:
            raise ValueError("system prompt provenance requires system_prompt_column")
        return
    if bool(template or expected) == bool(provenance or provenance_sha):
        raise ValueError("system prompt requires exactly one template or per-row provenance pair")
    path_value = template or provenance
    digest_value = expected or provenance_sha
    if not path_value or not digest_value:
        raise ValueError("system prompt provenance path/hash pair is incomplete")
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("system prompt provenance file is missing")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != str(digest_value):
        raise ValueError("system prompt provenance hash mismatch")
    if template:
        text = path.read_text(encoding="utf-8")
        for placeholder in ("{USER_FULL_NAME}", "{APPLICATION_NAME}"):
            if placeholder not in text:
                raise ValueError(f"conditioned system prompt template lacks {placeholder}")


def main(profile_config: "ProfileConfig", output_dir: str):
    """Main Gemma 3n LoRA training entry.

    Orchestrates dataset loading, model+processor initialization, LoRA injection,
    and Hugging Face Trainer execution. Mirrors the framework's shape used by
    Gemma modules to minimize integration risk.
    """

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    _validate_conditioned_prompt_file(profile_config)
    # Quiet down Hugging Face tokenizers dumping huge AddedToken lists
    try:
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Resolve device and dtype policy for Apple Silicon
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load dataset splits via existing CSV+patch pipeline
    dataset_name = profile_config["dataset"]
    text_column = profile_config.get("text_column", "text")
    max_samples = profile_config.get("max_samples")
    streaming_enabled = profile_config.get("streaming_enabled", False)
    modality = str(profile_config.get("modality", "audio")).strip().lower()
    text_sub_mode = str(profile_config.get("text_sub_mode", "instruction")).strip().lower()
    prompt_column = profile_config.get("prompt_column")
    max_seq_length = int(profile_config.get("max_seq_length", 2048))
    if device.type == "mps" and modality == "text":
        try:
            mps_seq_cap = int(os.environ.get("GEMMA_MPS_MAX_SEQ_LENGTH", "2048"))
        except ValueError:
            mps_seq_cap = 2048
        if max_seq_length > mps_seq_cap:
            logger.warning(
                "MPS text: max_seq_length=%s exceeds cap %s (raise GEMMA_MPS_MAX_SEQ_LENGTH if you need more; "
                "may OOM on unified memory). Truncating.",
                max_seq_length,
                mps_seq_cap,
            )
            max_seq_length = mps_seq_cap

    dataset_config = {
        "name": dataset_name,
        "text_column": text_column,
        "modality": modality,
        "text_sub_mode": text_sub_mode,
    }
    if prompt_column is not None:
        dataset_config["prompt_column"] = prompt_column
    if modality in ("image", "audiovisual"):
        dataset_config["image_sub_mode"] = str(profile_config.get("image_sub_mode", "caption")).strip().lower()
        ipc = profile_config.get("image_path_column")
        if ipc:
            dataset_config["image_path_column"] = str(ipc).strip()

    train_dataset, _ = load_dataset_split(
        split="train",
        dataset_config=dataset_config,
        max_samples=max_samples,
        streaming_enabled=streaming_enabled,
    )
    eval_dataset = None
    if profile_config.get("load_validation", True):
        try:
            eval_dataset, _ = load_dataset_split(
                split="validation",
                dataset_config=dataset_config,
                max_samples=None,
                streaming_enabled=streaming_enabled,
            )
        except Exception as e:
            logger.warning(f"Failed to load validation split; running without eval: {e}")

    image_path_column_resolved = str(profile_config.get("image_path_column") or "image_path").strip() or "image_path"
    if modality in ("image", "audiovisual"):
        dataset_dir = resolve_data_datasets_dir(dataset_name)

        def _resolve_image_paths(batch: dict) -> dict:
            col = image_path_column_resolved
            paths = batch[col]
            out = []
            for p in paths:
                if p and not os.path.isabs(str(p)):
                    out.append(os.path.join(dataset_dir, str(p)))
                else:
                    out.append(p)
            batch[col] = out
            return batch

        train_dataset = train_dataset.map(_resolve_image_paths, batched=True)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(_resolve_image_paths, batched=True)

    # Initialize processor and model
    model_id = profile_config.get("base_model", GemmaTrainingConstants.DEFAULT_BASE_MODEL_ID)
    attn_impl = profile_config.get("attn_implementation", "eager")

    # Wizard may have called gate_gemma_model already; repeating here is cheap and keeps CLI-only runs safe.
    family = detect_family(model_id)
    assert_family_supported(family)
    assert_entrypoint_support("finetune", family)
    logger.info("Gemma family: %s for model_id=%s", family.value, model_id)

    model_revision = profile_config.get("model_revision")
    if model_revision:
        logger.info(f"Using pinned revision: {model_revision}")

    processor = None
    text_tokenizer = None
    if modality == "text":
        logger.info(f"Loading tokenizer (text modality): {model_id}")
        text_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    else:
        logger.info(f"Loading processor ({modality} modality): {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, revision=model_revision)
        if modality in ("image", "audiovisual"):
            _itb = int(profile_config.get("image_token_budget", 280))
            apply_image_token_budget_to_processor(processor, _itb)

    # Determine dtype preference: prefer bfloat16 on MPS when available, else float32
    #
    # Implementation note: Comprehensive bfloat16 testing prevents training failures
    # that would only manifest during actual training. Simple tensor creation tests
    # are insufficient - operations and gradients must also work correctly.
    if device.type == "mps":
        use_bf16 = _test_mps_bfloat16_support(device)
    elif device.type == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
    else:
        use_bf16 = False
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if use_bf16:
        logger.info("bf16 supported on %s: enabling bfloat16 training", device.type)
    else:
        logger.info("bf16 not available on %s: using float32", device.type)

    # Reconcile dtype with profile_config so device.py and this module agree.
    # device.py:apply_device_defaults() may have set profile_config["dtype"] = "float32"
    # for MPS, but the bfloat16 test above may have determined we can use bfloat16.
    # Writing the actual dtype back here ensures downstream code (e.g. logging,
    # checkpoint metadata) reflects the real dtype in use.
    profile_config["dtype"] = "bfloat16" if torch_dtype == torch.bfloat16 else "float32"

    logger.info(f"Loading base model: {model_id}")
    model = load_base_model_for_gemma(
        model_id,
        family=family,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        revision=model_revision,
    )

    # LoRA configuration
    lora_r = int(profile_config.get("lora_r", GemmaTrainingConstants.LORA_R))
    lora_alpha = int(profile_config.get("lora_alpha", GemmaTrainingConstants.LORA_ALPHA))
    lora_dropout = float(profile_config.get("lora_dropout", GemmaTrainingConstants.LORA_DROPOUT))
    # Handle LoRA target modules from config with robust format support
    #
    # Cross-file connection: This resolves the config/code mismatch between:
    # - config.ini:435 which uses comma-separated string format: "q_proj,k_proj,v_proj,o_proj"
    # - Python code which expects list format: ["q_proj", "k_proj", "v_proj", "o_proj"]
    #
    # Called during LoRA configuration setup in main() function
    # Affects PEFT model creation via get_peft_model() later in this function
    target_regex = profile_config.get("lora_target_modules_regex")
    if target_regex and profile_config.get("lora_target_modules"):
        raise ValueError("lora_target_modules_regex and lora_target_modules are mutually exclusive")
    target_modules_from_config = profile_config.get("lora_target_modules") or GemmaTrainingConstants.LORA_TARGET_MODULES

    if isinstance(target_modules_from_config, str):
        # Handle config.ini format: comma-separated string
        # Example: "q_proj,k_proj,v_proj,o_proj" -> ["q_proj", "k_proj", "v_proj", "o_proj"]
        target_modules_list = [module_name.strip() for module_name in target_modules_from_config.split(",")]
    elif isinstance(target_modules_from_config, list):
        # Handle programmatic config format: already a list
        # Example: ["q_proj", "k_proj", "v_proj", "o_proj"] -> use as-is
        target_modules_list = target_modules_from_config
    else:
        # Fallback to safe defaults if config format is unexpected
        logger.warning(f"Unexpected LoRA target modules format: {type(target_modules_from_config)}. Using defaults.")
        target_modules_list = GemmaTrainingConstants.LORA_TARGET_MODULES

    # Validate target modules against actual model architecture.
    # This prevents LoRA injection failures from targeting non-existent modules.
    #
    # Two cases:
    # 1. User specified lora_target_modules in config → their list came through
    #    target_modules_list. If none exist in the model, that is a hard config
    #    error: raise ValueError with the available module names so they can fix it.
    # 2. No user config → target_modules_list came from GemmaTrainingConstants defaults.
    #    Use _discover_candidate_target_modules to find which defaults exist, and
    #    fall back to discovered defaults silently (no user mistake to surface).
    user_specified_targets = bool(profile_config.get("lora_target_modules"))
    discovered_modules = _discover_candidate_target_modules(model)

    # Intersect: which of the requested names are actually present in the model?
    validated_target_modules = [m for m in target_modules_list if m in discovered_modules]

    if target_regex:
        matched_target_modules = _validate_lora_target_regex(model, str(target_regex))
        lora_target_spec: str | List[str] = str(target_regex)
        logger.info("LoRA target regex matched %d linear modules", len(matched_target_modules))
    elif not validated_target_modules:
        if user_specified_targets:
            # User explicitly asked for modules that don't exist → hard error.
            available = sorted({name.split(".")[-1] for name, _ in model.named_modules() if "." in name})
            raise ValueError(
                f"None of the requested lora_target_modules {target_modules_list} exist in the model. "
                f"Available leaf module names include: {available[:40]}. "
                f"Fix 'lora_target_modules' in your profile config."
            )
        else:
            # Defaults weren't found — use whatever _discover found (may be empty,
            # in which case PEFT will raise its own informative error).
            validated_target_modules = discovered_modules
        lora_target_spec = validated_target_modules
    else:
        lora_target_spec = validated_target_modules

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_spec,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if not target_regex:
        _raise_if_lora_targets_use_peft_incompatible_linears(model, validated_target_modules)
    model = get_peft_model(model, lora_cfg)
    model = model.to(device)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainable_receipt = _trainable_parameter_receipt(model)
    (Path(output_dir) / "trainable_parameters.json").write_text(
        json.dumps(trainable_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Install kwarg filter at all PEFT nesting levels.
    # Prevents TypeErrors when HuggingFace Trainer injects unexpected kwargs
    # (e.g. num_items_in_batch for loss scaling) into the Gemma forward pass.
    # Must be called AFTER model.to(device) because device moves may re-wrap the model.
    # See models/common/utils.py:install_kw_filter for the list of stripped kwargs.
    install_kw_filter(model)
    if hasattr(model, "base_model"):
        install_kw_filter(model.base_model)
        if hasattr(model.base_model, "model"):
            install_kw_filter(model.base_model.model)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    def _hf_to_torch(ds: Union[HFDataset, Any]) -> Dataset:
        """Return a map-style Dataset for the Trainer.

        Only map-style datasets.Dataset is supported. Streaming (IterableDataset)
        cannot be used here because the Gemma collator requires random-access
        indexing and len(). Materialize the dataset before calling finetune().
        """
        if isinstance(ds, HFDataset):
            return ds
        from datasets import IterableDataset as _IterableDataset

        if isinstance(ds, _IterableDataset):
            raise ValueError(
                "Streaming IterableDataset is not supported for Gemma SFT. "
                "Load the dataset as a map-style Dataset (streaming=False) "
                "or call .to_iterable_dataset() and .map() to materialize it first."
            )
        raise TypeError(f"Unsupported dataset type for Gemma SFT: {type(ds)}")

    train_ds = _hf_to_torch(train_dataset)
    eval_ds = _hf_to_torch(eval_dataset) if eval_dataset is not None else None

    # Collator: multimodal processor (audio / image) or tokenizer-only (text)
    if modality == "text":
        data_collator = DataCollatorGemmaText(
            tokenizer=text_tokenizer,
            text_column=text_column,
            family=family,
            prompt_column=prompt_column,
            max_length=max_seq_length,
            sub_mode=text_sub_mode,
        )
    elif modality == "image":
        image_sub_mode = str(profile_config.get("image_sub_mode", "caption")).strip().lower()
        image_path_col = str(profile_config.get("image_path_column") or "image_path").strip() or "image_path"
        image_token_budget = int(profile_config.get("image_token_budget", 280))
        image_view_policy = str(profile_config.get("image_view_policy", "single"))
        system_prompt_column = profile_config.get("system_prompt_column")
        data_collator = DataCollatorGemmaImage(
            processor=processor,
            text_column=text_column,
            family=family,
            image_path_column=image_path_col,
            prompt_column=prompt_column,
            image_token_budget=image_token_budget,
            image_view_policy=image_view_policy,
            system_prompt_column=system_prompt_column,
            require_telepathic_contract=to_bool(profile_config.get("require_telepathic_contract", False)),
            completion_only_logits=to_bool(profile_config.get("completion_only_logits", False)),
            max_length=max_seq_length,
            sub_mode=image_sub_mode,
        )
    elif modality == "audiovisual":
        image_path_col = str(profile_config.get("image_path_column") or "image_path").strip() or "image_path"
        data_collator = DataCollatorGemmaAudioVisual(
            processor=processor,
            text_column=text_column,
            family=family,
            image_path_column=image_path_col,
            sampling_rate_hint=None,
        )
    else:
        data_collator = DataCollatorGemmaAudio(
            processor=processor, text_column=text_column, family=family, sampling_rate_hint=None
        )

    # WER metrics for speech runs only; text/image use loss / optional perplexity in train_results.
    compute_metrics_fn = None
    preprocess_logits_for_metrics = None
    if modality == "audio":
        _wer_metrics = build_wer_metrics(
            tokenizer=processor.tokenizer,
            decoder=processor.tokenizer,
            include_cer=False,
            local_files_only=True,
        )
        compute_metrics_fn = _wer_metrics["compute_fn"]

        def _preprocess_logits_for_metrics(logits, labels):
            """Argmax logits immediately after each eval batch to avoid accumulating [B, T, V] tensors.

            Standard HF pattern for causal LM evaluation with plain Trainer.
            Gemma's 256 k vocabulary makes storing full logits across the eval set
            prohibitively expensive (2 samples × 512 tokens × 256 k vocab × 4 bytes ≈ 1 GB
            per batch). Argmaxing here reduces each batch to [B, T] int64 token IDs.

            Called by:
            - Trainer.evaluation_loop() immediately after each eval batch, before
              tensors are gathered across devices/workers.

            Result shape [B, T] is what compute_metrics_fn receives as pred.predictions.
            The 3-D branch in compute_fn (metrics.py) becomes a no-op but does not break.
            """
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        preprocess_logits_for_metrics = _preprocess_logits_for_metrics

    # Guard: HF Trainer.__init__ raises ValueError when eval_strategy != "no" and
    # eval_dataset is None (verified against transformers trainer.py line 439).
    # The validation split load above is wrapped in a broad except, so eval_ds=None
    # is a real runtime path whenever the split is missing or fails to load.
    # We override to "no" here and log a warning so the failure is visible without
    # crashing the entire training run.
    requested_eval_strategy = str(profile_config.get("eval_strategy", GemmaTrainingConstants.DEFAULT_EVAL_STRATEGY))
    if eval_ds is None and requested_eval_strategy != "no":
        logger.warning(
            "eval_strategy=%r requested but no eval_dataset is available; overriding to 'no'.",
            requested_eval_strategy,
        )
        effective_eval_strategy = "no"
    else:
        effective_eval_strategy = requested_eval_strategy

    # Training arguments
    per_device_train_batch_size = int(profile_config.get("per_device_train_batch_size", 2))
    per_device_eval_batch_size = int(profile_config.get("per_device_eval_batch_size", per_device_train_batch_size))
    gradient_accumulation_steps = int(
        profile_config.get("gradient_accumulation_steps", GemmaTrainingConstants.DEFAULT_GRADIENT_ACCUMULATION)
    )
    learning_rate = float(profile_config.get("learning_rate", GemmaTrainingConstants.DEFAULT_LEARNING_RATE))
    num_train_epochs = int(profile_config.get("num_train_epochs", GemmaTrainingConstants.DEFAULT_NUM_TRAIN_EPOCHS))

    # MPS heuristics: text + Gemma 4 multimodal LoRA needs micro-batch 1 on unified memory; scale via grad accumulation.
    if device.type == "mps":
        if modality == "text":
            if per_device_train_batch_size > 1:
                logger.info(
                    "MPS text: clamping per_device_train_batch_size from %s to 1 (use gradient_accumulation_steps for effective batch).",
                    per_device_train_batch_size,
                )
            per_device_train_batch_size = min(per_device_train_batch_size, 1)
            if per_device_eval_batch_size > 1:
                logger.info(
                    "MPS text: clamping per_device_eval_batch_size from %s to 1.",
                    per_device_eval_batch_size,
                )
            per_device_eval_batch_size = min(per_device_eval_batch_size, 1)
        else:
            per_device_train_batch_size = min(per_device_train_batch_size, 2)
            per_device_eval_batch_size = min(per_device_eval_batch_size, 2)

    if device.type == "mps" and modality == "text":
        if os.environ.get("GEMMA_MPS_ALLOW_NO_GRADIENT_CHECKPOINTING", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            gradient_checkpointing = to_bool(profile_config.get("gradient_checkpointing"))
        else:
            gradient_checkpointing = True
    else:
        gradient_checkpointing = to_bool(profile_config.get("gradient_checkpointing"))

    if gradient_checkpointing:
        try:
            model.enable_input_require_grads()
        except Exception as e:
            logger.warning("enable_input_require_grads() failed (continuing): %s", e)

    _do_viz = bool(profile_config.get("visualize"))
    _logging_steps = int(profile_config.get("logging_steps", GemmaTrainingConstants.DEFAULT_LOGGING_STEPS))
    if _do_viz:
        # HF only calls TrainerCallback.on_log on the logging_steps schedule; the
        # dashboard follows that cadence. Force 1 so the visualizer updates every
        # optimizer step while viz is on (profile logging_steps is ignored).
        _logging_steps = 1
        logger.info("Visualization: logging_steps=1 (dashboard updates every training step).")

    train_kw = dict(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=float(profile_config.get("weight_decay", 0.0)),
        warmup_steps=int(profile_config.get("warmup_steps", 0)),
        warmup_ratio=float(profile_config.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(profile_config.get("lr_scheduler_type", "linear")),
        bf16=use_bf16,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=_logging_steps,
        save_steps=int(profile_config.get("save_steps", 1000)),
        save_total_limit=int(profile_config.get("save_total_limit", 10)),
        save_strategy=str(profile_config.get("save_strategy", GemmaTrainingConstants.DEFAULT_SAVE_STRATEGY)),
        eval_strategy=effective_eval_strategy,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        log_level="error",
        # Avoid Trainer.evaluate() stalling on MPS (transformers#27181); matches gemma4-guide.md.
        skip_memory_metrics=True,
        torch_compile=resolve_training_torch_compile(device, profile_config),
        seed=int(profile_config.get("seed", 42)),
        data_seed=int(profile_config.get("seed", 42)),
        full_determinism=to_bool(profile_config.get("full_determinism", False)),
    )
    _ms = profile_config.get("max_steps")
    if _ms is not None and _ms != "":
        train_kw["max_steps"] = int(_ms)
    args = TrainingArguments(**train_kw)

    # Seed for reproducibility
    set_seed(args.seed)

    trainer_callbacks: List[Any] = []
    trainer_callbacks.append(ImmutableCheckpointCallback())
    trainer_callbacks.append(RedactedTrainingMetricsCallback(output_dir))
    stop_after_step = profile_config.get("stop_after_step")
    if stop_after_step not in (None, ""):
        trainer_callbacks.append(StopAfterStepCallback(int(stop_after_step)))
    required_gradient_subsystems = str(profile_config.get("require_gradient_subsystems", "")).split(",")
    if any(value.strip() for value in required_gradient_subsystems):
        trainer_callbacks.append(
            GradientSubsystemReceiptCallback(output_dir, required_gradient_subsystems)
        )
    if _do_viz:
        trainer_callbacks.append(VisualizerTrainerCallback(update_every_steps=max(1, _logging_steps)))

    completion_only_logits = to_bool(profile_config.get("completion_only_logits", False))
    if _do_viz and completion_only_logits:
        raise ValueError("completion_only_logits is not compatible with visualization")
    trainer_cls = GemmaVizTrainer if _do_viz else (CompletionOnlyTrainer if completion_only_logits else Trainer)
    trainer_kwargs: dict[str, Any] = {}
    if _do_viz:
        trainer_kwargs["visualize"] = True

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=trainer_callbacks,
        **trainer_kwargs,
    )

    if _do_viz:
        for cb in trainer_callbacks:
            if isinstance(cb, VisualizerTrainerCallback):
                cb.bind_trainer(trainer)
        if gradient_checkpointing:
            logger.info(
                "Visualization: gradient checkpointing is on; some models omit attention tensors "
                "when checkpointing — attention heatmaps may stay empty unless the stack returns them."
            )

    logger.info("Starting Gemma LoRA training...")
    resume_from_checkpoint = profile_config.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        verify_complete_checkpoint(Path(str(resume_from_checkpoint)).expanduser().resolve())
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(Path(str(resume_from_checkpoint)).expanduser().resolve()) if resume_from_checkpoint else None
        )
    )
    logger.info("Training complete. Saving adapter...")
    trainer.save_model()
    if _do_viz:
        try:
            from gemma_tuner.visualizer import broadcast_training_finished

            broadcast_training_finished()
        except Exception as e:
            logger.debug("broadcast_training_finished failed (non-fatal): %s", e)

    _finalize_training_artifacts(
        output_dir,
        trainer=trainer,
        train_result=train_result,
        modality=modality,
        integrity_metadata={
            "model_id": model_id,
            "dtype": str(torch_dtype),
            "device": str(device),
            "modality": modality,
        },
    )

    empty_cache()
    return {"train_metrics": getattr(train_result, "metrics", {})}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Gemma 3n LoRA fine-tuning")
    parser.add_argument("--profile_config", required=True, help="JSON string or file path with profile configuration")
    parser.add_argument("--output_dir", required=True, help="Output directory for training artifacts")
    args = parser.parse_args()
    if args.profile_config.endswith(".json") and os.path.isfile(args.profile_config):
        with open(args.profile_config, "r") as f:
            cfg = json.load(f)
    else:
        cfg = json.loads(args.profile_config)
    main(cfg, args.output_dir)
