#!/usr/bin/env python3
"""
Model Export and Conversion Utility

Exports trained Gemma models to portable HF/SafeTensors directories.
Handles two cases transparently:

1. **LoRA adapter directory** (contains adapter_config.json):
   - Reads base_model_name_or_path from adapter_config.json
   - Loads base model + merges LoRA weights via PeftModel + merge_and_unload()
   - Saves the merged full model to {source_dir}-export/

2. **Full model directory or HuggingFace Hub id**:
   - Loads via the same family-aware loader as finetune (multimodal towers preserved for Gemma 3n/4)
   - Saves to {source_dir}-export/

Called by:
- core/ops.py:export() which is called by cli_typer.py:export()
- scripts/export_gemma_lora.py:main() as a direct entry point

Calls to:
- utils/device.py:get_device() for device selection
- gemma_tuner.models.gemma.base_model_loader:load_base_model_for_gemma (same family-aware path as finetune)
- peft.PeftModel for LoRA adapter merging
"""

import json
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor

from gemma_tuner.models.common.collators import apply_image_token_budget_to_processor
from gemma_tuner.models.gemma.base_model_loader import load_base_model_for_gemma
from gemma_tuner.models.gemma.family import gate_gemma_model
from gemma_tuner.utils.device import get_device
from gemma_tuner.utils.integrity import create_integrity_manifest

logger = logging.getLogger(__name__)

# Filename written by PEFT's save_pretrained() to mark a directory as a LoRA adapter.
# Presence of this file is the canonical signal that we need to merge before exporting.
_ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def export_model_dir(model_path_or_profile: str, model_revision: str | None = None) -> str:
    """Export a trained Gemma model (full or LoRA adapter) to a SafeTensors directory.

    Auto-detects whether the source is a LoRA adapter directory or a full model
    and handles each case correctly. LoRA adapters are merged into the base model
    before saving so the output is a self-contained, deployment-ready checkpoint.

    Called by:
    - core/ops.py:export()
    - scripts/export_gemma_lora.py:main()

    Args:
        model_path_or_profile: Local directory path or HuggingFace model id.
        model_revision: Optional specific model revision (commit hash) to load for reproducibility.

    Returns:
        str: Path to the exported model directory.

    Raises:
        FileNotFoundError: If model_path_or_profile is a local path that does not exist.
        KeyError: If adapter_config.json is missing base_model_name_or_path.
    """
    device = get_device()
    # bfloat16 is preferred on MPS/CUDA for memory efficiency; CPU uses float32 for compat.
    torch_dtype = torch.bfloat16 if device.type in ["cuda", "mps"] else torch.float32

    source_path = Path(model_path_or_profile)
    adapter_config_path = source_path / _ADAPTER_CONFIG_FILENAME

    if adapter_config_path.exists():
        # --- LoRA adapter path: merge weights into base model before saving ---
        logger.info("Detected LoRA adapter directory: %s", source_path)
        logger.info("Reading base model path from %s", _ADAPTER_CONFIG_FILENAME)

        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)

        base_model_id = adapter_cfg["base_model_name_or_path"]
        logger.info("Loading base model: %s", base_model_id)
        family = gate_gemma_model(base_model_id, entrypoint="export")

        if model_revision:
            logger.info("Using pinned revision for export: %s", model_revision)
        base_model = load_base_model_for_gemma(
            base_model_id,
            family=family,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            revision=model_revision,
        )

        # PeftModel loads the adapter weights on top of the base model, then
        # merge_and_unload() folds them into the base weights and returns a
        # plain model with no PEFT overhead (same multimodal class as training).
        from peft import PeftModel

        logger.info("Loading LoRA adapter weights from: %s", source_path)
        peft_model = PeftModel.from_pretrained(base_model, str(source_path))

        logger.info("Merging LoRA weights into base model...")
        model = peft_model.merge_and_unload()
        # Processor comes from the base model so downstream users can load
        # tokenizer + feature extractor from the exported directory alone.
        processor_source = base_model_id

    else:
        # --- Full model path or HuggingFace Hub id ---
        logger.info("Loading full model: %s", model_path_or_profile)
        family = gate_gemma_model(str(model_path_or_profile), entrypoint="export")
        if model_revision:
            logger.info("Using pinned revision for export: %s", model_revision)
        model = load_base_model_for_gemma(
            str(model_path_or_profile),
            family=family,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            revision=model_revision,
        )
        processor_source = model_path_or_profile

    out_dir = str(source_path) + "-export"
    logger.info("Saving exported model to: %s", out_dir)
    model.save_pretrained(out_dir)

    # Save processor (tokenizer + feature extractor) alongside the weights so
    # the exported directory is fully self-contained: any caller can do
    # AutoProcessor.from_pretrained(out_dir) without needing the original source.
    try:
        processor = AutoProcessor.from_pretrained(processor_source, revision=model_revision)
        for meta_base in (source_path, source_path.parent):
            meta_path = meta_base / "metadata.json"
            if not meta_path.is_file():
                continue
            try:
                with open(meta_path) as mf:
                    meta = json.load(mf)
                cfg = meta.get("config") or {}
                if str(cfg.get("modality", "")).lower() == "image" and cfg.get("image_token_budget") is not None:
                    apply_image_token_budget_to_processor(processor, int(cfg["image_token_budget"]))
                    logger.info(
                        "Applied image_token_budget=%s from %s for train/serve consistency.",
                        cfg["image_token_budget"],
                        meta_path,
                    )
            except Exception as exc_meta:
                logger.warning("Could not read image_token_budget from metadata (%s): %s", meta_path, exc_meta)
            break
        processor.save_pretrained(out_dir)
        logger.info("  Processor saved alongside model weights.")
    except Exception as exc:
        # Non-fatal: some base models may not have an AutoProcessor; log and continue.
        logger.warning("Could not save processor to export dir (non-fatal): %s", exc)

    logger.info("Export complete.")
    logger.info("  Source : %s", model_path_or_profile)
    logger.info("  Output : %s", out_dir)
    logger.info("  Device : %s", device)
    logger.info("  Dtype  : %s", torch_dtype)
    logger.info("  Params : %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # Create integrity manifest for exported model
    try:
        create_integrity_manifest(
            out_dir,
            metadata={
                "source": str(model_path_or_profile),
                "dtype": str(torch_dtype),
                "device": str(device),
                "is_lora_adapter": adapter_config_path.exists(),
            },
        )
    except Exception as e:
        logger.warning("Failed to create integrity manifest (non-fatal): %s", e)

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a model to HF/SafeTensors directory")
    parser.add_argument("model", help="Path or model id to export")
    parser.add_argument(
        "--revision",
        dest="model_revision",
        default=None,
        help="Specific model revision (commit hash) to load for reproducible export",
    )
    args = parser.parse_args()
    export_model_dir(args.model, model_revision=args.model_revision)
