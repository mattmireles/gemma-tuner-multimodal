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
   - Loads directly with AutoModelForCausalLM
   - Saves to {source_dir}-export/

Called by:
- core/ops.py:export() which is called by cli_typer.py:export()
- scripts/export_gemma_lora.py:main() as a direct entry point

Calls to:
- utils/device.py:get_device() for device selection
- transformers.AutoModelForCausalLM for model loading
- peft.PeftModel for LoRA adapter merging
"""

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from gemma_tuner.utils.device import get_device

logger = logging.getLogger(__name__)

# Filename written by PEFT's save_pretrained() to mark a directory as a LoRA adapter.
# Presence of this file is the canonical signal that we need to merge before exporting.
_ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def export_model_dir(model_path_or_profile: str) -> str:
    """Export a trained Gemma model (full or LoRA adapter) to a SafeTensors directory.

    Auto-detects whether the source is a LoRA adapter directory or a full model
    and handles each case correctly. LoRA adapters are merged into the base model
    before saving so the output is a self-contained, deployment-ready checkpoint.

    Called by:
    - core/ops.py:export()
    - scripts/export_gemma_lora.py:main()

    Args:
        model_path_or_profile: Local directory path or HuggingFace model id.

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

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # PeftModel loads the adapter weights on top of the base model, then
        # merge_and_unload() folds them into the base weights and returns a
        # plain AutoModelForCausalLM with no PEFT overhead.
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
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_profile,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        processor_source = model_path_or_profile

    out_dir = str(source_path) + "-export"
    logger.info("Saving exported model to: %s", out_dir)
    model.save_pretrained(out_dir)

    # Save processor (tokenizer + feature extractor) alongside the weights so
    # the exported directory is fully self-contained: any caller can do
    # AutoProcessor.from_pretrained(out_dir) without needing the original source.
    try:
        processor = AutoProcessor.from_pretrained(processor_source)
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

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a model to HF/SafeTensors directory")
    parser.add_argument("model", help="Path or model id to export")
    args = parser.parse_args()
    export_model_dir(args.model)
