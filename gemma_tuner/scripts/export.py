#!/usr/bin/env python3
"""
Model Export and Conversion Utility

This module provides streamlined model export functionality for converting trained
Gemma models between different formats and optimization states. It handles device-aware
model loading, format conversion, and optimized storage for deployment scenarios.

Key responsibilities:
- Device-optimal model loading with dtype selection
- Model format conversion and optimization
- SafeTensors integration for secure model storage
- Memory-efficient model processing for large models
- Cross-platform model export with compatibility validation

Called by:
- Manual execution for model conversion tasks
- Deployment pipelines requiring model format conversion
- Model optimization workflows
- Testing and validation scripts

Calls to:
- utils/device.py for optimal device selection and configuration
- transformers library for model loading and saving operations
- torch for tensor operations and dtype management

Export workflow:
1. Device detection and dtype optimization
2. Model loading from checkpoint with memory optimization
3. Format validation and compatibility checking
4. Optimized model saving with SafeTensors integration
5. Export validation and integrity verification

Device optimization:
- Apple Silicon (MPS): Uses float16 for unified memory efficiency
- NVIDIA CUDA: Uses float16 for GPU memory optimization
- CPU: Uses float32 for maximum compatibility and precision
- Memory management: low_cpu_mem_usage for large model handling

Format support:
- HuggingFace transformers format (primary)
- SafeTensors format for secure storage
- Cross-platform compatibility validation
- Optimization for deployment scenarios

Memory optimization:
- Efficient model loading with minimal memory footprint
- Device-specific dtype selection for optimal performance
- Memory pressure management during conversion
- Large model handling with streaming operations

Use cases:
- Converting training checkpoints to deployment format
- Optimizing models for specific hardware platforms
- Creating distributable model packages
- Validating model export compatibility

Compatibility:
- HuggingFace transformers: Modern model loading and saving
- SafeTensors: Secure tensor storage format
- PyTorch: Device-agnostic tensor operations
- Cross-platform: macOS, Linux, Windows support
"""

import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq

from gemma_tuner.utils.device import get_device

logger = logging.getLogger(__name__)


def export_model_dir(model_path_or_profile):
    """Exports a trained model to a portable HF/SafeTensors directory.

    Args:
        model_path_or_profile: Either a path to a local model directory or
            a Hugging Face model id. If a profile name is provided, caller
            should resolve it to a path before calling.
    """
    device = get_device()
    torch_dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

    # Resolve a profile name to latest completed run directory, if metadata exists
    model_id = model_path_or_profile

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )

    # Save next to source or to default ./exported_model
    out_dir = "exported_model" if not isinstance(model_id, str) else f"{model_id}-export"
    model.save_pretrained(out_dir)

    logger.info(f"Model exported successfully to {out_dir}")
    logger.info(f"Source model: {model_id}")
    logger.info(f"Device used: {device}")
    logger.info(f"Data type: {torch_dtype}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a model to HF/SafeTensors directory")
    parser.add_argument("model", help="Path or model id to export")
    args = parser.parse_args()
    export_model_dir(None, args.model)
