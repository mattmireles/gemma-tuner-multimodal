#!/usr/bin/env python3
"""
Fine-Tuning Orchestrator for Gemma Models

Routes training requests to the Gemma-specific training implementation.
Handles LoRA parameter detection, optional visualization server startup,
and dynamic module import.

Called by:
- core/ops.py:finetune() for all training workflows
- Batch training scripts processing multiple profiles
- CI/CD pipelines for automated model training

Calls to:
- gemma_tuner.models.gemma.finetune.main() for all Gemma model training

Each model-specific module must implement:
- main(profile_config, output_dir) function with identical signature
- Compatible parameter handling for profile configuration
- Consistent output directory structure for downstream evaluation
"""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_tuner.core.profile_config import ProfileConfig

logger = logging.getLogger(__name__)


class ModelDetectionConstants:
    """Named constants for model type detection and routing configuration."""

    # LoRA Parameter Detection
    # These parameter names indicate LoRA training when present in profile configuration
    LORA_PARAMETER_NAMES = [
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    ]

    # Model Type Detection Substrings
    GEMMA_MODEL_SUBSTRING = "gemma"

    # Module Path Templates
    MODULE_PATH_TEMPLATE = "gemma_tuner.models.{model_type}.finetune"

    # Visualization Server Configuration
    VISUALIZATION_HOST = "127.0.0.1"
    VISUALIZATION_PORT = 8080


def main(profile_config: "ProfileConfig", output_dir: str):
    """
    Orchestrates Gemma fine-tuning by routing to the Gemma training implementation.

    Called by:
    - core/ops.py:finetune() after profile configuration loading

    Calls to:
    - gemma_tuner.models.gemma.finetune.main() for all Gemma models

    Args:
        profile_config (dict): Complete training configuration including:
            - model: Model identifier (e.g., "gemma-3n-e4b-it")
            - dataset: Dataset configuration
            - hyperparameters: Learning rate, batch size, epochs, etc.
            - lora_*: LoRA-specific parameters (lora_r, lora_alpha, etc.)
        output_dir (str): Directory for saving training artifacts, checkpoints, and logs

    Raises:
        ValueError: If model name doesn't contain 'gemma'
        ImportError: If training module fails to import
    """

    model_name = profile_config["model"]

    # Validate this is a Gemma model
    if ModelDetectionConstants.GEMMA_MODEL_SUBSTRING not in model_name:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"This tool only supports Gemma models. "
            f"Model name must contain 'gemma' (e.g., 'gemma-3n-e4b-it', 'google/gemma-4-E2B-it')."
        )

    # Check for LoRA config (informational logging only - Gemma defaults to LoRA internally)
    has_lora_config = any(param in profile_config for param in ModelDetectionConstants.LORA_PARAMETER_NAMES)
    if has_lora_config:
        logger.info(f"Gemma LoRA training: model='{model_name}', explicit LoRA config present")
    else:
        logger.info(f"Gemma training: model='{model_name}' (LoRA applied by default)")

    model_type = "gemma"

    # Dynamic import of the Gemma training module
    module_path = ModelDetectionConstants.MODULE_PATH_TEMPLATE.format(model_type=model_type)

    try:
        finetune_module = importlib.import_module(module_path)
        logger.info(f"Successfully imported training module: {module_path}")
    except ImportError as e:
        file_path = module_path.replace(".", "/") + ".py"
        raise ImportError(
            f"Failed to import fine-tuning module '{module_path}' (file: '{file_path}'). "
            f"Error: {e}. "
            f"Verify the file exists and all dependencies are installed."
        )

    # Optional training visualization server
    visualize_enabled = profile_config.get("visualize", False)

    if visualize_enabled:
        logger.info("Starting training visualization server...")
        try:
            from gemma_tuner.visualizer import start_visualization_server

            start_visualization_server(
                host=ModelDetectionConstants.VISUALIZATION_HOST,
                port=ModelDetectionConstants.VISUALIZATION_PORT,
                # Match wizard / CLI UX: user opted into visualization — open the dashboard.
                open_browser=True,
            )
            logger.info(
                f"Visualization server started at "
                f"http://{ModelDetectionConstants.VISUALIZATION_HOST}:{ModelDetectionConstants.VISUALIZATION_PORT}"
            )
        except ImportError as e:
            logger.warning(f"Could not start training visualizer: {e}")
            logger.warning(
                "To enable visualization, install optional dependencies: "
                "pip install flask flask-socketio. "
                "Training will continue without visualization."
            )

    # Delegate to Gemma fine-tuning implementation
    result = finetune_module.main(profile_config, output_dir)
    return result if isinstance(result, dict) else {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma models with dynamic routing.")
    parser.add_argument(
        "--profile_config",
        required=True,
        help="Profile configuration as JSON string.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for training artifacts.")
    args = parser.parse_args()

    import json

    try:
        profile_config = json.loads(args.profile_config)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for profile_config: {e}")

    main(profile_config, args.output_dir)
