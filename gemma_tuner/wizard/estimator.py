#!/usr/bin/env python3

"""
Gemma Fine-Tuning Wizard - Training Estimation and Method Configuration

This module handles training time/resource estimation and method-specific
configuration (LoRA parameters).

All shared constants and utilities are imported from gemma_tuner.wizard.base to avoid
circular imports. NEVER import from the wizard package root.

Called by:
- wizard.runner.wizard_main() for method-specific config and time estimation
- wizard/__init__.py re-exports for backward compatibility

Integrates with:
- wizard.base: ModelSpecs, get_device_info, apple_style, console
"""

from datetime import datetime, timedelta
from typing import Any, Dict

import questionary

from gemma_tuner.wizard.base import (
    ModelSpecs,
    apple_style,
    console,
    get_device_info,
)


def configure_method_specifics(
    method: Dict[str, Any], model: str | tuple, seed: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Step 5: Method-specific configuration (progressive disclosure)"""
    from gemma_tuner.wizard.config_store import _read_config

    # Defensive: older call sites may pass a (model, seed) tuple.
    if isinstance(model, tuple):
        model, seed_from_tuple = model
        if seed is None and isinstance(seed_from_tuple, dict):
            seed = seed_from_tuple

    config = {} if seed is None else dict(seed)

    if method["key"] == "lora":
        console.print("\n[bold]Step 5: LoRA Configuration[/bold]")
        console.print("[dim]LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning[/dim]")

        # LoRA rank
        rank_choices = [
            {"name": "4 (Ultra lightweight)", "value": 4},
            {"name": "8 (Lightweight)", "value": 8},
            {"name": "16 (Balanced) ⭐ Recommended", "value": 16},
            {"name": "32 (High capacity)", "value": 32},
            {"name": "64 (Maximum capacity)", "value": 64},
        ]

        config["lora_r"] = questionary.select(
            "LoRA rank (higher = more parameters to train):", choices=rank_choices, style=apple_style
        ).ask()

        # LoRA alpha (smart default based on rank)
        default_alpha = config["lora_r"] * 2
        alpha_choices = [
            {"name": f"{default_alpha} (Recommended)", "value": default_alpha},
            {"name": f"{config['lora_r']} (Conservative)", "value": config["lora_r"]},
            {"name": f"{config['lora_r'] * 4} (Aggressive)", "value": config["lora_r"] * 4},
            {"name": "Custom value", "value": "custom"},
        ]

        alpha = questionary.select(
            "LoRA alpha (controls adaptation strength):", choices=alpha_choices, style=apple_style
        ).ask()

        if alpha == "custom":
            alpha_str = questionary.text("Enter custom alpha value:", default=str(default_alpha), style=apple_style).ask()
            try:
                alpha = int(alpha_str) if alpha_str is not None else default_alpha
            except ValueError:
                alpha = default_alpha

        config["lora_alpha"] = alpha
        config["lora_dropout"] = 0.1  # Smart default
        config["use_peft"] = True


    return config


def estimate_training_time(
    method: Dict[str, Any], model: str, dataset: Dict[str, Any], method_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Estimate training time and resource usage"""

    device_info = get_device_info()

    # Look up model specs, defaulting to the smaller Gemma variant
    default_specs = ModelSpecs.MODELS.get("gemma-4-e2b-it", list(ModelSpecs.MODELS.values())[0])
    model_specs = ModelSpecs.MODELS.get(model, default_specs)

    # Rough estimation based on dataset size
    if "files" in dataset:
        estimated_samples = dataset["files"] * 10  # Assume 10 samples per file on average
    else:
        estimated_samples = 100000  # Default assumption

    # Base time calculation (hours for 100k samples)
    base_hours = model_specs["hours_100k"]
    sample_ratio = estimated_samples / 100000
    method_multiplier = method["time_multiplier"]
    device_multiplier = device_info["performance_multiplier"]

    estimated_hours = base_hours * sample_ratio * method_multiplier * device_multiplier

    # Memory calculation
    base_memory = model_specs["memory_gb"]
    method_memory_multiplier = method["memory_multiplier"]
    estimated_memory = base_memory * method_memory_multiplier

    return {
        "hours": estimated_hours,
        "memory_gb": estimated_memory,
        "samples": estimated_samples,
        "eta": datetime.now() + timedelta(hours=estimated_hours),
    }
