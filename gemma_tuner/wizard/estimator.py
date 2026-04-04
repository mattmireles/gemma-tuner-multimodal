#!/usr/bin/env python3

"""
Gemma Fine-Tuning Wizard - Training Estimation and Method Configuration

This module handles training time/resource estimation and method-specific
configuration (LoRA parameters, distillation teacher/temperature selection).

All shared constants and utilities are imported from gemma_tuner.wizard.base to avoid
circular imports. NEVER import from the wizard package root.

Called by:
- wizard.runner.wizard_main() for method-specific config and time estimation
- wizard/__init__.py re-exports for backward compatibility

Integrates with:
- wizard.base: WizardConstants, ModelSpecs, _infer_num_mel_bins,
  get_device_info, apple_style, console
- wizard.config: _read_config for config.ini access
"""

from datetime import datetime, timedelta
from typing import Any, Dict

import questionary

from gemma_tuner.wizard.base import (
    ModelSpecs,
    _infer_num_mel_bins,
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

    elif method["key"] == "distillation":
        console.print("\n[bold]Step 5: Distillation Configuration[/bold]")
        # If user already chose Custom Hybrid in Step 2, skip asking architecture again
        arch_choice = (
            "custom" if (model == "__custom_hybrid__" or config.get("student_model_type") == "custom") else "standard"
        )

        # Define student model path
        if arch_choice == "custom":
            # Encoder/decoder sources
            if not config.get("student_encoder_from") or not config.get("student_decoder_from"):
                cfg = _read_config()
                available_models = [s.replace("model:", "") for s in cfg.sections() if s.startswith("model:")]
                large_like = [m for m in available_models if ("large" in m or "medium" in m)]
                small_like = [m for m in available_models if ("tiny" in m or "base" in m or "small" in m)]
                encoder_source = questionary.select(
                    "Choose an Encoder source (teacher model)",
                    choices=[{"name": m, "value": m} for m in large_like],
                    style=apple_style,
                ).ask()
                decoder_source = questionary.select(
                    "Choose a Decoder source (small/efficient model)",
                    choices=[{"name": m, "value": m} for m in small_like],
                    style=apple_style,
                ).ask()
                # Save in config
                config["student_model_type"] = "custom"
                config["student_encoder_from"] = encoder_source
                config["student_decoder_from"] = decoder_source
            else:
                encoder_source = config.get("student_encoder_from")
                decoder_source = config.get("student_decoder_from")

            # Teacher selection (guide to match encoder mel bins)
            teacher_models = ["gemma-3n-e4b-it", "gemma-3n-e4b-it", "gemma-3n-e4b-it"]
            student_mels = _infer_num_mel_bins(encoder_source)
            teacher_choices = []
            incompatible_count = 0
            for teacher in teacher_models:
                txt = teacher
                teacher_mels = _infer_num_mel_bins(teacher)
                if teacher_mels != student_mels:
                    txt += f" ({teacher_mels} mel bins vs student's {student_mels} - incompatible)"
                    incompatible_count += 1
                teacher_choices.append({"name": txt, "value": teacher})

            if incompatible_count == len(teacher_models):
                console.print(
                    f"[yellow]⚠️ Warning: All teacher models have incompatible mel bins with your encoder ({student_mels} mel bins).[/yellow]"
                )
                console.print(
                    "[yellow]Training may fail or produce poor results. Consider choosing a different encoder.[/yellow]"
                )

            teacher_choice = questionary.select(
                "Which teacher model should we distill knowledge from?",
                choices=teacher_choices,
                style=apple_style,
            ).ask()
        else:
            # Standard student: teacher from curated list with compatibility filter
            teacher_models = ["gemma-3n-e4b-it", "gemma-3n-e4b-it", "gemma-3n-e4b-it"]
            teacher_choices = []
            for teacher in teacher_models:
                if teacher != model:
                    choice_text = f"{teacher}"
                    teacher_choices.append({"name": choice_text, "value": teacher})
            student_mels = _infer_num_mel_bins(model)
            filtered_teacher_choices = []
            for ch in teacher_choices:
                t_model = ch["value"]
                if _infer_num_mel_bins(t_model) != student_mels:
                    ch = {"name": ch["name"] + " (incompatible mel bins; not recommended)", "value": t_model}
                filtered_teacher_choices.append(ch)
            teacher_choice = questionary.select(
                "Which teacher model should we distill knowledge from?",
                choices=filtered_teacher_choices,
                style=apple_style,
            ).ask()
        # Resolve to full HF repo id via config.ini when possible
        try:
            cfg = _read_config()
            sec = f"model:{teacher_choice}"
            if cfg.has_section(sec) and cfg.has_option(sec, "base_model"):
                resolved_teacher = cfg.get(sec, "base_model")
            else:
                resolved_teacher = (
                    f"openai/{teacher_choice}" if teacher_choice.startswith("whisper-") else teacher_choice
                )
        except Exception:
            resolved_teacher = teacher_choice
        config["teacher_model"] = resolved_teacher

        # Temperature
        temp_choices = [
            {"name": "2.0 (Conservative)", "value": 2.0},
            {"name": "5.0 (Balanced) ⭐ Recommended", "value": 5.0},
            {"name": "10.0 (Aggressive)", "value": 10.0},
            {"name": "Custom value", "value": "custom"},
        ]

        temperature = questionary.select(
            "Distillation temperature (higher = softer teacher guidance):", choices=temp_choices, style=apple_style
        ).ask()

        if temperature == "custom":
            temperature = questionary.text("Enter custom temperature:", default="5.0", style=apple_style).ask()
            temperature = float(temperature)

        config["temperature"] = temperature

    return config


def estimate_training_time(
    method: Dict[str, Any], model: str, dataset: Dict[str, Any], method_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Estimate training time and resource usage"""

    device_info = get_device_info()

    # Handle custom hybrid models by using encoder source for estimation
    if model == "__custom_hybrid__" and method_config:
        encoder_source = method_config.get("student_encoder_from", "gemma-3n-e4b-it")
        # Clean up model name to match ModelSpecs keys
        if "/" in encoder_source:
            encoder_source = encoder_source.split("/")[-1]
        model_specs = ModelSpecs.MODELS.get(encoder_source, ModelSpecs.MODELS["gemma-3n-e4b-it"])
    else:
        model_specs = ModelSpecs.MODELS.get(model, ModelSpecs.MODELS["gemma-3n-e4b-it"])

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
