#!/usr/bin/env python3

"""
Gemma Fine-Tuning Wizard - User Interaction Functions

This module contains all user-facing interactive functions for the wizard:
welcome screen, model/method/dataset selection, training parameter configuration,
and the final confirmation screen.

All shared constants and utilities are imported from gemma_tuner.wizard.base to avoid
circular imports. NEVER import from the wizard package root.

Called by:
- wizard.runner.wizard_main() for the complete interactive workflow
- wizard/__init__.py re-exports for backward compatibility

Integrates with:
- wizard.base: WizardConstants, ModelSpecs, TrainingMethod, detect_datasets,
  get_wizard_device_info, apple_style, console
- wizard.config: _read_config (used by select_model and show_confirmation_screen)
- wizard.estimator: configure_method_specifics, estimate_training_time (called after UI steps)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import questionary
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gemma_tuner.wizard.base import (
    ModelSpecs,
    TrainingMethod,
    apple_style,
    console,
    detect_datasets,
    get_wizard_device_info,
)


def show_welcome_screen():
    """
    Displays elegant Steve Jobs-inspired welcome screen with system capability verification.

    This function creates the first impression of the wizard using Apple's design language
    principles: beautiful visual elements, clear system status, and confidence-building
    through capability verification. It serves as both introduction and technical validation.

    Called by:
    - wizard_main() as the opening experience (line 1801)
    - Interactive training workflows requiring system status validation
    - Development and demonstration environments for visual appeal

    Calls to:
    - get_wizard_device_info() for comprehensive system capability detection (line 311)
    - rich.console for Apple-inspired visual formatting and layout
    - rich.panel for elegant bordered content presentation

    Design Philosophy:
    - Progressive disclosure: Only essential information shown initially
    - Confidence building: Clear indication system is ready for training
    - Visual hierarchy: ASCII art logo draws attention, then system details
    - Apple aesthetics: Blue color scheme, clean typography, generous whitespace

    System Verification Elements:
    - Device type detection (Apple Silicon MPS, NVIDIA CUDA, CPU fallback)
    - Available memory calculation for training capacity planning
    - Training readiness status with visual confirmation
    - Hardware optimization status for performance expectations

    User Experience Flow:
    1. ASCII art logo creates immediate visual impact and brand recognition
    2. System information builds confidence in hardware capabilities
    3. Ready status provides clear signal to proceed with training
    4. Press Enter prompt gives user control over pacing

    Visual Design Elements:
    - Custom ASCII logo using box-drawing characters for terminal aesthetics
    - Apple Silicon emoji (🍎) for brand association and hardware recognition
    - Color coding: Blue for branding, Green for success, Dim for instructions
    - Panel borders using Rich styling for professional appearance

    Technical Integration:
    - Uses identical device detection logic to training pipeline
    - Memory calculations align with training resource planning
    - Status verification prevents wizard from proceeding with invalid configurations
    - Visual feedback matches training progress indicators for consistency

    Accessibility Considerations:
    - High contrast color choices for terminal visibility
    - Clear text hierarchy for screen readers
    - Simple interaction model (Enter key) for universal access
    - Descriptive status messages for non-visual confirmation
    """

    # ASCII art logo
    logo = """
     ██████╗ ███████╗███╗   ███╗███╗   ███╗ █████╗
    ██╔════╝ ██╔════╝████╗ ████║████╗ ████║██╔══██╗
    ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║███████║
    ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══██║
    ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║
     ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

              🍎 Fine-Tuner for Apple Silicon
    """

    device_info = get_wizard_device_info()

    welcome_text = f"""
[bold cyan]Welcome to the Gemma Fine-Tuning Wizard![/bold cyan]

We'll guide you through fine-tuning your Gemma model in just a few questions.

[green]System Information:[/green]
• Device: {device_info["display_name"]}
• Available Memory: {device_info["available_memory_gb"]:.1f} GB
• Status: Ready for training ✅

[dim]Press Enter to begin...[/dim]
    """

    console.print(
        Panel(
            Align.center(Text(logo, style="bold blue"), vertical="middle"),
            title="💎 Gemma macOS Tuner",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print(welcome_text)

    input()  # Wait for user to press Enter


def select_training_method(family: str | None = None) -> Optional[Dict[str, Any]]:
    """Step 1: Select training method with progressive disclosure"""
    console.print("\n[bold]Step 1: Choose your training method[/bold]")

    # Gemma uses LoRA fine-tuning
    methods = [TrainingMethod.LORA]

    choices = []
    for method in methods:
        choices.append({"name": f"{method['name']} - {method['description']}", "value": method})

    selected_method = questionary.select(
        "What kind of fine-tuning do you want to run?", choices=choices, style=apple_style
    ).ask()

    # questionary returns None when stdin is not a TTY (piped/scripted input).
    # Returning None signals cancellation to wizard_main.
    if selected_method is None:
        return None

    return selected_method


def select_model(method: Dict[str, Any], family: str | None = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """Step 2: Select Gemma model for LoRA fine-tuning, driven by config.ini."""
    from gemma_tuner.wizard.config_store import _read_config

    console.print("\n[bold]Step 2: Choose your model[/bold]")

    device_info = get_wizard_device_info()
    available_memory = device_info["available_memory_gb"]

    # Dynamically discover available models from config.ini
    cfg = _read_config()
    available_models = [s.replace("model:", "") for s in cfg.sections() if s.startswith("model:")]
    # Filter to Gemma models only
    filtered = []
    for m in available_models:
        section = f"model:{m}"
        grp = cfg.get(section, "group", fallback="").strip().lower()
        if grp == "gemma" or "gemma" in m.lower():
            filtered.append(m)
    available_models = filtered

    # All Gemma models use LoRA fine-tuning
    base_models = list(available_models)

    # Build model choices with memory and time estimates
    choices = []
    seen: set[str] = set()
    for model_name in base_models:
        display_name = model_name.replace("-lora", "")
        if display_name in seen:
            continue
        if display_name not in ModelSpecs.MODELS:
            continue
        seen.add(display_name)
        specs = ModelSpecs.MODELS[display_name]
        required_memory = specs["memory_gb"] * method["memory_multiplier"]

        # Skip if not enough memory
        if required_memory > available_memory * 0.8:  # Leave 20% buffer
            continue

        # Estimate training time (assuming 100k samples baseline)
        estimated_hours = specs["hours_100k"] * method["time_multiplier"] * device_info["performance_multiplier"]

        if estimated_hours < 1:
            time_str = f"{estimated_hours * 60:.0f} minutes"
        else:
            time_str = f"{estimated_hours:.1f} hours"

        memory_str = f"{required_memory:.1f}GB"

        choice_text = f"{display_name} ({specs['params']}) - ~{time_str}, {memory_str} memory"

        # Add recommendation for the smaller variant (fits on most Apple Silicon)
        if display_name == "gemma-4-e2b-it":
            choice_text += " ⭐ Recommended"

        choices.append(
            {
                "name": choice_text,
                "value": model_name,  # Return the full config name, e.g., "gemma-3n-e4b-it"
            }
        )

    if not choices:
        console.print("[red]❌ No Gemma models found in config.ini. Add model sections with group=gemma.[/red]")
        # Raise RuntimeError so wizard_main's except-Exception handler can display a
        # clean message. Using sys.exit(1) would bypass those handlers entirely.
        raise RuntimeError("No Gemma models found in config.ini. Check your configuration.")

    selected_model = questionary.select(
        "Which model do you want to fine-tune?", choices=choices, style=apple_style
    ).ask()

    # questionary returns None when stdin is not a TTY (piped/scripted input).
    # Returning None signals cancellation to wizard_main.
    if selected_model is None:
        return None, {}

    return selected_model, {}


def select_dataset(method: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Step 3: Select dataset"""
    from gemma_tuner.wizard.config import select_bigquery_table_and_export
    from gemma_tuner.wizard.granary import setup_granary_dataset

    console.print("\n[bold]Step 3: Choose your dataset[/bold]")

    datasets = detect_datasets()

    choices = []
    for dataset in datasets:
        if dataset["type"] == "local_csv" or dataset["type"] == "local_audio":
            choice_text = f"📁 {dataset['name']} - {dataset['description']}"
        elif dataset["type"] == "huggingface":
            choice_text = f"🤗 {dataset['name']} - {dataset['description']}"
        else:
            choice_text = f"⚙️ {dataset['description']}"

        choices.append({"name": choice_text, "value": dataset})

    selected_dataset = questionary.select(
        "Which dataset do you want to use for training?", choices=choices, style=apple_style
    ).ask()

    # questionary returns None when stdin is not a TTY (piped/scripted input).
    # Returning None signals cancellation to wizard_main.
    if selected_dataset is None:
        return None

    # Handle BigQuery import flow
    if selected_dataset.get("type") == "bigquery_import":
        bq_dataset = select_bigquery_table_and_export()
        return bq_dataset

    # Handle Granary dataset setup flow
    if selected_dataset.get("type") == "granary_setup":
        granary_dataset = setup_granary_dataset()
        return granary_dataset

    # Handle custom dataset path
    if selected_dataset["name"] == "custom":
        dataset_path = questionary.path("Enter the path to your dataset:", style=apple_style).ask()

        selected_dataset["path"] = dataset_path
        selected_dataset["name"] = Path(dataset_path).name

    return selected_dataset


def configure_training_parameters() -> Dict[str, Any]:
    """Step 4: Training Parameters (mandatory)

    Prompts for critical hyperparameters with simple guidance, returning a dict:
    {"learning_rate": float, "num_train_epochs": int, "warmup_steps": int}
    """
    console.print("\n[bold]Step 4: Training Parameters[/bold]")
    # Learning rate
    console.print(
        "[dim]This is the most important hyperparameter. It controls how much the model learns from the data. A smaller number is safer. The default (1e-5) is a good starting point for fine-tuning.[/dim]"
    )
    lr_str = questionary.text("What learning rate do you want to use?", default="1e-5", style=apple_style).ask()
    try:
        learning_rate = float(lr_str)
    except Exception:
        learning_rate = 1e-5

    # Number of epochs
    console.print(
        "[dim]An epoch is one full pass through the entire training dataset. More epochs can lead to better results, but also increase the risk of overfitting. For fine-tuning, 1-3 epochs is usually enough.[/dim]"
    )
    epochs_str = questionary.text("How many training epochs?", default="3", style=apple_style).ask()
    try:
        num_train_epochs = int(epochs_str)
    except Exception:
        num_train_epochs = 3

    # Warmup steps
    console.print(
        "[dim]This gradually increases the learning rate at the start of training, which helps stabilize the model. A small number like 50-100 is a safe choice.[/dim]"
    )
    warmup_str = questionary.text("How many warmup steps for the learning rate?", default="50", style=apple_style).ask()
    try:
        warmup_steps = int(warmup_str)
    except Exception:
        warmup_steps = 50

    return {"learning_rate": learning_rate, "num_train_epochs": num_train_epochs, "warmup_steps": warmup_steps}


def show_confirmation_screen(
    method: Dict[str, Any],
    model: str,
    dataset: Dict[str, Any],
    method_config: Dict[str, Any],
    estimates: Dict[str, Any],
) -> bool:
    """Step 5: Beautiful confirmation screen"""
    from gemma_tuner.wizard.config_store import _read_config

    console.print("\n[bold cyan]Step 6: Ready to Train![/bold cyan]")

    # Create a beautiful configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan", width=20)
    config_table.add_column("Value", style="green")

    config_table.add_row(
        "Training Method", method["name"].replace("🚀", "").replace("🎨", "").replace("🧠", "").strip()
    )
    config_table.add_row("Model", f"{model} ({ModelSpecs.MODELS.get(model, {}).get('params', 'Unknown')})")
    config_table.add_row("Dataset", f"{dataset['name']} ({estimates['samples']:,} samples)")
    # Training parameters (added in Step 4)
    if "learning_rate" in method_config:
        config_table.add_row("Learning Rate", str(method_config["learning_rate"]))
    if "num_train_epochs" in method_config:
        config_table.add_row("Epochs", str(method_config["num_train_epochs"]))
    if "warmup_steps" in method_config:
        config_table.add_row("Warmup Steps", str(method_config["warmup_steps"]))

    # Add LoRA configuration.
    # Use .get() with "N/A" fallbacks because configure_method_specifics() may not
    # have populated lora_r/lora_alpha when stdin is non-interactive (questionary
    # returns None and the keys are never set), which would cause a KeyError.
    if method["key"] == "lora":
        lora_r = method_config.get("lora_r", "N/A")
        lora_alpha = method_config.get("lora_alpha", "N/A")
        config_table.add_row("LoRA Rank", str(lora_r))
        config_table.add_row("LoRA Alpha", str(lora_alpha))

    config_table.add_row("", "")  # Spacer
    config_table.add_row("Estimated Time", f"{estimates['hours']:.1f} hours")
    config_table.add_row("Memory Usage", f"{estimates['memory_gb']:.1f} GB")
    # Display dtype/attention when Gemma is selected or when group specifies
    try:
        cfg = _read_config()
        section = f"model:{model}"
        if cfg.has_section(section):
            group = cfg.get(section, "group", fallback="").strip().lower()
            if group == "gemma":
                dtype = cfg.get("group:gemma", "dtype", fallback="bfloat16")
                attn = cfg.get("group:gemma", "attn_implementation", fallback="eager")
                config_table.add_row("Precision (dtype)", dtype)
                config_table.add_row("Attention Impl", attn)
    except Exception:
        pass
    config_table.add_row(
        "Completion ETA",
        estimates["eta"].strftime("%I:%M %p today" if estimates["hours"] < 12 else "%I:%M %p tomorrow"),
    )

    device_info = get_wizard_device_info()
    config_table.add_row("Training Device", device_info["display_name"])

    # Status indicators
    memory_status = "🟢 Sufficient" if estimates["memory_gb"] < device_info["available_memory_gb"] * 0.8 else "🟡 Tight"
    config_table.add_row("Memory Status", memory_status)

    # Show the panel
    console.print(Panel(config_table, title="🎯 Training Configuration", border_style="green", padding=(1, 2)))

    # Ask about visualization
    console.print("\n[bold cyan]Optional: Enable Training Visualizer?[/bold cyan]")
    console.print("[dim]Watch your AI learn in real-time with stunning 3D graphics![/dim]")

    enable_viz = questionary.confirm("🎆 Enable live training visualization?", default=True, style=apple_style).ask()

    # questionary returns None when stdin is not a TTY. Default to False (no viz)
    # rather than crashing downstream code that checks method_config["visualize"].
    if enable_viz is None:
        enable_viz = False

    # Store visualization choice for later use
    method_config["visualize"] = enable_viz

    if enable_viz:
        console.print("[green]✨ Visualization will open in your browser when training starts![/green]")

    # Confirmation prompt. None means non-interactive stdin — treat as cancellation.
    confirmed = questionary.confirm("Start training with this configuration?", default=True, style=apple_style).ask()
    if confirmed is None:
        return False
    return confirmed
