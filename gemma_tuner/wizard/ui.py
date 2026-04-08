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

from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import questionary
from packaging.version import Version
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gemma_tuner.models.gemma.family import (
    MIN_TRANSFORMERS_GEMMA4,
    GemmaFamily,
    detect_family,
    gate_gemma_model,
)
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
    - wizard_main() as the opening experience
    - Interactive training workflows requiring system status validation
    - Development and demonstration environments for visual appeal

    Calls to:
    - get_wizard_device_info() for comprehensive system capability detection
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


def select_finetuning_kind() -> Optional[Dict[str, Any]]:
    """Step 1: Audio vs text task — gates later prompts (CSV columns, adapters)."""
    console.print("\n[bold]Step 1: What kind of fine-tuning?[/bold]")
    choices = [
        {
            "name": "Instruction tuning (text — prompt + response columns)",
            "value": {"modality": "text", "text_sub_mode": "instruction"},
        },
        {
            "name": "Speech-to-text (audio + transcript)",
            "value": {"modality": "audio", "text_sub_mode": "instruction"},
        },
        {
            "name": "Continued pretraining / style (text — single text column)",
            "value": {"modality": "text", "text_sub_mode": "completion"},
        },
        {
            "name": "Image captioning / OCR (image + caption)",
            "value": {"modality": "image", "image_sub_mode": "caption", "text_sub_mode": "instruction"},
        },
        {
            "name": "Visual question answering (image + question + answer)",
            "value": {"modality": "image", "image_sub_mode": "vqa", "text_sub_mode": "instruction"},
        },
    ]
    selected = questionary.select("Choose your task:", choices=choices, style=apple_style).ask()
    return selected


def select_training_method(family: str | None = None) -> Optional[Dict[str, Any]]:
    """Step 2: Select training method with progressive disclosure"""
    console.print("\n[bold]Step 2: Choose your training method[/bold]")

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
    """Step 3: Select Gemma model for LoRA fine-tuning, driven by config.ini."""
    from gemma_tuner.wizard.config_store import _read_config

    console.print("\n[bold]Step 3: Choose your model[/bold]")

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

    # Group by family and size so the wizard list reads top-to-bottom as
    # 3n-e2b → 3n-e4b → 4-e2b → 4-e4b, with base before instruct within each
    # size. Without this, ordering depends on whatever order the user (or the
    # wizard) happened to write [model:*] sections into config.ini, which is
    # how the 3n e2b/e4b pair ended up split across the list by Gemma 4 entries.
    def _model_sort_key(name: str) -> Tuple[int, int, int, str]:
        # Family: 3n before 4 (3n-e2b-it is also flagged ⭐ Recommended below).
        if "-3n-" in name:
            family_rank = 0
        elif "-4-" in name:
            family_rank = 1
        else:
            family_rank = 2
        # Size: e2b before e4b.
        if "-e2b" in name:
            size_rank = 0
        elif "-e4b" in name:
            size_rank = 1
        else:
            size_rank = 2
        # Variant: base ("") before instruct ("-it"), matching the
        # "primary targets" / "alternative starting point" comments in
        # config/config.ini.example.
        variant_rank = 1 if name.endswith("-it") else 0
        return (family_rank, size_rank, variant_rank, name)

    # All Gemma models use LoRA fine-tuning
    base_models = sorted(available_models, key=_model_sort_key)

    # Pick the ⭐ Recommended model based on what the current env can actually run.
    # We prefer gemma-4-e2b-it (newer, generally better quality at ~2B) but fall back
    # to gemma-3n-e2b-it on the default `pip install -e .` pin so a fresh user does
    # not get a "recommended" option that crashes in gate_gemma_model() because
    # transformers < 5.5.0 (the min for Gemma 4).
    try:
        _tf_ver = Version(metadata.version("transformers"))
    except metadata.PackageNotFoundError:
        _tf_ver = Version("0")
    _gemma4_ok = _tf_ver >= Version(MIN_TRANSFORMERS_GEMMA4)
    recommended_model = "gemma-4-e2b-it" if _gemma4_ok else "gemma-3n-e2b-it"

    # Build model choices with memory and time estimates.
    #
    # Track which choices correspond to Gemma 4 entries so the gate-fallback
    # loop below can filter them out once the user declines the install prompt —
    # without this, re-picking the same Gemma 4 model would re-prompt forever.
    choices = []
    gemma4_model_names: set[str] = set()
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

        # Distinguish base vs instruction-tuned variants visibly. The "-it" suffix
        # alone is easy to miss at the end of a long line, which made base/instruct
        # pairs (e.g. gemma-4-e2b vs gemma-4-e2b-it) look like duplicates because
        # they have identical memory_gb and hours_100k in ModelSpecs.
        variant_tag = "instruct" if display_name.endswith("-it") else "base"

        choice_text = f"{display_name} ({specs['params']}, {variant_tag}) - ~{time_str}, {memory_str} memory"

        try:
            tf_ver = Version(metadata.version("transformers"))
        except metadata.PackageNotFoundError:
            tf_ver = Version("0")
        hf_id = str(specs.get("hf_id") or "")
        is_gemma4_entry = False
        try:
            if detect_family(hf_id) == GemmaFamily.GEMMA_4:
                is_gemma4_entry = True
                if tf_ver < Version(MIN_TRANSFORMERS_GEMMA4):
                    choice_text += " (requires Gemma 4 install)"
        except RuntimeError:
            pass
        if is_gemma4_entry:
            gemma4_model_names.add(model_name)

        # Mark whichever model is recommended for the current env (see
        # recommended_model derivation above).
        if display_name == recommended_model:
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

    # Loop on the model picker so that if the user lands on a Gemma 4 entry in
    # an env that can't run Gemma 4 yet, we can offer to install the stack
    # (offer_gemma4_install re-execs the wizard on success) and on decline send
    # them back to the picker instead of crashing wizard_main with a stack trace.
    from gemma_tuner.wizard.runner import offer_gemma4_install

    declined_gemma4_install = False
    while True:
        # After the user has declined the Gemma 4 install once in this wizard run,
        # hide Gemma 4 entries from the picker so they cannot re-trigger the same
        # prompt by picking the same model again.
        active_choices = (
            [c for c in choices if c["value"] not in gemma4_model_names] if declined_gemma4_install else choices
        )
        if not active_choices:
            console.print(
                "[red]No Gemma models available without the Gemma 4 stack. "
                "Install it manually or re-run the wizard to retry the preflight.[/red]"
            )
            return None, {}

        selected_model = questionary.select(
            "Which model do you want to fine-tune?", choices=active_choices, style=apple_style
        ).ask()

        # questionary returns None when stdin is not a TTY (piped/scripted input).
        # Returning None signals cancellation to wizard_main.
        if selected_model is None:
            return None, {}

        section = f"model:{selected_model}"
        base_model_id = cfg.get(section, "base_model", fallback="").strip()
        if not base_model_id:
            raise RuntimeError(
                f"Missing base_model in [{section}] in config.ini. Set base_model to the Hugging Face model id."
            )
        # Fast-fail before training; finetune.main() runs the same gate again
        # (no duplicate user prompts). On Gemma 4 family + too-old transformers
        # this raises RuntimeError; we intercept that case and offer to fix it.
        try:
            gate_gemma_model(base_model_id, entrypoint="finetune")
        except RuntimeError as exc:
            msg = str(exc)
            try:
                family = detect_family(base_model_id)
            except RuntimeError:
                family = None
            if family == GemmaFamily.GEMMA_4 and "transformers" in msg.lower():
                console.print(f"\n[yellow]{selected_model}[/yellow] needs the Gemma 4 transformers stack.")
                # On success the wizard process is replaced via os.execv and
                # never returns here. On decline/failure the function returns
                # normally and we filter the Gemma 4 entries out of the picker
                # so we do not loop through the same prompt again.
                offer_gemma4_install(context=f"You picked [bold]{selected_model}[/bold] but Gemma 4 is not installed.")
                declined_gemma4_install = True
                console.print("[dim]Pick a Gemma 3n model instead, or cancel with Ctrl+C.[/dim]\n")
                continue  # back to the model picker
            raise

        return selected_model, {}


def select_dataset(method: Dict[str, Any], finetuning: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Step 4: Select dataset (text mode v1: local CSV only — no BigQuery/Granary)."""
    from gemma_tuner.wizard.config import select_bigquery_table_and_export
    from gemma_tuner.wizard.granary import setup_granary_dataset

    console.print("\n[bold]Step 4: Choose your dataset[/bold]")

    finetuning = finetuning or {}
    modality = str(finetuning.get("modality", "audio")).lower()

    datasets = detect_datasets()

    # Surface the bundled text sample at the top of the local-dataset list so
    # first-time users see "try this first" before scrolling. The sample only has
    # prompt/response columns (no audio_path, no image_path), so we only promote
    # it — and only show it at all — when the user picked text instruction tuning
    # in Step 1. Showing it for audio or image would lead to a guaranteed runtime
    # error during training.
    sample_promoted: list[Dict[str, Any]] = []
    other_local: list[Dict[str, Any]] = []
    other_virtual: list[Dict[str, Any]] = []
    text_sub_mode = str(finetuning.get("text_sub_mode", "instruction")).lower()
    sample_compatible = modality == "text" and text_sub_mode == "instruction"
    for dataset in datasets:
        if dataset.get("is_sample"):
            if sample_compatible:
                sample_promoted.append(dataset)
            # Skip the sample entirely for non-text modalities — it has no
            # audio_path / image_path column and would crash training.
            continue
        if dataset.get("type") in ("local_csv", "local_audio"):
            other_local.append(dataset)
        else:
            other_virtual.append(dataset)
    datasets = sample_promoted + other_local + other_virtual

    choices = []
    for dataset in datasets:
        if modality in ("text", "image") and dataset.get("type") in ("bigquery_import", "granary_setup"):
            continue
        if dataset.get("is_sample"):
            choice_text = f"✨ {dataset['name']} — {dataset['description']}"
        elif dataset["type"] == "local_csv" or dataset["type"] == "local_audio":
            choice_text = f"📁 {dataset['name']} - {dataset['description']}"
        else:
            choice_text = f"⚙️ {dataset['description']}"

        choices.append({"name": choice_text, "value": dataset})

    if not choices:
        console.print(
            "[red]No local datasets available for this task. "
            "For text or image fine-tuning, add CSV splits under data/datasets/<name>/.[/red]"
        )
        return None

    selected_dataset = questionary.select(
        "Which dataset do you want to use for training?", choices=choices, style=apple_style
    ).ask()

    # questionary returns None when stdin is not a TTY (piped/scripted input).
    # Returning None signals cancellation to wizard_main.
    if selected_dataset is None:
        return None

    # Handle BigQuery import flow
    if selected_dataset.get("type") == "bigquery_import":
        if modality in ("text", "image"):
            console.print("[red]Text/image fine-tuning (v1) needs a local CSV dataset, not BigQuery import.[/red]")
            return None
        bq_dataset = select_bigquery_table_and_export()
        return bq_dataset

    # Handle Granary dataset setup flow
    if selected_dataset.get("type") == "granary_setup":
        if modality in ("text", "image"):
            console.print("[red]Text/image fine-tuning (v1) needs a local CSV dataset, not Granary.[/red]")
            return None
        granary_dataset = setup_granary_dataset()
        return granary_dataset

    # Handle custom dataset path
    if selected_dataset["name"] == "custom":
        dataset_path = questionary.path("Enter the path to your dataset:", style=apple_style).ask()

        selected_dataset["path"] = dataset_path
        selected_dataset["name"] = Path(dataset_path).name

    return selected_dataset


def configure_text_columns(finetuning: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prompt for CSV column names when training on text-only data."""
    if not finetuning or finetuning.get("modality") != "text":
        return {}
    console.print("\n[bold]Text dataset columns[/bold]")
    console.print(
        "[dim]These must match column headers in your train/validation CSV files under data/datasets/<name>/.[/dim]"
    )
    sub = str(finetuning.get("text_sub_mode", "instruction")).lower()
    out: Dict[str, Any] = {}
    if sub == "instruction":
        tc = questionary.text("Response / target text column name:", default="response", style=apple_style).ask()
        out["text_column"] = (tc or "response").strip()
        pc = questionary.text("Prompt / instruction column name:", default="prompt", style=apple_style).ask()
        out["prompt_column"] = (pc or "prompt").strip()
    else:
        tc = questionary.text("Text column name (full sequence to train on):", default="text", style=apple_style).ask()
        out["text_column"] = (tc or "text").strip()
    return out


def _validate_image_token_budget_against_processor(model_id: str, budget: int) -> None:
    """Best-effort: compare chosen budget to ``AutoProcessor`` defaults (cached by Hugging Face)."""
    try:
        from transformers import AutoProcessor
    except ImportError:
        return
    try:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        console.print(f"[dim]Could not load AutoProcessor for {model_id!r} to validate image_token_budget ({e}).[/dim]")
        return
    if not hasattr(proc, "image_seq_length"):
        return
    isl = int(getattr(proc, "image_seq_length", 0) or 0)
    if isl <= 0:
        return
    b = int(budget)
    if b != isl:
        console.print(
            f"[yellow]Processor default image_seq_length is {isl}; your image_token_budget is {b}. "
            f"Training will align the processor to {b} — use the same value when serving.[/yellow]"
        )
    if b > isl * 4:
        console.print(
            f"[yellow]Warning: image_token_budget ({b}) is much larger than the processor default ({isl}); "
            f"expect high memory use.[/yellow]"
        )


def configure_image_columns(finetuning: Optional[Dict[str, Any]], model: Optional[str] = None) -> Dict[str, Any]:
    """Prompt for CSV column names and vision budget when training on image+text data."""
    if not finetuning or finetuning.get("modality") != "image":
        return {}
    console.print("\n[bold]Image dataset columns[/bold]")
    console.print(
        "[dim]Paths may be relative to data/datasets/<dataset>/; images are converted to RGB in the collator.[/dim]"
    )
    out: Dict[str, Any] = {}
    ipc = questionary.text("Image file path column name:", default="image_path", style=apple_style).ask()
    out["image_path_column"] = (ipc or "image_path").strip()
    sub = str(finetuning.get("image_sub_mode", "caption")).lower()
    if sub == "caption":
        tc = questionary.text("Caption / response text column:", default="caption", style=apple_style).ask()
        out["text_column"] = (tc or "caption").strip()
    else:
        qc = questionary.text("Question column name:", default="question", style=apple_style).ask()
        out["prompt_column"] = (qc or "question").strip()
        ac = questionary.text("Answer column name:", default="answer", style=apple_style).ask()
        out["text_column"] = (ac or "answer").strip()
    budget_choices = [
        {"name": "70 — smallest budget (fastest; good for smoke tests)", "value": 70},
        {"name": "140", "value": 140},
        {"name": "280 — default for captioning", "value": 280},
        {"name": "560 — VQA / more visual detail", "value": 560},
        {"name": "1120 — OCR / document parsing (heavy memory)", "value": 1120},
    ]
    itb = questionary.select(
        "Vision token budget (image_token_budget):",
        choices=budget_choices,
        style=apple_style,
    ).ask()
    if itb is None:
        itb = 280
    out["image_token_budget"] = int(itb)
    if model:
        _validate_image_token_budget_against_processor(model, out["image_token_budget"])
    return out


def configure_training_parameters() -> Dict[str, Any]:
    """Step 5: Training Parameters (mandatory)

    Prompts for critical hyperparameters with simple guidance, returning a dict:
    {"learning_rate": float, "num_train_epochs": int, "warmup_steps": int}
    """
    console.print("\n[bold]Step 5: Training Parameters[/bold]")
    # Learning rate
    #
    # Default is 2e-4 because this wizard only drives LoRA fine-tuning, where the
    # standard learning rate range is 1e-4 to 5e-4 — you are updating a tiny adapter,
    # not the base model weights, so the "a smaller number is safer" rule of thumb
    # from full fine-tuning does not apply. 2e-4 matches the built-in
    # [profile:sample-text] in config/config.ini.example so the happy-path smoke run
    # actually moves loss on the bundled 16-sample dataset. 1e-5 (the old default)
    # was borrowed from full-finetune lore and resulted in near-zero updates for
    # LoRA — the model learned nothing in typical short runs.
    console.print(
        "[dim]This is the most important hyperparameter. It controls how much the adapter learns from the data. "
        "For LoRA, 1e-4 to 5e-4 is the standard range; the default (2e-4) is a good starting point. "
        "Lower it toward 1e-4 if you see loss spikes; raise it toward 5e-4 for very small datasets.[/dim]"
    )
    lr_str = questionary.text("What learning rate do you want to use?", default="2e-4", style=apple_style).ask()
    try:
        learning_rate = float(lr_str)
    except Exception:
        learning_rate = 2e-4

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
    finetuning: Optional[Dict[str, Any]] = None,
) -> bool:
    """Step 7: Confirmation screen — shows full training configuration for final approval."""
    from gemma_tuner.wizard.config_store import _read_config

    console.print("\n[bold cyan]Step 7: Ready to Train![/bold cyan]")

    # Create a beautiful configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan", width=20)
    config_table.add_column("Value", style="green")

    config_table.add_row(
        "Training Method", method["name"].replace("🚀", "").replace("🎨", "").replace("🧠", "").strip()
    )
    ft = finetuning or {}
    if ft.get("modality") == "text":
        config_table.add_row("Task", f"text / {ft.get('text_sub_mode', 'instruction')}")
        if method_config.get("text_column"):
            config_table.add_row("Text column", str(method_config["text_column"]))
        if ft.get("text_sub_mode") == "instruction" and method_config.get("prompt_column"):
            config_table.add_row("Prompt column", str(method_config["prompt_column"]))
    elif ft.get("modality") == "image":
        config_table.add_row("Task", f"image / {ft.get('image_sub_mode', 'caption')}")
        if method_config.get("image_path_column"):
            config_table.add_row("Image path column", str(method_config["image_path_column"]))
        if method_config.get("text_column"):
            config_table.add_row("Text column", str(method_config["text_column"]))
        if ft.get("image_sub_mode") == "vqa" and method_config.get("prompt_column"):
            config_table.add_row("Question column", str(method_config["prompt_column"]))
        if method_config.get("image_token_budget") is not None:
            config_table.add_row("image_token_budget", str(method_config["image_token_budget"]))
    else:
        config_table.add_row("Task", "audio (speech-to-text)")
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
        from gemma_tuner.wizard.runner import ensure_viz_dependencies_installed

        ensure_viz_dependencies_installed(method_config)

    if method_config.get("visualize"):
        console.print("[green]✨ Visualization will open in your browser when training starts![/green]")

    # Confirmation prompt. None means non-interactive stdin — treat as cancellation.
    confirmed = questionary.confirm("Start training with this configuration?", default=True, style=apple_style).ask()
    if confirmed is None:
        return False
    return confirmed
