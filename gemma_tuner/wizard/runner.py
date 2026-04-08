"""Training execution and main wizard workflow.

This module contains the functions that drive the training pipeline after
the user has finished configuring their run through the interactive wizard.

Functions:
- execute_training(): Bridges wizard config to the main.py training subprocess.
- wizard_main(): Top-level orchestrator that ties every wizard step together.

Called by:
- wizard/__init__.py re-exports wizard_main for backward-compatible imports.
- ``entrypoints/wizard.py`` shim for ``python entrypoints/wizard.py`` invocations.
"""

import configparser
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Rich for beautiful terminal UI
# --- Cross-module wizard imports ---------------------------------------------------
from gemma_tuner.wizard.base import SAMPLE_DATASET_NAME, console
from gemma_tuner.wizard.config import generate_profile_config
from gemma_tuner.wizard.config_store import (
    _CONFIG_INI,
    _read_config,
    ensure_bundled_sample_config_sections,
)
from gemma_tuner.wizard.estimator import configure_method_specifics, estimate_training_time
from gemma_tuner.wizard.ui import (
    configure_image_columns,
    configure_text_columns,
    configure_training_parameters,
    select_dataset,
    select_finetuning_kind,
    select_model,
    select_training_method,
    show_confirmation_screen,
    show_welcome_screen,
)

# ---------------------------------------------------------------------------
# bootstrap helpers
# ---------------------------------------------------------------------------


def _ensure_config_ini_exists() -> None:
    """Make sure ``config/config.ini`` is present and contains the sample sections.

    Two scenarios both need to "just work":

    1. **Fresh checkout** — the user has never run the wizard. ``config.ini`` does
       not exist (it is gitignored because the wizard writes local paths and GCP
       project IDs into it). Without bootstrapping, the model selection step would
       fail with "No Gemma models found in config.ini" before the user could do
       anything. Fix: copy ``config.ini.example`` over.

    2. **Pre-existing user config** — the user already has a ``config.ini`` from
       before the bundled sample existed. Their file has ``[model:*]`` sections
       but no ``[dataset:sample-text]`` / ``[profile:sample-text]``. If they pick
       the sample in the dataset selection step, ``generate_profile_config()``
       would crash because ``load_model_dataset_config()`` requires the dataset
       section. Fix: idempotently inject the two sections when the sample
       directory exists on disk but the sections are missing from config.

    Both repairs are no-ops when nothing needs to change.
    """
    if not _CONFIG_INI.exists():
        example_path = _CONFIG_INI.with_name("config.ini.example")
        if example_path.exists():
            _CONFIG_INI.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(example_path), str(_CONFIG_INI))
            console.print(
                f"[green]Created [bold]{_CONFIG_INI.relative_to(_CONFIG_INI.parent.parent)}[/bold] "
                f"from [bold]config.ini.example[/bold].[/green]"
            )
            console.print(
                "[dim]This file is gitignored — edit it freely. The wizard will also write "
                "to it (e.g. BigQuery project IDs).[/dim]\n"
            )
        # If even the example is missing the checkout is broken; let the
        # downstream "no models" error fire so the user gets a clear signal.

    _ensure_sample_sections_present()


def _ensure_sample_sections_present() -> None:
    """Add ``[dataset:sample-text]`` / ``[profile:sample-text]`` if the sample exists on disk.

    Delegates to :func:`ensure_bundled_sample_config_sections` (single source of truth:
    ``config.ini.example``).
    """
    if ensure_bundled_sample_config_sections(
        config_ini=_CONFIG_INI,
        sample_dataset_name=SAMPLE_DATASET_NAME,
    ):
        console.print(
            f"[dim]Added bundled [bold]{SAMPLE_DATASET_NAME}[/bold] dataset/profile to config.ini.[/dim]"
        )


# ---------------------------------------------------------------------------
# execute_training
# ---------------------------------------------------------------------------


def execute_training(profile_config: Dict[str, Any]):
    """
    Executes fine-tuning training using the established main.py infrastructure with wizard integration.

    This function serves as the bridge between the wizard's interactive configuration and
    the production training system. It generates temporary configuration files, executes
    training via subprocess isolation, and provides comprehensive progress feedback with
    graceful error handling and cleanup.

    Called by:
    - wizard_main() after user confirms training configuration
    - Interactive training workflows initiated through the wizard interface
    - Automated training pipelines using wizard-generated configurations

    Calls to:
    - main.py:main() via subprocess for isolated training execution
    - core/config.py configuration loading through main.py integration
    - All training infrastructure through main.py's operation dispatch system
    - Temporary file management for wizard-generated configurations

    Configuration generation workflow:
    1. Temporary config directory creation with timestamp-based isolation
    2. Main config.ini parsing to preserve model and dataset definitions
    3. Wizard profile generation with user-selected parameters
    4. Temporary config file creation with complete training configuration
    5. Subprocess training execution with isolated environment
    6. Progress monitoring and error handling with user feedback
    7. Cleanup of temporary configuration files

    Subprocess isolation benefits:
    - Prevents import side effects from affecting wizard UI
    - Enables clean resource management and memory cleanup
    - Provides process-level isolation for training experiments
    - Supports graceful interruption without wizard corruption
    - Maintains compatibility with package installation scenarios

    Error handling strategies:
    - subprocess.CalledProcessError: Training process failures with exit codes
    - KeyboardInterrupt: User cancellation with checkpoint preservation
    - FileNotFoundError: Configuration file access issues
    - Exception: General error recovery with diagnostic information

    Configuration inheritance:
    - Preserves all model and dataset definitions from main config.ini
    - Inherits DEFAULT section with global training parameters
    - Maintains group configurations and hierarchical settings
    - Applies wizard-specific overrides only where necessary

    Progress feedback:
    - Real-time training status through subprocess stdout/stderr
    - Checkpoint saving notifications for long-running training
    - Success/failure status with actionable user guidance
    - Visual progress indicators using Rich console formatting

    Args:
        profile_config (Dict[str, Any]): Complete training configuration generated by wizard:
            - model: Model identifier and architecture settings
            - dataset: Dataset name and processing parameters
            - method: Training method (standard, LoRA, distillation)
            - method_config: Method-specific parameters (ranks, temperatures, etc.)
            - device_settings: Hardware optimization parameters
            - visualization: Optional training visualization settings

    Side Effects:
        - Creates temp_configs/ directory if it doesn't exist
        - Generates temporary wizard_config_{timestamp}.ini file
        - Executes training subprocess with main.py integration
        - Cleans up temporary files after training completion/failure
        - Updates system with trained model outputs in configured output directory

    Example:
        # Wizard-generated configuration for LoRA fine-tuning
        profile_config = {
            "model": "gemma-3n-e2b-it",
            "dataset": "librispeech_subset",
            "use_peft": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "visualize": True
        }
        execute_training(profile_config)
        # Outputs: Trained LoRA adapter in output/wizard_20250813_143052/
    """
    console.print("\n[bold green]🚀 Starting training...[/bold green]")

    # Anchor temp_configs to the project root (three levels up from this file:
    # wizard/runner.py -> wizard/ -> gemma_tuner/ -> project root).
    # Using a relative Path("temp_configs") would create the directory relative to
    # the caller's CWD, which breaks when the wizard is invoked from any other directory.
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    config_dir = _PROJECT_ROOT / "temp_configs"
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_config_path = config_dir / f"wizard_config_{timestamp}.ini"

    # Configuration inheritance from main config.ini
    # Ensures all model definitions, dataset configurations, and global defaults
    # are available to the training process without duplication
    main_config = _read_config()

    # New configuration generation with selective section copying
    # Preserves essential configuration structure while adding wizard profile
    config = configparser.ConfigParser()

    # DEFAULT section preservation for global training parameters
    # ConfigParser treats DEFAULT as special - not returned by sections() but contains
    # critical parameters like num_train_epochs, learning_rate, logging_steps
    try:
        config["DEFAULT"] = dict(main_config["DEFAULT"])  # Always present in parsed config
    except KeyError:
        # Fallback minimal defaults if DEFAULT is missing (corrupt or empty config)
        config["DEFAULT"] = {
            "output_dir": "output",
            "logging_dir": "logs",
            "num_train_epochs": "3",
            # Safety net: conservative default LR for fine-tuning
            "learning_rate": "1e-5",
        }

    # Essential section copying for training infrastructure support
    # Includes model definitions, dataset configurations, and group settings
    for section in main_config.sections():
        if section.startswith(("model:", "dataset:", "group:", "dataset_defaults")):
            config[section] = dict(main_config[section])

    # Wizard profile section creation with complete user configuration
    # ConfigParser requires all values to be strings. Serialize non-string types:
    # - lists become comma-joined strings
    # - bools become lowercase "true"/"false" (matching INI conventions)
    # - everything else is str()-converted
    profile_name = f"wizard_{timestamp}"
    serialized = {}
    for k, v in profile_config.items():
        if v is None:
            continue  # omit optional keys rather than writing "None"
        elif isinstance(v, list):
            serialized[k] = ",".join(str(item) for item in v)
        elif isinstance(v, bool):
            serialized[k] = "true" if v else "false"
        else:
            serialized[k] = str(v)
    config[f"profile:{profile_name}"] = serialized

    # Temporary configuration file generation for subprocess execution.
    # Written with 0o600 (owner read/write only) so GCP project IDs and
    # hyperparameters are not visible to other users on a shared system.
    fd = os.open(str(temp_config_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        config.write(f)

    # User progress feedback with realistic time expectations
    console.print("[dim]Training started! This may take several hours...[/dim]")
    console.print("[dim]Press Ctrl+C to interrupt (training will be saved at checkpoints)[/dim]")

    keep_config = False
    try:
        # Subprocess training execution with environment isolation
        # Uses module invocation for package compatibility and clean resource management
        module_cwd = Path(__file__).resolve().parent.parent.parent
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gemma_tuner.main",  # Legacy Typer entry (repo root main.py was removed)
                "finetune",  # Training operation
                profile_name,  # Generated wizard profile
                "--config",  # Custom configuration file
                str(temp_config_path.resolve()),  # Absolute path for cross-platform compatibility
            ],
            check=True,
            text=True,
            capture_output=False,
            cwd=str(module_cwd),
        )

        # Success feedback with actionable next steps
        console.print("\n[bold green]✅ Training completed successfully![/bold green]")
        console.print("[green]Model saved in output directory[/green]")
        console.print(f"[dim]Check output/wizard_{timestamp}/ for your trained model[/dim]")

    except subprocess.CalledProcessError as e:
        # Training process failure with diagnostic guidance
        console.print(f"\n[red]❌ Training failed with exit code {e.returncode}[/red]")
        console.print("[red]Check the logs for detailed error information[/red]")
        console.print(f"[dim]Log file: output/wizard_{timestamp}/run.log[/dim]")
        rc = e.returncode if e.returncode else 1
        raise SystemExit(rc) from e

    except KeyboardInterrupt:
        # User interruption with checkpoint preservation confirmation
        keep_config = True
        console.print("\n[yellow]⚠️ Training interrupted by user[/yellow]")
        console.print("[yellow]Progress saved at latest checkpoint[/yellow]")
        resume = (
            f"python -m gemma_tuner.main finetune {profile_name} --config {temp_config_path}  "
            f"(or: gemma-macos-tuner finetune {profile_name} --config {temp_config_path})"
        )
        console.print(f"[dim]Resume with: {resume}[/dim]")
        raise SystemExit(130)

    except Exception as e:
        # General error recovery with troubleshooting guidance
        keep_config = True
        console.print(f"\n[red]❌ Training execution failed: {str(e)}[/red]")
        console.print("[red]Check your configuration and try again[/red]")
        console.print(f"[dim]Configuration saved at: {temp_config_path}[/dim]")
        raise SystemExit(1) from e

    finally:
        # Temporary file cleanup with error tolerance
        # Clean up temporary configuration to prevent accumulation
        # Preserve config file when training was interrupted or failed so user can resume
        if not keep_config:
            try:
                temp_config_path.unlink()
            except Exception:
                # Ignore cleanup errors - temporary files will be cleaned up eventually
                pass


# ---------------------------------------------------------------------------
# wizard_main  --  top-level orchestrator
# ---------------------------------------------------------------------------


def wizard_main():
    """
    Main wizard orchestration function implementing Steve Jobs-inspired progressive disclosure.

    This function implements the complete wizard workflow using progressive disclosure principles:
    show only what's relevant at each step, ask one question at a time, and provide intelligent
    defaults for everything. The design creates a smooth, Apple-like experience that guides
    users from zero configuration to production-ready training.

    Called by:
    - Direct script execution: python entrypoints/wizard.py (project-root shim)
    - wizard/__init__.py re-export for ``from gemma_tuner.wizard import wizard_main``
    - Package entry points and command-line tools
    - Interactive training workflows and development environments
    - Automated training setup and configuration generation

    Calls to (complete wizard workflow):
    - show_welcome_screen() for elegant introduction and system status
    - select_training_method() for method selection with smart recommendations
    - select_model() for model selection with memory/performance constraints
    - select_dataset() for dataset selection with BigQuery integration
    - configure_training_parameters() for mandatory hyperparameters
    - configure_method_specifics() for advanced configuration with progressive disclosure
    - estimate_training_time() for realistic resource planning and expectations
    - show_confirmation_screen() for final configuration review and approval
    - generate_profile_config() for production configuration generation
    - execute_training() for single-machine training execution

    Progressive disclosure workflow:

    Step 0 (Welcome Screen):
    - System capability detection and hardware profiling
    - Visual welcome with Apple-inspired design language
    - Confidence building through status verification
    - Sets expectations for the complete workflow

    Step 1 (Training Method Selection):
    - LoRA fine-tuning for Gemma (only method exposed in the current UI)
    - Quality vs efficiency trade-offs explained simply
    - Smart defaults highlighted with recommendation badges
    - Technical complexity hidden until needed

    Step 2 (Model Selection):
    - Dynamic model filtering based on hardware constraints
    - Memory and time estimates for realistic expectations
    - Performance recommendations based on use case
    - Automatic incompatibility filtering for user safety

    Step 3 (Dataset Selection):
    - Automatic local dataset discovery with file counting
    - BigQuery import and Granary setup options
    - Custom dataset path support for advanced users

    Step 4 (Training Parameters):
    - Learning rate, epochs, warmup steps
    - Clear explanations for each hyperparameter
    - Safe defaults for beginners

    Step 5 (Method-Specific Configuration):
    - LoRA rank and alpha with smart defaults (Gemma LoRA path)

    Step 6 (Resource Estimation):
    - Realistic time and memory requirements
    - Hardware-specific performance calculations
    - Training completion estimates with ETA
    - Safety checks for resource constraints

    Step 7 (Confirmation & Execution):
    - Beautiful configuration summary table
    - Final approval with ability to cancel safely
    - Training execution with progress feedback
    - Success/failure handling with actionable guidance

    Error handling philosophy:
    - Graceful degradation: Never crash, always provide alternatives
    - User empowerment: Clear error messages with troubleshooting guidance
    - State preservation: Configuration saved even on failures
    - Recovery options: Instructions for manual continuation

    Exception handling:
    - KeyboardInterrupt: Clean cancellation without system changes
    - Configuration errors: Diagnostic information with recovery steps
    - Training failures: Checkpoint preservation and troubleshooting guidance
    - System errors: Comprehensive error reporting for issue resolution

    Side effects:
    - May create temporary configuration files (cleaned up automatically)
    - Updates config.ini with new dataset definitions (BigQuery imports)
    - Creates training output directories and model checkpoints
    - Generates comprehensive training logs and metrics

    Example workflow:
        $ python entrypoints/wizard.py

        Welcome Screen: "Ready for training ✅"
        Method Selection: "🎨 LoRA Fine-Tune"
        Model Selection: "gemma-3n-e2b-it (~2B) - … ⭐ Recommended"
        Dataset Selection: "📁 my_dataset - Local dataset with 3 CSV files"
        Configuration: [Smart defaults applied automatically]
        Confirmation: "Start training with this configuration? Yes"
        Training: [Progress monitoring with checkpoints]
        Completion: "✅ Training completed successfully! Model saved in output/"
    """
    try:
        # Bootstrap config.ini from the committed example on first run.
        # Must run before show_welcome_screen / select_model so the rest of the
        # wizard can read [model:*] and [dataset:*] sections without crashing.
        _ensure_config_ini_exists()

        # Step 0: Elegant introduction with system profiling
        # Creates confidence through hardware verification and beautiful design
        show_welcome_screen()

        # Step 1: Task kind (audio STT vs text) — gates dataset sources and column prompts
        finetuning = select_finetuning_kind()
        if finetuning is None:
            raise KeyboardInterrupt

        # Step 2: Training method selection (LoRA for Gemma)
        family = "gemma"
        method = select_training_method(family)
        # select_training_method returns None when questionary can't prompt (non-TTY stdin).
        if method is None:
            raise KeyboardInterrupt

        # Step 3: Model selection with intelligent constraints
        # Returns (model_key, seed_dict) tuple for configuration flexibility
        model, seed = select_model(method, family)
        # select_model returns (None, {}) when questionary can't prompt (non-TTY stdin).
        if model is None:
            raise KeyboardInterrupt

        # Step 4: Dataset selection with automatic discovery
        # Supports local files, BigQuery imports, and HuggingFace datasets
        dataset = select_dataset(method, finetuning)
        # select_dataset returns None when questionary can't prompt (non-TTY stdin).
        if dataset is None:
            raise KeyboardInterrupt

        # Step 5: Training parameters (mandatory hyperparameters)
        training_params = configure_training_parameters()

        # Step 6: Method-specific configuration with smart defaults
        # Reveals advanced options only when needed, passes seed for custom hybrids
        method_config = configure_method_specifics(method, model, seed)
        # Merge training parameters into method_config for downstream display and merging
        method_config.update(training_params)
        method_config.update(configure_text_columns(finetuning))
        method_config.update(configure_image_columns(finetuning, model))
        method_config["modality"] = finetuning["modality"]
        method_config["text_sub_mode"] = finetuning.get("text_sub_mode", "instruction")
        method_config["image_sub_mode"] = finetuning.get("image_sub_mode", "caption")

        # Step 7: Resource estimation with realistic expectations
        # Calculates training time and memory requirements based on hardware
        estimates = estimate_training_time(method, model, dataset, finetuning=finetuning)

        # Step 8: Beautiful confirmation screen with final approval
        # Comprehensive configuration review before committing to training
        if show_confirmation_screen(method, model, dataset, method_config, estimates, finetuning=finetuning):
            # Configuration generation for production training infrastructure
            profile_config = generate_profile_config(method, model, dataset, method_config)

            # Step 8: Start training
            # Steve Jobs magic: clean, obvious next step with single-machine defaults
            execute_training(profile_config)

        else:
            # Graceful cancellation with guidance for future use
            console.print("\n[yellow]Training cancelled by user.[/yellow]")
            console.print("[dim]Run the wizard again: python entrypoints/wizard.py  or  gemma-macos-tuner wizard[/dim]")

    except KeyboardInterrupt:
        # Clean interruption handling with system state preservation
        console.print("\n\n[yellow]Wizard interrupted by user.[/yellow]")
        console.print(
            "[dim]Interactive profile choices were not saved. If you already ran BigQuery import "
            "or Granary setup, config.ini may have been updated in those steps.[/dim]"
        )
        raise SystemExit(130)

    except Exception as e:
        console.print(f"\n[red]❌ Wizard error: {str(e)}[/red]")
        console.print("[red]Please report this issue or try manual configuration.[/red]")
        console.print("[dim]For manual setup: gemma-macos-tuner --help[/dim]")
        raise SystemExit(1) from e
