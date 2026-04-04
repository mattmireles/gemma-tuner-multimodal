#!/usr/bin/env python3

"""Granary-specific setup flow for the interactive wizard."""

from __future__ import annotations

from typing import Any, Dict

import questionary
from rich.align import Align
from rich.panel import Panel

from gemma_tuner.wizard.base import apple_style, console
from gemma_tuner.wizard.config_store import _read_config, _write_config


def setup_granary_dataset() -> Dict[str, Any]:
    """
    Interactive NVIDIA Granary dataset setup with guided corpus download workflow.

    This function implements a comprehensive Granary setup workflow that guides users through
    the process of configuring one of the world's largest public speech datasets. It provides
    step-by-step instructions for downloading external audio corpora, configures the necessary
    audio source mappings, and generates the required configuration sections.

    Called by:
    - select_dataset() when user chooses "Setup NVIDIA Granary Dataset" option

    Granary Dataset Overview:
    The NVIDIA Granary dataset combines ~643k hours of transcribed audio across 25 languages
    from multiple large-scale speech corpora:
    - VoxPopuli: Multilingual parliamentary speeches
    - YouTube Commons (YTC): Diverse web content
    - LibriLight: Large-scale English audiobooks
    - YODAS: Custom corpus (included in HuggingFace download)

    Setup Workflow:
    1. Introduction and value proposition explanation
    2. Language subset selection for focused training
    3. External corpus download guidance with specific links
    4. Audio source path configuration and validation
    5. Configuration generation and integration
    6. Optional preparation script execution

    Returns:
        Dict[str, Any]: Dataset descriptor for integration with training pipeline
    """
    console.print("\n" + "=" * 60)
    console.print(Align.center("🚀 [bold]NVIDIA GRANARY DATASET SETUP[/bold] 🚀"))
    console.print("=" * 60)

    # Step 1: Introduction and value proposition
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to Granary Setup![/bold cyan]\n\n"
            "The NVIDIA Granary dataset is one of the world's largest public speech datasets with:\n"
            "• 📊 ~643,000 hours of transcribed audio\n"
            "• 🌍 25 languages for multilingual training\n"
            "• 🎯 High-quality transcriptions for robust ASR models\n"
            "• 📚 Multiple diverse corpora (parliamentary speeches, web content, audiobooks)\n\n"
            "[yellow]Note: Granary requires external audio corpus downloads (several TB total)[/yellow]",
            title="About Granary",
            border_style="cyan",
        )
    )

    proceed = questionary.confirm("Ready to set up Granary for your project?", default=True, style=apple_style).ask()

    if not proceed:
        return {"name": "custom", "type": "custom", "description": "Manual dataset path"}

    # Step 2: Language subset selection
    console.print("\n[bold]Step 1: Choose Language Subset[/bold]")
    language_options = [
        {"name": "🇺🇸 English (en) - Most common choice", "value": "en"},
        {"name": "🇪🇸 Spanish (es) - Large corpus available", "value": "es"},
        {"name": "🇫🇷 French (fr) - High-quality parliamentary data", "value": "fr"},
        {"name": "🇩🇪 German (de) - Rich multilingual content", "value": "de"},
        {"name": "🌍 Other language (specify manually)", "value": "custom"},
    ]

    language_choice = questionary.select(
        "Which language subset do you want to prepare?", choices=language_options, style=apple_style
    ).ask()

    if language_choice == "custom":
        language_code = questionary.text("Enter language code (e.g., 'it', 'pt', 'nl'):", style=apple_style).ask()
    else:
        language_code = language_choice

    # Step 3: Download guidance with specific links
    console.print("\n[bold]Step 2: Download Required Audio Corpora[/bold]")
    console.print(
        Panel.fit(
            "[bold red]IMPORTANT: External Downloads Required[/bold red]\n\n"
            "Granary requires you to download audio files from external sources:\n\n"
            "📥 [bold]Required Downloads:[/bold]\n"
            "1. VoxPopuli: https://github.com/facebookresearch/voxpopuli\n"
            "2. YouTube Commons: https://research.google.com/youtube-cc/\n"
            "3. LibriLight: https://github.com/facebookresearch/libri-light\n\n"
            "[yellow]Total size: Several terabytes - ensure you have adequate storage![/yellow]\n\n"
            "💡 [bold]Tips:[/bold]\n"
            "• Download to a fast SSD for best training performance\n"
            "• Consider downloading only the language subset you need\n"
            "• Ensure stable internet connection for large downloads",
            title="Download Instructions",
            border_style="red",
        )
    )

    downloads_complete = questionary.confirm(
        "Have you completed downloading all required audio corpora?", default=False, style=apple_style
    ).ask()

    if not downloads_complete:
        console.print("\n[yellow]💡 Come back and run this setup again after downloading the audio corpora.[/yellow]")
        console.print("For now, I'll create a template configuration you can complete later.")

    # Step 4: Audio source path configuration
    console.print("\n[bold]Step 3: Configure Audio Source Paths[/bold]")

    audio_sources = {}
    corpus_info = [
        ("voxpopuli", "VoxPopuli parliamentary speeches"),
        ("ytc", "YouTube Commons diverse content"),
        ("librilight", "LibriLight English audiobooks"),
    ]

    for corpus_key, corpus_desc in corpus_info:
        if downloads_complete:
            path = questionary.path(f"Path to {corpus_desc} audio directory:", style=apple_style).ask()
            audio_sources[corpus_key] = path
        else:
            # Template paths for user to fill in later
            audio_sources[corpus_key] = f"/path/to/downloaded/{corpus_key}/audio"

    # Step 4: Validation configuration
    console.print("\n[bold]Step 4: Configure Audio Validation[/bold]")
    console.print(
        Panel.fit(
            "[bold cyan]Audio Validation Trade-offs[/bold cyan]\n\n"
            "Granary contains ~643k hours of audio. Validating every file takes time but prevents training failures.\n\n"
            "🔍 [bold]Full Validation (Recommended):[/bold] Checks every audio file exists (slow but safe)\n"
            "🎯 [bold]Sample Validation:[/bold] Checks a percentage of files (faster, some risk)\n"
            "🚀 [bold]Skip Validation:[/bold] No file checking (fastest, highest risk)\n\n"
            "[yellow]Recommendation: Use full validation for production, sampling for development[/yellow]",
            title="Validation Options",
            border_style="cyan",
        )
    )

    validation_mode = questionary.select(
        "How thorough should audio file validation be?",
        choices=[
            "Full validation (slowest, safest)",
            "Sample validation (faster, some risk)",
            "Skip validation (fastest, risky)",
        ],
        style=apple_style,
    ).ask()

    # Configure validation settings based on choice
    skip_audio_validation = False
    sample_validation_rate = 1.0

    if validation_mode == "Skip validation (fastest, risky)":
        skip_audio_validation = True
        console.print("\n[yellow]⚠️  Audio files will NOT be verified. Training may fail if files are missing.[/yellow]")
    elif validation_mode == "Sample validation (faster, some risk)":
        sample_rate_choice = questionary.select(
            "What percentage of files should be validated?",
            choices=[
                "10% (quick sanity check)",
                "25% (reasonable confidence)",
                "50% (high confidence)",
                "Custom percentage",
            ],
            style=apple_style,
        ).ask()

        if sample_rate_choice == "10% (quick sanity check)":
            sample_validation_rate = 0.1
        elif sample_rate_choice == "25% (reasonable confidence)":
            sample_validation_rate = 0.25
        elif sample_rate_choice == "50% (high confidence)":
            sample_validation_rate = 0.5
        else:  # Custom percentage
            while True:
                try:
                    custom_rate = questionary.text("Enter validation percentage (1-100):", style=apple_style).ask()
                    rate_float = float(custom_rate) / 100.0
                    if 0.01 <= rate_float <= 1.0:
                        sample_validation_rate = rate_float
                        break
                    console.print("[red]Please enter a number between 1 and 100[/red]")
                except (ValueError, TypeError):
                    console.print("[red]Please enter a valid number[/red]")

        console.print(f"\n[cyan]✅ Will validate {sample_validation_rate:.1%} of audio files[/cyan]")
    else:  # Full validation
        console.print("\n[green]✅ Will validate all audio files (recommended for production)[/green]")

    # Step 5: Generate configuration
    console.print("\n[bold]Step 5: Generate Configuration[/bold]")

    dataset_name = f"granary-{language_code}"
    config_section = f"""
[dataset:{dataset_name}]
source_type = granary
hf_name = nvidia/Granary
hf_subset = {language_code}
local_path = data/datasets/{dataset_name}
text_column = text
train_split = train
validation_split = validation
audio_source_voxpopuli = {audio_sources["voxpopuli"]}
audio_source_ytc = {audio_sources["ytc"]}
audio_source_librilight = {audio_sources["librilight"]}
skip_audio_validation = {str(skip_audio_validation).lower()}
sample_validation_rate = {sample_validation_rate}
"""

    console.print(
        Panel.fit(
            f"[bold]Configuration to add to config.ini:[/bold]\n{config_section}",
            title="Generated Configuration",
            border_style="green",
        )
    )

    # Add configuration to config.ini
    add_config = questionary.confirm(
        "Add this configuration to your config.ini file?", default=True, style=apple_style
    ).ask()

    if add_config:
        try:
            config = _read_config()

            section_name = f"dataset:{dataset_name}"
            if not config.has_section(section_name):
                config.add_section(section_name)

            config.set(section_name, "source_type", "granary")
            config.set(section_name, "hf_name", "nvidia/Granary")
            config.set(section_name, "hf_subset", language_code)
            config.set(section_name, "local_path", f"data/datasets/{dataset_name}")
            config.set(section_name, "text_column", "text")
            config.set(section_name, "train_split", "train")
            config.set(section_name, "validation_split", "validation")
            config.set(section_name, "audio_source_voxpopuli", audio_sources["voxpopuli"])
            config.set(section_name, "audio_source_ytc", audio_sources["ytc"])
            config.set(section_name, "audio_source_librilight", audio_sources["librilight"])

            # Add validation configuration
            config.set(section_name, "skip_audio_validation", str(skip_audio_validation).lower())
            config.set(section_name, "sample_validation_rate", str(sample_validation_rate))
            _write_config(config)

            console.print("✅ [green]Configuration added to config.ini successfully![/green]")

        except Exception as e:
            console.print(f"❌ [red]Failed to update config.ini: {e}[/red]")
            console.print("Please add the configuration manually.")

    # Step 6: Optional preparation execution
    preparation_succeeded = False
    if downloads_complete and add_config:
        console.print("\n[bold]Step 5: Prepare Dataset[/bold]")
        run_preparation = questionary.confirm(
            f"Run dataset preparation now? (gemma-macos-tuner prepare-granary {dataset_name})",
            default=True,
            style=apple_style,
        ).ask()

        if run_preparation:
            console.print(f"\n🚀 [bold]Running Granary preparation for {dataset_name}...[/bold]")
            try:
                from gemma_tuner.scripts.prepare_granary import prepare_granary

                manifest_path = prepare_granary(dataset_name)
                console.print(f"✅ [green]Preparation completed! Manifest: {manifest_path}[/green]")

                return {
                    "name": dataset_name,
                    "type": "local_csv",
                    "path": f"data/datasets/{dataset_name}",
                    "files": 1,
                    "description": f"NVIDIA Granary {language_code} dataset (~643k hours)",
                    "prepared": True,
                }

            except Exception as e:
                console.print(f"❌ [red]Preparation failed: {e}[/red]")
                console.print("You can run preparation later with:")
                console.print(f"[cyan]gemma-macos-tuner prepare-granary {dataset_name}[/cyan]")

    # Return dataset descriptor for training pipeline integration
    console.print("\n🎉 [bold green]Granary setup completed![/bold green]")
    console.print("You can now use this dataset for training once preparation is complete.")

    return {
        "name": dataset_name,
        "type": "granary_configured",
        "path": f"data/datasets/{dataset_name}",
        "files": 0,  # Will be 1 after preparation
        "description": f"NVIDIA Granary {language_code} dataset (setup complete, preparation needed)",
        "language": language_code,
        "audio_sources": audio_sources,
        "prepared": preparation_succeeded,
    }
