#!/usr/bin/env python3

"""
Whisper Fine-Tuning Wizard - Interactive CLI for Apple Silicon

A Steve Jobs-inspired command-line interface that guides users through the entire
Whisper fine-tuning process with progressive disclosure. Simple for beginners,
powerful for experts.

Design principles:
- Ask one question at a time
- Show only what's relevant
- Smart defaults for everything
- Beautiful visual feedback
- Zero configuration required

Called by:
- manage.py finetune-wizard command
- Direct execution: python wizard.py

Integrates with:
- main.py: Executes training using existing infrastructure
- config.ini: Can generate profile configs on the fly
- All existing model types: whisper, distil-whisper, LoRA variants
"""

import os
import sys
import configparser
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple


# Rich for beautiful terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import track
from rich.align import Align
from rich import print as rprint

# Questionary for interactive prompts
import questionary
from questionary import Style

# Import existing utilities
from utils.device import get_device

# Initialize console and styling
console = Console()

# Custom style for questionary prompts (Apple-inspired)
apple_style = Style([
    ('qmark', 'fg:#ff9500 bold'),          # Orange question mark (Apple orange)
    ('question', 'bold'),                   # Bold question text
    ('answer', 'fg:#007aff bold'),         # Blue answers (Apple blue)
    ('pointer', 'fg:#ff9500 bold'),        # Orange pointer
    ('highlighted', 'fg:#007aff bold'),    # Blue highlight
    ('selected', 'fg:#34c759 bold'),       # Green selected (Apple green)
    ('instruction', 'fg:#8e8e93'),         # Gray instructions
    ('text', ''),                          # Default text
])

class TrainingMethod:
    """Training method configurations with smart defaults"""
    
    STANDARD = {
        "key": "standard",
        "name": "🚀 Standard Fine-Tune (SFT)",
        "description": "Full model fine-tuning for best accuracy",
        "memory_multiplier": 1.0,
        "time_multiplier": 1.0,
        "quality": "highest"
    }
    
    LORA = {
        "key": "lora", 
        "name": "🎨 LoRA Fine-Tune",
        "description": "Memory-efficient parameter-efficient fine-tuning",
        "memory_multiplier": 0.4,
        "time_multiplier": 0.8,
        "quality": "high"
    }
    
    DISTILLATION = {
        "key": "distillation",
        "name": "🧠 Knowledge Distillation", 
        "description": "Train smaller models from larger teacher models",
        "memory_multiplier": 1.2,
        "time_multiplier": 1.5,
        "quality": "good"
    }

class ModelSpecs:
    """Model specifications for estimation calculations"""
    
    MODELS = {
        # OpenAI Standard Models
        "whisper-tiny": {"params": "39M", "memory_gb": 1.2, "hours_100k": 0.5, "hf_id": "openai/whisper-tiny"},
        "whisper-base": {"params": "74M", "memory_gb": 2.1, "hours_100k": 1.0, "hf_id": "openai/whisper-base"}, 
        "whisper-small": {"params": "244M", "memory_gb": 4.2, "hours_100k": 2.5, "hf_id": "openai/whisper-small"},
        "whisper-medium": {"params": "769M", "memory_gb": 8.4, "hours_100k": 6.0, "hf_id": "openai/whisper-medium"},
        "whisper-large-v2": {"params": "1550M", "memory_gb": 16.8, "hours_100k": 12.0, "hf_id": "openai/whisper-large-v2"},
        "whisper-large-v3": {"params": "1550M", "memory_gb": 16.8, "hours_100k": 12.0, "hf_id": "openai/whisper-large-v3"},
        
        # OpenAI English-Only Models
        "whisper-tiny.en": {"params": "39M", "memory_gb": 1.2, "hours_100k": 0.45, "hf_id": "openai/whisper-tiny.en"},
        "whisper-base.en": {"params": "74M", "memory_gb": 2.1, "hours_100k": 0.9, "hf_id": "openai/whisper-base.en"},
        "whisper-small.en": {"params": "244M", "memory_gb": 4.2, "hours_100k": 2.3, "hf_id": "openai/whisper-small.en"},
        "whisper-medium.en": {"params": "769M", "memory_gb": 8.4, "hours_100k": 5.5, "hf_id": "openai/whisper-medium.en"},

        # Distil-Whisper Models (Pre-trained by HuggingFace)
        "distil-small.en": {"params": "166M", "memory_gb": 3.2, "hours_100k": 1.8, "hf_id": "distil-whisper/distil-small.en"},
        "distil-medium.en": {"params": "394M", "memory_gb": 6.1, "hours_100k": 3.5, "hf_id": "distil-whisper/distil-medium.en"},
        "distil-large-v2": {"params": "756M", "memory_gb": 12.4, "hours_100k": 8.0, "hf_id": "distil-whisper/distil-large-v2"},
        "distil-large-v3": {"params": "756M", "memory_gb": 12.4, "hours_100k": 7.5, "hf_id": "distil-whisper/distil-large-v3"},
        
        # Custom Distillation Targets (Create your own distilled models)
        "distil-tiny-from-medium": {"params": "39M", "memory_gb": 2.8, "hours_100k": 1.2, "hf_id": "openai/whisper-tiny"},
        "distil-base-from-medium": {"params": "74M", "memory_gb": 4.2, "hours_100k": 2.0, "hf_id": "openai/whisper-base"},
        "distil-tiny.en-from-medium.en": {"params": "39M", "memory_gb": 2.8, "hours_100k": 1.1, "hf_id": "openai/whisper-tiny.en"},
        "distil-base.en-from-medium.en": {"params": "74M", "memory_gb": 4.2, "hours_100k": 1.8, "hf_id": "openai/whisper-base.en"},
        
        # Hybrid Encoder-Decoder Models (Fast decoding with quality encoding)
        "distil-large-encoder-tiny-decoder": {"params": "1550M→195M", "memory_gb": 10.2, "hours_100k": 6.0, "hf_id": "openai/whisper-large-v3"},
        "distil-medium-encoder-tiny-decoder": {"params": "769M→195M", "memory_gb": 6.8, "hours_100k": 3.5, "hf_id": "openai/whisper-medium"},
        "distil-small-encoder-tiny-decoder": {"params": "244M→195M", "memory_gb": 4.5, "hours_100k": 2.0, "hf_id": "openai/whisper-small"},
    }


def _infer_num_mel_bins(model_name_or_key: Any) -> int:
    """Return expected mel bins for a Whisper model key/name.
    
    Accepts a variety of inputs (string model key, tuple, or None) and
    normalizes to a lowercase string for robust handling.
    
    - Whisper large-v3 variants use 128 mel bins
    - Most other OpenAI Whisper models use 80 mel bins
    """
    # Normalize input to a lowercase string defensively
    value: Any = model_name_or_key
    if isinstance(value, tuple) and value:
        value = value[0]
    try:
        key = (value or "").lower()
    except Exception:
        key = str(value or "").lower()
    if "large-v3" in key:
        return 128
    return 80

def get_device_info() -> Dict[str, Any]:
    """Get device information for memory and time estimation"""
    device = get_device()
    
    # Get available memory (rough estimation)
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    device_info = {
        "type": device.type,
        "name": str(device),
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": available_memory_gb,
    }
    
    # Add device-specific optimizations
    if device.type == "mps":
        device_info["display_name"] = f"Apple Silicon ({device})"
        device_info["performance_multiplier"] = 1.0
    elif device.type == "cuda":
        device_info["display_name"] = f"NVIDIA GPU ({device})"  
        device_info["performance_multiplier"] = 0.7  # Generally faster
    else:
        device_info["display_name"] = f"CPU ({device})"
        device_info["performance_multiplier"] = 3.0  # Much slower
    
    return device_info

def show_welcome_screen():
    """Display an elegant welcome screen"""
    
    # ASCII art logo
    logo = """
    ██╗    ██╗██╗  ██╗██╗███████╗██████╗ ███████╗██████╗ 
    ██║    ██║██║  ██║██║██╔════╝██╔══██╗██╔════╝██╔══██╗
    ██║ █╗ ██║███████║██║███████╗██████╔╝█████╗  ██████╔╝
    ██║███╗██║██╔══██║██║╚════██║██╔═══╝ ██╔══╝  ██╔══██╗
    ╚███╔███╔╝██║  ██║██║███████║██║     ███████╗██║  ██║
     ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
                                                          
                 🍎 Fine-Tuner for Apple Silicon
    """
    
    device_info = get_device_info()
    
    welcome_text = f"""
[bold cyan]Welcome to the Whisper Fine-Tuning Wizard![/bold cyan]

We'll guide you through training your custom Whisper model in just a few questions.

[green]System Information:[/green]
• Device: {device_info['display_name']}
• Available Memory: {device_info['available_memory_gb']:.1f} GB
• Status: Ready for training ✅

[dim]Press Enter to begin...[/dim]
    """
    
    console.print(Panel(
        Align.center(Text(logo, style="bold blue"), vertical="middle"),
        title="🎯 Whisper Fine-Tuner",
        border_style="blue",
        padding=(1, 2)
    ))
    console.print(welcome_text)
    
    input()  # Wait for user to press Enter

def detect_datasets() -> List[Dict[str, Any]]:
    """Auto-detect available datasets under data/datasets plus curated sources.

    We intentionally scan only the immediate children of `data/datasets` to avoid
    treating the parent `data/` directory or the `datasets/` folder itself as a dataset.
    """
    datasets: List[Dict[str, Any]] = []

    # Prefer canonical layout: data/datasets/<name>
    root = Path("data/datasets")
    if root.exists():
        for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
            # Skip hidden and cache directories
            if subdir.name.startswith(".") or subdir.name in {".cache", "__pycache__"}:
                continue

            # Look for CSV files (common dataset format)
            csv_files = list(subdir.glob("*.csv"))
            if csv_files:
                datasets.append({
                    "name": subdir.name,
                    "type": "local_csv",
                    "path": str(subdir),
                    "files": len(csv_files),
                    "description": f"Local dataset with {len(csv_files)} CSV files",
                })

            # Look for audio files recursively inside this dataset folder
            audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
            audio_files: List[Path] = []
            for ext in audio_extensions:
                audio_files.extend(subdir.glob(f"**/{ext}"))
            if audio_files:
                datasets.append({
                    "name": subdir.name,
                    "type": "local_audio",
                    "path": str(subdir),
                    "files": len(audio_files),
                    "description": f"Local audio dataset with {len(audio_files)} files",
                })

    # Add BigQuery import option (virtual source)
    datasets.append({
        "name": "Import from Google BigQuery",
        "type": "bigquery_import",
        "description": "Query BQ, export surgical slice to _prepared.csv"
    })

    # Add common Hugging Face datasets
    hf_datasets = [
        {"name": "mozilla-foundation/common_voice_13_0", "type": "huggingface", "description": "Common Voice multilingual dataset"},
        {"name": "openslr/librispeech_asr", "type": "huggingface", "description": "LibriSpeech English ASR dataset"},
        {"name": "facebook/voxpopuli", "type": "huggingface", "description": "VoxPopuli multilingual dataset"},
    ]
    
    datasets.extend(hf_datasets)
    
    # Add custom dataset option
    datasets.append({
        "name": "custom",
        "type": "custom", 
        "description": "I'll specify my dataset path manually"
    })
    
    # Ensure the BigQuery import option appears first in the wizard list
    # without changing the relative order of the remaining entries.
    bigquery_first: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for item in datasets:
        if item.get("type") == "bigquery_import":
            bigquery_first.append(item)
        else:
            others.append(item)
    return bigquery_first + others

def select_training_method() -> Dict[str, Any]:
    """Step 1: Select training method with progressive disclosure"""
    
    console.print("\n[bold]Step 1: Choose your training method[/bold]")
    
    methods = [TrainingMethod.STANDARD, TrainingMethod.LORA, TrainingMethod.DISTILLATION]
    
    choices = []
    for method in methods:
        choices.append({
            "name": f"{method['name']} - {method['description']}",
            "value": method
        })
    
    selected_method = questionary.select(
        "What kind of fine-tuning do you want to run?",
        choices=choices,
        style=apple_style
    ).ask()
    
    return selected_method

def select_model(method: Dict[str, Any]):
    """Step 2: Select model based on training method, driven by config.ini.

    For distillation, this returns the STUDENT model key (not teacher).
    Teacher will be chosen in a later step via configure_method_specifics().
    """
    
    console.print(f"\n[bold]Step 2: Choose your model[/bold]")
    
    device_info = get_device_info()
    available_memory = device_info["available_memory_gb"]
    
    # Dynamically discover available models from config.ini
    cfg = _read_config()
    available_models = [s.replace("model:", "") for s in cfg.sections() if s.startswith("model:")]
    
    # Filter models based on the selected training method
    if method["key"] == "lora":
        base_models = [m for m in available_models if "lora" in m]
    elif method["key"] == "distillation":
        # For distillation, list base students (including medium) to fine-tune as student
        base_models = [
            m for m in available_models
            if ("tiny" in m or "base" in m or "small" in m or "medium" in m)
        ]
    else: # standard
        base_models = [m for m in available_models if "lora" not in m and "distil" not in m]

    # Build model choices with memory and time estimates
    choices = []
    # For distillation, restrict students to a clean set and add Custom Hybrid entry
    allowed_students = {
        "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
        "whisper-tiny.en", "whisper-base.en", "whisper-small.en", "whisper-medium.en",
    }
    seen: set[str] = set()
    for model_name in base_models:
        # Use a display-friendly name if the config name is long
        display_name = model_name.replace("-lora", "")
        if method["key"] == "distillation" and display_name not in allowed_students:
            continue
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
        
        # Create descriptive text for distillation models
        if "from-medium" in display_name:
            if "tiny" in display_name:
                model_desc = "Distill tiny (39M) from larger teacher"
            elif "base" in display_name:
                model_desc = "Distill base (74M) from larger teacher"
            else:
                model_desc = display_name
            choice_text = f"{model_desc} - ~{time_str}, {memory_str} memory"
        elif "encoder-tiny-decoder" in display_name:
            if "large" in display_name:
                model_desc = "Large Encoder / Tiny Decoder - Fast generation, best quality"
            elif "medium" in display_name:
                model_desc = "Medium Encoder / Tiny Decoder - Faster, good quality"
            elif "small" in display_name:
                model_desc = "Small Encoder / Tiny Decoder - Balanced speed & quality"
            else:
                model_desc = display_name
            choice_text = f"{model_desc} - ~{time_str}, {memory_str} memory"
        else:
            choice_text = f"{display_name} ({specs['params']}) - ~{time_str}, {memory_str} memory"
        
        # Add recommendation for optimal choice
        if display_name == "whisper-small" and method["key"] != "distillation":
            choice_text += " ⭐ Recommended"
        elif display_name == "distil-base-from-medium" and method["key"] == "distillation":
            choice_text += " ⭐ Recommended"
        
        choices.append({
            "name": choice_text,
            "value": model_name  # Return the full config name, e.g., "whisper-base"
        })

    # Distillation: add a Custom Hybrid option inline at Step 2
    if method["key"] == "distillation":
        choices.append({
            "name": "Build a Custom Hybrid (mix encoder and decoder)",
            "value": "__custom_hybrid__",
        })
    
    if not choices:
        console.print("[red]❌ No models available for your memory constraints. Consider using LoRA training.[/red]")
        sys.exit(1)
    
    prompt = (
        "Which model do you want to fine-tune?" if method["key"] != "distillation"
        else "Which student model do you want to train? (or choose Custom Hybrid)"
    )
    selected_model = questionary.select(
        prompt,
        choices=choices,
        style=apple_style
    ).ask()

    # If user chose Custom Hybrid, immediately ask how to customize (encoder/decoder)
    if method["key"] == "distillation" and selected_model == "__custom_hybrid__":
        cfg = _read_config()
        available_models = [s.replace("model:", "") for s in cfg.sections() if s.startswith("model:")]
        enc_sources = [m for m in available_models if ("large" in m or "medium" in m)]
        dec_sources = [m for m in available_models if ("tiny" in m or "base" in m or "small" in m)]

        enc_choice = questionary.select(
            "Choose an Encoder source (large/medium)",
            choices=[{"name": m, "value": m} for m in enc_sources],
            style=apple_style,
        ).ask()
        dec_choice = questionary.select(
            "Choose a Decoder source (tiny/base/small)",
            choices=[{"name": m, "value": m} for m in dec_sources],
            style=apple_style,
        ).ask()

        seed = {
            "student_model_type": "custom",
            "student_encoder_from": enc_choice,
            "student_decoder_from": dec_choice,
        }
        return selected_model, seed

    return selected_model, {}

def select_dataset(method: Dict[str, Any]) -> Dict[str, Any]:
    """Step 3: Select dataset"""
    
    console.print(f"\n[bold]Step 3: Choose your dataset[/bold]")
    
    datasets = detect_datasets()
    
    choices = []
    for dataset in datasets:
        if dataset["type"] == "local_csv" or dataset["type"] == "local_audio":
            choice_text = f"📁 {dataset['name']} - {dataset['description']}"
        elif dataset["type"] == "huggingface":
            choice_text = f"🤗 {dataset['name']} - {dataset['description']}"
        else:
            choice_text = f"⚙️ {dataset['description']}"
        
        choices.append({
            "name": choice_text,
            "value": dataset
        })
    
    selected_dataset = questionary.select(
        "Which dataset do you want to use for training?",
        choices=choices,
        style=apple_style
    ).ask()
    
    # Handle BigQuery import flow
    if selected_dataset.get("type") == "bigquery_import":
        bq_dataset = select_bigquery_table_and_export()
        return bq_dataset

    # Handle custom dataset path
    if selected_dataset["name"] == "custom":
        dataset_path = questionary.path(
            "Enter the path to your dataset:",
            style=apple_style
        ).ask()
        
        selected_dataset["path"] = dataset_path
        selected_dataset["name"] = Path(dataset_path).name
    
    return selected_dataset

def _read_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")
    return cfg

def _write_config(cfg: configparser.ConfigParser) -> None:
    with open("config.ini", "w") as f:
        cfg.write(f)

def _add_dataset_to_config(dataset_name: str, text_column: str) -> None:
    """Ensure `[dataset:dataset_name]` exists with source and text_column."""
    cfg = _read_config()
    section = f"dataset:{dataset_name}"
    if not cfg.has_section(section):
        cfg.add_section(section)
    cfg.set(section, "source", dataset_name)
    if text_column:
        cfg.set(section, "text_column", text_column)
    
    # BQ-created datasets have standard train/validation splits.
    # This ensures they are always present for the config validator.
    if not cfg.has_option(section, "train_split"):
        cfg.set(section, "train_split", "train")
    if not cfg.has_option(section, "validation_split"):
        cfg.set(section, "validation_split", "validation")
        
    _write_config(cfg)

def _update_bq_defaults(project_id: Optional[str], dataset_id: Optional[str]) -> None:
    cfg = _read_config()
    section = "bigquery"
    if not cfg.has_section(section):
        cfg.add_section(section)
    if project_id:
        cfg.set(section, "last_project_id", project_id)
    if dataset_id:
        cfg.set(section, "last_dataset_id", dataset_id)
    _write_config(cfg)

def _infer_candidate_columns(schema_fields: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    names = [f.get("name") if isinstance(f, dict) else getattr(f, "name", "") for f in schema_fields]
    names_lower = [str(n or "") for n in names]
    def pick(patterns: List[str]) -> List[str]:
        res: List[str] = []
        for p in patterns:
            for n in names:
                if n and n.lower() == p:
                    if n not in res:
                        res.append(n)
        return res
    audio_candidates = pick(["audio_path", "audio_url", "gcs_uri", "uri", "path", "audio"])
    transcript_candidates = pick(["text_perfect", "text_verbatim", "transcript", "asr_text", "text"]) 
    language_candidates = pick(["language", "lang", "locale"]) 
    # Add fallbacks if empty
    if not audio_candidates:
        audio_candidates = names[:5]
    if not transcript_candidates:
        transcript_candidates = names[:5]
    return audio_candidates, transcript_candidates, language_candidates

def select_bigquery_table_and_export() -> Dict[str, Any]:
    """Interactive flow to import a surgical slice from BigQuery and return a dataset dict.

    Produces a dataset directory under `data/datasets/` with a `_prepared.csv`,
    updates `config.ini`, and returns a `local_csv` dataset descriptor.
    """
    from core import bigquery as bq

    console.print("\n[bold]BigQuery Import[/bold]")

    # Auth check
    if not bq.check_gcp_auth():
        console.print("[yellow]GCP auth not detected. Run: gcloud auth application-default login[/yellow]")
        proceed = questionary.confirm("Continue anyway (may fail)?", default=False, style=apple_style).ask()
        if not proceed:
            return {"name": "custom", "type": "custom", "description": "Manual path"}

    # Defaults
    cfg = _read_config()
    last_project = cfg.get("bigquery", "last_project_id", fallback="")
    last_dataset = cfg.get("bigquery", "last_dataset_id", fallback="")

    # Project
    project_id = questionary.text("GCP Project ID:", default=last_project, style=apple_style).ask()

    # Dataset selection
    datasets = bq.list_datasets(project_id) or []
    if datasets:
        dataset_id = questionary.select("Dataset:", choices=datasets, style=apple_style).ask()
    else:
        dataset_id = questionary.text("Dataset ID:", default=last_dataset or "", style=apple_style).ask()

    # Table selection (single-table MVP)
    tables = bq.list_tables(project_id, dataset_id) or []
    if tables:
        table_id = questionary.select("Table:", choices=tables, style=apple_style).ask()
    else:
        table_id = questionary.text("Table ID:", style=apple_style).ask()

    _update_bq_defaults(project_id, dataset_id)

    # Schema and candidates
    schema = bq.get_table_schema(project_id, dataset_id, table_id)
    # Convert to serializable for inference helper
    schema_dicts = [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in schema]
    audio_cands, text_cands, lang_cands = _infer_candidate_columns(schema_dicts)

    audio_col = questionary.select("Audio path column:", choices=audio_cands, style=apple_style).ask()
    transcript_col = questionary.select("Transcript source column:", choices=text_cands, style=apple_style).ask()
    # The target column name should be the same as the source column name.
    # This removes the need for an extra user prompt.
    transcript_target = transcript_col

    use_language = False
    language_col = None
    languages: Optional[List[str]] = None
    if lang_cands:
        use_language = questionary.confirm("Filter by language?", default=True, style=apple_style).ask()
        if use_language:
            language_col = questionary.select("Language column:", choices=lang_cands, style=apple_style).ask()
            distinct = bq.get_distinct_languages(project_id, dataset_id, table_id, language_column=language_col) or []
            if distinct:
                languages = questionary.checkbox("Select languages (Space to toggle):", choices=distinct, style=apple_style).ask()
            else:
                languages = None

    # Sampling
    limit_str = questionary.text("Max rows to fetch (blank = no limit):", default="1000", style=apple_style).ask()
    try:
        limit = int(limit_str) if limit_str.strip() else None
    except Exception:
        limit = 1000
    sample_random = questionary.confirm("Random sample?", default=True, style=apple_style).ask()
    sample = "random" if sample_random else "first"

    extra_where = questionary.text("Advanced WHERE (optional):", default="", style=apple_style).ask()
    extra_where = extra_where.strip() or None

    # Execute export
    out_dir = Path("data/datasets")
    try:
        dataset_dir = bq.build_query_and_export(
            project_id=project_id,
            tables=[(dataset_id, table_id)],
            audio_col=audio_col,
            transcript_col=transcript_col,
            transcript_target=transcript_target,  # sets output column name
            language_col=language_col,
            languages=languages,
            limit=limit,
            sample=sample,  # "random" or "first"
            extra_where=extra_where,
            out_dir=out_dir,
        )
    except Exception as e:
        console.print(f"[red]BigQuery export failed:[/red] {e}")
        raise

    dataset_name = dataset_dir.name
    # Update config.ini for dataset resolution and text_column
    # Use the source column name (transcript_col) which is now the same as transcript_target
    _add_dataset_to_config(dataset_name, transcript_target)

    # Return dataset descriptor compatible with downstream flow
    return {
        "name": dataset_name,
        "type": "local_csv",
        "path": str(dataset_dir),
        "files": 1,
        "description": f"Imported from BigQuery {project_id}.{dataset_id}.{table_id}",
    }

def configure_method_specifics(method: Dict[str, Any], model: str | tuple, seed: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Step 4: Method-specific configuration (progressive disclosure)"""
    # Defensive: older call sites may pass a (model, seed) tuple.
    if isinstance(model, tuple):
        model, seed_from_tuple = model
        if seed is None and isinstance(seed_from_tuple, dict):
            seed = seed_from_tuple

    config = {} if seed is None else dict(seed)
    
    if method["key"] == "lora":
        console.print(f"\n[bold]Step 4: LoRA Configuration[/bold]")
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
            "LoRA rank (higher = more parameters to train):",
            choices=rank_choices,
            style=apple_style
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
            "LoRA alpha (controls adaptation strength):",
            choices=alpha_choices,
            style=apple_style
        ).ask()
        
        if alpha == "custom":
            alpha = questionary.text(
                "Enter custom alpha value:",
                default=str(default_alpha),
                style=apple_style
            ).ask()
            alpha = int(alpha)
        
        config["lora_alpha"] = alpha
        config["lora_dropout"] = 0.1  # Smart default
        config["use_peft"] = True
        
    elif method["key"] == "distillation":
        console.print(f"\n[bold]Step 4: Distillation Configuration[/bold]")
        # If user already chose Custom Hybrid in Step 2, skip asking architecture again
        arch_choice = "custom" if (model == "__custom_hybrid__" or config.get("student_model_type") == "custom") else "standard"
        
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
            teacher_models = ["whisper-large-v3", "whisper-large-v2", "whisper-medium"]
            student_mels = _infer_num_mel_bins(encoder_source)
            teacher_choices = []
            for teacher in teacher_models:
                txt = teacher
                if _infer_num_mel_bins(teacher) != student_mels:
                    txt += " (incompatible mel bins; not recommended)"
                teacher_choices.append({"name": txt, "value": teacher})
            teacher_choice = questionary.select(
                "Which teacher model should we distill knowledge from?",
                choices=teacher_choices,
                style=apple_style,
            ).ask()
        else:
            # Standard student: teacher from curated list with compatibility filter
            teacher_models = ["whisper-large-v3", "whisper-large-v2", "whisper-medium"]
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
                resolved_teacher = f"openai/{teacher_choice}" if teacher_choice.startswith("whisper-") else teacher_choice
        except Exception:
            resolved_teacher = teacher_choice
        config["teacher_model"] = resolved_teacher
        
        # Temperature
        temp_choices = [
            {"name": "2.0 (Conservative)", "value": 2.0},
            {"name": "5.0 (Balanced) ⭐ Recommended", "value": 5.0},
            {"name": "10.0 (Aggressive)", "value": 10.0},
            {"name": "Custom value", "value": "custom"}
        ]
        
        temperature = questionary.select(
            "Distillation temperature (higher = softer teacher guidance):",
            choices=temp_choices,
            style=apple_style
        ).ask()
        
        if temperature == "custom":
            temperature = questionary.text(
                "Enter custom temperature:",
                default="5.0",
                style=apple_style
            ).ask()
            temperature = float(temperature)
        
        config["temperature"] = temperature
    
    return config

def estimate_training_time(method: Dict[str, Any], model: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate training time and resource usage"""
    
    device_info = get_device_info()
    model_specs = ModelSpecs.MODELS.get(model, ModelSpecs.MODELS["whisper-base"])
    
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
        "eta": datetime.now() + timedelta(hours=estimated_hours)
    }

def show_confirmation_screen(method: Dict[str, Any], model: str, dataset: Dict[str, Any], 
                           method_config: Dict[str, Any], estimates: Dict[str, Any]) -> bool:
    """Step 5: Beautiful confirmation screen"""
    
    console.print(f"\n[bold cyan]Step 5: Ready to Train![/bold cyan]")
    
    # Create a beautiful configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan", width=20)
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Training Method", method["name"].replace("🚀", "").replace("🎨", "").replace("🧠", "").strip())
    # Distillation: show student architecture details (standard vs custom)
    if method["key"] == "distillation" and method_config.get("student_model_type") == "custom":
        config_table.add_row("Student", "Custom Hybrid")
        config_table.add_row("Encoder From", str(method_config.get("student_encoder_from")))
        config_table.add_row("Decoder From", str(method_config.get("student_decoder_from")))
    else:
        config_table.add_row("Model", f"{model} ({ModelSpecs.MODELS.get(model, {}).get('params', 'Unknown')})")
    config_table.add_row("Dataset", f"{dataset['name']} ({estimates['samples']:,} samples)")
    
    # Add method-specific configuration
    if method["key"] == "lora":
        config_table.add_row("LoRA Rank", str(method_config["lora_r"]))
        config_table.add_row("LoRA Alpha", str(method_config["lora_alpha"]))
    elif method["key"] == "distillation":
        config_table.add_row("Teacher Model", method_config["teacher_model"])
        config_table.add_row("Temperature", str(method_config["temperature"]))
    
    config_table.add_row("", "")  # Spacer
    config_table.add_row("Estimated Time", f"{estimates['hours']:.1f} hours")
    config_table.add_row("Memory Usage", f"{estimates['memory_gb']:.1f} GB")
    config_table.add_row("Completion ETA", estimates['eta'].strftime("%I:%M %p today" if estimates['hours'] < 12 else "%I:%M %p tomorrow"))
    
    device_info = get_device_info()
    config_table.add_row("Training Device", device_info['display_name'])
    
    # Status indicators
    memory_status = "🟢 Sufficient" if estimates['memory_gb'] < device_info['available_memory_gb'] * 0.8 else "🟡 Tight"
    config_table.add_row("Memory Status", memory_status)
    
    # Show the panel
    console.print(Panel(
        config_table,
        title="🎯 Training Configuration",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Ask about visualization
    console.print(f"\n[bold cyan]Optional: Enable Training Visualizer?[/bold cyan]")
    console.print("[dim]Watch your AI learn in real-time with stunning 3D graphics![/dim]")
    
    enable_viz = questionary.confirm(
        "🎆 Enable live training visualization?",
        default=True,
        style=apple_style
    ).ask()
    
    # Store visualization choice for later use
    method_config['visualize'] = enable_viz
    
    if enable_viz:
        console.print("[green]✨ Visualization will open in your browser when training starts![/green]")
    
    # Confirmation prompt
    return questionary.confirm(
        "Start training with this configuration?",
        default=True,
        style=apple_style
    ).ask()

def generate_profile_config(method: Dict[str, Any], model: str, dataset: Dict[str, Any], 
                          method_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate config dict for the existing training infrastructure by leveraging the core config loader."""
    
    from core.config import load_model_dataset_config
    
    # Load the base configuration from config.ini using the robust, hierarchical loader.
    # This ensures that all central defaults are respected.
    cfg = _read_config()
    # For custom hybrid, load base defaults from decoder source instead of sentinel model
    model_for_loader = model
    if method["key"] == "distillation" and method_config.get("student_model_type") == "custom" and model == "__custom_hybrid__":
        model_for_loader = method_config.get("student_decoder_from") or model
    profile_config = load_model_dataset_config(cfg, model_for_loader, dataset["name"])

    # CRITICAL: Add the model and dataset keys that are required by load_profile_config
    # These are not included in load_model_dataset_config but are required for profile sections
    profile_config["model"] = model_for_loader
    profile_config["dataset"] = dataset["name"]

    # Layer the user's interactive choices on top of the base configuration.
    # This overrides the defaults with the specific parameters selected in the wizard.
    
    # Method-specific configuration
    if method["key"] == "lora":
        profile_config.update({
            "use_peft": True,
            "peft_method": "lora",
            "lora_r": method_config["lora_r"],
            "lora_alpha": method_config["lora_alpha"], 
            "lora_dropout": method_config.get("lora_dropout", 0.1), # Sensible default
            # Use canonical key expected by trainer; leave as list, not string
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        })
    elif method["key"] == "distillation":
        profile_config.update({
            "teacher_model": method_config["teacher_model"],
            "distillation_temperature": method_config["temperature"],
            "distillation_alpha": 0.5,  # Balance between hard and soft targets
        })
        # Propagate custom student architecture if selected
        if method_config.get("student_model_type") == "custom":
            profile_config["student_model_type"] = "custom"
            profile_config["student_encoder_from"] = method_config.get("student_encoder_from")
            profile_config["student_decoder_from"] = method_config.get("student_decoder_from")
    
    # Dataset-specific configuration
    if dataset["type"] == "huggingface":
        profile_config["dataset_name"] = dataset["name"]
        profile_config["dataset_config"] = "en"  # Default to English
        profile_config["train_split"] = "train"
        profile_config["eval_split"] = "validation"
    elif dataset["type"] in ["local_csv", "local_audio"]:
        profile_config["train_dataset_path"] = dataset["path"]
        profile_config["eval_dataset_path"] = dataset["path"]  # Same for now
    
    # Add visualization flag if enabled
    if method_config.get('visualize', False):
        profile_config['visualize'] = True
    
    # Ensure required splits are always present for validation
    # These are required by the configuration validator
    if "train_split" not in profile_config:
        profile_config["train_split"] = "train"
    if "validation_split" not in profile_config:
        profile_config["validation_split"] = "validation"
    
    return profile_config

def execute_training(profile_config: Dict[str, Any]):
    """Execute training using the existing main.py infrastructure"""
    
    console.print(f"\n[bold green]🚀 Starting training...[/bold green]")
    
    import subprocess
    import argparse
    
    # Create a temporary config file
    config_dir = Path("temp_configs")
    config_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_config_path = config_dir / f"wizard_config_{timestamp}.ini"
    
    # Start by reading the main config.ini to get all model and dataset definitions
    main_config = _read_config()
    
    # Create a new config that copies all necessary sections from main config
    config = configparser.ConfigParser()
    
    # Copy DEFAULT section
    # Note: ConfigParser treats DEFAULT as a special section that is not returned by
    # has_section("DEFAULT"). We must read it directly to preserve global defaults
    # like num_train_epochs, logging_steps, etc., which training requires.
    try:
        config["DEFAULT"] = dict(main_config["DEFAULT"])  # always present if file parsed
    except Exception:
        # Fallback minimal defaults if main config is malformed
        config["DEFAULT"] = {
            "output_dir": "output",
            "logging_dir": "logs",
        }
    
    # Copy all essential sections from main config
    # This ensures model definitions, dataset definitions, and group configs are available
    for section in main_config.sections():
        if section.startswith(("model:", "dataset:", "group:", "dataset_defaults")):
            config[section] = dict(main_config[section])
    
    # Create the wizard profile section with user's selections
    profile_name = f"wizard_{timestamp}"
    config[f"profile:{profile_name}"] = profile_config
    
    # Write temporary config
    with open(temp_config_path, 'w') as f:
        config.write(f)
    
    console.print("[dim]Training started! This may take several hours...[/dim]")
    console.print("[dim]Press Ctrl+C to interrupt (training will be saved at checkpoints)[/dim]")
    
    try:
        # Execute training via subprocess to avoid import side effects
        # Use module invocation so this works when installed as a package
        module_cwd = Path(__file__).resolve().parent
        result = subprocess.run([
            sys.executable,
            "-m", "main",
            "finetune",
            profile_name,
            "--config",
            str(temp_config_path.resolve())
        ], check=True, text=True, capture_output=False, cwd=str(module_cwd))
        
        console.print(f"\n[bold green]✅ Training completed successfully![/bold green]")
        console.print(f"[green]Model saved in output directory[/green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]❌ Training failed with exit code {e.returncode}[/red]")
        console.print(f"[red]Check the logs for detailed error information[/red]")
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⚠️ Training interrupted by user[/yellow]")
        console.print(f"[yellow]Progress saved at latest checkpoint[/yellow]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Training execution failed: {str(e)}[/red]")
        console.print(f"[red]Check your configuration and try again[/red]")
        
    finally:
        pass  # No cleanup needed with subprocess approach
        
        # Clean up temporary config
        try:
            temp_config_path.unlink()
        except:
            pass

def wizard_main():
    """Main wizard entry point - orchestrates the entire flow"""
    
    try:
        # Step 0: Welcome screen
        show_welcome_screen()
        
        # Step 1: Select training method
        method = select_training_method()
        
        # Step 2: Select model (returns (model_key, seed))
        model, seed = select_model(method)
        
        # Step 3: Select dataset  
        dataset = select_dataset(method)
        
        # Step 4: Method-specific configuration (pass seed for custom hybrids)
        method_config = configure_method_specifics(method, model, seed)
        
        # Step 5: Estimate time and resources
        estimates = estimate_training_time(method, model, dataset)
        
        # Step 6: Confirmation screen
        if show_confirmation_screen(method, model, dataset, method_config, estimates):
            
            # Generate configuration
            profile_config = generate_profile_config(method, model, dataset, method_config)
            
            # Execute training
            execute_training(profile_config)
            
        else:
            console.print(f"\n[yellow]Training cancelled by user.[/yellow]")
            console.print(f"[dim]Run the wizard again anytime with: python manage.py finetune-wizard[/dim]")
            
    except KeyboardInterrupt:
        console.print(f"\n\n[yellow]Wizard interrupted by user.[/yellow]")
        console.print(f"[dim]No changes made to your system.[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Wizard error: {str(e)}[/red]")
        console.print(f"[red]Please report this issue or try manual configuration.[/red]")
        raise

if __name__ == "__main__":
    wizard_main()