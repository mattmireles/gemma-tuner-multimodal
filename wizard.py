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

# Ensure this project's root is first on sys.path to avoid name collisions
# with other projects that may also define a top-level `constants` module.
try:
    import sys as _sys
    from pathlib import Path as _Path
    _project_root = _Path(__file__).resolve().parent
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))
except Exception:
    pass

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

# Robust import of MemoryLimits with safe fallback
try:
    from constants import MemoryLimits  # type: ignore
except Exception:
    class MemoryLimits:  # Fallback defaults
        MPS_DEFAULT_FRACTION = 0.8
        CUDA_DEFAULT_FRACTION = 0.9
    print(
        "constants module not found or incompatible. Using default MemoryLimits. "
        "Ensure this project's constants.py is available."
    )

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
        "whisper-tiny": {"params": "39M", "memory_gb": 1.2, "hours_100k": 0.5},
        "whisper-base": {"params": "74M", "memory_gb": 2.1, "hours_100k": 1.0}, 
        "whisper-small": {"params": "244M", "memory_gb": 4.2, "hours_100k": 2.5},
        "whisper-medium": {"params": "769M", "memory_gb": 8.4, "hours_100k": 6.0},
        "whisper-large": {"params": "1550M", "memory_gb": 16.8, "hours_100k": 12.0},
        "whisper-large-v2": {"params": "1550M", "memory_gb": 16.8, "hours_100k": 12.0},
        "whisper-large-v3": {"params": "1550M", "memory_gb": 16.8, "hours_100k": 12.0},
        "distil-whisper-small": {"params": "166M", "memory_gb": 3.2, "hours_100k": 1.8},
        "distil-whisper-medium": {"params": "394M", "memory_gb": 6.1, "hours_100k": 3.5},
        "distil-whisper-large-v2": {"params": "756M", "memory_gb": 12.4, "hours_100k": 8.0},
    }

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
    """Auto-detect available datasets"""
    datasets = []
    
    # Check for local datasets
    data_dir = Path("data")
    if data_dir.exists():
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                # Look for CSV files (common dataset format)
                csv_files = list(subdir.glob("*.csv"))
                if csv_files:
                    datasets.append({
                        "name": subdir.name,
                        "type": "local_csv",
                        "path": str(subdir),
                        "files": len(csv_files),
                        "description": f"Local dataset with {len(csv_files)} CSV files"
                    })
                
                # Look for audio files
                audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
                audio_files = []
                for ext in audio_extensions:
                    audio_files.extend(list(subdir.glob(f"**/{ext}")))
                
                if audio_files:
                    datasets.append({
                        "name": subdir.name,
                        "type": "local_audio",
                        "path": str(subdir),
                        "files": len(audio_files),
                        "description": f"Local audio dataset with {len(audio_files)} files"
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
    
    return datasets

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

def select_model(method: Dict[str, Any]) -> str:
    """Step 2: Select model based on training method"""
    
    console.print(f"\n[bold]Step 2: Choose your model[/bold]")
    
    device_info = get_device_info()
    available_memory = device_info["available_memory_gb"]
    
    # Filter models based on method and memory constraints
    if method["key"] == "standard":
        base_models = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"]
    elif method["key"] == "lora":
        base_models = ["whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"]  # LoRA can handle larger models
    else:  # distillation
        base_models = ["distil-whisper-small", "distil-whisper-medium", "distil-whisper-large-v2"]
    
    # Build model choices with memory and time estimates
    choices = []
    for model_name in base_models:
        if model_name not in ModelSpecs.MODELS:
            continue
            
        specs = ModelSpecs.MODELS[model_name]
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
        
        choice_text = f"{model_name} ({specs['params']}) - ~{time_str}, {memory_str} memory"
        
        # Add recommendation for optimal choice
        if model_name == "whisper-small" and method["key"] != "distillation":
            choice_text += " ⭐ Recommended"
        elif model_name == "distil-whisper-small" and method["key"] == "distillation":
            choice_text += " ⭐ Recommended"
        
        choices.append({
            "name": choice_text,
            "value": model_name
        })
    
    if not choices:
        console.print("[red]❌ No models available for your memory constraints. Consider using LoRA training.[/red]")
        sys.exit(1)
    
    selected_model = questionary.select(
        f"Which model do you want to {'fine-tune' if method['key'] != 'distillation' else 'train'}?",
        choices=choices,
        style=apple_style
    ).ask()
    
    return selected_model

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
    
    # Handle custom dataset path
    if selected_dataset["name"] == "custom":
        dataset_path = questionary.path(
            "Enter the path to your dataset:",
            style=apple_style
        ).ask()
        
        selected_dataset["path"] = dataset_path
        selected_dataset["name"] = Path(dataset_path).name
    
    return selected_dataset

def configure_method_specifics(method: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Step 4: Method-specific configuration (progressive disclosure)"""
    
    config = {}
    
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
        
        # Teacher model selection
        teacher_models = ["whisper-large-v3", "whisper-large-v2", "whisper-medium"]
        teacher_choices = []
        
        for teacher in teacher_models:
            if teacher != model:  # Don't allow same model as teacher and student
                choice_text = f"{teacher}"
                if teacher == "whisper-large-v3":
                    choice_text += " ⭐ Recommended"
                teacher_choices.append({"name": choice_text, "value": teacher})
        
        config["teacher_model"] = questionary.select(
            "Which teacher model should we distill knowledge from?",
            choices=teacher_choices,
            style=apple_style
        ).ask()
        
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
    """Generate config dict for the existing training infrastructure"""
    
    # Base configuration
    profile_config = {
        "model": model,
        "dataset": dataset["name"],
        "dataset_path": dataset.get("path", ""),
        "learning_rate": "1e-5",  # Smart default
        "batch_size": "16",
        "num_epochs": "3",
        "warmup_steps": "500",
        "eval_steps": "1000",
        "save_steps": "1000",
        "logging_steps": "100",
        "gradient_checkpointing": True,
        "fp16": True,
        "dataloader_num_workers": "4",
        "remove_unused_columns": False,
        "label_smoothing_factor": "0.1",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": "3",
    }
    
    # Method-specific configuration
    if method["key"] == "lora":
        profile_config.update({
            "use_peft": True,
            "peft_method": "lora",
            "lora_r": method_config["lora_r"],
            "lora_alpha": method_config["lora_alpha"], 
            "lora_dropout": method_config["lora_dropout"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        })
    elif method["key"] == "distillation":
        profile_config.update({
            "teacher_model": method_config["teacher_model"],
            "distillation_temperature": method_config["temperature"],
            "distillation_alpha": 0.5,  # Balance between hard and soft targets
        })
    
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
    
    # Generate INI format config
    config = configparser.ConfigParser()
    config["DEFAULT"] = {
        "output_dir": "output",
        "logging_dir": "logs",
    }
    
    # Create a profile section
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
        result = subprocess.run([
            sys.executable,
            "-m", "main",
            "finetune",
            profile_name,
            "--config",
            str(temp_config_path)
        ], check=True, text=True, capture_output=False)
        
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
        
        # Step 2: Select model
        model = select_model(method)
        
        # Step 3: Select dataset  
        dataset = select_dataset(method)
        
        # Step 4: Method-specific configuration
        method_config = configure_method_specifics(method, model)
        
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