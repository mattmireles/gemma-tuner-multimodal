import argparse
import configparser
import importlib
import os
from datetime import datetime
from scripts.utils import update_metadata
import json
import torch
import traceback
from filelock import FileLock
from utils.device import get_device, set_memory_fraction

# Get the best available device
device = get_device()

# Device-specific optimizations
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
elif device.type == "mps":
    # Set memory limit for MPS to prevent system-wide memory pressure
    set_memory_fraction(0.8)

def get_next_run_id(output_dir):
    """Gets the next available run ID."""
    lock_file = os.path.join(output_dir, "next_run_id.txt.lock")
    lock = FileLock(lock_file)

    with lock:
        try:
            with open(os.path.join(output_dir, "next_run_id.txt"), "r") as f:
                next_id = int(f.read())
        except FileNotFoundError:
            next_id = 1

        with open(os.path.join(output_dir, "next_run_id.txt"), "w") as f:
            f.write(str(next_id + 1))

    return next_id

def find_finetuning_run_dir(output_dir, profile_name):
    """
    Finds the directory of the latest finetuning run for a given profile.

    Args:
        output_dir: The base output directory.
        profile_name: The name of the profile.

    Returns:
        The path to the latest finetuning run directory, or None if no such run is found.
    """
    finetuning_runs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and f"-{profile_name}" in d and os.path.exists(os.path.join(output_dir, d, "completed"))
    ]
    if not finetuning_runs:
        return None
    finetuning_runs.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)), reverse=True)
    return os.path.join(output_dir, finetuning_runs[0])


def mark_run_as_completed(run_dir):
    """Marks a run as completed by creating a 'completed' file."""
    with open(os.path.join(run_dir, "completed"), "w") as f:
        f.write("completed")

def create_run_directory(output_dir, profile_name, run_id, run_type, model_name=None, dataset_name=None):
    """Creates a directory for a run and initializes metadata.json."""
    if run_type == "finetuning":
        run_dir = os.path.join(output_dir, f"{run_id}-{profile_name}")
    elif run_type == "evaluation" or run_type == "blacklist":
        if profile_name and not dataset_name:
            # Profile-based evaluation
            finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            run_dir = os.path.join(finetuning_run_dir, "eval")
        elif model_name and dataset_name:
            # Model+dataset-based evaluation
            run_id = f"{model_name}+{dataset_name}"  # Set run_id to model+dataset
            run_dir = os.path.join(output_dir, run_id, "eval")
        elif profile_name and dataset_name:
            # profile + dataset evaluation
            finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            run_dir = os.path.join(finetuning_run_dir, f"eval-{dataset_name}")
        else:
            raise ValueError("Invalid run type of evaluation parameters")
    else:
        raise ValueError(f"Invalid run type: {run_type}")

    os.makedirs(run_dir, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "run_type": run_type,
        "status": "running",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "profile": profile_name,
        "model": model_name,
        "dataset": dataset_name,
        "config": {},
        "metrics": {},
        "finetuning_run_id": None
    }

    if run_type == "evaluation" and profile_name:
        finetuning_run_dir = find_latest_finetuning_run(output_dir, profile_name)
        if finetuning_run_dir:
            finetuning_run_id = os.path.basename(finetuning_run_dir).split("-")[0]
            metadata["finetuning_run_id"] = finetuning_run_id

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return run_dir

def find_latest_finetuning_run(output_dir, profile_name):
    """Finds the latest finetuning run directory for a given profile."""
    finetuning_runs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and f"-{profile_name}" in d
    ]
    if not finetuning_runs:
        return None

    finetuning_runs.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)), reverse=True)
    return os.path.join(output_dir, finetuning_runs[0])

def update_run_metadata(run_dir, **kwargs):
    """Updates the metadata.json file for a run."""
    metadata_file = os.path.join(run_dir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    metadata.update(kwargs)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

def main():
    parser = argparse.ArgumentParser(
        description="Main entry point for data preparation, finetuning, evaluation, and export."
    )
    parser.add_argument("operation", choices=["prepare", "finetune", "evaluate", "export", "pseudo_label", "gather", "validate_data", "system_check", "blacklist"], help="Operation to perform")
    parser.add_argument("profile_or_model_dataset", nargs="?", help="Name of the profile to use (from config.ini) or model+dataset combination")
    parser.add_argument("--config", default="config.ini", help="Path to the configuration file.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training or evaluation samples to use.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use for evaluation (overrides profile dataset).")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use for evaluation/blacklisting (overrides profile split).")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    output_dir = config["DEFAULT"]["output_dir"]

    if args.operation == "gather":
        if args.profile_or_model_dataset:
            args.profiles = [p.strip() for p in args.profile_or_model_dataset.split(',')]

    if args.operation == "prepare":
        from scripts.prepare_data import prepare_data
        prepare_data(args.config)

    elif args.operation == "finetune":
        if not args.profile_or_model_dataset:
            parser.error("The 'finetune' operation requires a profile name.")
        if "+" in args.profile_or_model_dataset:
            parser.error("Profile names for 'finetune' cannot contain '+'.")

        profile_name = args.profile_or_model_dataset
        run_id = get_next_run_id(output_dir)
        run_dir = create_run_directory(output_dir, profile_name, run_id, "finetuning")

        try:
            # Load profile settings
            profile_config = load_profile_config(config, profile_name)

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            # Dynamically import the finetune module
            finetune_module = importlib.import_module("scripts.finetune")

            # Update metadata with config
            update_run_metadata(run_dir, config=profile_config, model=profile_config["model"], dataset=profile_config["dataset"])

            # fallbacks
            profile_config["force_languages"] = profile_config.get("force_languages", False)
            profile_config["languages"] = profile_config.get("languages", "all")

            # Call the main function of the finetune module, passing the profile config as a string
            finetune_module.main(profile_config, run_dir)
            mark_run_as_completed(run_dir)
            update_run_metadata(run_dir, status="completed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        except Exception as e:
            print(f"Error during finetuning: {e}")
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=str(e))

    elif args.operation == "evaluate":
        if not args.profile_or_model_dataset:
            parser.error("The 'evaluate' operation requires a profile name or a model+dataset combination.")

        run_id = get_next_run_id(output_dir)

        if "+" in args.profile_or_model_dataset:
            # Handle model+dataset combination
            model_name, dataset_name = args.profile_or_model_dataset.split("+")
            profile_config = load_model_dataset_config(config, model_name, dataset_name)
            run_dir = create_run_directory(output_dir, None, run_id, "evaluation", model_name=model_name, dataset_name=dataset_name)
            profile_config['model_name_or_path'] = profile_config["base_model"]

        else:
            # Handle profile or profile+dataset
            profile_name = args.profile_or_model_dataset
            profile_config = load_profile_config(config, profile_name)

            if args.dataset:
                # Profile + Dataset evaluation
                dataset_name = args.dataset
                run_dir = create_run_directory(output_dir, profile_name, run_id, "evaluation", dataset_name=dataset_name)
            else:
                # Profile evaluation with dataset from profile_config
                dataset_name = profile_config["dataset"]
                run_dir = create_run_directory(output_dir, profile_name, run_id, "evaluation")

            # Find latest finetuning run for the profile
            latest_run_dir = find_latest_finetuning_run(output_dir, profile_name)
            if latest_run_dir:
                profile_config["model_name_or_path"] = latest_run_dir
            else:
                print(f"Error: No finetuning runs found for profile: {profile_name}")
                return

        try:
            # Update metadata with config
            update_run_metadata(run_dir, config=profile_config)

            from scripts.evaluate import run_evaluation
            profile_config["force_languages"] = profile_config.get("force_languages", False)
            profile_config["languages"] = profile_config.get("languages", "all")
            profile_config["dataset"] = dataset_name

            if 'language_mode' not in profile_config:
                print("Warning: Defaulting to 'strict' language mode")
                profile_config['language_mode'] = 'strict'

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            metrics = run_evaluation(profile_config, run_dir)

            if metrics:
                update_run_metadata(run_dir, metrics=metrics)

            mark_run_as_completed(run_dir)
            update_run_metadata(run_dir, status="completed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        except Exception as e:
            print(f"Error during evaluation: {e}")
            update_run_metadata(run_dir, status="failed", end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), error_message=str(e))

    elif args.operation == "export":
        from scripts.export import export_ggml
        export_ggml(config, args.profile_or_model_dataset)

    elif args.operation == "pseudo_label":
        from scripts.pseudo_label import main as pseudo_label_main
        pseudo_label_main()

    elif args.operation == "gather":
        from scripts.gather import gather_predictions
        gather_predictions(args.profiles)

    elif args.operation == "validate_data":
        from scripts.validate_data import main as validate_data_main
        validate_data_main(config)

    elif args.operation == "system_check":
        from scripts.system_check import main as system_check_main
        system_check_main()

    elif args.operation == "blacklist":

        finetuning_run_dir = find_finetuning_run_dir(output_dir, args.profile_or_model_dataset)
        run_dir = os.path.join(finetuning_run_dir, "blacklist")

        profile_name = args.profile_or_model_dataset
        run_id = get_next_run_id(output_dir)
        run_dir = create_run_directory(output_dir, profile_name, run_id, "blacklist")

        try:
            profile_config = load_profile_config(config, profile_name)

            # Use the specified split or fallback to the profile's train split
            split = args.split if args.split else profile_config["train_split"]
            profile_config["split"] = split
            profile_config['model_name_or_path'] = finetuning_run_dir

            # Add max_samples to profile_config if provided
            if args.max_samples is not None:
                profile_config["max_samples"] = args.max_samples

            from scripts.blacklist import create_blacklist
            blacklist_path = create_blacklist(profile_config, run_dir)

            print(f"Blacklist created at: {blacklist_path}")

        except Exception as e:
            print(f"Error during blacklist creation: {e}")
            traceback.print_exc()

    else:
        print(f"Invalid operation: {args.operation}")

def get_latest_run_directory(base_dir):
    """Finds the latest run directory based on timestamps in the directory names."""
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("20")]
    if not run_dirs:
        return None
    run_dirs.sort(reverse=True)
    return os.path.join(base_dir, run_dirs[0])

def load_profile_config(config, profile_name):
    """Loads the configuration for the specified profile, including defaults."""
    profile_section = f"profile:{profile_name}"
    if not config.has_section(profile_section):
        raise ValueError(f"Profile '{profile_name}' not found in config.ini.")

    profile_config = {}

    # Load defaults
    if config.has_section("DEFAULT"):
        profile_config.update(config["DEFAULT"])
    if config.has_section("dataset_defaults"):
        profile_config.update(config["dataset_defaults"])

    # Load model group and model defaults
    model_name = config.get(profile_section, "model")
    model_section = f"model:{model_name}"
    if config.has_section(model_section):
        group_name = config.get(model_section, "group")
        group_section = f"group:{group_name}"
        if config.has_section(group_section):
            profile_config.update(config[group_section])
        profile_config.update(config[model_section])

    # Load dataset defaults
    dataset_name = config.get(profile_section, "dataset")
    dataset_section = f"dataset:{dataset_name}"
    if config.has_section(dataset_section):
        profile_config.update(config[dataset_section])

    # Load profile settings (overrides defaults)
    profile_config.update(config[profile_section])

    return profile_config

def load_model_dataset_config(config, model_name, dataset_name):
    """Loads the configuration for a given model and dataset combination."""
    model_section = f"model:{model_name}"
    dataset_section = f"dataset:{dataset_name}"

    if not config.has_section(model_section):
        raise ValueError(f"Model '{model_name}' not found in config.ini.")
    if not config.has_section(dataset_section):
        raise ValueError(f"Dataset '{dataset_name}' not found in config.ini.")

    config_dict = {}

    # Load defaults
    if config.has_section("DEFAULT"):
        config_dict.update(config["DEFAULT"])
    if config.has_section("dataset_defaults"):
        config_dict.update(config["dataset_defaults"])

    # Load model group defaults
    group_name = config.get(model_section, "group")
    group_section = f"group:{group_name}"
    if config.has_section(group_section):
        config_dict.update(config[group_section])

    # Load model and dataset settings
    config_dict.update(config[model_section])
    config_dict.update(config[dataset_section])

    return config_dict

if __name__ == "__main__":
    main()
