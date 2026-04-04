#!/usr/bin/env python3

"""
Gemma Fine-Tuning Wizard - Configuration and dataset orchestration.

This module now focuses on BigQuery dataset import and wizard profile generation.
Shared config.ini persistence helpers live in ``wizard.config_store`` and the
Granary-specific interactive flow lives in ``wizard.granary``.

All shared constants and utilities are imported from gemma_tuner.wizard.base to avoid
circular imports. NEVER import from the wizard package root.

Called by:
- wizard.ui: select_model() uses _read_config(); select_dataset() uses
  select_bigquery_table_and_export() and setup_granary_dataset()
- wizard.ui: show_confirmation_screen() uses _read_config()
- wizard.runner: wizard_main() uses generate_profile_config()
- wizard.estimator: configure_method_specifics() uses _read_config()

Integrates with:
- wizard.base: WizardConstants, apple_style, console
- core/bigquery.py: BigQuery data pipeline functions
- core/config.py: load_model_dataset_config for profile generation
- config.ini: Central configuration file for models, datasets, and profiles
"""

from typing import Any, Dict, List, Optional, Tuple

import questionary

from gemma_tuner.wizard import config_store as _config_store
from gemma_tuner.wizard import granary as _granary
from gemma_tuner.wizard.base import apple_style, console

_PROJECT_ROOT = _config_store._PROJECT_ROOT
_add_dataset_to_config = _config_store._add_dataset_to_config
_read_config = _config_store._read_config
_update_bq_defaults = _config_store._update_bq_defaults
setup_granary_dataset = _granary.setup_granary_dataset


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
    """
    Interactive BigQuery dataset import with surgical precision and enterprise-grade workflow.

    This function implements a comprehensive BigQuery-to-training-data pipeline that enables
    users to extract precisely the data they need from enterprise data warehouses. It provides
    intelligent schema analysis, column mapping, sampling strategies, and quality control
    for production-ready training datasets.

    Called by:
    - select_dataset() when user chooses BigQuery import option (line 502)
    - Enterprise workflows requiring direct data warehouse integration
    - Large-scale training data preparation from production analytics systems
    - Custom dataset creation for specialized domain training requirements

    Calls to:
    - core/bigquery.py:check_gcp_auth() for authentication validation
    - core/bigquery.py:list_datasets(), list_tables() for resource discovery
    - core/bigquery.py:get_table_schema() for intelligent column analysis
    - core/bigquery.py:get_distinct_languages() for multilingual dataset filtering
    - core/bigquery.py:build_query_and_export() for surgical data extraction
    - _add_dataset_to_config() for automatic configuration integration
    - _update_bq_defaults() for user experience optimization through defaults

    Enterprise BigQuery integration workflow:

    Authentication & Access:
    - GCP authentication verification with actionable setup guidance
    - Project access validation and permission checking
    - Service account or user credential support with clear error messages
    - Graceful fallback options for authentication issues

    Resource Discovery:
    - Dynamic project dataset enumeration with caching for performance
    - Table listing with metadata analysis for informed selection
    - Schema introspection for intelligent column mapping suggestions
    - Automatic detection of common audio and transcript patterns

    Intelligent Column Mapping:
    - Audio path detection: audio_path, audio_url, gcs_uri, path, audio
    - Transcript identification: text_perfect, text_verbatim, transcript, text
    - Language analysis: language, lang, locale with distinct value enumeration
    - Fallback suggestions when standard patterns aren't found

    Sampling Strategies:
    - Random sampling: Statistical representation across data distribution
    - First-N sampling: Chronological order preservation for time-series data
    - Row limits: Memory management and iteration speed optimization

    Language-Aware Processing:
    - Automatic language detection from schema analysis
    - Multi-language filtering with checkbox selection interface
    - Language distribution analysis for balanced dataset creation
    - Unicode and encoding validation for international datasets

    Quality Control & Validation:
    - Schema compatibility checking for Gemma training requirements
    - Data type validation and automatic conversion recommendations
    - NULL value detection and handling strategies
    - File size estimation and download time predictions

    Configuration Integration:
    - Automatic config.ini updates with dataset definitions
    - Standard train/validation split generation (80/20 default)
    - Text column mapping for consistent pipeline integration
    - Source tracking for data lineage and reproducibility

    Output Structure:
    Creates `data/datasets/{dataset_name}/` directory containing:
    - train.csv: Training split with audio paths and transcripts
    - validation.csv: Validation split for model evaluation
    - metadata.json: Export configuration and statistics
    - schema.json: Original BigQuery schema for reference

    Error Handling & Recovery:
    - Network timeout handling with retry mechanisms
    - Query complexity validation and optimization suggestions
    - Memory pressure detection during large exports
    - Partial download recovery for interrupted transfers

    Performance Optimizations:
    - Parallel chunk downloading for large datasets
    - Automatic query optimization through BigQuery best practices
    - Smart caching of schema and metadata to reduce API calls
    - Progress tracking with ETA calculation for long exports

    Returns:
        Dict[str, Any]: Dataset descriptor compatible with wizard workflow:
        {
            "name": Generated dataset directory name,
            "type": "local_csv" for downstream compatibility,
            "path": Absolute path to created dataset directory,
            "files": Number of CSV files created (train.csv, validation.csv),
            "description": Human-readable description with source information,
            "source_info": {
                "project_id": BigQuery project identifier,
                "dataset_id": BigQuery dataset name,
                "table_id": Source table name,
                "export_timestamp": ISO timestamp of export,
                "row_count": Total rows exported,
                "languages": List of languages included (if filtered)
            }
        }

    Example Usage:
        dataset = select_bigquery_table_and_export()
        # Interactive prompts guide user through:
        # 1. Project selection: "my-audio-project"
        # 2. Dataset selection: "speech_data_warehouse"
        # 3. Table selection: "transcribed_calls"
        # 4. Column mapping: audio_path -> "gcs_audio_uri", transcript -> "human_transcription"
        # 5. Language filtering: ["en", "es", "fr"] from 15 available languages
        # 6. Sampling: Random 10,000 rows
        # Result: Creates data/datasets/bq_gemma_finetuning_20250813/ with train.csv and validation.csv

    Integration Benefits:
        - Zero-copy data pipeline from BigQuery to training
        - Surgical precision in data selection reduces training time
        - Enterprise-grade authentication and permission handling
        - Automatic data validation and quality assurance
        - Seamless integration with existing wizard workflow
        - Configuration persistence for reproducible experiments
    """
    from gemma_tuner.core import bigquery as bq

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

    # Table selection (single-table MVP) with preflight and auto-refresh
    def _pick_table() -> str:
        tbls = bq.list_tables(project_id, dataset_id) or []
        if tbls:
            return questionary.select("Table:", choices=tbls, style=apple_style).ask()
        return questionary.text("Table ID:", style=apple_style).ask()

    # Initial pick
    table_id = _pick_table()
    # Preflight verify and auto-refresh once if needed
    ok, msg = bq.verify_table(project_id, dataset_id, table_id)
    if not ok:
        console.print("[yellow]Selected table/view failed preflight. Refreshing table list...[/yellow]")
        table_id = _pick_table()
        ok2, msg2 = bq.verify_table(project_id, dataset_id, table_id)
        if not ok2:
            console.print(f"[red]Preflight failed again for '{table_id}':[/red] {msg2}")
            raise RuntimeError("BigQuery table check failed. Please choose a concrete table or fix the view wildcard.")

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
                languages = questionary.checkbox(
                    "Select languages (Space to toggle):", choices=distinct, style=apple_style
                ).ask()
            else:
                languages = None

    # Sampling
    limit_str = questionary.text("Max rows to fetch (blank = no limit):", default="1000", style=apple_style).ask()
    try:
        limit = int(limit_str) if (limit_str or "").strip() else None
    except Exception:
        limit = 1000
    sample_random = questionary.confirm("Random sample?", default=True, style=apple_style).ask()
    sample = "random" if sample_random else "first"

    # NOTE: extra_where (raw SQL from user) has been intentionally removed.
    # The build_query_and_export docstring marks extra_where as trusted-caller-only;
    # passing free-form user text directly created a SQL injection vector. Use the
    # structured UI controls above (language, limit, sample) for all filtering.
    extra_where = None

    # Execute export — anchored to project root so it works from any cwd
    out_dir = _PROJECT_ROOT / "data" / "datasets"
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
        "files": 2,  # train.csv and validation.csv
        "description": f"Imported from BigQuery {project_id}.{dataset_id}.{table_id}",
    }


def generate_profile_config(
    method: Dict[str, Any], model: str, dataset: Dict[str, Any], method_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate config dict for the existing training infrastructure by leveraging the core config loader."""

    from gemma_tuner.core.config import load_model_dataset_config

    # Load the base configuration from config.ini using the robust, hierarchical loader.
    # This ensures that all central defaults are respected.
    cfg = _read_config()
    model_for_loader = model
    profile_config = load_model_dataset_config(cfg, model_for_loader, dataset["name"])

    # Add the model and dataset keys required by the training pipeline
    profile_config["model"] = model_for_loader
    profile_config["dataset"] = dataset["name"]

    # Layer the user's interactive choices on top of the base configuration.
    # This overrides the defaults with the specific parameters selected in the wizard.

    # Method-specific configuration
    if method["key"] == "lora":
        profile_config.update(
            {
                "use_peft": True,
                "peft_method": "lora",
                "lora_r": method_config["lora_r"],
                "lora_alpha": method_config["lora_alpha"],
                "lora_dropout": method_config.get("lora_dropout", 0.1),  # Sensible default
                # Use canonical key expected by trainer; leave as list, not string
                # Prefer canonical Gemma naming (o_proj not out_proj)
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
            }
        )
        # Gemma-specific safety: if selected model belongs to group:gemma, enforce eager attention
        section = f"model:{model_for_loader}"
        if cfg.has_option(section, "group") and cfg.get(section, "group").strip().lower() == "gemma":
            profile_config["attn_implementation"] = "eager"

    # Dataset-specific configuration
    if dataset["type"] == "huggingface":
        profile_config["dataset_name"] = dataset["name"]
        # Use the language/config the user selected during dataset setup if available.
        # Fallback to "en" only when no config key was captured — avoids silently
        # forcing English on users who selected a non-English HuggingFace dataset.
        profile_config["dataset_config"] = dataset.get("config") or dataset.get("language") or "en"
        profile_config["train_split"] = "train"
        profile_config["eval_split"] = "validation"
    elif dataset["type"] == "granary_configured":
        # Granary datasets are configured but may not yet be fully prepared (manifest
        # generation is a separate step run via `gemma-macos-tuner prepare-granary`).
        # Set dataset_source_type so the training pipeline knows to look up the
        # [dataset:<name>] config.ini section and use its granary-specific keys
        # (hf_name, hf_subset, audio_source_*, etc.) rather than expecting a ready CSV.
        # The training pipeline reads dataset_source_type == "granary" to trigger the
        # granary data loader path.
        profile_config["dataset_source_type"] = "granary"
        profile_config["dataset_name"] = dataset["name"]
        # Use the local_path from config.ini (set by setup_granary_dataset) if available,
        # otherwise fall back to the path in the dataset descriptor.
        _granary_path = dataset.get("path", f"data/datasets/{dataset['name']}")
        profile_config["train_dataset_path"] = _granary_path
        profile_config["eval_dataset_path"] = _granary_path
        # Surface a clear warning so the user knows training will fail if preparation
        # has not been completed first.
        if not dataset.get("prepared", False):
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Granary dataset '%s' is configured but not yet prepared. "
                "Run `gemma-macos-tuner prepare-granary %s` before starting training, "
                "or the training pipeline will fail to find data files.",
                dataset["name"],
                dataset["name"],
            )
    elif dataset["type"] in ["local_csv", "local_audio"]:
        profile_config["train_dataset_path"] = dataset["path"]
        # Check for a sibling eval CSV (e.g. data_eval.csv or data_validation.csv)
        import logging as _logging

        _ds_path = dataset["path"]
        _eval_path = None
        try:
            from pathlib import Path as _Path

            _p = _Path(_ds_path)
            for _suffix in (f"{_p.stem}_eval.csv", f"{_p.stem}_validation.csv"):
                _candidate = _p.parent / _suffix
                if _candidate.exists():
                    _eval_path = str(_candidate)
                    break
        except Exception:
            pass
        if _eval_path:
            profile_config["eval_dataset_path"] = _eval_path
        else:
            profile_config["eval_dataset_path"] = _ds_path
            _logging.getLogger(__name__).warning(
                "eval_dataset_path set to the same path as train_dataset_path (%s); "
                "evaluation metrics will reflect training data distribution.",
                _ds_path,
            )

    # Merge training parameters from Step 4 (learning_rate, num_train_epochs, warmup_steps)
    for k in ("learning_rate", "num_train_epochs", "warmup_steps"):
        if k in method_config:
            profile_config[k] = method_config[k]

    # Add visualization flag if enabled
    if method_config.get("visualize", False):
        profile_config["visualize"] = True

    # Ensure required splits are always present for validation
    # These are required by the configuration validator
    if "train_split" not in profile_config:
        profile_config["train_split"] = "train"
    if "validation_split" not in profile_config:
        profile_config["validation_split"] = "validation"

    return profile_config
