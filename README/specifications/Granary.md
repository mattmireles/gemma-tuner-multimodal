# Granary Dataset Integration Product Specification

## Executive Summary

This document outlines the product specification and implementation plan for integrating the [NVIDIA Granary dataset](https://huggingface.co/datasets/nvidia/Granary) into the Whisper Fine-Tuner framework. The goal is to provide users with an optional, large-scale, multilingual data source for training high-quality Automatic Speech Recognition (ASR) and Speech Translation (AST) models. The integration is architected around a robust preparation script to handle the dataset's unique, multi-corpus structure, ensuring reliability and simplifying the core training pipeline.

### Key Capabilities
- **Large-Scale Data Access**: Leverage ~643k hours of transcribed audio across 25 languages.
- **Multi-Corpus Support**: Seamlessly handle data from YODAS, VoxPopuli, YouTube-Commons, and LibriLight.
- **Reliable Data Preparation**: An offline preparation script validates the presence of all required audio files *before* training begins, preventing mid-run failures.
- **Simplified Integration**: The prepared dataset conforms to our existing internal format, requiring no changes to the core training loops.
- **User-Friendly Workflow**: Clear documentation and configuration for downloading, preparing, and using the dataset.

### Core Value Proposition
"Unlock massive improvements in model quality by training on one of the world's largest public speech datasets, with a reliable workflow that prevents common data pipeline failures."

---

## Technical Architecture

### Key Design Decisions

1.  **Preparation Script Over Direct Streaming (Reliability First)**: We are deliberately choosing a one-time, offline preparation script instead of a direct streaming approach. The `Granary` dataset requires users to download terabytes of audio separately from the metadata manifests. A preparation script allows us to **validate the existence of all audio files upfront**, preventing training runs from failing hours or days in. This prioritizes reliability and repeatable success over the elegance of pure streaming.

2.  **Decoupled and Simple (Maintainability)**: The preparation script's output is a standardized `.csv` manifest with absolute paths. This decouples the complexity of `Granary`'s multi-corpus structure from our core training pipeline. The existing `Dataset` loaders can consume this manifest without modification, adhering to our core principle of **simplicity over complexity**.

### Data Flow Diagram
```mermaid
graph TD
    subgraph "User's Local Environment"
        A[External Audio Downloads<br/>(e.g., VoxPopuli, YTC)] --> B{Local Audio Storage};
    end

    subgraph "Hugging Face Hub"
        C[nvidia/Granary Metadata Manifests] --> D{Hugging Face Hub};
    end

    subgraph "Whisper Fine-Tuner Workflow"
        E[scripts/prepare_granary.py] --> F[data/datasets/granary_prepared.csv<br/><b>Validated & Unified Manifest</b>];
        F --> G[Standard Training Pipeline<br/>(No changes needed)];
        G --> H[Fine-Tuned Model];
    end

    D --> E;
    B --> E;

    style A fill:#ffe0b2
    style C fill:#d1c4e9
    style E fill:#c8e6c9
    style F fill:#a5d6a7
```

---

## Implementation Plan

This plan is broken down into five actionable steps for the engineering team. The implementation should be straightforward and isolated, primarily involving the creation of a new script.

### Step 1: Extend Configuration System (`core/config.py`)

The system needs to know how to find the externally downloaded audio files. We'll add new, clearly-named keys to our dataset configuration for this.

-   **Action:** Extend the configuration loader to recognize `audio_source_*` keys.
-   **File to Modify:** `core/config.py`.
-   **Details:**
    -   In the `load_profile_config` function (or equivalent), add logic to iterate through keys in a dataset section.
    -   If a key starts with `audio_source_`, parse it and add it to a dictionary called `audio_sources`. The key should be the part after the prefix (e.g., `voxpopuli`) and the value should be the path.
    -   This `audio_sources` dictionary will be part of the final configuration passed to the preparation script.
-   **Example `config.ini` Section:**
    ```ini
    [dataset:granary-en]
    source_type = granary
    hf_name = nvidia/Granary
    hf_subset = en
    local_path = data/datasets/granary-en
    # User must provide absolute paths to their downloaded audio corpora
    audio_source_voxpopuli = /path/to/downloaded/voxpopuli/audio
    audio_source_ytc = /path/to/downloaded/ytc/audio
    audio_source_librilight = /path/to/downloaded/librilight/audio
    # The 'yodas' corpus is included in the HF download, so it doesn't need a path here.
    ```

### Step 2: Create Data Preparation Script (`scripts/prepare_granary.py`)

This is the core of the implementation. This script will be robust, user-friendly, and contain all the dataset-specific logic.

-   **New File:** `scripts/prepare_granary.py`
-   **Pseudocode / Structure:**
    ```python
    import argparse
    import os
    from pathlib import Path
    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    from whisper_tuner.core.config import load_profile_config # Or equivalent

    def prepare_granary(profile_name):
        """
        Downloads Granary metadata, validates against local audio files,
        and creates a unified manifest for training.
        """
        # 1. Load Configuration
        config = load_profile_config(profile_name)
        audio_sources = config.get("audio_sources", {})
        hf_subset = config["hf_subset"]
        output_path = Path(config["local_path"])
        output_path.mkdir(parents=True, exist_ok=True)

        # 2. Download Metadata
        print(f"Downloading metadata for 'nvidia/Granary' subset: {hf_subset}...")
        metadata_ds = load_dataset("nvidia/granary", hf_subset, split="train") # Or handle multiple splits

        # 3. Validate Audio Paths and Build Manifest
        manifest_data = []
        print("Validating audio files... (This may take a while)")
        for item in tqdm(metadata_ds):
            source = item["dataset_source"]
            relative_path = item["audio_filepath"]

            if source in audio_sources:
                base_path = Path(audio_sources[source])
                # Granary paths can be inconsistent, so we need to be flexible.
                # Sometimes they include the source, sometimes not.
                # Example: "voxpopuli/fr/audio.flac"
                # We build the path by joining the user's base path with the relative path.
                absolute_path = base_path / relative_path
            elif source == "yodas":
                # YODAS audio is included in the HF download, path is often relative
                # This needs investigation on how HF datasets library handles it.
                # For now, we assume it needs a base path as well.
                # If HF caches it, we need to find that cache path.
                # Let's assume for now it also needs a user-provided path for consistency.
                pass # Logic to handle YODAS source
            
            if not absolute_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found for source '{source}': {absolute_path}\n"
                    f"Please ensure you have downloaded the '{source}' corpus and that the path in your config.ini is correct."
                )

            manifest_data.append({
                "id": item["utt_id"],
                "audio_path": str(absolute_path),
                "text": item["answer"], # 'answer' contains the transcription
                "language": item["source_lang"],
                "duration": item["duration"],
            })

        # 4. Save Unified Manifest
        manifest_df = pd.DataFrame(manifest_data)
        output_file = output_path / f"granary_{hf_subset}_prepared.csv"
        manifest_df.to_csv(output_file, index=False)
        print(f"✅ Successfully created manifest at: {output_file}")

    if __name__ == "__main__":
        # Add argparse to run from CLI
        parser = argparse.ArgumentParser()
        parser.add_argument("--profile", required=True, help="Dataset profile name from config.ini")
        args = parser.parse_args()
        prepare_granary(args.profile)
    ```

### Step 3: Minimal Dataset Loader Integration (`utils/dataset_utils.py`)

No changes should be necessary here, which is the entire point of our decoupled approach. This step is for verification.

-   **Action:** Verify that the existing `load_dataset_split` function can load the manifest created by `prepare_granary.py` without modification.
-   **Verification:** A unit test should be added that runs `prepare_granary` on a tiny, mocked dataset and then feeds the resulting manifest path to `load_dataset_split` to ensure it loads correctly.

### Step 4: Add CLI and Wizard Hooks (`main.py`, `wizard.py`)

-   **File to Modify:** `main.py`
    -   Add a new command `prepare-granary` that calls the `prepare_granary.py` script. This keeps our main entry point consistent.
-   **File to Modify:** `wizard.py`
    -   In the dataset selection step, add an option like "Setup a new large-scale dataset (e.g., Granary)...".
    -   If selected, the wizard should print a clear, multi-line message explaining the manual steps required (download audio, edit `config.ini`) and then offer to run the `prepare-granary` command for the user once they've done so.

### Step 5: High-Quality Documentation (`README/specifications/Datasets.md`)

This is critical for user success.

-   **Action**: Create a new, detailed section in the dataset documentation.
-   **Content Checklist:**
    -   [ ] Overview of the Granary dataset and its value.
    -   [ ] **Explicit links** to the download pages for each required corpus (VoxPopuli, YTC, etc.).
    -   [ ] A copy-pasteable `config.ini` template section.
    -   [ ] A clear, numbered list of steps for the user to follow.
    -   [ ] Example CLI commands for preparation and subsequent training.

---

## Definition of Done

The integration will be considered complete when an engineer can successfully:
1.  Configure a new `[dataset:granary-en]` section in `config.ini` with local paths to downloaded audio.
2.  Run `whisper-tuner prepare-granary granary-en` and have it successfully generate a `_prepared.csv` file without errors.
3.  Launch a training run using a profile that points to this new dataset, and have the training start successfully, loading data from the prepared manifest.
4.  All new code is accompanied by relevant unit tests and documentation.

## Future Enhancements (Out of Scope for V1)

-   **Automated Downloaders**: Scripts to optionally automate the download and extraction of the external audio corpora.
-   **Streaming Validation**: A "validation-only" streaming mode that could check for file existence without creating a full manifest, for quicker checks.
-   **Partial Preparation**: Allow preparation of only a subset of the data (e.g., `--max_samples`) for faster iteration.ok, 
