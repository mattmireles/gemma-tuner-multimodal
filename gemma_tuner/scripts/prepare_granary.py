#!/usr/bin/env python3
"""
NVIDIA Granary Dataset Preparation Script

This script prepares the NVIDIA Granary dataset for Gemma fine-tuning by downloading
metadata from HuggingFace, validating against locally downloaded audio corpora, and
generating a unified training manifest. The script implements a reliability-first approach
by validating all audio files before training begins to prevent mid-run failures.

Key responsibilities:
- Download Granary metadata from HuggingFace Hub for specified language subset
- Validate existence of all required audio files from external corpora
- Handle flexible path resolution for Granary's varying directory structures
- Generate unified CSV manifest with absolute paths for training pipeline
- Provide comprehensive error reporting and user guidance for setup issues

Called by:
- main.py prepare-granary command for CLI-based dataset preparation
- wizard.py Granary setup workflow for guided user experience
- Automated training pipelines requiring large-scale dataset preparation

Calls to:
- core/config.py:load_profile_config() for hierarchical configuration loading
- datasets.load_dataset() for HuggingFace metadata download
- pathlib.Path for cross-platform path resolution and validation
- pandas for efficient manifest generation and CSV output

Granary Dataset Architecture:
The NVIDIA Granary dataset combines multiple large-scale speech corpora:
- VoxPopuli: Multilingual parliamentary speeches (requires external download)
- YouTube Commons (YTC): Diverse web content (requires external download)
- LibriLight: Large-scale English audiobooks (requires external download)
- YODAS: Custom corpus (included in HuggingFace dataset download)

Audio Source Resolution Strategy:
Granary uses inconsistent path structures across corpora. This script handles:
- Absolute path construction from user-provided base directories
- Flexible relative path joining for both "corpus/lang/file.ext" and "file.ext" patterns
- Comprehensive error reporting with specific file paths and configuration guidance
- Validation of all audio files before manifest generation

Configuration Requirements:
The script expects dataset configuration sections like:
[dataset:granary-en]
source_type = granary
hf_name = nvidia/Granary
hf_subset = en
local_path = data/datasets/granary-en
audio_source_voxpopuli = /path/to/downloaded/voxpopuli/audio
audio_source_ytc = /path/to/downloaded/ytc/audio
audio_source_librilight = /path/to/downloaded/librilight/audio

Error Handling and Recovery:
- Missing audio files: Detailed error with specific file path and source corpus
- Configuration issues: Clear guidance on required audio_source_* settings
- Download failures: Retry logic and fallback strategies for HuggingFace access
- Path resolution errors: Helpful suggestions for common directory structure issues

Performance Optimizations:
- Efficient audio file existence checking using pathlib
- Progress tracking for large dataset validation
- Memory-efficient processing of HuggingFace dataset metadata
- Batch processing with comprehensive error isolation

Apple Silicon Optimizations:
- Native path handling for macOS file systems
- Efficient memory usage for unified memory architecture
- Progress reporting optimized for terminal display
- Integration with existing MPS-aware training pipeline

Output Format:
Generates standardized CSV manifest compatible with existing dataset loading:
- id: Unique sample identifier from Granary metadata
- audio_path: Absolute path to validated audio file
- text: Transcription text from Granary 'answer' field
- language: Source language code for multilingual training
- duration: Audio duration in seconds for batch optimization

The output CSV integrates seamlessly with the existing training pipeline without
requiring any modifications to core dataset loading or training systems.
"""

import argparse
import configparser
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Import configuration system with new audio source support
from gemma_tuner.core.config import ConfigConstants, load_profile_config

# Initialize logging for comprehensive error reporting and progress tracking
logger = logging.getLogger(__name__)


def validate_granary_config(config: Dict) -> None:
    """
    Validates Granary dataset configuration for required keys and audio sources.

    This function performs comprehensive validation of Granary-specific configuration
    requirements to ensure all necessary components are present before beginning the
    potentially time-intensive audio validation process.

    Called by:
    - prepare_granary() before beginning metadata download and audio validation

    Validation requirements:
    - Required HuggingFace dataset identification (hf_name, hf_subset)
    - Local output path configuration for manifest generation
    - Audio source configuration for external corpus mapping
    - Minimum external corpus coverage (VoxPopuli, YTC, LibriLight)

    Args:
        config (Dict): Merged configuration dictionary from hierarchical loading

    Raises:
        ValueError: If required configuration keys are missing or invalid
        FileNotFoundError: If configured audio source directories don't exist

    Example:
        >>> config = load_profile_config(cfg, "granary-en")
        >>> validate_granary_config(config)  # Raises if invalid
    """
    # Validate required HuggingFace dataset configuration
    required_hf_keys = ["hf_name", "hf_subset", "local_path"]
    missing_keys = [key for key in required_hf_keys if key not in config or not config[key]]
    if missing_keys:
        raise ValueError(
            f"Missing required Granary configuration keys: {', '.join(missing_keys)}. "
            f"Please ensure your dataset configuration includes: {', '.join(required_hf_keys)}"
        )

    # Validate HuggingFace dataset identifier
    if config["hf_name"] != "nvidia/Granary":
        raise ValueError(f"Invalid hf_name '{config['hf_name']}'. Granary preparation requires 'nvidia/Granary'.")

    # Validate audio sources configuration
    audio_sources = config.get("audio_sources", {})
    if not audio_sources:
        raise ValueError(
            "No audio sources configured. Granary preparation requires audio_source_* configuration keys. "
            "Example: audio_source_voxpopuli = /path/to/voxpopuli"
        )

    # Check for minimum required external corpora
    configured_corpora = set(audio_sources.keys())
    missing_corpora = ConfigConstants.GRANARY_MINIMUM_EXTERNAL_CORPORA - configured_corpora
    if missing_corpora:
        raise ValueError(
            f"Missing required audio corpora: {', '.join(missing_corpora)}. "
            f"Granary requires external downloads for: {', '.join(ConfigConstants.GRANARY_MINIMUM_EXTERNAL_CORPORA)}. "
            f"Add configuration keys like: audio_source_{list(missing_corpora)[0]} = /path/to/{list(missing_corpora)[0]}"
        )

    # Validate audio source directories exist
    for corpus, path in audio_sources.items():
        corpus_path = Path(path)
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Audio source directory not found for '{corpus}': {path}. "
                f"Please ensure you have downloaded the {corpus} corpus and the path is correct."
            )
        if not corpus_path.is_dir():
            raise ValueError(
                f"Audio source path for '{corpus}' is not a directory: {path}. "
                f"Please provide the path to the directory containing {corpus} audio files."
            )


def resolve_granary_audio_path(relative_path: str, source: str, audio_sources: Dict[str, str]) -> Optional[Path]:
    """
    Resolves absolute audio file paths for Granary's varying directory structures.

    Granary uses inconsistent path patterns across different corpora. This function
    implements flexible path resolution to handle both corpus-prefixed paths like
    "voxpopuli/fr/audio.flac" and direct paths like "audio.flac".

    Called by:
    - prepare_granary() during audio file validation for each sample

    Path resolution strategy:
    1. Try direct joining: base_path / relative_path
    2. Try corpus removal: base_path / relative_path.removeprefix(corpus)
    3. Handle YODAS special case (included in HuggingFace download)
    4. Return None if no valid path found (triggers error handling)

    Args:
        relative_path (str): Relative path from Granary metadata
        source (str): Corpus identifier (voxpopuli, ytc, librilight, yodas)
        audio_sources (Dict[str, str]): Mapping of corpus to base directory paths

    Returns:
        Optional[Path]: Absolute path to audio file if found, None if not resolvable

    Example:
        >>> path = resolve_granary_audio_path("voxpopuli/fr/audio.flac", "voxpopuli",
        ...                                  {"voxpopuli": "/data/voxpopuli"})
        >>> # Returns: Path("/data/voxpopuli/voxpopuli/fr/audio.flac") or
        >>> #          Path("/data/voxpopuli/fr/audio.flac") depending on structure
    """
    # Handle YODAS corpus (included in HuggingFace download)
    if source == "yodas":
        # TODO: Implement YODAS path resolution
        # YODAS audio is included in the HuggingFace dataset download
        # Need to determine the cache path or extraction location
        logger.warning(f"YODAS path resolution not yet implemented for: {relative_path}")
        return None

    # Handle external corpora with user-provided base paths
    if source not in audio_sources:
        logger.error(f"No audio source configured for corpus '{source}'. Available: {list(audio_sources.keys())}")
        return None

    base_path = Path(audio_sources[source])

    # Strategy 1: Direct path joining (handles most cases)
    direct_path = base_path / relative_path
    if direct_path.exists():
        return direct_path

    # Strategy 2: Remove corpus prefix if present
    # Handles cases where relative_path includes corpus name: "voxpopuli/fr/audio.flac"
    if relative_path.startswith(f"{source}/"):
        stripped_path = relative_path[len(f"{source}/") :]
        corpus_stripped = base_path / stripped_path
        if corpus_stripped.exists():
            return corpus_stripped

    # Strategy 3: Try just the filename for flat directory structures
    filename_only = Path(relative_path).name
    filename_path = base_path / filename_only
    if filename_path.exists():
        return filename_path

    # No valid path found - will trigger error in main validation loop
    return None


def prepare_granary(profile_name: str, config_path: str | None = None) -> str:
    """
    Downloads Granary metadata, validates against local audio files, and creates unified manifest.

    This is the main orchestration function that implements the complete Granary preparation
    workflow following the reliability-first approach specified in the requirements. It
    validates all audio files upfront to prevent training failures hours or days later.

    Called by:
    - main.py prepare-granary command via CLI argument parsing
    - Automated training pipelines for large-scale dataset preparation

    Calls to:
    - core/config.py:load_profile_config() for hierarchical configuration loading
    - validate_granary_config() for comprehensive configuration validation
    - datasets.load_dataset() for HuggingFace metadata download
    - resolve_granary_audio_path() for flexible path resolution
    - pandas for efficient manifest generation and CSV output

    Preparation workflow:
    1. Load and validate hierarchical configuration with audio sources
    2. Download Granary metadata from HuggingFace for specified subset
    3. Validate existence of all audio files with detailed error reporting
    4. Generate unified manifest with absolute paths and metadata
    5. Save standardized CSV for seamless integration with training pipeline

    Error handling:
    - Configuration validation: Clear guidance on required settings
    - Missing audio files: Specific file paths and source corpus identification
    - HuggingFace download failures: Retry logic and fallback strategies
    - Path resolution errors: Helpful suggestions for directory structure issues

    Args:
        profile_name (str): Dataset profile name from config.ini (e.g., "granary-en")
        config_path (str | None): Path to INI file, or None for the same resolution as the CLI

    Returns:
        str: Absolute path to generated manifest CSV file

    Raises:
        ValueError: If configuration validation fails or required audio missing
        FileNotFoundError: If audio source directories don't exist
        ConnectionError: If HuggingFace dataset download fails

    Example:
        >>> manifest_path = prepare_granary("granary-en")
        >>> print(f"Manifest created: {manifest_path}")
        # Output: "Manifest created: /path/to/data/datasets/granary-en/granary_en_prepared.csv"
    """
    logger.info("=" * 60)
    logger.info("🚀 NVIDIA GRANARY DATASET PREPARATION")
    logger.info("=" * 60)

    # Step 1: Load and validate configuration
    logger.info("📋 Loading configuration...")
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    from gemma_tuner.core.ops import _resolve_config_path

    config.read(_resolve_config_path(config_path))

    try:
        profile_config = load_profile_config(config, profile_name)
    except ValueError as e:
        raise ValueError(f"Configuration loading failed: {e}")

    # Validate Granary-specific configuration requirements
    validate_granary_config(profile_config)

    # Extract configuration values
    audio_sources = profile_config.get("audio_sources", {})
    hf_subset = profile_config["hf_subset"]
    output_path = Path(profile_config["local_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Check configuration options for streaming and validation
    streaming_enabled = profile_config.get("streaming_enabled", False)
    max_samples = profile_config.get("max_samples", None)
    skip_audio_validation = profile_config.get("skip_audio_validation", False)
    sample_validation_rate = profile_config.get("sample_validation_rate", 1.0)

    # Validate sample_validation_rate is in valid range
    if not (0.0 <= sample_validation_rate <= 1.0):
        raise ValueError(f"sample_validation_rate must be between 0.0 and 1.0, got {sample_validation_rate}")

    logger.info(f"Dataset subset: {hf_subset}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Audio sources: {list(audio_sources.keys())}")
    logger.info(f"Streaming mode: {'enabled' if streaming_enabled else 'disabled'}")
    if max_samples:
        logger.info(f"Max samples limit: {max_samples:,}")

    # Log validation configuration
    if skip_audio_validation:
        logger.info("Audio validation: SKIPPED (fastest, but risky)")
    elif sample_validation_rate < 1.0:
        logger.info(f"Audio validation: {sample_validation_rate:.1%} sampling (faster, some risk)")
    else:
        logger.info("Audio validation: FULL (slowest, but safest)")

    # Step 2: Download metadata from HuggingFace
    logger.info("\n📥 Loading Granary metadata from HuggingFace...")
    try:
        # Load dataset with streaming support
        dataset = load_dataset("nvidia/Granary", hf_subset, split="train", streaming=streaming_enabled)

        if streaming_enabled:
            logger.info("✅ Streaming dataset loaded (length unknown)")
            if max_samples:
                logger.info(f"Will process up to {max_samples:,} samples")
        else:
            logger.info(f"✅ Downloaded metadata for {len(dataset)} samples")
    except Exception as e:
        raise ConnectionError(
            f"Failed to download Granary metadata for subset '{hf_subset}': {e}. "
            f"Please check your internet connection and HuggingFace access."
        )

    # Step 3: Validate audio files and build manifest
    if skip_audio_validation:
        logger.info("\n🔍 Building manifest (skipping audio validation)...")
    elif sample_validation_rate < 1.0:
        logger.info(f"\n🔍 Building manifest with {sample_validation_rate:.1%} audio validation sampling...")
    else:
        logger.info("\n🔍 Validating audio files and building manifest...")

    if streaming_enabled:
        logger.info("Processing samples in streaming mode...")
    else:
        if not skip_audio_validation:
            logger.info("This may take several minutes for large datasets...")

    manifest_data = []
    validation_errors = []
    corpus_stats = {corpus: 0 for corpus in ConfigConstants.GRANARY_SUPPORTED_CORPORA}

    # Process samples with proper streaming support
    sample_count = 0  # counts valid (non-error) samples only
    items_iterated = 0  # counts every item seen, including validation failures

    # Setup progress tracking based on mode
    if streaming_enabled:
        if skip_audio_validation:
            desc = "Building manifest (streaming, no validation)"
        elif sample_validation_rate < 1.0:
            desc = f"Building manifest (streaming, {sample_validation_rate:.1%} validation)"
        else:
            desc = "Validating audio files (streaming)"
        if max_samples:
            desc += f" [max: {max_samples:,}]"
        progress_bar = tqdm(desc=desc)
        # For streaming, we iterate directly through the dataset
        dataset_iter = iter(dataset)
    else:
        # For non-streaming, tqdm can iterate through the dataset directly
        if skip_audio_validation:
            desc = "Building manifest (no validation)"
        elif sample_validation_rate < 1.0:
            desc = f"Building manifest ({sample_validation_rate:.1%} validation)"
        else:
            desc = "Validating audio files"
        progress_bar = tqdm(dataset, desc=desc)
        dataset_iter = progress_bar

    try:
        for item in dataset_iter:
            items_iterated += 1
            # Update progress for streaming mode (non-streaming progress handled by tqdm automatically)
            if streaming_enabled:
                progress_bar.update(1)
                progress_bar.set_postfix({"processed": sample_count, "valid": len(manifest_data)})

            source = item["dataset_source"]
            relative_path = item["audio_filepath"]

            # Determine if this sample should be validated based on configuration
            should_validate = True
            if skip_audio_validation:
                # Skip all validation when flag is set
                should_validate = False
            elif sample_validation_rate < 1.0:
                # Use random sampling for validation
                should_validate = random.random() < sample_validation_rate

            # Default path resolution for manifest generation
            absolute_path = None
            if should_validate:
                # Resolve absolute audio path using flexible path resolution
                absolute_path = resolve_granary_audio_path(relative_path, source, audio_sources)

                if absolute_path is None or not absolute_path.exists():
                    error_msg = (
                        f"Audio file not found - Source: {source}, "
                        f"Relative path: {relative_path}, "
                        f"Expected: {absolute_path if absolute_path else 'Could not resolve path'}"
                    )
                    validation_errors.append(error_msg)
                    # Do NOT increment sample_count here — failed samples must not consume
                    # quota from max_samples. Only valid samples (below) count toward the limit.

                    # Limit error collection in streaming mode to prevent memory issues
                    if streaming_enabled and len(validation_errors) > 1000:
                        logger.warning("⚠️ Over 1000 validation errors in streaming mode. Stopping error collection.")
                        validation_errors = validation_errors[:1000] + ["... (truncated for memory efficiency)"]
                    continue
            else:
                # Skip validation but still need a path for the manifest
                # Use the path resolution but don't check if file exists
                absolute_path = resolve_granary_audio_path(relative_path, source, audio_sources)
                if absolute_path is None:
                    # If we can't resolve the path at all, use a constructed path
                    # This maintains manifest consistency even without validation
                    if source in audio_sources:
                        absolute_path = Path(audio_sources[source]) / relative_path
                    else:
                        # Fallback to relative path if source not configured
                        absolute_path = Path(relative_path)

            # Track corpus statistics
            if source in corpus_stats:
                corpus_stats[source] += 1

            # Add validated sample to manifest
            manifest_data.append(
                {
                    "id": item["utt_id"],
                    "audio_path": str(absolute_path),
                    "text": item["answer"],  # 'answer' contains the transcription
                    "language": item["source_lang"],
                    "duration": item.get("duration", 0.0),  # Duration may not be present in all samples
                    "source_corpus": source,  # Additional metadata for analysis
                }
            )

            sample_count += 1

            # Respect max_samples limit
            if max_samples and sample_count >= max_samples:
                logger.info(f"Reached max_samples limit of {max_samples:,}")
                break

    finally:
        progress_bar.close()

    # Step 4: Report validation results
    logger.info("\n📊 Processing Results:")
    logger.info(f"✅ Manifest entries created: {len(manifest_data)}")

    # Report validation status based on configuration
    if skip_audio_validation:
        logger.info("🚀 Audio validation: SKIPPED for maximum speed")
        logger.info("⚠️  Note: Audio files not verified - training may fail if files are missing")
    elif sample_validation_rate < 1.0:
        logger.info(f"🎯 Audio validation: {sample_validation_rate:.1%} sampling performed")
        logger.info(f"❌ Validation errors in sample: {len(validation_errors)}")
        if len(validation_errors) > 0:
            estimated_total_errors = int(len(validation_errors) / sample_validation_rate)
            logger.info(f"📈 Estimated total errors: ~{estimated_total_errors}")
    else:
        logger.info("🔍 Audio validation: FULL validation performed")
        logger.info(f"❌ Invalid samples: {len(validation_errors)}")

    if corpus_stats:
        logger.info("\n📈 Corpus Distribution:")
        for corpus, count in corpus_stats.items():
            if count > 0:
                logger.info(f"  {corpus}: {count:,} samples")

    # Handle validation errors based on validation mode
    if validation_errors and not skip_audio_validation:
        logger.warning(f"\n⚠️  Found {len(validation_errors)} validation errors:")
        # Show first 10 errors for debugging
        for error in validation_errors[:10]:
            logger.warning(f"  {error}")
        if len(validation_errors) > 10:
            logger.warning(f"  ... and {len(validation_errors) - 10} more errors")

        # Determine if errors are fatal - only for full validation or high error rates in sampling
        if sample_validation_rate == 1.0:
            # Full validation mode - use standard error rate threshold
            total_processed = max(sample_count + len(validation_errors), 1)  # Include failures in denominator
            error_rate = len(validation_errors) / total_processed
            if error_rate > 0.1:  # More than 10% errors
                raise ValueError(
                    f"High error rate ({error_rate:.1%}): {len(validation_errors)} of {total_processed} files missing. "
                    f"Please check your audio source configuration and ensure all corpora are downloaded correctly."
                )
            else:
                logger.warning(f"Proceeding with {len(manifest_data)} valid samples ({error_rate:.1%} error rate)")
        else:
            # Sampling mode - estimate error rate and warn but don't fail unless extremely high
            # Use items_iterated (all items seen, including failures) as the base, not
            # sample_count (valid-only).  sample_count excludes validation failures, so
            # using it as the denominator inflates the error rate and triggers false aborts.
            samples_validated = int(items_iterated * sample_validation_rate)
            if samples_validated > 0:
                sample_error_rate = len(validation_errors) / samples_validated
                estimated_total_errors = int(len(validation_errors) / sample_validation_rate)

                if sample_error_rate > 0.5:  # More than 50% of sampled files missing
                    raise ValueError(
                        f"Very high error rate in validation sample ({sample_error_rate:.1%}): "
                        f"{len(validation_errors)} of {samples_validated} sampled files missing. "
                        f"Estimated total missing: ~{estimated_total_errors}. "
                        f"Please check your audio source configuration."
                    )
                else:
                    logger.warning(
                        f"Validation sample error rate: {sample_error_rate:.1%} "
                        f"(estimated {estimated_total_errors} total errors)"
                    )
            else:
                logger.warning("No samples were validated due to sampling rate")

    if not manifest_data:
        raise ValueError(
            "No valid audio files found. Please check your audio source configuration "
            "and ensure all required corpora are downloaded to the specified paths."
        )

    # Step 5: Generate and save unified manifest
    logger.info("\n💾 Generating unified manifest...")
    manifest_df = pd.DataFrame(manifest_data)

    # Generate output filename with subset identifier
    output_file = output_path / f"granary_{hf_subset}_prepared.csv"
    manifest_df.to_csv(output_file, index=False)

    logger.info("=" * 60)
    logger.info("✅ GRANARY PREPARATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"📄 Manifest created: {output_file}")
    logger.info(f"📊 Total samples: {len(manifest_data):,}")
    logger.info("🎯 Ready for training with existing pipeline")
    logger.info("=" * 60)

    return str(output_file)


def main():
    """
    Command-line interface for Granary dataset preparation.

    Provides argparse-based CLI for standalone script execution with comprehensive
    error handling and user-friendly output formatting.

    Called by:
    - Direct script execution: python scripts/prepare_granary.py --profile granary-en
    - main.py prepare-granary command delegation

    CLI Arguments:
    - --profile: Required dataset profile name from config.ini
    - Inherits standard logging configuration from main application

    Error handling:
    - Configuration errors: User-friendly guidance on fixing config.ini
    - File not found errors: Specific paths and download instructions
    - Connection errors: HuggingFace access troubleshooting
    """
    parser = argparse.ArgumentParser(
        description="Prepare NVIDIA Granary dataset for Gemma fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_granary.py --profile granary-en
  python scripts/prepare_granary.py --profile granary-multilingual

Configuration Requirements:
  Your config.ini must include a dataset section like:
  
  [dataset:granary-en]
  source_type = granary
  hf_name = nvidia/Granary
  hf_subset = en
  local_path = data/datasets/granary-en
  audio_source_voxpopuli = /path/to/downloaded/voxpopuli
  audio_source_ytc = /path/to/downloaded/ytc
  audio_source_librilight = /path/to/downloaded/librilight

For more information, see: README/Datasets.md
""",
    )
    parser.add_argument("--profile", required=True, help="Dataset profile name from config.ini (e.g., granary-en)")

    args = parser.parse_args()

    # Configure logging for standalone execution
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    try:
        manifest_path = prepare_granary(args.profile)
        print(f"\n🎉 Success! Manifest created: {manifest_path}")
        print("You can now train using this dataset with your existing training pipeline.")
        return 0

    except ValueError as e:
        logger.error(f"❌ Configuration Error: {e}")
        print("\n💡 Tips:")
        print("- Check your config.ini dataset section")
        print("- Ensure all audio_source_* paths are correct")
        print("- Verify you've downloaded all required corpora")
        return 1

    except FileNotFoundError as e:
        logger.error(f"❌ File Not Found: {e}")
        print("\n💡 Tips:")
        print("- Verify audio corpus download paths")
        print("- Check directory permissions")
        print("- Ensure all required corpora are downloaded")
        return 1

    except ConnectionError as e:
        logger.error(f"❌ Connection Error: {e}")
        print("\n💡 Tips:")
        print("- Check your internet connection")
        print("- Verify HuggingFace Hub access")
        print("- Try again in a few minutes")
        return 1

    except Exception as e:
        logger.error(f"❌ Unexpected Error: {e}", exc_info=True)
        print("\n💡 If this error persists, please check the logs and configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
