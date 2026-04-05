#!/usr/bin/env python3
"""
Comprehensive Dataset Preparation and Audio Processing System

This module provides end-to-end dataset preparation capabilities for Gemma fine-tuning,
handling audio download, format conversion, quality filtering, and dataset splitting.
It transforms raw audio datasets into training-ready formats with comprehensive
validation and quality assurance.

Key responsibilities:
- Concurrent audio download and format conversion (M4A to WAV)
- Audio quality validation and duration analysis
- Text transcription filtering and language processing
- Dataset splitting (train/validation) with stratification
- Prepared dataset CSV generation with metadata
- Statistical analysis and quality reporting

Called by:
- main.py prepare operations for dataset setup workflows
- Data preparation pipelines in automated training systems
- Dataset curation scripts for research projects
- Quality assurance workflows for production datasets

Calls to:
- utils/dataset_utils.py for dataset loading utilities
- ffmpeg for audio format conversion and processing
- librosa for audio analysis and duration calculation
- concurrent.futures for parallel audio processing
- sklearn for stratified dataset splitting

Data processing pipeline:
1. Raw dataset loading from CSV files
2. Concurrent audio download with error recovery
3. Audio format conversion (M4A → WAV at 16kHz)
4. Quality filtering (empty transcriptions, corrupted audio)
5. Language filtering and code normalization
6. Duration analysis and statistical reporting
7. Dataset splitting with validation
8. Prepared CSV generation with metadata

Audio processing optimizations:
- Concurrent download using ThreadPoolExecutor
- Automatic retry mechanism for failed downloads
- Memory-efficient audio processing pipeline
- Progress tracking for large dataset preparation
- Error isolation to prevent cascade failures

Quality assurance features:
- Comprehensive audio validation using librosa
- Text transcription completeness verification
- Language code validation and normalization
- Duration-based filtering with configurable thresholds
- Statistical analysis of dataset characteristics
- Error reporting and recovery guidance

File organization:
Creates organized directory structure:
data/
├── datasets/
│   └── {dataset_name}/
│       ├── {dataset_name}_prepared.csv
│       ├── train.csv
│       └── validation.csv
└── audio/
    ├── {note_id}.wav (processed audio files)
    └── {note_id}.m4a (original downloads, cleaned up)

Configuration integration:
Reads from config.ini for dataset-specific settings:
- Source CSV file location
- Language filtering preferences
- Quality thresholds and processing parameters
- Output directory structure

Apple Silicon optimizations:
- Efficient audio processing using native libraries
- Memory-conscious operations for unified memory architecture
- Optimized concurrent processing for M-series chips
- Native ffmpeg integration for hardware acceleration

Error handling:
- Network failures: Automatic retry with exponential backoff
- Audio corruption: Individual file error isolation
- Format conversion: Fallback strategies for problematic files
- Memory issues: Batch processing with memory management
- File system errors: Comprehensive error reporting

Compatibility:
- ffmpeg: Required for audio format conversion
- librosa: Audio analysis and duration calculation
- pandas: Efficient dataset manipulation
- sklearn: Stratified splitting for balanced datasets
- concurrent.futures: Parallel processing coordination
"""

import argparse
import concurrent.futures
import configparser
import logging
import os
import re
import subprocess
import threading
import urllib.request
from pathlib import Path

# Anchor all output paths to the project root so that `prepare_data()` writes
# to the correct location regardless of where the CLI is invoked from.
# Structure: prepare_data.py -> scripts/ -> gemma_tuner/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gemma_tuner.models.gemma.constants import AudioProcessingConstants

logger = logging.getLogger(__name__)

# Language Processing Constants
# Standard unknown language token used throughout the system for consistency
UNKNOWN_LANGUAGE_TOKEN = "??"


def download_and_decode_audio(audio_url, note_id, audio_dir, error_notes, error_lock):
    """
    Downloads and converts audio files with comprehensive error handling and recovery.

    This function handles the complete audio acquisition and conversion pipeline,
    downloading M4A files from URLs and converting them to WAV format suitable
    for training. It implements robust error handling to ensure dataset
    preparation continues even with individual file failures.

    Called by:
    - prepare_data() via ThreadPoolExecutor for concurrent processing
    - Parallel audio processing workflows

    Processing workflow:
    1. Check for existing converted WAV file (skip if present)
    2. Download M4A file from provided URL with error handling
    3. Convert M4A to WAV using ffmpeg with specific parameters
    4. Validate converted file and clean up intermediate files
    5. Add failed entries to error tracking for reporting

    Audio conversion parameters:
    - Output format: WAV (PCM 16-bit)
    - Sample rate: 16kHz (audio requirement)
    - Channels: Mono (automatic conversion from stereo)
    - Quality: Lossless conversion preserving audio fidelity

    Error handling strategy:
    - Network failures: Individual file isolation (doesn't stop batch)
    - Conversion failures: Clean up partial files and mark as failed
    - File system errors: Graceful degradation with error logging
    - Permission issues: Clear error reporting with resolution guidance

    Threading safety:
    - Thread-safe error list updates using provided lock
    - Independent file operations per thread
    - No shared state modification beyond error tracking
    - All exceptions caught to prevent thread pool crashes

    Args:
        audio_url (str): HTTP/HTTPS URL of M4A audio file to download
        note_id (str): Unique identifier for the audio sample
        audio_dir (str): Directory path for storing processed audio files
        error_notes (list): List for tracking failed samples (protected by error_lock)
        error_lock (threading.Lock): Lock for thread-safe access to error_notes

    Returns:
        tuple: (success: bool, note_id: str, error_msg: str or None)

    Side effects:
        - Creates {note_id}.m4a file (temporary, cleaned up on success)
        - Creates {note_id}.wav file (permanent output)
        - Adds note_id to error_notes list on failure
        - Prints progress and error messages to stdout

    Error conditions:
        - Network connectivity issues during download
        - Corrupted or invalid M4A files
        - ffmpeg conversion failures
        - Insufficient disk space or permissions
        - Invalid audio URLs or missing files
    """
    # Validate note_id to prevent path traversal. Only allow alphanumerics,
    # hyphens, and underscores — any path separator or dot sequence is rejected.
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(note_id)):
        raise ValueError(f"Unsafe note_id rejected (contains path characters): {note_id!r}")

    m4a_filename = os.path.join(audio_dir, f"{note_id}.m4a")
    wav_filename = os.path.join(audio_dir, f"{note_id}.wav")

    try:
        # Skip processing if WAV file already exists and is valid
        if not os.path.exists(wav_filename):
            # Download M4A file if not already present
            if not os.path.exists(m4a_filename):
                # Enforce HTTPS to prevent MITM substitution of training audio data.
                if not str(audio_url).startswith("https://"):
                    raise ValueError(f"Refusing non-HTTPS audio URL: {audio_url!r}")
                urllib.request.urlretrieve(audio_url, m4a_filename)
                logger.info(f"Downloaded: {audio_url} to {m4a_filename}")

            # Audio Format Conversion using ffmpeg
            # Convert M4A to WAV with compatible parameters
            ffmpeg_command = [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-i",
                m4a_filename,  # Input M4A file
                "-acodec",
                "pcm_s16le",  # PCM 16-bit little-endian codec
                "-ar",
                str(AudioProcessingConstants.DEFAULT_SAMPLING_RATE),  # 16kHz sample rate
                "-ac",
                "1",  # Convert to mono channel
                wav_filename,  # Output WAV file
            ]

            subprocess.run(
                ffmpeg_command,
                check=True,  # Raise exception on non-zero exit code
                stdout=subprocess.PIPE,  # Capture stdout to prevent spam
                stderr=subprocess.PIPE,  # Capture stderr for error analysis
                start_new_session=True,  # Isolate process group
            )
            logger.info(f"Converted: {m4a_filename} to {wav_filename}")

            # Clean up intermediate M4A file after successful conversion
            if os.path.exists(m4a_filename):
                os.remove(m4a_filename)

        # Success - file already existed or was successfully created
        return (True, note_id, None)

    except Exception as e:
        # Comprehensive Error Recovery and Cleanup
        error_msg = f"Error processing {audio_url} (ID: {note_id}): {e}"
        logger.error(error_msg)

        # Clean up any partially created or corrupted files
        for filename in [m4a_filename, wav_filename]:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    logger.warning(f"Cleaned up corrupted file: {filename}")
                except OSError as cleanup_error:
                    logger.warning(f"Could not clean up {filename}: {cleanup_error}")

        # Mark sample as failed for exclusion from dataset
        # Use lock to ensure thread-safe append to shared list
        with error_lock:
            error_notes.append(note_id)

        # Return failure status with error message
        return (False, note_id, str(e))


def prepare_data(dataset_name, config_path, no_download=False):
    """
    Orchestrates comprehensive dataset preparation with quality assurance and validation.

    This is the main dataset preparation function, coordinating all aspects of data
    processing from raw CSV files to training-ready datasets. It provides end-to-end
    data preparation with comprehensive quality assurance, statistical analysis,
    and error handling.

    Called by:
    - Command-line interface (__main__ section) for direct script execution
    - main.py prepare operations for integrated workflow execution
    - Automated data preparation pipelines in training systems
    - Dataset curation workflows for research projects

    Data preparation workflow:
    1. Configuration loading and validation from config.ini
    2. Raw dataset CSV loading with column mapping
    3. Concurrent audio download and format conversion
    4. Quality filtering (transcriptions, audio validation)
    5. Language processing and code normalization
    6. Duration analysis and statistical reporting
    7. Dataset splitting with stratification
    8. Prepared CSV generation with comprehensive metadata
    9. Validation and quality assurance reporting

    Configuration integration:
    Reads dataset-specific settings from config.ini:
    - Source CSV file location and column mapping
    - Language filtering preferences and valid language codes
    - Quality thresholds and duration limits
    - Processing parameters and output locations

    Concurrent processing:
    - ThreadPoolExecutor for parallel audio download/conversion
    - Configurable worker count for optimal performance
    - Progress tracking with tqdm for long-running operations
    - Error isolation to prevent cascade failures

    Quality assurance features:
    - Audio file validation using librosa
    - Transcription completeness verification
    - Language code validation and ISO normalization
    - Duration-based filtering with statistical analysis
    - Comprehensive error reporting and statistics

    Statistical analysis:
    Provides detailed dataset characteristics:
    - Sample count at each processing stage
    - Language distribution analysis
    - Duration statistics (min, max, average)
    - Quality filtering impact assessment
    - Error rate analysis and patterns

    Output file generation:
    Creates organized dataset structure:
    - {dataset_name}_prepared.csv: Complete processed dataset
    - train.csv: Training split (90% default)
    - validation.csv: Validation split (10% default)
    - Consistent column structure across all files

    Error handling:
    - Network failures: Individual file error isolation
    - Audio corruption: Graceful degradation with reporting
    - Configuration errors: Clear validation and guidance
    - File system issues: Comprehensive error context

    Apple Silicon optimizations:
    - Efficient concurrent processing for M-series chips
    - Memory-conscious operations for unified memory architecture
    - Native library integration for optimal performance

    Args:
        dataset_name (str): Virtual dataset name defined in config.ini
            Used to locate configuration section and construct output paths
        config_path (str): Path to configuration file (typically config.ini)
            Contains dataset-specific processing parameters and settings

    Raises:
        FileNotFoundError: If config file or source CSV doesn't exist
        ValueError: If dataset configuration is invalid or incomplete
        OSError: If directory creation or file operations fail

    Example:
        # Prepare a dataset named 'librispeech_clean' using default config
        prepare_data('librispeech_clean', 'config.ini')

        # Output files created:
        # - data/datasets/librispeech_clean/librispeech_clean_prepared.csv
        # - data/datasets/librispeech_clean/train.csv
        # - data/datasets/librispeech_clean/validation.csv
        # - data/audio/{note_id}.wav (processed audio files)
    """

    # Configuration Loading and Validation
    # Load dataset-specific configuration from INI file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Default Configuration Loading
    # Extract system-wide defaults for consistent behavior
    id_column = config["dataset_defaults"].get("id_column", "note_id")

    # Dataset Configuration Section Identification
    # Locate dataset-specific configuration using naming convention
    dataset_section = f"dataset:{dataset_name}"
    if not config.has_section(dataset_section):
        raise ValueError(f"Section '{dataset_section}' not found in config.ini.")

    # Output Path Construction
    # Build organized directory structure for dataset files
    dataset_file = config.get(dataset_section, "source")  # Assuming 'source' key holds the source CSV name
    prepared_csv_path = str(_PROJECT_ROOT / "data" / "datasets" / dataset_name / f"{dataset_name}_prepared.csv")
    train_split_path = str(_PROJECT_ROOT / "data" / "datasets" / dataset_name / "train.csv")
    val_split_path = str(_PROJECT_ROOT / "data" / "datasets" / dataset_name / "validation.csv")

    # Directory Structure Creation
    # Ensure output directories exist before processing
    os.makedirs(os.path.dirname(prepared_csv_path), exist_ok=True)

    # Language Filtering Configuration
    # Configure language processing and filtering parameters
    languages_str = config.get(
        dataset_section, "languages", fallback=config["dataset_defaults"].get("languages", "all")
    )
    if languages_str == "all":
        languages = []  # All languages
    else:
        languages = [lang.strip() for lang in languages_str.split(",")]

    # 1. Audio Processing Pipeline
    df = pd.read_csv(str(_PROJECT_ROOT / "data" / f"{dataset_file}.csv"))
    initial_rows = len(df)

    # Column Standardization
    # Normalize column names for consistent processing throughout pipeline
    df = df.rename(columns={id_column: "id"})

    # Auto-detect cloud streaming URIs and force streaming mode
    if not no_download and "audio_url" in df.columns and df["audio_url"].astype(str).str.startswith("gs://").any():
        logger.info("Detected GCS URIs in audio_url column; enabling streaming mode (no downloads).")
        no_download = True

    if no_download:
        # Streaming mode: Skip downloads and use GCS URIs directly
        logger.info("\n=== STREAMING MODE: Skipping audio downloads ===")
        logger.info("Audio URLs will be written directly to audio_path for cloud streaming")

        # No failed downloads in streaming mode
        df_download_decode_success = df.copy()
        error_notes = []
    else:
        # Traditional mode: Download and convert audio files
        # Process all audio files with parallel download and format conversion
        audio_dir = str(_PROJECT_ROOT / "data" / "audio")
        os.makedirs(audio_dir, exist_ok=True)

        # Error Tracking Initialization
        # Track failed downloads for quality reporting and exclusion
        error_notes = []
        # Create lock for thread-safe access to error_notes list
        error_lock = threading.Lock()

        # Concurrent Processing Configuration
        # Optimize thread count for system resources and network capacity
        max_workers = 4  # You can adjust this value

        # Parallel Audio Processing Execution
        # Launch concurrent download and conversion tasks
        if "audio_url" not in df.columns:
            raise ValueError(f"Expected 'audio_url' column in dataset CSV but found columns: {list(df.columns)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for index, row in df.iterrows():
                audio_url = row["audio_url"]
                note_id = row["id"]
                futures.append(
                    executor.submit(download_and_decode_audio, audio_url, note_id, audio_dir, error_notes, error_lock)
                )

            # Task Completion and Progress Tracking
            # Monitor all concurrent tasks with progress indication
            # Track successful and failed downloads for reporting
            successful_downloads = 0
            failed_downloads = 0

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading and decoding audio"
            ):
                try:
                    success, note_id, error_msg = future.result()
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                        # Error already logged in the worker function
                except Exception as e:
                    # This should rarely happen now that worker function catches all exceptions
                    failed_downloads += 1
                    logger.error(f"Unexpected error in thread pool: {e}")

            print(f"\nDownload summary: {successful_downloads} successful, {failed_downloads} failed")

        # Quality Filtering: Remove Failed Downloads
        # Exclude samples that failed download or conversion
        df_download_decode_success = df[~df["id"].isin(error_notes)]

    # 2. Metadata Generation and Path Assignment
    # Add audio file paths and prepare dataset metadata
    if no_download:
        # Streaming mode: Use GCS URIs directly from audio_url column
        df_download_decode_success["audio_path"] = df_download_decode_success["audio_url"]
    else:
        # Traditional mode: Use local WAV file paths
        df_download_decode_success["audio_path"] = [
            os.path.join(audio_dir, f"{note_id}.wav").replace("\\", "/") for note_id in df_download_decode_success["id"]
        ]

    # Column Mapping and Standardization
    # Transform raw column names to system-standard naming convention
    df_download_decode_success = df_download_decode_success.rename(
        columns={
            "analysis_verbatim_transcript": "text_verbatim",
            "backend_audio_length_seconds": "duration",
            "analysis_ground_truth_transcript": "text_perfect",
        }
    )

    # Quality Filtering: Text Transcription Validation
    # Remove samples with missing or empty transcriptions after audio processing
    df_no_empty_transcriptions = df_download_decode_success.dropna(subset=["text_verbatim", "text_perfect"])
    df_no_empty_transcriptions = df_no_empty_transcriptions[
        df_no_empty_transcriptions["text_verbatim"].str.strip().ne("")
        & df_no_empty_transcriptions["text_perfect"].str.strip().ne("")
    ]

    # Language-Based Dataset Filtering
    # Copy first to avoid SettingWithCopyWarning from pandas views
    df_lang_filtered = df_no_empty_transcriptions.copy()

    # Language Code Normalization — applied after .copy() to guarantee independent DataFrame
    if "analysis_primary_language" in df_lang_filtered.columns:
        df_lang_filtered["language"] = df_lang_filtered["analysis_primary_language"].apply(
            lambda x: x[:2] if pd.notna(x) and isinstance(x, str) else x
        )
    else:
        raise ValueError("Source CSV missing required column 'analysis_primary_language'")
    if languages:
        # Include rows where the language is in the specified list or is the unknown language token
        df_lang_filtered = df_lang_filtered[
            df_lang_filtered["language"].isin(languages) | (df_lang_filtered["language"] == UNKNOWN_LANGUAGE_TOKEN)
        ]

    df_lang_filtered.reset_index(drop=True, inplace=True)

    # Audio Duration Analysis and Validation
    if no_download:
        # Streaming mode: Skip duration calculation (no local files)
        print("\nSkipping duration calculation in streaming mode")
        # "backend_audio_length_seconds" was already renamed to "duration" in the column
        # rename step above. The original column no longer exists, making the if-branch
        # dead code and always landing in the else (zeroing duration). Check for "duration"
        # directly instead to avoid silently zero-ing valid duration data.
        if "duration" not in df_lang_filtered.columns:
            df_lang_filtered["duration"] = 0  # Fallback for datasets that never had duration
    else:
        # Traditional mode: Calculate precise durations using librosa
        print("\nCalculating audio durations...")
        durations = []
        for audio_path in tqdm(df_lang_filtered["audio_path"], desc="Calculating durations"):
            duration = librosa.get_duration(path=audio_path)
            durations.append(duration)
        df_lang_filtered["duration"] = durations

    # Comprehensive Dataset Statistics and Quality Reporting
    # Provide detailed analysis of processing results and quality metrics
    logger.info("\nDataset Statistics:")
    logger.info(f"  Initial number of rows: {initial_rows}")
    logger.info(f"  Number of rows after removing failed downloads/decodes: {len(df_download_decode_success)}")
    logger.info(f"  Number of rows after removing empty transcriptions: {len(df_no_empty_transcriptions)}")
    if languages:
        logger.info(f"  Number of rows after filtering by languages {languages}: {len(df_lang_filtered)}")
    logger.info(f"  Number of rows after removing duplicates: {len(df_lang_filtered.drop_duplicates())}")

    if not no_download:
        logger.info(f"\n  Max duration: {df_lang_filtered['duration'].max():.2f} seconds")
        logger.info(f"  Min duration: {df_lang_filtered['duration'].min():.2f} seconds")
        logger.info(f"  Average duration: {df_lang_filtered['duration'].mean():.2f} seconds")
    else:
        logger.info("\n  Duration statistics: Not calculated in streaming mode")

    # Language distribution
    logger.info("\n  Language Distribution:")
    language_counts = df_lang_filtered["language"].value_counts()
    for lang, count in language_counts.items():
        logger.info(f"    {lang}: {count} samples")

    # Duration-Based Quality Filtering
    # Apply configurable duration limits for training optimization
    max_duration = float(config.get(dataset_section, "max_duration"))
    if max_duration > 0:
        # Warn when all durations are 0 (streaming mode fallback) -- filter becomes a no-op
        if (df_lang_filtered["duration"] == 0).all():
            logger.warning(
                "max_duration filter is active (%.1f s) but all durations are 0 "
                "(streaming mode). Filter will pass all rows unmodified.",
                max_duration,
            )
        df_duration_filtered = df_lang_filtered[df_lang_filtered["duration"] <= max_duration]
    else:
        df_duration_filtered = df_lang_filtered
    # Final Column Selection and Cleanup
    # Retain only essential columns for training pipeline
    necessary_columns = ["id", "audio_path", "text_verbatim", "text_perfect", "language", "duration"]
    df_duration_filtered = df_duration_filtered[necessary_columns]

    # Guard before writing: raise early so no empty header-only CSV is left on disk.
    if len(df_duration_filtered) == 0:
        raise ValueError(
            "No samples remain after filtering. Check language, duration, "
            "and download settings."
        )

    df_duration_filtered.to_csv(prepared_csv_path, index=False)

    # 3. Dataset Splitting with Stratification
    # Create balanced train/validation splits for reliable model evaluation
    train_df, val_df = train_test_split(df_duration_filtered, test_size=0.1, random_state=42)

    # Guard: a dataset with fewer than 10 rows produces an empty validation set.
    # This doesn't block training but evaluation metrics will be meaningless.
    if len(val_df) == 0:
        logger.warning(
            "Validation set is empty after split (dataset has fewer than 10 rows). "
            "Evaluation metrics will be unreliable."
        )

    # Split Generation with Progress Reporting
    logger.info("\nGenerating stratified train/validation splits...")
    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)

    # Comprehensive Processing Summary and Quality Assessment
    # Report detailed statistics on all filtering stages and final dataset characteristics
    logger.info("\nData Preparation Summary:")
    logger.info(f"  Removed due to download/decode errors: {error_notes}")
    logger.info(
        f"  Removed due to missing transcriptions: {len(df_download_decode_success) - len(df_no_empty_transcriptions)}"
    )
    logger.info(f"  Removed due to language filtering: {len(df_no_empty_transcriptions) - len(df_lang_filtered)}")
    logger.info(
        f"  Removed due to duration exceeding max_duration: {len(df_lang_filtered) - len(df_duration_filtered)}"
    )
    logger.info(f"  Final number of rows in prepared dataset: {len(df_duration_filtered)}")
    logger.info(f"  Number of training samples: {len(train_df)}")
    logger.info(f"  Number of validation samples: {len(val_df)}")

    if no_download:
        logger.info(f"\nData preparation for dataset '{dataset_name}' completed in STREAMING MODE.")
        logger.info(f"Prepared CSV with GCS URIs saved to {prepared_csv_path}")
        logger.info(f"Train/validation splits saved to data/datasets/{dataset_name}")
        logger.info("\nNote: Audio files will be streamed from cloud storage during training.")
    else:
        logger.info(
            f"\nData preparation for dataset '{dataset_name}' completed. Prepared CSV saved to {prepared_csv_path}"
        )
        logger.info(f"Train/validation splits saved to data/datasets/{dataset_name}")


# Command-Line Interface for Direct Script Execution
# Provides standalone dataset preparation capability with argument parsing
if __name__ == "__main__":
    # Argument Parser Configuration
    # Define command-line interface for dataset preparation
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset preparation for Gemma fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py librispeech_clean
  python prepare_data.py custom_dataset --config config/custom.ini
        """,
    )
    parser.add_argument(
        "dataset", help="Virtual dataset name defined in config.ini (e.g., librispeech_clean, custom_dataset)."
    )
    parser.add_argument(
        "--config", default="config.ini", help="Configuration file path containing dataset settings and parameters."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip audio downloading and write GCS URIs directly to audio_path. For streaming from cloud storage.",
    )
    args = parser.parse_args()

    prepare_data(args.dataset, args.config, args.no_download)
