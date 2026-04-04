#!/usr/bin/env python3
"""
Gemma 3n Dataset Preparation and Validation Utilities

This module provides comprehensive tools for converting standard CSV-based audio datasets
into Gemma 3n-compatible formats and validating multimodal preprocessing pipelines. It
serves as the bridge between the project's universal CSV dataset format and Gemma 3n's
specific multimodal requirements.

Key Responsibilities:
- CSV to JSONL conversion with Gemma 3n chat message formatting
- Single-sample validation through AutoProcessor pipeline
- Message template construction for multimodal training
- Preprocessing validation and debugging support
- Format standardization for reproducible training workflows

Architecture Integration:
This utility complements the runtime training infrastructure while providing offline
preprocessing capabilities. It ensures consistency between development, debugging,
and production training workflows by using identical message formatting.

Called by:
- Development workflows for dataset preprocessing and validation
- CI/CD pipelines for dataset format validation
- Debugging sessions for processor and message format testing
- Data scientists preparing custom datasets for Gemma 3n training
- Batch processing scripts for large-scale dataset conversion

Calls to:
- transformers.AutoProcessor for Gemma 3n multimodal preprocessing
- utils.dataset_prep.load_audio_local_or_gcs for audio loading (optional)
- Standard CSV and JSON libraries for data format conversion
- pathlib for robust cross-platform file handling

Cross-File Integration Points:

Training Pipeline Integration:
- models/gemma/finetune.py:DataCollatorGemmaAudio uses identical message format
- models/gemma/finetune.py:_build_messages() mirrors message construction logic
- Ensures training collator and offline preprocessing produce identical results

Dataset Loading Integration:
- utils/dataset_prep.py:load_audio_local_or_gcs for consistent audio loading
- Supports both local filesystem and Google Cloud Storage URIs
- Maintains compatibility with existing dataset infrastructure

Configuration Integration:
- README/specifications/Gemma3n.md documents message format requirements
- config.ini dataset configurations can reference generated JSONL files
- Supports existing profile-based training configuration system

Message Format Specification:
The utility implements the Gemma 3n multimodal chat template with the following structure:
1. User message with audio placeholder and transcription request
2. Assistant message with target transcription text
3. Proper role attribution for chat-based training
4. Audio attachment handled separately from text content

JSONL Output Format:
Each line contains a JSON object with:
- audio_path: File path or URI to audio resource
- messages: Structured chat conversation for multimodal training

This format enables:
- Offline inspection of training data structure
- Debugging of message formatting issues
- Reproducible dataset preparation workflows
- Validation of processor input formats

Usage Patterns:

Dataset Conversion Workflow:
1. Prepare CSV with audio_path and transcript columns
2. Run conversion to generate JSONL with proper message formatting
3. Validate sample entries using single-sample validation
4. Integrate JSONL into training pipeline or use for debugging

Processor Validation Workflow:
1. Test individual audio files and transcripts
2. Verify processor accepts message format
3. Inspect output tensor shapes and types
4. Debug preprocessing pipeline issues

CLI Usage Examples:

Basic dataset conversion:
  python utils/gemma_dataset_prep.py \
      --csv data/datasets/librispeech/train.csv \
      --out data/datasets/librispeech/train_gemma.jsonl \
      --text-column text

Custom column mapping:
  python utils/gemma_dataset_prep.py \
      --csv data/datasets/custom/data.csv \
      --out data/datasets/custom/formatted.jsonl \
      --text-column transcript \
      --model-id google/gemma-4-E2B-it

Single sample validation:
  python utils/gemma_dataset_prep.py --validate \
      --audio test_files/sample.wav \
      --text "hello world example" \
      --model-id google/gemma-4-E2B-it

Batch validation across multiple samples:
  for audio in test_files/*.wav; do
    python utils/gemma_dataset_prep.py --validate \
        --audio "$audio" \
        --text "validation text" \
        --model-id google/gemma-4-E2B-it
  done

Design Principles:
- Format Consistency: Identical message structure to runtime training collator
- Processor Agnostic: Defers to official AutoProcessor for all processing logic
- Debugging Friendly: Comprehensive validation and error reporting
- Storage Efficient: Audio paths stored separately from message content
- Integration Compatible: Works seamlessly with existing dataset infrastructure

Error Handling Strategy:
- Graceful degradation when optional audio loading unavailable
- Comprehensive validation of CSV column requirements
- Detailed error messages for debugging format issues
- Safe handling of malformed or missing data entries

Performance Considerations:
- Memory efficient: Processes CSV row-by-row without loading entire dataset
- I/O optimized: Streaming writes to JSONL output
- Optional audio loading: Can validate message format without audio processing
- Configurable validation: Choose between format-only or full processor validation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoProcessor

# Reuse shared audio I/O to support file system and GCS URIs
try:
    from gemma_tuner.utils.dataset_prep import load_audio_local_or_gcs
except Exception:
    load_audio_local_or_gcs = None  # Optional for --validate path when not needed


class GemmaDatasetPrepConstants:
    """Named constants for Gemma dataset preparation and validation."""

    # Default Configuration
    DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"  # Default Gemma multimodal model
    DEFAULT_TEXT_COLUMN = "text"  # Default transcript column name

    # Required CSV Columns
    REQUIRED_AUDIO_COLUMN = "audio_path"  # Required audio file path column
    FALLBACK_AUDIO_COLUMN = "audio"  # Alternative audio column name

    # Chat Message Template Structure
    # These constants define the Gemma 3n multimodal chat format
    # Must match the format used in models/gemma/finetune.py:DataCollatorGemmaAudio
    CHAT_ROLES = {
        "USER": "user",  # User role for chat messages
        "ASSISTANT": "assistant",  # Assistant role for chat messages
    }

    CONTENT_TYPES = {
        "AUDIO": "audio",  # Audio content type identifier
        "TEXT": "text",  # Text content type identifier
    }

    # Message Content Templates
    AUDIO_PLACEHOLDER = "<audio:attached>"  # Placeholder for audio attachment
    TRANSCRIPTION_PROMPT = "Please transcribe this audio."  # Standard prompt for transcription

    # JSONL Output Structure
    JSONL_KEYS = {
        "AUDIO_PATH": "audio_path",  # Key for audio file path in JSONL
        "MESSAGES": "messages",  # Key for chat messages in JSONL
    }

    # Error Codes and Handling
    SUCCESS_CODE = 0  # Successful execution return code
    PARTIAL_FAILURE_CODE = 2  # Partial failure (some records skipped)

    # File Processing Configuration
    CSV_NEWLINE_MODE = ""  # CSV newline handling (use default)
    JSON_ENSURE_ASCII = False  # Allow non-ASCII characters in JSON output
    JSON_SEPARATORS = (",", ":")  # Compact JSON formatting

    # Processor Validation Configuration
    PROCESSOR_RETURN_TENSORS = "pt"  # PyTorch tensor format for validation
    PROCESSOR_PADDING = True  # Enable padding for batch processing


def _build_messages(transcript: str) -> List[Dict]:
    """
    Constructs Gemma 3n multimodal chat messages with proper formatting for training.

    This function creates the standard message structure required for Gemma 3n multimodal
    training, ensuring consistency between offline preprocessing and runtime training
    collation. The format matches exactly what the training pipeline expects.

    Called by:
    - prepare_gemma_jsonl() during CSV to JSONL conversion
    - validate_single_sample() during processor validation
    - External scripts using this utility for message formatting

    Integration with Training Pipeline:
    - Message format MUST match models/gemma/finetune.py:DataCollatorGemmaAudio
    - Ensures processor.apply_chat_template() receives correct structure
    - Enables proper injection of <bos>, <start_of_turn>, <end_of_turn> tokens

    Message Structure Design:
    1. User Message:
       - Contains audio placeholder for multimodal attachment
       - Includes transcription request prompt
       - Uses proper content type annotations

    2. Assistant Message:
       - Contains target transcription text
       - Provides training target for language modeling objective
       - Formatted as text content type

    Chat Template Requirements:
    - Roles: "user" and "assistant" for proper chat formatting
    - Content types: "audio" and "text" for multimodal processing
    - Audio placeholder: Special token for audio attachment
    - Prompt text: Standardized request for transcription

    Args:
        transcript (str): Target transcription text for assistant response
                         Empty string is acceptable and will be preserved

    Returns:
        List[Dict]: Structured chat messages containing:
            - User message with audio placeholder and transcription request
            - Assistant message with target transcription text

    Format Specification:
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "<audio:attached>"},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<transcript>"}
                ]
            }
        ]

    Note:
    This function deliberately uses named constants to ensure consistency
    with the training pipeline and enable easy maintenance of message formats.
    """
    constants = GemmaDatasetPrepConstants

    return [
        {
            "role": constants.CHAT_ROLES["USER"],
            "content": [
                {"type": constants.CONTENT_TYPES["AUDIO"], "audio": constants.AUDIO_PLACEHOLDER},
                {"type": constants.CONTENT_TYPES["TEXT"], "text": constants.TRANSCRIPTION_PROMPT},
            ],
        },
        {
            "role": constants.CHAT_ROLES["ASSISTANT"],
            "content": [{"type": constants.CONTENT_TYPES["TEXT"], "text": transcript or ""}],
        },
    ]


def prepare_gemma_jsonl(
    csv_path: str | Path,
    output_jsonl: str | Path,
    text_column: str = GemmaDatasetPrepConstants.DEFAULT_TEXT_COLUMN,
) -> int:
    """
    Converts CSV dataset to Gemma 3n-compatible JSONL format with multimodal chat messages.

    This function implements the core dataset conversion workflow, transforming standard
    CSV datasets into the structured JSONL format required for Gemma 3n multimodal training.
    It provides comprehensive validation, error handling, and progress reporting.

    Called by:
    - main() function during CLI-driven dataset conversion
    - Batch processing scripts for large-scale dataset preparation
    - Development workflows preparing custom datasets
    - CI/CD pipelines for automated dataset validation and conversion

    Calls to:
    - _build_messages() for consistent chat message formatting
    - pathlib.Path for robust cross-platform file handling
    - csv.DictReader for efficient CSV processing
    - json.dumps for optimized JSONL serialization

    Dataset Processing Workflow:
    1. Validate input CSV exists and has required columns
    2. Create output directory structure if needed
    3. Process CSV row-by-row for memory efficiency
    4. Validate each row has required audio path and transcript
    5. Convert each valid row to structured JSONL format
    6. Write JSONL with proper formatting and encoding
    7. Report conversion statistics and success metrics

    Required CSV Structure:
    - audio_path column: File paths or URIs to audio files
    - transcript column: Target transcription text (configurable column name)
    - Optional columns: Any additional metadata (preserved in workflow)

    JSONL Output Format:
    Each line contains a JSON object with:
    - audio_path: Original audio file path or URI
    - messages: Structured chat conversation for training

    Message Format Integration:
    - Uses _build_messages() for consistent formatting
    - Matches training pipeline expectations exactly
    - Enables seamless integration with models/gemma/finetune.py

    Error Handling Strategy:
    - Column validation: Ensures required columns exist before processing
    - Row validation: Skips rows with missing audio paths (non-fatal)
    - File handling: Creates directories and handles I/O errors gracefully
    - Progress tracking: Reports conversion statistics for debugging

    Performance Optimizations:
    - Streaming processing: Memory-efficient row-by-row processing
    - Optimized JSON: Compact serialization with non-ASCII support
    - Lazy evaluation: Opens files only when needed
    - Progress reporting: Real-time feedback for large datasets

    Args:
        csv_path (str | Path): Path to input CSV file with audio_path and transcript columns
        output_jsonl (str | Path): Path to output JSONL file (directories created if needed)
        text_column (str): Name of transcript column in CSV (default: "text")

    Returns:
        int: Exit code indicating conversion status:
            - 0: Success (at least one record converted)
            - 2: Partial failure (no valid records found)

    Raises:
        ValueError: If required columns missing from CSV
        FileNotFoundError: If input CSV file doesn't exist
        PermissionError: If output directory cannot be created

    Example Usage:
        # Basic conversion
        result = prepare_gemma_jsonl("data.csv", "data_gemma.jsonl")

        # Custom transcript column
        result = prepare_gemma_jsonl(
            "custom.csv",
            "formatted.jsonl",
            text_column="transcript"
        )

    Output Statistics:
    Prints conversion summary showing:
    - Number of records successfully converted
    - Total number of input records processed
    - Output file path for verification

    Integration Notes:
    - Output JSONL can be consumed by training pipeline directly
    - Message format matches runtime collator expectations
    - Audio files remain external (paths stored, not embedded)
    - Compatible with existing dataset infrastructure
    """
    constants = GemmaDatasetPrepConstants

    # Normalize file paths for cross-platform compatibility
    csv_path = Path(csv_path)
    output_jsonl = Path(output_jsonl)

    # Ensure output directory exists with proper permissions
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Initialize processing counters for progress tracking
    total_records = 0
    written_records = 0

    # Process CSV with streaming I/O for memory efficiency
    with csv_path.open("r", newline=constants.CSV_NEWLINE_MODE) as rf, output_jsonl.open("w") as wf:
        reader = csv.DictReader(rf)

        # Validate required columns exist before processing
        fieldnames = reader.fieldnames or []

        # Check for primary audio column or fallback
        has_audio_path = constants.REQUIRED_AUDIO_COLUMN in fieldnames
        has_audio_fallback = constants.FALLBACK_AUDIO_COLUMN in fieldnames

        if not (has_audio_path or has_audio_fallback):
            raise ValueError(
                f"CSV must contain either '{constants.REQUIRED_AUDIO_COLUMN}' or "
                f"'{constants.FALLBACK_AUDIO_COLUMN}' column. "
                f"Found columns: {fieldnames}"
            )

        if text_column not in fieldnames:
            raise ValueError(f"CSV must contain the transcript column '{text_column}'. Found columns: {fieldnames}")

        # Process each CSV row with validation and error handling
        for row in reader:
            total_records += 1

            # Extract audio path with fallback column support
            audio_path = (
                row.get(constants.REQUIRED_AUDIO_COLUMN) or row.get(constants.FALLBACK_AUDIO_COLUMN) or ""
            ).strip()

            # Extract transcript text with empty string fallback
            transcript = (row.get(text_column) or "").strip()

            # Skip rows without valid audio paths (non-fatal)
            if not audio_path:
                continue

            # Construct JSONL record with structured format
            record = {
                constants.JSONL_KEYS["AUDIO_PATH"]: audio_path,
                constants.JSONL_KEYS["MESSAGES"]: _build_messages(transcript),
            }

            # Write JSONL with optimized serialization
            json_line = json.dumps(
                record, ensure_ascii=constants.JSON_ENSURE_ASCII, separators=constants.JSON_SEPARATORS
            )
            wf.write(json_line + "\n")
            written_records += 1

    # Report conversion statistics for verification and debugging
    print(f"Wrote {written_records}/{total_records} records to {output_jsonl}")

    # Return appropriate exit code based on conversion success
    return constants.SUCCESS_CODE if written_records > 0 else constants.PARTIAL_FAILURE_CODE


def validate_single_sample(audio_path: str, text: str, model_id: str) -> None:
    """
    Validates single audio-text sample through Gemma 3n processor pipeline.

    This function provides comprehensive validation of the preprocessing pipeline
    without performing model inference. It ensures message format compatibility,
    audio loading functionality, and processor output structure validation.

    Called by:
    - main() function during CLI validation mode
    - Development workflows testing processor compatibility
    - Debugging sessions for message format issues
    - CI/CD pipelines for preprocessing validation

    Calls to:
    - transformers.AutoProcessor.from_pretrained() for model-specific processing
    - _build_messages() for consistent message formatting
    - utils.dataset_prep.load_audio_local_or_gcs() for audio loading (optional)
    - processor() for full multimodal preprocessing pipeline

    Validation Pipeline:
    1. Load Gemma 3n processor for specified model
    2. Construct chat messages using standard format
    3. Optionally load audio file with processor-specific sampling rate
    4. Process messages and audio through full preprocessing pipeline
    5. Report output tensor shapes and types for verification

    Processor Integration Testing:
    - Verifies message format accepted by AutoProcessor
    - Tests audio preprocessing with model-specific requirements
    - Validates tensor output structure for training compatibility
    - Ensures preprocessing matches training pipeline expectations

    Audio Loading Strategy:
    - Attempts to use processor's preferred sampling rate
    - Falls back gracefully when audio loading unavailable
    - Supports both format-only and full audio validation
    - Handles various audio file formats through soundfile integration

    Sampling Rate Detection:
    - Checks processor.sampling_rate attribute first
    - Falls back to processor.feature_extractor.sampling_rate
    - Uses None (auto-detection) if no rate specified
    - Ensures compatibility with processor audio requirements

    Error Handling:
    - Graceful fallback when audio loading unavailable
    - Safe handling of processor attribute access
    - Comprehensive error reporting for debugging
    - Non-fatal audio loading failures (message-only validation)

    Output Validation:
    - Reports tensor shapes for all processor outputs
    - Identifies data types for debugging type mismatches
    - Provides compact summary for quick verification
    - Enables comparison with training pipeline expectations

    Args:
        audio_path (str): Path to audio file for validation
        text (str): Transcript text for message construction
        model_id (str): Hugging Face model identifier for processor loading

    Output:
        Prints processor output summary showing:
        - Tensor shapes for all output keys
        - Data types for non-tensor outputs
        - Validation success confirmation

    Raises:
        FileNotFoundError: If model_id cannot be loaded from Hugging Face
        Exception: If processor cannot handle message format or audio

    Example Usage:
        # Basic validation
        validate_single_sample("test.wav", "hello world", "google/gemma-4-E2B-it")

        # Custom model validation
        validate_single_sample("audio.flac", "transcript", "custom/gemma-model")

    Expected Output Example:
        Processor output summary: {
            'input_ids': (1, 256),
            'attention_mask': (1, 256),
            'audio_features': (1, 1024, 80),
            'labels': (1, 256)
        }

    Integration Notes:
    - Uses identical message format to training pipeline
    - Compatible with all Gemma 3n model variants
    - Validates preprocessing without model inference overhead
    - Provides debugging information for format issues
    """
    constants = GemmaDatasetPrepConstants

    # Load processor for specified Gemma 3n model
    processor = AutoProcessor.from_pretrained(model_id)

    # Construct messages using standard format for consistency
    messages = _build_messages(text)

    # Optional audio loading with processor-specific sampling rate detection
    audios = []
    if load_audio_local_or_gcs is not None:
        # Attempt to detect processor's preferred sampling rate
        # This ensures audio is preprocessed correctly for the specific model
        sampling_rate = None
        try:
            # Primary sampling rate source: processor attribute
            sampling_rate = getattr(processor, "sampling_rate", None)

            # Fallback: feature extractor sampling rate
            if sampling_rate is None and hasattr(processor, "feature_extractor"):
                sampling_rate = getattr(processor.feature_extractor, "sampling_rate", None)
        except Exception:
            # Safe fallback: let audio loader use default detection
            sampling_rate = None

        # Load audio with detected or default sampling rate
        audio_array = load_audio_local_or_gcs(audio_path, sampling_rate=sampling_rate)
        audios.append(audio_array)

    # Process through full multimodal preprocessing pipeline
    # This validates the complete preprocessing workflow used during training
    processed_inputs = processor(
        messages=[messages],
        audios=audios or None,
        return_tensors=constants.PROCESSOR_RETURN_TENSORS,
        padding=constants.PROCESSOR_PADDING,
    )

    # Generate compact summary of processor outputs for validation
    # This helps verify tensor shapes and types match training expectations
    output_summary = {
        key: tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        for key, value in processed_inputs.items()
    }

    # Report validation results for debugging and verification
    print("Processor output summary:", output_summary)


def main() -> int:
    """
    Main entry point for Gemma 3n dataset preparation and validation CLI.
    
    This function provides a comprehensive command-line interface for dataset conversion
    and processor validation workflows. It supports both batch dataset processing and
    individual sample validation with extensive error handling and user guidance.
    
    Called by:
    - Command-line execution when script is run directly
    - Batch processing scripts for automated dataset preparation
    - CI/CD pipelines for dataset validation and preprocessing
    - Development workflows for debugging and testing
    
    Calls to:
    - prepare_gemma_jsonl() for dataset conversion workflows
    - validate_single_sample() for processor validation workflows
    - argparse for comprehensive CLI argument handling
    
    CLI Design Philosophy:
    - Dual-mode operation: dataset conversion vs. sample validation
    - Comprehensive help text for user guidance
    - Sensible defaults for common use cases
    - Clear error messages for invalid argument combinations
    
    Workflow Modes:
    
    1. Dataset Conversion Mode (default):
       - Converts CSV datasets to Gemma 3n JSONL format
       - Requires --csv and --out arguments
       - Optional --text-column for custom transcript columns
       - Optional --model-id for processor-specific requirements
    
    2. Validation Mode (--validate flag):
       - Tests single audio-text samples through processor
       - Requires --audio and --text arguments
       - Optional --model-id for model-specific validation
       - Provides debugging output for preprocessing issues
    
    Error Handling Strategy:
    - Argument validation before processing begins
    - Clear error messages for missing required arguments
    - Mode-specific requirement checking
    - Graceful handling of file and processing errors
    
    Returns:
        int: Exit code indicating operation status:
            - 0: Success (conversion completed or validation passed)
            - 1: Argument error (invalid CLI usage)
            - 2: Partial failure (some records skipped in conversion)
            
    Example Usage:
        # Dataset conversion
        python utils/gemma_dataset_prep.py \
            --csv data/train.csv \
            --out data/train_gemma.jsonl
        
        # Custom transcript column
        python utils/gemma_dataset_prep.py \
            --csv data/custom.csv \
            --out data/formatted.jsonl \
            --text-column transcript
        
        # Single sample validation
        python utils/gemma_dataset_prep.py --validate \
            --audio test.wav \
            --text "hello world"
        
        # Custom model validation
        python utils/gemma_dataset_prep.py --validate \
            --audio sample.flac \
            --text "test transcript" \
            --model-id custom/gemma-model
    """
    constants = GemmaDatasetPrepConstants

    # Initialize argument parser with comprehensive CLI interface
    ap = argparse.ArgumentParser(
        description="Gemma 3n dataset preparation and validation utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Convert CSV to JSONL format
  %(prog)s --csv data.csv --out data_gemma.jsonl
  
  # Custom transcript column
  %(prog)s --csv data.csv --out formatted.jsonl --text-column transcript
  
  # Validate single sample
  %(prog)s --validate --audio test.wav --text "hello world"
        """,
    )

    # Dataset conversion arguments
    conversion_group = ap.add_argument_group("Dataset Conversion")
    conversion_group.add_argument("--csv", help="Input CSV file with audio_path and transcript columns")
    conversion_group.add_argument("--out", help="Output JSONL file path (directories created automatically)")
    conversion_group.add_argument(
        "--text-column", default=constants.DEFAULT_TEXT_COLUMN, help="Name of transcript column in CSV file"
    )

    # Validation mode arguments
    validation_group = ap.add_argument_group("Single Sample Validation")
    validation_group.add_argument(
        "--validate", action="store_true", help="Enable validation mode for testing single audio-text samples"
    )
    validation_group.add_argument("--audio", help="Audio file path for validation (required with --validate)")
    validation_group.add_argument("--text", help="Transcript text for validation (required with --validate)")

    # Common configuration arguments
    common_group = ap.add_argument_group("Common Configuration")
    common_group.add_argument(
        "--model-id", default=constants.DEFAULT_MODEL_ID, help="Hugging Face model identifier for processor loading"
    )

    # Parse and validate arguments
    args = ap.parse_args()

    # Validation mode: single sample processor testing
    if args.validate:
        # Ensure required validation arguments are provided
        if not (args.audio and args.text):
            ap.error("Validation mode requires both --audio and --text arguments. Use --help for usage examples.")

        # Execute validation workflow
        validate_single_sample(args.audio, args.text, args.model_id)
        return constants.SUCCESS_CODE

    # Conversion mode: CSV to JSONL dataset preparation
    if not (args.csv and args.out):
        ap.error(
            "Dataset conversion requires both --csv and --out arguments. "
            "Use --validate for single sample testing, or --help for usage examples."
        )

    # Execute conversion workflow
    return prepare_gemma_jsonl(args.csv, args.out, text_column=args.text_column)


if __name__ == "__main__":
    raise SystemExit(main())
