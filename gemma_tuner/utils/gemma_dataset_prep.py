#!/usr/bin/env python3
"""CSV → JSONL for Gemma multimodal (audio + transcript) and optional ``--validate`` probe.

Message layout matches ``DataCollatorGemmaAudio`` in ``gemma_tuner/models/gemma/finetune.py``.
Entry point: ``python -m gemma_tuner.utils.gemma_dataset_prep`` (see ``--help``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoProcessor

from gemma_tuner.models.gemma.constants import GemmaTrainingConstants, resolve_processor_sampling_rate

# Reuse shared audio I/O to support file system and GCS URIs.
# Only catch ImportError (missing optional deps such as librosa/soundfile) so that
# programmer errors — AttributeError, SyntaxError, etc. — are never swallowed.
try:
    from gemma_tuner.utils.dataset_prep import load_audio_local_or_gcs
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import load_audio_local_or_gcs ({e}). "
        "Audio loading will be unavailable. Install librosa and soundfile.",
        ImportWarning,
        stacklevel=2,
    )
    load_audio_local_or_gcs = None  # Optional for --validate path when not needed


class GemmaDatasetPrepConstants:
    """Named constants for Gemma dataset preparation and validation."""

    # Default Configuration (aligned with finetune.py when base_model is unset)
    DEFAULT_MODEL_ID = GemmaTrainingConstants.DEFAULT_BASE_MODEL_ID
    DEFAULT_TEXT_COLUMN = "text"  # Default transcript column name

    # Required CSV Columns
    REQUIRED_AUDIO_COLUMN = "audio_path"  # Required audio file path column
    FALLBACK_AUDIO_COLUMN = "audio"  # Alternative audio column name

    # Must match models/gemma/finetune.py:DataCollatorGemmaAudio
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
    """User (audio placeholder + transcription prompt) and assistant (``transcript``) messages.

    Must stay aligned with ``DataCollatorGemmaAudio`` in ``gemma/finetune.py``.
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
    """Read CSV (``audio_path`` or ``audio``, plus ``text_column``), write JSONL lines.

    Returns ``0`` if at least one row written, ``2`` if none. Raises ``ValueError`` if
    required columns are missing.
    """
    constants = GemmaDatasetPrepConstants

    csv_path = Path(csv_path)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    written_records = 0

    with csv_path.open("r", newline=constants.CSV_NEWLINE_MODE) as rf, output_jsonl.open("w") as wf:
        reader = csv.DictReader(rf)
        fieldnames = reader.fieldnames or []

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

        for row in reader:
            total_records += 1

            audio_path = (
                row.get(constants.REQUIRED_AUDIO_COLUMN) or row.get(constants.FALLBACK_AUDIO_COLUMN) or ""
            ).strip()

            transcript = (row.get(text_column) or "").strip()

            if not audio_path:
                continue

            record = {
                constants.JSONL_KEYS["AUDIO_PATH"]: audio_path,
                constants.JSONL_KEYS["MESSAGES"]: _build_messages(transcript),
            }

            json_line = json.dumps(
                record, ensure_ascii=constants.JSON_ENSURE_ASCII, separators=constants.JSON_SEPARATORS
            )
            wf.write(json_line + "\n")
            written_records += 1

    print(f"Wrote {written_records}/{total_records} records to {output_jsonl}")
    return constants.SUCCESS_CODE if written_records > 0 else constants.PARTIAL_FAILURE_CODE


def validate_single_sample(audio_path: str, text: str, model_id: str) -> None:
    """Load ``model_id`` processor, ``apply_chat_template`` + ``processor(text=..., audio=...)``, print shapes.

    Uses :func:`~gemma_tuner.models.gemma.constants.resolve_processor_sampling_rate` like the
    training collator. If ``load_audio_local_or_gcs`` could not be imported, runs without audio.
    """
    constants = GemmaDatasetPrepConstants

    processor = AutoProcessor.from_pretrained(model_id)
    messages = _build_messages(text)

    sampling_rate = resolve_processor_sampling_rate(processor)

    audios = []
    if load_audio_local_or_gcs is not None:
        audio_array = load_audio_local_or_gcs(audio_path, sampling_rate=sampling_rate)
        audios.append(audio_array)

    prompts = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=False,
    )
    proc_kwargs: Dict[str, object] = {
        "text": prompts,
        "return_tensors": constants.PROCESSOR_RETURN_TENSORS,
        "padding": constants.PROCESSOR_PADDING,
    }
    if audios:
        proc_kwargs["audio"] = audios
        proc_kwargs["sampling_rate"] = sampling_rate
    processed_inputs = processor(**proc_kwargs)

    output_summary = {
        key: tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        for key, value in processed_inputs.items()
    }

    print("Processor output summary:", output_summary)


def main() -> int:
    """CLI: ``--csv``/``--out`` for conversion, or ``--validate`` with ``--audio``/``--text``."""
    constants = GemmaDatasetPrepConstants

    ap = argparse.ArgumentParser(
        description="CSV→JSONL for Gemma multimodal training, or validate one sample with AutoProcessor",
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

    args = ap.parse_args()

    if args.validate:
        if not (args.audio and args.text):
            ap.error("Validation mode requires both --audio and --text arguments. Use --help for usage examples.")

        validate_single_sample(args.audio, args.text, args.model_id)
        return constants.SUCCESS_CODE

    if not (args.csv and args.out):
        ap.error(
            "Dataset conversion requires both --csv and --out arguments. "
            "Use --validate for single sample testing, or --help for usage examples."
        )

    return prepare_gemma_jsonl(args.csv, args.out, text_column=args.text_column)


if __name__ == "__main__":
    raise SystemExit(main())
