"""
Structured Logging System for Gemma Fine-Tuning Pipeline

This module provides a unified logging infrastructure with support for both
human-readable and machine-parseable JSON output formats. It implements
structured logging patterns optimized for production monitoring, debugging,
and log aggregation systems.

Key responsibilities:
- Structured JSON logging for machine parsing and aggregation
- Human-readable format for development and debugging
- Consistent log formatting across all modules
- Library noise suppression for cleaner output
- File and console handler management

Called by:
- main.py:main() for initial logging setup (line 199)
- All modules via logging.getLogger(__name__)
- Training scripts for file-based logging
- Evaluation scripts for metrics logging

Output formats:
1. Human-readable (default):
   2024-01-15 10:30:45 INFO scripts.finetune: Starting training run

2. JSON structured (json_format=True):
   {"time": "2024-01-15 10:30:45", "level": "INFO", "name": "scripts.finetune", "message": "Starting training run"}

Integration with monitoring:
JSON format enables integration with log aggregation systems like ELK stack,
Datadog, or CloudWatch for production monitoring and alerting.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Logging configuration — inline constants to avoid an unnecessary namespace class.
_DEFAULT_LEVEL = "INFO"
_URLLIB3_LEVEL = logging.WARNING  # Suppress connection pool messages
_TRANSFORMERS_LEVEL = logging.INFO  # Keep model loading messages
_HUMAN_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_STREAM = sys.stdout  # Use stdout for container compatibility


class _JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging output.

    This formatter converts log records to JSON lines format, enabling
    machine parsing and integration with log aggregation systems. Each
    log entry becomes a single JSON object on one line.

    Used by:
    - init_logging() when json_format=True
    - add_file_handler() for JSON file logging
    - Production deployments requiring structured logs

    JSON schema:
    {
        "time": "2024-01-15 10:30:45",      # Human-readable timestamp
        "level": "INFO",                     # Log level name
        "name": "module.name",               # Logger name (usually module path)
        "message": "Log message",            # Actual log message
        "exc_info": "Traceback..."           # Exception details if present
    }

    Benefits:
    - Machine-parseable for log aggregation
    - Preserves structure for complex messages
    - Supports Unicode without escaping
    - Single-line format for streaming logs
    """

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        """
        Formats a log record as a JSON line.

        Args:
            record: LogRecord to format

        Returns:
            str: JSON-formatted log line
        """
        payload = {
            "time": self.formatTime(record, datefmt=_TIME_FORMAT),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include exception traceback if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # Single line JSON for streaming compatibility
        return json.dumps(payload, ensure_ascii=False)


def init_logging(level: str | int = "INFO", json_format: bool = False) -> None:
    """
    Initializes the root logger with consistent formatting and configuration.

    This function sets up the logging infrastructure for the entire application,
    configuring output format, log level, and library suppressions. It ensures
    clean, consistent logging across all modules.

    Called by:
    - main.py:main() at application startup (line 199)
    - Test fixtures for logging setup
    - Subprocess scripts requiring logging initialization

    Configuration strategy:
    1. Clear any existing handlers (prevents duplicates)
    2. Set up console handler with appropriate formatter
    3. Configure root logger level
    4. Suppress noisy third-party libraries

    Args:
        level (str | int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                          or integer level value
        json_format (bool): If True, use JSON formatter; else human-readable

    Side effects:
        - Removes all existing root logger handlers
        - Adds new StreamHandler to stdout
        - Sets library-specific log levels

    Example:
        >>> init_logging("DEBUG", json_format=True)  # Development with JSON
        >>> init_logging("INFO")  # Production with human-readable logs

    Library suppression:
        - urllib3: WARNING level (hides connection pool messages)
        - transformers: INFO level (shows model loading progress)
    """
    # Convert string level to logging constant
    lvl = getattr(logging, str(level).upper(), logging.INFO) if isinstance(level, str) else level

    # Clear existing handlers to prevent duplicate logs
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # Configure console handler with appropriate formatter
    handler = logging.StreamHandler(_DEFAULT_STREAM)
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT))

    root.setLevel(lvl)
    root.addHandler(handler)

    # Suppress noisy third-party libraries
    logging.getLogger("urllib3").setLevel(_URLLIB3_LEVEL)
    logging.getLogger("transformers").setLevel(_TRANSFORMERS_LEVEL)


def add_file_handler(log_path: str, json_format: bool = False, level: Optional[int] = None) -> None:
    """
    Attaches a file handler to the root logger for persistent log storage.

    This function adds file-based logging to complement console output,
    enabling log persistence for debugging, auditing, and post-mortem analysis.
    Supports both human-readable and JSON formats for different use cases.

    Called by:
    - Training scripts for run-specific logging
    - main.py when --log-file argument provided
    - Production deployments requiring persistent logs
    - Debug sessions needing detailed trace logs

    File logging patterns:
    1. Training logs: output/{run_id}/training.log
    2. Evaluation logs: output/{run_id}/eval/evaluation.log
    3. Debug logs: debug_{timestamp}.log
    4. Production logs: /var/log/gemma-tuner.log

    Args:
        log_path (str): Path to log file (created if doesn't exist)
        json_format (bool): If True, use JSON formatter for structured logs
        level (Optional[int]): Specific level for file handler (None uses root level)

    Side effects:
        - Creates log file and parent directories if needed
        - Adds FileHandler to root logger (cumulative with existing handlers)
        - File remains open until process exits or handler removed

    Example:
        >>> # Training run with detailed file logging
        >>> add_file_handler("output/5-gemma/training.log", level=logging.DEBUG)
        >>>
        >>> # Production with JSON logs for aggregation
        >>> add_file_handler("/var/log/gemma-tuner.json", json_format=True)

    Note:
        Multiple file handlers can be added for different log files.
        Each handler operates independently with its own level and format.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)

    # Configure formatter based on format preference
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT))

    # Set handler-specific level if provided
    if level is not None:
        handler.setLevel(level)

    # Add to root logger (cumulative with existing handlers)
    logging.getLogger().addHandler(handler)
