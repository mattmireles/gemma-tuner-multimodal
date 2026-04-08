#!/usr/bin/env python3
"""
Setup helper for gemma-tuner-multimodal.

This script helps new users get started by:
1. Checking if config.ini exists
2. Creating it from config.ini.example if needed
3. Verifying the setup is complete

Run this after cloning the repository to get started quickly.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find the repository root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "gemma_tuner").exists():
            return current
        current = current.parent
    return Path.cwd()


def setup_config() -> tuple[bool, str]:
    """
    Set up config.ini from the example file.

    Returns:
        (success, message): Success status and descriptive message.
    """
    repo_root = find_repo_root()
    config_dir = repo_root / "config"
    config_example = config_dir / "config.ini.example"
    config_file = config_dir / "config.ini"

    # Check if config.ini already exists
    if config_file.exists():
        return (
            True,
            f"✓ config.ini already exists at {config_file}\n"
            "  You're ready to go! Edit config.ini to customize your settings.",
        )

    # Check if example exists
    if not config_example.exists():
        return (
            False,
            f"✗ config.ini.example not found at {config_example}\n"
            "  Please ensure you cloned the repository correctly.",
        )

    # Create config directory if needed
    config_dir.mkdir(parents=True, exist_ok=True)

    # Copy example to config.ini
    try:
        shutil.copy(config_example, config_file)
        return (
            True,
            f"✓ Created config.ini at {config_file}\n"
            f"  Edit this file to configure your models and datasets.\n"
            f"  See README.md for detailed configuration instructions.",
        )
    except Exception as e:
        return (False, f"✗ Failed to create config.ini: {e}")


def check_environment() -> list[tuple[bool, str]]:
    """
    Check the environment for potential issues.

    Returns:
        List of (is_ok, message) tuples.
    """
    checks = []

    # Check Python version
    if sys.version_info >= (3, 10):
        checks.append((True, f"✓ Python {sys.version_info.major}.{sys.version_info.minor}"))
    else:
        checks.append(
            (
                False,
                f"✗ Python {sys.version_info.major}.{sys.version_info.minor} detected. "
                "Python 3.10+ is required.",
            )
        )

    # Check for .env file
    repo_root = find_repo_root()
    env_file = repo_root / ".env"
    if env_file.exists():
        checks.append((True, "✓ .env file found"))
    else:
        checks.append(
            (
                False,
                "⚠ .env file not found. Create it for HF_TOKEN and other secrets:\n"
                "   echo 'HF_TOKEN=your_token_here' > .env",
            )
        )

    # Check for virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        checks.append((True, "✓ Running in virtual environment"))
    else:
        checks.append(
            (
                False,
                "⚠ Not in a virtual environment. Recommended:\n"
                "   python -m venv venv && source venv/bin/activate  # macOS/Linux",
            )
        )

    return checks


def main() -> int:
    """Run the setup helper."""
    print("=" * 60)
    print("Gemma Tuner - Setup Helper")
    print("=" * 60)
    print()

    # Setup config.ini
    print("Step 1: Configuration Setup")
    print("-" * 40)
    success, message = setup_config()
    print(message)
    print()

    # Environment checks
    print("Step 2: Environment Checks")
    print("-" * 40)
    checks = check_environment()
    for is_ok, msg in checks:
        print(msg)

    # Summary
    print()
    print("=" * 60)
    if success:
        print("Setup complete! Next steps:")
        print()
        print("1. Edit config/config.ini with your model and dataset settings")
        print("2. Run: pip install -e .")
        print("3. Run: gemma-macos-tuner --help")
        print()
        print("See README.md for detailed usage instructions.")
    else:
        print("Setup encountered issues. Please resolve them above.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
