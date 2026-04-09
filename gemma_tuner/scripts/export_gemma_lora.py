"""Gemma LoRA export entry point.

Thin wrapper around scripts/export.py:export_model_dir(), which handles both
LoRA adapter directories and full model directories transparently.

Called by:
- Direct execution: python export_gemma_lora.py <adapter_dir>

Calls to:
- scripts/export.py:export_model_dir() for all export logic
"""

import argparse

from gemma_tuner.scripts.export import export_model_dir


def main():
    parser = argparse.ArgumentParser(description="Merge and export a Gemma LoRA adapter to SafeTensors")
    parser.add_argument("adapter_dir", help="Path to a PEFT adapter directory (contains adapter_config.json)")
    parser.add_argument(
        "--revision",
        dest="model_revision",
        default=None,
        help="Optional specific base-model revision to pin during export",
    )
    args = parser.parse_args()
    export_model_dir(args.adapter_dir, model_revision=args.model_revision)


if __name__ == "__main__":
    main()
