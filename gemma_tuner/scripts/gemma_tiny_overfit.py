#!/usr/bin/env python3
"""
Tiny overfit sanity run for Gemma 3n LoRA on MPS.

- Uses config.ini [profile:gemma-lora-test] with small limits
- Trains on a tiny subset (e.g., 16 samples) and expects loss to decrease

Usage:
  python scripts/gemma_tiny_overfit.py --profile gemma-lora-test --max-samples 32
"""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path

from gemma_tuner.core.config import load_profile_config
from gemma_tuner.models.gemma.finetune import main as gemma_train


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma tiny overfit")
    ap.add_argument("--profile", default="gemma-lora-test", help="Profile name in config.ini")
    ap.add_argument("--config", default="config/config.ini", help="Path to config.ini")
    ap.add_argument("--max-samples", type=int, default=32, help="Limit training samples")
    ap.add_argument("--output", default="output/gemma_tiny_overfit", help="Output directory")
    args = ap.parse_args()

    # Load profile
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    profile = load_profile_config(cfg, args.profile)
    # Apply tiny limits and stability settings
    profile["max_samples"] = args.max_samples
    profile["logging_steps"] = 5
    profile["num_train_epochs"] = 1
    profile["gradient_accumulation_steps"] = 8

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Call Gemma trainer entry
    gemma_train(profile, args.output)
    print("[OK] Tiny overfit run completed. Check logs and loss trend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
