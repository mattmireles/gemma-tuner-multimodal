"""Launch one bounded Plan 31 continuation from an explicit frozen profile."""

from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve(strict=True)
    os.environ["GEMMA_TUNER_CONFIG"] = str(config_path)

    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.utils.checkpoints import verify_complete_checkpoint

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(config_path):
        raise FileNotFoundError(config_path)
    profile = load_profile_config(config, args.profile)
    checkpoint = Path(str(profile["resume_from_checkpoint"])).resolve(strict=True)
    verify_complete_checkpoint(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) != int(profile["telemetry_start_step"]):
        raise ValueError("resume checkpoint does not match the bounded segment start")
    if int(profile["stop_after_step"]) <= int(profile["telemetry_start_step"]):
        raise ValueError("segment stop does not follow the sealed resume step")

    import transformers

    if transformers.__version__ != str(profile["required_transformers_version"]):
        raise RuntimeError("the corrected Transformers version is required")
    if args.preflight_only:
        print(f"preflight_ok start={state['global_step']} stop={profile['stop_after_step']}")
        return

    from gemma_tuner.models.gemma.finetune import main as train

    train(profile, str(args.output_dir))


if __name__ == "__main__":
    run()
