"""Resumable Plan 33 MLX generation plus blinded single-candidate judging.

Reuses the frozen Plan 32 judge rendering, vote parsing, judge dispatch, and
quadrant views byte-for-byte. Adds the Plan 33 row panels (exposed-20 screen
or the 60-row selection panel, six modes each), a Plan 33 MLX model receipt,
and per-row peak MLX memory. Private prompts, outputs, and votes stay in an
ignored output directory; judges never see arm, checkpoint, or mode identity.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gemma_tuner.models.common.plan31_input_modes import MODES, render_plan31_input
from gemma_tuner.utils.plan32_projection import sha_file, verify_projection
from tools.run_plan32_literal_eval import (
    MODELS,
    append_jsonl,
    canonical,
    digest,
    image_views,
    parse_vote,
    prompt_sections,
    read_jsonl,
    render_judge,
    run_judge,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT / "data/datasets/tt-screenshot-plan33-literal-v3-deploy/epoch-1"
PROJECTION_SCHEMA = "plan33_literal_projection_v1"
JUDGE_PROMPT_SHA256 = "f29503bdbe4522db17e30a08cd593d0fc215bfc7bf238089a3c4e3f2e84f1b85"
MAX_TOKENS = 8192
RUNTIME = {"mlx": "0.32.2", "mlx-vlm": "0.7.1"}
FENCE = re.compile(r"```(?:json)?\n(.*)\n```", re.S)


def parse_plan33_vote(raw: str) -> dict[str, str]:
    """Plan 32 vote parsing, additionally accepting exactly one outer JSON code fence."""
    stripped = raw.strip()
    match = FENCE.fullmatch(stripped)
    return parse_vote(match.group(1) if match else stripped)


def panel_rows(panel: str, receipt_sha: str) -> list[tuple[str, dict[str, str], str]]:
    """Return (mode, row, image_sha) in frozen order: mode-major over the panel."""
    receipt = verify_projection(PROJECTION, expected_receipt_sha256=receipt_sha, epoch=1,
                                schema_version=PROJECTION_SCHEMA)
    hashes = {row["id"]: row["image_sha256"] for row in read_jsonl(PROJECTION / "validation-image-sha256.jsonl")}
    csv.field_size_limit(1 << 30)
    if panel == "exposed20":
        with (PROJECTION / "validation.csv").open(encoding="utf-8", newline="") as handle:
            base = list(csv.DictReader(handle))[:20]
        selection = {str(row["id"]) for row in read_jsonl(PROJECTION / "validation-60.jsonl")}
        if len(base) != 20 or selection & {row["id"] for row in base}:
            raise ValueError("exposed-20 must be the first 20 validation rows, disjoint from selection-60")
        per_mode = {mode: [{**row, "input_mode": mode} for row in base] for mode in MODES}
    elif panel == "selection60":
        per_mode = {}
        for mode in MODES:
            with (PROJECTION / f"validation-panel-{mode}.csv").open(encoding="utf-8", newline="") as handle:
                per_mode[mode] = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unknown Plan 33 panel {panel!r}")
    result = []
    for mode in MODES:
        for row in per_mode[mode]:
            image = (PROJECTION / row["image_path"]).resolve(strict=True)
            row["image_path"] = str(image)
            result.append((mode, row, hashes[row["id"]]))
    if receipt["test_rows_read"] != 0:
        raise ValueError("sealed test must stay closed")
    return result


def verify_model(path: Path) -> str:
    """Bind an MLX model directory to its Plan 33 receipt and the frozen runtime."""
    receipt_path = path / "plan33-mlx-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "plan33_mlx_model_v1" or receipt.get("precision") not in ("bfloat16", "6bit"):
        raise ValueError("model lacks a Plan 33 MLX receipt")
    for name, expected in receipt["files_sha256"].items():
        if sha_file(path / name) != expected:
            raise ValueError(f"MLX model file hash mismatch: {name}")
    for package, version in RUNTIME.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"runtime version drift: {package}")
    return sha_file(receipt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("exposed20", "selection60"), required=True)
    parser.add_argument("--projection-receipt-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="private arm label; never shown to judges")
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agent", type=Path, default=Path.home() / ".local/bin/agent")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-judge", action="store_true", help="generate only; judge later")
    args = parser.parse_args()

    out = args.out.resolve()
    ignored = subprocess.run(["git", "check-ignore", "-q", str(out / "generations.jsonl")], cwd=ROOT, check=False)
    if ignored.returncode != 0:
        raise ValueError("Plan 33 evaluation output must be ignored by Git")
    if not args.no_judge and not args.agent.is_file():
        raise ValueError("judge agent is missing")
    rows = panel_rows(args.panel, args.projection_receipt_sha256)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    model_path = args.model.resolve()
    model_sha = verify_model(model_path)
    system, user = prompt_sections(args.prompt_template.resolve(), JUDGE_PROMPT_SHA256)
    config_sha = digest(canonical({
        "schema": "plan33_eval_config_v1", "panel": args.panel, "arm": args.arm,
        "projection": args.projection_receipt_sha256, "model": model_sha,
        "judge_prompt": JUDGE_PROMPT_SHA256, "max_tokens": MAX_TOKENS,
        "thinking": False, "temperature": 0.0, "modes": MODES, "runtime": RUNTIME,
    }))
    generation_path, votes_path = out / "generations.jsonl", out / "votes.jsonl"
    generations = read_jsonl(generation_path)
    expected = [(mode, row["id"]) for mode, row, _ in rows]
    if ([(item["mode"], item["id"]) for item in generations] != expected[: len(generations)]
            or any(item.get("config_sha256") != config_sha for item in generations)):
        raise ValueError("generation ledger conflicts with frozen row order/config")
    seen_votes = {item["key"] for item in read_jsonl(votes_path)}

    def judge(generated: dict[str, Any], row: dict[str, str], images: list[Path], judge_name: str) -> None:
        prompt = render_judge(system, user, row, generated["text"])
        key = digest(canonical({
            "config": config_sha, "mode": generated["mode"], "id": generated["id"],
            "candidate": generated["text_sha256"], "prompt": digest(prompt),
            "images": [sha_file(image) for image in images], "selector": MODELS[judge_name],
        }))
        if key in seen_votes:
            return
        try:
            raw, latency = run_judge(args.agent, MODELS[judge_name], prompt, images)
            try:
                verdict, state = parse_plan33_vote(raw), "valid"
            except (ValueError, json.JSONDecodeError):
                verdict, state = None, "invalid"
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            raw, latency, verdict, state = "", None, None, type(error).__name__
        append_jsonl(votes_path, {
            "schema_version": "plan33_single_vote_v1", "key": key, "config_sha256": config_sha,
            "mode": generated["mode"], "id": generated["id"], "judge": judge_name,
            "selector": MODELS[judge_name], "state": state, "verdict": verdict,
            "raw_output": raw, "raw_sha256": digest(raw), "latency_seconds": latency,
        })
        seen_votes.add(key)

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(str(model_path))
    from tools.plan33_mlx_vision_lora import VISION_LORA_FILE, apply_vision_lora

    if (model_path / VISION_LORA_FILE).exists():
        print(canonical({"event": "vision_lora", "applied": apply_vision_lora(model, model_path)}), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = []
        for index, (mode, row, image_sha) in enumerate(rows):
            views, files = image_views(Path(row["image_path"]), out / "views", image_sha)
            if index < len(generations):
                generated = generations[index]
            else:
                selected, messages = render_plan31_input(
                    mode=mode, full_prompt=row["prompt"], system_prompt=row["system_prompt"], full_views=views,
                )
                model_prompt = apply_chat_template(
                    processor, model.config, messages, num_images=len(selected), enable_thinking=False
                )
                mx.reset_peak_memory()
                started = time.monotonic()
                result = generate(model=model, processor=processor, prompt=model_prompt, image=selected,
                                  max_tokens=MAX_TOKENS, temperature=0.0, enable_thinking=False, verbose=False)
                generated = {
                    "schema_version": "plan33_single_generation_v1", "config_sha256": config_sha,
                    "mode": mode, "id": row["id"], "image_sha256": image_sha,
                    "prompt_sha256": digest(str(model_prompt)), "prompt_tokens": result.prompt_tokens,
                    "text": result.text, "text_sha256": digest(result.text),
                    "finish_reason": result.finish_reason, "generation_tokens": result.generation_tokens,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_mlx_bytes": int(mx.get_peak_memory()),
                }
                append_jsonl(generation_path, generated)
                print(canonical({"event": "generation", "completed": index + 1, "mode": mode,
                                 "tokens": result.generation_tokens, "finish": result.finish_reason}), flush=True)
            if not args.no_judge:
                pending.extend(pool.submit(judge, generated, row, files, name) for name in MODELS)
        for future in pending:
            future.result()
    print(canonical({"generations": len(read_jsonl(generation_path)), "votes": len(read_jsonl(votes_path)),
                     "config_sha256": config_sha}), flush=True)


if __name__ == "__main__":
    main()
