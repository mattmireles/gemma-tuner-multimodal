"""Resume one-candidate Plan 32 MLX generation and blind screenshot validation.

Private prompts, images, outputs, and votes belong in an ignored output directory.
The frozen 60-row panel, source hashes, conversion receipt, and judge prompt
are verified before dispatch. Judges see one candidate and never see its arm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from gemma_tuner.models.common.plan31_input_modes import (
    MODES,
    OCR_CLOSE,
    OCR_OPEN,
    render_plan31_input,
    without_ocr,
)
from gemma_tuner.utils.plan32_projection import sha_file, verify_projection

MODELS = {"grok": "cursor-grok-4.6-medium", "composer": "composer-2.5[fast=false]"}
STATUSES = {"APPROVED", "REJECTED", "NEEDS_REVIEW"}
LOCK = threading.Lock()
GUARDED_CHILD_ENV = "PLAN32_ANE_GUARDED_CHILD"
ROW_CHILD_ENV = "PLAN32_ANE_ROW_CHILD"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"partial JSONL ledger: {path.name}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def supervise_ane_child(
    *,
    generation_path: Path,
    target_generations: int,
    max_swap_used_mb: float,
    max_swap_growth_mb: float,
    min_free_memory_percent: float,
) -> int:
    from tools.plan32_ane_vision import host_pressure_snapshot, pressure_violation

    while True:
        before = len(read_jsonl(generation_path))
        if before >= target_generations:
            return 0
        baseline = host_pressure_snapshot()
        initial = pressure_violation(
            baseline,
            baseline,
            max_swap_used_mb=max_swap_used_mb,
            max_swap_growth_mb=max_swap_growth_mb,
            min_free_memory_percent=min_free_memory_percent,
        )
        if initial:
            raise RuntimeError(f"unsafe ANE host pressure before launch: {initial}")
        environment = os.environ.copy()
        environment[GUARDED_CHILD_ENV] = "1"
        environment[ROW_CHILD_ENV] = "1"
        child = subprocess.Popen([sys.executable, *sys.argv], env=environment)
        while child.poll() is None:
            time.sleep(2)
            try:
                snapshot = host_pressure_snapshot()
                violation = pressure_violation(
                    snapshot,
                    baseline,
                    max_swap_used_mb=max_swap_used_mb,
                    max_swap_growth_mb=max_swap_growth_mb,
                    min_free_memory_percent=min_free_memory_percent,
                )
            except Exception as error:  # fail closed if host telemetry disappears
                snapshot = None
                violation = f"host telemetry failed: {type(error).__name__}: {error}"
            if violation:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                print(
                    canonical(
                        {
                            "event": "pressure_guard_terminated_child",
                            "reason": violation,
                            "baseline": baseline,
                            "snapshot": snapshot,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return 75
        if child.returncode != 0:
            return int(child.returncode)
        after = len(read_jsonl(generation_path))
        if after != before + 1:
            raise RuntimeError(
                f"ANE row child must append exactly one generation: before={before}, after={after}"
            )


def prompt_sections(path: Path, expected_sha: str) -> tuple[str, str]:
    if sha_file(path) != expected_sha:
        raise ValueError("Plan 32 judge prompt hash mismatch")
    source = path.read_text(encoding="utf-8")
    system = source.split("## System instruction\n", 1)[1].split("\n## User message template\n", 1)[0].strip()
    user = source.split("## User message template\n", 1)[1].split("\n## Rendering contract\n", 1)[0].strip()
    for marker in (
        "${canonical_full_system_instruction}",
        "${canonical_full_user_message}",
        "${structured_ocr}",
        "${primary_analyst_output}",
    ):
        if user.count(marker) != 1:
            raise ValueError(f"judge template marker count changed: {marker}")
    return system, user


def parse_vote(raw: str) -> dict[str, str]:
    value = json.loads(raw.strip())
    if not isinstance(value, dict) or set(value) != {"status", "feedback"}:
        raise ValueError("judge response must have status and feedback only")
    if value["status"] not in STATUSES or not isinstance(value["feedback"], str):
        raise ValueError("judge verdict or feedback invalid")
    if not value["feedback"].strip():
        raise ValueError("judge feedback is empty")
    return {"status": value["status"], "feedback": value["feedback"]}


def render_judge(system: str, user: str, row: dict[str, str], candidate: str) -> str:
    full_prompt = row["prompt"]
    no_ocr = without_ocr(full_prompt)
    ocr = full_prompt.split(OCR_OPEN, 1)[1].split(OCR_CLOSE, 1)[0]
    replacements = {
        "${canonical_full_system_instruction}": row["system_prompt"],
        "${canonical_full_user_message}": no_ocr,
        "${structured_ocr}": ocr,
        "${primary_analyst_output}": candidate,
    }
    # Check the template, not the rendered text: OCR or model output may contain "${" literally.
    template = user
    for marker in replacements:
        if marker not in template:
            raise ValueError(f"judge template lacks marker {marker}")
        template = template.replace(marker, "", 1)
    if "${" in template:
        raise ValueError("unfilled judge template marker")
    rendered = user
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value, 1)
    return system + "\n\n" + rendered


def frozen_rows(
    root: Path, receipt_sha: str, *, subset_only: bool = False, panel_rows: int = 60
) -> list[tuple[str, dict[str, str], str]]:
    if subset_only:
        receipt_path = root / "projection.receipt.json"
        if sha_file(receipt_path) != receipt_sha:
            raise ValueError("Plan 32 projection receipt changed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("epoch") != 2 or receipt.get("fresh_panel_rows") != 60):
            raise ValueError("Plan 32 frozen panel receipt changed")
        for name, expected in receipt["outputs_sha256"].items():
            if sha_file(root / name) != expected:
                raise ValueError(f"Plan 32 projection file changed: {name}")
    else:
        receipt = verify_projection(root, expected_receipt_sha256=receipt_sha, epoch=2)
    panel = read_jsonl(root / "validation-60.jsonl")
    ids = [str(item["id"]) for item in panel]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("frozen Plan 32 panel is not 60 unique rows")
    selected_ids = set(ids[:panel_rows])
    result = []
    for mode in (("full",) if subset_only else MODES):
        path = root / f"validation-panel-{mode}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if [row["id"] for row in rows] != ids:
            raise ValueError(f"{mode} panel order changed")
        for row, item in zip(rows, panel):
            if subset_only and row["id"] not in selected_ids:
                continue
            image = (root / row["image_path"]).resolve(strict=True)
            if sha_file(image) != item["image_sha256"]:
                raise ValueError("frozen screenshot hash changed")
            row["image_path"] = str(image)
            result.append((mode, row, item["image_sha256"]))
    if receipt["outputs_sha256"]["validation-60.jsonl"] != sha_file(root / "validation-60.jsonl"):
        raise ValueError("frozen panel receipt mismatch")
    return result


def verify_model(path: Path) -> str:
    conversion_path = path / "conversion-receipt.json"
    quantization_path = path / "quantization-receipt.json"
    if conversion_path.exists():
        receipt_path = conversion_path
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("profile") != "plan30-corrected"
            or receipt.get("precision") != "bfloat16"
            or receipt.get("quantization") is not None
            or receipt.get("coverage", {}).get("mapped_inference_active_targets") != 371
        ):
            raise ValueError("model conversion is not corrected BF16 with all 371 targets")
        expected_files = receipt["output"]["files_sha256"]
    else:
        receipt_path = quantization_path
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        bits = receipt.get("bits")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version") != "plan32_decoder_quantization_v1"
            or bits not in (4, 6)
            or receipt.get("quantized_language_modules") != 345
            or receipt.get("quantized_nonlanguage_modules") != 0
            or config.get("quantization") != {"group_size": 64, "bits": bits, "mode": "affine"}
        ):
            raise ValueError("model is not the frozen Plan 32 decoder-only quantization")
        expected_files = receipt["output_sha256"]
    for name, expected in expected_files.items():
        if sha_file(path / name) != expected:
            raise ValueError(f"MLX model file hash mismatch: {name}")
    for package, version in {"mlx": "0.32.2", "mlx-vlm": "0.7.1"}.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"runtime version drift: {package}")
    return sha_file(receipt_path)


def image_views(path: Path, cache: Path, image_sha: str) -> tuple[list[Image.Image], list[Path]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    x, y = image.width // 2, image.height // 2
    views = [
        image,
        image.crop((0, 0, x, y)),
        image.crop((x, 0, image.width, y)),
        image.crop((0, y, x, image.height)),
        image.crop((x, y, image.width, image.height)),
    ]
    directory = cache / image_sha
    directory.mkdir(parents=True, exist_ok=True)
    files = [path, *(directory / f"quadrant-{index}.png" for index in range(1, 5))]
    for view, destination in zip(views[1:], files[1:]):
        if not destination.exists():
            view.save(destination, format="PNG")
    return views, files


def run_judge(agent: Path, selector: str, prompt: str, images: list[Path]) -> tuple[str, float]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="plan32-judge-") as temp:
        workspace = Path(temp)
        copied = []
        for index, image in enumerate(images):
            destination = workspace / f"view-{index}{image.suffix.lower()}"
            shutil.copy2(image, destination)
            copied.append(destination)
        instruction = (
            prompt
            + "\n\nRead only these five image files: "
            + " ".join(str(path) for path in copied)
            + "\nDo not use shell, search, MCP, or other files. Return only the JSON response specified above."
        )
        completed = subprocess.run(
            [
                str(agent),
                "-p",
                instruction,
                "--model",
                selector,
                "--mode",
                "ask",
                "--workspace",
                str(workspace),
                "--print",
                "--output-format",
                "text",
                "--trust",
                "--sandbox",
                "enabled",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"judge process failed with exit {completed.returncode}")
    return completed.stdout.strip(), time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--projection-receipt-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ane-vision-root", type=Path)
    parser.add_argument("--ane-source-conversion", type=Path)
    parser.add_argument("--ane-compiled-root", type=Path)
    parser.add_argument("--max-swap-used-mb", type=float, default=4096)
    parser.add_argument("--max-swap-growth-mb", type=float, default=2048)
    parser.add_argument("--min-free-memory-percent", type=float, default=3)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--prompt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agent", type=Path, default=Path.home() / ".local/bin/agent")
    parser.add_argument("--max-rows", type=int, default=360)
    parser.add_argument("--panel-rows", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--prefill-step-size", type=int)
    parser.add_argument("--latency-only", action="store_true", help="Generate without invoking judges")
    args = parser.parse_args()
    if not 1 <= args.max_rows <= 360 or args.max_tokens != 8192:
        raise ValueError("Plan 32 evaluation requires 1–360 rows and max 8192 tokens")
    if not 1 <= args.panel_rows <= 60:
        raise ValueError("Plan 32 evaluation requires 1–60 validation rows per mode")
    if args.prefill_step_size is not None and not 32 <= args.prefill_step_size <= 2048:
        raise ValueError("prefill step size must be between 32 and 2048")
    if args.ane_vision_root and os.environ.get(GUARDED_CHILD_ENV) != "1":
        raise SystemExit(
            supervise_ane_child(
                generation_path=args.out.resolve() / "generations.jsonl",
                target_generations=min(
                    args.max_rows,
                    args.panel_rows if args.latency_only else args.panel_rows * len(MODES),
                ),
                max_swap_used_mb=args.max_swap_used_mb,
                max_swap_growth_mb=args.max_swap_growth_mb,
                min_free_memory_percent=args.min_free_memory_percent,
            )
        )
    root, model_path, out = args.projection.resolve(), args.model.resolve(), args.out.resolve()
    if not args.latency_only:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(out / "generations.jsonl")],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        if not args.agent.is_file() or ignored.returncode != 0:
            raise ValueError("missing agent or output is not ignored by Git")
    rows = [
        item
        for item in frozen_rows(
            root, args.projection_receipt_sha256,
            subset_only=args.latency_only, panel_rows=args.panel_rows,
        )
        if item[1]["id"] in {
            row["id"] for row in read_jsonl(root / "validation-60.jsonl")[: args.panel_rows]
        }
    ]
    model_sha = verify_model(model_path)
    ane_receipts = None
    ane_compiled_receipts = None
    if args.ane_vision_root:
        if not args.ane_source_conversion or not args.ane_compiled_root:
            raise ValueError(
                "ANE run requires its exact BF16 source receipt and target-local compiled root"
            )
        source = json.loads(args.ane_source_conversion.read_text(encoding="utf-8"))
        if (model_path / "quantization-receipt.json").is_file():
            quant = json.loads((model_path / "quantization-receipt.json").read_text(encoding="utf-8"))
            if (
                quant["source_conversion_identity_sha256"] != source["conversion_identity_sha256"]
                or quant["source_adapter_weights_sha256"] != source["adapter"]["weights_sha256"]
            ):
                raise ValueError("ANE vision and quantized decoder are not the same checkpoint")
        ane_receipts = {
            path.parent.name: sha_file(path)
            for path in sorted(args.ane_vision_root.glob("plan32-cp616-ane-*/export-receipt.json"))
        }
        if not ane_receipts:
            raise ValueError("no verified ANE export receipts")
        ane_compiled_receipts = {
            path.name: sha_file(path)
            for path in sorted(args.ane_compiled_root.glob("*.json"))
        }
        if not ane_compiled_receipts:
            raise ValueError("no target-local compiled ANE receipts")
    system, user = prompt_sections(args.prompt_template.resolve(), args.prompt_sha256)
    config_sha = digest(
        canonical(
            {
                "panel": sha_file(root / "validation-60.jsonl"),
                "projection": args.projection_receipt_sha256,
                "model": model_sha,
                "ane_receipts": ane_receipts,
                "ane_compiled_receipts": ane_compiled_receipts,
                "ane_source_conversion": sha_file(args.ane_source_conversion)
                if args.ane_source_conversion else None,
                "judge_prompt": args.prompt_sha256,
                "max_tokens": args.max_tokens,
                "prefill_step_size": args.prefill_step_size,
                "thinking": False,
                "temperature": 0.0,
                "modes": MODES,
                "panel_rows": args.panel_rows,
                "latency_only": args.latency_only,
                "host_pressure_gate": {
                    "max_swap_used_mb": args.max_swap_used_mb,
                    "max_swap_growth_mb": args.max_swap_growth_mb,
                    "min_free_memory_percent": args.min_free_memory_percent,
                },
            }
        )
    )
    generation_path, votes_path = out / "generations.jsonl", out / "votes.jsonl"
    generations = read_jsonl(generation_path)
    expected = [(mode, row["id"]) for mode, row, _ in rows]
    if [(item["mode"], item["id"]) for item in generations] != expected[: len(generations)] or any(
        item.get("config_sha256") != config_sha for item in generations
    ):
        raise ValueError("generation ledger conflicts with frozen row order/config")
    votes = read_jsonl(votes_path)
    vote_keys = [item["key"] for item in votes]
    if len(vote_keys) != len(set(vote_keys)):
        raise ValueError("duplicate judge vote key")
    seen_votes = set(vote_keys)

    def judge(generated: dict[str, Any], row: dict[str, str], images: list[Path], judge_name: str) -> None:
        prompt = render_judge(system, user, row, generated["text"])
        key = digest(
            canonical(
                {
                    "config": config_sha,
                    "mode": generated["mode"],
                    "id": generated["id"],
                    "candidate": generated["text_sha256"],
                    "prompt": digest(prompt),
                    "images": [sha_file(image) for image in images],
                    "selector": MODELS[judge_name],
                }
            )
        )
        if key in seen_votes:
            return
        try:
            raw, latency = run_judge(args.agent, MODELS[judge_name], prompt, images)
            try:
                verdict = parse_vote(raw)
                state = "valid"
            except (ValueError, json.JSONDecodeError):
                verdict, state = None, "invalid"
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            raw, latency, verdict, state = "", None, None, type(error).__name__
        record = {
            "schema_version": "plan32_single_vote_v1",
            "key": key,
            "config_sha256": config_sha,
            "mode": generated["mode"],
            "id": generated["id"],
            "judge": judge_name,
            "selector": MODELS[judge_name],
            "state": state,
            "verdict": verdict,
            "raw_output": raw,
            "raw_sha256": digest(raw),
            "latency_seconds": latency,
        }
        append_jsonl(votes_path, record)
        seen_votes.add(key)
        print(canonical({"event": "judge", "mode": generated["mode"], "judge": judge_name, "state": state}), flush=True)

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    if args.ane_vision_root:
        from tools.plan32_ane_vision import assert_safe_host_pressure

        assert_safe_host_pressure(
            max_swap_used_mb=args.max_swap_used_mb,
            min_free_memory_percent=args.min_free_memory_percent,
        )
    model, processor = load(str(model_path))
    if args.ane_vision_root:
        from tools.plan32_ane_vision import ANEVisionTower

        model.vision_tower = ANEVisionTower(
            args.ane_vision_root.resolve(),
            args.ane_source_conversion.resolve(),
            model.vision_tower,
            out / "ane-real-image-parity.jsonl",
            compiled_root=args.ane_compiled_root.resolve(),
            max_swap_used_mb=args.max_swap_used_mb,
            min_free_memory_percent=args.min_free_memory_percent,
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = []
        for index, (mode, row, image_sha) in enumerate(rows[: args.max_rows]):
            if index < len(generations):
                generated = generations[index]
                _, files = image_views(Path(row["image_path"]), out / "views", image_sha)
            else:
                if args.ane_vision_root:
                    assert_safe_host_pressure(
                        max_swap_used_mb=args.max_swap_used_mb,
                        min_free_memory_percent=args.min_free_memory_percent,
                    )
                views, files = image_views(Path(row["image_path"]), out / "views", image_sha)
                selected, messages = render_plan31_input(
                    mode=mode,
                    full_prompt=row["prompt"],
                    system_prompt=row["system_prompt"],
                    full_views=views,
                )
                model_prompt = apply_chat_template(
                    processor, model.config, messages, num_images=len(selected), enable_thinking=False
                )
                started = time.monotonic()
                result = generate(
                    model=model,
                    processor=processor,
                    prompt=model_prompt,
                    image=selected,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    enable_thinking=False,
                    prefill_step_size=args.prefill_step_size,
                    verbose=False,
                )
                generated = {
                    "schema_version": "plan32_single_generation_v1",
                    "config_sha256": config_sha,
                    "mode": mode,
                    "id": row["id"],
                    "image_sha256": image_sha,
                    "prompt_sha256": digest(str(model_prompt)),
                    "text": result.text,
                    "text_sha256": digest(result.text),
                    "finish_reason": result.finish_reason,
                    "generation_tokens": result.generation_tokens,
                    "elapsed_seconds": time.monotonic() - started,
                }
                append_jsonl(generation_path, generated)
                print(
                    canonical(
                        {
                            "event": "generation",
                            "completed": index + 1,
                            "mode": mode,
                            "tokens": result.generation_tokens,
                            "finish": result.finish_reason,
                        }
                    ),
                    flush=True,
                )
                if os.environ.get(ROW_CHILD_ENV) == "1":
                    if not args.latency_only:
                        for judge_name in MODELS:
                            pending.append(pool.submit(judge, generated, row, files, judge_name))
                    break
            if not args.latency_only:
                for judge_name in MODELS:
                    pending.append(pool.submit(judge, generated, row, files, judge_name))
        for future in pending:
            future.result()
    print(
        canonical(
            {
                "generations": len(read_jsonl(generation_path)),
                "votes": len(read_jsonl(votes_path)),
                "config_sha256": config_sha,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
