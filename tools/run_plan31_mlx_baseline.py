"""Resume the frozen 20-row, six-mode Plan 31 MLX baseline privately."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image

from gemma_tuner.models.common.plan31_input_modes import MODES, render_plan31_input


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def views_for(path: Path) -> list[Image.Image]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    x, y = image.width // 2, image.height // 2
    return [image, image.crop((0, 0, x, y)), image.crop((x, 0, image.width, y)),
            image.crop((0, y, x, image.height)), image.crop((x, y, image.width, image.height))]


def read_ledger(path: Path, cases: list[dict], model_hash: str) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("partial Plan 31 baseline ledger line")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = [(mode, case["context_id"]) for mode in MODES for case in cases]
    actual = [(row["mode"], row["context_id"]) for row in rows]
    if actual != expected[:len(rows)] or any(row.get("model_hash") != model_hash for row in rows):
        raise ValueError("Plan 31 baseline ledger conflicts with frozen order or model")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--cases-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=120)
    args = parser.parse_args()
    if sha256(args.cases) != args.cases_sha256:
        raise ValueError("frozen case file changed")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines()]
    if len(cases) != 20 or len({case["context_id"] for case in cases}) != 20:
        raise ValueError("expected exactly 20 unique exposed diagnostic cases")
    if not 1 <= args.max_rows <= 20 * len(MODES):
        raise ValueError("max-rows must be within the frozen 120-row panel")
    receipt = json.loads((args.model / "conversion-receipt.json").read_text())
    if receipt.get("profile") != "plan30-corrected":
        raise ValueError("baseline requires corrected Plan 30 checkpoint")
    model_hash = receipt["conversion_identity_sha256"]
    rows = read_ledger(args.out, cases, model_hash)
    if len(rows) == 20 * len(MODES):
        return
    model, processor = load(str(args.model))
    for mode in MODES:
        for case in cases:
            ordinal = MODES.index(mode) * 20 + cases.index(case)
            if ordinal < len(rows):
                continue
            if ordinal >= args.max_rows:
                return
            image_path = Path("/") / case["image_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            selected, messages = render_plan31_input(
                mode=mode, full_prompt=case["user_prompt"],
                system_prompt=case["system_prompt"], full_views=views_for(image_path),
            )
            prompt = apply_chat_template(
                processor, model.config, messages,
                num_images=len(selected), enable_thinking=False,
            )
            started = time.perf_counter()
            result = generate(
                model=model, processor=processor, prompt=prompt, image=selected,
                max_tokens=args.max_tokens, temperature=0.0,
                enable_thinking=False, verbose=False,
            )
            row = {
                "mode": mode, "context_id": case["context_id"], "model_hash": model_hash,
                "image_sha256": sha256(image_path),
                "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
                "views": len(selected), "generated_text": result.text,
                "finish_reason": result.finish_reason,
                "generation_tokens": result.generation_tokens,
                "elapsed_seconds": time.perf_counter() - started,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(row)
            print(json.dumps({"completed": len(rows), "mode": mode,
                              "tokens": result.generation_tokens,
                              "stop": result.finish_reason}), flush=True)


if __name__ == "__main__":
    main()
