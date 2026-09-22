#!/usr/bin/env python3
"""Capture tensor goldens from a pinned Hugging Face Gemma 4 revision.

This tool intentionally writes only to an ignored receipt directory.  Tracked
fixture manifests contain hashes; private request bodies and media are resolved
from ``GEMMA_E4B_FIXTURE_ROOT`` or ``--private-root``.
"""

from __future__ import annotations

import argparse
import copy
import platform
import sys
import types
from pathlib import Path
from typing import Any

from reference_common import (
    canonical_json,
    load_manifest,
    load_private_fixture,
    private_root_from,
    select_fixture,
    sha256_file,
    stable_manifest_hash,
)


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(part) for part in value.split(",") if part.strip()})
    if not layers or layers[0] < 0:
        raise ValueError("--layers must contain non-negative layer indexes")
    return layers


def resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(torch_module: Any, name: str) -> Any:
    return {
        "bf16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "fp32": torch_module.float32,
    }[name]


def synchronize(torch_module: Any, device: str) -> None:
    if device == "mps":
        torch_module.mps.synchronize()
    elif device.startswith("cuda"):
        torch_module.cuda.synchronize()


def _load_audio(path: Path, sampling_rate: int) -> Any:
    import librosa

    samples, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return samples


def prepare_inputs(processor: Any, fixture: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from PIL import Image

    request = fixture["request"]
    messages = copy.deepcopy(request.get("messages"))
    if not isinstance(messages, list):
        raise ValueError("private request.messages must be a list")
    mode = fixture["mode"]
    media = fixture["_resolved_media"]
    if mode == "image_to_text":
        image_keys = sorted(media)
        user_messages = [message for message in messages if message.get("role") == "user"]
        if len(user_messages) != 1 or not isinstance(user_messages[0].get("content"), str):
            raise ValueError("image_to_text requires one string-valued user message")
        user_messages[0]["content"] = [
            *[{"type": "image"} for _ in image_keys],
            {"type": "text", "text": user_messages[0]["content"]},
        ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs: dict[str, Any] = {"text": [prompt], "padding": True, "return_tensors": "pt"}
    if mode == "image_to_text":
        image_keys = sorted(media)
        kwargs["images"] = [Image.open(media[key]).convert("RGB") for key in image_keys]
    elif mode == "audio_to_text":
        sampling_rate = int(request.get("sampling_rate", 16000))
        if set(media) != {"audio"}:
            raise ValueError("audio_to_text requires exactly one media entry named audio")
        kwargs["audio"] = [_load_audio(media["audio"], sampling_rate)]
        kwargs["sampling_rate"] = sampling_rate
    elif media:
        raise ValueError("text_to_text fixtures must not declare media")
    return dict(processor(**kwargs)), prompt


def move_inputs(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in inputs.items()}


def _capture_cache(
    torch_module: Any,
    cache: Any,
    tensors: dict[str, Any],
    prefix: str,
    *,
    capture_shared_states: bool = False,
    full_state_tensors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"type": type(cache).__name__}
    if hasattr(cache, "get_seq_length"):
        metadata["sequence_length"] = int(cache.get_seq_length())
    layers = getattr(cache, "layers", [])
    metadata["layer_count"] = len(layers)
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        metadata.setdefault("populated_layers", []).append(index)
        # A small deterministic slice is enough to detect ownership/layout drift.
        tensors[f"{prefix}.layer_{index}.key_edge"] = keys.detach().cpu()[..., :1, :8].contiguous()
        tensors[f"{prefix}.layer_{index}.value_edge"] = values.detach().cpu()[..., :1, :8].contiguous()
        if full_state_tensors is not None:
            full_state_tensors[f"{prefix}.layer_{index}.key"] = keys.detach().cpu().contiguous()
            full_state_tensors[f"{prefix}.layer_{index}.value"] = values.detach().cpu().contiguous()
    shared = getattr(cache, "shared_layers", {})
    metadata["shared_layer_indexes"] = sorted(int(index) for index in shared)
    for index, pair in sorted(shared.items()):
        tensors[f"{prefix}.shared_{index}.key_edge"] = pair[0].detach().cpu()[..., :1, :8].contiguous()
        tensors[f"{prefix}.shared_{index}.value_edge"] = pair[1].detach().cpu()[..., :1, :8].contiguous()
        if capture_shared_states:
            tensors[f"{prefix}.shared_{index}.key"] = pair[0].detach().cpu().contiguous()
            tensors[f"{prefix}.shared_{index}.value"] = pair[1].detach().cpu().contiguous()
    return metadata


def capture(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    manifest_rows = load_manifest(args.manifest)
    row = select_fixture(manifest_rows, args.fixture_id)
    private_root = private_root_from(args.private_root)
    fixture = load_private_fixture(row, private_root)
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)

    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).eval()
    safe_per_layer_gather = False
    if device == "mps":
        language_model = model.model.language_model
        per_layer_weight = language_model.embed_tokens_per_layer.weight.detach()
        if per_layer_weight.numel() * per_layer_weight.element_size() > 2**32:
            safe_per_layer_gather = True
            cpu_per_layer_weight = per_layer_weight
            cpu_scale = torch.tensor(
                language_model.embed_tokens_per_layer.scalar_embed_scale,
                dtype=cpu_per_layer_weight.dtype,
            )

            def safe_get_per_layer_inputs(
                self: Any,
                input_ids: Any,
                _inputs_embeds: Any,
            ) -> Any:
                if input_ids is None:
                    raise RuntimeError("safe oversized per-layer gather requires input_ids")
                ids_cpu = input_ids.detach().cpu()
                gathered = torch.nn.functional.embedding(ids_cpu, cpu_per_layer_weight) * cpu_scale
                gathered = gathered.reshape(
                    *ids_cpu.shape,
                    self.config.num_hidden_layers,
                    self.config.hidden_size_per_layer_input,
                )
                return gathered.to(input_ids.device)

            language_model.get_per_layer_inputs = types.MethodType(  # type: ignore[method-assign]
                safe_get_per_layer_inputs,
                language_model,
            )
    model.to(device)

    inputs, _ = prepare_inputs(processor, fixture)
    tensors: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            tensors[f"processor.{name}"] = value.detach().cpu().contiguous()
    device_inputs = move_inputs(inputs, device)
    synchronize(torch, device)
    with torch.inference_mode():
        outputs = model(
            **device_inputs,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
    synchronize(torch, device)

    hidden_states = outputs.hidden_states or ()
    hidden_tensors: dict[str, Any] = {}
    for index in args.layers:
        if index >= len(hidden_states):
            raise ValueError(f"requested hidden state {index}, but model returned {len(hidden_states)}")
        destination = hidden_tensors if args.hidden_output is not None else tensors
        destination[f"decoder.hidden_{index}"] = hidden_states[index].detach().cpu().contiguous()
    tensors["decoder.last_logits"] = outputs.logits[:, -1:, :].detach().cpu().contiguous()
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tensors["decoder.greedy_next_token"] = next_token.detach().cpu().contiguous()
    if outputs.image_hidden_states is not None:
        tensors["encoder.image_soft_tokens"] = outputs.image_hidden_states.detach().cpu().contiguous()
    if outputs.audio_hidden_states is not None:
        tensors["encoder.audio_soft_tokens"] = outputs.audio_hidden_states.detach().cpu().contiguous()

    full_cache_tensors: dict[str, Any] = {}
    cache_metadata = _capture_cache(
        torch,
        outputs.past_key_values,
        tensors,
        "cache.prefill",
        capture_shared_states=True,
        full_state_tensors=full_cache_tensors if args.full_cache_output is not None else None,
    )
    transition_metadata: dict[str, Any] = {"attempted": True}
    attention_mask = device_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(device_inputs["input_ids"], device=device)
    extended_mask = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)], dim=-1
    )
    cache_position = torch.tensor([int(cache_metadata.get("sequence_length", attention_mask.shape[-1]))], device=device)
    prepared = model.prepare_inputs_for_generation(
        next_token.to(device),
        past_key_values=outputs.past_key_values,
        attention_mask=extended_mask,
        cache_position=cache_position,
        use_cache=True,
    )
    for name, value in prepared.items():
        if isinstance(value, torch.Tensor):
            tensors[f"transition.{name}"] = value.detach().cpu().contiguous()
    synchronize(torch, device)
    with torch.inference_mode():
        next_outputs = model(**prepared, output_hidden_states=True, return_dict=True)
    synchronize(torch, device)
    tensors["decoder.transition_logits"] = next_outputs.logits[:, -1:, :].detach().cpu().contiguous()
    transition_token = next_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tensors["decoder.transition_greedy_next_token"] = transition_token.detach().cpu().contiguous()
    for index, hidden_state in enumerate(next_outputs.hidden_states or ()):
        tensors[f"decoder.transition_hidden_{index}"] = hidden_state.detach().cpu().contiguous()
    transition_metadata.update(
        _capture_cache(
            torch,
            next_outputs.past_key_values,
            tensors,
            "cache.transition",
            full_state_tensors=full_cache_tensors if args.full_cache_output is not None else None,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / f"{row['fixture_id']}.safetensors"
    save_file(tensors, str(tensor_path))
    hidden_metadata = None
    if args.hidden_output is not None:
        hidden_path = args.hidden_output.resolve()
        if hidden_path == tensor_path.resolve():
            raise ValueError("--hidden-output must differ from the primary tensor receipt")
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(hidden_tensors, str(hidden_path))
        hidden_metadata = {
            "path": str(hidden_path),
            "sha256": sha256_file(hidden_path),
            "tensor_names": sorted(hidden_tensors),
        }
    full_cache_metadata = None
    if args.full_cache_output is not None:
        full_cache_path = args.full_cache_output.resolve()
        if full_cache_path in {tensor_path.resolve(), args.hidden_output.resolve() if args.hidden_output else None}:
            raise ValueError("--full-cache-output must differ from the other tensor receipts")
        full_cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(full_cache_tensors, str(full_cache_path))
        full_cache_metadata = {
            "path": str(full_cache_path),
            "sha256": sha256_file(full_cache_path),
            "tensor_names": sorted(full_cache_tensors),
        }
    metadata = {
        "schema_version": "gemma4-e4b-reference-capture-v1",
        "fixture_id": row["fixture_id"],
        "mode": row["mode"],
        "manifest_sha256": stable_manifest_hash(manifest_rows),
        "model": {"id": args.model, "revision": args.revision, "dtype": args.dtype},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device,
            "torch": torch.__version__,
        },
        "capture": {
            "hidden_state_indexes": args.layers,
            "safe_per_layer_embedding_gather": safe_per_layer_gather,
            "cache_prefill": cache_metadata,
            "cache_transition": transition_metadata,
            "tensor_names": sorted([*tensors, *hidden_tensors]),
        },
        "tensors": {"path": tensor_path.name, "sha256": sha256_file(tensor_path)},
    }
    if hidden_metadata is not None:
        metadata["hidden_tensors"] = hidden_metadata
    if full_cache_metadata is not None:
        metadata["full_cache_tensors"] = full_cache_metadata
    metadata_path = args.output_dir / f"{row['fixture_id']}.json"
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-id")
    parser.add_argument("--private-root")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0,1,5,6,23,24,25,41,42"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runtime-receipts/reference"))
    parser.add_argument(
        "--hidden-output",
        type=Path,
        help="write requested decoder hidden states to a separate SafeTensors file",
    )
    parser.add_argument(
        "--full-cache-output",
        type=Path,
        help="write complete prefill and transition physical cache tensors separately",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest)
    row = select_fixture(rows, args.fixture_id)
    if args.validate_only:
        root = private_root_from(args.private_root)
        load_private_fixture(row, root)
        print(canonical_json({"fixture_id": row["fixture_id"], "manifest_sha256": stable_manifest_hash(rows)}))
        return 0
    print(canonical_json(capture(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
