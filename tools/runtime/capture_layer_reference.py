#!/usr/bin/env python3
"""Capture complete Gemma 4 E4B decoder-layer stages from a frozen receipt."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--hidden-receipt",
        type=Path,
        help="optional separate receipt containing decoder.hidden_* tensors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import DynamicCache, Gemma4ForConditionalGeneration
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    args = parse_args()
    with ExitStack() as stack:
        receipt = stack.enter_context(safe_open(args.receipt, framework="pt"))
        hidden_receipt = receipt
        if args.hidden_receipt is not None:
            hidden_receipt = stack.enter_context(safe_open(args.hidden_receipt, framework="pt"))
        model_input_cpu = hidden_receipt.get_tensor("decoder.hidden_0")
        hidden_cpu = hidden_receipt.get_tensor(f"decoder.hidden_{args.layer}")
        expected = hidden_receipt.get_tensor(f"decoder.hidden_{args.layer + 1}")
        input_ids_cpu = receipt.get_tensor("processor.input_ids")
        mm_types_cpu = receipt.get_tensor("processor.mm_token_type_ids")
        attention_mask_cpu = receipt.get_tensor("processor.attention_mask")
        shared_states_cpu = {}
        for owner in (22, 23):
            key_name = f"cache.prefill.shared_{owner}.key"
            value_name = f"cache.prefill.shared_{owner}.value"
            if key_name in receipt.keys() and value_name in receipt.keys():
                shared_states_cpu[owner] = (
                    receipt.get_tensor(key_name),
                    receipt.get_tensor(value_name),
                )

    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).eval()
    language_model = model.model.language_model
    llm_input_ids_cpu = input_ids_cpu.clone()
    llm_input_ids_cpu[mm_types_cpu != 0] = language_model.padding_idx
    # PyTorch/MPS silently wraps byte offsets for this 5.64 GB table once a
    # row begins beyond 2^32 bytes. Build the reference gather on CPU so token
    # IDs above 199,728 retain their official checkpoint values.
    with torch.inference_mode():
        raw_per_layer_cpu = language_model.embed_tokens_per_layer(llm_input_ids_cpu)
        raw_per_layer_cpu = raw_per_layer_cpu.reshape(
            *llm_input_ids_cpu.shape,
            language_model.config.num_hidden_layers,
            language_model.config.hidden_size_per_layer_input,
        )
    model.to(args.device)
    model_input = model_input_cpu.to(args.device)
    hidden = hidden_cpu.to(args.device)
    attention_mask = attention_mask_cpu.to(args.device)
    raw_per_layer = raw_per_layer_cpu.to(args.device)
    layer = language_model.layers[args.layer]
    layer_type = language_model.config.layer_types[args.layer]

    captured: dict[str, Any] = {}

    def hook(name: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach()

        return capture

    modules = {
        "input_norm": layer.input_layernorm,
        "q_proj": layer.self_attn.q_proj,
        "q_norm": layer.self_attn.q_norm,
        "k_proj": layer.self_attn.k_proj,
        "k_norm": layer.self_attn.k_norm,
        "v_proj": layer.self_attn.v_proj,
        "v_norm": layer.self_attn.v_norm,
        "o_proj": layer.self_attn.o_proj,
        "post_attention_norm": layer.post_attention_layernorm,
        "pre_feedforward_norm": layer.pre_feedforward_layernorm,
        "mlp": layer.mlp,
        "post_feedforward_norm": layer.post_feedforward_layernorm,
        "per_layer_gate": layer.per_layer_input_gate,
        "per_layer_projection": layer.per_layer_projection,
        "post_per_layer_norm": layer.post_per_layer_input_norm,
    }
    handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
    try:
        projected_per_layer = language_model.per_layer_model_projection(model_input)
        projected_per_layer = projected_per_layer * language_model.per_layer_model_projection_scale
        projected_per_layer = projected_per_layer.reshape(
            *model_input.shape[:-1],
            language_model.config.num_hidden_layers,
            language_model.config.hidden_size_per_layer_input,
        )
        projected_per_layer = language_model.per_layer_projection_norm(projected_per_layer)
        per_layer_inputs = (projected_per_layer + raw_per_layer) * language_model.per_layer_input_scale
        per_layer_input = per_layer_inputs[:, :, args.layer, :]
        position_ids = torch.arange(hidden.shape[1], device=args.device).unsqueeze(0)
        position_embeddings = language_model.rotary_emb(hidden, position_ids, layer_type)
        mask_factory = (
            create_sliding_window_causal_mask if layer_type == "sliding_attention" else create_causal_mask
        )
        attention_bias = mask_factory(
            config=language_model.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        reference_cache = DynamicCache(config=language_model.config)
        reference_cache.shared_layers = {
            owner: (key.to(args.device), value.to(args.device))
            for owner, (key, value) in shared_states_cpu.items()
        }
        with torch.inference_mode():
            raw_output = layer(
                hidden,
                per_layer_input,
                position_embeddings=position_embeddings,
                attention_mask=attention_bias,
                position_ids=position_ids,
                past_key_values=reference_cache,
            )
            output = language_model.norm(raw_output) if args.layer + 1 == len(language_model.layers) else raw_output
        torch.mps.synchronize() if args.device == "mps" else None
    finally:
        for handle in handles:
            handle.remove()

    prefix = f"layer{args.layer}"
    tensors = {f"{prefix}.{name}": value.detach().cpu().contiguous() for name, value in captured.items()}
    tensors[f"{prefix}.per_layer_token"] = raw_per_layer[:, :, args.layer, :].detach().cpu().contiguous()
    tensors[f"{prefix}.per_layer_projected"] = projected_per_layer[:, :, args.layer, :].detach().cpu().contiguous()
    tensors[f"{prefix}.per_layer_input"] = per_layer_input.detach().cpu().contiguous()
    tensors[f"{prefix}.raw_output"] = raw_output.detach().cpu().contiguous()
    tensors[f"{prefix}.output"] = output.detach().cpu().contiguous()
    tensors[f"{prefix}.expected_output"] = expected.contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)
    difference = (tensors[f"{prefix}.output"].float() - expected.float()).abs()
    print(
        f"saved {len(tensors)} tensors; corrected-vs-original mae={difference.mean().item():.10f} "
        f"max_abs={difference.max().item():.10f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
