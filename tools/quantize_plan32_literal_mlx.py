"""Quantize only the decoder of a sealed Plan 32 MLX BF16 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from mlx_vlm.convert import convert


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize(source: Path, output: Path, bits: int) -> dict:
    if bits not in (4, 6):
        raise ValueError("only 4-bit and 6-bit comparisons are frozen")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    receipt = json.loads((source / "conversion-receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("profile") != "plan30-corrected"
        or receipt.get("precision") != "bfloat16"
        or receipt.get("quantization") is not None
        or receipt.get("coverage", {}).get("mapped_inference_active_targets") != 371
    ):
        raise ValueError("source is not the complete corrected BF16 conversion")
    for name, expected in receipt["output"]["files_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")

    with tempfile.TemporaryDirectory(prefix="plan32-quant-source-", dir=output.parent) as temporary:
        staged = Path(temporary)
        for item in source.iterdir():
            if item.name.endswith(".safetensors"):
                (staged / item.name).symlink_to(item)
            elif item.name in {
                "config.json",
                "model.safetensors.index.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "generation_config.json",
                "chat_template.jinja",
                "README.md",
            }:
                shutil.copyfile(item, staged / item.name)
        convert(
            hf_path=str(staged),
            mlx_path=str(output),
            quantize=True,
            q_group_size=64,
            q_bits=bits,
            q_mode="affine",
            quant_method="rtn",
            dtype="bfloat16",
            quant_predicate=lambda path, _module: path.startswith("language_model."),
        )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization") != {"group_size": 64, "bits": bits, "mode": "affine"}:
        raise ValueError("quantization recipe mismatch")
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    keys = set(index["weight_map"])
    language = [key for key in keys if key.startswith("language_model.") and key.endswith(".scales")]
    nonlanguage = [key for key in keys if not key.startswith("language_model.") and key.endswith(".scales")]
    if len(language) != 345 or nonlanguage:
        raise ValueError(f"unexpected quantized modules: decoder={len(language)}, other={len(nonlanguage)}")
    output_hashes = {item.name: sha256(item) for item in output.iterdir() if item.is_file()}
    result = {
        "schema_version": "plan32_decoder_quantization_v1",
        "source_conversion_identity_sha256": receipt["conversion_identity_sha256"],
        "source_adapter_weights_sha256": receipt["adapter"]["weights_sha256"],
        "method": "mlx-vlm 0.7.1 affine RTN; decoder only; group size 64",
        "bits": bits,
        "quantized_language_modules": len(language),
        "quantized_nonlanguage_modules": len(nonlanguage),
        "output_sha256": output_hashes,
    }
    (output / "quantization-receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 6), required=True)
    args = parser.parse_args()
    result = quantize(args.source.resolve(strict=True), args.output.resolve(), args.bits)
    print(json.dumps({"bits": result["bits"], "decoder_modules": result["quantized_language_modules"]}))


if __name__ == "__main__":
    main()
