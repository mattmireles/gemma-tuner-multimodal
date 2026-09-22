#!/usr/bin/env python3
"""Write the native TSV index for existing SafeTensors without copying data."""

from __future__ import annotations

import argparse
from pathlib import Path

from gemma_tuner.runtime.package_schema import inspect_safetensors, tensor_index_bytes, tensor_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SafeTensors file or directory")
    parser.add_argument("output", type=Path, help="destination tensor-index.tsv")
    parser.add_argument(
        "--payload-root",
        type=Path,
        help="emit real shard paths relative to this root (useful for symlinked Hub snapshots)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if source.is_file():
        entries = inspect_safetensors(source)
        payload_root = source.parent
    elif source.is_dir():
        entries = tensor_inventory(source)["entries"]
        payload_root = source
    else:
        raise FileNotFoundError(source)
    if args.payload_root is not None:
        payload_root = args.payload_root.resolve()
        for entry in entries:
            shard = source if source.is_file() else source / entry["shard"]
            entry["shard"] = shard.resolve().relative_to(payload_root).as_posix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tensor_index_bytes(entries, path_prefix=""))
    print(f"indexed {len(entries)} tensors under {payload_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
