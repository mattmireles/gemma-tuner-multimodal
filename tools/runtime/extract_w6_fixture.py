#!/usr/bin/env python3
"""Extract selected indexed W6 triples into a compact run-only fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

HEADER = "gemma4-tensor-index-v1"


def parse_index(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != HEADER:
        raise ValueError("unsupported tensor index")
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError("malformed tensor index row")
        rows.append(fields)
    return rows


def extract(args: argparse.Namespace) -> None:
    selected_names = {f"{base}.{suffix}" for base in args.base for suffix in ("weight", "scales", "biases")}
    selected = [row for row in parse_index(args.index) if row[0] in selected_names]
    found = {row[0] for row in selected}
    missing = sorted(selected_names - found)
    if missing:
        raise ValueError(f"fixture tensors are missing: {missing}")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    payload_path = args.output / "weights.bin"
    index_rows = [HEADER]
    with payload_path.open("wb") as destination:
        destination.write(b"\0")
        for name, dtype, relative, source_offset, length, rank, dimensions in selected:
            offset = destination.tell()
            source_path = args.payload_root / relative
            with source_path.open("rb") as source:
                source.seek(int(source_offset))
                data = source.read(int(length))
            if len(data) != int(length):
                raise OSError(f"short tensor read: {name}")
            destination.write(data)
            index_rows.append(
                "\t".join((name, dtype, "weights.bin", str(offset), length, rank, dimensions))
            )
            destination.write(b"\0")
    (args.output / "tensor-index.tsv").write_text("\n".join(index_rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", action="append", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
