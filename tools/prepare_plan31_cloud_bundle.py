"""Remap the frozen Plan 31 CSVs to content-addressed bundle image paths.

Only image_path changes. Local symlinks permit an exact content check without
copying 1 GB of private screenshots. The VM reuses the sealed Plan 29 images.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gemma_tuner.utils.plan31_projection import verify_projection
from tools.prepare_plan31_mixed import FIELDS, OUTPUT as SOURCE, sha_file, write_csv_once


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/datasets/tt-screenshot-plan31-mixed-v4-deploy"
SOURCE_RECEIPT_SHA256 = "b569cf0852e237e0e41aa5dadbf652a34f631424d13aa1ae9beaa2e2682f8c78"


def _copy_once(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"conflicting Plan 31 deploy artifact: {path.name}")
    if not path.exists():
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def build(source: Path = SOURCE, destination: Path = DESTINATION) -> dict:
    source_receipt = verify_projection(source, SOURCE_RECEIPT_SHA256)
    target_root = destination / "full"
    image_root = destination / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name in sorted(source_receipt["outputs_sha256"]):
        src = source / name
        if not name.endswith(".csv"):
            output_hashes[name] = _copy_once(target_root / name, src.read_bytes())
            continue
        with src.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            original = Path(row["image_path"])
            image_sha = sha_file(original)
            # The sealed Plan 29 GCS bundle keys screenshots by source row ID,
            # not by image-content SHA. Retain that exact reusable image layout.
            link = image_root / f"{row['id']}{original.suffix.lower()}"
            if link.exists() or link.is_symlink():
                if link.resolve() != original.resolve() or sha_file(link) != image_sha:
                    raise ValueError("Plan 31 deploy image link conflicts with source")
            else:
                link.symlink_to(original)
            row["image_path"] = f"../images/{link.name}"
        output_hashes[name] = write_csv_once(target_root / name, rows)
    receipt = {**source_receipt, "outputs_sha256": output_hashes}
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _copy_once(target_root / "projection-v3.receipt.json", receipt_bytes)
    verify_projection(target_root, hashlib.sha256(receipt_bytes).hexdigest())
    return {"schema_version": "plan31_deploy_bundle_v1",
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
            "deploy_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "image_links": len(list(image_root.iterdir())),
            "outputs_sha256": output_hashes}


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
