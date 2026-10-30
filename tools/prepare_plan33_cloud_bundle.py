"""Pack the Plan 33 trainer overlay: code, frozen config, and projected CSVs.

Screenshots are not packed. The trainer VM boots from the corrected Plan 31/32
snapshot whose image cache uses the same row-ID layout, and projection
verification rehashes every referenced image on the VM before training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "data/datasets/tt-screenshot-plan33-literal-v3-deploy"
PATHS = (
    "gemma_tuner",
    "config/plan33-e2b-literal-three-epochs.ini",
    "tools/run_plan33_segment.py",
    "tools/verify_plan33_segment.py",
    "tools/probe_plan33_shared_kv.py",
    "tools/probe_plan33_merge_parity.py",
    "tests/test_plan33_profile.py",
    "tests/test_plan33_resume.py",
    "tests/test_plan31_input_modes.py",
    "tests/test_validation_telemetry.py",
    "tests/test_plan31_exposure_ledger.py",
    "tests/test_plan31_resume_sampler.py",
    "tests/test_plan32_projection.py",
    "requirements/requirements-gemma4-telepathic-corrected.lock",
    DATASET,
)


def _files() -> list[Path]:
    files: list[Path] = []
    for relative in PATHS:
        path = ROOT / relative
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*"))
                         if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(relative)
    return files


def build(output: Path) -> dict:
    manifest = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=6) as archive:
        for path in _files():
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            manifest[relative] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(relative)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("plan33-bundle-manifest.json")
        info.size, info.mtime, info.mode = len(raw), 0, 0o644
        archive.addfile(info, io.BytesIO(raw))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(buffer.getvalue())
    return {"bundle": str(output), "bundle_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "files": len(manifest), "manifest_sha256": hashlib.sha256(raw).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), sort_keys=True))
