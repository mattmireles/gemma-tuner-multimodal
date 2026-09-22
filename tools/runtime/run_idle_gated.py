#!/usr/bin/env python3
"""Run a benchmark only when the host passes the Phase 0 idle gate."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
from reference_common import canonical_json, sha256_text

ML_MARKERS = (
    "benchmark_mlx",
    "capture_reference",
    "coreml",
    "eval_screenshot",
    "gemma",
    "kokoro",
    "llama-server",
    "mlx_lm",
    "mlx_vlm",
    "ollama",
    "pytorch",
    "qwen",
    "transformers",
    "whisper",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _thermal_state() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"supported": False, "warning": None, "performance_warning": None, "pass": False}
    result = subprocess.run(["pmset", "-g", "therm"], check=False, capture_output=True, text=True)
    output = result.stdout + result.stderr
    no_thermal = "No thermal warning level has been recorded" in output
    no_performance = "No performance warning level has been recorded" in output
    return {
        "supported": result.returncode == 0,
        "warning": not no_thermal,
        "performance_warning": not no_performance,
        "pass": result.returncode == 0 and no_thermal and no_performance,
    }


def _looks_like_ml_job(name: str, command: list[str]) -> bool:
    executable = Path(command[0]).name.casefold() if command else ""
    process_name = name.casefold()
    if any(marker in executable for marker in ML_MARKERS):
        return True
    if any(process_name.startswith(marker) for marker in ("kokoro", "llama-server", "ollama")):
        return True
    if not (executable.startswith("python") or executable in {"node", "bun", "deno"}):
        return False
    identifiers: list[str] = []
    for token in command[1:4]:
        if token.startswith("-"):
            continue
        identifiers.append(Path(token).name.casefold())
        break
    return any(any(marker in identifier for marker in ML_MARKERS) for identifier in identifiers)


def sample_host(seconds: float, process_limit_percent: float, allowed_pids: set[int]) -> dict[str, Any]:
    before_swap = psutil.swap_memory()
    psutil.cpu_times_percent(interval=None)
    processes: list[tuple[psutil.Process, bool]] = []
    ml_jobs: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid in allowed_pids:
                continue
            process.cpu_percent(interval=None)
            name = process.info.get("name") or "unknown"
            command = process.info.get("cmdline") or []
            processes.append((process, _looks_like_ml_job(name, command)))
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue

    time.sleep(seconds)
    cpu = psutil.cpu_times_percent(interval=None)
    offenders: list[dict[str, Any]] = []
    for process, is_ml_job in processes:
        try:
            usage = float(process.cpu_percent(interval=None))
            if usage > process_limit_percent:
                offenders.append({"pid": process.pid, "name": process.name(), "cpu_percent": usage})
            if is_ml_job and usage > 1.0:
                ml_jobs.append({"pid": process.pid, "name": process.name(), "cpu_percent": usage})
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    after_swap = psutil.swap_memory()
    thermal = _thermal_state()
    sample = {
        "seconds": seconds,
        "cpu_idle_percent": float(cpu.idle),
        "swap_in_delta_bytes": max(0, int(after_swap.sin - before_swap.sin)),
        "swap_out_delta_bytes": max(0, int(after_swap.sout - before_swap.sout)),
        "cpu_offenders": sorted(offenders, key=lambda item: (-item["cpu_percent"], item["pid"])),
        "other_ml_jobs": sorted(ml_jobs, key=lambda item: item["pid"]),
        "thermal": thermal,
    }
    return sample


def sample_passes(sample: dict[str, Any], minimum_idle_percent: float) -> bool:
    return bool(
        float(sample["cpu_idle_percent"]) >= minimum_idle_percent
        and int(sample["swap_in_delta_bytes"]) == 0
        and int(sample["swap_out_delta_bytes"]) == 0
        and not sample["cpu_offenders"]
        and not sample["other_ml_jobs"]
        and sample["thermal"]["pass"]
    )


def measure_gate(
    samples: int,
    seconds: float,
    minimum_idle_percent: float,
    process_limit_percent: float,
    allowed_pids: set[int],
) -> dict[str, Any]:
    measured = [sample_host(seconds, process_limit_percent, allowed_pids) for _ in range(samples)]
    return {
        "pass": all(sample_passes(sample, minimum_idle_percent) for sample in measured),
        "samples": measured,
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--minimum-idle-percent", type=float, default=90.0)
    parser.add_argument("--process-limit-percent", type=float, default=100.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 2 or args.seconds <= 0:
        raise ValueError("the gate requires at least two positive-duration samples")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    allowed_pids = {os.getpid()}
    receipt: dict[str, Any] = {
        "schema_version": "gemma4-e4b-idle-gate-v1",
        "started_at": _utc_now(),
        "settings": {
            "samples": args.samples,
            "seconds": args.seconds,
            "minimum_idle_percent": args.minimum_idle_percent,
            "process_limit_percent": args.process_limit_percent,
        },
        "command_sha256": sha256_text(canonical_json(command)) if command else None,
    }
    receipt["preflight"] = measure_gate(
        args.samples,
        args.seconds,
        args.minimum_idle_percent,
        args.process_limit_percent,
        allowed_pids,
    )
    if not receipt["preflight"]["pass"]:
        receipt["status"] = "preflight_failed"
        receipt["finished_at"] = _utc_now()
        _write_receipt(args.output, receipt)
        print(canonical_json({"status": receipt["status"], "output": str(args.output)}))
        return 2

    if command:
        started = time.perf_counter()
        completed = subprocess.run(command, check=False)
        receipt["command"] = {
            "exit_code": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
        }
        receipt["postflight"] = measure_gate(
            args.samples,
            args.seconds,
            args.minimum_idle_percent,
            args.process_limit_percent,
            allowed_pids,
        )
        if completed.returncode != 0:
            receipt["status"] = "command_failed"
        elif not receipt["postflight"]["pass"]:
            receipt["status"] = "postflight_failed"
        else:
            receipt["status"] = "pass"
    else:
        receipt["status"] = "pass"
    receipt["finished_at"] = _utc_now()
    _write_receipt(args.output, receipt)
    print(canonical_json({"status": receipt["status"], "output": str(args.output)}))
    if receipt["status"] == "pass":
        return 0
    if receipt["status"] == "command_failed":
        return int(receipt["command"]["exit_code"]) or 1
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
