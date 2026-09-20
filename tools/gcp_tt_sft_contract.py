#!/usr/bin/env python3
"""Render, but never execute, a budget-bounded GCE launch for this experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "config" / "telepathic_context_sft_phase0.json"
IMAGE_FAMILY = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT = "deeplearning-platform-release"
CAPACITY_FALLBACK_ZONES = frozenset({"us-central1-c", "us-central1-f"})
A100_80_MACHINE_TYPE = "a2-ultragpu-1g"
A100_80_STANDARD_USD_PER_HOUR = 5.06879789


class ContractError(ValueError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    budget = contract["budget_usd"]
    allocated = sum(budget[key] for key in ("gcp_compute_max", "terra_evaluation_max", "gcs_max", "contingency"))
    if not math.isclose(allocated, budget["owner_total"], abs_tol=1e-9):
        raise ContractError("budget allocation does not equal owner total")
    if budget["owner_total"] != 500.0:
        raise ContractError("owner ceiling changed")
    return contract


def render_launch(
    contract: dict[str, Any], phase: str, hours: float, *, zone: str | None = None,
    gpu_memory_gb: int = 40, spot: bool = False,
) -> dict[str, Any]:
    gcp, budget = contract["gcp"], contract["budget_usd"]
    selected_zone = zone or gcp["zone"]
    if selected_zone != gcp["zone"] and selected_zone not in CAPACITY_FALLBACK_ZONES:
        raise ContractError("zone must be the frozen primary or an approved same-region capacity fallback")
    maximum = budget["compute_phase_max_hours"].get(phase)
    if maximum is None:
        raise ContractError("paid compute is allowed only for phases 1 through 4")
    if not math.isfinite(hours) or hours <= 0:
        raise ContractError("lease hours must be positive and finite")
    if hours > maximum or hours > gcp["maximum_single_lease_hours"]:
        raise ContractError("lease exceeds the frozen phase or single-lease ceiling")
    if gpu_memory_gb == 40:
        machine_type = gcp["machine_type"]
        hourly_rate = gcp["standard_usd_per_hour"]
        instance_suffix = ""
    elif gpu_memory_gb == 80:
        machine_type = A100_80_MACHINE_TYPE
        hourly_rate = A100_80_STANDARD_USD_PER_HOUR
        instance_suffix = "-80"
    else:
        raise ContractError("gpu_memory_gb must be 40 or 80")
    cost = hours * hourly_rate
    if cost > budget["gcp_compute_max"]:
        raise ContractError("lease exceeds the frozen compute dollar ceiling")
    provisioning_suffix = "-spot" if spot else ""
    instance = (
        f"gemma4-e4b-telepathic-{phase.replace('_', '-')}"
        f"{instance_suffix}{provisioning_suffix}"
    )
    seconds = math.floor(hours * 3600)
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance,
        f"--project={gcp['project']}",
        f"--account={gcp['account']}",
        f"--zone={selected_zone}",
        f"--machine-type={machine_type}",
        f"--provisioning-model={'SPOT' if spot else 'STANDARD'}",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--max-run-duration={seconds}s",
        "--instance-termination-action=STOP",
        f"--service-account={gcp['account']}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        f"--image-family={IMAGE_FAMILY}",
        f"--image-project={IMAGE_PROJECT}",
        "--boot-disk-size=300GB",
        "--boot-disk-type=pd-balanced",
        "--metadata="
        f"experiment-id={contract['experiment_id']},phase={phase},"
        f"idle-shutdown-minutes={gcp['idle_shutdown_minutes']},"
        f"checkpoint-prefix={gcp['checkpoint_prefix']}",
    ]
    if gpu_memory_gb == 80:
        command.append("--discard-local-ssds-at-termination-timestamp=true")
    return {
        "schema_version": "gemma4_e4b_telepathic_gcp_launch_v1",
        "identity": {"project": gcp["project"], "account": gcp["account"]},
        "phase": phase,
        "zone": selected_zone,
        "capacity_fallback": selected_zone != gcp["zone"],
        "hardware": {
            "machine_type": machine_type,
            "gpu": f"NVIDIA A100 {gpu_memory_gb}GB",
            "provisioning_model": "SPOT" if spot else "STANDARD",
            "standard_usd_per_hour": hourly_rate,
        },
        "lease_hours": hours,
        "maximum_cost_usd": round(cost, 6),
        "checkpoint_prefix": gcp["checkpoint_prefix"],
        "create_command": command,
        "stop_command": [
            "gcloud",
            "compute",
            "instances",
            "stop",
            instance,
            f"--project={gcp['project']}",
            f"--account={gcp['account']}",
            f"--zone={selected_zone}",
            "--quiet",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phase", required=True, choices=("phase_1", "phase_2", "phase_3", "phase_4"))
    parser.add_argument("--hours", required=True, type=float)
    parser.add_argument("--zone")
    parser.add_argument("--gpu-memory-gb", type=int, choices=(40, 80), default=40)
    parser.add_argument("--spot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        render_launch(
            load_contract(args.contract.resolve()), args.phase, args.hours,
            zone=args.zone, gpu_memory_gb=args.gpu_memory_gb, spot=args.spot,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
