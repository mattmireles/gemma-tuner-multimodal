from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.gcp_tt_sft_contract import ContractError, load_contract, render_launch
from tools.prepare_tt_telepathic_sft import owner_diverse_subset, parse_context, render_conditioned, stable_key


def test_stable_order_is_deterministic_and_seeded() -> None:
    values = ["c", "a", "b"]
    assert sorted(values, key=lambda value: stable_key("seed", value)) == sorted(
        values, key=lambda value: stable_key("seed", value)
    )
    assert sorted(values, key=lambda value: stable_key("other", value)) != sorted(
        values, key=lambda value: stable_key("seed", value)
    )


def test_conditioned_rendering_resolves_exact_placeholders() -> None:
    row = {
        "user_prompt": (
            "instruction\n\n<context>\n"
            '{"context":{"application":"Code","user":"Ada"}}'
            "\n</context>\n\n<first_pass_screenshot_ocr></first_pass_screenshot_ocr>\n"
        )
    }
    assert parse_context(row["user_prompt"])["user"] == "Ada"
    assert (
        render_conditioned("For {USER_FULL_NAME} in {APPLICATION_NAME}; help {USER_FULL_NAME}.", row)
        == "For Ada in Code; help Ada."
    )


def test_conditioned_rendering_rejects_missing_placeholder() -> None:
    row = {"user_prompt": '<context>\n{"context":{"application":"Code","user":"Ada"}}\n</context>'}
    with pytest.raises(ValueError, match="must contain"):
        render_conditioned("For {USER_FULL_NAME}.", row)


def test_conditioned_rendering_uses_frozen_generic_name_fallback() -> None:
    row = {"user_prompt": ('<context>\n{"context":{"application":"Code","user":"{USER_FULL_NAME}"}}\n</context>')}
    assert render_conditioned("Help {USER_FULL_NAME} in {APPLICATION_NAME}.", row) == "Help the user in Code."


def test_owner_diverse_subset_round_robins_before_reusing_owner() -> None:
    rows = [
        {"context_id": "a1", "owner_id": "a"},
        {"context_id": "a2", "owner_id": "a"},
        {"context_id": "b1", "owner_id": "b"},
    ]
    selected = owner_diverse_subset(rows, "seed", 2)
    owner_by_id = {row["context_id"]: row["owner_id"] for row in rows}
    assert len({owner_by_id[item] for item in selected}) == 2


def test_phase0_contract_budget_sums_and_hashes_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "config" / "telepathic_context_sft_phase0.json"
    contract = json.loads(contract_path.read_text())
    budget = contract["budget_usd"]
    assert (
        budget["gcp_compute_max"] + budget["terra_evaluation_max"] + budget["gcs_max"] + budget["contingency"]
        == budget["owner_total"]
    )
    assert budget["owner_total"] == 500.0
    assert sum(budget["compute_phase_max_hours"].values()) == contract["gcp"]["maximum_program_compute_hours"]
    assert contract["messages"]["compact_system_role"] is False
    assert contract["model"]["revision"] == "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
    assert len(contract["model"]["files"]["model.safetensors"]) == 64


def test_gcp_launch_is_identity_and_lease_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root / "config" / "telepathic_context_sft_phase0.json")
    rendered = render_launch(contract, "phase_2", 3)
    command = rendered["create_command"]
    assert "--project=gist-is-backend" in command
    assert "--account=whisper-gcp-sft-runner@gist-is-backend.iam.gserviceaccount.com" in command
    assert "--max-run-duration=10800s" in command
    assert rendered["maximum_cost_usd"] == pytest.approx(11.020155)
    with pytest.raises(ContractError, match="ceiling"):
        render_launch(contract, "phase_2", 9)


def test_gcp_launch_allows_only_named_same_region_capacity_fallbacks() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root / "config" / "telepathic_context_sft_phase0.json")
    rendered = render_launch(contract, "phase_2", 3, zone="us-central1-c")
    assert rendered["zone"] == "us-central1-c"
    assert rendered["capacity_fallback"] is True
    assert "--zone=us-central1-c" in rendered["create_command"]
    with pytest.raises(ContractError, match="same-region"):
        render_launch(contract, "phase_2", 3, zone="us-west1-a")


def test_gcp_80gb_four_hour_lease_stays_inside_phase2_cost_cap() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root / "config" / "telepathic_context_sft_phase0.json")
    rendered = render_launch(contract, "phase_2", 4, gpu_memory_gb=80)
    assert rendered["hardware"]["machine_type"] == "a2-ultragpu-1g"
    assert rendered["maximum_cost_usd"] == pytest.approx(20.275192)
    assert rendered["maximum_cost_usd"] < contract["budget_usd"]["compute_phase_max_usd"]["phase_2"]
    assert "--max-run-duration=14400s" in rendered["create_command"]
    assert "--discard-local-ssds-at-termination-timestamp=true" in rendered["create_command"]


def test_spot_launch_is_explicit_and_keeps_standard_rate_as_cost_ceiling() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root / "config" / "telepathic_context_sft_phase0.json")
    rendered = render_launch(contract, "phase_2", 2, gpu_memory_gb=80, spot=True)
    assert rendered["hardware"]["provisioning_model"] == "SPOT"
    assert rendered["maximum_cost_usd"] == pytest.approx(2 * 5.06879789)
    assert "--provisioning-model=SPOT" in rendered["create_command"]
    assert rendered["create_command"][4] == "gemma4-e4b-telepathic-phase-2-80-spot"
