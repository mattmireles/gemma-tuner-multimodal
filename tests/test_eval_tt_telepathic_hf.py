from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
import torch

MODULE = Path(__file__).parents[1] / "tools" / "eval_tt_telepathic_hf.py"
SPEC = importlib.util.spec_from_file_location("eval_tt_telepathic_hf", MODULE)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"context_analysis":{}}', True),
        ('{"context_analysis":{},"extra":1}', False),
        ('```json\n{"context_analysis":{}}\n```', False),
        ('[]', False),
    ],
)
def test_strict_context_json(text: str, expected: bool) -> None:
    assert evaluation.strict_context_json(text) is expected


def test_resolve_device_prefers_cuda_then_mps_then_cpu() -> None:
    class Available:
        def __init__(self, value: bool) -> None:
            self.value = value

        def is_available(self) -> bool:
            return self.value

    class Runtime:
        def __init__(self, cuda: bool, mps: bool) -> None:
            self.cuda = Available(cuda)
            self.backends = type("Backends", (), {"mps": Available(mps)})()

    assert evaluation.resolve_device(Runtime(True, True), "auto") == "cuda"
    assert evaluation.resolve_device(Runtime(False, True), "auto") == "mps"
    assert evaluation.resolve_device(Runtime(False, False), "auto") == "cpu"
    assert evaluation.resolve_device(Runtime(True, True), "cpu") == "cpu"


def test_select_rows_uses_frozen_order_and_rejects_missing_ids() -> None:
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert [row["id"] for row in evaluation.select_rows(rows, ids=["c", "a"], count=None)] == ["c", "a"]
    with pytest.raises(ValueError, match="missing 1"):
        evaluation.select_rows(rows, ids=["missing"], count=None)


def test_read_csv_resolves_image_paths_relative_to_csv(tmp_path: Path) -> None:
    image = tmp_path / "images" / "fixture.png"
    image.parent.mkdir()
    image.write_bytes(b"fixture")
    csv_path = tmp_path / "arm" / "train.csv"
    csv_path.parent.mkdir()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path"])
        writer.writeheader()
        writer.writerow({"id": "a", "image_path": "../images/fixture.png"})

    rows = evaluation.read_csv(csv_path)
    assert rows[0]["image_path"] == str(image.resolve())


def test_move_batch_to_device_preserves_scalar_model_kwargs() -> None:
    batch = {"input_ids": torch.tensor([[1, 2]]), "logits_to_keep": 7}
    moved = evaluation.move_batch_to_device(batch, "cpu")
    assert moved["input_ids"].device.type == "cpu"
    assert moved["logits_to_keep"] == 7


def test_adapter_integrity_accepts_sealed_checkpoint_and_stock(tmp_path: Path) -> None:
    assert evaluation.adapter_integrity_sha256(None) is None
    marker = tmp_path / ".complete.json"
    marker.write_text('{"global_step":78}\n', encoding="utf-8")
    assert evaluation.adapter_integrity_sha256(tmp_path) == evaluation.sha256_file(marker)


def test_adapter_integrity_rejects_unsealed_adapter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no checkpoint completion"):
        evaluation.adapter_integrity_sha256(tmp_path)


def test_completion_loss_contract_accepts_attention_mask() -> None:
    from gemma_tuner.models.gemma.finetune import completion_only_causal_loss

    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[-100, 2, 3]])
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    loss = completion_only_causal_loss(logits, labels, attention_mask)
    assert torch.isfinite(loss)


def test_merge_adapter_requires_an_adapter() -> None:
    # The fail-closed condition is evaluated before an adapter can be merged.
    with pytest.raises(ValueError, match="without an adapter"):
        evaluation.load_runtime(None, "cpu", merge_adapter=True)


def test_release_device_cache_empties_mps_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    runtime = type("Runtime", (), {
        "mps": type("MPS", (), {"empty_cache": lambda self: calls.append("mps")})()
    })()
    monkeypatch.setattr(evaluation.gc, "collect", lambda: calls.append("gc"))
    evaluation.release_device_cache(runtime, "mps")
    assert calls == ["gc", "mps"]


def test_cyclic_permutation_never_reuses_own_image() -> None:
    rows = [
        {"id": "a", "image_path": "a.png"},
        {"id": "b", "image_path": "b.png"},
        {"id": "c", "image_path": "c.png"},
    ]
    permutation = evaluation.cyclic_permutation(rows)
    assert set(permutation) == {"a", "b", "c"}
    assert all(permutation[row["id"]] != row["image_path"] for row in rows)


def test_dependence_summary_reports_loss_reduction_and_correct_image_margin() -> None:
    rows = [
        {"image_dependence": {"base_correct": 10.0, "correct": 1.0, "permuted": 1.5}},
        {"image_dependence": {"base_correct": 8.0, "correct": 0.8, "permuted": 0.7}},
    ]
    summary = evaluation.summarize_dependence(rows)
    assert summary is not None
    assert summary["assistant_loss_reduction"] == pytest.approx(0.9)
    assert summary["correct_lower_nll"] == 1
    assert summary["mean_permuted_minus_correct_nll"] == pytest.approx(0.2)
    assert summary["one_sided_sign_test_p"] == pytest.approx(0.75)
    assert summary["passes_preregistered_gate"] is False


def test_preregistered_image_dependence_gate_requires_22_of_32_and_positive_mean() -> None:
    rows = [
        {"image_dependence": {"base_correct": 10.0, "correct": 1.0, "permuted": 2.0}}
        for _ in range(22)
    ] + [
        {"image_dependence": {"base_correct": 10.0, "correct": 2.0, "permuted": 1.0}}
        for _ in range(10)
    ]
    summary = evaluation.summarize_dependence(rows)
    assert summary is not None
    assert summary["correct_lower_nll"] == 22
    assert summary["one_sided_sign_test_p"] == pytest.approx(0.025051229866221547)
    assert summary["mean_permuted_minus_correct_nll"] > 0
    assert summary["passes_preregistered_gate"] is True


def test_generation_messages_preserve_arm_difference_only() -> None:
    row = {"prompt": "user", "system_prompt": "intent"}
    views = [object()] * 5
    compact = evaluation.messages_for_generation({"prompt": "user"}, "compact", views)
    conditioned = evaluation.messages_for_generation(row, "conditioned", views)
    assert [message["role"] for message in compact] == ["user"]
    assert [message["role"] for message in conditioned] == ["system", "user"]
    assert compact[-1] == conditioned[-1]
    full = evaluation.messages_for_generation(row, "full", views)
    assert [message["role"] for message in full] == ["system", "user"]
    assert full == conditioned
