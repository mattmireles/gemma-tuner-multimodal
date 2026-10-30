from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "tools" / "eval_tt_telepathic_moondream.py"
SPEC = importlib.util.spec_from_file_location("eval_tt_telepathic_moondream", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_render_question_preserves_both_prompt_parts() -> None:
    rendered = MODULE.render_question({"system_prompt": "SYSTEM", "prompt": "USER"})
    assert rendered == "System instructions:\nSYSTEM\n\nUser request:\nUSER"


def test_strict_context_json_rejects_fences_and_extra_keys() -> None:
    assert MODULE.strict_context_json('{"context_analysis":{}}')
    assert not MODULE.strict_context_json('```json\n{"context_analysis":{}}\n```')
    assert not MODULE.strict_context_json('{"context_analysis":{},"extra":1}')


def test_read_rows_resolves_images_and_freezes_prefix(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    Image.new("RGB", (2, 2)).save(image)
    csv_path = tmp_path / "validation.csv"
    csv_path.write_text(
        "id,owner_id,image_path,prompt,response,image_view_policy,system_prompt\n"
        "a,o,shot.png,p,r,v,s\n"
        "b,o,shot.png,p,r,v,s\n",
        encoding="utf-8",
    )
    rows = MODULE.read_rows(csv_path, 1)
    assert [row["id"] for row in rows] == ["a"]
    assert rows[0]["image_path"] == str(image.resolve())


def test_read_prior_fails_closed_on_settings_mismatch(tmp_path: Path) -> None:
    ledger = tmp_path / "out.jsonl"
    ledger.write_text(json.dumps({"example_id": "a", "settings_sha256": "old"}) + "\n")
    with pytest.raises(ValueError, match="settings mismatch"):
        MODULE.read_prior(ledger, "new")


def test_mps_load_disables_flex_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    class Inner:
        use_flex_decoding = True

        class Config:
            class Text:
                max_context = 4096

            text = Text()

        config = Config()

        def _refresh_runtime_buffers(self) -> None:
            pass

    class FakeModel:
        model = Inner()

        def to(self, device: str):
            assert device == "mps"
            return self

        def eval(self) -> None:
            pass

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    import sys
    import types

    fake_torch = types.SimpleNamespace(bfloat16="bf16")
    fake_transformers = types.SimpleNamespace(AutoModelForCausalLM=AutoModel)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _, model = MODULE.load_model("mps")
    assert model.model.use_flex_decoding is False
    assert model.model.config.text.max_context == 12288
