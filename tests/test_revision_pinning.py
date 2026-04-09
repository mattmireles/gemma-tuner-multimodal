from __future__ import annotations

from types import SimpleNamespace

from gemma_tuner.models.gemma.family import GemmaFamily


def test_load_base_model_threads_revision_to_autoconfig_and_model_loader(monkeypatch):
    from gemma_tuner.models.gemma import base_model_loader as loader

    seen: dict[str, object] = {}

    def fake_config_from_pretrained(model_id: str, **kwargs):
        seen["config_model_id"] = model_id
        seen["config_kwargs"] = kwargs
        return SimpleNamespace(architectures=["Gemma2ForCausalLM"], model_type="gemma2")

    def fake_causallm_from_pretrained(model_id: str, **kwargs):
        seen["model_model_id"] = model_id
        seen["model_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", staticmethod(fake_config_from_pretrained))
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_causallm_from_pretrained))

    model = loader.load_base_model_for_gemma(
        "google/gemma-3n-E2B-it",
        family=GemmaFamily.GEMMA_3N,
        torch_dtype="float32",
        attn_implementation="eager",
        revision="abc123",
    )

    assert model is not None
    assert seen["config_model_id"] == "google/gemma-3n-E2B-it"
    assert seen["model_model_id"] == "google/gemma-3n-E2B-it"
    assert seen["config_kwargs"] == {"trust_remote_code": True, "revision": "abc123"}
    assert seen["model_kwargs"]["revision"] == "abc123"


def test_export_model_dir_threads_revision_to_processor(monkeypatch, tmp_path):
    from gemma_tuner.scripts import export as export_mod

    model_dir = tmp_path / "gemma-3n-local-model"
    model_dir.mkdir()

    class _FakeModel:
        def save_pretrained(self, out_dir: str) -> None:
            seen["saved_model_dir"] = out_dir

        def parameters(self):
            return []

    class _FakeProcessor:
        def save_pretrained(self, out_dir: str) -> None:
            seen["saved_processor_dir"] = out_dir

    seen: dict[str, object] = {}

    monkeypatch.setattr(export_mod, "get_device", lambda: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(export_mod, "load_base_model_for_gemma", lambda *args, **kwargs: _FakeModel())

    def fake_processor_from_pretrained(source: str, **kwargs):
        seen["processor_source"] = source
        seen["processor_kwargs"] = kwargs
        return _FakeProcessor()

    monkeypatch.setattr(export_mod.AutoProcessor, "from_pretrained", staticmethod(fake_processor_from_pretrained))

    out_dir = export_mod.export_model_dir(str(model_dir), model_revision="abc123")

    assert out_dir.endswith("-export")
    assert seen["processor_source"] == str(model_dir)
    assert seen["processor_kwargs"]["revision"] == "abc123"
    assert seen["saved_model_dir"] == out_dir
    assert seen["saved_processor_dir"] == out_dir


def test_ops_export_forwards_model_revision(monkeypatch):
    from gemma_tuner.core import ops
    from gemma_tuner.scripts import export as export_mod

    seen: dict[str, object] = {}

    def fake_export_model_dir(model_path_or_profile: str, model_revision: str | None = None) -> None:
        seen["model_path_or_profile"] = model_path_or_profile
        seen["model_revision"] = model_revision

    monkeypatch.setattr(export_mod, "export_model_dir", fake_export_model_dir)

    ops.export("google/gemma-3n-E2B-it", model_revision="abc123")

    assert seen == {
        "model_path_or_profile": "google/gemma-3n-E2B-it",
        "model_revision": "abc123",
    }
