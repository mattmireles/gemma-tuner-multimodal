from pathlib import Path

import pytest
import torch
from flask import Flask
from flask_socketio import SocketIO

from gemma_tuner.visualization.events import build_training_event
from gemma_tuner.visualization.payload import finalize_training_payload


def test_visualizer_template_uses_local_assets_only():
    template = Path("templates/index.html").read_text()
    assert "https://" not in template
    assert "http://" not in template
    assert "{{ asset_paths.socketio }}" in template
    assert "{{ asset_paths.chartjs }}" in template


def test_build_training_event_extracts_bounded_payload():
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([param], lr=0.01, weight_decay=0.1)

    class FakeOutputs:
        def __init__(self):
            self.attentions = [torch.ones(1, 2, 30, 30)]
            self.logits = torch.randn(1, 6, 10)

    event = build_training_event(
        step=10,
        epoch=1.0,
        loss=0.5,
        gradient_norm=1.2,
        learning_rate=0.01,
        memory_gb=0.25,
        batch={"input_features": torch.ones(1, 80, 120)},
        outputs=FakeOutputs(),
        optimizer=optimizer,
        steps_per_second=2.5,
        total_time=12.0,
        architecture={"encoder_layers": 12},
    )

    payload = finalize_training_payload(event.as_payload())
    assert payload["viz_schema_version"] == 1
    assert "panels_status" in payload
    assert payload["step"] == 10
    assert len(payload["attention"]) == 20
    assert len(payload["attention"][0]) == 20
    assert len(payload["token_probs"]["values"]) == 5
    assert payload["optimizer_stats"]["lr"] == 0.01
    assert payload["optimizer_stats"]["weight_decay"] == 0.1


def test_flask_socketio_emit_must_not_use_broadcast_kwarg():
    """Regression: ``broadcast=`` is not valid on ``SocketIO.emit`` (Flask-SocketIO 5)."""
    app = Flask(__name__)
    sio = SocketIO(app, async_mode="threading")
    with app.app_context():
        with pytest.raises(TypeError, match="broadcast"):
            sio.emit("training_update", {"step": 1}, namespace="/", broadcast=True)
        sio.emit("training_update", {"step": 1}, namespace="/")
