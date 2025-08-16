import sys
import types
from pathlib import Path

import importlib
import builtins


def test_single_node_injection(monkeypatch, tmp_path):
    # Arrange: fake finetune orchestrator to capture trainer injection
    captured = {}

    def fake_finetune(profile_config, output_dir, trainer_class=None, trainer_kwargs=None):
        captured["profile"] = profile_config
        captured["output_dir"] = output_dir
        captured["trainer_class"] = trainer_class
        captured["strategy_cls"] = type(trainer_kwargs.get("strategy")) if trainer_kwargs else None
        return None

    # Provide a minimal config.ini with a tiny profile pointing to existing model/dataset names
    cfg = """
[DEFAULT]
output_dir = output

[group:whisper]

[model:whisper-small]
base_model = openai/whisper-small
group = whisper

[dataset:librispeech]
text_column = text
train_split = train
validation_split = validation
max_duration = 30.0
max_label_length = 256

[profile:test]
model = whisper-small
dataset = librispeech
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
num_train_epochs = 1
logging_steps = 10
save_steps = 1000
save_total_limit = 1
learning_rate = 1e-5
    """.strip()
    config_path = tmp_path / "config.ini"
    config_path.write_text(cfg)

    # Monkeypatch imports and argv
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setitem(sys.modules, 'scripts.finetune', types.SimpleNamespace(main=fake_finetune))

    # Build argv for single-node path
    argv = [
        'train_distributed.py',
        '--profile', 'test',
        '--output_dir', str(tmp_path / 'out'),
        '--num_nodes', '1',
        '--strategy', 'allreduce',
        '--config', str(config_path),
    ]
    monkeypatch.setattr(sys, 'argv', argv)

    # Act: import and run main()
    m = importlib.import_module('train_distributed')
    m.main()

    # Assert: trainer injection happened
    from distributed.trainer import DistributedWhisperTrainer
    from gym.exogym.strategy.strategy import SimpleReduceStrategy
    assert captured["trainer_class"] is DistributedWhisperTrainer
    assert captured["strategy_cls"] is SimpleReduceStrategy


def test_spawn_called(monkeypatch, tmp_path):
    # Arrange fake config
    cfg = """
[DEFAULT]
output_dir = output

[group:whisper]

[model:whisper-small]
base_model = openai/whisper-small
group = whisper

[dataset:librispeech]
text_column = text
train_split = train
validation_split = validation
max_duration = 30.0
max_label_length = 256

[profile:test]
model = whisper-small
dataset = librispeech
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
num_train_epochs = 1
logging_steps = 10
save_steps = 1000
save_total_limit = 1
learning_rate = 1e-5
    """.strip()
    config_path = tmp_path / "config.ini"
    config_path.write_text(cfg)

    # Fake finetune to avoid heavy imports
    def fake_finetune(*args, **kwargs):
        return None

    # Capture mp.spawn call
    called = {}
    def fake_spawn(fn, args=(), nprocs=1, join=True, start_method="spawn"):
        called["nprocs"] = nprocs
        called["args_len"] = len(args)
        # Do not execute worker in tests
        return None

    monkeypatch.setitem(sys.modules, 'scripts.finetune', types.SimpleNamespace(main=fake_finetune))

    import importlib
    m = importlib.import_module('train_distributed')
    monkeypatch.setattr(m.mp, 'spawn', fake_spawn)

    argv = [
        'train_distributed.py',
        '--profile', 'test',
        '--output_dir', str(tmp_path / 'out'),
        '--num_nodes', '2',
        '--strategy', 'allreduce',
        '--config', str(config_path),
    ]
    monkeypatch.setattr(sys, 'argv', argv)

    # Act
    m.main()

    # Assert
    assert called.get("nprocs") == 2
