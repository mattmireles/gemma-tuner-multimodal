from typer.testing import CliRunner

from cli_typer import app


def test_root_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Whisper Fine-Tuner" in result.stdout


def test_runs_help():
    runner = CliRunner()
    result = runner.invoke(app, ["runs", "--help"])
    assert result.exit_code == 0
    assert "Manage, list, and inspect" in result.stdout


def test_finetune_help():
    runner = CliRunner()
    result = runner.invoke(app, ["finetune", "--help"])
    assert result.exit_code == 0
    assert "Fine-tune a Whisper model" in result.stdout


def test_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Evaluate a fine-tuned Whisper model" in result.stdout


def test_distributed_push_code_help():
    runner = CliRunner()
    result = runner.invoke(app, ["distributed-push-code", "--help"])
    assert result.exit_code == 0
    assert "Synchronize the project directory to all worker nodes" in result.stdout
