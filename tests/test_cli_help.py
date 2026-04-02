from typer.testing import CliRunner

from whisper_tuner.cli_typer import app


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
    assert "Execute model fine-tuning" in result.stdout


def test_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Evaluate fine-tuned Whisper models" in result.stdout
