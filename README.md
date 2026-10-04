# Gemma Multimodal Fine-Tuner

Fine-tune Hugging Face Gemma models on text, images, and audio using PEFT LoRA.
The repository includes a command-line wizard and training tools for Apple Silicon.

## Install

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp config/config.ini.example config/config.ini
```

Review the configuration before running training. Model downloads may require
Hugging Face authentication and acceptance of the model license.

## Use

```sh
gemma-macos-tuner --help
gemma-macos-tuner system-check
```

Implementation lives in `gemma_tuner/`; additional tools live in `tools/`.
See `pyproject.toml` for dependencies and command-line entry points.

## Repository scope

This public repository contains code and public usage documentation. Personal
plans, research notes, experiment receipts, and agent configuration are maintained
separately and are excluded here. Contributions must use clean code changes;
do not merge branches containing private development history.

## License

See [LICENSE](LICENSE). Bundled third-party code retains its own license notices.
