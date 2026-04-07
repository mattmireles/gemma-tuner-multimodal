# Gemma Multimodal Fine-Tuner

```text
     ██████╗ ███████╗███╗   ███╗███╗   ███╗ █████╗
    ██╔════╝ ██╔════╝████╗ ████║████╗ ████║██╔══██╗
    ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║███████║
    ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══██║
    ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║
     ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

              🍎 Fine-Tuner for Apple Silicon
```

**Fine-tune Gemma with audio, on your Mac, on data that doesn't fit on your Mac.**

- 🎙️ **Audio + text LoRA** — not just text. MLX can't do this yet.
- ☁️ **Stream from GCS / BigQuery** — train on terabytes without filling your SSD.
- 🍎 **Runs on Apple Silicon** — MPS-native, no NVIDIA box required.

**Source:** [github.com/mattmireles/gemma-tuner-multimodal](https://github.com/mattmireles/gemma-tuner-multimodal) (public).

---

## Why not just use…?

| | **This** | MLX-LM | Unsloth | axolotl |
| --- | :-: | :-: | :-: | :-: |
| Fine-tune Gemma (text) | ✅ | ✅ | ✅ | ✅ |
| Fine-tune Gemma **audio + text** | ✅ | ❌ | ❌ | ⚠️ CUDA only |
| Runs on Apple Silicon (MPS) | ✅ | ✅ | ❌ | ❌ |
| **Stream training data from cloud** | ✅ | ❌ | ❌ | ⚠️ partial |
| No NVIDIA GPU required | ✅ | ✅ | ❌ | ❌ |

If you want to fine-tune Gemma on **audio** without renting an H100 or copying a terabyte of WAVs to your laptop, this is the only toolkit that does all three.

Under the hood: Hugging Face Gemma checkpoints + PEFT LoRA, supervised fine-tuning in [`gemma_tuner/models/gemma/finetune.py`](gemma_tuner/models/gemma/finetune.py), exported as a merged HF / SafeTensors tree by [`gemma_tuner/scripts/export.py`](gemma_tuner/scripts/export.py). For Core ML conversion and GGUF inference tooling, see [`README/guides/README.md`](README/guides/README.md) — this repo's *training* path is Gemma-only by design.

**Deeper reading:** [`README/guides/README.md`](README/guides/README.md) · [`README/specifications/Gemma3n.md`](README/specifications/Gemma3n.md)

---

## What you can build with this

- **Domain-specific ASR** — fine-tune on medical dictation, legal depositions, call-center recordings, or any field where off-the-shelf Whisper / Gemma mishears the jargon.
- **Accent and dialect adaptation** — adapt a base Gemma model to underrepresented accents using your own labeled audio.
- **Low-resource languages** — train on a few hours of paired audio + transcript and ship a model that actually understands your language.
- **Audio-grounded assistants** — extend Gemma's text reasoning with audio understanding for transcription + Q&A pipelines.
- **Private, on-device pipelines** — train and run entirely on your Mac. Audio never leaves the machine; weights never touch a third-party API.

If your data lives in GCS or BigQuery, you can do all of this on a laptop without copying terabytes of audio locally — the dataloader streams shards on demand.

---

## Supported models

Training targets **Gemma multimodal (audio + text)** checkpoints loaded via `base_model` in [`config.ini`](config.ini) and routed to [`gemma_tuner/models/gemma/finetune.py`](gemma_tuner/models/gemma/finetune.py). The default file ships these **`[model:…]`** entries (LoRA on top of the Hub weights):

| Model key (`config.ini`) | Hugging Face `base_model` | Notes |
| --- | --- | --- |
| `gemma-4-e2b-it` | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) | Gemma 4 instruct, ~2B effective — usual default |
| `gemma-4-e4b-it` | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | Gemma 4 instruct, ~4B effective |
| `gemma-4-e2b` | [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) | Gemma 4 base (not instruct-tuned) |
| `gemma-4-e4b` | [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) | Gemma 4 base (not instruct-tuned) |
| `gemma-3n-e2b-it` | [`google/gemma-3n-E2B-it`](https://huggingface.co/google/gemma-3n-E2B-it) | Gemma 3n instruct, ~2B effective |
| `gemma-3n-e4b-it` | [`google/gemma-3n-E4B-it`](https://huggingface.co/google/gemma-3n-E4B-it) | Gemma 3n instruct, ~4B effective |

Add your own **`[model:your-name]`** section with `group = gemma` and a compatible `base_model` if you need another **any-to-any** Gemma 3n / Gemma 4 E2B–E4B checkpoint. **Larger Gemma 4 weights** on Hugging Face (for example 26B or 31B class) use a different Transformers architecture than this trainer’s `AutoModelForCausalLM` audio path—they are **not** supported here yet.

Wizard time and memory hints come from [`gemma_tuner/wizard/base.py`](gemma_tuner/wizard/base.py) (`ModelSpecs`).

---

## Architecture (what actually calls what)

| Piece | Role |
| --- | --- |
| [`gemma_tuner/cli_typer.py`](gemma_tuner/cli_typer.py) | Canonical CLI (`gemma-macos-tuner`). Imports `core.bootstrap` early so MPS env vars exist before Torch wakes up. |
| [`gemma_tuner/core/ops.py`](gemma_tuner/core/ops.py) | Dispatches prepare → `scripts.prepare_data`, finetune → `scripts.finetune`, evaluate → `scripts.evaluate`, export → `scripts.export`. |
| [`gemma_tuner/scripts/finetune.py`](gemma_tuner/scripts/finetune.py) | **Router**: only models whose name contains `gemma` → [`gemma_tuner/models/gemma/finetune.py`](gemma_tuner/models/gemma/finetune.py). |
| [`gemma_tuner/utils/device.py`](gemma_tuner/utils/device.py) | MPS → CUDA → CPU selection, sync helpers, memory hints. |
| [`gemma_tuner/utils/dataset_utils.py`](gemma_tuner/utils/dataset_utils.py) | CSV loads, patches, blacklist/protection semantics. |
| [`gemma_tuner/wizard/`](gemma_tuner/wizard/) | Questionary + Rich UI; training is spawned with `python -m main finetune …` from the repo root (see [`gemma_tuner/wizard/runner.py`](gemma_tuner/wizard/runner.py)). |

**Run layout** (typical):

```text
output/
├── {id}-{profile}/
│   ├── metadata.json
│   ├── metrics.json
│   ├── checkpoint-*/
│   └── adapter_model/          # LoRA artifacts when applicable
```

**Configuration:** hierarchical INI—defaults, groups, models, datasets, then profiles—read by `gemma_tuner/core/config.py`. Set `GEMMA_TUNER_CONFIG` if you invoke the CLI outside the repo root.

---

## Requirements

| | |
| --- | --- |
| **Python** | **3.10+** (matches `pyproject.toml`; 3.8 is a fond memory) |
| **macOS** | 12.3+ for MPS; use **native arm64** Python—not Rosetta |
| **RAM** | 16 GB workable for small Gemma runs; more is calmer |
| **CUDA** | Optional; install the CUDA build of PyTorch that matches your driver |

---

## Installation

### 1. Create a virtual environment (do this first)

macOS’s built-in Python is 3.9 — too old. This project requires **Python 3.10+**.
Homebrew has a newer one; install it if you haven’t:

```bash
brew install python@3.12
```

Then create a virtual environment (this also gives you `pip` — macOS doesn’t ship it standalone):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt changes to `(.venv) …`. Every command below assumes the venv is active.
To reactivate in a new terminal: `source .venv/bin/activate`.

### 2. Prove you are on arm64 (Mac)

```bash
python -c "import platform; print(platform.machine())"
# arm64  ← good
# x86_64 ← wrong Python; fix before blaming MPS
```

If you see `x86_64`, your Python is running under Rosetta. Install a native arm64 Python
from [python.org](https://www.python.org/downloads/macos/) or via Homebrew (`brew install python@3.12`),
then recreate the venv.

### 3. Install PyTorch

```bash
pip install torch torchaudio
```

### 4. Install this package

```bash
pip install -e .
```

### 5. Run the wizard

```bash
gemma-macos-tuner wizard
```

![Gemma macOS Tuner wizard: system check, then LoRA / model / dataset steps](README/assets/wizard-cli.png)

The wizard walks you through model selection, dataset config, and training — answering questions and writing `config.ini` for you.

> **Before the wizard downloads model weights**, you need a Hugging Face account with access to Gemma.
> Accept the license on the [model card](https://huggingface.co/google/gemma-3n-E2B-it), then authenticate:
> ```bash
> huggingface-cli login
> ```
> Or set `HF_TOKEN` in your environment.

If something seems broken, run `gemma-macos-tuner system-check` first.

---

## CLI cheat sheet

```bash
# Dataset prep (profile names come from config.ini)
gemma-macos-tuner prepare <dataset-profile>

# Train (model in profile must be a Gemma id / local path with "gemma" in the string)
gemma-macos-tuner finetune <profile> --json-logging

# Evaluate
gemma-macos-tuner evaluate <profile-or-run>

# Export merged HF/SafeTensors tree (LoRA merged when adapter_config.json is present)
gemma-macos-tuner export <run-dir-or-profile>

# Blacklist generation from errors
gemma-macos-tuner blacklist <profile>

# Run index
gemma-macos-tuner runs list

# Guided setup
gemma-macos-tuner wizard
```

**Migration from `main.py` / old habits:** [`MIGRATION.md`](MIGRATION.md). Runs management moved to the `runs` subcommand—not a separate `manage.py` in this tree.

---

## Gemma 3n / Gemma 4 on Apple Silicon

End-to-end notes live in [`README/specifications/Gemma3n.md`](README/specifications/Gemma3n.md). Short version:

```bash
python -m gemma_tuner.scripts.gemma_preflight
python -m gemma_tuner.scripts.gemma_profiler --model google/gemma-3n-E2B-it

gemma-macos-tuner wizard

python -m gemma_tuner.scripts.gemma_tiny_overfit --profile gemma-lora-test --max-samples 32

python tools/eval_gemma_asr.py \
  --csv data/datasets/<your_dataset>/validation.csv \
  --model google/gemma-3n-E2B-it \
  --adapters output/<your_run>/ \
  --text-column text \
  --limit 200
```

**MPS reality check:** prefer bf16 when supported; attention is forced to `eager` for stability; do not leave `PYTORCH_ENABLE_MPS_FALLBACK=1` on in production (it hides silent CPU fallbacks).

---

## Data: CSVs, GCS, BigQuery

- **Local / HTTP / GCS paths** in your prepared CSV; use `gemma-macos-tuner prepare <profile> --no-download` to avoid copying GCS audio locally.
- **BigQuery import** (wizard or scripts): needs `pip install .[gcp]` and Application Default Credentials (`gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`). The wizard can materialize `_prepared.csv` and append a dataset section to `config.ini`.

Patch layout (by dataset `source`):

```text
data_patches/{source}/
├── override_text_perfect/
├── do_not_blacklist/
└── delete/
```

---

## Training visualizer (optional)

Install `viz` extras, set `visualize=true` in the profile, open the URL the trainer prints (default bind `127.0.0.1`, port starting at 8080). If Flask isn’t installed, training continues without drama.

---

## NVIDIA Granary & streaming

Large-corpus workflows: `gemma-macos-tuner prepare-granary <profile>` and streaming-oriented dataset keys—see [`README/Datasets.md`](README/Datasets.md).

---

## Apple Silicon knobs

```bash
# Debug only—surfaces unsupported ops by falling back to CPU (slow)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Cap MPS allocator appetite (try 0.7–0.9)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
```

Preprocessing worker count and dataloader settings are controlled from `config.ini`; defaults favor using available CPU cores for `Dataset.map`.

---

## CI & tests

Workflows under [`.github/workflows/`](.github/workflows/): lint (`ruff`), fast tests (`pytest -k "not slow"`), macOS smoke. Regenerate lockfiles with `pip-compile` when you change `pyproject.toml`—see comments in [`requirements.txt`](requirements.txt).

---

## Experiment index

Runs update `output/experiments.csv` and optional SQLite—handy SQL examples are still valid; swap profile names for whatever you actually train.

---

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| `Unsupported model` from finetune | Use a Gemma model id / path containing `gemma`. |
| MPS not available | macOS 12.3+, arm64 Python, current PyTorch. |
| OOM / swap storm | Smaller batch, gradient checkpointing, lower `PYTORCH_MPS_HIGH_WATERMARK_RATIO`. |
| Slow training with fallback env on | Unset `PYTORCH_ENABLE_MPS_FALLBACK` after debugging. |
| Config not found | `GEMMA_TUNER_CONFIG` or run from the directory that contains `config.ini`. |
| 401 / gated model / cannot download weights | Accept the license on the model’s Hugging Face page; run `huggingface-cli login` or set `HF_TOKEN`. |

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Prefer extending `cli_typer.py` and shared helpers in `gemma_tuner/core/` over one-off scripts.

---

## Acknowledgments

Google’s Gemma team, Hugging Face Transformers & PEFT, PyTorch MPS maintainers—and everyone who filed an issue after watching Activity Monitor turn red.

---

## License

Released under the [MIT License](LICENSE).
