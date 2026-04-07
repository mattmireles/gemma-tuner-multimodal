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

Train **Google Gemma** multimodal (audio + text) models on your own data—with **LoRA**, **PyTorch**, and first-class **Apple Silicon (MPS)** support. The CLI is boring on purpose; your models do not have to be.

**Source:** [github.com/mattmireles/gemma-tuner-multimodal](https://github.com/mattmireles/gemma-tuner-multimodal) (public).

---

## The honest pitch

This repository is a **Gemma-first** toolkit: the training path loads Hugging Face Gemma checkpoints, injects PEFT LoRA adapters, and runs the supervised fine-tuning loop in `gemma_tuner/models/gemma/finetune.py`. If your profile’s model name does not contain `gemma`, `gemma_tuner/scripts/finetune.py` will refuse—by design.

Export produces a **merged or plain Hugging Face / SafeTensors directory** for downstream use (`gemma_tuner/scripts/export.py`). For Core ML conversion and GGUF-style local inference tooling, start with [`README/guides/README.md`](README/guides/README.md); this repo’s **training** path is Gemma-only.

---

## Features

- **Apple Silicon first**: MPS-friendly defaults, memory pressure knobs, and docs that admit when Metal is quirky.
- **Gemma multimodal LoRA**: Audio + text via Transformers + PEFT; bf16 on MPS when the hardware agrees.
- **Cross-platform**: CUDA and CPU fall back cleanly when you are not on a Mac.
- **Typer CLI**: `gemma-macos-tuner` is the interface you want; `main.py` remains for automation and habits.
- **Data hygiene**: Patch directories, blacklists, and protection lists so one bad clip does not wreck a run.
- **Streaming data**: Stream data from GCS to your local machine, pull slices from BigQuery (optional extras), prepare Granary-scale corpora.
- **Optional eye candy**: Live training visualizer behind the `viz` extra (Flask + Socket.IO).
- **Interactive wizard**: `gemma-macos-tuner wizard`—questions, sane defaults, fewer foot-guns.

**Deeper reading:** curated field guides in [`README/guides/README.md`](README/guides/README.md); Gemma product notes in [`README/specifications/Gemma3n.md`](README/specifications/Gemma3n.md).

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

### 1. Prove you are on arm64 (Mac)

```bash
python -c "import platform; print(platform.machine())"
# arm64  ← good
# x86_64 ← wrong Python; fix before blaming MPS
```

### 2. Install PyTorch (you choose the flavor)

```bash
# Apple Silicon / CPU wheels
pip install torch torchaudio

# NVIDIA (example CUDA 12.x index—adjust to PyTorch’s current docs)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install this package

```bash
pip install .
# or
pip install -e .
```

Optional extras (see `pyproject.toml`):

- `gemma-macos-tuner[torch]` — declares Torch if your tooling insists
- `eval` — WER/CER via `jiwer`
- `gcp` — BigQuery + GCS client libraries
- `viz` — training visualizer
- `dev` — ruff, pytest

### 4. Smoke-test the stack

```bash
gemma-macos-tuner system-check
```

### 5. Before you run training (first time)

- **Hugging Face:** Gemma checkpoints are **gated** on the Hub. Open each model card (see [Supported models](#supported-models)), accept Google’s terms, then authenticate so downloads work: install [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/quick-start) and run `huggingface-cli login`, or set **`HF_TOKEN`** in the environment for CI/servers.
- **Config file:** The CLI loads **`config.ini`** from the current working directory unless you set **`GEMMA_TUNER_CONFIG`** to an absolute path—handy when you are not in the repo root.
- **Data:** Training reads **prepared CSVs** and **`[dataset:…]`** sections in that INI (local paths, GCS, Granary, BigQuery flows)—not raw Hugging Face Hub dataset names. See [Data: CSVs, GCS, BigQuery](#data-csvs-gcs-bigquery) and [`README/Datasets.md`](README/Datasets.md).

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

# Legacy entrypoints (narrow)
gemma-macos-tuner legacy main   # forwards to gemma_tuner.main if you really need it
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
