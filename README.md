# Gemma Multimodal Fine-Tuner

![Gemma macOS Tuner wizard: system check, then LoRA / model / dataset steps](README/assets/wizard-cli.png)

**Fine-tune Gemma on text, images, *and* audio — on your Mac, on data that doesn't fit on your Mac.**

- 🖼️ **Image + text LoRA** — captioning and VQA on local CSV.
- 🎙️ **Audio + text LoRA** — Apple-Silicon-native, no CUDA required.
- 📝 **Text-only LoRA** — instruction or completion on CSV.
- ☁️ **Stream from GCS / BigQuery** — train on terabytes without filling your SSD.
- 🍎 **Runs on Apple Silicon** — MPS-native, no NVIDIA box required.

**Source:** [github.com/mattmireles/gemma-tuner-multimodal](https://github.com/mattmireles/gemma-tuner-multimodal) (public).

---

## Watch your model learn

![Real-time training visualizer: loss curve, attention heatmap, gradient signal, memory, and token predictions — updating live as training runs on Apple Silicon](README/assets/training-visualizer.gif)

Loss curve. Attention heatmap. Gradient signal strength. Memory pressure. Token-by-token predictions — all updating in real time, in your browser, while the model trains on your Mac. No TensorBoard. No notebook. One flag in your config, one URL in your terminal.

→ [Setup takes 30 seconds](#training-visualizer)

---

## LoRA for Gemma 4 & 3n — why not just use…?

| | **This** | MLX-LM | Unsloth | axolotl |
| --- | :-: | :-: | :-: | :-: |
| Fine-tune Gemma (text-only CSV) | ✅ | ✅ | ✅ | ✅ |
| Fine-tune Gemma **image + text** (caption / VQA CSV) | ✅ | ⚠️ varies | ⚠️ varies | ⚠️ varies |
| Fine-tune Gemma **audio + text** | ✅ | ❌ | ❌ | ⚠️ CUDA only |
| Runs on Apple Silicon (MPS) | ✅ | ✅ | ❌ | ❌ |
| **Stream training data from cloud** | ✅ | ❌ | ❌ | ⚠️ partial |
| No NVIDIA GPU required | ✅ | ✅ | ❌ | ❌ |

Fine-tune Gemma on **text, images, or audio** without renting an H100 or copying a terabyte of data to your laptop. All three modalities run on Apple Silicon.

**Text-only fine-tuning** (instruction or completion on CSV) is supported: set `modality = text` in your profile and use local CSV splits under `data/datasets/<name>/`. See [Text-only fine-tuning](#text-only-fine-tuning) below.

**Image + text fine-tuning** (captioning or VQA on local CSV) uses `modality = image`, `image_sub_mode`, and `image_token_budget`; see [Image fine-tuning](#image-fine-tuning) below. v1 is **local CSV only** (same constraint as text-only).

**How it works:** Hugging Face Gemma checkpoints + PEFT LoRA, supervised fine-tuning in [`gemma_tuner/models/gemma/finetune.py`](gemma_tuner/models/gemma/finetune.py), exported as a merged HF / SafeTensors tree by [`gemma_tuner/scripts/export.py`](gemma_tuner/scripts/export.py). For Core ML conversion and GGUF inference tooling, see [`README/guides/README.md`](README/guides/README.md) — this repo's *training* path is Gemma-only by design.

**Deeper reading:** [`README/guides/README.md`](README/guides/README.md) · [`README/specifications/Gemma3n.md`](README/specifications/Gemma3n.md)

---

## What you can build with this

- **Domain-specific ASR** — fine-tune on medical dictation, legal depositions, call-center recordings, or any field where off-the-shelf Whisper / Gemma mishears the jargon.
- **Domain-specific vision** — captioning or VQA on receipts, charts, screenshots, manufacturing defects, medical imagery — any visual domain where generic models hallucinate.
- **Document & screen understanding** — train on screenshot → structured-output pairs for UI agents, OCR-adjacent pipelines, or chart QA.
- **Accent, dialect, and low-resource language adaptation** — adapt a base Gemma model to underrepresented voices and languages with your own labeled audio.
- **Multimodal assistants** — extend Gemma's text reasoning with image *or* audio grounding for transcription, captioning, and Q&A pipelines.
- **Private, on-device pipelines** — train and run entirely on your Mac. Data never leaves the machine; weights never touch a third-party API.

If your data lives in GCS or BigQuery, you can do all of this on a laptop without copying terabytes locally — the dataloader streams shards on demand.

---

## Supported models

Training targets **Gemma multimodal (text + image + audio)** checkpoints loaded via `base_model` in [`config/config.ini`](config/config.ini) and routed to [`gemma_tuner/models/gemma/finetune.py`](gemma_tuner/models/gemma/finetune.py). The default file ships these **`[model:…]`** entries (LoRA on top of the Hub weights):

| Model key (in `config/config.ini`) | Hugging Face `base_model` | Notes |
| --- | --- | --- |
| `gemma-4-e2b-it` | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) | Gemma 4 instruct, ~2B — requires `requirements/requirements-gemma4.txt` (see Installation) |
| `gemma-4-e4b-it` | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | Gemma 4 instruct, ~4B — requires Gemma 4 stack |
| `gemma-4-e2b` | [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) | Gemma 4 base — requires Gemma 4 stack |
| `gemma-4-e4b` | [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) | Gemma 4 base — requires Gemma 4 stack |
| `gemma-3n-e2b-it` | [`google/gemma-3n-E2B-it`](https://huggingface.co/google/gemma-3n-E2B-it) | Gemma 3n instruct, ~2B — **default** on the base `pip install -e .` pin |
| `gemma-3n-e4b-it` | [`google/gemma-3n-E4B-it`](https://huggingface.co/google/gemma-3n-E4B-it) | Gemma 3n instruct, ~4B |

Add your own **`[model:your-name]`** section with `group = gemma` and a compatible `base_model` if you need another **any-to-any** Gemma 3n / Gemma 4 E2B–E4B checkpoint. **Larger Gemma 4 weights** on Hugging Face (for example 26B or 31B class) use a different Transformers architecture than this trainer’s `AutoModelForCausalLM` audio path—they are **not** supported here yet.

Wizard time and memory hints come from [`gemma_tuner/wizard/base.py`](gemma_tuner/wizard/base.py) (`ModelSpecs`).

---

## Architecture

| Piece | Role |
| --- | --- |
| [`gemma_tuner/cli_typer.py`](gemma_tuner/cli_typer.py) | Canonical CLI (`gemma-macos-tuner`). Imports `core.bootstrap` early so MPS env vars are set before Torch is loaded. |
| [`gemma_tuner/core/ops.py`](gemma_tuner/core/ops.py) | Dispatches prepare → `scripts.prepare_data`, finetune → `scripts.finetune`, evaluate → `scripts.evaluate`, export → `scripts.export`. |
| [`gemma_tuner/scripts/finetune.py`](gemma_tuner/scripts/finetune.py) | **Router**: only models whose name contains `gemma` → [`gemma_tuner/models/gemma/finetune.py`](gemma_tuner/models/gemma/finetune.py). |
| [`gemma_tuner/utils/device.py`](gemma_tuner/utils/device.py) | MPS → CUDA → CPU selection, sync helpers, memory hints. |
| [`gemma_tuner/utils/dataset_utils.py`](gemma_tuner/utils/dataset_utils.py) | CSV loads, patches, blacklist/protection semantics. |
| [`gemma_tuner/wizard/`](gemma_tuner/wizard/) | Questionary + Rich UI; training is spawned with `python -m gemma_tuner.main finetune …` from the repo root (see [`gemma_tuner/wizard/runner.py`](gemma_tuner/wizard/runner.py)). |

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
| **Python** | **3.10+** (matches `pyproject.toml`) |
| **macOS** | 12.3+ for MPS; use **native arm64** Python, not Rosetta |
| **RAM** | 16 GB minimum for the smaller Gemma runs; 32 GB+ recommended |
| **CUDA** | Optional; install the CUDA build of PyTorch that matches your driver |

---

## Installation

### 1. Create a Python 3.10+ virtual environment

macOS's built-in Python is 3.9, which is too old. Install a newer one with Homebrew:

```bash
brew install python@3.12
```

Then create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Every command below assumes the venv is active. To reactivate in a new terminal:
`source .venv/bin/activate`.

### 2. Confirm you are on arm64 (Apple Silicon)

```bash
python -c "import platform; print(platform.machine())"
# arm64  -> good
# x86_64 -> Python is running under Rosetta; install a native arm64 Python and recreate the venv
```

A native arm64 Python is available from [python.org](https://www.python.org/downloads/macos/)
or Homebrew (`brew install python@3.12`).

### 3. Install PyTorch

```bash
pip install torch torchaudio
```

### 4. Install this package

```bash
pip install -e .
```

### 5. Authenticate with Hugging Face

Gemma weights are gated. Accept the license on the
[model card](https://huggingface.co/google/gemma-3n-E2B-it), then either log in or
export a token:

```bash
huggingface-cli login
# or:  export HF_TOKEN=hf_...
```

### 6. Gemma 4 (optional)

The base install (`pip install -e .`) pins Transformers ≥5.5 — both **Gemma 3n** and **Gemma 4** families work out of the box. Gemma 4 checkpoints need a slightly newer PEFT:

```bash
pip install -r requirements/requirements-gemma4.txt
```

`finetune` and `export` are family-aware. A few non-training commands (`gemma_generate`, multimodal probing, ASR eval) still reject Gemma 4 ids until those code paths are upgraded.

### 7. Run the wizard

```bash
gemma-macos-tuner wizard
```

The wizard is the primary UI: it picks the model, walks you through dataset and
hyperparameter selection, and starts training. On first run it creates
`config/config.ini` for you from the committed
[`config/config.ini.example`](config/config.ini.example) template (the live config
is gitignored because the wizard writes local paths and GCP project IDs into it).

If a command fails, run `gemma-macos-tuner system-check` first to surface
environment issues.

---

## Zero to training in 90 seconds

The repo ships a 16-row instruction-tuning dataset at [`data/datasets/sample-text/`](data/datasets/sample-text/) — translations, summaries, trivia, haiku, JSON conversion. Small enough to finish in under a minute. Large enough to prove the full pipeline works: data loading, tokenization, LoRA, checkpointing, export.

```bash
gemma-macos-tuner wizard
```

Pick **Instruction tuning → gemma-3n-e2b-it → sample-text**, accept the defaults, and watch it train. First run downloads ~5 GB of base weights from Hugging Face (step 5 above must be done). Every run after that starts in seconds.

Or skip the wizard entirely:

```bash
gemma-macos-tuner finetune sample-text
```

Once the sample run finishes, drop your own CSV under `data/datasets/<your-name>/` and run the wizard again — it picks up new datasets automatically.

---

## What kind of data do I need?

All training data is **CSV** under `data/datasets/<name>/`, with one row per
example and a header row. The required columns depend on the modality. Each
dataset directory holds at least:

```text
data/datasets/<name>/
├── train.csv
└── validation.csv
```

There is no JSONL / Parquet / Hugging Face dataset format requirement — just CSV.
The column **names** are configurable via `prompt_column`, `text_column`, and
`image_path_column` in your profile; the names below are the defaults used by
[`config/config.ini.example`](config/config.ini.example).

### Text instruction (`modality = text`, `text_sub_mode = instruction`)

```csv
id,prompt,response
1,Translate to French: Good morning.,Bonjour.
2,What is the capital of Japan?,Tokyo.
```

The prompt is masked from the loss; the model only learns to generate `response`.
This is what the bundled `sample-text` dataset uses.

### Text completion (`modality = text`, `text_sub_mode = completion`)

```csv
id,text
1,"Once upon a time, in a small village by the sea, ..."
```

A single text column; the full sequence is trained (no prompt mask). Useful for
domain pretraining-style adaptation.

### Image + text (`modality = image`)

```csv
id,image_path,caption
1,images/receipt_001.jpg,"Total: $42.18, paid in cash"
2,images/receipt_002.jpg,"Subtotal $19.99, tax $1.60, total $21.59"
```

`image_path` is resolved relative to the dataset directory (or an absolute path).
For VQA, set `image_sub_mode = vqa` and use `image_path,question,answer` columns.
See [Image fine-tuning](#image-fine-tuning) for details.

### Audio + text (`modality = audio`, the default)

```csv
id,audio_path,text,language,duration
1,audio/sample_001.wav,"the quick brown fox jumps over the lazy dog",en,2.4
```

`audio_path` points at decoded WAV files (16 kHz mono recommended). The
`gemma-macos-tuner prepare` command will fetch and decode audio for you if you
provide an `audio_url` column instead. See [`README/Datasets.md`](README/Datasets.md)
for the full schema and the GCS / BigQuery streaming variants.

---

## CLI cheat sheet

```bash
# Dataset prep (profile names come from config/config.ini)
gemma-macos-tuner prepare <dataset-profile>

# Train (model in profile must be a Gemma id / local path with "gemma" in the string)
gemma-macos-tuner finetune <profile> --json-logging

# Evaluate
gemma-macos-tuner evaluate <profile-or-run>

# Export merged HF/SafeTensors tree (LoRA merged when adapter_config.json is present)
gemma-macos-tuner export <run-dir-or-profile>

# Exported models and completed runs include a .integrity.json manifest for
# corruption/drift detection. Verification is intentionally strict about
# unexpected extra tracked files. This is integrity only, not signing/authenticity.

# Blacklist generation from errors
gemma-macos-tuner blacklist <profile>

# Run index
gemma-macos-tuner runs list

# Guided setup
gemma-macos-tuner wizard
```

**Migration from `main.py` / old habits:** [`docs/MIGRATION.md`](docs/MIGRATION.md). Runs management moved to the `runs` subcommand—not a separate `manage.py` in this tree.

---

## Text-only fine-tuning

Train on **CSV text** (local splits under `data/datasets/<name>/`) without audio. v1 supports **local CSV only** — not BigQuery or Granary streaming (those remain audio-oriented).

Set in your `[profile:…]` (see also [`README/Datasets.md`](README/Datasets.md)):

- `modality = text`
- `text_sub_mode = instruction` — user/assistant turns: set `prompt_column` and `text_column` (response).
- `text_sub_mode = completion` — one column; the full sequence is trained (no prompt mask).

Optional: `max_seq_length` (default `2048`).

**Instruction example** (profile snippet):

```ini
modality = text
text_sub_mode = instruction
text_column = response
prompt_column = prompt
max_seq_length = 2048
```

**Completion example**:

```ini
modality = text
text_sub_mode = completion
text_column = text
max_seq_length = 2048
```

The checkpoint is still a multimodal Gemma `AutoModelForCausalLM`; the USM audio tower weights remain in memory in v1 even when you only train on text. See [`README/KNOWN_ISSUES.md`](README/KNOWN_ISSUES.md).

---

## Image fine-tuning

Train on **image + text** pairs from **local CSV** splits under `data/datasets/<name>/` (`train.csv` / `validation.csv`). v1 supports **captioning** (`image_sub_mode = caption`) and **VQA** (`image_sub_mode = vqa`). See [`README/Datasets.md`](README/Datasets.md) for all keys.

- **Caption / OCR-style:** user turn = image + fixed instruction (“Describe this image.”); assistant = your caption column.
- **VQA:** user turn = image + question (`prompt_column`); assistant = answer (`text_column`).

**Profile snippet (caption):**

```ini
modality = image
image_sub_mode = caption
text_column = caption
image_path_column = image_path
image_token_budget = 280
```

**Profile snippet (VQA):**

```ini
modality = image
image_sub_mode = vqa
prompt_column = question
text_column = answer
image_path_column = image_path
image_token_budget = 560
```

`image_token_budget` must be one of **70, 140, 280, 560, 1120**. Use the **same** value at inference as during training. Higher budgets improve detail but increase memory and step time on MPS. Export saves the processor next to weights; if `metadata.json` from the run is present, export reapplies the stored budget to the processor for consistency.

---

## Gemma 3n / Gemma 4 on Apple Silicon

End-to-end notes live in [`README/specifications/Gemma3n.md`](README/specifications/Gemma3n.md). Multimodal Gemma 4 + MPS field guide: [`README/guides/apple-silicon/gemma4-guide.md`](README/guides/apple-silicon/gemma4-guide.md). Common commands:

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

**MPS notes:** prefer bf16 when supported; attention is forced to `eager` for stability; unset `PYTORCH_ENABLE_MPS_FALLBACK=1` after debugging — leaving it on hides silent CPU fallbacks.

---

## Data: CSVs, GCS, BigQuery

- **Local / HTTP / GCS paths** in your prepared CSV; use `gemma-macos-tuner prepare <profile> --no-download` to avoid copying GCS audio locally.
- **BigQuery import** (wizard or scripts): needs `pip install .[gcp]` and Application Default Credentials (`gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`). The wizard can materialize `_prepared.csv` and append a dataset section to `config/config.ini`.

Patch layout (by dataset `source`):

```text
data_patches/{source}/
├── override_text_perfect/
├── do_not_blacklist/
└── delete/
```

---

## Training visualizer

Six live panels in your browser while the model trains:

| Panel | What it shows |
| --- | --- |
| **Loss curve** | Per-step loss over time — the single most important number in training |
| **Attention heatmap** | Where the model is looking across the input, layer by layer |
| **Signal strength** | Gradient norm — are the updates meaningful or vanishing? |
| **Step size** | Learning rate at each step (schedule + warmup visible at a glance) |
| **Memory** | GPU/MPS memory in GB — catch pressure before it becomes a crash |
| **Token predictions** | Top-5 next-token probabilities — watch the model's guesses sharpen in real time |

**Setup:**

```bash
pip install -e ".[viz]"
```

Then set `visualize = true` in your profile and run training. The trainer prints a URL (default `127.0.0.1:8080`). Open it. That's it.

If Flask isn't installed, training still runs — the visualizer is skipped silently. No dependency, no breakage.

---

## NVIDIA Granary & streaming

Large-corpus workflows: `gemma-macos-tuner prepare-granary <profile>` and streaming-oriented dataset keys—see [`README/Datasets.md`](README/Datasets.md).

---

## Apple Silicon knobs

```bash
# Debug only—surfaces unsupported ops by falling back to CPU (slow)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Cap MPS allocator high-water mark (try 0.7–0.9)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
```

Preprocessing worker count and dataloader settings are controlled from `config/config.ini`; defaults favor using available CPU cores for `Dataset.map`.

---

## CI & tests

Workflows under [`.github/workflows/`](.github/workflows/): lint (`ruff`), fast tests (`pytest -k "not slow"`), macOS smoke. Regenerate lockfiles with `pip-compile` when you change `pyproject.toml`—see comments in [`requirements/requirements.txt`](requirements/requirements.txt).

---

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| `Unsupported model` from finetune | Use a Gemma model id / path containing `gemma`. |
| MPS not available | macOS 12.3+, arm64 Python, current PyTorch. |
| OOM / swap storm | Smaller batch, gradient checkpointing, lower `PYTORCH_MPS_HIGH_WATERMARK_RATIO`. |
| Slow training with fallback env on | Unset `PYTORCH_ENABLE_MPS_FALLBACK` after debugging. |
| Config not found | `GEMMA_TUNER_CONFIG`, or run from the repo with `config/config.ini`, or pass `--config`. |
| 401 / gated model / cannot download weights | Accept the license on the model’s Hugging Face page; run `huggingface-cli login` or set `HF_TOKEN`. |

---

## Contributing

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). Prefer extending `cli_typer.py` and shared helpers in `gemma_tuner/core/` over one-off scripts.

---

## Acknowledgments

Google's Gemma team, Hugging Face Transformers & PEFT, and the PyTorch MPS maintainers.

---

## License

If your data lives in a bucket and your GPU lives in your lap, this was built for you.

Released under the [MIT License](LICENSE).
