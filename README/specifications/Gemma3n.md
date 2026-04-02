# Gemma 3n Multimodal Fine-Tuning Product Specification

Related: condensed developer notes — [`../guides/apple-silicon/gemma3n.md`](../guides/apple-silicon/gemma3n.md).

## Executive Summary

This document outlines the integration of Google's Gemma 3n, a state-of-the-art open multimodal model, into the Whisper Fine-Tuner framework. This extension enables users to perform parameter-efficient fine-tuning (PEFT) on Gemma 3n's audio capabilities, leveraging the framework's existing Apple Silicon (MPS) optimizations. The core of this project involves engineering a robust data pipeline to accommodate Gemma's unique audio processing requirements and extending the CLI wizard to provide a seamless, guided user experience for this new model family.

### Key Capabilities

- **Gemma 3n Audio Fine-Tuning**: Adapt Gemma 3n for domain-specific audio transcription tasks.
- **Parameter-Efficient Training**: Utilize LoRA and QLoRA to fine-tune Gemma 3n on consumer hardware.
- **Seamless Wizard Integration**: A guided, zero-configuration workflow for setting up Gemma 3n training runs.
- **Apple Silicon First**: Optimized for PyTorch on MPS, leveraging unified memory for efficient training.
- **Multimodal Data Pipeline**: A new data processor to handle the Universal Speech Model (USM) feature extraction required by Gemma 3n.

## Technical Architecture

### System Integration

Gemma 3n will be integrated as a new model family alongside Whisper. The existing architecture (`core/ops.py` dispatch, `models/*/finetune.py` structure) will be extended.

- **Model Implementation**: A new directory `models/gemma/finetune.py` will be created. It will leverage Hugging Face's `transformers` library to load `AutoModelForCausalLM` and `AutoProcessor` for Gemma 3n models.
- **Training Framework**: The project will use the `trl.SFTTrainer`, which is well-suited for Gemma's chat-based format. This requires a specialized data pipeline to format audio-text pairs into the required conversational structure.
- **Primary Toolkit**: **PyTorch with MPS** is the designated framework for this task, as recommended in the developer field guide for its mature ecosystem (`peft`, `trl`) and flexibility with complex multimodal models. MLX is explicitly avoided due to its current instability with Gemma's audio tower.

### Core Technical Challenges & Solutions

1.  **Data Preprocessing (The Critical Path)**:
    - **Challenge**: Gemma's audio encoder uses Google's Universal Speech Model (USM), which requires a specific feature extraction process, unlike Whisper's simple log-mel spectrogram.
    - **Solution**: We will create a new data preparation script, `utils/gemma_dataset_prep.py`. This script will use the official `transformers.GemmaProcessor` to handle all audio processing. This ensures perfect replication of the required feature extraction and tokenization.

2.  **Conversational Data Formatting**:
    - **Challenge**: `SFTTrainer` for Gemma requires input data in a specific chat format with special tokens (`<bos>`, `<start_of_turn>`, `<end_of_turn>`).
    - **Solution**: The new data prep script will include a function to transform simple `(audio, text)` pairs into the required JSONL format with the correct conversational structure. The `<bos>` token will be prepended to every training example, a strict requirement for stable training.

3.  **Numerical Stability on MPS**:
    - **Challenge**: Gemma was pre-trained using `bfloat16`. The PyTorch MPS backend can be sensitive to floating-point precision, potentially leading to `NaN` loss values when using the default `float16`.
    - **Solution**: The training script (`models/gemma/finetune.py`) and wizard will default to using `bfloat16` (`bf16=True` in `SFTConfig`) when an MPS device is detected. If the hardware does not support `bfloat16`, it will fall back to full `float32`, and the user will be warned about increased memory usage.

## CLI Wizard Integration (`wizard.py`) - ✅ COMPLETED

The wizard has been successfully extended to make Gemma 3n a first-class citizen with full progressive disclosure support.

### Implemented Wizard Flow

1.  **✅ Top-Level Model Family Selection**:
    The wizard now presents model family choice as the first step after welcome.
    ```
    ? Choose the model family you want to work with:
      ❯ 🌬️ Whisper - The robust ASR model from OpenAI.
        💎 Gemma - The new multimodal model from Google.
    ```

2.  **✅ Gemma Model Selection with Hardware Gating**:
    When "Gemma" is selected, the wizard displays hardware-appropriate options.
    ```
    ? Which Gemma 3n model do you want to fine-tune?
      ❯ gemma-3n-e2b-it (Elastic 2B) - Faster, smaller memory footprint. ⭐ Recommended
        gemma-3n-e4b-it (Elastic 4B) - Maximum capability, higher memory usage.
    ```
    **Memory Gating**: Uses `ModelSpecs.MODES` with 20% safety buffer to hide infeasible options based on available system memory.

3.  **✅ Training Method Restriction**:
    For Gemma models, only LoRA is available due to memory requirements.
    ```
    ? Choose your training method for Gemma:
      ❯ 🎨 LoRA Fine-Tune - Optimized for Gemma 3n on consumer hardware.
    ```
    **Note**: Standard fine-tuning is automatically hidden for Gemma models to prevent memory issues.

4.  **✅ Automatic Configuration Management**:
    The wizard handles all Gemma-specific optimizations transparently:
    - **Data Type Optimization**: Prefers `bfloat16` on MPS; falls back to `float32` if unsupported
    - **Attention Implementation**: Forces `eager` attention for MPS stability
    - **Chat Template**: Automatically configures proper multimodal message formatting
    - **LoRA Configuration**: Uses optimal settings (`rank=16`, `alpha=32`) for Gemma architecture
    - **Memory Settings**: Applies conservative memory limits for stable training

5.  **✅ Enhanced Confirmation Screen**:
    The confirmation screen displays Gemma-specific configuration details:
    ```
    ┌─────────────────────────────────────┐
    │ Training Configuration              │
    ├─────────────────────────────────────┤
    │ Family:     💎 Gemma                 │
    │ Model:      gemma-3n-e2b-it         │
    │ Method:     🎨 LoRA Fine-Tune       │
    │ Dataset:    common_voice (50k)      │
    │ Data Type:  bfloat16                │
    │ Attention:  eager                   │
    │ LoRA Rank:  16                      │
    │ Device:     Apple Silicon (mps)     │
    └─────────────────────────────────────┘
    ```

### Wizard Integration Features

- **✅ Progressive Disclosure**: Complex Gemma settings are handled automatically
- **✅ Hardware Awareness**: Memory gating prevents selection of incompatible models
- **✅ Platform Optimization**: Automatic MPS/CUDA/CPU configuration
- **✅ Error Prevention**: Invalid combinations are prevented at selection time
- **✅ User Experience**: Seamless flow with clear feedback and recommendations

## Configuration System (`config.ini`)

To support Gemma, the configuration system will be extended with a new group and model profiles.

### New `[group:gemma]` Section

```ini
[group:gemma]
# Common settings for all Gemma models
attn_implementation = eager
dtype = bfloat16 ; Critical for MPS stability
optim = paged_adamw_32bit
```

### New Model/Profile Sections

```ini
[model:gemma-3n-e2b-it]
base_model = google/gemma-3n-E2B-it
group = gemma

[profile:gemma-lora-test]
inherits = DEFAULT
model = gemma-3n-e2b-it
dataset = test_streaming
method = lora
lora_r = 16
lora_alpha = 32
target_modules = q_proj,k_proj,v_proj,o_proj
```

## Limitations and Challenges

- **High Memory Requirements**: Even with LoRA, fine-tuning Gemma 3n's audio tower is memory-intensive. Full SFT will be impractical on most consumer hardware.
- **Data Pipeline Complexity**: The dependency on the `GemmaProcessor` and the specific chat template makes the data pipeline more fragile than Whisper's. Any deviation will lead to poor results.
- **Initial Scope**: The initial integration will focus on LoRA fine-tuning for audio transcription. Other modalities (vision) and training methods (distillation) are out of scope for the first version.
- **MLX Instability**: As noted in the field guide, `mlx-lm` has known issues with Gemma's audio tower. This integration will **only** support the PyTorch MPS backend.

## Implementation Progress (track your progress and take notes below)

- [x] Created `models/gemma/finetune.py` with Gemma 3n LoRA trainer
  - Loads `AutoModelForCausalLM` + `AutoProcessor`
  - Prefers bf16 on MPS (probe); falls back to float32
  - Uses eager attention; injects LoRA (`q_proj,k_proj,v_proj,o_proj` with auto-discovery fallback)
  - Implements `DataCollatorGemmaAudio` that delegates multimodal packing to the processor
  - Integrates with existing dataset loader (`utils.dataset_utils.load_dataset_split`)
  - Saves adapters and `train_results.json`
- [x] Added Gemma routing in orchestrator: `scripts/finetune.py` now detects `gemma` models and dispatches `models.gemma.finetune`
- [x] Added environment preflight: `scripts/gemma_preflight.py` (arm64, MPS availability, bf16 probe, memory tips)
- [x] Added quick profiler: `scripts/gemma_profiler.py` (loads model, runs tiny forward, reports dtype/time/RSS)
- [x] Updated `config.ini`
  - Added `[group:gemma]` with `dtype=bfloat16`, `attn_implementation=eager`
  - Added `[model:gemma-3n-e2b-it]` and `[model:gemma-3n-e4b-it]`
  - Added example `[profile:gemma-lora-test]` using `test_streaming`
- [x] Add `utils/gemma_dataset_prep.py` (JSONL writer; optional if processor-based collator suffices)
- [x] Add `scripts/gemma_generate.py` (load base + adapters; transcribe a WAV)
- [ ] Wizard integration
  - [ ] Add top-level Model Family selection (Whisper/Gemma)
  - [x] Add top-level Model Family selection (Whisper/Gemma)
  - [x] Add Gemma models to wizard gating table for memory checks (`ModelSpecs.MODES`)
  - [x] Include Gemma models when user selects LoRA method in wizard model list
  - [x] Gemma-only method: LoRA (SFT hidden for now)
  - [x] Memory gating for E2B/E4B choices (uses ModelSpecs and available memory with 20% safety buffer)
  - [x] Confirmation screen: show dtype/attention for Gemma and enforce `attn_implementation=eager` in profile
- [ ] Tests
  - [x] Unit: `DataCollatorGemmaAudio` produces required keys; `<bos>` presence via processor template
  - [x] Tiny overfit (16–64 samples) sanity on MPS (`scripts/gemma_tiny_overfit.py`)
- [ ] Eval utilities
  - [x] `tools/eval_gemma_asr.py` (WER/CER via jiwer)
- [ ] Docs
  - [x] Add README section with setup, preflight, example run and known caveats

Notes:
- bitsandbytes is not used on macOS; optimizer defaults to AdamW.
- TRL SFTTrainer was not assumed; using vanilla Trainer + custom collator for robustness.

## Quickstart: Gemma 3n on Apple Silicon (MPS)

### 1) Environment & Installation

```
python -c "import platform; print(platform.platform())"  # Must show arm64

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate peft jiwer soundfile
```

Recommended MPS env (memory pressure control):

```
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
```

### 2) Preflight & Profiling

```
python scripts/gemma_preflight.py
python scripts/gemma_profiler.py --model google/gemma-3n-E2B-it
```

### 3) Run the Wizard (LoRA on Gemma)

```
python wizard.py
# Step 0: Choose "Gemma" family → LoRA → gemma-3n-e2b-it (⭐ Recommended)
# Wizard enforces attn_implementation=eager for Gemma; bf16 preferred on MPS.
```

### 4) Tiny Overfit Sanity (Optional)

```
python scripts/gemma_tiny_overfit.py --profile gemma-lora-test --max-samples 32
```

### 5) Evaluate (WER/CER)

```
python tools/eval_gemma_asr.py \
  --csv data/datasets/<your_dataset>/validation.csv \
  --model google/gemma-3n-E2B-it \
  --adapters output/<your_run>/ \
  --text-column text \
  --limit 200
```

### Known Caveats (MPS)

- Gemma prefers bfloat16. On MPS, we probe bf16; if unavailable we fall back to float32.
- Attention implementation is forced to `eager` for stability on MPS.
- Do not enable `PYTORCH_ENABLE_MPS_FALLBACK` in production; it silently moves ops to CPU.
- Avoid frequent `.item()` calls during training; they force GPU sync and hurt throughput.
- `bitsandbytes`/QLoRA are not used on macOS; we use AdamW with gradient accumulation.
