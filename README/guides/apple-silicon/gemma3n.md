# Gemma 3n on Apple Silicon: A Developer's Field Guide

> **Note:** Gemma 4 (`google/gemma-4-E2B`) is now the primary fine-tuning target. The architecture covered here (USM audio encoder, LoRA targets, MPS setup) applies equally to Gemma 4. Gemma 3n remains supported. For Gemma 4–specific multimodal LoRA + MPS pitfalls, see [`gemma4-guide.md`](gemma4-guide.md).

Product integration and CLI behavior: [`../../specifications/Gemma3n.md`](../../specifications/Gemma3n.md).

This guide provides a condensed overview of the Gemma 3n architecture, with a practical focus on fine-tuning and deployment on Apple Silicon hardware.

---

## Part 1: Gemma 3n Architecture Essentials

Gemma 3n is an open model family from Google, purpose-built for on-device, or "edge," computing. It is engineered for efficiency and real-time responsiveness on consumer devices.

### The MatFormer Engine: Elastic Inference
At its core is the **MatFormer (Matryoshka Transformer)** architecture, which enables "elastic inference." A larger model contains smaller, fully functional versions of itself.

-   **E2B & E4B Models**: Released as 5B and 8B raw parameter models, they can operate with the memory footprint of traditional 2B and 4B models. The E4B model contains the E2B model as a nested, fully optimized sub-model.
-   **Deployment Options**:
    -   Deploy **E4B** for maximum capability.
    -   Deploy the standalone **E2B** sub-model for up to 2x faster inference and a smaller memory footprint.

### Memory and Compute Management
Three key innovations make it feasible to run these models on devices with as little as 2-3 GB of available memory.

1.  **Per-Layer Embedding (PLE) Caching**: Only core transformer weights are loaded into the accelerator (GPU). The large but less-frequently accessed per-layer embedding parameters are kept on the CPU and paged into accelerator memory as needed.
2.  **Conditional Parameter Loading**: Developers can skip loading entire modality-specific components (vision, audio) if they are not required for a given task (e.g., a text-only chatbot), significantly saving memory.
3.  **KV Cache Sharing**: The keys (K) and values (V) from the middle attention layer are shared with all subsequent top layers. This reduces redundant computations during prompt processing (prefill), delivering a ~2x improvement in prefill performance.

---

## Part 2: Multimodal Components

### Vision System: MobileNet-V5
-   **Encoder**: `MobileNet-V5-300M`, optimized for mobile hardware.
-   **Performance**: Can process video at up to 60 FPS on a Google Pixel device.
-   **Input Resolutions**: Natively supports `256x256`, `512x512`, and `768x768` pixels, allowing a trade-off between speed and detail.
-   **Output**: Encodes image data into a fixed-size sequence of 256 "soft tokens" for the language model.

### Audio System: USM Integration
-   **Encoder**: Based on Google's Universal Speech Model (USM).
-   **Tasks**: Performs on-device Automatic Speech Recognition (ASR) and Automatic Speech Translation (AST).
-   **Processing Rate**: Generates one token for every 160 milliseconds of sound (~6.25 tokens/second).
-   **Constraint**: The initial release is limited to processing audio clips of up to 30 seconds. The underlying architecture is capable of streaming, which may be enabled in future updates.

### Language Core and Text Processing
-   **Architecture**: Standard decoder-only transformer.
-   **Context Window**: 32,000 tokens.
-   **Multilingual Support**: Trained on text from over 140 languages.
-   **Chat Template (Crucial)**: Turns must be structured with special tokens.
    ```
    <start_of_turn>user
    ...user message...
    <end_of_turn>
    <start_of_turn>model
    ...model response...
    <end_of_turn>
    ```
-   **Fine-Tuning Requirement (Crucial)**: Each training example **must** begin with a beginning-of-sequence (`<bos>`) token to ensure stable training.

---

## Part 3: A Practical Guide to Fine-Tuning

### Supervised Fine-Tuning (SFT) with TRL

**1. Environment Setup**
```bash
# Install core and Hugging Face libraries
pip install -U torch transformers datasets accelerate peft trl bitsandbytes
```

**2. Dataset Preparation**
Training data for conversational SFT must be a list of dictionaries with "role" and "content" keys.

```python
from datasets import load_dataset

# Example function to format a dataset
def format_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["response"]}
        ]
    }

# Load and format a dataset from the Hub
raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
formatted_dataset = raw_dataset.map(format_conversation, remove_columns=raw_dataset.features)
```

**3. Training Configuration**
Use the `SFTConfig` class to manage hyperparameters.

```python
from trl import SFTConfig

training_args = SFTConfig(
    output_dir="./gemma-3n-sft-dolly",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    bf16=True, # Set to True if your GPU supports bfloat16
    logging_steps=10,
    save_strategy="epoch",
)
```

**4. Execution**
Instantiate the `SFTTrainer` and start training.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

model_id = "google/gemma-4-E2B"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
    max_seq_length=1024,
)

# Start the training process
trainer.train()
```

### Parameter-Efficient Tuning with LoRA/QLoRA

For hardware with limited VRAM, LoRA freezes most model weights and injects small, trainable "adapter" matrices.

**1. LoRA Configuration**
Use `LoraConfig` to define the adapters.

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**2. Quantization with QLoRA**
Combine LoRA with 4-bit quantization to further reduce memory usage.

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
```
The `SFTTrainer` seamlessly integrates with this by passing the `peft_config` to its constructor. For maximum performance, consider using the [Unsloth](https://github.com/unslothai/unsloth) library, which can provide up to 2x faster training speeds.

---

## Part 4: Fine-Tuning on Apple Silicon

### The Native Path: MLX

MLX is Apple's high-performance array framework for Apple Silicon.

**1. Environment Setup & Model Conversion**
```bash
pip install mlx-lm
```
Models must be converted from Hugging Face format to an MLX-compatible format.

```bash
# Convert and quantize a model to 4-bit MLX format
python -m mlx_lm.convert --hf-path google/gemma-4-E2B -q
```

**2. Data Formatting (JSONL)**
MLX requires data in JSON Lines (`.jsonl`) format. Each line must be a JSON object with a single `"text"` key.

```json
{"text": "<bos><start_of_turn>user\nWhat is the capital of France?<end_of_turn><start_of_turn>model\nParis is the capital of France.<end_of_turn>"}
{"text": "<bos><start_of_turn>user\nExplain the theory of relativity.<end_of_turn><start_of_turn>model\nAlbert Einstein's theory of relativity is...<end_of_turn>"}
```
The data must be split into `train.jsonl` and `valid.jsonl` files within a single directory.

**3. Executing LoRA Fine-Tuning**
Use the `mlx_lm.lora` command-line tool.

```bash
mlx_lm.lora \
    --model path/to/mlx_model \
    --train \
    --data path/to/data_directory \
    --iters 1000 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --num-layers 8 \
    --grad-checkpoint \
    --adapter-path ./my_gemma3n_adapter
```
**Key Memory-Saving Arguments**:
-   `--batch-size`: Reduce to avoid OOM errors.
-   `--num-layers`: Reduce the number of layers LoRA is applied to.
-   `--grad-checkpoint`: Highly recommended. Trades computation for memory.

### The Alternative: PyTorch + Metal Performance Shaders (MPS)

**1. Environment Setup & Device Configuration**
Install a nightly or stable build of PyTorch with MPS support.

```bash
# Install nightly PyTorch with MPS support
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
In your script, specify `"mps"` as the device and move the model and data to it.

```python
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Move model and data to the MPS device
model.to(device)
# In training loop:
# input_ids = batch["input_ids"].to(device)
```

**2. Numerical Stability**
Gemma models were pre-trained using `bfloat16`. The MPS backend may default to `float16`, which has a more limited numerical range and can cause NaN/infinity values in the loss.
-   **Solution**: Use `bfloat16` for mixed-precision training if available. If not, fall back to full `float32` precision.

### Head-to-Head: MLX vs. PyTorch MPS

| Feature        | MLX                                                                      | PyTorch with MPS                                                                  | Recommendation                                                                                                  |
|----------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Performance**    | Higher performance due to native optimization for unified memory.        | Lower performance, operates through a backend abstraction layer.                  | For raw speed, **MLX** is the clear winner.                                                                         |
| **Memory Model**   | Unified memory model eliminates costly CPU-GPU data transfers.           | Traditional device-mapping paradigm can introduce latency from data movement.     | MLX's architecture is inherently better suited to Apple Silicon.                                                |
| **Ease of Use**    | High. `mlx-lm` provides a simple CLI for text-based LoRA fine-tuning.    | Moderate. Requires writing a standard, more verbose PyTorch training loop.        | For quick text-only LoRA fine-tuning, **MLX** is more user-friendly.                                                |
| **Ecosystem**      | Newer and more focused.                                                  | Extremely mature and extensive (e.g., `torchaudio`, `torchvision`, `trl`, `peft`). | **PyTorch**'s ecosystem is a major advantage for complex tasks requiring specialized data processing.           |
| **Recommendation** | **Use MLX for text-only LoRA fine-tuning.**                              | **Use PyTorch/MPS for complex multimodal fine-tuning tasks.**                       | Choose based on the task: MLX for simplicity and speed, PyTorch for flexibility and complex data pipelines. |

---

## Part 5: Field Notes & Known Issues

### Troubleshooting Guide

| Issue                                     | Symptom                                                                              | Root Cause                                                                                              | Solution / Workaround                                                                                                                              |
|-------------------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| **MLX: Incorrect Output w/o Cache**       | Model generates nonsensical output when KV cache is not used.                        | Bug in `mlx-lm` that failed to initialize an empty cache.                                               | **Fixed.** Update to the latest version of `mlx-lm`.                                                                                                 |
| **MLX: `ValueError: Model type gemma3n`** | Error when trying to load or convert a Gemma 3n model.                               | Early versions of `mlx-lm` lacked support for the `gemma3n` architecture type.                          | **Fixed.** Update `mlx-lm` to the latest version (`pip install -U mlx-lm`).                                                                          |
| **MLX: Audio Tower Parameter Mismatch**   | Error "Received X parameters not in model" when loading with `mlx-vlm`.              | `mlx-vlm` was not fully equipped to handle the audio encoder architecture.                              | **Ongoing.** Audio functionality in `mlx-vlm` for Gemma 3n may be unstable. Use a framework with official support like Hugging Face Transformers. |
| **General: Numerical Instability**        | Training loss becomes `NaN` or `inf`.                                                | Floating-point precision mismatch. Gemma was trained with `bfloat16`, but hardware may default to `float16`. | 1. Use `bfloat16` if hardware supports it. 2. If not, fall back to `float32`. 3. Lower the learning rate significantly as a last resort.      |
| **General: High Initial Training Loss**   | Training loss starts abnormally high.                                                | Omission of the beginning-of-sequence (`<bos>`) token at the start of each training example.            | **Ensure every single training example is prefixed with the `<bos>` token.** This is a strict requirement.                                          |
| **General: Poor Long-Context Perf.**      | Fine-tuned model performs poorly on long contexts.                                   | Early library bug where RoPE was incorrectly calculated in `bfloat16` instead of `float32`.             | **Fixed.** Ensure libraries are up to date (e.g., `transformers >= 4.38.2`).                                                                       |

### Core Challenges in On-Device Multimodal Fine-Tuning

1.  **Prohibitive Memory Requirements**: Fine-tuning vision or audio layers requires significant VRAM (often >15GB)
2.  **Data Pipeline Complexity**: Preparing and processing interleaved multimodal data is an order of magnitude more complex than handling text alone, requiring significant custom engineering effort.
3.  **Data Heterogeneity**: Real-world user data is often incomplete or varied (e.g., image-only, text-only), posing a significant challenge for standard fine-tuning algorithms.
