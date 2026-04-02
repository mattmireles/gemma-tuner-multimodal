# **Field Guide to Whisper Model Export, GGUF Integration, and CoreML Hybrid Deployment**

April 2, 2026

## **Executive Summary**

The transition of trained speech models from PyTorch or Hugging Face Transformers into edge-ready deployment environments using whisper.cpp requires navigating a complex matrix of tensor mappings, precision limits, and hardware-specific quirks. This guide provides an operational manual for engineers responsible for the end-to-end export, validation, and deployment of Whisper architectures.

* **Export formats require strict alignment:** The whisper.cpp repository utilizes a custom binary format (GGML/GGUF). Hugging Face fine-tunes must be converted using the convert-h5-to-ggml.py script, whereas original OpenAI checkpoints mandate the convert-pt-to-ggml.py script.1  
* **Adapter merging is irreversible and precision-sensitive:** Low-Rank Adaptation (LoRA) adapters must be merged into the base weights in PyTorch using merge\_and\_unload() prior to export. GGUF conversion scripts will fail on unmerged parameter-efficient fine-tuning (PEFT) weights, and mixed-precision merges will permanently destroy model accuracy.2  
* **Safetensors metadata is highly brittle:** Attempting to convert Hugging Face .safetensors files using standard PyTorch scripts often results in a fatal KeyError: 'dims' because safetensors strip the proprietary pickled metadata required by the standard conversion scripts. Custom conversion routing or metadata patching is strictly required.5  
* **Apple Neural Engine (ANE) offloading is asymmetrical:** CoreML hybrid deployments on Apple Silicon exclusively offload the Whisper *encoder* to the ANE. The decoder remains on the CPU. This architectural split dictates that scaling dynamics for large batch sizes will inevitably bottleneck at the CPU-bound decoder.7  
* **macOS Sonoma is a hard production dependency for CoreML:** Deployments targeting macOS versions prior to Sonoma (version 14\) suffer from severe ANE-induced transcription hallucinations due to improper handling of floating-point division in the positional embeddings.10  
* **Quantization degrades speech models non-linearly:** While Q8\_0 maintains near-perfect parity with FP16, configurations below 4-bit (e.g., 2-bit or 1.58-bit ternary) experience catastrophic failure, yielding Word Error Rates (WER) exceeding 90%. Acoustic models do not share the extreme low-bit resilience of Large Language Models (LLMs).12  
* **Audio frontend implementations diverge:** The whisper.cpp repository exhibits minor deviations from the standard PyTorch audio frontend, including missing reflective padding in the Mel spectrogram generation and discrepancies in the Hann window denominator, which can cause spectral leakage during the Short-Time Fourier Transform (STFT).14  
* **Repetitive token hallucination requires runtime mitigation:** Short, noisy audio segments frequently trigger temperature fallback loops in the C++ runtime. Disabling temperature fallback (temperature\_inc \= \-1.0f) is a mandatory fix for short-utterance command workloads to prevent infinite CPU grinding.15  
* **Distil-Whisper requires manual CoreML routing:** Distilled models utilize the teacher's original encoder but feature a heavily compressed decoder. Deploying Distil-Whisper via CoreML requires manually renaming the teacher's compiled .mlmodelc encoder to perfectly match the distilled binary's filename prefix.16  
* **Performance scaling diminishes rapidly on edge CPUs:** Increasing thread counts beyond the number of physical performance cores yields diminishing returns. Excessive threading on the autoregressive decoder can actually increase latency due to memory bandwidth saturation.18  
* **Vocabulary mismatches corrupt inference silently:** If a fine-tuning process adds tokens to added\_tokens.json without resizing the underlying model embedding matrix, the conversion script will silently drop out-of-bounds tokens, shifting the entire downstream vocabulary and resulting in gibberish output.19  
* **Memory constraints dictate mobile architecture:** Deploying models larger than small.en (466 MB) to iOS devices is a severe anti-pattern that frequently results in OS-level out-of-memory (OOM) terminations. Mobile realtime pipelines must rely on tiny.en or base.en with Q8\_0 quantization.18  
* **Distil-Whisper long-form chunking is unsupported:** The Hugging Face Distil-Whisper implementation relies on a custom 15-second overlapping chunk strategy. whisper.cpp forces these models through the standard 30-second sequential pipeline, which can lead to suboptimal timestamp alignment on lengthy audio files.21  
* **Speculative decoding offers a high-performance workaround:** Combining a heavily quantized large-v3 model with a Distil-Whisper draft model via speculative decoding cuts latency by nearly 50% while preserving absolute teacher accuracy, representing a highly effective edge deployment pattern.23  
* **Exact parity validation is mandatory before shipping:** Discrepancies between PyTorch and GGUF outputs are expected, but they must be quantified. Calculating the normalized Levenshtein edit distance between the PyTorch output and the GGUF output is the only mathematically sound method to approve a model for production.24

## **Decision Framework**

Deploying Whisper models involves navigating strict, physical trade-offs between latency, accuracy, and memory footprint. The following frameworks dictate architectural choices before the export pipeline begins.

### **When GGUF Export is Worth It vs. When to Stay in PyTorch**

The decision to export a model from PyTorch to the GGUF format for use with whisper.cpp hinges entirely on the target deployment environment. Engineers should retain models in PyTorch (or compile them via ONNX/TensorRT) when deploying to centralized cloud infrastructure equipped with high-end NVIDIA GPUs (e.g., A100, H100). In these environments, batched throughput is the primary metric, and memory is abundant. Frameworks like faster-whisper (utilizing CTranslate2) or insanely-fast-whisper (utilizing Flash Attention and dynamic batching) drastically outperform CPU-based execution for massive, concurrent transcription workloads.26

Conversely, GGUF and whisper.cpp are the correct architectural targets when deployment dictates execution on the edge. This includes consumer hardware such as MacBooks, iOS devices, Raspberry Pis, and Windows laptops. GGUF is explicitly designed for memory-constrained environments lacking dedicated VRAM, utilizing dynamic RAM allocation and highly optimized CPU intrinsics (AVX2, AVX-512, NEON).10 Furthermore, GGUF minimizes dependencies; shipping a massive Python and PyTorch runtime to an end-user device is often unacceptable, whereas whisper.cpp can be compiled into a lightweight, standalone binary.10

### **When to Use whisper.cpp Alone vs. Hybrid CoreML \+ whisper.cpp**

On Apple Silicon hardware (M1-M4), engineers must choose between executing the model entirely on the CPU via Metal acceleration, or utilizing a hybrid approach that leverages the Apple Neural Engine (ANE) via CoreML.

Executing the model entirely through pure whisper.cpp (Metal/CPU) is highly recommended when the application requires maximum simplicity, as it relies on a single .bin file without the need for external compiled bundles. This approach is also mandatory when the deployment targets multi-platform edge devices simultaneously, maintaining a unified codebase.18 Additionally, because CoreML provides no acceleration for the autoregressive decoder, workloads that rely heavily on the decoder—such as translating complex audio from one language to another—see minimal benefit from ANE offloading.7

Hybrid CoreML deployments, where the encoder is offloaded to the ANE and the decoder runs via whisper.cpp on the CPU, become necessary under specific conditions. If battery life on portable Apple devices is a strict constraint, ANE offloading significantly reduces power consumption while sustaining high inference speeds.8 This hybrid architecture is particularly effective for workloads heavily skewed toward long-form transcription. Because the encoder processes the entire 30-second audio chunk at once, offloading it to the ANE yields up to a 3x speedup for the encoding phase, freeing the CPU to focus entirely on the sequential decoding phase.10

### **When Low-Bit Quantization is Worth It vs. Too Destructive**

Quantization compresses model weights, reducing memory bandwidth bottlenecks at the cost of representation fidelity. For Whisper models, the relationship between bit-depth and Word Error Rate (WER) is decidedly non-linear. While Large Language Models (LLMs) can often survive extreme quantization, acoustic models like Whisper rely on continuous, high-fidelity embeddings and break down rapidly when pushed too far.12

| Quantization Level | File Size Reduction | Speedup (Edge CPU) | WER Impact | Production Recommendation |
| :---- | :---- | :---- | :---- | :---- |
| **FP16** (Baseline) | 0% | 1.0x | None | Reserved for high-accuracy offline transcription where memory is not a limiting factor. |
| **Q8\_0** | \~75% | 1.2x \- 1.5x | \< 0.5% degradation | **Default for most edge deployments.** Near lossless parity with FP16 while drastically reducing the memory footprint.12 |
| **Q5\_K\_M** | \~80% | 1.5x \- 2.0x | \~1.0% degradation | Optimal for mobile/iOS hybrid deployments where memory pressure is high but accuracy remains critical.18 |
| **Q4\_K\_M** | \~87.5% | 2.0x \- 2.5x | 1.5% \- 3.0% degradation | Viable for real-time streaming applications or voice activity detection pipelines where speed outweighs strict accuracy.12 |
| **\< Q3\_K** | \> 90% | \> 2.5x | Catastrophic (\> 15% drop) | **Avoid entirely.** Speech models break down below 4-bit, resulting in severe attention failure and infinite hallucination loops.13 |

## **Migration Playbook**

Exporting a fine-tuned Whisper model is a multi-stage process requiring strict validation at each phase. Failure to adhere to this order of operations will result in silent quality regressions that are nearly impossible to debug at runtime.

### **Phase 1: Audit Checklist Before Export**

Prior to invoking any conversion script, the state of the PyTorch or Hugging Face repository must be rigorously validated.

1. **Adapter Status:** Confirm that all Low-Rank Adaptation (LoRA) matrices are fully merged into the base model weights. Unmerged adapters will be silently ignored by the GGML conversion scripts, or they will cause shape mismatches during tensor extraction.3  
2. **Vocabulary Integrity:** Verify that vocab.json and added\_tokens.json are present and uncorrupted in the model directory. whisper.cpp relies heavily on these files to construct the Byte Pair Encoding (BPE) dictionary during conversion.19  
3. **Config Hygiene:** Ensure config.json accurately reflects the architecture. Verify that max\_source\_positions, d\_model, encoder\_layers, and decoder\_layers match the expected dimensions of the checkpoint being exported.33  
4. **Format Verification:** Determine if the model is saved as PyTorch .pt or Hugging Face .safetensors. This dictates which conversion script will be used.

### **Phase 2: Ordered Export Steps**

The export pipeline relies on bridging Hugging Face naming conventions to GGML naming conventions via a predetermined conv\_map.33

**Step 2.1: Merge LoRA Adapters (If Applicable)**

To prevent precision degradation, load the base model in the exact precision used during training (e.g., bfloat16 or float16), apply the adapter, and execute the merge.

**Step 2.2: Execute GGML Conversion** For models originating from Hugging Face, utilize the convert-h5-to-ggml.py script. This script translates tensors such as self\_attn.k\_proj to attn.key and embeds the necessary Mel filters directly into the binary header.33

**Step 2.3: Quantize the Binary**

Process the resulting ggml-model.bin into the desired quantization format using the compiled C++ quantize tool.

### **Phase 3: Validation Checklist After Each Step**

After the conversion and quantization phases, validation must secure three vectors: transcription parity, performance baselines, and architectural stability.

1. **Generate a Benchmark Dataset:** Assemble 5 to 10 diverse audio files representing the target domain (e.g., noisy environments, clinical dictation, rapid speech).  
2. **Execute the Baseline:** Run the original PyTorch model over the dataset, strictly enforcing deterministic parameters (e.g., beam\_size=1, temperature=0.0). Record the Word Error Rate (WER) and Character Error Rate (CER).  
3. **Execute the Exported Model:** Run the quantized GGUF model over the same dataset using the exact same decoding parameters.  
4. **Compare Metrics:** Compute the delta between the PyTorch WER and the GGUF WER. A deviation greater than 1.5% indicates a catastrophic failure in the conversion map, tokenizer mapping, or a collapse at the quantization boundary.

## **Failure Modes Catalog**

This section details the most critical failure modes encountered when shipping Whisper models via whisper.cpp, organized by symptom, cause, and concrete resolution.

### **Issue 1: Safetensors Metadata Missing (KeyError: 'dims')**

**Symptom:** Running the standard conversion script on a downloaded community fine-tune crashes immediately with a KeyError: 'dims' traceback.5

**Root Cause:** The convert-pt-to-ggml.py script is legacy code designed for the original PyTorch checkpoints generated by OpenAI. It uses torch.load to search for a proprietary dims dictionary containing model hyperparameters. Modern Hugging Face models serialized in the safetensors format do not bundle this custom pickled metadata, causing the script to fail.5

**Minimal Fix:** Route the conversion through the Hugging Face-specific script, convert-h5-to-ggml.py, which parses the standard config.json rather than relying on pickled PyTorch dictionaries.1

**Robust Fix:** Ensure the repository contains the full suite of Hugging Face configuration files. Load the safetensors model using the transformers library, extract the state dictionary, and verify it against the config.json before feeding it to the GGML script.

**Code Example:**

Bash

\# Do NOT use convert-pt-to-ggml.py for Hugging Face models  
\# Instead, ensure the model directory has config.json, vocab.json, etc.  
git clone https://github.com/openai/whisper  
git clone https://github.com/ggml-org/whisper.cpp

\# Execute the robust HF conversion script  
python3./whisper.cpp/models/convert-h5-to-ggml.py \\  
   ./my-hf-safetensors-model/ \\  
   ./whisper \\  
   ./exported-ggml-dir

**Verification:** The script should execute cleanly, outputting a list of tensor mappings, reading hyperparameters directly from config.json, and concluding with a log indicating the ggml ctx size.34

### **Issue 2: Severe Precision Loss After LoRA merge\_and\_unload**

**Symptom:** The PyTorch PEFT evaluation yields an excellent WER of 8.5%. However, after merging the adapter and exporting to GGUF, the WER spikes to 35%+, and the model outputs frequent misspellings and disjointed grammar.

**Root Cause:** The adapter was merged while the base model was loaded in a mixed or lower precision state (e.g., load\_in\_8bit=True or FP32 mixed with FP16). The merge\_and\_unload() function executes destructive mathematical casts if the precisions of the base weights and the adapter weights are mismatched. This permanently destroys the fine-tuned feature alignment, creating a fractured state dictionary.2

**Minimal Fix:** Ensure both the base model and the PEFT adapter are loaded explicitly in torch.float16 before invoking the merge\_and\_unload() function.4

**Robust Fix:** Implement a strict pre-flight assertion in the merging pipeline that checks base\_model.dtype \== peft\_model.dtype. Never apply quantization *before* or *during* the merge process; only quantize the final, high-precision .bin file using the whisper.cpp quantize executable.2

**Code Example:**

Python

import torch  
from transformers import AutoModelForSpeechSeq2Seq  
from peft import PeftModel

base\_id \= "openai/whisper-large-v3"  
adapter\_id \= "./my-trained-lora"  
export\_path \= "./merged-whisper"

\# 1\. Load base in strictly enforced float16  
base\_model \= AutoModelForSpeechSeq2Seq.from\_pretrained(  
    base\_id,   
    torch\_dtype=torch.float16,   
    device\_map="cpu",  
    low\_cpu\_mem\_usage=True  
)

\# 2\. Attach adapter, enforcing the exact same dtype  
peft\_model \= PeftModel.from\_pretrained(  
    base\_model,   
    adapter\_id,   
    torch\_dtype=torch.float16  
)

assert base\_model.dtype \== peft\_model.dtype, "Precision mismatch detected\!"

\# 3\. Merge safely in high precision  
merged\_model \= peft\_model.merge\_and\_unload()  
merged\_model.save\_pretrained(export\_path)  
print(f"Safe, high-precision merge saved to {export\_path}")

**Verification:** Differencing the final merged state dictionary tensors against a known-good FP16 baseline should show values that deviate only by the mathematically expected ![][image1] formulation.

### **Issue 3: CoreML Encoder Induces Aggressive Hallucination**

**Symptom:** The exported model performs perfectly when run on the CPU via Metal. However, when compiled with WHISPER\_COREML=1 and the .mlmodelc is loaded, the model outputs infinite repetitive phrases or completely hallucinates context that does not exist in the audio.10

**Root Cause:** The Apple Neural Engine (ANE) on macOS Ventura (13.x) and earlier handles certain FP16 convolutional operations idiosyncratically, leading to numerical overflow in the Whisper encoder's cross-attention blocks. Furthermore, the deprecated \_\_floordiv\_\_ function utilized in the PyTorch trace causes incorrect rounding for negative values in the positional embeddings when translated to CoreML.10

**Minimal Fix:** Upgrade the deployment and build environment to macOS Sonoma (version 14.0+), which resolves the underlying CoreML framework bugs at the operating system level.10

**Robust Fix:** When invoking generate-coreml-model.sh, mandate that the Python environment uses strictly coremltools \>= 7.0 and torch \>= 2.1. These versions replace the deprecated floor division operations during the neural network trace, ensuring the exported topology aligns with ANE specifications.36

**Code Example:**

Bash

\# Setup a strict environment for CoreML generation  
python3.11 \-m venv coreml\_env  
source coreml\_env/bin/activate

\# Install specific versions known to avoid the floordiv bug  
pip install torch\>=2.1.0 coremltools\>=7.0 ane\_transformers openai-whisper

\# Generate the CoreML encoder artifact  
./models/generate-coreml-model.sh base.en

**Verification:** Run a zero-temperature (greedy) decode of a standard sample (e.g., jfk.wav) using the CoreML build. The text output must match the CPU-only output byte-for-byte.

### **Issue 4: Distil-Whisper CoreML Filename Mismatch**

**Symptom:** A developer successfully converts distil-large-v2 to GGUF and generates the CoreML encoder using the provided shell script. However, at runtime, whisper.cpp fails to load the ANE integration, falling back to the CPU silently.

**Root Cause:** Distil-Whisper models utilize the exact same 32-layer encoder as their OpenAI "teacher" models (e.g., large-v2). Therefore, the generate-coreml-model.sh script appropriately produces an artifact named ggml-large-v2-encoder.mlmodelc. However, the developer's GGUF file is named ggml-distil-large-v2.bin. The whisper.cpp runtime enforces a strict naming convention: it expects the encoder folder to exactly match the prefix of the .bin file.16

**Minimal Fix:** Manually rename the generated .mlmodelc directory to match the binary.16

**Robust Fix:** Modify the deployment packaging script to extract the base name of the target .bin file dynamically and enforce a symlink or rename operation automatically prior to packaging the release artifact.

**Code Example:**

Bash

\# The generated encoder takes the teacher's name  
generate\_artifact="models/ggml-large-v2-encoder.mlmodelc"

\# The compiled binary uses the distilled name  
target\_binary="models/ggml-distil-large-v2.bin"

\# Extract prefix and rename  
prefix=$(basename "$target\_binary".bin)  
mv "$generate\_artifact" "models/${prefix}\-encoder.mlmodelc"

echo "Renamed encoder to match binary prefix: ${prefix}"

**Verification:** Launch ./whisper-cli. The initialization log must explicitly state: whisper\_init\_state: loading Core ML model from 'models/ggml-distil-large-v2-encoder.mlmodelc'.7

### **Issue 5: Infinite Repetition / Gibberish on Short Audio**

**Symptom:** Passing a short, repetitive audio clip (e.g., a 3-second clip of the word "six") causes whisper.cpp to hang. CPU utilization spikes for upwards of 60 seconds, eventually outputting severe gibberish strings like "6 6 6 \[random unrelated words\]".15

**Root Cause:** The default runtime behavior of Whisper incorporates a temperature fallback mechanism. If the model is uncertain, predicts repetitive tokens, or encounters dead silence, it discards the output, increases the temperature (introducing randomness to break the loop), and retries. On extremely short segments lacking context, the model spirals into a fallback loop, generating increasingly improbable and computationally expensive gibberish.15

**Minimal Fix:** Disable the temperature increment mechanism entirely for short-command workloads by passing the appropriate flag or setting the parameter temperature\_inc \= \-1.0f.15

**Robust Fix:** In addition to disabling temperature fallback, dynamically cap the maximum generation length relative to the audio duration. Human speech has a maximum density limit. Calculate the allowed tokens and explicitly terminate backend processing before a CPU grind can occur.15

**Code Example (C++ Implementation via API):**

C++

whisper\_full\_params wparams \= whisper\_full\_default\_params(WHISPER\_SAMPLING\_GREEDY);

// Disable temperature fallback to prevent hallucination loops  
wparams.temperature\_inc \= \-1.0f;

// Cap max tokens based on audio length (assuming 16kHz audio)  
// 4 tokens per second is a safe upper bound for rapid human speech  
int audio\_seconds \= pcm\_buffer.size() / WHISPER\_SAMPLE\_RATE;  
wparams.max\_tokens \= 4 \* audio\_seconds;

// Execute inference  
whisper\_full(ctx, wparams, pcm\_buffer.data(), pcm\_buffer.size());

**Verification:** Input 3 seconds of static background noise. The model should return execution within 200ms with a blank string or a single hallucinated word, rather than looping for a minute.

### **Issue 6: Tokenizer Mismatch and "Added Tokens" Corruption**

**Symptom:** The exported model outputs completely alien text, or structural tags (e.g., \<|startoftranscript|\>) are rendered as literal text strings in the final output rather than acting as invisible control tokens.37

**Root Cause:** The mapping between the BPE vocabulary and the GGML binary relies on reading vocab.json and added\_tokens.json. If a fine-tuning script adds custom tokens but fails to resize the model's underlying embedding matrix (lm\_head), the conversion script parses out-of-bounds IDs (where token\_id \>= vocab\_size). The script logs a warning, silently drops the special tokens, and shifts the entire generation dictionary, destroying the token-to-id mapping.19

**Minimal Fix:** Ensure the added\_tokens.json file is present in the source directory before conversion, as the Python script explicitly looks for it to map overrides.

**Robust Fix:** Audit the config.json prior to export to ensure the vocab\_size integer accurately reflects the base vocabulary *plus* any tokens appended during fine-tuning. If vocab\_size is 51865, but added tokens push the necessary matrix to 51868, the embeddings must be resized in PyTorch before export.34

**Code Example:**

Python

import json

\# Pre-flight check for vocabulary mismatch  
with open("config.json", "r") as f:  
    config \= json.load(f)  
      
with open("added\_tokens.json", "r") as f:  
    added\_tokens \= json.load(f)

vocab\_size \= config.get("vocab\_size", 51865)  
max\_added\_id \= max(added\_tokens.values())

if max\_added\_id \>= vocab\_size:  
    raise ValueError(  
        f"CRITICAL: added\_tokens contain ID {max\_added\_id}, "  
        f"but config.json vocab\_size is only {vocab\_size}. "  
        "The GGUF conversion script will silently drop tokens and corrupt output\!"  
    )  
print("Vocabulary boundaries are safe for export.")

**Verification:** Run the exported model with the \--print-special flag. Control tokens should be parsed, mapped to their respective IDs, and handled by the runtime, rather than printed as raw UTF-8 strings.

## **Best Practices**

**1\. Production Artifact Packaging and Naming Hygiene** Standardize the naming conventions for all produced binaries to prevent deployment confusion. A production artifact should clearly denote the model architecture, the language focus, and the quantization level. For example: ggml-large-v3-turbo-en-q8\_0.bin. When dealing with CoreML deployments, remember that the compiled .mlmodelc artifact is a macOS bundle (a directory). It must remain completely intact. Never compress it into a .zip for runtime reading, rename its internal files, or alter the folder structure, as the ANE runtime requires the specific signature and shape mapping organized by the macOS CoreML compiler.20

**2\. Thread Count Optimization** Do not blindly set thread counts to the total number of logical cores on the target device. Whisper's memory bandwidth bounds the autoregressive decoder. Over-threading causes severe cache trashing. As a strict rule of thumb, set maxThreads equal to the number of physical *performance* cores on the device. Frequently, profiling with the bench() tool will reveal that 4 threads drastically outperform 8 threads on mobile and desktop edge processors.18

**3\. Explicit Language and Task Declaration** Avoid relying on the model's auto-language detection for mission-critical deployments. Auto-detection consumes the entire first segment of processing time and is highly susceptible to misclassification due to background noise. Hardcode the target language flag (e.g., \--language en or language="en") to force the initial decoder prompt, significantly accelerating time-to-first-token.18

## **Worst Practices / Anti-patterns**

**1\. Deploying Large Models on Mobile Edge Devices** Shipping medium (1.5 GB) or large-v3 (3 GB) models to iOS or Android targets is a severe anti-pattern.18 The memory pressure induced by loading these models frequently triggers the operating system's jetsam process, resulting in instantaneous out-of-memory (OOM) application crashes. Mobile realtime pipelines must rely exclusively on tiny.en or base.en architectures utilizing Q8\_0 quantization.18

**2\. Mixing Precisions During Parity Evaluation**

A common anti-pattern is using FP32 for the PyTorch benchmark and Q4\_K\_M for the edge benchmark, and then improperly attributing all WER degradation to the whisper.cpp framework. The audio processing frontends differ fundamentally (PyTorch uses PyTorch audio ops, whisper.cpp uses a custom C++ FFT implementation). Engineers must isolate variables: benchmark PyTorch FP16 against whisper.cpp FP16 to identify frontend bugs, and then benchmark whisper.cpp FP16 against whisper.cpp Q4\_K\_M to isolate the precise quantization degradation.

**3\. Quantizing the First Decoder Call** The Whisper decoder possesses a distinct forward pass signature for the first token generation (which has an empty key-value cache) versus all subsequent tokens (which utilize a populated key-value cache). Applying aggressive, experimental sub-4-bit quantization to the initial context-embedding pass disproportionately degrades context awareness. Tools like OpenVINO explicitly split these into two distinct models to avoid this; while whisper.cpp handles it internally, users should avoid custom scripts that aggressively quantize the projection layers.39

## **Weird Patterns That Work Surprisingly Well in Practice**

**1\. Speculative Decoding with Distil-Whisper Draft Models** While running Distil-Whisper standalone in whisper.cpp faces limitations due to chunking strategy mismatches 1, combining it with large-v3 via speculative decoding yields exceptional results. Because the encoder architecture is frozen and identical between the two models, the massive large-v3 encoder is executed only once. The tiny Distil-Whisper decoder then acts as a rapid draft model, proposing strings of tokens, which the robust large-v3 decoder verifies in batches. This architectural hack cuts latency by nearly 50% on edge devices while preserving the absolute accuracy of the massive large-v3 teacher model.23

**2\. Faking Stereo Diarization via Downmixing** whisper.cpp does not natively support complex multi-channel audio for distinct speaker diarization. However, a highly effective, albeit brute-force, pattern involves extracting the left and right channels using ffmpeg into separate mono files. Run instances of the model independently on each channel, using specific \--prompt inputs derived from the opposing channel's output to maintain contextual awareness, and manually merge the timelines. This acts as a highly effective diarization pipeline without requiring external, heavy embedding models.40

## **Limitations and Unsolved Problems**

**1\. Audio Frontend Spectral Leakage** The PyTorch reference implementation utilizes an exact padding of 480,000 samples and applies reflective padding to handle edge effects. The whisper.cpp implementation is less robust, occasionally skipping reflective padding and exhibiting minor frame count calculation errors.14 Furthermore, the C++ standard library trigonometric functions default to FP64, while PyTorch uses FP32. This mismatch causes subtle discrepancies in the Mel spectrogram generation, particularly at lower angles, contributing to a baseline WER delta between the frameworks even when both are running in FP16.14

**2\. Distil-Whisper Chunking Support Gaps** The Hugging Face Distil-Whisper implementation achieves its high performance on long audio by relying on a custom 15-second overlapping chunk strategy. This handles long-form audio accurately without the timestamp shift errors inherent in the model's compression. whisper.cpp currently lacks native support for this specific chunking algorithm, forcing the distilled models through the standard sequential 30-second pipeline. This can lead to suboptimal timestamp alignment and increased hallucination on files exceeding 10 minutes.21

**3\. Extreme Low-Bit Quantization Collapse** Research attempts to compress Whisper using 2-bit or 1.58-bit (ternary) weight formats have completely stalled. While Large Language Models (LLMs) demonstrate remarkable resilience at these extreme bit-depths, acoustic models rely on continuous, high-fidelity embeddings to map non-discrete audio waves to discrete tokens. Moving below 4-bit quantization in whisper.cpp causes an immediate, catastrophic breakdown in the attention mapping mechanism. This results in severe hallucination loops and WERs exceeding 90%, rendering sub-4-bit speech models an unsolved problem and a current dead end.13

## **Copy-Paste Reference Snippets**

### **Automated Parity Validation (WER Calculation)**

To validate a GGUF export against the PyTorch original, exact WER computation is necessary. Use this script to calculate the normalized Levenshtein edit distance. Normalization is critical to prevent casing and punctuation from artificially inflating error rates.24

Python

import evaluate  
from transformers.models.whisper.english\_normalizer import BasicTextNormalizer

def calculate\_strict\_parity(ref\_text: str, hyp\_text: str):  
    """  
    Computes normalized WER between PyTorch and whisper.cpp outputs.  
    """  
    wer\_metric \= evaluate.load("wer")  
    normalizer \= BasicTextNormalizer()  
      
    \# Normalize texts to strip punctuation, standardize casing, and expand contractions  
    normalized\_ref \= normalizer(ref\_text).strip()  
    normalized\_hyp \= normalizer(hyp\_text).strip()  
      
    if not normalized\_ref:  
        print("Reference is empty, skipping calculation.")  
        return 0.0  
          
    wer \= 100 \* wer\_metric.compute(  
        references=\[normalized\_ref\],   
        predictions=\[normalized\_hyp\]  
    )  
      
    print(f"Strict Normalized WER: {wer:.2f}%")  
    return wer

\# Example Execution  
\# reference\_pytorch \= "I wish to change my address, please."  
\# prediction\_gguf \= "i wish to change my adress please"  
\# calculate\_strict\_parity(reference\_pytorch, prediction\_gguf)

### **CoreML Deployment Stub & Validation Harness**

Execute this sequence to generate the Apple Neural Engine artifact, compile the binary, and execute a validation run.10

Bash

\#\!/bin/bash  
\# Exit on any failure  
set \-e

MODEL\_NAME="base.en"

\# 1\. Setup isolated environment  
python3.11 \-m venv coreml\_env  
source coreml\_env/bin/activate  
pip install ane\_transformers openai-whisper coremltools torch\>=2.1.0

\# 2\. Generate the CoreML artifact  
echo "Generating CoreML encoder for ${MODEL\_NAME}..."  
./models/generate-coreml-model.sh ${MODEL\_NAME}

\# 3\. Compile whisper.cpp with CoreML flag enabled  
echo "Compiling whisper.cpp with CoreML support..."  
cmake \-B build \-DWHISPER\_COREML=1  
cmake \--build build \-j \--config Release

\# 4\. Execute with validation  
echo "Validating hybrid deployment..."  
./build/bin/whisper-cli \-m models/ggml-${MODEL\_NAME}.bin \-f samples/jfk.wav \> deployment\_test.log 2\>&1

\# 5\. Assert CoreML initialization  
if grep \-q "whisper\_init\_state: Core ML model loaded" deployment\_test.log; then  
    echo "SUCCESS: CoreML Hybrid Deployment Verified."  
else  
    echo "FATAL: CoreML integration failed. Falling back to CPU."  
    exit 1  
fi

## **Final "Red Flags" Checklist Before Shipping**

Do not merge the pull request or ship the binary artifact until all the following assertions pass:

* \[ \] **Precision Audit:** The LoRA adapter was merged via merge\_and\_unload() while explicitly loaded in torch.float16. No mixed precision existed in the pipeline.  
* \[ \] **Metadata Integrity:** The convert-h5-to-ggml.py script was used for Hugging Face models, successfully preventing the dims safetensors crash.  
* \[ \] **Quantization Floor:** The target deployment binary is quantized no lower than Q4\_K\_M.  
* \[ \] **Tokenizer Parity:** A dry run with \--print-special demonstrates that control tokens (\<|startoftranscript|\>) are accurately parsed by the GGML runtime, and are not printed as raw text strings in the output.  
* \[ \] **CoreML Sibling Validation:** For Apple Silicon hybrid builds, the \[model\_name\]-encoder.mlmodelc directory is physically located in the exact same directory as the .bin file, and the prefix matches perfectly.  
* \[ \] **OS Versioning:** The deployment target OS constraints explicitly require macOS 14.0+ (Sonoma) to mitigate ANE hallucination matrices.  
* \[ \] **Command Truncation:** For short-audio (command and control) applications, temperature\_inc is set to \-1.0f to prevent infinite CPU fallback grinding.  
* \[ \] **WER Delta:** The normalized WER of the final whisper.cpp output on a 5-minute benchmark audio file deviates from the PyTorch inference by no more than 1.5%.  
* \[ \] **Memory Profiling:** The model fits within the memory constraints of the target edge device (e.g., using tiny or base for iOS deployment) without triggering jetsam events.  
* \[ \] **Thread Constraint:** Runtime thread configurations are hardcoded or dynamically limited to the physical performance core count, avoiding decoder memory bandwidth saturation.

#### **Works cited**

1. 5.75 kB \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/spaces/Xenobd/whisper.cpp/resolve/b4969ff304b29adf4ae564feb0550e0912004bd9/models/README.md?download=true](https://huggingface.co/spaces/Xenobd/whisper.cpp/resolve/b4969ff304b29adf4ae564feb0550e0912004bd9/models/README.md?download=true)  
2. Merging LoRA Adapters with Base Model \- Kaggle, accessed April 2, 2026, [https://www.kaggle.com/code/ebinbt007/merging-lora-adapters-with-base-model](https://www.kaggle.com/code/ebinbt007/merging-lora-adapters-with-base-model)  
3. Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue \#7062 · ggml-org/llama.cpp \- GitHub, accessed April 2, 2026, [https://github.com/ggml-org/llama.cpp/issues/7062](https://github.com/ggml-org/llama.cpp/issues/7062)  
4. Help with merging LoRA weights back into base model :-) \- Beginners, accessed April 2, 2026, [https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968](https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968)  
5. KeyError: 'dims' when running convert-pt-to-ggml.py on safetensors/pytorch models \#3315, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp/issues/3315](https://github.com/ggml-org/whisper.cpp/issues/3315)  
6. whisper.cpp/models/convert-pt-to-ggml.py", line 210 KeyError: 'dims' · Issue \#2730 · ggml-org/whisper.cpp \- GitHub, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp/issues/2730](https://github.com/ggml-org/whisper.cpp/issues/2730)  
7. whisper.cpp/examples/whisper.swiftui/README.md at master · ggml-org/whisper.cpp · GitHub, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp/blob/master/examples/whisper.swiftui/README.md](https://github.com/ggml-org/whisper.cpp/blob/master/examples/whisper.swiftui/README.md)  
8. altalt-org/Lightning-SimulWhisper: An MLX/CoreML implementation of SimulStreaming. \~15x increase in performance \- GitHub, accessed April 2, 2026, [https://github.com/altalt-org/Lightning-SimulWhisper](https://github.com/altalt-org/Lightning-SimulWhisper)  
9. faster-whisper vs whisper.cpp with CoreML · SYSTRAN faster-whisper · Discussion \#368 \- GitHub, accessed April 2, 2026, [https://github.com/SYSTRAN/faster-whisper/discussions/368](https://github.com/SYSTRAN/faster-whisper/discussions/368)  
10. ggml-org/whisper.cpp: Port of OpenAI's Whisper model in C/C++ \- GitHub, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)  
11. v1.5.5 · ggml-org whisper.cpp · Discussion \#2064 \- GitHub, accessed April 2, 2026, [https://github.com/ggerganov/whisper.cpp/discussions/2064](https://github.com/ggerganov/whisper.cpp/discussions/2064)  
12. Quantization for OpenAI's Whisper Models: A Comparative Analysis \- alphaXiv, accessed April 2, 2026, [https://www.alphaxiv.org/overview/2503.09905](https://www.alphaxiv.org/overview/2503.09905)  
13. Faster and Smaller Whisper: A Deep Dive into Quantization and Torch Compilation \- Dropbox, accessed April 2, 2026, [https://dropbox.github.io/whisper-static-cache-blog/](https://dropbox.github.io/whisper-static-cache-blog/)  
14. Significantly improve whisper.cpp inference quality \#1148 \- SemanticDiff, accessed April 2, 2026, [https://app.semanticdiff.com/gh/ggml-org/whisper.cpp/pull/1148/overview](https://app.semanticdiff.com/gh/ggml-org/whisper.cpp/pull/1148/overview)  
15. Short sequences of numbers can cause extremely long repetitive inference · Issue \#412 · ggml-org/whisper.cpp \- GitHub, accessed April 2, 2026, [https://github.com/ggerganov/whisper.cpp/issues/412](https://github.com/ggerganov/whisper.cpp/issues/412)  
16. Converted distil-whisper model with generate-coreml-model.sh works incorrectly · Issue \#1558 · ggml-org/whisper.cpp \- GitHub, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp/issues/1558](https://github.com/ggml-org/whisper.cpp/issues/1558)  
17. Audio Transcription Effortlessly with Distill Whisper AI | DigitalOcean, accessed April 2, 2026, [https://www.digitalocean.com/community/tutorials/distill-whisper](https://www.digitalocean.com/community/tutorials/distill-whisper)  
18. Performance Optimization \- whisper.rn \- Mintlify, accessed April 2, 2026, [https://www.mintlify.com/mybigday/whisper.rn/advanced/optimization](https://www.mintlify.com/mybigday/whisper.rn/advanced/optimization)  
19. accessed April 2, 2026, [https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert\_hf\_to\_gguf.py](https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert_hf_to_gguf.py)  
20. aarush67/whisper-coreml-models \- Apple-Silicon \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/aarush67/whisper-coreml-models](https://huggingface.co/aarush67/whisper-coreml-models)  
21. Distil-Whisper Models: 6 times speed up and 49% smaller size? \#1414 \- GitHub, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp/discussions/1414](https://github.com/ggml-org/whisper.cpp/discussions/1414)  
22. distil-whisper/distil-medium.en \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/distil-whisper/distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en)  
23. distil-whisper/distil-large-v3.5 \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5)  
24. How to Build Medical Transcription Software in Python \- Picovoice, accessed April 2, 2026, [https://picovoice.ai/blog/build-medical-transcription-software/](https://picovoice.ai/blog/build-medical-transcription-software/)  
25. SeamlessM4T vs. Whisper — A Speech-to-Text Benchmark | by Jeremy K | AI monks.io, accessed April 2, 2026, [https://medium.com/aimonks/seamlessm4t-vs-whisper-a-speech-to-text-benchmark-6dc873154825](https://medium.com/aimonks/seamlessm4t-vs-whisper-a-speech-to-text-benchmark-6dc873154825)  
26. Showdown of Whisper Variants \- Quids, accessed April 2, 2026, [https://quids.tech/blog/showdown-of-whisper-variants/](https://quids.tech/blog/showdown-of-whisper-variants/)  
27. Comparison between Hugging Face Transformers and llama.cpp | by Jsong \- Medium, accessed April 2, 2026, [https://medium.com/@jsong\_49820/comparison-between-hugging-face-transformers-and-llama-cpp-5ee8affe1f27](https://medium.com/@jsong_49820/comparison-between-hugging-face-transformers-and-llama-cpp-5ee8affe1f27)  
28. Choosing between Whisper variants: faster-whisper, insanely-fast-whisper, WhisperX \- Modal, accessed April 2, 2026, [https://modal.com/blog/choosing-whisper-variants](https://modal.com/blog/choosing-whisper-variants)  
29. Efficient Deep Learning in Mobile and Embedded Computing Environments \- PhD Dissertation \- Ioannis Panopoulos \- Artemis \- ΕΘΝΙΚΟ ΜΕΤΣΟΒΙΟ ΠΟΛΥΤΕΧΝΕΙΟ, accessed April 2, 2026, [http://artemis.cslab.ece.ntua.gr:8080/jspui/bitstream/123456789/19600/1/PhD\_Dissertation\_Ioannis\_Panopoulos.pdf](http://artemis.cslab.ece.ntua.gr:8080/jspui/bitstream/123456789/19600/1/PhD_Dissertation_Ioannis_Panopoulos.pdf)  
30. A Starter Guide to Whisper CPP \- Medium, accessed April 2, 2026, [https://medium.com/@bhuwanmishra\_59371/a-starter-guide-to-whisper-cpp-f238817fd876](https://medium.com/@bhuwanmishra_59371/a-starter-guide-to-whisper-cpp-f238817fd876)  
31. Demystifying LLM Quantization Suffixes: What Q4\_K\_M, Q8\_0, and Q6\_K Really Mean | by Paul Ilvez | Medium, accessed April 2, 2026, [https://medium.com/@paul.ilvez/demystifying-llm-quantization-suffixes-what-q4-k-m-q8-0-and-q6-k-really-mean-0ec2770f17d3](https://medium.com/@paul.ilvez/demystifying-llm-quantization-suffixes-what-q4-k-m-q8-0-and-q6-k-really-mean-0ec2770f17d3)  
32. README.md \- ruvnet/RuVector \- GitHub, accessed April 2, 2026, [https://github.com/ruvnet/ruvector/blob/main/README.md](https://github.com/ruvnet/ruvector/blob/main/README.md)  
33. whisper.cpp/models/convert-h5-to-ggml.py at master · ggml-org ..., accessed April 2, 2026, [https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-h5-to-ggml.py](https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-h5-to-ggml.py)  
34. Convert hugginface model to ggml? · Issue \#157 · ggml-org/whisper.cpp \- GitHub, accessed April 2, 2026, [https://github.com/ggerganov/whisper.cpp/issues/157](https://github.com/ggerganov/whisper.cpp/issues/157)  
35. Fast Whisper-Large-v2 Fine-Tuning with LoRA \- Kaggle, accessed April 2, 2026, [https://www.kaggle.com/code/imtiazprio/fast-whisper-large-v2-fine-tuning-with-lora](https://www.kaggle.com/code/imtiazprio/fast-whisper-large-v2-fine-tuning-with-lora)  
36. Getting bizarre output on Metal \#1862 \- ggml-org/whisper.cpp \- GitHub, accessed April 2, 2026, [https://github.com/ggerganov/whisper.cpp/issues/1862](https://github.com/ggerganov/whisper.cpp/issues/1862)  
37. Prompt tokenization does not match openai/whisper · Issue \#1098 · ggml-org/whisper.cpp, accessed April 2, 2026, [https://github.com/ggml-org/whisper.cpp/issues/1098](https://github.com/ggml-org/whisper.cpp/issues/1098)  
38. ai: OpenWebUI Missing Mac Manual with Ollama and ComfyUI notes v2.1 2025-05-04, accessed April 2, 2026, [https://tongfamily.com/2025/05/02/ai-openwebui-missing-mac-manual-with-ollama-and-comfyui-notes-v2-0-2025-04-30/](https://tongfamily.com/2025/05/02/ai-openwebui-missing-mac-manual-with-ollama-and-comfyui-notes-v2-0-2025-04-30/)  
39. Optimizing Whisper and Distil-Whisper for Speech Recognition with OpenVINO and NNCF, accessed April 2, 2026, [https://blog.openvino.ai/blog-posts/optimizing-whisper-and-distil-whisper-for-speech-recognition-with-openvino-and-nncf](https://blog.openvino.ai/blog-posts/optimizing-whisper-and-distil-whisper-for-speech-recognition-with-openvino-and-nncf)  
40. pszemraj/LocalLLaMA-posts · Datasets at Hugging Face, accessed April 2, 2026, [https://huggingface.co/datasets/pszemraj/LocalLLaMA-posts/viewer](https://huggingface.co/datasets/pszemraj/LocalLLaMA-posts/viewer)  
41. \[Distil-Whisper\] Add support for Distil-Whisper · Issue \#1423 · ggml-org/whisper.cpp \- GitHub, accessed April 2, 2026, [https://github.com/ggerganov/whisper.cpp/issues/1423](https://github.com/ggerganov/whisper.cpp/issues/1423)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAAAYCAYAAAAYuwRKAAAFY0lEQVR4Xu2aachtUxjHn2Oe6+qW8YPMig+EjPlAkjmEECLTDZHhkw+ESG5RhiRfTIWSEkmmmzG5UsZML645Eq7pGtfvrLXOfvZz1tpnn/2ed5/jfc+vnnve/X/2sNaznzXuK9IKHSv8/5mHVZoyZcoC5ujSUes9XCsP3NEK84dW4jc07zlbbMUKNnP2T0mZzHpZXnO2ixUjqzv70dm/yr4LPjLyF+N7P/jgc+M7SfkWKtc5u8KKAyCpiN84WeHsVyne5Upn3zv7SWn39M4ucHpnfStq4k1TxBunONfZJVZcoKwm+TjlOEuq4zsShugAKcdjVnTcJt53s9EPFt/5ZB/yhuQrV1XxP/VB5t7j4truv+0V6knpD/wgYmxz8W2TY8SXYzfrcNAr4fvLOsTrW1gx8pD4ExjvNadKvqu+xdnGVpwgrrfCHEOM1rBiBU8728rZ15KOb45LB7SVTaxQk9clX46l4n1XGh2Ya71gxQhzAy7c3+i/O3sx+ELQetV6J/4xbjKBvtEKDVjL2U3ObnW2edlVeurWkn8pKdZ19kn4+xkZ7tr9nL1lxcD2zr6wYk1yPScLEfQfrCNwoqSv63KmeOfpSrvP2YbO7g2+HZQvV7FJglY2Gz4UX881wzExuLpwl3rEG6QiuAmY00ZIWhvfQRwo5YUUcP1XRjNkmqCHMpDk2zjbVvyKj54I/ezeWelbZOt+gHjnNUpbHn6vCr5Dw/Ei8cNgH+lnjo1ZJFaHFz9jRN0yL3C2pfItU75BkBS6N71I/LXHKa0OBzn7IPxNUjGkNuUI8WU4Q3z5MO5PQyonVpps3Zl84XwgHH+qfKcF34XheJXyRcjyd0UNjy0n2e4Jo6e1WrQq4nyCVZ6GrRn0nf1vqYbES8esCvsSDgsaDbif6kDG5PrWOoakan51nnjfkoqi4F/HihGcy93Fm0q5m2dMx8eKh4ocpXwaCkerHgdHJuyRhBatCpImF2T0+8UvszUzUsyZqrhL/N6Qtp/F3/dhdV5dWMHRU/H8WmSSg+fn6ryReN9v1qHAz7QpCU4q+rfRWSniezTh0+QK1g79EWs6FFIPtl9S4GND2fKE2N3zfugBU9cC92VONwwkVZxT7SnDX6/h+Y9bMbBEvJ9GkaPy3cespVeyRF/VUhY/nzPelvJeyB3ihyVWl88rHS4LOgmtYZf3VanXC+SYTWLdacUAvvOtKL437wW3P8e78EUjLgQsMb51Ib5fGo3kshP6Opwg/tl7W0egTtkq/ThXZYKCb8aKCuYJuqLxQQ+Kn5wCE0K/S1vwWfjl81DcQ6NVxw+cdK8MxU1omli03D+MdogUQxYtl/nWPsrPMj8XXHqquGWTgmSr8/IizPFy87k9xDfuYVgh6WcfLl5nlFrb+DTHSvr6HjhZ8aWovNDxirOT1XE8X193t/R/R4sB3Ssch4lxdxd7pbNzgt6EpokFt0tRNspBuWDXoNkGAuh2wn+8+IbCdzeuKX2pED98Rj+/+Ad9IluaafyR7ayQgYbCTnp3A7xT1JdjyvGUlBtPjpecvWzFwQyoRUAnEAnGypFfrfM3G47xnuuJb60EMp53ufgWNApmk1hNYGXmPyONk3rvS4Y4MUHpWt5dboifNbS4CK2AnWWS6OOgcRyT52LxQ5xNOthJyju8i8V/VmqBhoEuLtN1DDS855wx8vLs6+6Z2n4aEZ1uxjJksErRy85nxXeT/C+IN6XYdIWPnD0nfptCw040e2JM6LN7TiMP0Whg55p9nwll5FFj/jXM99F2GHk15+CODfjG2QZWnIfwEZ0555QWOcUKddFNYyKaSZ46E/spUxY6E96M5z014v8fZoxZe5kwb2gAAAAASUVORK5CYII=>