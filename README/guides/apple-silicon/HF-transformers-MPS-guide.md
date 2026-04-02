# **Hugging Face Transformers on Apple Silicon: S-Tier Field Guide for PyTorch MPS**

April 2, 2026

## **1\. Executive Summary**

This report establishes the definitive operational protocol for engineering teams actively shipping, debugging, and maintaining Hugging Face Transformers workflows—specifically seq2seq architectures, Whisper pipelines, autoregressive generation, and LoRA fine-tuning—on Apple Silicon. As of 2026, the PyTorch Metal Performance Shaders (MPS) backend has cemented its status as a highly capable but uniquely volatile "inference and local prototyping island".1 Achieving production-grade reliability requires navigating strict hardware limitations, framework fragmentation between PyTorch and Apple's native libraries, and silent regressions across the torch and transformers ecosystems.

The following brutal realities govern Apple Silicon machine learning operations:

* **The Swap Death Trap Over OOM:** Unlike NVIDIA CUDA architectures, which gracefully halt execution and throw an Out-of-Memory (OOM) exception when VRAM is exhausted, macOS attempts to salvage operations by pushing excessive tensor allocations to the solid-state drive (SSD) swap file.1 This architectural difference causes inference pipelines to silently degrade from 20 tokens per second to 0.01 tokens per second, effectively freezing the Python process indefinitely without raising an actionable error trace.1  
* **Version Instability is the Baseline Constraint:** The transition to transformers v5.3.0 and torch 2.6.0/2.7.0 introduces severe, pipeline-breaking changes. Legacy commands like the transformers chat CLI are deprecated, the underlying HTTP backend has migrated from requests to httpx, and unified decoding APIs fundamentally alter how batch generation outputs are processed.2  
* **Silent NaNs from Memory Layout Divergence:** Optimizers and low-level mathematical operations (such as addcmul\_) operating on non-contiguous tensors routinely fail silently on the MPS backend. This results in zero-gradient updates or silent NaN propagation during LoRA training without triggering framework-level exceptions.4  
* **Hardware Buffer Limits Break SDPA:** The hardware buffer allocations governed by the Metal Shading Language strictly prevent the standard Scaled Dot-Product Attention (SDPA) mechanism from processing sequences beyond approximately 12,000 tokens.5 Exceeding this boundary triggers hard crashes, demanding manual sequence chunking or memory-efficient attention fallbacks.5  
* **Seq2SeqTrainer Evaluation Freezes are Systemic:** Utilizing predict\_with\_generate=True within the Hugging Face Seq2SeqTrainer reliably freezes on MPS during the evaluation loop. This occurs because the trainer accumulates hidden states in the unified memory pool until swapping occurs, unless the eval\_accumulation\_steps parameter is strictly managed.6  
* **macOS Dataloader Multiprocessing is Fatally Broken:** The default spawn and fork behaviors in the Darwin OS clash fatally with MPS tensor allocations and Metal context initialization. The dataloader\_num\_workers parameter in TrainingArguments must strictly remain 0 to prevent immediate process termination.8  
* **Distributed Training is Non-Existent:** The MPS backend currently offers zero support for nccl or gloo communication protocols for GPU-accelerated distributed execution. Multi-GPU accelerate configurations or Fully Sharded Data Parallel (FSDP) wrappers will fail immediately on Apple Silicon.7  
* **Sparse Tensor Omissions Require Fallbacks:** Key mathematical operations required by Whisper architectures, such as aten::\_sparse\_coo\_tensor\_with\_dims\_and\_tensors, are entirely missing from the MPS backend. This necessitates explicit environmental configuration to force CPU fallbacks.11  
* **Generation Configs Suffer from Legacy Collisions:** Legacy forced\_decoder\_ids attributes in Whisper generation configs are deprecated and frequently conflict with the newer language and task attributes, resulting in infinite generation loops, repetition penalties, or gibberish text outputs.12  
* **Low-Precision Formats are Emulated Hazards:** Apple Silicon lacks native hardware logic for FP8 and FP4 precision formats. These data types are emulated by upcasting to BF16 or FP32 under the hood. Consequently, a model that converges numerically on a Mac may immediately explode on an NVIDIA Blackwell or Hopper cluster due to the hidden loss of dynamic range.1  
* **Garbage Collection is a Strict Mandate:** Relying solely on torch.mps.empty\_cache() is insufficient for addressing memory leaks, particularly with LSTMs and heavy autoregressive generation loops. Python's gc.collect() must be explicitly invoked prior to clearing the MPS cache to force the Metal driver to release unified memory.14  
* **The Fallback Variable is Mandatory:** Setting the environment variable PYTORCH\_ENABLE\_MPS\_FALLBACK=1 is not optional for production seq2seq workloads. Without it, the invocation of unsupported operations triggers immediate segmentation faults or process deaths instead of safely routing the operation to the CPU.16  
* **Hardware Watermarks Dictate Survival:** Maximum unified memory usage must be strictly controlled via the PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO environment variable. This prevents the OS from killing the Python process during peak generation steps by enforcing a hard ceiling on Metal allocations.18  
* **Asynchronous Execution Traps Abound:** MPS operations are fundamentally asynchronous. Any attempt to time execution blocks, profile memory, or dynamically assert tensor shapes inside generation loops requires explicit torch.mps.synchronize() calls to ensure mathematical correctness and accurate tracing.20  
* **Compilation is Prototype-Only:** Attempting to accelerate transformers pipelines on MPS using torch.compile routinely results in SyntaxError shader generation failures. The Triton-Inductor pipeline is severely broken for MSL, particularly for complex encoder-decoder models like T5 and Whisper.1

## **2\. Decision Framework**

The volatility of the MPS compiler stack requires engineering teams to know exactly when to leverage high-level Hugging Face abstractions and when the framework is actively fighting the hardware. The following framework dictates architectural choices for seq2seq and Whisper pipelines on Apple M-series chips.

### **When plain transformers \+ manual loop is best**

A manual PyTorch training and evaluation loop built directly over base transformers classes (e.g., AutoModelForSpeechSeq2Seq, AutoProcessor) without higher-level orchestration is required under the following conditions:

1. **Processing Long-Form Audio:** When transcribing audio files longer than 30 seconds, the sequence length exceeds the memory buffers of the SDPA implementation on MPS.5 A manual loop allows for custom chunking logic, passing strict 30-second audio windows to the Whisper model iteratively while explicitly clearing memory between segments.23 The Trainer abstraction cannot handle this seamlessly without triggering the OS swap death.  
2. **Debugging Silent NaNs or Zero-Gradient Updates:** The MPS backend is notorious for silent failures when optimizing non-contiguous tensors.4 Manual loops allow engineers to insert explicit tensor.is\_contiguous() assertions and custom gradient magnitude checks. Relying on torch.autograd.detect\_anomaly() is unfeasible, as it slows MPS execution by over 100x and often crashes large models outright.1  
3. **Executing Generative Evaluation:** If the workflow relies on predict\_with\_generate=True, the Seq2SeqTrainer is highly prone to freezing as it accumulates hidden states in memory.24 A manual loop permits explicit memory purging via gc.collect() and torch.mps.empty\_cache() at the end of each generation batch, keeping the unified memory footprint stable.15

### **When Seq2SeqTrainer is acceptable**

The Seq2SeqTrainer should only be utilized on Apple Silicon if the workflow adheres to strict, constrained boundaries:

1. **Short-Form Tasks and Adapters:** The objective is standard sequence classification, short-form ASR fine-tuning (e.g., audio segments strictly under 10 seconds), or LoRA adapter training where the maximum sequence lengths are rigidly capped.  
2. **Evaluation Accumulation is Managed:** Evaluation during training does not rely on heavy autoregressive generation, or the parameter eval\_accumulation\_steps=1 is explicitly defined in Seq2SeqTrainingArguments. This forces the trainer to offload predictions to the CPU after every step, preventing the exhaustion of the Metal memory pool.7  
3. **Single-Threaded Dataloading:** The engineering team accepts that dataloader\_num\_workers must be set to 0\. This bounds the pipeline's overall throughput to the CPU's single-threaded data preparation speed, but it is entirely unavoidable to prevent multiprocessing crashes on macOS.8

### **When accelerate is useful**

The accelerate library is highly recommended for MPS workflows, provided its usage is restricted strictly to single-device paradigms. It provides significant value when:

1. **Standardizing Deployment Code:** The primary goal is to write a single training script that runs seamlessly on a local Apple Silicon machine for prototyping, and can then be deployed to single-node NVIDIA cloud instances without refactoring the boilerplate.7  
2. **Dynamic Device Placement:** accelerate handles the automated routing of tensors to the mps device and naturally respects the absence of distributed backends (gloo/nccl) on macOS, falling back to single-device execution gracefully.10  
3. **Gradient Accumulation:** Utilizing accelerator.accumulate(model) simplifies the process of bypassing the strict memory limits of the unified memory architecture by enabling larger effective batch sizes without triggering OOM crashes.7

### **When to avoid higher-level abstractions**

High-level abstractions like the pipeline() object or the torch.compile engine must be actively avoided on Apple Silicon when:

1. **Precise Whisper Timestamp Processing is Required:** The pipeline object abstracts away the slicing of duplicate timestamp tokens in the middle of long-form generation. This is a known breaking change and edge case in newer transformers iterations that requires direct access to the model.generate outputs to handle correctly.12  
2. **Compilation is Attempted:** The torch.compile mechanism relies on the Triton-Inductor pipeline. Apple has not fully embraced the Triton stack, relying instead on its proprietary Metal Shading Language.1 Complex fusions, particularly for models like T5 and Whisper, either fail to generate shaders entirely (throwing SyntaxError) or execute significantly slower than eager mode due to continuous CPU fallbacks.1

## **3\. Migration Playbook**

Transitioning an existing seq2seq or Whisper pipeline to run reliably on MPS in 2026 requires strict version adherence, comprehensive environmental auditing, and structured validation.

### **Recommended Version-Pinning Strategy**

The following matrix represents the most stable configuration for Apple Silicon workflows as of early 2026\.

| Package | Pinned Version | Engineering Rationale |
| :---- | :---- | :---- |
| torch / torchaudio | \>=2.6.0 | Resolves critical unified memory leaks in LSTMs and getStridedMPSNDArray allocations. Fixes enable\_gqa crashes during attention calculation.28 |
| transformers | \==5.3.0 | Contains the unified decoding API (where decode behavior perfectly matches batch\_decode) and cleanly deprecates bloated legacy chat wrappers.2 |
| accelerate | \>=1.1.1 | Ensures robust integration with huggingface\_hub \>= 1.0.0 and guarantees proper device placement heuristics for the mps backend.3 |
| datasets | \==3.1.0 | Guarantees stable compatibility with transformers v5.x data collators and avoids iterable dataset streaming bugs during generation loops.30 |
| evaluate | \==0.4.x | Provides a stable metrics API that is fully compatible with the recent httpx backend HTTP transition implemented in the Transformers ecosystem.31 |

### **Audit Checklist for an Existing Hugging Face Codepath**

Before executing code on an M-series chip, the repository must be audited against the following checklist:

* **Purge Hardcoded CUDA:** Remove all instances of .cuda() or device="cuda". Replace with dynamic assignment: device \= torch.device("mps" if torch.backends.mps.is\_available() else "cpu").32  
* **Audit Tensor Layouts for Optimizers:** Search for dynamically generated tensors, particularly during custom loss calculations or LoRA weight initializations. Ensure they explicitly call .contiguous() before being passed to optimizers. Failure to do so results in the addcmul\_ silent failure where weights do not update.4  
* **Enforce Single-Threaded Dataloading:** Search the codebase for num\_workers. If torch.backends.mps.is\_available() evaluates to True, this parameter must be hardcoded to 0\.9  
* **Inject the Fallback Variable:** Ensure os.environ \= "1" is executed at the very start of the script, *before* importing torch.16  
* **Sanitize Generation Configs:** Strip out legacy forced\_decoder\_ids from Whisper configurations. Replace them with explicit model.generation\_config.language \= "en" and model.generation\_config.task \= "transcribe".12  
* **Verify AutoProcessor File Loading:** Ensure that initialization logic accounts for the v5.3.0 changes, where passing vocab and merges as file paths directly to tokenizer initialization requires proper format detection.3

### **Ordered Migration Steps**

**Step 1: Environmental Initialization and Verification** Export all necessary environment variables, initialize the Accelerator, and instantiate the AutoProcessor. *Validation:* Pass a dummy audio tensor through the processor to ensure input\_features are correctly shaped and the httpx backend successfully pulls the configuration from the Hugging Face Hub without throwing HTTP errors.3

**Step 2: Safe Model Loading and Device Placement**

Load the model using from\_pretrained. Utilize low\_cpu\_mem\_usage=True to leverage memory mapping (safetensors), preventing the model from duplicating its memory footprint in RAM before transitioning to the unified GPU context. Load in torch.float32.

*Validation:* Assert model.device.type \== 'mps'.

**Step 3: Data Pipeline Execution** Pass a single, constrained batch through the forward pass of the model. *Validation:* Monitor the system using Activity Monitor. Ensure that the "Physical footprint" unified memory usage remains stable and does not rapidly inflate into the yellow/red "swap" zone.35

**Step 4: Generative Evaluation Pass** Execute a greedy decoding pass using model.generate(). *Validation:* Ensure the text output is coherent and not a repeating string of \`\` tokens, which is a primary symptom of sequence length buffer overflows or missing aten:: operators failing silently during the autoregressive loop.36

## **4\. Failure Modes Catalog**

This section details the most critical failure boundaries encountered when operating Hugging Face Transformers on the MPS backend.

### **Issue 1: Seq2SeqTrainer Freezes During Evaluation**

* **Symptom:** The training loop progresses perfectly, but the progress bar stalls indefinitely (e.g., stopping exactly at step 1001\) during the evaluation phase. The GPU memory remains reserved, but compute utilization drops to zero.6  
* **Root Cause:** When predict\_with\_generate=True is enabled, the trainer accumulates all generated tokens and hidden states in MPS memory across the entire evaluation dataset before calculating metrics. This exhausts the Metal buffer, triggering OS-level SSD swapping that effectively hangs the Python process.1  
* **Minimal Fix:** Add eval\_accumulation\_steps=1 to the Seq2SeqTrainingArguments to force the offloading of predictions to the CPU after every discrete step.7  
* **Robust Fix:** Abandon the Seq2SeqTrainer for generative tasks. Implement a manual accelerate evaluation loop with explicit garbage collection.  
* **Code Example (Minimal Fix):**  
  Python  
  training\_args \= Seq2SeqTrainingArguments(  
      output\_dir="./whisper-mps-checkpoints",  
      predict\_with\_generate=True,  
      evaluation\_strategy="steps",  
      eval\_steps=500,  
      eval\_accumulation\_steps=1, \# CRITICAL: Forces CPU offload on MPS  
      dataloader\_num\_workers=0,  \# CRITICAL: Prevents macOS fork crashes  
      fp16=False \# Prefer FP32 on Mac to avoid overflow  
  )

* **Verification:** The evaluation loop completes successfully, and macOS Activity Monitor shows a stable memory footprint without high swap usage.

### **Issue 2: Silent Training Failures (Zero Gradient Updates)**

* **Symptom:** The training loss plateaus instantly. Specific model layers (e.g., the encoder in a seq2seq architecture) freeze at their initialized values, while others train normally. No NaNs or exceptions are thrown to the console.4  
* **Root Cause:** Apple's MPS backend silently fails when executing specific in-place operations (addcmul\_, addcdiv\_) on non-contiguous tensors. When the Adam optimizer attempts to update momentum and variance matrices on a transposed or non-contiguous view, the tensor simply does not update, masking the bug as a hyperparameter issue.4  
* **Minimal Fix:** Ensure all model weights are contiguous before passing them to the optimizer initialization.  
* **Robust Fix:** Enforce .contiguous() memory layouts during custom weight initialization (especially crucial for LoRA adapters) and upgrade to PyTorch \>= 2.6.0, where in-place view mutations properly raise RuntimeError instead of failing silently.4  
* **Code Example:**  
  Python  
  \# Anti-pattern (Causes Silent Failure):   
  \# encoder.weight \= decoder.weight.T.clone() 

  \# Robust Fix:  
  for name, param in model.named\_parameters():  
      if not param.is\_contiguous():  
          \# Force reallocation in unified memory to a contiguous block  
          param.data \= param.data.contiguous()

  optimizer \= torch.optim.AdamW(model.parameters(), lr=1e-5)

* **Verification:** Print param.grad and observe a non-zero update in the layer weights after optimizer.step().

### **Issue 3: Dataloader Multiprocessing Crash**

* **Symptom:** RuntimeError: DataLoader worker (pid X) exited unexpectedly with exit code 1\. The training script crashes instantly upon invoking the first batch from the dataloader.8  
* **Root Cause:** macOS utilizes spawn rather than Linux's fork for process creation in multiprocessing contexts. Spawning new worker processes that attempt to interact with the MPS device context concurrently leads to immediate termination, as the Metal framework does not support this type of shared memory context.8  
* **Minimal Fix:** Set num\_workers=0 in the DataLoader or TrainingArguments.  
* **Robust Fix:** If dataset preprocessing is heavily CPU-bound and creates a bottleneck, decouple data preparation from training. Pre-tokenize and map the dataset to disk offline, then load the cached .arrow dataset during training with num\_workers=0.  
* **Code Example:**  
  Python  
  train\_dataloader \= DataLoader(  
      tokenized\_dataset,  
      batch\_size=16,  
      shuffle=True,  
      collate\_fn=data\_collator,  
      num\_workers=0 \# MUST BE 0 ON macOS Apple Silicon  
  )

* **Verification:** The training loop initiates and processes batches without throwing a PID exit error.

### **Issue 4: Missing Sparse Tensor Operations in Whisper**

* **Symptom:** NotImplementedError: Could not run 'aten::\_sparse\_coo\_tensor\_with\_dims\_and\_tensors' with arguments from the 'SparseMPS' backend..11  
* **Root Cause:** Whisper architectures and certain embedding lookups optimize specific operations by utilizing sparse tensors. The MPS backend lacks coverage for various sparse tensor operations, and the SparseMPS framework is incomplete.11  
* **Minimal Fix:** Enable the PyTorch CPU fallback environment variable.  
* **Robust Fix:** Set the environment variable globally at the top of the execution script to ensure any missing operator transparently drops to the CPU.  
* **Code Example:**  
  Python  
  import os  
  \# Must be set BEFORE importing torch to hook into the backend initialization  
  os.environ \= "1"  
  import torch  
  from transformers import WhisperForConditionalGeneration

* **Verification:** The Whisper forward pass completes successfully. Execution speed will slightly decrease when the specific fallback operator executes, but it prevents a fatal crash.

### **Issue 5: Out of Memory on Long Sequence Generation**

* **Symptom:** RuntimeError: MPS backend out of memory... Tried to allocate X MB on private pool. This reliably occurs when decoding audio files longer than 30 seconds or processing text prompts exceeding 12,000 tokens.5  
* **Root Cause:** The scaled\_dot\_product\_attention (SDPA) mechanism natively allocates ![][image1] memory for the attention matrix. On MPS, Metal enforces strict maximum buffer limits. When the sequence length exceeds the threshold, PyTorch attempts to allocate a buffer larger than Metal permits, resulting in a direct OOM crash.5  
* **Minimal Fix:** Set PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0 to disable the upper allocation limit, though this risks system-wide freezing.18  
* **Robust Fix:** Implement dynamic sequence chunking (audio segmenting) prior to generation, passing strict 30-second audio windows to the Whisper model iteratively and manually aggregating the text.23  
* **Code Example (Audio Chunking Strategy):**  
  Python  
  chunk\_length\_s \= 30.0  
  chunk\_samples \= int(chunk\_length\_s \* 16000)  
  transcriptions \=

  for start\_idx in range(0, len(audio), chunk\_samples):  
      audio\_chunk \= audio\[start\_idx : start\_idx \+ chunk\_samples\]  
      inputs \= processor(audio\_chunk, return\_tensors="pt", sampling\_rate=16000)  
      inputs \= inputs.to("mps")

      \# Free memory between iterations to prevent Metal buffer exhaustion  
      import gc  
      gc.collect()  
      torch.mps.empty\_cache()

      with torch.no\_grad():  
          predicted\_ids \= model.generate(\*\*inputs)

      decoded \= processor.batch\_decode(predicted\_ids, skip\_special\_tokens=True)  
      transcriptions.append(decoded)

* **Verification:** The full audio file processes iteratively without triggering the Metal buffer allocation error.

### **Issue 6: use\_cache and Legacy Config Collisions**

* **Symptom:** Generation produces repeating loops of text, outputs an endless sequence of \`\` tokens, or raises severe warnings about conflicting decoder IDs.13  
* **Root Cause:** In versions prior to transformers 5.x, developers manipulated forced\_decoder\_ids to force English transcription. Modern versions handle this via the generation\_config.language and task properties. Combining legacy configs with modern flags, or using use\_cache=True with misaligned position IDs, breaks the autoregressive loop.12  
* **Minimal Fix:** Strip forced\_decoder\_ids entirely from both the model and generation configurations.  
* **Robust Fix:** Explicitly clear the legacy attribute and assign the modern parameters directly onto the generation\_config object before invoking .generate().  
* **Code Example:**  
  Python  
  model \= WhisperForConditionalGeneration.from\_pretrained("openai/whisper-small").to("mps")

  \# Purge legacy attributes to prevent collisions  
  model.config.forced\_decoder\_ids \= None  
  model.generation\_config.forced\_decoder\_ids \= None

  \# Apply modern explicit configuration  
  model.generation\_config.language \= "en"  
  model.generation\_config.task \= "transcribe"  
  model.generation\_config.use\_cache \= True \# Now safe to enable

* **Verification:** Warning logs regarding forced\_decoder\_ids disappear, and standard beam search operates correctly without repeating loops.

### **Issue 7: Accidental float64 Promotion Traps**

* **Symptom:** TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.  
* **Root Cause:** The MPS backend fundamentally does not support double-precision (float64) math operations.1 Within custom loss functions or metrics calculations, PyTorch operations (like certain normalization steps or divisions) may implicitly promote a float32 tensor to float64 to preserve precision. When this promoted tensor is passed back to an MPS operation, the framework crashes.  
* **Minimal Fix:** Cast the tensor back to float32 before continuing.  
* **Robust Fix:** Wrap custom loss calculations or metrics logging in explicit dtype casts to ensure mathematical intermediaries remain in 32-bit precision.  
* **Code Example:**  
  Python  
  def compute\_custom\_loss(logits, labels):  
      \# Anti-pattern: Division by a Python float might trigger promotion  
      \# loss \= torch.sum(logits) / len(labels)

      \# Robust Fix: Explicitly enforce float32  
      loss \= torch.sum(logits, dtype=torch.float32) / len(labels)  
      return loss.to(torch.float32)

* **Verification:** The training step completes without a TypeError crash.

### **Issue 8: Gradient Checkpointing NotImplementedError**

* **Symptom:** When gradient\_checkpointing=True is enabled, the pipeline crashes with a NotImplementedError regarding missing unique or view operations during the backward pass.41  
* **Root Cause:** Gradient checkpointing relies heavily on reentrant autograd graphs and specific tensor view operations that are highly unstable or missing on the MPS backend.41 The interaction between use\_reentrant=False (the new default in PyTorch) and the Metal graph compiler frequently fails.43  
* **Minimal Fix:** Disable gradient checkpointing if memory permits.  
* **Robust Fix:** If checkpointing is strictly required to fit the model in unified memory, force the legacy reentrant behavior in the training arguments or model config.  
* **Code Example:**  
  Python  
  \# Force legacy reentrant checkpointing which is more stable on MPS  
  model.config.use\_cache \= False \# Must be False when checkpointing  
  model.gradient\_checkpointing\_enable(gradient\_checkpointing\_kwargs={"use\_reentrant": True})

* **Verification:** The backward pass successfully computes gradients without throwing a NotImplementedError.

### **Issue 9: Memory Leak with MPS Caching**

* **Symptom:** Memory usage steadily increases over time during model training or inference, unlike the stable memory usage observed with the CPU backend. torch.mps.empty\_cache() has no effect.14  
* **Root Cause:** The memory is held at the Metal driver level. Missing autorelease pools in the PyTorch backend (particularly for LSTMs and heavy generation loops) prevent the OS from reclaiming the memory.14  
* **Minimal Fix:** Update to PyTorch 2.6.0 which addresses the getStridedMPSNDArray memory leak.29  
* **Robust Fix:** Implement aggressive, multi-stage garbage collection inside the training/inference loop.  
* **Code Example:**  
  Python  
  import gc  
  import torch

  \# Inside your loop:  
  del outputs  
  del loss  
  gc.collect() \# Stage 1: Clear Python reference counters  
  torch.mps.empty\_cache() \# Stage 2: Clear MPS cached allocations

* **Verification:** Activity Monitor shows the memory footprint dropping after each iteration rather than climbing monotonically.

## **5\. Best Practices**

### **Attention Backend Selection**

| Backend | MPS Support | Implication | Recommendation |
| :---- | :---- | :---- | :---- |
| **FlashAttention-2/3** | No 1 | Fails completely; requires CUDA. | Avoid. |
| **xFormers** | No 44 | CUDA-only custom kernels. | Avoid. |
| **SDPA (Math Fallback)** | Yes 45 | Computes attention using standard PyTorch ops. Stable for sequences \< 12k. | **Default choice**. Handled automatically by transformers \>= 4.36 on PyTorch \>= 2.0. |

### **Dtype Enforcement Strategies**

| Dtype | MPS Status | Usage Guideline |
| :---- | :---- | :---- |
| torch.float32 | Native | **Gold Standard.** Use for all training and base model loading. |
| torch.float16 | Native | Use for inference to save 50% memory. Prone to overflow gradients during training. |
| torch.bfloat16 | Emulated 1 | Avoid. Computations are upcast under the hood. Causes severe numerical divergence compared to A100/H100 runs. |
| torch.float64 | Unsupported 45 | Will crash. Explicitly cast down to FP32 if custom metrics promote tensors. |

### **Safe Model Loading**

Leverage low\_cpu\_mem\_usage=True. This parameter relies on safetensors to memory-map the model weights directly, preventing the system from loading the entire model into system RAM before moving it to the GPU. On Apple Silicon, because RAM and VRAM are the same unified pool, failing to use this can cause an immediate 2x memory spike that triggers OS swapping.

Python

model \= AutoModelForSpeechSeq2Seq.from\_pretrained(  
    "openai/whisper-large-v3",  
    torch\_dtype=torch.float32,  
    low\_cpu\_mem\_usage=True, \# Critical for Unified Memory  
    use\_safetensors=True  
).to("mps")

## **6\. Worst Practices / Anti-patterns**

* **Relying on torch.autograd.detect\_anomaly:** Activating this debugging tool on MPS will drop execution speeds by over 100x and often mask actual hardware-level timing bugs. It is practically unusable for large models on Mac.1 Debug NaNs manually via explicit tensor inspection.  
* **Attempting Distributed Paradigms:** Initializing DDP, FSDP, or passing devices \> 1 in accelerate or PyTorch Lightning is futile. macOS gloo backends operate entirely over the CPU RAM, destroying communication latency, and nccl is fundamentally unavailable.1  
* **Blindly Trusting torch.compile:** PyTorch 2.6/2.7 explicitly lists MPS support for torch.compile as an "early prototype." Multi-stage reductions, RMS Norm tracing, and dynamic shapes consistently fail, reverting to CPU kernels and causing massive performance degradation compared to standard eager mode.1  
* **Assuming Identical RNG:** The MPS backend does not strictly mirror the stochastic rounding probabilities or random number generation logic of NVIDIA hardware.1 Setting torch.manual\_seed(42) on Mac and CUDA will yield mathematically divergent training loss curves almost immediately. Verification of numerical convergence must occur on NVIDIA hardware.

## **7\. Weird Patterns That Work Surprisingly Well on Apple Silicon**

* **Aggressive Garbage Collection Before Cache Clearing:** In PyTorch MPS, the Metal driver holds memory hostage even after Python variable deletion. Executing del model followed by torch.mps.empty\_cache() does not free memory. The effective pattern requires invoking the Python garbage collector *between* the deletion and the cache dump to sever the reference counters before Metal is queried.15  
* **Selective Offloading via CPU Swapping:** If a specific transformer layer or processing step is causing OOM on a 16GB Mac, explicitly mapping a tensor to('cpu'), performing the operation, and mapping back to('mps') is surprisingly performant. Because Apple Silicon uses Unified Memory, the PCIe bus bottleneck does not exist. The memory is shared, making "transfers" between CPU and GPU virtually instantaneous compared to discrete PCIe setups.10

## **8\. Limitations and Unsolved Problems**

* **Algorithmic Drift and Non-Determinism:** As established, local debugging of loss convergence is functionally impossible due to algorithmic drift.1 Models prototyped on Mac must be validated on CUDA before assuming the hyperparameter configuration is sound.  
* **MLX Fragmentation:** Apple's proprietary framework, MLX, routinely outperforms PyTorch MPS by 2-3x for generation tasks by exploiting hardware-specific features like the Neural Engine and KV-cache sharing. PyTorch MPS remains comparatively slower as it relies on more generic Metal mappings.1  
* **Memory Fragmentation on Dynamic Shapes:** Feeding batches of varying sequence lengths to scaled\_dot\_product\_attention on MPS causes massive memory fragmentation, as the Metal allocator struggles to resize buffers dynamically.5 Padding to a fixed sequence length actually *saves* memory by reusing the same buffer context, counter to standard NLP optimization intuition.

## **9\. Copy-Paste Reference Snippets**

### **A. Core Training Loop with Accelerate (MPS Optimized)**

This snippet demonstrates a robust, single-device accelerate setup optimized to prevent MPS memory exhaustion and handle missing sparse operations.

Python

import os  
import gc  
import torch  
from accelerate import Accelerator  
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

\# 1\. Mandatory Fallback for unsupported Aten ops  
os.environ \= "1"

\# 2\. Initialize Accelerator without unsupported distributed paradigms  
accelerator \= Accelerator(mixed\_precision="no") \# FP32 is safest for MPS gradients  
device \= accelerator.device \# Automatically resolves to MPS if available

\# 3\. Safe Model Loading  
model \= AutoModelForSeq2SeqLM.from\_pretrained(  
    "openai/whisper-small",  
    low\_cpu\_mem\_usage=True,  
    use\_safetensors=True  
)  
optimizer \= torch.optim.AdamW(model.parameters(), lr=1e-5)

\# 4\. Anti-Silent-NaN Protocol  
for param in model.parameters():  
    if not param.is\_contiguous():  
        param.data \= param.data.contiguous()

model, optimizer, dataloader \= accelerator.prepare(model, optimizer, my\_dataloader)

\# 5\. Execution Loop  
for epoch in range(3):  
    model.train()  
    for batch in dataloader:  
        optimizer.zero\_grad()  
        outputs \= model(\*\*batch)  
        loss \= outputs.loss  
        accelerator.backward(loss)  
        optimizer.step()  
          
    \# 6\. Crucial MPS Cleanup Pattern (Execute per epoch or per N steps)  
    gc.collect()  
    torch.mps.empty\_cache()

### **B. Correctness Validation (Parity Test CPU vs MPS)**

Because of numerical drift and missing operators, validating output equivalence between CPU and MPS is a required step when upgrading transformers or torch.

Python

import torch  
import os  
os.environ \= "1"

def validate\_mps\_correctness(model, processor, sample\_audio):  
    \# Process inputs  
    inputs \= processor(sample\_audio, return\_tensors="pt", sampling\_rate=16000)  
      
    \# 1\. CPU Baseline Execution  
    model.to("cpu")  
    model.eval()  
    with torch.no\_grad():  
        cpu\_outputs \= model.generate(  
            \*\*inputs,   
            output\_scores=True,   
            return\_dict\_in\_generate=True  
        )  
      
    \# 2\. MPS Execution  
    model.to("mps")  
    inputs\_mps \= {k: v.to("mps") for k, v in inputs.items()}  
    with torch.no\_grad():  
        mps\_outputs \= model.generate(  
            \*\*inputs\_mps,   
            output\_scores=True,   
            return\_dict\_in\_generate=True  
        )  
      
    \# 3\. Calculate Divergence  
    \# We examine the first generation step's raw logits to identify drift  
    cpu\_logits \= cpu\_outputs.scores  
    mps\_logits \= mps\_outputs.scores.cpu()  
      
    max\_diff \= torch.max(torch.abs(cpu\_logits \- mps\_logits)).item()  
    print(f"Max Absolute Divergence (CPU vs MPS): {max\_diff:.6f}")  
      
    if max\_diff \> 1e-3:  
        print("WARNING: Significant numerical drift detected between backends. Review precision.")  
        return False  
    return True

## **10\. Final "Red Flags" Checklist Before Shipping**

Before deploying a pipeline to production or pushing code meant to be run by end-users on Macs, audit the pull request against this final checklist:

1. \[ \] **Is num\_workers exactly 0?** If any Dataloader sets num\_workers \> 0 on macOS, the program will crash on the first epoch due to OS-level spawning conflicts with Metal.  
2. \[ \] **Is PYTORCH\_ENABLE\_MPS\_FALLBACK=1 injected?** This is essential for preventing hard crashes on sparse tensor lookups (common in Whisper) or missing matrix reductions.  
3. \[ \] **Are contiguous() memory checks passed?** Are weights and tensors explicitly verified for contiguous layout before being touched by Adam or addcmul\_ to prevent silent freezing?  
4. \[ \] **Is eval\_accumulation\_steps configured?** If using Seq2SeqTrainer with generative evaluation, is this flag explicitly set to 1 to prevent indefinite memory accumulation and swap freezing?  
5. \[ \] **Are legacy forced\_decoder\_ids purged?** Is the generation configuration properly leveraging the modern language and task flags for Whisper, avoiding infinite generation loops?  
6. \[ \] **Is gc.collect() present?** Are you invoking the Python garbage collector immediately prior to torch.mps.empty\_cache() to ensure the Metal driver actually frees unified memory?  
7. \[ \] **Is Dtype explicitly defined?** Are you avoiding torch.bfloat16 and torch.float64 to prevent emulation overhead and direct crashes, respectively?

#### **Works cited**

1. State of PyTorch Hardware Acceleration 2025, accessed April 2, 2026, [https://tunguz.github.io/PyTorch\_Hardware\_2025/](https://tunguz.github.io/PyTorch_Hardware_2025/)  
2. transformers by Hugging Face \- Release Notes \- March 2026 Latest Updates \- Releasebot, accessed April 2, 2026, [https://releasebot.io/updates/huggingface/transformers](https://releasebot.io/updates/huggingface/transformers)  
3. transformers/MIGRATION\_GUIDE\_V5.md at main \- GitHub, accessed April 2, 2026, [https://github.com/huggingface/transformers/blob/main/MIGRATION\_GUIDE\_V5.md](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)  
4. the bug that taught me more about PyTorch than years of using it \- Elana Simon, accessed April 2, 2026, [https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)  
5. Optimizing PyTorch MPS Attention: Memory-Efficient Large Sequence Processing Without Accuracy Trade-offs | by Raksheka R | Medium, accessed April 2, 2026, [https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b)  
6. Huggingface Seq2seqTrainer freezes on evaluation \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/78128694/huggingface-seq2seqtrainer-freezes-on-evaluation](https://stackoverflow.com/questions/78128694/huggingface-seq2seqtrainer-freezes-on-evaluation)  
7. Trainer \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/transformers/v4.33.0/main\_classes/trainer](https://huggingface.co/docs/transformers/v4.33.0/main_classes/trainer)  
8. DataLoader with num\_workers\>0 fails when running with "spawn" and using joblib · Issue \#44687 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/44687](https://github.com/pytorch/pytorch/issues/44687)  
9. The DataLoader can't work in Apple Silicon. · Issue \#70344 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/70344](https://github.com/pytorch/pytorch/issues/70344)  
10. accelerate/docs/source/usage\_guides/mps.md at main \- GitHub, accessed April 2, 2026, [https://github.com/huggingface/accelerate/blob/main/docs/source/usage\_guides/mps.md](https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/mps.md)  
11. MPS Backend: NotImplementedError for aten::\_sparse\_coo\_tensor\_with\_dims\_and\_tensors when running Whisper \#141711 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/141711](https://github.com/pytorch/pytorch/issues/141711)  
12. transformers/src/transformers/models/whisper/generation\_whisper.py at main \- GitHub, accessed April 2, 2026, [https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation\_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py)  
13. Trainer freezes/crashes after evaluation step \- Transformers \- Hugging Face Forums, accessed April 2, 2026, [https://discuss.huggingface.co/t/trainer-freezes-crashes-after-evaluation-step/77556](https://discuss.huggingface.co/t/trainer-freezes-crashes-after-evaluation-step/77556)  
14. Memory Leak in MPS Backend During LSTM Iterations (Out of Memory Error) · Issue \#145374 · pytorch/pytorch \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/145374](https://github.com/pytorch/pytorch/issues/145374)  
15. How to free GPU memory in PyTorch \- python \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch](https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch)  
16. Apple Silicon \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/transformers/perf\_train\_special](https://huggingface.co/docs/transformers/perf_train_special)  
17. MPS training (basic) — PyTorch Lightning 2.6.1 documentation, accessed April 2, 2026, [https://lightning.ai/docs/pytorch/stable/accelerators/mps\_basic.html](https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html)  
18. MPS Back End Out of Memory on GitHub Action \- PyTorch Forums, accessed April 2, 2026, [https://discuss.pytorch.org/t/mps-back-end-out-of-memory-on-github-action/189773](https://discuss.pytorch.org/t/mps-back-end-out-of-memory-on-github-action/189773)  
19. mps\_environment\_variables.rst.txt \- PyTorch.org, accessed April 2, 2026, [https://docs.pytorch.org/docs/2.6/\_sources/mps\_environment\_variables.rst.txt](https://docs.pytorch.org/docs/2.6/_sources/mps_environment_variables.rst.txt)  
20. \[MPS\] Regression from macOS 14.3 to 14.4 in PyTorch 2.2.0/2.2.1 · Issue \#122016 · pytorch/pytorch \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/122016](https://github.com/pytorch/pytorch/issues/122016)  
21. torch.mps.synchronize — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)  
22. torch.compile on MPS progress tracker · Issue \#150121 · pytorch ..., accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/150121](https://github.com/pytorch/pytorch/issues/150121)  
23. Hugging face model not transcribing the entire length of the audio file \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/76850304/hugging-face-model-not-transcribing-the-entire-length-of-the-audio-file](https://stackoverflow.com/questions/76850304/hugging-face-model-not-transcribing-the-entire-length-of-the-audio-file)  
24. Why does Seq2SeqTrainer produces error during evaluation when using T5?, accessed April 2, 2026, [https://stackoverflow.com/questions/78182686/why-does-seq2seqtrainer-produces-error-during-evaluation-when-using-t5](https://stackoverflow.com/questions/78182686/why-does-seq2seqtrainer-produces-error-during-evaluation-when-using-t5)  
25. Trainer.evaluate() freezing \- Transformers \- Hugging Face Forums, accessed April 2, 2026, [https://discuss.huggingface.co/t/trainer-evaluate-freezing/75067](https://discuss.huggingface.co/t/trainer-evaluate-freezing/75067)  
26. m-ric/transformers\_documentation\_en · Datasets at Hugging Face, accessed April 2, 2026, [https://huggingface.co/datasets/m-ric/transformers\_documentation\_en/viewer/default/train](https://huggingface.co/datasets/m-ric/transformers_documentation_en/viewer/default/train)  
27. Accelerated PyTorch Training on Mac \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/accelerate/usage\_guides/mps](https://huggingface.co/docs/accelerate/usage_guides/mps)  
28. scaled\_dot\_product\_attention crashes on apple silicon · Issue \#149132 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/149132](https://github.com/pytorch/pytorch/issues/149132)  
29. MPS Memory Leak · Issue \#154329 · pytorch/pytorch \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/154329](https://github.com/pytorch/pytorch/issues/154329)  
30. How to fine-tune open LLMs in 2025 with Hugging Face \- Philschmid, accessed April 2, 2026, [https://www.philschmid.de/fine-tune-llms-in-2025](https://www.philschmid.de/fine-tune-llms-in-2025)  
31. How to Fine-Tune an OPUS MT Model? \- Kaggle, accessed April 2, 2026, [https://www.kaggle.com/code/alvations/how-to-fine-tune-an-opus-mt-model](https://www.kaggle.com/code/alvations/how-to-fine-tune-an-opus-mt-model)  
32. PyTorch model running on CPU despite MPS (Apple Silicon) being available and detected, accessed April 2, 2026, [https://stackoverflow.com/questions/79144698/pytorch-model-running-on-cpu-despite-mps-apple-silicon-being-available-and-det](https://stackoverflow.com/questions/79144698/pytorch-model-running-on-cpu-despite-mps-apple-silicon-being-available-and-det)  
33. Accelerated PyTorch training on Mac \- Metal \- Apple Developer, accessed April 2, 2026, [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)  
34. Using HuggingFace pipeline on pytorch mps device M1 pro \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/72861962/using-huggingface-pipeline-on-pytorch-mps-device-m1-pro](https://stackoverflow.com/questions/72861962/using-huggingface-pipeline-on-pytorch-mps-device-m1-pro)  
35. \[D\] How optimized is Pytorch for apple silicon \- Reddit, accessed April 2, 2026, [https://www.reddit.com/r/pytorch/comments/1elechb/d\_how\_optimized\_is\_pytorch\_for\_apple\_silicon/](https://www.reddit.com/r/pytorch/comments/1elechb/d_how_optimized_is_pytorch_for_apple_silicon/)  
36. evalstate/transformers-pr · Datasets at Hugging Face, accessed April 2, 2026, [https://huggingface.co/datasets/evalstate/transformers-pr](https://huggingface.co/datasets/evalstate/transformers-pr)  
37. Huggingface trainer leaves residual memory \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/76179395/huggingface-trainer-leaves-residual-memory](https://stackoverflow.com/questions/76179395/huggingface-trainer-leaves-residual-memory)  
38. Releases · pytorch/pytorch \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/releases](https://github.com/pytorch/pytorch/releases)  
39. Dataloader crashes if num\_worker\>0 on MacOS with Python 3.8 · Issue \#46648 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/46648](https://github.com/pytorch/pytorch/issues/46648)  
40. evalstate/transformers-pr · Datasets at Hugging Face, accessed April 2, 2026, [https://huggingface.co/datasets/evalstate/transformers-pr/viewer/](https://huggingface.co/datasets/evalstate/transformers-pr/viewer/)  
41. gopikrsmscs/torch-issues · Datasets at Hugging Face, accessed April 2, 2026, [https://huggingface.co/datasets/gopikrsmscs/torch-issues](https://huggingface.co/datasets/gopikrsmscs/torch-issues)  
42. Getting RuntimeError: view size is not compatible with input tensor's size and stride while using mps \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/79274150/getting-runtimeerror-view-size-is-not-compatible-with-input-tensors-size-and-s](https://stackoverflow.com/questions/79274150/getting-runtimeerror-view-size-is-not-compatible-with-input-tensors-size-and-s)  
43. torch.utils.checkpoint \- PyTorch documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/checkpoint.html](https://docs.pytorch.org/docs/stable/checkpoint.html)  
44. Memory-efficient attention (without xformers) · Issue \#1892 · huggingface/diffusers \- GitHub, accessed April 2, 2026, [https://github.com/huggingface/diffusers/issues/1892](https://github.com/huggingface/diffusers/issues/1892)  
45. torch.nn.functional.scaled\_dot\_product\_attention — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled\_dot\_product\_attention.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)  
46. (Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA) — PyTorch Tutorials 2.5.0+cu124 documentation, accessed April 2, 2026, [https://pytorch-cn.com/tutorials/intermediate/scaled\_dot\_product\_attention\_tutorial.html](https://pytorch-cn.com/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)  
47. Apple Intelligence Foundation Language Models Tech Report 2025, accessed April 2, 2026, [https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAYCAYAAAC4CK7hAAACsUlEQVR4XuWWy+uOQRTHz+uWS24lVih/gBW5LYSSNUKSH0qxsVHCws4lG34lsbDAghJ/gYXfQikpl41LWUgSFhQh1/N9ZuZ55/nOeead35tXyqdO78z3nDNzZp5n5n2kI/81A1v+XbVfarfZ8Xdormtto1fOu6h9Se1b1AdbqN+Gucvz1c6pnVGbRj6LPWoHWSwET+KEb6MY9GtUWKA/D2LN61lOixtou+/P05Q3+vulG5IwV2NesdiLlkIWCy3EM6x2nkWLMeIGGEHHmOS72k8WPcibyGIOY/wAXqutLHqsBSYg6DmLEavFxawhfbna19CJC8wU28YF8eehJfey4BVrcYKXYqyW4sMTu9aUqx3s92zE7FJb6tsrYkfEdKlqsFeyUlyBt0hnZoqLe086tEmkBa6orY/6m9SuSrfgwCJxh32H2m61JxDtcqv5zMsHO1ryjm8TF3c/0qZ6zeKH/4V/g7gNmCFu0dDiKxV9tjbgO4QGL9QlspryVFwsrtnAKq8xKHbYjwk/LooYaDdJ8/QsBLkXq1YUOts7rGIYK26noYGFamPV5ojzT266K21/6PQsvclntTssYjIMCmeOjeLiRkgf8nobZyX1L6u0jqt/lIsAn9TusQisnWZcTDorDq2RWwfCx58br73eL8i9wSKm/CD5gV+I849vylWx4SZjPQDf8Vjw2rGo3Uq6bxXIOcxiAM6HRuZbSXeUQe4EFpVZ4nx8NUPDf9IStb3kq+mWkhSFfGxgK/gCRRAOEs4M2pisF3jl6oMbsVnsHX8kTj/FjgKmiD3maEl2BxxQ+8jiQOhUX+PXWf6TYJfGsTgAck/D3GVp103WqT1jsZyiuY6oHS0LLSAzzklx30mDAH+sj1msyRRVRjJAZ4iVlCSphH0s9Elfk9dks7POf40BF/sb1neUgXKuhioAAAAASUVORK5CYII=>