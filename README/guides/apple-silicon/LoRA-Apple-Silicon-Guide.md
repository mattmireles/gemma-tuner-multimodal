# **Operational Field Guide: PEFT and LoRA Workflows on Apple Silicon (MPS)**

April 2, 2026

## **1\. Executive Summary**

This field guide establishes the definitive operational protocols for engineering teams actively shipping Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA) workflows on Apple Silicon. With a specific focus on the PyTorch Metal Performance Shaders (MPS) backend, this manual dictates the rigorous handling required to stabilize training, ensure evaluation parity, maintain checkpoint hygiene, and export deployment-safe sequence-to-sequence (seq2seq) and audio models, particularly the Whisper architecture.

* **Precision Realities and the BF16 Trap:** Apple Silicon lacks native BFloat16 (BF16) hardware instruction sets. Executing BF16 triggers software emulation and upcasting, introducing a severe "correctness gap" between local fine-tuning on unified memory and datacenter deployment.1 Standardize on FP32 for gradient stability or strictly utilize FP16 when memory reduction is paramount.2  
* **The Silent NaN Threat:** The MPS backend operates with significantly less strict floating-point exception handling than NVIDIA CUDA. Models will frequently continue training for hours while accumulating "Silent NaNs" (Not a Number) in the loss gradients, silently poisoning the checkpoint.1 Explicit gradient monitoring is mandatory.  
* **Memory Watermark Overrides:** MPS restricts PyTorch memory allocations aggressively to protect operating system stability. Overriding this constraint with PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0 is an absolute necessity for training larger 7B+ parameter models or Whisper-Large variants, though it risks OS-level swap crashes if physical RAM is exceeded.5  
* **Garbage Collection Mandates:** The standard Hugging Face Trainer lacks native MPS cache clearing mechanisms during high-throughput batches. Manual injection of torch.mps.empty\_cache() and gc.collect() within custom Trainer callbacks or manual loops is required to prevent out-of-memory (OOM) halts.8  
* **Target Module Optimization for Seq2Seq:** For Whisper architectures, targeting only the self-attention \["q\_proj", "v\_proj"\] matrices leaves significant alignment capacity untapped. Extending targets to include cross-attention projections and feed-forward networks (often targeted via all-linear) is required for optimal audio-to-text alignment, especially when utilizing advanced initializations.11  
* **Merging Discrepancies and Precision Loss:** Executing merge\_and\_unload() directly on the MPS device in FP16 frequently introduces catastrophic cancellation and precision degradation. The most robust workflow necessitates moving both the base model and the adapter to the CPU, casting to FP32, performing the matrix addition, and serializing to disk prior to CoreML or GGUF export.2  
* **Resumption Pitfalls:** Resuming seq2seq model training from a PEFT checkpoint routinely drops the decoder\_start\_token\_id. This causes the decoder to initiate without the Begin-Of-Sentence (BOS) token, leading to immediate alignment divergence.14 Checkpoint configurations must explicitly retain these generation parameters.  
* **Quantization Evolution:** As of early 2026, the mps-bitsandbytes library enables native Metal GPU acceleration for 4-bit (NF4/FP4) and 8-bit quantization.16 However, legacy pipelines relying on CUDA-only bitsandbytes must utilize CPU fallbacks or transition to Apple's native MLX framework.18  
* **Hardware Fallback Flags:** When unsupported operations halt the computational graph, setting PYTORCH\_ENABLE\_MPS\_FALLBACK=1 routes the specific operation to the CPU.19 While this prevents hard crashes, it introduces severe PCIe bottlenecking during the forward pass and should only be used as a last resort.  
* **Graph Break Penalties in Debugging:** Attempting to trace gradient errors using torch.autograd.detect\_anomaly on MPS slows execution by up to 100x, rendering it operationally unviable for large-scale seq2seq debugging.1  
* **Gradient Checkpointing Conflicts:** Utilizing use\_gradient\_checkpointing=True alongside PEFT requires careful unwrapping prior to evaluation generation loops. Wrapped modules frequently lack the gradient\_checkpointing\_disable method required by unwrap\_model\_for\_generation, causing immediate application crashes.21  
* **Trainer vs. Manual Loop Memory Dynamics:** The Hugging Face Trainer is heavily optimized for the CUDA caching allocator. When operating on Apple Silicon, manual training loops with explicit micro-batching and synchronous memory clearing often achieve higher throughput and greater stability than the default Trainer abstractions.

## **2\. Decision Framework**

The following frameworks govern the architectural, mathematical, and operational decisions required for deploying stable fine-tuning workflows on Apple's Unified Memory architectures.

### **2.1 When to use LoRA vs. Full Fine-Tune on Apple Silicon**

The decision between full fine-tuning and Low-Rank Adaptation on Apple Silicon is dictated almost entirely by Unified Memory bandwidth, capacity constraints, and the mathematical properties of the MPS allocator.

| Condition | Recommended Approach | MPS & Architectural Considerations |
| :---- | :---- | :---- |
| **Model Size \< 1B Parameters** | Full Fine-Tune or LoRA | Feasible in FP32. Full fine-tuning provides maximum adaptability without introducing adapter inference latency.2 |
| **Model Size 1B \- 7B Parameters** | LoRA | Full fine-tuning will trigger OOM failures on 16GB/32GB Mac architectures. LoRA reduces trainable gradient states by approximately 99%, allowing convergence within unified memory limits.22 |
| **Whisper Large (1.5B)** | LoRA with Mixed Precision | Requires \~24GB VRAM for full fine-tuning. PEFT reduces this requirement to \<8GB, fitting comfortably on baseline M-series chips while maintaining word error rate (WER) parity.24 |
| **Domain-Specific Adaptation** | LoRA | Enables maintaining multiple specialized variants (e.g., medical transcription, legal dictation) by hot-swapping adapters dynamically without duplicating the base model's memory footprint.2 |

### **2.2 Target Module Selection Strategy for Seq2Seq and Audio Models**

Selecting the correct target\_modules within the LoraConfig dictates the expressive capacity of the adapter. For encoder-decoder architectures like Whisper, the placement of low-rank matrices critically impacts the model's ability to map 80-dimensional log-Mel spectrogram frames to text tokens.

| Target Selection | Parameter Footprint | Application / Model Type | Tradeoffs on Apple Silicon |
| :---- | :---- | :---- | :---- |
| \["q\_proj", "v\_proj"\] | Minimal (\~1-2%) | Standard seq2seq, baseline Whisper.28 | Fastest training, lowest memory footprint. Often insufficient for complex multilingual audio alignments or heavy acoustic noise.26 |
| \["q\_proj", "k\_proj", "v\_proj", "out\_proj"\] | Moderate (\~3-5%) | Complex reasoning, deep audio transcription.30 | Highly balanced. Recommended for Whisper models handling diverse accents or background noise, providing broader attention modification. |
| all-linear | High (\~5-10%) | Generalization tasks, LoftQ initialization.11 | Maximizes adaptation capability by injecting adapters into the feed-forward networks (FFNs) alongside attention. Increases memory pressure; requires strict batch size tuning on MPS. |
| Selective Cross-Attention | Variable | Asymmetric seq2seq models, speech translation.31 | Targets the specific mechanism bridging the encoder's audio representation with the decoder's text generation. Requires custom regex patterns (e.g., .\*decoder.\*(q\_proj|v\_proj)). Ideal if the encoder representations are already robust. |

### **2.3 Merging Adapters vs. Keeping Them Separate**

The lifecycle of an adapter requires a decisive shift between the training phase and the deployment phase.

* **Keep Adapters Separate:** This is mandatory during the experimentation phase, during evaluation parity checks, or when deploying via a cloud endpoint that dynamically routes requests to different domain experts. Unmerged models preserve the exact mathematical precision of the base model during inference.22  
* **Merge Adapters (merge\_and\_unload):** Merging is strictly required prior to exporting models to CoreML via coremltools or to GGUF formats for edge deployment.32 Merged models execute faster locally by eliminating the computational overhead of the parallel adapter forward pass. However, as noted in the executive summary, merging on the MPS device carries severe precision risks.

### **2.4 Quantization: Necessity vs. Trap**

* **When Quantization is Worth It:** Utilizing mps-bitsandbytes (which supports NF4/FP4 and 8-bit blockwise quantization natively on Metal) or QLoRA is highly effective when dealing with 7B+ parameter language models or Whisper Large v3 Turbo on constrained hardware (e.g., MacBooks with 16GB RAM).16  
* **When Quantization is a Trap:** Simulating FP4/Int4 via generic torchao abstractions on Apple Silicon is a dangerous anti-pattern. Apple hardware lacks native Tensor Cores engineered for these specific precision formats. Consequently, the MPS backend executes these as emulated operations, upcasting them to BF16 or FP32 in the background. This creates a severe correctness gap: the model achieves numerical convergence locally due to the hidden high-precision math, but diverges catastrophically when pushed to datacenter hardware lacking those wide accumulation registers.1 Legacy CUDA-bound bitsandbytes implementations will also fail completely, requiring CPU fallbacks that destroy training throughput.18

## **3\. Migration Playbook**

When refactoring a legacy full-fine-tuning seq2seq script to support MPS-accelerated PEFT, the deployment team must adhere strictly to the following ordered checklist to prevent architectural mismatches and memory leaks.

### **3.1 Audit Checklist: Safe Version Combinations**

The Apple Silicon machine learning ecosystem is highly sensitive to version mismatches. The following table outlines the minimum required versions for stable MPS training as of 2026\.

| Framework | Minimum Safe Version | Justification for MPS Stability |
| :---- | :---- | :---- |
| torch | \>= 2.3.0 | Contains critical fixes for torch.mps.empty\_cache() and Metal memory leak resolutions.36 |
| transformers | \>= 4.45.0 | Integrates updated Seq2Seq trainer logic that properly handles device mapping for un-merged PEFT models during generation loops.21 |
| peft | \>= 0.18.0 | Contains fixes for PiSSA and LoftQ initialization methods, and proper state dict handling for merge\_and\_unload precision.21 |
| accelerate | \>= 0.33.0 | Ensures that accelerator.free\_memory() correctly hooks into the MPS cache rather than defaulting to CUDA.30 |
| mps-bitsandbytes | \>= 0.7.0 | Required for native Metal GPU acceleration of NF4/FP4. Legacy bitsandbytes will trigger CPU fallbacks.16 |

### **3.2 Ordered Migration Steps**

1. **Environment Variable Injection:** Prior to importing PyTorch, inject the necessary MPS safety flags into the shell or Python environment to prevent immediate allocation crashes.  
   Bash  
   export PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0  
   export PYTORCH\_ENABLE\_MPS\_FALLBACK=1

2. **Explicit Device Mapping:** Remove any instances of device\_map="auto". While convenient on CUDA, this abstraction frequently misallocates tensors on mixed unified memory systems. Explicitly cast the base model: model.to(torch.device("mps")).30  
3. **Data Collator Restructuring for Seq2Seq:** The data collator must be rewritten to dynamically pad audio features and text labels. Crucially, padding tokens in the label sequences must be replaced with \-100. This ensures the PyTorch CrossEntropyLoss algorithm ignores them correctly during the backward pass.27  
4. **PEFT Wrapper Injection:**  
   * Initialize the LoraConfig with appropriate target modules.  
   * Wrap the base model: model \= get\_peft\_model(model, peft\_config).  
   * Verify the trainable parameter ratio using model.print\_trainable\_parameters().27  
5. **Gradient Checkpointing Adjustments:** If the memory budget necessitates model.gradient\_checkpointing\_enable(), the evaluation logic must be modified. Ensure generation loops temporarily detach the wrapper or suppress the checkpointing state, as wrapped modules frequently lack the gradient\_checkpointing\_disable method required by the unwrapper, resulting in AttributeError exceptions.21

### **3.3 Post-Migration Validation Checklist**

* \[ \] **Overfit a Single Batch:** Run 50 training steps on a single batch. The loss must approach 0\. If it flatlines or emits NaNs, the decoder\_start\_token\_id is likely misconfigured, or an MPS accumulation error has occurred.  
* \[ \] **Memory Profiling Verification:** Execute torch.mps.current\_allocated\_memory() before and after the backward pass.41 Confirm that the memory usage stabilizes and does not display a monotonic upward leak per epoch.  
* \[ \] **Parity Export Verification:** Save the adapter, unload it, and reload it. Confirm the inference logits match the active training graph exactingly before proceeding to full-scale training.

## **4\. Failure Modes Catalog**

Engineering on the Metal Performance Shaders backend requires navigating a distinct set of hardware constraints and software compilation anomalies. The following catalog details the most critical failures, their root causes, and explicit, pasteable fixes.

### **4.1 Issue: The Silent NaN (Loss Explosion)**

* **Symptom:** The training loss suddenly spikes to infinity, or output generations become endless repetitive loops of a single token. PyTorch throws no errors, and the progress bar continues normally.  
* **Root Cause:** The MPS hardware is significantly less strict about floating-point exceptions (overflows and underflows) than NVIDIA CUDA.1 A single division by zero or gradient overflow in an adapter matrix will multiply through the entire computational graph silently, poisoning the model weights.  
* **Minimal Fix:** Switch the training precision to pure FP32 (fp16=False, bf16=False in TrainingArguments).  
* **Robust Fix:** Implement an explicit NaN detection and gradient clipping hook directly in the training loop. This prevents the optimizer from writing poisoned weights into the checkpoint.  
* **Code Example:**  
  Python  
  import math  
  import torch

  def safe\_backward\_step(loss, optimizer, model, max\_norm=1.0):  
      \# 1\. Detect Silent NaNs immediately  
      if math.isnan(loss.item()):  
          raise ValueError("Silent NaN detected in loss. Halting training to prevent checkpoint corruption.")

      loss.backward()

      \# 2\. Explicit gradient clipping for MPS stability  
      torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm)

      optimizer.step()  
      optimizer.zero\_grad()

* **How to Verify:** Inject an artificial NaN into the loss tensor during a dry run. The script should intentionally crash and preserve the last valid checkpoint rather than writing corrupted, NaN-poisoned weights to the disk.

### **4.2 Issue: RuntimeError: MPS backend out of memory**

* **Symptom:** Training crashes mid-epoch with an MPS memory allocation error, despite the macOS Activity Monitor showing ample free unified memory.  
* **Root Cause:** PyTorch implements a strict upper bound on how much unified memory the MPS framework can claim to prevent starving the operating system.6 Furthermore, computational graphs cache memory eagerly and do not always release it back to the OS synchronously during high-throughput loops.  
* **Minimal Fix:** Execute export PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0 in the shell prior to running the training script.7  
* **Robust Fix:** Combine the watermark override with aggressive manual garbage collection inside the epoch loop, explicitly instructing the MPS driver to clear its cache.8  
* **Code Example:**  
  Python  
  import gc  
  import torch  
  import os

  \# Disable the PyTorch artificial memory ceiling  
  os.environ \= "0.0"

  def execute\_epoch\_with\_gc(dataloader, model, optimizer):  
      for step, batch in enumerate(dataloader):  
          \#... forward/backward pass execution...

          \# Aggressive MPS memory reclamation at safe intervals  
          if step % 50 \== 0:  
              gc.collect()  
              torch.mps.empty\_cache()

* **How to Verify:** Implement a logging hook calling torch.mps.current\_allocated\_memory().41 The reported peak memory should display a sawtooth pattern, dropping cleanly to baseline at the end of every collection interval.

### **4.3 Issue: ValueError: Attempting to unscale FP16 gradients**

* **Symptom:** The script crashes during the optimizer.step() phase when mixed-precision (FP16) is enabled via Accelerate or the Hugging Face Trainer.42  
* **Root Cause:** The MPS backend handles sparse gradients and specific element-wise operations differently than CUDA. If a gradient tensor contains uncoalesced sparse values, or if an unsupported operation triggers a CPU fallback during the forward pass, the FP16 Automatic Mixed Precision (AMP) scaler fails to unscale the values properly, leading to a state mismatch.42  
* **Minimal Fix:** Disable mixed precision entirely and train in full FP32.  
* **Robust Fix:** Ensure all target modules are strictly linear. Avoid targeting embedding layers (wte, embed\_tokens) with LoRA if using FP16. If custom loss functions are used, ensure scaler.update() is correctly synchronized and that sparse gradients are manually coalesced before the unscale operation.  
* **Code Example:**  
  Python  
  \# Inside a custom training loop using AMP on MPS  
  scaler \= torch.cuda.amp.GradScaler(enabled=True) \# Works for MPS abstractly

  with torch.autocast(device\_type='mps', dtype=torch.float16):  
      outputs \= model(\*\*batch)  
      loss \= outputs.loss

  scaler.scale(loss).backward()

  \# Check for sparse gradients causing unscale failures  
  for param in model.parameters():  
      if param.grad is not None and param.grad.is\_sparse:  
          if param.grad.dtype \== torch.float16:  
              param.grad \= param.grad.coalesce()

  scaler.step(optimizer)  
  scaler.update()

* **How to Verify:** The optimizer.step() completes consistently across multiple epochs without throwing the unscale error, and the loss curves demonstrate standard convergence.

### **4.4 Issue: Severe Precision Loss Post-Merge (Parity Failure)**

* **Symptom:** Evaluation metrics (such as Word Error Rate for Whisper, or Perplexity for LLMs) degrade significantly immediately after calling model.merge\_and\_unload().  
* **Root Cause:** The mathematical operation required to merge a LoRA adapter involves adding the scaled low-rank matrices to the base weights: W \= W \+ (B @ A) \* (alpha / r).26 Executing this matrix addition directly on the MPS device in FP16 results in severe truncation and rounding errors, as the intermediate B @ A matrix contains wide ranges of values that cause catastrophic cancellation when added to the massive base weight matrix.2  
* **Minimal Fix:** Merge the model using the CPU device rather than MPS.  
* **Robust Fix:** Cast both the base model and the PEFT adapter to FP32, move them entirely to the CPU, perform the merge operation to preserve the mantissa, serialize the result to disk, and then reload the merged model for inference.13  
* **Code Example:**  
  Python  
  import torch  
  from transformers import AutoModelForCausalLM  
  from peft import PeftModel

  \# 1\. Load Base Model to CPU in FP32 strictly  
  base\_model \= AutoModelForCausalLM.from\_pretrained(  
      "base\_model\_path",   
      torch\_dtype=torch.float32,   
      device\_map="cpu"  
  )

  \# 2\. Load Adapter to CPU  
  peft\_model \= PeftModel.from\_pretrained(  
      base\_model,   
      "adapter\_path",   
      torch\_dtype=torch.float32,  
      device\_map="cpu"  
  )

  \# 3\. Merge Weights safely in high precision  
  merged\_model \= peft\_model.merge\_and\_unload()

  \# 4\. Save to safetensors for deployment  
  merged\_model.save\_pretrained("merged\_output\_safe", safe\_serialization=True)

* **How to Verify:** Execute a parity check comparing the logits of the unmerged model against the merged model. The mean absolute error (MAE) between the two output tensors must be less than 1e-5.

### **4.5 Issue: Resumed Checkpoint Divergence (Seq2Seq / Whisper)**

* **Symptom:** Training resumes successfully without explicit errors, but the loss immediately spikes, or the model begins outputting hallucinated garbage text.  
* **Root Cause:** When the Hugging Face Trainer saves a PEFT checkpoint, it serializes only the adapter weights and the adapter configuration (adapter\_model.safetensors, adapter\_config.json).37 It frequently fails to persist the tokenizer state or specific generation configuration parameters—most critically, the decoder\_start\_token\_id.14 Upon resumption, the newly instantiated base model lacks its BOS (Begin-Of-Sentence) token override. Consequently, the seq2seq model begins decoding at an arbitrary token index, misaligning the entire sequence and causing immediate gradient explosions.  
* **Minimal Fix:** Explicitly pass the decoder\_start\_token\_id when defining the Seq2SeqTrainingArguments or when calling model.generate().  
* **Robust Fix:** Hardcode the configuration parameters directly into the model object prior to re-initiating the training loop from the checkpoint.  
* **Code Example:**  
  Python  
  from transformers import WhisperProcessor, WhisperForConditionalGeneration

  processor \= WhisperProcessor.from\_pretrained("openai/whisper-large-v3-turbo")  
  model \= WhisperForConditionalGeneration.from\_pretrained("openai/whisper-large-v3-turbo")

  \# MUST inject these overrides before wrapping with PEFT and resuming  
  model.config.decoder\_start\_token\_id \= processor.tokenizer.bos\_token\_id  
  model.config.pad\_token\_id \= processor.tokenizer.pad\_token\_id

  \# For Whisper, force language tokens to prevent language hallucination  
  model.config.forced\_decoder\_ids \= processor.get\_decoder\_prompt\_ids(  
      language="en", task="transcribe"  
  )

  \# Now wrap and resume  
  model \= get\_peft\_model(model, peft\_config)  
  trainer.train(resume\_from\_checkpoint="path/to/peft/checkpoint")

* **How to Verify:** Inspect the first batch of generated predictions after resumption. The output sequence should correctly begin with the \<|startoftranscript|\> and \<|en|\> tokens, and the loss should continue smoothly from the previous epoch's final value.

### **4.6 Issue: Generation Crashes with Gradient Checkpointing**

* **Symptom:** Calling model.generate() during the evaluation phase of a training loop results in an AttributeError, specifically stating that the model or wrapper lacks the gradient\_checkpointing\_disable method.21  
* **Root Cause:** The unwrap\_model\_for\_generation utility inside the Trainer attempts to temporarily disable gradient checkpointing to allow autoregressive generation. However, PEFT wrappers (like PolicyAndValueWrapper in TRL or complex PeftModel instances) copy the is\_gradient\_checkpointing=True flag but fail to implement the actual enabling/disabling methods, causing the unwrapper to crash.21  
* **Minimal Fix:** Disable gradient checkpointing entirely if memory permits.  
* **Robust Fix:** Manually patch the is\_gradient\_checkpointing flag to False immediately before the evaluation loop, or implement a custom evaluation step that relies on the base model's generate method rather than the wrapped model's generate method.

### **4.7 Issue: Trainer vs. Manual Loop Memory Leaks**

* **Symptom:** Training with the Hugging Face Trainer OOMs consistently on MPS after exactly one epoch, while a manual PyTorch loop with the exact same batch size completes successfully.  
* **Root Cause:** The HF Trainer relies on PyTorch's native memory allocators. On CUDA, the caching allocator is highly mature and recycles memory automatically. On MPS, the allocator is less mature and frequently holds onto graph states until the Python process terminates. The Trainer does not aggressively call torch.mps.empty\_cache().10  
* **Minimal Fix:** Switch to a fully manual training loop.  
* **Robust Fix:** If the Trainer must be used, implement a custom Hugging Face TrainerCallback that hooks into the on\_step\_end and on\_epoch\_end events to force garbage collection.  
* **Code Example:**  
  Python  
  from transformers import TrainerCallback  
  import torch  
  import gc

  class MPSMemoryCallback(TrainerCallback):  
      def on\_step\_end(self, args, state, control, \*\*kwargs):  
          if state.global\_step % 50 \== 0:  
              gc.collect()  
              if torch.backends.mps.is\_available():  
                  torch.mps.empty\_cache()

      def on\_epoch\_end(self, args, state, control, \*\*kwargs):  
          gc.collect()  
          if torch.backends.mps.is\_available():  
              torch.mps.empty\_cache()

  \# Pass to trainer  
  trainer \= Trainer(..., callbacks=)

* **How to Verify:** The Trainer completes multiple epochs without crashing, and system memory swap usage remains stable.

## **5\. Best Practices**

* **Rank (r) and Alpha Scaling Tradeoffs:** On Apple Silicon memory budgets, avoid excessively high ranks. Typical values of r=8 or r=16 paired with lora\_alpha=32 provide excellent representational capacity without overwhelming the unified memory bandwidth.2 To further stabilize training, apply Rank-Stabilized LoRA by setting use\_rslora=True. This adjusts the scaling factor mathematically to lora\_alpha/math.sqrt(r) rather than lora\_alpha/r, enhancing convergence stability across deep transformer layers.12  
* **Dropout on MPS:** Standardize your configuration to lora\_dropout=0.05. Setting dropout to exactly 0.0 can trigger specific, optimized fast-paths in certain model architectures (such as DoRA), which may behave unpredictably or bypass essential safety checks on the MPS backend.11  
* **Naming, Packaging, and Artifact Hygiene:** When saving model artifacts, always utilize safe\_serialization=True to output .safetensors files, ensuring deployment safety and avoiding pickle-based vulnerabilities.13 Maintain strict directory separation between the base\_model, the adapter\_checkpoint, and the merged\_final\_model. Never overwrite the base model directory, as this corrupts the ability to verify parity later.  
* **Custom Data Collators:** When training Whisper or other speech models, the data collator must dynamically pad audio features (to 3000 frames) and text labels. Ensure padding tokens in the labels are replaced with \-100 so the PyTorch CrossEntropyLoss algorithm ignores them correctly.27  
* **Hardware Fallback Prudence:** Only utilize PYTORCH\_ENABLE\_MPS\_FALLBACK=1 when an explicit NotImplementedError is thrown by the MPS backend for a specific, isolated operator.19 Do not leave this flag enabled globally in production training pipelines, as silent fallbacks to the CPU will decimate your training throughput and mask underlying architectural incompatibilities.

## **6\. Worst Practices and Anti-Patterns**

* **Anti-Pattern 1: Relying on BFloat16 (BF16).** Do not under any circumstances use bf16=True on Apple Silicon. Despite PyTorch allowing the flag and executing the code without throwing a syntax error, M-series hardware lacks native BF16 instructions. It will emulate the format entirely in software, resulting in severe rounding errors and creating a massive correctness gap between your local development environment and cloud deployment.1  
* **Anti-Pattern 2: Profiling with detect\_anomaly.** Using torch.autograd.set\_detect\_anomaly(True) on an MPS device causes the graph execution to slow down by up to 100x.1 It renders training pipelines functionally frozen and is not a viable method for tracing gradient issues on macOS. Rely on explicit tensor checking instead.  
* **Anti-Pattern 3: Saving Full Models During PEFT.** Overriding PEFT's default saving mechanisms to write out the entire 13GB+ model at every evaluation step exhausts SSD write-cycles and storage rapidly. Trust the PEFT framework to save only the lightweight adapter weights (adapter\_model.safetensors), which are typically under 100MB.37  
* **Anti-Pattern 4: Merging in Evaluation Loops.** Attempting to dynamically merge and unmerge LoRA weights during the validation phase of a training loop to test "true" inference performance will continuously trigger memory fragmentation and eventual OOM crashes on MPS. Evaluate using the active, separate adapter state during training, and only merge once the final model is selected.

## **7\. Weird Patterns That Work Surprisingly Well on Apple Silicon**

* **The FP32 CPU Shuffle:** When encountering unsolvable NaNs or persistent unscale errors on MPS during complex, custom loss calculations (such as contrastive loss for audio embeddings), a weird but highly effective pattern is transferring the logits and labels to the CPU via .to("cpu"), calculating the loss in FP32, computing the backward pass on the CPU, and then transferring the gradients back to the MPS device before calling optimizer.step().  
* **Deduplicating Target Modules by Regex and Types:** Instead of listing exhaustive strings (which frequently break across different Hugging Face architecture updates), using Python set comprehensions to dynamically probe the model architecture and find all linear layers works perfectly. This bypasses hardcoded dictionary structures and ensures all relevant feed-forward and attention networks are adapted.29  
* **Forcing Language Tokens unconditionally:** PEFT models often lose the ability to auto-detect language during decoding because the base weights are frozen and the adapter is highly specialized. Forcing the starting tokens (forced\_decoder\_ids) unconditionally during model.generate() instantly restores language generation stability, bypassing the degraded classification head.25

## **8\. Limitations and Unsolved Problems**

* **The Emulation Trap and Tensor Cores:** Apple Silicon fundamentally lacks the equivalent of NVIDIA's Tensor Cores engineered specifically for FP8 and FP4 mathematics. While the new mps-bitsandbytes library implements Metal kernels for these operations, they are executed via dynamic codebook quantization.17 Optimization efforts achieved locally on Mac will not directly translate to 1:1 throughput gains on an NVIDIA H100 or B200.  
* **Distributed Training Isolation:** Prototyping distributed training logic (using torch.distributed with the gloo backend) locally on a Mac does not simulate the realities of high-speed interconnects like NVLink. There is a steep "interconnect cliff" when moving from an MPS environment to a multi-node datacenter setup, and throughput calculations will not scale linearly.1  
* **8-bit Optimizer Precision Failures:** Even with native Metal support via mps-bitsandbytes, 8-bit optimizers (such as 8-bit Adam or RMSprop) exhibit known relative error tolerance failures on MPS (e.g., relerr \> 0.0016). They also suffer from hardcoded blocksizes that deviate from standard CUDA implementations, leading to subtle gradient divergence over long training runs.17

## **9\. Copy-Paste Reference Snippets**

### **9.1 Whisper-Specific LoRA Configuration and Target Selection**

This script demonstrates how to dynamically target all cross-attention and linear layers in a Whisper model, avoiding hardcoded string lists that inevitably break across library versions.

Python

import torch  
from transformers import WhisperForConditionalGeneration  
from peft import LoraConfig, get\_peft\_model

model\_id \= "openai/whisper-large-v3-turbo"  
model \= WhisperForConditionalGeneration.from\_pretrained(  
    model\_id,   
    torch\_dtype=torch.float32, \# Ensure safe MPS baseline  
    device\_map="mps"  
)

\# Dynamically target all linear projection layers (attention \+ FFN)  
\# Crucial: Avoid targeting embeddings or convolutional layers  
target\_modules \= \[  
    name for name, module in model.named\_modules()  
    if isinstance(module, torch.nn.Linear) and ("proj" in name or "fc" in name)  
\]

peft\_config \= LoraConfig(  
    r=16,  
    lora\_alpha=32,  
    target\_modules=target\_modules, \# Resolves dynamically to \["q\_proj", "v\_proj", "fc1", "fc2",...\]  
    lora\_dropout=0.05,  
    bias="none",  
    task\_type="SEQ\_2\_SEQ\_LM"  
)

model \= get\_peft\_model(model, peft\_config)  
model.print\_trainable\_parameters()

### **9.2 Apple-Silicon Memory-Safe Manual Training Loop**

This example demonstrates a manual training loop implementing explicit memory management, garbage collection, and safety bounds specifically tailored for MPS hardware.

Python

import gc  
import math  
import torch  
import os

\# Crucial memory overrides for large models  
os.environ \= "0.0"

def train\_epoch\_mps\_safe(model, dataloader, optimizer):  
    model.train()  
    total\_loss \= 0  
      
    for step, batch in enumerate(dataloader):  
        \# Move inputs explicitly to the MPS device  
        input\_features \= batch\["input\_features"\].to("mps")  
        labels \= batch\["labels"\].to("mps")  
          
        outputs \= model(input\_features=input\_features, labels=labels)  
        loss \= outputs.loss  
          
        \# Immediate Silent NaN Prevention Hook  
        if math.isnan(loss.item()):  
            print(f"CRITICAL: Silent NaN detected at step {step}. Halting to preserve checkpoint.")  
            break  
              
        loss.backward()  
          
        \# Explicit gradient clipping to prevent overflow  
        torch.nn.utils.clip\_grad\_norm\_(model.parameters(), 1.0)  
          
        optimizer.step()  
        optimizer.zero\_grad()  
        total\_loss \+= loss.item()  
          
        \# Aggressive memory reclamation at safe, synchronous intervals  
        if step % 50 \== 0:  
            gc.collect()  
            torch.mps.empty\_cache()  
              
    return total\_loss / len(dataloader)

### **9.3 Evaluation Parity: Adapter-Attached vs. Merged Model**

This script provides the mathematical verification workflow required to prove that a LoRA adapter has been merged without precision degradation on Apple hardware.

Python

import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from peft import PeftModel

\# Prepare dummy input for evaluation  
tokenizer \= AutoTokenizer.from\_pretrained("base\_model\_path")  
inputs \= tokenizer("Verify adapter parity rigorously.", return\_tensors="pt")

\# \--- Phase 1: Generate Logits from Unmerged Model \---  
\# Load base model to CPU to establish the baseline truth  
base\_model \= AutoModelForCausalLM.from\_pretrained(  
    "base\_model\_path",   
    torch\_dtype=torch.float32,  
    device\_map="cpu"  
)  
peft\_model \= PeftModel.from\_pretrained(base\_model, "adapter\_path")  
peft\_model.eval()

with torch.no\_grad():  
    logits\_unmerged \= peft\_model(\*\*inputs).logits

\# \--- Phase 2: Merge Safely on CPU \---  
\# Using the CPU explicitly prevents MPS FP16 truncation errors during matrix addition  
merged\_model \= peft\_model.merge\_and\_unload()  
merged\_model.eval()

\# \--- Phase 3: Generate Logits from Merged Model \---  
with torch.no\_grad():  
    logits\_merged \= merged\_model(\*\*inputs).logits

\# \--- Phase 4: Verify Parity \---  
diff \= torch.abs(logits\_unmerged \- logits\_merged).max().item()  
mean\_diff \= torch.abs(logits\_unmerged \- logits\_merged).mean().item()

print(f"Max absolute difference: {diff}")  
print(f"Mean absolute difference: {mean\_diff}")

if diff \< 1e-4:  
    print("SUCCESS: Merged model parity verified. Safe to export to GGUF/CoreML.")  
else:  
    print("WARNING: Significant precision loss detected during the merge operation.")

## **10\. Final "Red Flags" Checklist Before Shipping**

Before taking a PEFT model trained on Apple Silicon and pushing it to a deployment pipeline (whether cloud inference or on-device CoreML), verify the following conditions to prevent catastrophic runtime failures:

* \[ \] **The BF16 Audit:** Grep the training configuration, TrainingArguments, and underlying codebase to ensure bf16=True is completely disabled. Confirm that all artifacts and checkpoints were serialized via FP32 or FP16 logic.  
* \[ \] **Decoder Tokens Configured:** Confirm that model.config.decoder\_start\_token\_id is explicitly set in the final config.json inside the artifact directory. Do not assume the downstream inference engine or generation script will default to it correctly.  
* \[ \] **Merged on CPU:** Confirm that the final merge\_and\_unload() sequence was executed on the CPU using FP32 precision. If the build logs indicate it was executed on device="mps", recalculate the merge immediately to prevent silent rounding errors from shipping.  
* \[ \] **Safetensors Exclusivity:** Verify the output artifact directory contains model.safetensors and adapter\_model.safetensors, and strictly lacks any pytorch\_model.bin (pickle files) to ensure loading security.  
* \[ \] **CoreML State Isolation:** If exporting the final model to CoreML for iOS/macOS deployment using coremltools, ensure dynamic states (like KV-caches) were cleanly isolated during the tracing step, and that the PEFT adapter was entirely merged into the base weights prior to attempting the conversion.33  
* \[ \] **Quantization Dependency Cleared:** If deploying the adapted model to a different hardware architecture, ensure any local hacks relying on mps-bitsandbytes or torchao fallbacks are stripped from the inference code, preventing runtime ImportError failures on non-Apple hardware.

#### **Works cited**

1. State of PyTorch Hardware Acceleration 2025, accessed April 2, 2026, [https://tunguz.github.io/PyTorch\_Hardware\_2025/](https://tunguz.github.io/PyTorch_Hardware_2025/)  
2. How to Fine-Tune LLMs Locally: The Complete LoRA Guide \- Medium, accessed April 2, 2026, [https://medium.com/@matteo28/how-to-fine-tune-llms-locally-the-complete-lora-guide-420fe9e1278d](https://medium.com/@matteo28/how-to-fine-tune-llms-locally-the-complete-lora-guide-420fe9e1278d)  
3. Defeating the Training-Inference Mismatch via FP16 \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2510.26788v1](https://arxiv.org/html/2510.26788v1)  
4. Reflection and the Never-Ending Confusion Between FP16 and BF16 \- Reddit, accessed April 2, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1fcjtpo/reflection\_and\_the\_neverending\_confusion\_between/](https://www.reddit.com/r/LocalLLaMA/comments/1fcjtpo/reflection_and_the_neverending_confusion_between/)  
5. Stable Diffusion XL 1.0 model, accessed April 2, 2026, [https://stable-diffusion-art.com/sdxl-model/](https://stable-diffusion-art.com/sdxl-model/)  
6. MPS memory issue, MPS backend out of memory, but works if I empty the MPS cache \#105839 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/105839](https://github.com/pytorch/pytorch/issues/105839)  
7. Fine Tuning/GGML Quantiziation on Apple Silicon Guide : r/LocalLLaMA \- Reddit, accessed April 2, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/15y9m64/fine\_tuningggml\_quantiziation\_on\_apple\_silicon/](https://www.reddit.com/r/LocalLLaMA/comments/15y9m64/fine_tuningggml_quantiziation_on_apple_silicon/)  
8. Python Archives \- Satyaki De's Blog, accessed April 2, 2026, [https://satyakide.com/tag/python/](https://satyakide.com/tag/python/)  
9. hv\_train\_network.py · svjack/Yato\_wan\_2\_1\_1\_3\_B\_text2video\_lora at main, accessed April 2, 2026, [https://huggingface.co/svjack/Yato\_wan\_2\_1\_1\_3\_B\_text2video\_lora/blob/main/hv\_train\_network.py](https://huggingface.co/svjack/Yato_wan_2_1_1_3_B_text2video_lora/blob/main/hv_train_network.py)  
10. transformers/src/transformers/trainer\_utils.py at main · huggingface/transformers \- GitHub, accessed April 2, 2026, [https://github.com/huggingface/transformers/blob/main/src/transformers/trainer\_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py)  
11. LoRA \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/peft/developer\_guides/lora](https://huggingface.co/docs/peft/developer_guides/lora)  
12. LoRA \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/peft/package\_reference/lora](https://huggingface.co/docs/peft/package_reference/lora)  
13. Fine-Tuning Mistral-7B on Apple Silicon: A Mac User's Journey with Axolotl & LoRA, accessed April 2, 2026, [https://medium.com/@plawanrath/fine-tuning-mistral-7b-on-apple-silicon-a-mac-users-journey-with-axolotl-lora-c6ff53858e7d](https://medium.com/@plawanrath/fine-tuning-mistral-7b-on-apple-silicon-a-mac-users-journey-with-axolotl-lora-c6ff53858e7d)  
14. 3.35 MB \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/mishig/llms-txt/resolve/main/transformers.txt?download=true](https://huggingface.co/mishig/llms-txt/resolve/main/transformers.txt?download=true)  
15. Content \- 4351ecffd10f629db4942a2c5dbe64aa967cac7b \- ccb490f/examples/pytorch/summarization/run\_summarization\_no\_trainer.py \- the Software Heritage archive, accessed April 2, 2026, [https://archive.softwareheritage.org/browse/content/sha1\_git:4351ecffd10f629db4942a2c5dbe64aa967cac7b/?branch=refs/heads/inherited\_causal\_lm\_tests\&path=examples/pytorch/summarization/run\_summarization\_no\_trainer.py\&snapshot\_id=b0e644832319d05788a44e69f94a2fd247b4a7d6](https://archive.softwareheritage.org/browse/content/sha1_git:4351ecffd10f629db4942a2c5dbe64aa967cac7b/?branch=refs/heads/inherited_causal_lm_tests&path=examples/pytorch/summarization/run_summarization_no_trainer.py&snapshot_id=b0e644832319d05788a44e69f94a2fd247b4a7d6)  
16. mps-bitsandbytes \- PyPI, accessed April 2, 2026, [https://pypi.org/project/mps-bitsandbytes/](https://pypi.org/project/mps-bitsandbytes/)  
17. Add MPS backend for Apple Silicon by imperatormk · Pull Request ..., accessed April 2, 2026, [https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853)  
18. Apple Silicon vs NVIDIA CUDA: AI Comparison 2025, Benchmarks, Advantages and Limitations \- Consultant freelance Jean-Jerome Levy, accessed April 2, 2026, [https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/](https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/)  
19. Fundamentals: Chapter 0, accessed April 2, 2026, [https://arena-chapter0-fundamentals.streamlit.app/](https://arena-chapter0-fundamentals.streamlit.app/)  
20. Trainer \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/transformers/v4.33.0/main\_classes/trainer](https://huggingface.co/docs/transformers/v4.33.0/main_classes/trainer)  
21. \[PPO\] 'PolicyAndValueWrapper' has no attribute 'gradient\_checkpointing\_disable' in unwrap\_model\_for\_generation · Issue \#4954 · huggingface/trl \- GitHub, accessed April 2, 2026, [https://github.com/huggingface/trl/issues/4954](https://github.com/huggingface/trl/issues/4954)  
22. LLM Fine-Tuning with PEFT: Example Application | by Sulbha Jain | Medium, accessed April 2, 2026, [https://sulbhajain.medium.com/llm-fine-tuning-with-peft-a6bdfe789323](https://sulbhajain.medium.com/llm-fine-tuning-with-peft-a6bdfe789323)  
23. Parameter Efficient Fine Tuning (PEFT) Techniques for Large Models | by Shashank Guda, accessed April 2, 2026, [https://shashankguda.medium.com/parameter-efficient-fine-tuning-peft-techniques-for-large-models-e536726e12c2](https://shashankguda.medium.com/parameter-efficient-fine-tuning-peft-techniques-for-large-models-e536726e12c2)  
24. Parameter-Efficient Fine-Tuning on Multilingual ASR Whisper Model for Frisian \- Studenttheses Campus Fryslan, accessed April 2, 2026, [https://campus-fryslan.studenttheses.ub.rug.nl/539/1/MA-5521904-X-Liu.pdf](https://campus-fryslan.studenttheses.ub.rug.nl/539/1/MA-5521904-X-Liu.pdf)  
25. Vaibhavs10/fast-whisper-finetuning \- GitHub, accessed April 2, 2026, [https://github.com/Vaibhavs10/fast-whisper-finetuning](https://github.com/Vaibhavs10/fast-whisper-finetuning)  
26. LoRA-Finetuned Whisper ASR \- Emergent Mind, accessed April 2, 2026, [https://www.emergentmind.com/topics/lora-finetuned-whisper](https://www.emergentmind.com/topics/lora-finetuned-whisper)  
27. Fine-tuning Whisper with LoRA \- Medium, accessed April 2, 2026, [https://medium.com/@anitaliubfsu/fine-tuning-whisper-with-lora-c796781f00f5](https://medium.com/@anitaliubfsu/fine-tuning-whisper-with-lora-c796781f00f5)  
28. Fork of Fine-tuning\_whisper\_with\_LoRA \- Kaggle, accessed April 2, 2026, [https://www.kaggle.com/code/davidramos18/fork-of-fine-tuning-whisper-with-lora](https://www.kaggle.com/code/davidramos18/fork-of-fine-tuning-whisper-with-lora)  
29. Target modules for applying PEFT / LoRA on different models \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models](https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models)  
30. Low performance on mps backed · Issue \#2041 · huggingface/peft, accessed April 2, 2026, [https://github.com/huggingface/peft/issues/2041](https://github.com/huggingface/peft/issues/2041)  
31. Fine-tuning only Whisper decoder · openai whisper · Discussion \#1707 \- GitHub, accessed April 2, 2026, [https://github.com/openai/whisper/discussions/1707](https://github.com/openai/whisper/discussions/1707)  
32. Parameter-Efficient Fine-Tuning of DINOv2 for Large-Scale Font Classification \- arXiv.org, accessed April 2, 2026, [https://arxiv.org/html/2602.13889v1](https://arxiv.org/html/2602.13889v1)  
33. Converting Models to Core ML \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/blog/fguzman82/frompytorch-to-coreml](https://huggingface.co/blog/fguzman82/frompytorch-to-coreml)  
34. Multifunction Models — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html)  
35. Oute\_TTS\_(1B).ipynb \- Colab \- Google, accessed April 2, 2026, [https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Oute\_TTS\_(1B).ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Oute_TTS_\(1B\).ipynb)  
36. bitsandbytes \- PyPI, accessed April 2, 2026, [https://pypi.org/project/bitsandbytes/](https://pypi.org/project/bitsandbytes/)  
37. Parameter-efficient fine-tuning · Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/transformers/peft](https://huggingface.co/docs/transformers/peft)  
38. \`skip\_memory\_metrics=False\` breaks training loop when on MPS device · Issue \#27181 · huggingface/transformers \- GitHub, accessed April 2, 2026, [https://github.com/huggingface/transformers/issues/27181](https://github.com/huggingface/transformers/issues/27181)  
39. Fast Whisper-Large-v2 Fine-Tuning with LoRA \- Kaggle, accessed April 2, 2026, [https://www.kaggle.com/code/imtiazprio/fast-whisper-large-v2-fine-tuning-with-lora](https://www.kaggle.com/code/imtiazprio/fast-whisper-large-v2-fine-tuning-with-lora)  
40. GitHub \- huggingface/peft: PEFT: State-of-the-art Parameter-Efficient Fine-Tuning., accessed April 2, 2026, [https://github.com/huggingface/peft](https://github.com/huggingface/peft)  
41. torch.mps.profiler. \- PyTorch documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html](https://docs.pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html)  
42. kye/all-conceptofmind-code · Datasets at Hugging Face, accessed April 2, 2026, [https://huggingface.co/datasets/kye/all-conceptofmind-code/viewer/default/train?p=3](https://huggingface.co/datasets/kye/all-conceptofmind-code/viewer/default/train?p=3)  
43. Issues when using HuggingFace \`accelerate\` with \`fp16\` \- Stack Overflow, accessed April 2, 2026, [https://stackoverflow.com/questions/75802877/issues-when-using-huggingface-accelerate-with-fp16](https://stackoverflow.com/questions/75802877/issues-when-using-huggingface-accelerate-with-fp16)  
44. Understanding PEFT — Using Adapter Techniques | by Adrien Riaux \- Medium, accessed April 2, 2026, [https://medium.com/@adrien.riaux/understanding-peft-using-adapter-techniques-16b8e9152759](https://medium.com/@adrien.riaux/understanding-peft-using-adapter-techniques-16b8e9152759)  
45. TowardsDataScience-2023-博客中文翻译-十九- \- 绝不原创的飞龙, accessed April 2, 2026, [https://www.cnblogs.com/apachecn/p/18461379](https://www.cnblogs.com/apachecn/p/18461379)  
46. Interactive Guide: Train an LLM From Scratch, accessed April 2, 2026, [https://www.e-accelerate.com/](https://www.e-accelerate.com/)  
47. LLM Fine-Tuning on a Budget: Top FAQs on Adapters, LoRA, and Other Parameter-Efficient Methods \- Runpod, accessed April 2, 2026, [https://www.runpod.io/articles/guides/llm-fine-tuning-on-a-budget-top-faqs-on-adapters-lora-and-other-parameter-efficient-methods](https://www.runpod.io/articles/guides/llm-fine-tuning-on-a-budget-top-faqs-on-adapters-lora-and-other-parameter-efficient-methods)  
48. CoreML On | Skills Marketplace \- LobeHub, accessed April 2, 2026, [https://lobehub.com/skills/comeonoliver-skillshub-coreml](https://lobehub.com/skills/comeonoliver-skillshub-coreml)