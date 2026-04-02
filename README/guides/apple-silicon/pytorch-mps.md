# **Apple Silicon Production Field Guide: PyTorch MPS and CoreML Workflows for Speech Models**

April 2, 2026

## **Executive Summary**

The transition from CUDA-first or CPU-first PyTorch workflows to Apple Silicon’s Metal Performance Shaders (MPS) requires a fundamental rewiring of engineering assumptions. The architecture relies on unified memory, strict but often silent operational fallbacks, and a highly specific set of supported hardware precision formats. For teams fine-tuning Whisper, managing LoRA ranks, or experimenting with Mamba-style state space models, the following practical realities govern MPS execution in 2026:

* **Float16 is Mandatory; Bfloat16 is a Trap**: Despite industry trends favoring bfloat16 for gradient stability, MPS execution of bfloat16 relies on unoptimized pathways that can degrade throughput by up to 10x compared to float16.1  
* **The Silent NaN Contiguity Bug**: PyTorch operations like addcmul\_ and addcdiv\_ will silently fail—leaving tensors populated with zeros or NaNs—if the underlying tensors are not contiguous in memory. This is the primary cause of stalled training loss on MPS.4  
* **Unified Memory Illusion**: Apple Silicon shares RAM between the CPU and GPU. Applying CUDA-style VRAM instincts will cause system-wide crashes. The PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO environment variable is the only defense against hard out-of-memory (OOM) kernel panics.6  
* **Zero-Worker Dataloaders**: Due to macOS process spawning mechanics, setting num\_workers \> 0 in PyTorch DataLoader instances typically destroys performance. Default to num\_workers=0 for standard speech batches.8  
* **Explicit CPU Fallbacks**: Missing MPS operators must be explicitly routed to the CPU using PYTORCH\_ENABLE\_MPS\_FALLBACK=1, but relying on this global flag within the inner training loop creates catastrophic synchronization bottlenecks.6  
* **CoreML for Encoders, MPS for Decoders**: The Apple Neural Engine (ANE) heavily penalizes dynamic shapes and autoregressive decoding. Whisper’s audio encoder should be exported to CoreML, while the text decoder remains on MPS.10  
* **Beware Scaled Dot-Product Attention (SDPA)**: Long-sequence attention on MPS attempts to allocate massive contiguous buffers, frequently triggering Metal assertion failures. Explicit chunking is required for sequences exceeding 12,000 tokens.13  
* **Avoid Float64 Completely**: Double-precision operations lack native hardware support on Apple Silicon GPUs and will either fail or silently cast, corrupting gradient accumulation.14  
* **Mamba / Custom Ops Require Hybrid Execution**: The selective scan operation native to Mamba lacks Triton/MPS kernels. It must be executed via localized CPU fallback or unrolled in plain PyTorch.15  
* **Graph Compilation is Immature**: torch.compile on MPS remains unstable for complex dynamic-graph models. It frequently triggers recompilations or raises NotImplementedError during export.17  
* **Distillation Demands Memory Asymmetry**: When performing teacher-student distillation, load the frozen teacher on the CPU and the training student on MPS. Loading both on MPS will immediately exhaust the unified memory high watermark.19  
* **Determinism is Expensive**: Reproducibility requires torch.use\_deterministic\_algorithms(True), but this disables crucial MPS optimizations and drastically reduces training throughput.21  
* **Garbage Collection is Manual**: Unlike CUDA, MPS relies heavily on the OS pager. Interleaving torch.mps.empty\_cache() between large batch epochs is a required pattern for sustained training loops.22  
* **LoRA is the Standard for Local Fine-Tuning**: Full parameter fine-tuning of Whisper Large-v3 will exceed Apple Silicon memory bandwidth. LoRA reduces memory requirements by a factor of 3 and is fully supported by the MPS backend.24  
* **Validation Requires Parity Checks**: Never assume an MPS model is mathematically equivalent to its CPU counterpart without running torch.testing.assert\_close across the first training step.25

## **Decision Framework**

Selecting the correct execution backend on Apple Silicon determines the ceiling for both iteration speed and deployment viability. Engineers must route workloads based on the architectural characteristics of the model and the phase of the development lifecycle.

| Workload Profile | Primary Backend Strategy | Rationale & Tradeoffs |
| :---- | :---- | :---- |
| **Whisper LoRA Fine-Tuning** | PyTorch MPS (Native) | MPS provides sufficient acceleration for the matrix multiplications heavily utilized in LoRA. Avoids the overhead of exporting to MLX. Memory is managed via strict batch sizing and 8-bit quantization where applicable. 24 |
| **Mamba / SSM Training** | Hybrid (MPS \+ CPU Fallback) | Triton kernels for Mamba's selective scan do not compile on Metal. The backbone runs on MPS, but the scan operation falls back to CPU. This introduces a severe latency penalty but ensures mathematical correctness. 16 |
| **Teacher-Student Distillation** | Hybrid Memory Split | The teacher model resides on cpu to preserve the unified memory pool allocated to the GPU allocator, while the student model resides on mps. Activations are moved across the bus per step. 20 |
| **Production ASR Inference** | MLX | MLX provides superior inference latency on Apple Silicon by utilizing native Metal optimizations that bypass PyTorch overhead. Requires converting standard checkpoints to MLX arrays. 28 |
| **On-Device Mobile Deployment** | CoreML (ANE) | Exporting the Whisper encoder to CoreML allows it to run on the Apple Neural Engine. This provides massive power efficiency gains, but strictly requires static shapes and locks the model into the Apple ecosystem. 12 |

### **When to use plain PyTorch on MPS**

Use plain PyTorch MPS for standard Transformer architectures where all operators are officially supported in the PyTorch Metal backend. This includes standard multi-head attention, linear layers, and convolutions. It is the optimal path for research, prototyping, and LoRA fine-tuning where rapid iteration is prioritized over raw hardware utilization. The integration with standard Hugging Face workflows allows engineers to rely on pre-existing pipelines without rewriting model architectures into a new framework.24

### **When to use hybrid MPS \+ CPU fallback**

Deploy a hybrid strategy when a model requires custom CUDA kernels (e.g., FlashAttention, Mamba SSMs) or unsupported loss functions. By isolating the unsupported operations within custom nn.Module wrappers, the bulk of the computational graph remains accelerated by MPS, while the unsupported node calculates its forward and backward pass on the CPU.16 This is a necessary compromise for novel architectures that outpace Apple's kernel development timeline.

### **When to use MLX instead**

Transition to Apple's MLX framework when inference throughput is the primary objective. PyTorch MPS introduces significant framework overhead and struggles with specific gradient scatter/gather operations. MLX is designed natively for unified memory, features lazy computation, and routinely outperforms PyTorch MPS for pure autoregressive text generation and speech decoding. However, adopting MLX requires rewriting custom layers and adopting a NumPy-style functional API, which can increase developer friction.28

### **When to export to CoreML**

Export to CoreML only for the final deployment phase targeting macOS or iOS applications. CoreML enables access to the Apple Neural Engine (ANE), a specialized matrix math coprocessor that provides massive power-efficiency gains over the standard GPU. However, CoreML severely limits dynamic tensor shaping. It should be strictly reserved for the static-shape components of a model, such as the Whisper Audio Encoder, which processes fixed 30-second audio chunks.11

## **Migration Playbook**

Converting a CUDA-first repository to an Apple Silicon-ready codebase requires systematic auditing and phased implementation. The unified memory architecture fundamentally alters how memory bottlenecks present themselves during training.

### **Audit checklist for an existing PyTorch codebase**

1. **Rosetta Emulation Check**: If the system reports that the MPS accelerator is unavailable despite running on M-series hardware, the Python executable is running under x86 emulation via Rosetta. Native arm64 Python must be installed. Emulated Python will permanently lock PyTorch to the CPU.36  
2. **Dtype Audit**: Search the codebase for torch.bfloat16 and torch.float64. These must be conditionally replaced with torch.float16 and torch.float32 respectively when running on MPS. The M-series GPU hardware lacks native pathways for these formats, leading to emulation overhead or outright crashes.1  
3. **Contiguous Tensor Audit**: Search for .T, .transpose(), .view(), and .expand(). These operations create non-contiguous memory layouts. On older macOS versions and PyTorch builds, passing non-contiguous tensors to optimizers triggers silent gradient failures. Append .contiguous() where applicable.4  
4. **Dataloader Audit**: Locate all DataLoader instantiations. Verify that num\_workers is not hardcoded to high values (e.g., 8 or 16). On macOS, multiprocessing relies on spawn, which incurs massive overhead for initializing new Python interpreters.8  
5. **Unsupported Ops Audit**: Identify custom C++/CUDA extensions, Triton kernels, or experimental PyTorch operators. These will require isolation and explicit CPU fallback wrappers.15

### **Ordered migration steps**

**Step 1: Dynamic Device Detection**

Implement a centralized device resolution function that gracefully falls back from CUDA to MPS to CPU. This ensures cross-platform compatibility without branching logic scattered throughout the repository. Hardcoding device="mps" is an anti-pattern that breaks continuous integration (CI) pipelines running on Linux runners.

**Step 2: Environment Configuration** Inject the necessary environment variables at the application entry point before torch is imported. Set PYTORCH\_ENABLE\_MPS\_FALLBACK=1 to prevent immediate crashes upon encountering unsupported ops. Tune PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO to 0.85 or 0.9 to give the OS pager breathing room and prevent hard system panics during peak memory allocation.6

**Step 3: Precision Alignment** Implement a precision abstraction layer. When the device is MPS, force the Automatic Mixed Precision (AMP) autocast context to utilize float16. Disable bfloat16 entirely for the MPS execution path to prevent the 10x throughput collapse observed on consumer M-series chips.2

**Step 4: Memory-Aware Training Loops** Modify the training loop to explicitly clear the MPS cache. Apple Silicon's unified memory does not behave like discrete VRAM; the MPSAllocator pools memory and relies on aggressive garbage collection. Inject torch.mps.empty\_cache() between validation and training phases, and after any large tensor deallocations.22 Ensure that accumulated losses use .item() to extract the float value, rather than accumulating the tensor graph itself, which will rapidly trigger an OOM error.38

### **Validation checklist after each step**

* Run a single forward pass on the CPU and the exact same pass on MPS. Use torch.testing.assert\_close(cpu\_out, mps\_out.cpu(), rtol=1e-3, atol=1e-3) to verify numerical parity.25  
* Monitor memory utilization using torch.mps.current\_allocated\_memory() during the first epoch to ensure the high watermark is not breached.39  
* Verify that the training loss decreases over 50 steps. A stalled loss curve immediately indicates a non-contiguous tensor bug in the optimizer state.5

## **Failure Modes Catalog**

The PyTorch MPS backend contains unique, often poorly documented failure modes that rarely surface on CUDA hardware. The following catalog addresses the most critical issues encountered in speech and sequential modeling on Apple Silicon.

### **Issue 1: The Silent Optimizer Freeze (Non-Contiguous Gradient Bug)**

* **Symptom**: The training loss plateaus instantly or behaves erratically. The model appears to execute normally, and throughput is stable, but specific weights (often encoder weights or transposed projections) do not update. No error is thrown to the console.  
* **Root Cause**: PyTorch MPS backend operations addcmul\_ and addcdiv\_—which serve as the core mathematical components of the Adam and AdamW optimizers—fail silently when writing to non-contiguous output tensors. If a weight is initialized as a transpose (e.g., encoder.weight \= decoder.weight.T.clone()), the optimizer state inherits this non-contiguous layout. The weight update is simply bypassed by the underlying Metal kernel, creating a silent failure where the model appears to learn but stalls because an entire block of weights remains frozen.4  
* **Minimal Fix**: Ensure all parameters are explicitly made contiguous before passing them to the optimizer.  
* **Robust Fix**: Subclass or wrap the optimizer initialization to enforce contiguity checks across the state\_dict, or mandate an upgrade to PyTorch ![][image1] 2.4 and macOS 15+ where the underlying Metal bug was officially addressed by Apple.5  
* **Code Example**:  
  Python  
  \# BAD: Inherits non-contiguous layout from the transpose operation  
  encoder.weight \= nn.Parameter(decoder.weight.T.clone())

  \# GOOD: Forces contiguous memory allocation before becoming a Parameter  
  encoder.weight \= nn.Parameter(decoder.weight.T.contiguous().clone())

* **Verification**: Print torch.norm(model.encoder.weight.grad) and torch.norm(model.encoder.weight) before and after calling optimizer.step(). The weight norm must demonstrably change. If the norm is static while gradients are non-zero, the bug is present.

### **Issue 2: MPS Backend Out of Memory (High Watermark Crash)**

* **Symptom**: The training script crashes abruptly with the message: RuntimeError: MPS backend out of memory... Tried to allocate X MB on private pool. Use PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.0 to disable upper limit...  
* **Root Cause**: Apple Silicon utilizes a unified memory architecture, meaning the CPU and GPU share the exact same physical RAM. To prevent the GPU from indiscriminately consuming all system memory and triggering a fatal OS panic, PyTorch enforces a default "high watermark" (usually 1.7x the recommended maximum working set size limit dictated by Metal). Extremely large batch sizes, long un-chunked audio sequences, or failure to explicitly delete intermediate tensors causes the MPSAllocator to hit this hard ceiling.6  
* **Minimal Fix**: Reduce the global batch size, decrease the maximum sequence length, and ensure that PYTORCH\_MPS\_HIGH\_WATERMARK\_RATIO=0.9 is set to force adaptive commit garbage collection before crashing.6 Setting this value to 0.0 disables the limit and is a dangerous anti-pattern that will cause macOS to lock up.  
* **Robust Fix**: Implement explicit garbage collection within the inner training loop. Delete intermediate logits immediately after the loss calculation, and utilize gradient accumulation to simulate large batches without holding vast activation graphs in memory.  
* **Code Example**:  
  Python  
  import os  
  \# Set soft limit for adaptive garbage collection  
  os.environ \= "0.85"

  \# Inside the training loop:  
  loss \= criterion(outputs, targets)  
  loss.backward()  
  optimizer.step()

  \# Robust memory management: aggressively delete heavy tensors  
  del outputs  
  del loss

  \# Force the allocator to flush unused pools back to the OS  
  if step % 10 \== 0:  
      torch.mps.empty\_cache()

* **Verification**: Implement a telemetry callback that logs torch.mps.current\_allocated\_memory() / (1024\*\*3) (GB) at the end of every step. The charted memory usage should oscillate stably in a sawtooth pattern rather than monotonically increasing.39

### **Issue 3: Dataloader Spawn Bottleneck**

* **Symptom**: GPU utilization sits constantly at low percentages (e.g., 10-20%), and iteration speed is incredibly slow. Profiling reveals that the time between batches heavily outweighs the actual forward and backward pass computation time.  
* **Root Cause**: On Linux systems, PyTorch dataloaders utilize fork for highly efficient multiprocessing. On macOS, multiprocessing must rely on spawn, which incurs massive computational overhead due to initializing entirely new Python interpreters for each worker. For relatively lightweight datasets (e.g., pre-processed Mel-spectrograms), the cost of spawning workers and serializing data across process boundaries far exceeds the cost of simply loading the data on the main thread.8  
* **Minimal Fix**: Set num\_workers=0 in all DataLoader instantiations.  
* **Robust Fix**: Set num\_workers=0 as the default for Apple Silicon environments. If complex, CPU-bound audio augmentations are strictly required on-the-fly, carefully tune num\_workers=2 and explicitly set persistent\_workers=True to pay the spawn penalty only once per epoch rather than continuously.8  
* **Code Example**:  
  Python  
  import platform  
  from torch.utils.data import DataLoader

  is\_mac \= platform.system() \== "Darwin"

  \# Best practice configuration for Apple Silicon  
  trainloader \= DataLoader(  
      trainset,   
      batch\_size=32,   
      shuffle=True,   
      \# Bypass spawn overhead entirely on macOS  
      num\_workers=0 if is\_mac else 8,   
      \# pin\_memory is a PCIe/CUDA concept; disable for unified memory  
      pin\_memory=False if is\_mac else True   
  )

* **Verification**: Time the epoch duration using time.time() with num\_workers=0 versus num\_workers=4. On Apple Silicon, 0 will typically yield the lowest overall latency for standard audio loading operations.8

### **Issue 4: BFloat16 Throughput Collapse**

* **Symptom**: Training with torch.bfloat16 mixed precision is exceptionally slow, routinely benchmarking at up to 10x slower than the equivalent model run in float32 or float16.  
* **Root Cause**: The transition to bfloat16 has been a cornerstone of optimizing memory bandwidth on Nvidia's Ampere and Hopper architectures due to its robust dynamic range. However, migrating these assumptions to Apple Silicon is an anti-pattern. PyTorch's MPS backend currently lacks highly optimized native Metal kernels for bfloat16 arithmetic in core operations (such as multi-dimensional convolutions and certain linear projections). When the backend encounters these operations, it frequently falls back to slow emulation pathways or suboptimal execution grids, destroying throughput.1  
* **Minimal Fix**: Switch the Automatic Mixed Precision (AMP) context to use torch.float16.  
* **Robust Fix**: Enforce a strict float16 autocast context specifically for MPS execution, and use a dynamic gradient scaler (torch.amp.GradScaler) to meticulously manage the reduced dynamic range and prevent gradient underflow (which bfloat16 natively avoids).  
* **Code Example**:  
  Python  
  device\_type \= "mps"  
  \# Force float16 on MPS, reserving bfloat16 strictly for CUDA hardware  
  pt\_dtype \= torch.float16 if device\_type \== "mps" else torch.bfloat16

  \# Initialize the gradient scaler (required for float16 stability)  
  scaler \= torch.amp.GradScaler(device=device\_type)

  with torch.amp.autocast(device\_type=device\_type, dtype=pt\_dtype):  
      outputs \= model(inputs)  
      loss \= criterion(outputs, targets)

  scaler.scale(loss).backward()  
  scaler.step(optimizer)  
  scaler.update()

* **Verification**: Measure the tokens/second or steps/second. Transitioning an LLM or audio encoder from bfloat16 to float16 on an M3 Max chip will yield an immediate, highly measurable surge in throughput.2

### **Issue 5: SDPA Long Sequence Buffer Crash**

* **Symptom**: The model crashes abruptly with a Metal assertion error when passing long audio sequences (typically exceeding 12,000 tokens) through standard attention layers.  
* **Root Cause**: PyTorch’s torch.nn.functional.scaled\_dot\_product\_attention (SDPA) is highly optimized on CUDA via FlashAttention. However, the MPS implementation attempts to allocate a single, massive contiguous buffer for the attention matrix (![][image2] memory scaling). For long sequences, this buffer easily exceeds Metal's maximum supported buffer size per individual tensor, resulting in a hard crash. Furthermore, insidious output shape bugs currently exist where the last dimension of the output matches the query head instead of the value head.13  
* **Minimal Fix**: Truncate audio sequences to ensure they remain under 10,000 tokens prior to processing.  
* **Robust Fix**: Implement a chunked attention mechanism for long sequences, or disable SDPA entirely at the framework level and fall back to manual attention computation.  
* **Code Example**:  
  Python  
  \# For Whisper models exported or run natively, explicitly disable SDPA  
  \# to avoid both buffer crashes and dimension mismatch bugs on MPS.  
  import whisper.model  
  whisper.model.MultiHeadAttention.use\_sdpa \= False 

* **Verification**: Pass a dummy tensor of shape (1, 1, 15000, 64\) through the target attention block. With SDPA disabled, it should execute and return the correct shape without raising an out-of-memory exception or Metal buffer assertion error.

### **Issue 6: torch.compile Recompilation Loops**

* **Symptom**: Model execution is extremely slow during the first several epochs, and warning logs frequently state that the graph is being recompiled due to varying integer values or tensor shapes.  
* **Root Cause**: While torch.compile is the default optimization standard for PyTorch 2.0+ on Linux, the Inductor backend for MPS remains highly immature. It suffers from severe graph breaks, repetitive recompilation when simple integer values (like loop counters) change, and frequently raises NotImplementedError when handling advanced attention mechanisms.17  
* **Minimal Fix**: Disable torch.compile when detecting the mps backend.  
* **Robust Fix**: Keep the codebase in eager execution mode for Apple Silicon deployments. Rely on operator-level optimizations rather than full-graph compilation until the Metal Inductor backend matures in later PyTorch releases.

## **Best Practices**

**Precision Strategy on MPS** Apple Silicon neural operations, specifically within the matrix co-processors, are highly optimized for float16 arithmetic. Because float16 suffers from a limited dynamic range—rendering it prone to underflow in small gradients—engineers must utilize torch.amp.GradScaler alongside torch.amp.autocast. Never manually cast an entire model to float16 via model.half(), as critical loss calculations, normalizations, and reduction functions require the numerical stability of float32 to converge correctly.1

**Unified Memory Awareness** Unlike discrete discrete GPUs, moving data from cpu to mps does not traverse a slow PCIe bus; it merely reassigns memory pointers within the unified RAM architecture. Consequently, "pinning memory" (pin\_memory=True), a standard CUDA optimization intended to stage memory for fast PCIe transfer, is virtually useless on Apple Silicon. In some edge cases, it can actually cause memory allocator confusion and degrade performance.37 Treat the CPU and GPU as sharing the exact same physical space, and manage tensor lifecycles accordingly.

**Whisper-Specific Optimizations** When running Whisper models, aggressive model selection based on hardware constraints is paramount. The base.en (142 MB) and small.en (466 MB) models provide the optimal latency-to-accuracy ratio for MPS execution. Avoid deploying the medium and large variants for any real-time processing tasks, as the parameter count exceeds the memory bandwidth required for fast autoregressive generation.10 Furthermore, strictly enforce greedy decoding (beamSize=1). Beam search creates a combinatorial explosion of tensor allocations that forces the MPS allocator to continuously resize pools, devastating throughput.10

**Distillation Load Balancing** When performing knowledge distillation, standard practice dictates loading both a massive teacher model and a compact student model onto the GPU. On a standard 32GB Mac, loading a Whisper Large-v3 teacher alongside a Whisper Small student onto the mps device will easily trigger the high-watermark OOM limit.19 Because the physical memory is shared, the optimal strategy is memory asymmetry: instantiate the frozen teacher on the cpu and the trainable student on mps. Compute the teacher logits on the CPU, move those specific output tensors across the bus to the mps device, and calculate the KL Divergence loss natively on the GPU. This bypasses the Metal allocator limits entirely while still utilizing hardware acceleration where it matters.20

## **Worst Practices / Anti-Patterns**

**Blindly Replicating CUDA Infrastructure** Attempting to map NVIDIA Nsight systems, cuDNN benchmarking flags, and multi-GPU DDP (Distributed Data Parallel) scripts directly to MPS is a fundamental anti-pattern. PyTorch MPS currently does not support the gloo or nccl distributed backends.9 Therefore, attempting to run multi-GPU training scripts or invoking torch.distributed will instantly fail on Apple Silicon.

**Ignoring Explicit Data Types in View Operations** Relying on implicit type promotion during .view() or .reshape() operations will catastrophically crash the MPS backend if the underlying tensor happens to be evaluated in float64. The MPS backend aggressively rejects double-precision data types.14 Engineers must ensure all tensors are explicitly cast to float32 or float16 prior to executing structural transformations.

## **Weird Patterns that Work Surprisingly Well on Apple Silicon**

**The Identity Addition Mask Fix** In specific versions of PyTorch executing on MPS, combining MultiheadAttention with a boolean attention mask and standard dropout generates unexplainable NaNs in the output tensor. Bizarrely, inserting a no-op identity transformation—specifically x \= x \+ 0—immediately following the attention output forces the underlying MPS Graph to correctly re-evaluate the tensor and reliably eliminates the NaNs. This is a known compiler quirk where the identity operation forces a memory sync that resolves a race condition.25

**Zero-Worker Dataloading** As analyzed previously, setting num\_workers=0 violates every established best practice on Linux/CUDA systems. Yet, due to macOS process spawning mechanics, running data loading entirely on the main thread is consistently the fastest, most stable approach for small to medium-sized speech datasets.8

## **Limitations and Unsolved Problems**

**Mamba Selective Scan Support** State Space Models like Mamba rely heavily on a hardware-aware parallel prefix-sum (scan) algorithm to achieve their signature throughput. The official implementations of these models rely on highly customized Triton kernels explicitly tailored for CUDA execution. Currently, these kernels do not translate to Apple's Metal Shading Language. Engineers must rely on slower, unrolled PyTorch implementations or write custom Metal compute shaders from scratch. Consequently, Mamba inference on MPS will fundamentally lag behind CUDA parity for the foreseeable future.15

**Float64 Operations** The lack of native double-precision (float64) support on Apple Silicon GPUs is a hard hardware limitation. Any scientific computing operations, customized loss functions, or precision-sensitive accumulations explicitly requiring float64 will crash or be silently down-casted, potentially destroying the mathematical integrity of the model.14

**Silent Gradient Failures on Older macOS** As identified in the failure catalog, PyTorch versions prior to 2.4 and macOS versions prior to 15.0 exhibit unpatchable kernel bugs regarding non-contiguous tensor writes.4 Engineers locked into legacy macOS environments cannot rely on the optimizer to update certain layers without manually cloning all tensors, which wastes memory and compute cycles.

**Determinism is Expensive** Achieving perfect reproducibility across runs requires invoking torch.use\_deterministic\_algorithms(True). However, doing so on MPS disables crucial asynchronous kernel optimizations and forces rigid synchronization barriers. This guarantees reproducibility but drastically reduces training throughput, turning a fast prototype loop into a crawl.21

## **Copy-Paste Reference Snippets**

### **Device Detection and Safe Fallback Hierarchy**

Use this standard routing function to dynamically assign the execution device across hybrid development teams.

Python

import torch

def get\_compute\_device() \-\> torch.device:  
    """Safely resolves the optimal compute backend."""  
    if torch.cuda.is\_available():  
        return torch.device("cuda")  
    elif torch.backends.mps.is\_available():  
        \# Requires macOS 12.3+ and arm64 architecture  
        return torch.device("mps")  
    else:  
        return torch.device("cpu")

device \= get\_compute\_device()

### **Memory-Aware Training Step for Apple Silicon**

This snippet interleaves garbage collection and handles precision scaling natively, preventing OOM crashes during Whisper LoRA fine-tuning.

Python

import torch  
import torch.nn as nn  
from torch.amp import GradScaler, autocast

\# Setup  
device \= torch.device("mps")  
model \= MySpeechModel().to(device)  
optimizer \= torch.optim.AdamW(model.parameters(), lr=1e-4)

\# Initialize scaler specifically for MPS  
scaler \= GradScaler(device="mps") 

def memory\_safe\_train\_step(batch\_idx, inputs, targets):  
    optimizer.zero\_grad()  
      
    \# Enforce float16 for MPS, avoiding the bfloat16 performance penalty  
    with autocast(device\_type="mps", dtype=torch.float16):  
        outputs \= model(inputs)  
        loss \= criterion(outputs, targets)  
          
    \# Scale the loss to prevent float16 underflow  
    scaler.scale(loss).backward()  
    scaler.step(optimizer)  
    scaler.update()  
      
    \# Explicit garbage collection for unified memory  
    del outputs  
    del loss  
    if batch\_idx % 20 \== 0:  
        torch.mps.empty\_cache()

### **Mixed CPU/MPS Execution for Unsupported Ops (Custom Loss)**

When a custom operator (like a specialized ASR loss function) throws an MPS NotImplemented error, wrap it in a fallback context to maintain the pipeline without setting global environment variables that obscure other bugs.

Python

import torch  
import torch.nn as nn

class HybridCustomLoss(nn.Module):  
    def \_\_init\_\_(self):  
        super().\_\_init\_\_()  
        \# Initialize the complex loss module  
        self.complex\_loss \= nn.MSELoss()

    def forward(self, mps\_logits: torch.Tensor, mps\_targets: torch.Tensor) \-\> torch.Tensor:  
        \# 1\. Pull tensors to CPU for the unsupported operation.  
        \# This incurs a sync penalty, but guarantees execution.  
        cpu\_logits \= mps\_logits.cpu()  
        cpu\_targets \= mps\_targets.cpu()  
          
        \# 2\. Perform the complex/unsupported operation on CPU  
        cpu\_loss \= self.complex\_loss(cpu\_logits, cpu\_targets) \* 0.5   
          
        \# 3\. Push the resulting scalar back to MPS to maintain the autograd graph  
        return cpu\_loss.to("mps")

### **Parity Testing Between CPU and MPS**

Before trusting a new workflow or a complex custom layer, execute this numeric parity check to ensure MPS emulation perfectly matches the CPU baseline.

Python

import torch  
import torch.nn as nn

def verify\_mps\_parity(model: nn.Module, dummy\_input: torch.Tensor):  
    model.eval()  
      
    \# Run CPU Baseline  
    model\_cpu \= model.to("cpu")  
    input\_cpu \= dummy\_input.to("cpu")  
    with torch.no\_grad():  
        out\_cpu \= model\_cpu(input\_cpu)  
          
    \# Run MPS Evaluation  
    model\_mps \= model.to("mps")  
    input\_mps \= dummy\_input.to("mps")  
    with torch.no\_grad():  
        out\_mps \= model\_mps(input\_mps)  
          
    \# Assert numeric closeness. Note that mps output must be brought to cpu for comparison.  
    try:  
        torch.testing.assert\_close(out\_cpu, out\_mps.cpu(), rtol=1e-3, atol=1e-3)  
        print("MPS and CPU outputs are numerically equivalent.")  
    except AssertionError as e:  
        print(f"PARITY FAILURE: {e}")

### **PyTorch-to-CoreML Export Preparation Stub**

CoreML export requires static shapes and careful tracing. This snippet demonstrates preparing an encoder-heavy speech model for the Apple Neural Engine, bypassing the dynamic shape limitations of the decoder.

Python

import torch  
import coremltools as ct

def export\_encoder\_to\_coreml(encoder\_model: torch.nn.Module, sample\_audio\_tensor: torch.Tensor):  
    \# 1\. Prepare model for static tracing (disables dropout, fixes batch norm)  
    encoder\_model.eval()  
      
    \# 2\. Trace the PyTorch execution graph using a dummy input  
    traced\_encoder \= torch.jit.trace(encoder\_model, sample\_audio\_tensor)  
      
    \# 3\. Convert to CoreML targeting the Apple Neural Engine (ANE)  
    mlmodel \= ct.convert(  
        traced\_encoder,  
        inputs=,  
        \# Force the model to leverage the highly efficient Neural Engine  
        compute\_units=ct.ComputeUnit.CPU\_AND\_NE   
    )  
      
    mlmodel.save("WhisperEncoder.mlpackage")  
    print("Exported Encoder successfully for ANE execution.")

## **Final "Red Flags" Checklist Before Shipping**

Before finalizing a PR targeting Apple Silicon workflows, systematically verify the following conditions:

* \[ \] PYTORCH\_ENABLE\_MPS\_FALLBACK=1 is securely exported in the deployment environment to prevent hard crashes on obscure ops.  
* \[ \] No occurrences of torch.bfloat16 exist within the MPS execution pathway, preventing massive latency spikes.  
* \[ \] No occurrences of torch.float64 or implicit .double() promotions exist, avoiding silent casts and runtime failures.  
* \[ \] torch.mps.empty\_cache() is called strategically during training to prevent Unified Memory exhaustion and OS paging.  
* \[ \] Model weights and optimizers have been explicitly verified as .contiguous() to prevent the silent gradient failure bug.  
* \[ \] Dataloader num\_workers is set to 0 unless extensive local profiling dictates that the spawn overhead is justified by heavy augmentation.  
* \[ \] Long sequence attention mechanisms are truncated or explicitly chunked to prevent Metal buffer overflow assertions.  
* \[ \] CoreML transitions are strictly reserved for static-shape encoders (e.g., audio feature extraction), never dynamic autoregressive decoders.

#### **Works cited**

1. Automatic Mixed Precision package \- torch.amp — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html)  
2. Why is there such a huge performance gap between bfloat16, float16, and float32?, accessed April 2, 2026, [https://discuss.pytorch.org/t/why-is-there-such-a-huge-performance-gap-between-bfloat16-float16-and-float32/219536](https://discuss.pytorch.org/t/why-is-there-such-a-huge-performance-gap-between-bfloat16-float16-and-float32/219536)  
3. bfloat16 Conv2d slower than float16 on 4090 · Issue \#154351 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/154351](https://github.com/pytorch/pytorch/issues/154351)  
4. MPS Random in-place operations fail silently on non-contiguous tensors (macOS \< 15.0) · Issue \#165257 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/165257](https://github.com/pytorch/pytorch/issues/165257)  
5. the bug that taught me more about PyTorch than years of using it \- Elana Simon, accessed April 2, 2026, [https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)  
6. MPS Environment Variables — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/mps\_environment\_variables.html](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)  
7. MPS backend out of memory \- PyTorch Forums, accessed April 2, 2026, [https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879](https://discuss.pytorch.org/t/mps-backend-out-of-memory/183879)  
8. Efficient Data Pipelines in PyTorch: Lessons from \`num\_workers\` | by Allam Satyanarayana, accessed April 2, 2026, [https://medium.com/@allam.satyanarayana/efficient-data-pipelines-in-pytorch-lessons-from-num-workers-4d49eb6b384d](https://medium.com/@allam.satyanarayana/efficient-data-pipelines-in-pytorch-lessons-from-num-workers-4d49eb6b384d)  
9. PyTorch training on Apple silicon \- Hugging Face, accessed April 2, 2026, [https://huggingface.co/docs/transformers/v4.48.0/perf\_train\_special](https://huggingface.co/docs/transformers/v4.48.0/perf_train_special)  
10. Performance Optimization \- whisper.rn \- Mintlify, accessed April 2, 2026, [https://www.mintlify.com/mybigday/whisper.rn/advanced/optimization](https://www.mintlify.com/mybigday/whisper.rn/advanced/optimization)  
11. Converting from PyTorch — Guide to Core ML Tools \- Apple, accessed April 2, 2026, [https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html)  
12. Convert PyTorch models to Core ML \- Tech Talks \- 비디오 \- Apple Developer, accessed April 2, 2026, [https://developer.apple.com/kr/videos/play/tech-talks/10154/?time=137](https://developer.apple.com/kr/videos/play/tech-talks/10154/?time=137)  
13. Optimizing PyTorch MPS Attention: Memory-Efficient Large Sequence Processing Without Accuracy Trade-offs | by Raksheka R | Medium, accessed April 2, 2026, [https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b](https://medium.com/@rakshekaraj/optimizing-pytorch-mps-attention-memory-efficient-large-sequence-processing-without-accuracy-5239f565f07b)  
14. State of PyTorch Hardware Acceleration 2025, accessed April 2, 2026, [https://tunguz.github.io/PyTorch\_Hardware\_2025/](https://tunguz.github.io/PyTorch_Hardware_2025/)  
15. Implementing Mamba in PyTorch \- Medium, accessed April 2, 2026, [https://medium.com/@torotoki/getting-started-with-mamba-implementing-mamba-in-pytorch-33d56ccd8393](https://medium.com/@torotoki/getting-started-with-mamba-implementing-mamba-in-pytorch-33d56ccd8393)  
16. purohit10saurabh/mamba-ssm-macos: (Unofficial) Mamba ... \- GitHub, accessed April 2, 2026, [https://github.com/purohit10saurabh/mamba-ssm-macos](https://github.com/purohit10saurabh/mamba-ssm-macos)  
17. Weekly GitHub Report for Pytorch: September 15, 2025 \- Buttondown, accessed April 2, 2026, [https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-september-15/](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-september-15/)  
18. Weekly GitHub Report for Pytorch: May 26, 2025 \- June 02, 2025 (12:05:35) \- Buttondown, accessed April 2, 2026, [https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-26-2025-june-5528/](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-pytorch-may-26-2025-june-5528/)  
19. How to optimize memory usage in PyTorch? \- GeeksforGeeks, accessed April 2, 2026, [https://www.geeksforgeeks.org/deep-learning/how-to-optimize-memory-usage-in-pytorch/](https://www.geeksforgeeks.org/deep-learning/how-to-optimize-memory-usage-in-pytorch/)  
20. A Friendly Guide to Knowledge Distillation (with PyTorch code you can paste today), accessed April 2, 2026, [https://mohamed-stifi.medium.com/a-friendly-guide-to-knowledge-distillation-with-pytorch-code-you-can-paste-today-5a764762e7c7](https://mohamed-stifi.medium.com/a-friendly-guide-to-knowledge-distillation-with-pytorch-code-you-can-paste-today-5a764762e7c7)  
21. Reproducibility — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/notes/randomness.html](https://docs.pytorch.org/docs/stable/notes/randomness.html)  
22. WhisperUI\_4Mac\_MPS is a macOS application leveraging Apple Silicon for efficient video-to-audio conversion and multilingual transcription using OpenAI's Whisper Large v3 Turbo. It features CUDA support, YouTube integration, batch processing, and real-time resource monitoring. Optimize your audio workflow with this powerful tool. · GitHub, accessed April 2, 2026, [https://github.com/Git-Fg/WhisperUI\_4Mac\_MPS](https://github.com/Git-Fg/WhisperUI_4Mac_MPS)  
23. torch.mps — PyTorch 2.11 documentation, accessed April 2, 2026, [https://docs.pytorch.org/docs/stable/mps.html](https://docs.pytorch.org/docs/stable/mps.html)  
24. Fine-tune Whisper models on Amazon SageMaker with LoRA | Artificial Intelligence \- AWS, accessed April 2, 2026, [https://aws.amazon.com/blogs/machine-learning/fine-tune-whisper-models-on-amazon-sagemaker-with-lora/](https://aws.amazon.com/blogs/machine-learning/fine-tune-whisper-models-on-amazon-sagemaker-with-lora/)  
25. \[MPS\] MultiheadAttention with masks and dropout produces NaNs · Issue \#151667 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/151667](https://github.com/pytorch/pytorch/issues/151667)  
26. Torch MPS Basic Speedup Test \- Sungwon Kim, accessed April 2, 2026, [https://sungwon-kim.com/blog/2025/torch-mps-speedup-test/](https://sungwon-kim.com/blog/2025/torch-mps-speedup-test/)  
27. A Complete Guide to PyTorch Loss Functions \- Lightly AI, accessed April 2, 2026, [https://www.lightly.ai/blog/pytorch-loss-functions](https://www.lightly.ai/blog/pytorch-loss-functions)  
28. Benchmarking On-Device Machine Learning on Apple Silicon with MLX (Oct 2025), accessed April 2, 2026, [https://www.youtube.com/watch?v=r-r3r4X6yVE](https://www.youtube.com/watch?v=r-r3r4X6yVE)  
29. \[2510.18921\] Benchmarking On-Device Machine Learning on Apple Silicon with MLX \- arXiv, accessed April 2, 2026, [https://arxiv.org/abs/2510.18921](https://arxiv.org/abs/2510.18921)  
30. Converting PyTorch Model to CoreML Using ONNX, accessed April 2, 2026, [https://webpages.charlotte.edu/ialzouby/modelconversion.html](https://webpages.charlotte.edu/ialzouby/modelconversion.html)  
31. Optimized Deep Learning on Apple Silicon with PyTorch MPS \- Apple Books, accessed April 2, 2026, [https://books.apple.com/us/book/optimized-deep-learning-on-apple-silicon-with-pytorch-mps/id6751339705](https://books.apple.com/us/book/optimized-deep-learning-on-apple-silicon-with-pytorch-mps/id6751339705)  
32. Accelerated PyTorch training on Mac \- Metal \- Apple Developer, accessed April 2, 2026, [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)  
33. How to implement a custom loss in pytorch, accessed April 2, 2026, [https://discuss.pytorch.org/t/how-to-implement-a-custom-loss-in-pytorch/197938](https://discuss.pytorch.org/t/how-to-implement-a-custom-loss-in-pytorch/197938)  
34. \[Performance\] PyTorch (MPS) is faster than MLX in backward of convolution layer \#1313, accessed April 2, 2026, [https://github.com/ml-explore/mlx/issues/1313](https://github.com/ml-explore/mlx/issues/1313)  
35. MLX vs MPS vs CUDA: a Benchmark | Towards Data Science, accessed April 2, 2026, [https://towardsdatascience.com/mlx-vs-mps-vs-cuda-a-benchmark-c5737ca6efc9/](https://towardsdatascience.com/mlx-vs-mps-vs-cuda-a-benchmark-c5737ca6efc9/)  
36. MPS training (basic) — PyTorch Lightning 2.6.1 documentation, accessed April 2, 2026, [https://lightning.ai/docs/pytorch/stable/accelerators/mps\_basic.html](https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html)  
37. Speed Up Model Training — PyTorch Lightning 2.6.1 documentation, accessed April 2, 2026, [https://lightning.ai/docs/pytorch/stable/advanced/speed.html](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)  
38. PyTorch Memory Management Tips | PDF | Applied Mathematics \- Scribd, accessed April 2, 2026, [https://www.scribd.com/document/890258525/fewfwefwefsdfsdf](https://www.scribd.com/document/890258525/fewfwefwefsdfsdf)  
39. pytorch/torch/mps/\_\_init\_\_.py at main \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/blob/main/torch/mps/\_\_init\_\_.py](https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py)  
40. MPS Memory Leaks across core MPS files · Issue \#164299 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/164299](https://github.com/pytorch/pytorch/issues/164299)  
41. MPS: scaled\_dot\_product\_attention returns wrong output shape when value dim \!= query/key dim · Issue \#176767 \- GitHub, accessed April 2, 2026, [https://github.com/pytorch/pytorch/issues/176767](https://github.com/pytorch/pytorch/issues/176767)  
42. Guidelines for assigning num\_workers to DataLoader \- PyTorch Forums, accessed April 2, 2026, [https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813)  
43. AutoTrain Python automatically using MPS on Mac \- How to switch to CPU, accessed April 2, 2026, [https://discuss.huggingface.co/t/autotrain-python-automatically-using-mps-on-mac-how-to-switch-to-cpu/116449](https://discuss.huggingface.co/t/autotrain-python-automatically-using-mps-on-mac-how-to-switch-to-cpu/116449)  
44. Implementation of Mamba in one file of PyTorch | Hacker News, accessed April 2, 2026, [https://news.ycombinator.com/item?id=38708730](https://news.ycombinator.com/item?id=38708730)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAA60lEQVR4XpWSPQoCMRCFJ9h4ALGxFrG1WUGwsxLsvYLgHaxsPIWNZ7D2ELYeQVtt9OUHk5lk3PjBI5P3MknYLJGGkXVqsFkoeZ5ZeZw7jJbYoy3STs7X506C3rrA9I1xw1yG+dHumZHf5CADS2wotUaG0BM6yeAfetAdusiAkV2EG10YN4xXqMOSSvrQAzrLQMEdP4JeqI8iJHm9lDlC++X3MhCwF1xTxZtLtuSbVjJoY4e9Jr7kf1Kwaqlfqa8NvhanDKCllwmjWk9jm8PY37DBMY0dTVIXNI59Ae16Zb/sVsPa3aSwYcH68gGB6RgJkHhrkgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAYCAYAAAC4CK7hAAACsUlEQVR4XuWWy+uOQRTHz+uWS24lVih/gBW5LYSSNUKSH0qxsVHCws4lG34lsbDAghJ/gYXfQikpl41LWUgSFhQh1/N9ZuZ55/nOeead35tXyqdO78z3nDNzZp5n5n2kI/81A1v+XbVfarfZ8Xdormtto1fOu6h9Se1b1AdbqN+Gucvz1c6pnVGbRj6LPWoHWSwET+KEb6MY9GtUWKA/D2LN61lOixtou+/P05Q3+vulG5IwV2NesdiLlkIWCy3EM6x2nkWLMeIGGEHHmOS72k8WPcibyGIOY/wAXqutLHqsBSYg6DmLEavFxawhfbna19CJC8wU28YF8eehJfey4BVrcYKXYqyW4sMTu9aUqx3s92zE7FJb6tsrYkfEdKlqsFeyUlyBt0hnZoqLe086tEmkBa6orY/6m9SuSrfgwCJxh32H2m61JxDtcqv5zMsHO1ryjm8TF3c/0qZ6zeKH/4V/g7gNmCFu0dDiKxV9tjbgO4QGL9QlspryVFwsrtnAKq8xKHbYjwk/LooYaDdJ8/QsBLkXq1YUOts7rGIYK26noYGFamPV5ojzT266K21/6PQsvclntTssYjIMCmeOjeLiRkgf8nobZyX1L6u0jqt/lIsAn9TusQisnWZcTDorDq2RWwfCx58br73eL8i9wSKm/CD5gV+I849vylWx4SZjPQDf8Vjw2rGo3Uq6bxXIOcxiAM6HRuZbSXeUQe4EFpVZ4nx8NUPDf9IStb3kq+mWkhSFfGxgK/gCRRAOEs4M2pisF3jl6oMbsVnsHX8kTj/FjgKmiD3maEl2BxxQ+8jiQOhUX+PXWf6TYJfGsTgAck/D3GVp103WqT1jsZyiuY6oHS0LLSAzzklx30mDAH+sj1msyRRVRjJAZ4iVlCSphH0s9Elfk9dks7POf40BF/sb1neUgXKuhioAAAAASUVORK5CYII=>