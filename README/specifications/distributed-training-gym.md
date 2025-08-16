# Distributed Training Gym - Product Specification

(Add Details Here)

## Executive Summary

The Distributed Training integration enables multi-GPU fine-tuning of Whisper models, dramatically accelerating the training process by leveraging the `ExoGym` framework. It is implemented as an optional, minimally invasive layer that extends the existing Hugging Face-based training pipeline, rather than replacing it. This allows users to scale their training workflows to multiple GPUs on a single machine with simple CLI flags or wizard options, unlocking significant performance gains without altering their existing profiles or configurations.

### Key Capabilities

- **Accelerated Training**: Achieve near-linear training speed improvements with multiple GPUs.
- **Advanced Strategies**: Access sophisticated distributed training strategies like DiLoCo for communication-efficient training, in addition to standard data parallelism.
- **Seamless Integration**: No changes are required for existing training profiles. Distributed training is a parallel, opt-in workflow.
- **Hardware Agnostic**: Supports multi-GPU training on both NVIDIA (via NCCL) and Apple Silicon (via Gloo) systems.
- **User-Friendly**: Integrated directly into the CLI wizard for a guided, zero-config user experience.

### Core Value Proposition

"Fine-tune Whisper models up to 4x faster on multi-GPU systems without rewriting your training logic."

---

## Technical Architecture

### Core Principle: Wrap and Extend, Don't Replace

The integration is architected to be a wrapper around the existing `Seq2SeqTrainer`. We will not replace the battle-tested Hugging Face training loop. Instead, we intercept the optimization step and delegate it to a `gym` strategy. This allows us to inject sophisticated distributed communication logic with minimal changes to the existing codebase.

### System Architecture Diagram

```mermaid
graph TD
    subgraph Main Process
        User --> Launcher[train_distributed.py];
        Launcher -->|Parses args| GymTrainer[gym.exogym.trainer.LocalTrainer];
        GymTrainer -->|mp.spawn()| Workers;
    end

    subgraph "Worker Processes (N x GPUs)"
        Workers --> Node[WhisperTrainNode];
        Node -->|Instantiates| HFTrainer[distributed.trainer.DistributedWhisperTrainer];
        HFTrainer -->|Delegates step to| Strategy[gym.exogym.strategy.Strategy];
        Strategy -->|Handles comms & optim| TorchDist[torch.distributed];
    end

    style User fill:#e1f5fe
    style Launcher fill:#fff3e0
```

### Component Deep Dive

1.  **`train_distributed.py` (The Launcher)**
    - **Role**: The new, optional entry point for distributed training.
    - **Functionality**:
        - Parses CLI arguments (`--profile`, `--num_nodes`, `--strategy`).
        - If `num_nodes` is 1, it calls the original `main.py finetune` script for backward compatibility.
        - Loads the base model onto the CPU (crucial for `mp.spawn` safety).
        - Instantiates the appropriate `gym` `Strategy` based on user input.
        - Initializes the `gym.exogym.trainer.LocalTrainer` and starts the training process.
        - Receives the final, averaged model state upon completion and saves it.

2.  **`gym.exogym.trainer.LocalTrainer` (The Process Manager)**
    - **Role**: A component from the `gym` library responsible for managing the distributed environment on a single machine.
    - **Functionality**:
        - Handles the `torch.multiprocessing.spawn()` call to create worker processes.
        - Sets up the `torch.distributed` process group for communication (NCCL for NVIDIA, Gloo for Apple Silicon/CPU).
        - Passes a serialized `TrainingConfig` object to each worker.
        - Collects the final model state dictionaries from each worker and averages them to produce the final model.

3.  **`WhisperTrainNode` (The Worker Task)**
    - **Role**: A custom `gym.exogym.train_node.TrainNode` that runs inside each spawned process.
    - **Functionality**:
        - Replicates the setup logic from the original `models/whisper/finetune.py` script (e.g., loading datasets, collators, and `Seq2SeqTrainingArguments`). This requires refactoring the original script into modular helper functions.
        - Initializes the `gym` strategy on the worker's target device, which in turn creates the optimizer.
        - Instantiates our custom `DistributedWhisperTrainer`, providing it with the model, datasets, arguments, and the `gym` strategy.
        - Calls the `.train()` method on the Hugging Face trainer to start the training loop.
        - Returns the final model's state dictionary to the main process for averaging.

4.  **`distributed.trainer.DistributedWhisperTrainer` (The Bridge)**
    - **Role**: The heart of the integration, this class subclasses `transformers.Seq2SeqTrainer`.
    - **Functionality**:
        - Accepts a `gym.Strategy` object in its `__init__` method.
        - **Overrides the `training_step` method**. This is the key modification. Inside this method, it performs the standard forward pass and `loss.backward()`.
        - Instead of calling the optimizer directly, it calls `self.strategy.step()`.

5.  **`gym.exogym.strategy.Strategy` (The Brains)**
    - **Role**: An object from the `gym` library that encapsulates the distributed communication and optimization logic.
    - **Functionality**:
        - The `step()` method is called by our custom trainer.
        - For `SimpleReduceStrategy` (standard DDP), it will perform an `all_reduce` operation on the gradients across all GPUs and then call `optimizer.step()`.
        - For `DiLoCoStrategy`, it will perform a local `optimizer.step()`, and only communicate (by averaging model weights) every `H` steps.

---

## User Journey

### 1. Wizard-Based Workflow

The `wizard.py` script will be updated to make distributed training a simple, guided choice.

1.  **New Question**: After selecting the "finetune" operation, the user is asked if they want to enable distributed training.
    ```
    ? Enable distributed training (multi-GPU)? (y/N)
    ```
2.  **Conditional Questions**: If yes, the wizard asks for the number of GPUs and the desired strategy.
    ```
    ? How many GPUs to use? (2)
    ? Which distributed strategy?
      ❯ allreduce (standard)
        diloco (communication-efficient)
    ```
3.  **Command Generation**: The wizard constructs and executes the appropriate `python train_distributed.py ...` command. The user experience remains seamless.

### 2. CLI-Based Workflow

Power users can directly invoke the new launcher script.

```bash
# Standard Data Parallelism with 2 GPUs (all-reduce each step)
python train_distributed.py \
  --profile whisper-small-librispeech \
  --output_dir output/dist-run-1 \
  --num_nodes 2 \
  --strategy allreduce

# DiLoCo with 4 GPUs and a communication interval of 100 steps
python train_distributed.py \
  --profile whisper-small-librispeech \
  --output_dir output/dist-run-2 \
  --num_nodes 4 \
  --strategy diloco \
  --h_param 100

# Single-device (uses bridge trainer under the hood, no spawn)
python train_distributed.py \
  --profile whisper-small-librispeech \
  --output_dir output/single \
  --num_nodes 1 \
  --strategy allreduce
```

---

## Configuration Reference

### Launcher CLI Arguments

The `train_distributed.py` script will accept the following arguments:

| Argument | Type | Description | Default |
|---|---|---|---|
| `--profile` | str | The training profile from `config.ini` to use. | **Required** |
| `--output_dir` | str | The directory to save the final model and logs. | **Required** |
| `--num_nodes` | int | The number of GPUs to use for training. | `1` |
| `--strategy` | str | The distributed strategy to use. Choices: `allreduce`, `diloco`.| `allreduce` |
| `--h_param` | int | The communication interval for DiLoCo (`H` parameter). | `100` |
| `...` | | Other standard training arguments like `learning_rate` will be added. | |

### Profile Configuration (`config.ini`)

**No changes are required.** All existing profiles will work with the distributed launcher without modification. The launcher passes the profile configuration to each worker, which then sets up the training environment accordingly.

---

## Supported Strategies

1.  **`allreduce` (Standard Data Parallelism)**
    - **How it works**: Averages gradients after every single batch. This is equivalent to PyTorch's `DistributedDataParallel` (DDP).
    - **When to use**: This is the default and most stable strategy. It is highly effective on systems with fast GPU-to-GPU interconnects (e.g., NVIDIA NVLink).

2.  **`DiLoCo` (Distributed Low-Communication)**
    - **How it works**: Each GPU performs local optimizer steps independently and only communicates to average the full model weights periodically (every `H` steps).
    - **When to use**: This can be more efficient on systems with slower interconnects (e.g., standard PCIe, or multiple Apple Silicon devices). It reduces communication overhead but may require tuning of the `H` parameter to maintain convergence quality.

---

## Performance Expectations

Distributed training offers significant reductions in training time. The actual speedup depends on the model size, batch size, and the system's interconnect bandwidth.

| Number of GPUs | Theoretical Speedup | Expected Real-World Speedup |
|---|---|---|
| 1 | 1.0x | 1.0x (baseline) |
| 2 | 2.0x | ~1.7x - 1.9x |
| 4 | 4.0x | ~3.2x - 3.8x |
| 8 | 8.0x | ~6.0x - 7.5x |

*Note: These are estimates. Larger models and batch sizes tend to see better scaling.*

---

## Limitations

- **Single-Machine Only**: This implementation is designed for multi-GPU training on a single machine and does not support multi-node (cross-machine) clusters.
- **Hugging Face `Trainer` Dependency**: The workflow is still fundamentally based on the Hugging Face `Trainer` API. Any bugs or limitations within that API are inherited.
- **Strategy Compatibility**: The initial implementation will focus on `SimpleReduceStrategy` (allreduce) and `DiLoCoStrategy`. While other `gym` strategies exist, they would require testing and potential adaptation to be compatible with this workflow.

Of course. A detailed plan is essential for execution. Here is the complete, step-by-step guide for your engineer. This plan is designed to be minimally invasive, introducing distributed training as an optional, parallel workflow that doesn't disrupt any of the existing single-device functionality.

## Implementation Progress (Track Your Progress Below)

### Status at a glance

- [x] Core `gym` framework vendored and wired
  - [x] `gym/exogym/trainer.py` → `LocalTrainer` for single-machine multi-GPU (CUDA/NCCL, MPS/Gloo, CPU/Gloo)
  - [x] `gym/exogym/train_node.py` → `TrainNode` implementing the distributed training loop, gradient accumulation, validation, and logging hooks
  - [x] `gym/exogym/strategy/strategy.py` → Strategy base + `SimpleReduceStrategy` (gradient all-reduce each step)
  - [x] Additional strategies scaffolding present (DiLoCo, Federated, Sparta variants) pending Whisper integration validation

- [x] HF bridge trainer
  - [x] `distributed/trainer.py` → `DistributedWhisperTrainer(Seq2SeqTrainer)` that overrides `training_step()` and delegates optimizer work to a `gym` Strategy via `strategy.step()`

- [ ] Launcher and workflow glue
  - [x] `train_distributed.py` (launcher) implemented:
    - parses `--profile`, `--output_dir`, `--num_nodes`, `--strategy`, `--h_param`
    - loads profile via existing config system
    - Single-process: injects `DistributedWhisperTrainer` + `SimpleReduceStrategy`
    - Multi-process: spawns `num_nodes` workers (torch.distributed init per-rank), selects backend (NCCL/Gloo), runs HF pipeline with injected strategy per rank, isolates rank outputs
    - Current limitation: `diloco` under HF/DDP falls back to `allreduce` (conflicting reducers); DiLoCo will be enabled when switching to `LocalTrainer`
  - [x] Worker wiring for Whisper (minimal HF-bridge path)
    - Reuse `models/whisper/finetune.py` via injected `trainer_class=DistributedWhisperTrainer`
    - Strategy constructed per-rank in launcher; HF `main()` handles datasets/collators/args
    - Note: rank 0 writes to `--output_dir`; other ranks write to `--output_dir/rank_{i}`

  - [x] Wizard integration (baseline)
    - [x] Adds prompt to enable distributed training
    - [x] Prompts for `num_nodes` and `strategy` (`allreduce`/`diloco`, optional `H`)
    - [x] Executes `train_distributed.py` with the generated temporary profile

- [ ] Tests and validation
  - [ ] Sanity test on Apple Silicon (MPS/Gloo) single-device fallbacks
  - [ ] E2E test: config → launcher → multi-rank training → final artifacts

- [ ] Documentation and examples
  - [x] Add README usage examples mirroring the spec’s CLI snippets (see CLI section above)
  - [x] Note MPS caveats and environment variables for debugging (`PYTORCH_ENABLE_MPS_FALLBACK=1`)

### Notes

- Current implementation uses the HF bridge path (no custom `WhisperTrainNode`). If needed, we can swap to `gym.exogym.trainer.LocalTrainer` later for model-state averaging workflows. For now, per-step synchronization occurs via strategy `.step()`; rank output isolation avoids checkpoint contention.
- Gradient sync double-counting is avoided by detecting DDP-wrapped models inside `SimpleReduceStrategy` and skipping manual all-reduce when applicable.

