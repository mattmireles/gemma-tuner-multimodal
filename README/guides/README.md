# Field guides

Long-form reference material for Apple Silicon training, export, and adjacent tooling. Product behavior and roadmap live under [`../specifications/`](../specifications/).

Mamba-specific guides were moved to the standalone `mamba-asr-mps` repository so this guide index stays focused on the Whisper product surface.

## Apple Silicon (`apple-silicon/`)

| Guide | Description |
| --- | --- |
| [`pytorch-mps.md`](apple-silicon/pytorch-mps.md) | PyTorch MPS production notes, Core ML tradeoffs, speech-model pitfalls on Metal. |
| [`metal-mpsgraph-whisper.md`](apple-silicon/metal-mpsgraph-whisper.md) | Low-level Whisper fine-tuning with MPSGraph (not PyTorch MPS). |
| [`pytorch-to-mlx-migration.md`](apple-silicon/pytorch-to-mlx-migration.md) | Mental model and migration patterns from PyTorch to MLX / MLX Swift. |
| [`gemma3n.md`](apple-silicon/gemma3n.md) | Condensed Gemma 3n on-device architecture; complements the product spec. |

## Core ML (`coreml/`)

| Guide | Description |
| --- | --- |
| [`pytorch-tensorflow-to-coreml.md`](coreml/pytorch-tensorflow-to-coreml.md) | Generic PyTorch / TensorFlow → Core ML conversion (coremltools, common failures). |

## Whisper export & deployment (`whisper/`)

| Guide | Description |
| --- | --- |
| [`coreml-hybrid-whispercpp.md`](whisper/coreml-hybrid-whispercpp.md) | Whisper encoder on ANE via Core ML + decoder in whisper.cpp; environment and pitfalls. |
| [`gguf-export-and-coreml-deployment.md`](whisper/gguf-export-and-coreml-deployment.md) | End-to-end GGUF export, hybrid deployment, quantization, and validation playbook. |

## Integrations (`integrations/`)

| Guide | Description |
| --- | --- |
