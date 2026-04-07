# Field guides

Long-form reference for Apple Silicon training and Hugging Face + PyTorch MPS behavior. Gemma training details: [`../specifications/Gemma3n.md`](../specifications/Gemma3n.md). Text-only CSV fine-tuning (profile keys, datasets): [`../Datasets.md`](../Datasets.md) and the root [`README.md`](../README.md#text-only-fine-tuning) “Text-only fine-tuning” section.

## Apple Silicon (`apple-silicon/`)

| Guide | Description |
| --- | --- |
| [`pytorch-mps.md`](apple-silicon/pytorch-mps.md) | PyTorch MPS production notes and Metal pitfalls. |
| [`gemma3n.md`](apple-silicon/gemma3n.md) | Condensed Gemma 3n architecture notes. |
| [`gemma4-guide.md`](apple-silicon/gemma4-guide.md) | Gemma 4 multimodal LoRA on MPS (stack pins, PEFT, modalities). |
| [`HF-transformers-MPS-guide.md`](apple-silicon/HF-transformers-MPS-guide.md) | Transformers on MPS (device maps, dtype, debugging). |
| [`LoRA-Apple-Silicon-Guide.md`](apple-silicon/LoRA-Apple-Silicon-Guide.md) | LoRA / PEFT workflows on Apple Silicon. |
