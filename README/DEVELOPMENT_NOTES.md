# Development Notes
## Dependency Lockfile

We support deterministic installs via a lockfile using `uv` (preferred) or `pip-tools`.

- Generate (uv):
  - `uv pip compile requirements/requirements.txt -o requirements.lock`
- Generate (pip-tools):
  - `pip-compile --generate-hashes -o requirements.lock requirements/requirements.txt`

Install from lockfile:
- `uv pip install -r requirements.lock` or `pip install -r requirements.lock`


## Training Tips

### Clear Data Cache
If modifying data, clear the HuggingFace cache:
```bash
rm -rf ~/.cache/huggingface/datasets/*
```

### bfloat16 Training
- Works reliably with `accelerate launch` on CUDA
- NOT recommended for Apple Silicon MPS - bfloat16 is emulated and slower
- Use float16 instead for native MPS performance
- Be careful with config booleans being read as strings

## Dataset Implementation

### Local Dataset Loading
The dataset is loaded using a generator file copied to:
```
../../data/datasets/${DATASET}/dataset_loader.py
```

This is a workaround for local dataset handling with HuggingFace datasets.

### Dataset Format
- `id`: Database ID
- `audio`: Original audio file
- `text_verbatim`: Transcription including disfluencies
- `text_perfect`: Improved transcriptions excluding disfluencies
- `language`: Two-letter language code or `??` for unknown

## Export artifacts

Training produces Hugging Face–compatible checkpoints and adapters (`gemma_tuner/scripts/export.py`). Third-party GGUF / local-inference conversion pipelines are not maintained in this repository.

## Data Quality Notes

### Common Data Issues
- Language detection may be incorrect for some samples
- Some transcriptions may be translations rather than verbatim
- Audio quality varies across samples

## Dataset Schema Example

### Expected CSV Format
```csv
id,audio_url,language,duration_seconds,text_verbatim,text_perfect,recording_type
1234,path/to/audio.wav,en,15.2,"um, hello world","Hello world",dictation
```

### Key Fields
- `id`: Unique identifier
- `audio_url`: Path to audio file (local or remote)
- `language`: ISO 639-1 code or `??` for unknown
- `duration_seconds`: Audio length
- `text_verbatim`: Raw transcription with disfluencies
- `text_perfect`: Cleaned transcription
- `recording_type`: Type of audio (dictation, conversation, etc.)

## Architecture Notes

### Source Tree
- Training code lives under `gemma_tuner/models/gemma/` and shared utilities in `gemma_tuner/utils/`.

### Pending Tasks
- [ ] Make finetune.py print TensorBoard link
- [ ] Improve local dataset handling
- [ ] Add automated outlier detection

## Configuration Gotchas

- **Boolean Config Values**: Can be read as strings and always evaluate to True
- **Language Codes**: Use ISO 639-1 two-letter codes
- **Unknown Language**: Use `??` token consistently