# Development Notes
## Dependency Lockfile

We support deterministic installs via a lockfile using `uv` (preferred) or `pip-tools`.

- Generate (uv):
  - `uv pip compile requirements.txt -o requirements.lock`
- Generate (pip-tools):
  - `pip-compile --generate-hashes -o requirements.lock requirements.txt`

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

## Export to GGML

Convert models for whisper.cpp:
```bash
python ./convert-h5-to-ggml.py <checkpoint_dir> <whisper_openai_dir>
```

Example:
```bash
python ./convert-h5-to-ggml.py \
  ../../whisper/distil-whisper/training/output/checkpoint-100-epoch-7/ \
  ../../whisper-openai
```

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
- `distil-whisper/`: Modified from official DistilWhisper repo
- Modified files: `run_eval.py`, `run_distillation.py`
- TODO: Extract modified files instead of keeping entire repo

### Pending Tasks
- [ ] Make finetune.py print TensorBoard link
- [ ] Extract distil-whisper modifications
- [ ] Improve local dataset handling
- [ ] Add automated outlier detection

## Configuration Gotchas

- **Boolean Config Values**: Can be read as strings and always evaluate to True
- **Language Codes**: Use ISO 639-1 two-letter codes
- **Unknown Language**: Use `??` token consistently