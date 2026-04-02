# Deployment Guide: From Training to Production

This guide shows you how to deploy your fine-tuned Whisper models for real-world inference across different platforms and use cases. Whether you're building a mobile app, server API, or edge device application, this guide provides the code and best practices you need.

## Quick Decision Guide

Choose your deployment format based on your use case:

| Use Case | Recommended Format | Why |
|----------|-------------------|-----|
| **iOS/macOS App** | CoreML + GGUF | ANE acceleration, native integration |
| **Server/Cloud API** | HuggingFace (SafeTensors) | Full framework support, GPU acceleration |
| **Edge Device/Raspberry Pi** | GGUF (quantized) | Minimal memory, CPU optimized |
| **Android App** | GGUF | Cross-platform, efficient |
| **Real-time Streaming** | GGUF | Low latency, streaming support |
| **Research/Experimentation** | HuggingFace | Maximum flexibility |

## Performance Comparison

| Format | Speed | Memory | Quality | Platform |
|--------|-------|--------|---------|----------|
| **HuggingFace** | Baseline | High (2-4GB) | 100% | All |
| **GGUF (FP16)** | 10-50x faster | Medium (1-2GB) | 99.9% | All |
| **GGUF (INT8)** | 15-75x faster | Low (500MB-1GB) | 99% | All |
| **CoreML** | 20x faster* | Low (500MB) | 99.9% | Apple only |

*On Apple Neural Engine

---

## Method 1: Using GGUF with whisper.cpp

GGUF format provides the fastest CPU inference and is automatically generated after training. It's ideal for production deployments where speed and efficiency matter.

### Installation

```bash
# Clone whisper.cpp if not already present
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

# Build for your platform
make -j  # Linux/Mac
# OR
cmake -B build && cmake --build build -j  # Cross-platform
```

### Basic Transcription

After training, your GGUF model is at `output/<run-id>/model-f16.gguf`. Use it directly:

```bash
# Simple transcription
./whisper.cpp/main -m output/47-wizard_20250812_175449/model-f16.gguf -f audio.wav

# With language forcing (for better accuracy if language is known)
./whisper.cpp/main -m output/47-wizard_20250812_175449/model-f16.gguf -f audio.wav -l en

# Output to file
./whisper.cpp/main -m output/47-wizard_20250812_175449/model-f16.gguf -f audio.wav -otxt -of output
```

### Advanced Options

```bash
# Real-time transcription from microphone
./whisper.cpp/stream -m output/47-wizard_20250812_175449/model-f16.gguf -t 8 --step 0 --length 5000

# Batch processing multiple files
for file in *.wav; do
    ./whisper.cpp/main -m output/47-wizard_20250812_175449/model-f16.gguf -f "$file" -otxt
done

# With Voice Activity Detection (VAD) for long audio
./whisper.cpp/main -m output/47-wizard_20250812_175449/model-f16.gguf -f podcast.mp3 --vad

# JSON output with timestamps
./whisper.cpp/main -m output/47-wizard_20250812_175449/model-f16.gguf -f audio.wav -ojson -ml 1
```

### Python Integration

```python
import subprocess
import json

def transcribe_with_whisper_cpp(audio_path, model_path):
    """Transcribe audio using whisper.cpp via subprocess."""
    cmd = [
        "./whisper.cpp/main",
        "-m", model_path,
        "-f", audio_path,
        "-ojson",      # JSON output
        "-np",         # No prints
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Parse JSON output
        output_file = audio_path.replace('.wav', '.json')
        with open(output_file, 'r') as f:
            transcription = json.load(f)
        return transcription['transcription']
    else:
        raise Exception(f"Transcription failed: {result.stderr}")

# Usage
model = "output/47-wizard_20250812_175449/model-f16.gguf"
text = transcribe_with_whisper_cpp("audio.wav", model)
print(f"Transcription: {text}")
```

### Quantization for Smaller Models

```bash
# Convert to INT8 (2x smaller, 1.5x faster, <1% quality loss)
./whisper.cpp/quantize output/47-wizard_20250812_175449/model-f16.gguf output/model-q8_0.gguf q8_0

# Convert to INT4 (4x smaller, 2x faster, ~1-2% quality loss)
./whisper.cpp/quantize output/47-wizard_20250812_175449/model-f16.gguf output/model-q4_0.gguf q4_0

# Use quantized model
./whisper.cpp/main -m output/model-q8_0.gguf -f audio.wav
```

---

## Method 2: CoreML Deployment (Apple Devices)

CoreML provides the best performance on Apple devices by leveraging the Apple Neural Engine (ANE).

### Export to CoreML

```bash
# Export your trained model to CoreML format
python -m whisper_tuner.scripts.export_coreml output/47-wizard_20250812_175449

# This creates:
# output/47-wizard_20250812_175449/coreml/whisper-encoder.mlmodelc
# output/47-wizard_20250812_175449/model-f16.gguf (decoder)
```

### Swift Integration (iOS/macOS)

```swift
import CoreML
import AVFoundation

class WhisperInference {
    let encoder: MLModel
    let decoderPath: String
    
    init(modelPath: String) throws {
        // Load CoreML encoder
        let encoderURL = URL(fileURLWithPath: "\(modelPath)/coreml/whisper-encoder.mlmodelc")
        self.encoder = try MLModel(contentsOf: encoderURL)
        
        // Store decoder path for whisper.cpp
        self.decoderPath = "\(modelPath)/model-f16.gguf"
    }
    
    func transcribe(audioURL: URL) async throws -> String {
        // 1. Load and preprocess audio to mel-spectrogram
        let audioData = try Data(contentsOf: audioURL)
        let melFeatures = preprocessAudio(audioData)  // Your preprocessing
        
        // 2. Run encoder on ANE
        let encoderInput = WhisperEncoderInput(mel_features: melFeatures)
        let encoderOutput = try await encoder.prediction(from: encoderInput)
        
        // 3. Run decoder with whisper.cpp (hybrid approach)
        // This is pseudocode - integrate with whisper.cpp C API
        let transcription = runWhisperDecoder(
            encoderOutput: encoderOutput,
            decoderPath: decoderPath
        )
        
        return transcription
    }
}

// Usage
let whisper = try WhisperInference(modelPath: "output/47-wizard_20250812_175449")
let text = try await whisper.transcribe(audioURL: audioFileURL)
```

### Python with CoreML

```python
import coremltools as ct
import numpy as np
import librosa

class CoreMLWhisperInference:
    def __init__(self, model_path):
        # Load CoreML encoder
        encoder_path = f"{model_path}/coreml/whisper-encoder.mlmodelc"
        self.encoder = ct.models.MLModel(encoder_path)
        
        # Store decoder path for hybrid inference
        self.decoder_path = f"{model_path}/model-f16.gguf"
        
    def transcribe(self, audio_path):
        # 1. Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        mel_features = self.extract_mel_features(audio)
        
        # 2. Run encoder on ANE
        encoder_input = {"mel_features": mel_features}
        encoder_output = self.encoder.predict(encoder_input)
        
        # 3. Run decoder with whisper.cpp subprocess
        # (CoreML encoder + GGUF decoder hybrid approach)
        transcription = self.run_gguf_decoder(encoder_output['output'])
        
        return transcription
    
    def extract_mel_features(self, audio):
        """Extract mel-spectrogram features."""
        # Implementation depends on your model's preprocessing
        # This is a simplified example
        mel = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_mels=80, n_fft=400, hop_length=160
        )
        log_mel = np.log10(np.maximum(mel, 1e-10))
        return log_mel

# Usage
model = CoreMLWhisperInference("output/47-wizard_20250812_175449")
text = model.transcribe("audio.wav")
print(f"Transcription: {text}")
```

---

## Method 3: HuggingFace Models (Python)

The HuggingFace format provides maximum flexibility and is ideal for research, experimentation, and server deployments with GPU acceleration.

### Basic Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Load your fine-tuned model
model_path = "output/47-wizard_20250812_175449"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Transcribe audio
def transcribe(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio to features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    
    # Decode tokens to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Usage
text = transcribe("audio.wav")
print(f"Transcription: {text}")
```

### Using core/inference.py Module

The project includes a unified inference module that handles audio processing, language modes, and scoring:

```python
import sys
sys.path.append('/path/to/whisper-fine-tuner-macos')

from whisper_tuner.core.inference import prepare_features, generate, decode_and_score
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load model and processor
model_path = "output/47-wizard_20250812_175449"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

def transcribe_with_language_control(audio_path, language_mode="mixed", forced_language=None):
    """
    Transcribe with language control using core.inference module.
    
    Language modes:
    - "mixed": Auto-detect language
    - "strict": Use provided language
    - "override:en": Force English (or any language)
    """
    # Prepare audio features
    input_features = prepare_features(audio_path, processor.feature_extractor)
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
    
    # Generate with language control
    generated_ids = generate(
        model=model,
        processor=processor,
        input_features=input_features,
        language_mode=language_mode,
        forced_language=forced_language,
        gen_kwargs={"max_new_tokens": 256}
    )
    
    # Decode to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Examples
text = transcribe_with_language_control("audio.wav", language_mode="mixed")
text_en = transcribe_with_language_control("audio.wav", language_mode="override:en")
text_es = transcribe_with_language_control("audio.wav", forced_language="es")
```

### Batch Processing

```python
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, audio_dir, processor):
        self.audio_files = list(Path(audio_dir).glob("*.wav"))
        self.processor = processor
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        return {
            "input_features": inputs.input_features.squeeze(0),
            "path": str(audio_path)
        }

# Batch transcription
def batch_transcribe(audio_dir, model_path, batch_size=8):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    dataset = AudioDataset(audio_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            generated_ids = model.generate(input_features)
            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for path, text in zip(batch["path"], transcriptions):
                results.append({"file": path, "transcription": text})
    
    return results

# Usage
results = batch_transcribe("audio_files/", "output/47-wizard_20250812_175449", batch_size=4)
for r in results:
    print(f"{r['file']}: {r['transcription']}")
```

---

## Platform-Specific Deployment

### Mobile Apps (iOS/Android)

**iOS (Swift)**:
```swift
// Use CoreML for best performance
// See CoreML section above for implementation
```

**Android (Kotlin)**:
```kotlin
// Use whisper.cpp Android bindings
// Build whisper.cpp for Android:
// cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake
// cmake --build build-android

class WhisperTranscriber(context: Context) {
    init {
        System.loadLibrary("whisper")
    }
    
    external fun transcribe(modelPath: String, audioPath: String): String
    
    fun transcribeAudio(audioFile: File): String {
        val modelPath = "${context.filesDir}/model-f16.gguf"
        return transcribe(modelPath, audioFile.absolutePath)
    }
}
```

### Server/API Deployment

**FastAPI Server Example**:
```python
from fastapi import FastAPI, UploadFile, File
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import io

app = FastAPI()

# Load model once at startup
MODEL_PATH = "output/47-wizard_20250812_175449"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read audio file
    audio_bytes = await file.read()
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    # Process and transcribe
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return {"transcription": transcription}

# Run with: uvicorn app:app --reload
```

### Edge Devices (Raspberry Pi)

```bash
# Use quantized GGUF for minimal memory usage
./whisper.cpp/quantize output/47-wizard_20250812_175449/model-f16.gguf model-q4_0.gguf q4_0

# Run with reduced threads for stability
./whisper.cpp/main -m model-q4_0.gguf -f audio.wav -t 2
```

### Real-time Streaming

```python
import subprocess
import threading
import queue

class WhisperStreamer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def start_streaming(self):
        """Start whisper.cpp stream mode."""
        cmd = [
            "./whisper.cpp/stream",
            "-m", self.model_path,
            "-t", "4",
            "--step", "0",
            "--length", "5000"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Read output in separate thread
        def read_output():
            for line in process.stdout:
                if line.strip():
                    self.result_queue.put(line.strip())
        
        thread = threading.Thread(target=read_output)
        thread.daemon = True
        thread.start()
        
        return process
    
    def get_transcription(self, timeout=1.0):
        """Get latest transcription."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# Usage
streamer = WhisperStreamer("output/47-wizard_20250812_175449/model-f16.gguf")
process = streamer.start_streaming()

# Read transcriptions as they come
while True:
    text = streamer.get_transcription()
    if text:
        print(f"Transcribed: {text}")
```

---

## Performance Optimization

### Memory Optimization

```python
# For HuggingFace models - use half precision
model = model.half()  # FP16 for GPU

# Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

### Speed Optimization

```python
# 1. Use torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 2. Use larger batch sizes for GPU
batch_size = 16 if torch.cuda.is_available() else 1

# 3. Disable gradient computation
model.eval()
with torch.no_grad():
    # inference code
```

### Quantization Comparison

| Quantization | Model Size | Speed | Quality Loss |
|--------------|------------|-------|--------------|
| FP16 (default) | 100% | 1x | 0% |
| INT8 (q8_0) | 50% | 1.5x | <1% |
| INT5 (q5_1) | 35% | 1.8x | ~1% |
| INT4 (q4_0) | 25% | 2x | 1-2% |

---

## Production Checklist

Before deploying to production, ensure:

- [ ] **Model Testing**: Validate on representative test set
- [ ] **Error Handling**: Implement fallbacks for failures
- [ ] **Monitoring**: Add logging and metrics
- [ ] **Resource Limits**: Set memory and CPU limits
- [ ] **Security**: Validate input files, sanitize outputs
- [ ] **Scaling**: Plan for concurrent requests
- [ ] **Backup**: Keep original model files backed up
- [ ] **Documentation**: Document model version and training data

---

## Troubleshooting

### Common Issues

**Issue**: "Model not found" error
```python
# Solution: Use absolute paths
import os
model_path = os.path.abspath("output/47-wizard_20250812_175449")
```

**Issue**: Out of memory errors
```bash
# Solution: Use quantized models or reduce batch size
./whisper.cpp/quantize model-f16.gguf model-q8_0.gguf q8_0
```

**Issue**: Slow inference on CPU
```python
# Solution: Use GGUF format instead of HuggingFace
# GGUF is 10-50x faster on CPU
```

**Issue**: Language detection issues
```python
# Solution: Force language when known
generated_ids = model.generate(
    input_features,
    language="en",  # Force English
    task="transcribe"
)
```

---

## Summary

You now have three powerful deployment options:

1. **GGUF with whisper.cpp**: Best for production, edge devices, and CPU inference
2. **CoreML**: Optimal for Apple devices with ANE acceleration  
3. **HuggingFace**: Maximum flexibility for research and GPU servers

Choose based on your platform and requirements. The automatic GGUF export after training means you're immediately ready for production deployment with whisper.cpp, while the HuggingFace format gives you full framework capabilities when needed.

For most production use cases, GGUF provides the best balance of speed, efficiency, and ease of deployment.
