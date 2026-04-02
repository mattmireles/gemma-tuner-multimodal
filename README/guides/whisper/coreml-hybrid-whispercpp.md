# **Whisper CoreML Conversion Guide**

## **Part 1: Technical Foundation**

### **1.1 Key Architectures**

- **Whisper:** Encoder-decoder Transformer. Audio → 16kHz → 80-channel log-mel spectrogram → encoder → hidden states → decoder → text tokens. The stateless encoder (parallelizable) and stateful decoder (sequential) split drives the conversion strategy.
- **Distil-Whisper:** Knowledge-distilled models with fewer decoder layers but identical encoders. Example: `distil-large-v3` is 6.3x faster, 49% smaller, within 1% WER of `whisper-large-v3`. Key insight: `distil-large-v2` uses the exact same encoder as `whisper-large-v2`, enabling conversion shortcuts.
    
- **CoreML:** Abstraction layer dispatching to CPU/GPU/ANE. First load triggers `ANECompilerService` to partition the graph and compile device-specific code (cached for future use).
    

### **1.2 ANE Requirements**

- **FP16 Only:** ANE requires 16-bit precision. FP32 operations fall back to GPU/CPU.
- **Data Layout `(B, C, 1, S)`:** ANE needs 4D channels-first format. PyTorch uses 3D `(B, S, C)`. Solution: Replace `nn.Linear` with 1x1 `nn.Conv2d`.
- **Limited Operators:** ANE supports a subset of ops. Unsupported ops break the chain and fall back to GPU/CPU. Recent additions include `scaled_dot_product_attention` (iOS 18+).
    
- **Dynamic Shapes:** Use `ct.RangeDim` or `ct.EnumeratedShapes`. Fully dynamic reshapes unsupported. `EnumeratedShapes` safer for ANE.
    

### **1.3 Environment Setup**

**Requirements:**
- Python 3.11 (whisper.cpp standard)
- Full Xcode installation (not just CLI tools)
- Python packages: `torch`, `openai-whisper`, `coremltools>=8.0`, `ane_transformers`
```bash
conda create -n whisper-coreml python=3.11 -y
conda activate whisper-coreml

pip install torch torchvision openai-whisper 'coremltools>=8.0' ane_transformers

xcode-select --install
```
    

## **Part 2: The `whisper.cpp` Hybrid Approach (Recommended)**

### **2.1 Why This Works Best**

- **Split architecture:** ANE-accelerated encoder (stateless, parallel) + C++ decoder (stateful, sequential)
- **Avoids decoder conversion issues:** No KV cache management in CoreML
- **Mixed quantization:** FP16 encoder on ANE + 4/5-bit quantized decoder on CPU

### **2.2 Quick Start**

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp/models
./generate-coreml-model.sh base.en  # Creates ggml-base.en-encoder.mlmodelc

# Build with CoreML support
cd ..
make WHISPER_COREML=1
```

### **2.3 What the Conversion Script Does**

Key modifications for ANE compatibility:
- **Linear→Conv2d:** Replaces `nn.Linear` with 1x1 `nn.Conv2d` using `ane_transformers`
- **Disable SDPA:** Sets `use_sdpa = False` to avoid PyTorch version conflicts
- **Dynamic shapes:** Uses `ct.RangeDim(1, 3000)` for variable audio lengths
- **FP16 conversion:** Weights converted to 16-bit precision

### **2.4 Runtime Integration**

- **Naming convention:** `ggml-base.en.bin` → looks for `ggml-base.en-encoder.mlmodelc`
- **Execution flow:**
  1. C++ loads audio → log-mel spectrogram
  2. Objective-C++ wrapper → CoreML encoder (ANE)
  3. Encoder output → C++ decoder (CPU with ggml)
  4. Decoder performs beam search, sampling → final text

## **Part 3: Pure CoreML Conversion (Advanced)**

### **3.1 The Challenge**

- **Stateful decoder:** Autoregressive generation requires maintaining KV cache across inference calls
- **Performance overhead:** Passing large KV cache tensors between app and CoreML kills performance

### **3.2 Encoder Conversion**

Same process as whisper.cpp but manual:
1. Load PyTorch model
2. Apply ANE patches (Linear→Conv2d)
3. Disable SDPA
4. Convert with `ct.RangeDim`
    

### **3.3 Decoder Conversion: KV Cache Solutions**

**Pre-iOS 18 (Manual KV Cache):**
- Model is stateless, KV cache passed as input/output each step
- Swift app manages loop: token + KV cache in → logits + new KV cache out
- **Problem:** Copying large KV cache every token kills performance
**iOS 18+ (Stateful Models with `MLState`):**
- CoreML maintains KV cache internally via `MLState` object
- App only passes new tokens, no cache copying
- Much more efficient, but requires latest OS
```swift
// Conceptual MLState-based decoding
let decoderState = try decoderModel.makeState()
var tokens: [Int] = [START_TOKEN]

for _ in 0..<MAX_LENGTH {
    let input = MLFeatureProvider(token: tokens.last!)
    let output = try decoderModel.prediction(from: input, state: decoderState)
    
    let nextToken = argmax(output.logits)
    tokens.append(nextToken)
    
    if nextToken == END_TOKEN { break }
}
```
    

## **Part 4: The Distil-Whisper Shortcut**

### **4.1 Key Insight**

Distil-Whisper models keep the parent encoder unchanged - only the decoder is reduced. Example: `distil-large-v2` uses the same encoder as `whisper-large-v2`.

### **4.2 Skip Conversion: Just Rename**

1. Download `ggml-distil-large-v2.bin` from Hugging Face
2. Download parent encoder: `ggml-large-v2-encoder.mlmodelc.zip`
3. Rename to match: `ggml-distil-large-v2-encoder.mlmodelc`
4. Place both in same directory and run

No conversion needed - the encoders are identical.

## **Part 5: Troubleshooting**

### **5.1 ANECompilerService Hangs**

**Symptom:** Conversion freezes, ANECompilerService at 100% CPU

**Fix:**
```bash
sudo kill -9 $(pgrep ANECompilerService)
```

The conversion will complete immediately after killing the process.

### **5.2 Model Not Running on ANE**

**Diagnosis:**
1. Xcode: Drag `.mlpackage` → Performance tab → Check device assignment
2. Use CoreMLProfiler for detailed reasons

**Common fixes:**
- Update `coremltools` (new ops added regularly)
- Ensure all ops are FP16
- Check operator support table below

| Operator | ANE Support | Notes |
| --- | --- | --- |
| Linear (as Conv2d) | ✅ | Requires conversion |
| LayerNorm, GELU, Softmax | ✅ | Well supported |
| SDPA | ✅ (iOS 18+) | Manual attention for older |
| einsum | ⚠️ | Only specific patterns |
| Dynamic Reshape | ❌ | Use static shapes |
| Transpose, Gather | ✅ | May have overhead |

### **5.3 PyTorch Version Conflicts**

**Error:** TracerWarning with `scaled_dot_product_attention`

**Fix:**
```python
# Before loading model
import whisper.model
whisper.model.MultiHeadAttention.use_sdpa = False

model = whisper.load_model("base")
```
    

### **5.4 Gibberish Output**

**Cause:** Usually preprocessing mismatch or FP16 precision loss

**Debug:**
1. Test with CPU-only (FP32) to isolate precision issues:
   ```python
   mlmodel = ct.convert(model, compute_units=ct.ComputeUnit.CPU_ONLY)
   ```
2. Verify preprocessing matches exactly:
   ```python
   assert np.allclose(python_mel, native_mel, atol=1e-5)
   ```

## **Part 6: Verification & Performance**

### **6.1 Verify Correctness**

```python
# Compare PyTorch vs CoreML outputs
torch_output = torch_encoder(input_mel).detach().numpy()
coreml_output = coreml_encoder.predict({'mel': input_mel.numpy()})['output']

assert np.allclose(torch_output, coreml_output, atol=1e-2)  # FP16 tolerance
```
    

### **6.2 Profile Performance**

**Xcode:** Select `.mlpackage` → Performance tab → Generate report
- Shows device assignment per layer (ANE/GPU/CPU)
- Loading and prediction times

**Instruments:** Use Core ML template for runtime profiling

**CoreMLProfiler:** Provides detailed reasons why ops aren't on ANE

### **6.3 Quantization Options**

- **CoreML encoder:** Already FP16, can use 8-bit or 4-bit (coremltools 8.0+)
- **whisper.cpp decoder:** Supports mature 4/5/8-bit quantization via ggml
- Always re-verify accuracy after quantization

## **Part 7: Summary**

### **7.1 Choose Your Approach**

| Approach | Best For | Trade-offs |
|----------|----------|------------|
| **whisper.cpp Hybrid** | Production, all OS versions | C++ dependency |
| **Pure CoreML + MLState** | Swift-only, iOS 18+ | Latest OS only, experimental |
| **Distil-Whisper Rename** | Quick deployment | Only for distil models |


### **7.2 Quick Checklist**

- [ ] Python 3.11 + ARM64 environment
- [ ] For Distil-Whisper: try rename first
- [ ] Apply ANE patches: Linear→Conv2d, disable SDPA
- [ ] Verify outputs match: `np.allclose(torch_out, coreml_out, atol=1e-2)`
- [ ] Verify preprocessing matches exactly
- [ ] Profile in Xcode to confirm ANE usage
- [ ] If hanging: `kill -9 ANECompilerService`