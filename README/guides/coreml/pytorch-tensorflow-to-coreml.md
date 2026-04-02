# CUDA to CoreML Conversion Guide

## Prerequisites & Setup

```bash
pip install coremltools torch torchvision
# Use CoreML Tools 8.0b1+ for latest features
```

**Critical Hardware Context:**
- **Apple Neural Engine (ANE)**: Power-efficient AI accelerator, FP16 only, not directly programmable
- **Unified Memory Architecture**: CPU/GPU/ANE share memory - no explicit transfers needed
- **ANE falls back to GPU/CPU** for unsupported ops, dynamic shapes, or FP32 precision

## Direct PyTorch → CoreML (Recommended)

**DO NOT use ONNX intermediate format** - it's legacy and unmaintained.

```python
import torch
import coremltools as ct

# 1. CRITICAL: Set model to eval mode
model = torch.load('model.pth').eval()

# 2. Trace the model
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 3. Convert with proper preprocessing
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(
        shape=example_input.shape,
        scale=1.0/(0.229*255.0),  # 1/std per channel
        bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],  # -mean/std
        color_layout=ct.colorlayout.RGB
    )],
    convert_to="mlprogram",  # Use modern format
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL
)

# 4. Save
coreml_model.save("model.mlpackage")
```

## TensorFlow → CoreML

```python
import tensorflow as tf
import coremltools as ct

# Load as Keras model first (validates model integrity)
keras_model = tf.keras.models.load_model('model.h5')

coreml_model = ct.convert(
    keras_model,
    inputs=[ct.ImageType(
        shape=(1, 224, 224, 3),
        scale=2.0/255.0,  # Example: [0,255] → [-1,1]
        bias=-1.0
    )],
    convert_to="mlprogram"
)
```

## Common Failures & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `NotImplementedError: Unsupported op: X` | Missing CoreML operator | Create composite operator (see below) |
| `can't convert cuda:0 device type tensor` | Tensors on GPU during trace | `model.cpu()` and `example_input.cpu()` |
| Output is garbage/NaN | Wrong preprocessing or precision | Fix scale/bias OR force FP32 |
| Crashes with different input size | Fixed shape conversion | Use `EnumeratedShapes` or `RangeDim` |
| Model works in simulator, fails on device | Dynamic shape issue | Re-convert with flexible inputs |

### Fix Unsupported Operations

**Preferred: Composite Operators**
```python
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

@register_torch_op
def selu(context, node):
    x = _get_inputs(context, node, expected=1)[0]
    
    # SELU = scale * ELU(x, alpha)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    
    x = mb.elu(x=x, alpha=alpha)
    x = mb.mul(x=x, y=scale, name=node.name)
    context.add(x)
```

### Fix Dynamic Shapes

```python
# Option 1: Enumerate specific shapes (best for ANE)
flexible_input = ct.TensorType(
    shape=ct.EnumeratedShapes([(1, 128), (1, 256), (1, 512)])
)

# Option 2: Range of shapes
flexible_input = ct.TensorType(
    shape=(1, ct.RangeDim(lower_bound=1, upper_bound=1024))
)

model = ct.convert(traced_model, inputs=[flexible_input])
```

### Fix Precision Issues

```python
# Force FP32 for debugging
model = ct.convert(
    traced_model,
    compute_precision=ct.precision.FLOAT32,
    compute_units=ct.ComputeUnit.CPU_ONLY  # Ensures FP32 execution
)

# Verify conversion accuracy
pytorch_output = pytorch_model(test_input).detach().numpy()
coreml_output = coreml_model.predict({'input': test_input.numpy()})['output']
np.testing.assert_allclose(pytorch_output, coreml_output, rtol=1e-3)
```

## Optimization Workflow

### 1. Quantization (Apply First)

```python
# INT8 quantization (4x compression, ANE optimized)
quantized_model = ct.optimize.coreml.linear_quantize_weights(
    model, config=ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8"
        )
    )
)

# 6-bit palettization (5.3x compression)
palettized_model = ct.optimize.coreml.palettize_weights(model, nbits=6)

# Mixed-bit optimization
config = ct.optimize.coreml.OptimizationConfig(
    global_config={"n_bits": 6},
    op_type_configs={
        "conv": {"n_bits": 4},
        "linear": {"n_bits": 8}
    }
)
```

### 2. Model-Specific Optimizations

**Large Language Models:**
```python
# Stateful models with KV-cache
mlmodel = ct.convert(
    llm_model,
    convert_to="mlprogram",
    states=[ct.StateType(
        wrapped_type=ct.TensorType(shape=(batch, heads, seq_len, head_dim)),
        name="key_cache"
    )]
)

# INT4 quantization for LLMs
quantized = ct.optimize.coreml.linear_quantize_weights(
    mlmodel,
    config=ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block"
        )
    )
)
```

## Hardware-Specific Considerations

### Neural Engine Compatibility Rules
- **FP16 only** (FP32 forces CPU/GPU fallback)
- **No dynamic shapes** (use EnumeratedShapes with ≤128 variants)
- **No dilated convolutions** (dilation_rate > 1)
- **No custom layers**
- **4D tensor preference**: (Batch, Channels, 1, Sequence)

### Compute Unit Selection
```swift
let config = MLModelConfiguration()
config.computeUnits = .all  // Default: Let CoreML optimize

// Decision matrix:
// .all → Best for standard models, ANE optimization
// .cpuAndGPU → Large models, custom operations
// .cpuOnly → Debugging, guaranteed FP32 precision
```

### Performance by Chip
| Chip | Memory Bandwidth | ANE TOPS | Best For |
|------|------------------|----------|----------|
| M1 Max | 400 GB/s | 15.8 | Large models |
| M2 Ultra | 800 GB/s | 31.6 | Multi-model pipelines |
| M4 Max | 546 GB/s | 38 | Latest optimizations |

## Debugging & Profiling

### Xcode Performance Reports
1. Open `.mlpackage` in Xcode
2. Navigate to Performance tab
3. Run test on connected device
4. Check which operations run on ANE vs GPU/CPU

### Programmatic Debugging
```swift
let computePlan = try model.newComputePlan()
for operation in computePlan.operations {
    print("Operation: \(operation.name)")
    print("Compute device: \(operation.computeDevice)")
    print("Estimated cost: \(operation.estimatedCost)")
}
```

### Memory Analysis
```python
# Check model size progression
original_size = os.path.getsize("original_model.mlpackage")
quantized_size = os.path.getsize("quantized_model.mlpackage")
print(f"Compression ratio: {original_size / quantized_size:.1f}x")
```

## Real-World Performance Targets

**Stable Diffusion (512x512, 20 steps):**
- iPhone 14 Pro Max: 7.9s
- M2 Ultra: ~20s for SDXL 1024x1024

**DistilBERT on iPhone 13:**
- ANE: 3.47ms @ 0.454W (10x faster than CPU)

**Llama-3.1-8B:**
- M1 Max: ~33 tokens/s with optimizations

## Production Checklist

- [ ] Model in `eval()` mode before tracing
- [ ] Correct preprocessing (scale/bias) baked into model
- [ ] Use `mlprogram` format
- [ ] Test accuracy with `assert_allclose`
- [ ] Apply quantization (start with INT8)
- [ ] Profile on actual device (not simulator)
- [ ] Verify ANE usage in Xcode Performance Reports
- [ ] Test with representative input variations
- [ ] Measure memory usage under load
- [ ] Implement fallback for edge cases

## Integration Examples

### Swift Integration
```swift
import CoreML

guard let model = try? MLModel(contentsOf: modelURL) else { return }

let prediction = try model.prediction(from: input)
let output = prediction.featureValue(for: "output")?.multiArrayValue
```

### Memory-Efficient Loading
```swift
class ModelManager {
    private var cachedModels: [String: MLModel] = [:]
    private let maxModels = 3
    
    func loadModel(_ name: String) -> MLModel? {
        if cachedModels.count >= maxModels {
            cachedModels.removeAll()
        }
        return try? MLModel(contentsOf: modelURL(for: name))
    }
}
```

## When Things Break

1. **Always start with CPU-only conversion** to isolate hardware-specific issues
2. **Force FP32** to eliminate precision problems
3. **Use smaller test inputs** to debug shape issues
4. **Check Apple Developer Forums** for device-specific quirks
5. **Test on oldest supported hardware** to catch edge cases

This guide covers 90% of real-world conversion scenarios. For edge cases, refer to Apple's CoreML documentation and the coremltools GitHub issues.