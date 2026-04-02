# Fine-Tuning Whisper with Metal Performance Shaders Graph

## Introduction

This guide shows how to fine-tune OpenAI Whisper models directly on Apple Silicon using MPSGraph. Unlike high-level frameworks (PyTorch-MPS, MLX, Core ML), direct MPSGraph gives you complete control over GPU execution, memory management, and operator fusion for maximum performance.

You'll need Swift/Objective-C experience, Metal API knowledge, and transformer architecture understanding.

---

# Section 1: Mapping Whisper Architecture to MPSGraph

## 1.1 Encoder-Decoder Overview

Whisper is a sequence-to-sequence transformer with two components:
- **Audio encoder**: Processes fixed-length spectrograms (static graph, good for fusion)
- **Text decoder**: Auto-regressive generation (dynamic sequences, requires KV caching)

The encoder runs once per audio clip. The decoder generates tokens one at a time, requiring careful state management for efficient inference.

## 1.2 Input Pipeline

Convert raw audio to Mel spectrograms:
1. Resample audio to 16kHz
2. Create log-magnitude Mel spectrogram:
   - Whisper v1/v2: 80 channels
   - Whisper v3: 128 channels  
   - 25ms window, 10ms stride
3. Normalize to [-1,1] range

Use Accelerate (vDSP) for CPU processing, then create `MTLBuffer` and wrap in `MPSGraphTensor`. Shape: `[batch_size, num_mel_bins, sequence_length]`.

## 1.3 Encoder Components

### Convolutional Stem
Two 1D convolutions (filter width 3, GELU activation). Second conv has stride 2.
- Use `convolution2D` (treat 1D as 2D with height=1) + `gelu`

### Positional Embeddings  
Fixed sinusoidal embeddings added to conv stem output.
- Build with `sine`, `cosine`, `addition`, `multiplication` operations

### Transformer Encoder Block
Each block contains:
- **Multi-Head Self-Attention**: Use `scaledDotProductAttention` (single fused operation, much faster than manual Q/K/V projections)
- **Feed-Forward Network**: Two `matrixMultiplication` + `gelu` 
- **Residual + LayerNorm**: `addition` + `layerNormalization`
    

## 1.4 Decoder Components

### Transformer Decoder Block
- **Masked Self-Attention**: Causal mask prevents attending to future tokens. Use triangular mask with `scaledDotProductAttention`
- **Cross-Attention**: Decoder attends to encoder output. Query from decoder, Key/Value from encoder
- **FFN + Residual + LayerNorm**: Same as encoder

### Output Projection
`matrixMultiplication` to vocab size + `softMax` for token probabilities.

## 1.5 Parameter Mapping

| WhisperConfig Parameter | MPSGraph Implementation |
|---|---|
| `encoder_layers`, `decoder_layers` | Number of transformer blocks to stack |
| `d_model` | Hidden dimension size for most tensors |
| `encoder_attention_heads` | `numberOfHeads` for encoder attention |
| `decoder_attention_heads` | `numberOfHeads` for decoder attention |
| `encoder_ffn_dim`, `decoder_ffn_dim` | FFN intermediate dimensions |
| `activation_function` | Activation node type (e.g., `gelu`) |

---

# Section 2: Building the Training Graph

## 2.1 Graph Components

Three tensor types define training graphs:

- **Placeholders**: Dynamic inputs (audio batches, labels) - `graph.placeholder(shape:dataType:name:)`
- **Variables**: Trainable parameters (weights, biases) - `graph.variable(with:shape:dataType:name:)`  
- **Constants**: Fixed values (masks, hyperparameters) - `graph.constant(...)`

## 2.2 Automatic Differentiation

Use `graph.gradient(of:withRespectTo:name:)` to compute gradients from scalar loss to all variables.

**Debugging Strategy**: The gradient graph is opaque. Add "tap points" to fetch intermediate values (attention scores, activations) during forward pass. Check for `NaN`, infinity, or extreme values on CPU side - gradient issues often start in forward pass.

## 2.3 Graph Execution

Two execution methods:
- **`run`**: Synchronous, good for debugging/testing
- **`encode`**: Asynchronous, high-performance, gives full GPU timeline control

Both use `feeds` dictionary (placeholders → data) and `targetTensors` array (which outputs to read back).

---

# Section 3: Complete Training Implementation

## 3.1 Loading Weights from .safetensors

Use `.safetensors` format (more secure than `.pt`, simpler specification).

Process:
1. Read `.safetensors` file (JSON header + binary data)
2. Parse JSON manifest (tensor names, types, shapes)  
3. Extract byte slices for each tensor
4. Create `MTLBuffer` with correct data type
5. Initialize `MPSGraphVariable`s with these buffers

**Critical**: Match tensor names exactly between `.safetensors` and graph variables (e.g., `encoder.layers.0.self_attn.q_proj.weight`).

## 3.2 Cross-Entropy Loss

Build manually from primitives:
1. `softMax` on logits → probabilities
2. `log` on probabilities 
3. `oneHot` on ground truth labels
4. Element-wise `multiplication` 
5. `reductionSum` across vocab + `negate`
6. `reductionMean` across batch → scalar loss

## 3.3 AdamW Optimizer Implementation

MPSCNN has built-in optimizers, but MPSGraph requires building optimizer logic into the graph itself.

Create `MPSGraphVariable`s for optimizer state (m, v vectors). AdamW update sequence:

1. **Update moments**: `m = β1*m + (1-β1)*g`, `v = β2*v + (1-β2)*g²`
2. **Bias correction**: `m̂ = m/(1-β1^t)`, `v̂ = v/(1-β2^t)`  
3. **Calculate update**: `update = η*m̂/(√v̂ + ε)`
4. **Weight decay**: `w = w - η*λ*w`
5. **Apply update**: `w = w - update`

Build with `addition`, `subtraction`, `multiplication`, `square`, `squareRoot` operations. End with `graph.assign()` to update variables in-place.

## 3.4 Training Loop

```swift
func runTrainingStep(audioBatch: MPSGraphTensorData, labelBatch: MPSGraphTensorData) -> Float {
    var loss: Float = 0.0
    
    guard let commandBuffer = commandQueue.makeCommandBuffer() else { return Float.nan }
    
    let feeds = [
        audioPlaceholder: audioBatch,
        labelPlaceholder: labelBatch
    ]
    
    let targetTensors = [lossTensor]
    
    let results = trainingGraph.encode(
        to: commandBuffer,
        feeds: feeds,
        targetTensors: targetTensors,
        targetOperations: updateOperations
    )
    
    commandBuffer.addCompletedHandler { buffer in
        guard buffer.error == nil else {
            print("Command buffer failed: \(buffer.error!)")
            return
        }
        
        let lossData = results![lossTensor]!
        lossData.mpsndarray().readBytes(&loss, strideBytes: nil)
    }
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    return loss
}

// Training loop
for epoch in 0..<numEpochs {
    for (audioBatch, labelBatch) in dataLoader {
        let loss = runTrainingStep(audioBatch: audioBatch, labelBatch: labelBatch)
        print("Epoch \(epoch), Loss: \(loss)")
    }
}
```

---

# Section 4: Common Pitfalls and Solutions

## 4.1 Memory Management with MTLHeap

**Problem**: Memory fragmentation during training causes OOM crashes despite sufficient total memory.

**Solution**: Use `MTLHeap` for manual memory management:

1. **Measure peak usage**: Run dry training step or use `MPSHintTemporaryMemoryHighWaterMark`
2. **Create heap**: `MTLHeap` with `MTLHeapType.placement` for peak size
3. **Manual allocation**: Create `MTLBuffer`s from heap for large tensors
4. **Resource aliasing**: Reuse memory regions for non-overlapping tensor lifetimes
5. **Sync**: Use `MTLFence`/`MTLEvent` to prevent race conditions

## 4.2 Variable-Length Audio Optimization

**Problem**: Fixed 30-second padding wastes computation on shorter clips.

**Solution**: Dynamic shapes + bucketing:

1. **Dynamic placeholders**: Use `-1` for sequence length (e.g., `[-1, 80, -1]`)
2. **JIT compilation**: MPSGraph compiles specialized versions for each shape  
3. **Bucketing**: Sort dataset by length, batch similar lengths together

```swift
func createPaddedBatches(from spectrograms: [[Float]], batchSize: Int) -> [PaddedBatch] {
    let sortedSpectrograms = spectrograms.sorted { $0.count < $1.count }
    var batches: [PaddedBatch] = []
    
    for i in stride(from: 0, to: sortedSpectrograms.count, by: batchSize) {
        let batchSpectrograms = Array(sortedSpectrograms[i..<min(i + batchSize, sortedSpectrograms.count)])
        let maxLengthInBatch = batchSpectrograms.map { $0.count }.max() ?? 0
        let paddedData = batchSpectrograms.map { pad($0, to: maxLengthInBatch) }
        batches.append(PaddedBatch(data: paddedData, sequenceLength: maxLengthInBatch))
    }
    return batches
}
```

## 4.3 Mixed-Precision Training

**Problem**: `float16` training causes NaN values due to limited dynamic range.

**Solution**: Strategic precision + loss scaling:

1. **Strategic precision**: 
   - `float16`: Matrix ops, convolutions  
   - `float32`: Weights, optimizer state, loss calculation
2. **Loss scaling**: 
   - Scale loss by large factor (65536) before gradients
   - Unscale gradients before optimizer update

## 4.4 Custom Metal Kernels for Optimizer

**Problem**: Many small MPSGraph operations create CPU dispatch overhead (10-15μs vs 0.1μs GPU execution).

**Solution**: Custom MSL kernel for AdamW:

1. Write `.metal` file implementing full AdamW logic
2. Hybrid model: MPSGraph for main computation, custom kernel for optimizer
3. Single kernel dispatch per variable vs dozens of MPSGraph operations

## 4.5 Debugging GPU Crashes

**Problem**: Generic Metal errors with no insight into which operation failed.

**Solution**: Multi-tool debugging workflow:

1. **Export IR**: Set `MPS_GRAPH_DUMP_IR=1` to dump MLIR representation
2. **Visualize**: Use `mpsgraphtool` to create `.mpsgraphpackage` file  
3. **Inspect**: Open in Xcode MPSGraph Viewer to see exact graph topology, fusion, and tensor shapes

---

# Section 5: Performance Optimization

## 5.1 Quantization-Aware Training

QAT adapts weights to lower precision during training (vs post-training quantization).

**Implementation**: Insert fake quantization in forward pass:
1. `graph.quantize()` → `graph.dequantize()` for each weight
2. Master weights stay `float32`, used weights experience quantization loss  
3. **Key optimization**: `dequantize` + `matrixMultiplication` fuses into `quantizedMatrixMultiplication`

## 5.2 KV Cache for Inference

**Problem**: Recomputing K/V projections for entire sequence each token is O(n²).

**Solution**: Cache K/V tensors, append new tokens:
1. **Stateful variables**: `MPSGraphVariable`s for K/V caches per decoder layer
2. **In-place updates**: `graph.sliceUpdateDataTensor()` to append new K/V  
3. **Efficient attention**: Matrix-matrix → matrix-vector multiplication

## 5.3 Profiling with Instruments

- **Metal System Trace**: CPU/GPU timeline, identify idle "bubbles" from dispatch overhead
- **GPU Counters**: ALU occupancy, memory bandwidth, instruction mix for kernel optimization

---

## Conclusion

Direct MPSGraph implementation provides maximum control and performance for Whisper fine-tuning on Apple Silicon. Key techniques covered:

**Architecture**: Map transformer components to MPSGraph operations using fused attention and proper variable management.

**Training**: Build complete training graphs with automatic differentiation, proper loss functions, and graph-native optimizers.

**Optimization**: Use `MTLHeap` for memory management, dynamic shapes for variable audio lengths, mixed-precision training, and custom Metal kernels where needed.

**Debugging**: Leverage MPSGraph Viewer and tap points for visibility into complex training graphs.

This approach trades framework convenience for ultimate performance and control - essential for production Apple Silicon ML applications.