# Advanced LoRA Guide

This guide provides a comprehensive technical overview of Low-Rank Adaptation (LoRA) for Whisper models. It covers everything from fundamental concepts to advanced optimization strategies, with a special focus on Apple Silicon deployment.

## Introduction: Why LoRA Changes Everything

Low-Rank Adaptation (LoRA) represents a paradigm shift in how we approach model fine-tuning. Instead of updating all parameters of a pre-trained model, LoRA injects small, trainable adapter modules while keeping the original model frozen. This approach reduces trainable parameters by 99%+ while maintaining comparable performance.

### The Core Insight

LoRA is based on a fundamental observation: the weight updates during fine-tuning often have a low intrinsic rank. Instead of updating a weight matrix **W** directly, LoRA approximates the update **ΔW** as the product of two smaller matrices:

```
ΔW = B × A
```

Where:
- **A** is a matrix of size `[rank, input_dim]`
- **B** is a matrix of size `[output_dim, rank]`
- **rank** << min(input_dim, output_dim)

During forward pass: `output = (W + ΔW) × input = W × input + B × A × input`

### Why This Matters for Whisper

Whisper models are large (244M+ parameters) and memory-intensive to fine-tune. LoRA enables:

- **Accessibility**: Fine-tune large models on consumer hardware
- **Efficiency**: 50-80% reduction in memory usage and training time
- **Modularity**: Create domain-specific adapters that can be swapped instantly
- **Preservation**: Keep the original model intact for multiple use cases

## Strategic LoRA Application for Whisper

### Understanding Whisper's Architecture for LoRA

Whisper consists of two main components, each with different LoRA considerations:

#### Encoder: The Audio Understanding Engine
- **Function**: Transforms audio spectrograms into semantic representations
- **Layers**: Primarily attention mechanisms for temporal modeling
- **LoRA Impact**: Adapting encoder attention helps with audio domain adaptation (noise characteristics, recording quality, speaker accents)

#### Decoder: The Text Generation Engine  
- **Function**: Converts semantic representations to text tokens
- **Layers**: Causal attention + cross-attention + feedforward networks
- **LoRA Impact**: Adapting decoder layers helps with vocabulary, linguistic style, and domain-specific terminology

### Target Module Selection Strategy

The framework targets specific modules within each transformer layer:

| Module | Function | LoRA Impact | Priority |
|--------|----------|-------------|----------|
| `q_proj` | Query projection for self-attention | High - affects what model pays attention to | **Essential** |
| `k_proj` | Key projection for self-attention | High - affects attention patterns | **Essential** |
| `v_proj` | Value projection for self-attention | High - affects information extraction | **Essential** |
| `out_proj` | Output projection after attention | Medium - affects information integration | **Recommended** |
| `fc1` | First feedforward layer | Medium - affects feature transformation | **Recommended** |
| `fc2` | Second feedforward layer | Medium - affects final representation | **Recommended** |

**Default Configuration**: All six modules are targeted for balanced adaptation across attention and feedforward components.

## Configuration Deep Dive

### Core LoRA Parameters

#### `lora_r` (Rank) - The Capacity Controller
- **Range**: 1-512 (practical range: 8-128)
- **Default**: 32
- **Trade-off**: Higher rank = more adaptation capacity but more parameters

| Rank | Trainable Params (whisper-small) | Use Case |
|------|--------------------------------|----------|
| 8 | ~0.1M (0.04%) | Subtle domain adaptation |
| 16 | ~0.2M (0.08%) | Light fine-tuning |
| 32 | ~0.4M (0.16%) | **Recommended balance** |
| 64 | ~0.8M (0.33%) | Heavy adaptation |
| 128 | ~1.6M (0.66%) | Maximum adaptation (rarely needed) |

#### `lora_alpha` (Scaling Factor) - The Adaptation Strength
- **Range**: 1-256 (practical range: rank to 4×rank)
- **Default**: 64 (2× rank)
- **Formula**: `scaling = lora_alpha / lora_r`
- **Purpose**: Controls how much the LoRA adaptation affects the final output

**Recommended Alpha Values**:
- Conservative adaptation: `alpha = rank` (scaling = 1.0)
- Balanced adaptation: `alpha = 2 × rank` (scaling = 2.0) **← Default**
- Aggressive adaptation: `alpha = 4 × rank` (scaling = 4.0)

#### `lora_dropout` - Regularization for Adapters
- **Range**: 0.0-0.3
- **Default**: 0.07
- **Purpose**: Prevents overfitting in LoRA layers
- **Recommendation**: Start with default; increase if overfitting occurs

### Hardware-Specific Optimization

#### Apple Silicon Configuration

For optimal MPS performance, the framework uses specific settings:

```ini
[group:whisper-lora]
dtype = float32              # MPS compatibility (vs bfloat16)
attn_implementation = sdpa   # Scaled Dot Product Attention
lora_r = 32                 # Balanced for M1/M2 memory bandwidth
lora_alpha = 64             # 2x scaling for stable training
enable_8bit = False         # Disabled by default for stability
```

**Why `dtype = float32`?**
- Broader MPS operation support than `bfloat16`
- More stable numerics for parameter-efficient training
- Negligible memory difference at LoRA scale

#### Memory Usage by Apple Silicon Variant (With Flash Attention 2)

| Model Size | M1/M2/M3 (8GB) | M1/M2/M3 (16GB) | M1/M2/M3 Pro (16-32GB) | M1/M2/M3 Max (32-64GB) | M1/M2/M3 Ultra (64-192GB) |
|------------|-----------------|------------------|------------------------|------------------------|---------------------------|
| **whisper-small + LoRA** | ❌ Too tight | ✅ Batch 12-20 | ✅ Batch 20-28 | ✅ Batch 28-40+ | ✅ Batch 40-60+ |
| **whisper-medium + LoRA** | ❌ No | ✅ Batch 6-10 | ✅ Batch 10-18 | ✅ Batch 18-28 | ✅ Batch 28-40+ |
| **whisper-large-v2 + LoRA** | ❌ No | ❌ Too tight | ✅ Batch 4-8 | ✅ Batch 8-14 | ✅ Batch 14-24+ |

*Note: With Flash Attention 2 enabled (default in PyTorch 2.3+), memory usage is ~28% lower, allowing these larger batch sizes.*

### Advanced Configuration Strategies

#### Strategy 1: Encoder-Only Adaptation
For audio domain adaptation (new microphones, environments, accents):

```ini
[profile:encoder-only-lora]
model = whisper-small-lora
lora_target_modules = encoder.q_proj,encoder.k_proj,encoder.v_proj,encoder.out_proj
lora_r = 48
lora_alpha = 96
```

#### Strategy 2: Decoder-Only Adaptation  
For vocabulary/style adaptation (medical terms, legal language, slang):

```ini
[profile:decoder-only-lora]
model = whisper-small-lora
lora_target_modules = decoder.q_proj,decoder.k_proj,decoder.v_proj,decoder.out_proj,decoder.fc1,decoder.fc2
lora_r = 64
lora_alpha = 128
```

#### Strategy 3: Minimal Resource Adaptation
For extremely memory-constrained scenarios:

```ini
[profile:minimal-lora]
model = whisper-small-lora
lora_target_modules = q_proj,v_proj
lora_r = 16
lora_alpha = 16
lora_dropout = 0.1
```

## 8-Bit Quantization with LoRA

### When to Use 8-Bit + LoRA

8-bit quantization reduces the base model's memory footprint by ~50% while keeping LoRA adapters in full precision. This is ideal for:

- **Extreme memory constraints**: Training large models on 8GB systems
- **Batch size maximization**: More samples per batch within memory limits
- **Multi-adapter training**: Running multiple experiments simultaneously

### Configuration and Considerations

```ini
[group:whisper-lora-8bit]
dtype = float32
enable_8bit = True
lora_r = 32
lora_alpha = 64
per_device_train_batch_size = 32  # Can increase due to lower memory usage
```

**Important Notes**:
- Base model is quantized to 8-bit, adapters remain float32
- Slight quality trade-off for significant memory savings
- MPS support varies by PyTorch version
- Always validate results against full-precision baseline

### Memory Impact Comparison

| Configuration | whisper-small Memory | whisper-medium Memory | whisper-large-v2 Memory |
|---------------|---------------------|----------------------|------------------------|
| Standard Fine-tuning | ~16-24GB | ~28-40GB | ~48-64GB |
| LoRA (float32) | ~4-8GB | ~8-12GB | ~16-24GB |
| LoRA + 8-bit | ~2-4GB | ~4-6GB | ~8-12GB |

## Multiple Adapter Management

### The Modular Advantage

One of LoRA's greatest strengths is the ability to train multiple specialized adapters for a single base model:

```
whisper-small (base model, 244M params)
├── medical_adapter (0.4M params)
├── legal_adapter (0.4M params)
├── conversational_adapter (0.4M params)
└── podcast_adapter (0.4M params)
```

### Training Multiple Domain Adapters

#### Sequential Training Strategy
Train adapters one at a time for different domains:

```bash
# Medical domain
python main.py finetune medical-lora-data

# Legal domain  
python main.py finetune legal-lora-data

# Conversational domain
python main.py finetune conversation-lora-data
```

#### Parallel Training Strategy
Train multiple adapters simultaneously (requires sufficient memory):

```bash
# Terminal 1
python main.py finetune medical-lora-data --output_dir output/medical

# Terminal 2
python main.py finetune legal-lora-data --output_dir output/legal

# Terminal 3  
python main.py finetune conversation-lora-data --output_dir output/conversation
```

### Adapter Loading and Switching

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

# Load base model once
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load different adapters as needed
medical_model = PeftModel.from_pretrained(base_model, "output/medical/adapter_model")
legal_model = PeftModel.from_pretrained(base_model, "output/legal/adapter_model")

# Or switch adapters dynamically
model = PeftModel.from_pretrained(base_model, "output/medical/adapter_model")
model.load_adapter("output/legal/adapter_model", adapter_name="legal")
model.set_adapter("legal")  # Switch to legal adapter
```

## Apple Silicon Optimization Guide

### MPS-Specific Best Practices

#### Environment Setup
```bash
# Essential for development/testing
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Memory management (adjust based on your Mac's RAM)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

# Enable Flash Attention 2 (PyTorch 2.3+) - reduces memory by ~28%
export SDPA_ALLOW_FLASH_ATTN=1

# Debugging (optional)
export PYTORCH_DEBUG_MPS_ALLOCATOR=1
```

#### Flash Attention 2 Support (New in PyTorch 2.3)

Starting with PyTorch 2.3 (July 2025), Apple Silicon now supports Flash Attention 2:

- **Memory Reduction**: ~28% lower peak memory usage
- **Performance**: Faster attention computation for sequences ≤ 4096
- **Compatibility**: Works seamlessly with LoRA adapters
- **Enable**: Set `export SDPA_ALLOW_FLASH_ATTN=1` before training

This dramatically improves LoRA training efficiency, allowing larger batch sizes and faster convergence on M-series chips.

#### Batch Size Optimization Strategy

Start conservative and scale up:

```bash
# Step 1: Verify training works
python main.py finetune small-lora-test --override per_device_train_batch_size=4

# Step 2: Find optimal batch size
python main.py finetune small-lora-test --override per_device_train_batch_size=8
python main.py finetune small-lora-test --override per_device_train_batch_size=16
python main.py finetune small-lora-test --override per_device_train_batch_size=24

# Step 3: Use gradient accumulation if needed
python main.py finetune small-lora-test --override per_device_train_batch_size=16 --override gradient_accumulation_steps=2
```

#### Memory Monitoring

Monitor memory pressure during training:

```bash
# Terminal 1: Start training
python main.py finetune medium-lora-data3

# Terminal 2: Monitor memory
watch -n 1 'memory_pressure && echo "---" && top -l 1 -s 0 | grep "python"'
```

**Warning Signs**:
- Memory pressure: yellow/red
- Swap usage increasing
- Training suddenly slows down
- System becomes unresponsive

### Performance Optimization Techniques

#### Gradient Accumulation for Effective Large Batches
```ini
[profile:large-effective-batch]
model = whisper-medium-lora
per_device_train_batch_size = 8    # Physical batch size
gradient_accumulation_steps = 4    # Effective batch size = 8 × 4 = 32
```

#### Mixed Precision Training (when supported)
```ini
[profile:mixed-precision]
model = whisper-small-lora
fp16 = False  # Disabled for MPS compatibility
# LoRA training is already memory-efficient
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: "Cannot determine model type"
```
ValueError: Cannot determine model type from model name: whisper-small and configuration.
```

**Solution**: Ensure LoRA parameters are present in your profile:
```ini
[profile:my-lora-profile]
model = whisper-small-lora
lora_r = 32              # This parameter triggers LoRA detection
lora_alpha = 64
# ... other LoRA parameters
```

#### Issue: "ImportError: cannot import name 'prepare_model_for_int8_training'"
**Solution**: This is handled automatically by the framework. If you see warnings, install bitsandbytes:
```bash
pip install bitsandbytes
```

#### Issue: MPS Training Fails with "NotImplementedError"
**Solution**: Enable CPU fallback temporarily:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py finetune your-lora-profile
```

#### Issue: Training Extremely Slow on Apple Silicon
**Causes & Solutions**:
1. **Using Rosetta Python**: Verify ARM64 Python installation
2. **Memory swapping**: Reduce batch size
3. **Fallback mode enabled**: Disable `PYTORCH_ENABLE_MPS_FALLBACK` after initial testing

#### Issue: Poor Performance Despite Training Success
**Diagnostic Steps**:
1. **Check rank/alpha ratio**: Ensure `alpha >= rank`
2. **Verify target modules**: Include both attention and feedforward layers
3. **Increase rank**: Try higher rank (64, 128) for complex adaptations
4. **Check data quality**: Validate your training dataset

### Advanced Debugging

#### Monitor Gradient Flow
```python
# Check that LoRA parameters are receiving gradients
for name, param in model.named_parameters():
    if 'lora' in name and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

#### Verify Adapter Loading
```python
# Confirm adapters are loaded correctly
print(model.peft_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")
```

## Advanced Usage Scenarios

### Scenario 1: Multi-Domain Voice Assistant

**Goal**: Create specialized adapters for different conversation contexts.

**Strategy**:
1. **Base Training**: Start with general conversational data
2. **Specialized Adapters**: Train domain-specific adapters
3. **Runtime Switching**: Switch adapters based on conversation context

```ini
[profile:assistant-base]
model = whisper-small-lora
dataset = conversational-general
lora_r = 32
lora_alpha = 64

[profile:assistant-technical]
model = whisper-small-lora
dataset = technical-support
lora_r = 48  # Higher rank for technical vocabulary
lora_alpha = 96

[profile:assistant-casual]
model = whisper-small-lora
dataset = casual-conversation
lora_r = 24  # Lower rank for simpler adaptation
lora_alpha = 48
```

### Scenario 2: Incremental Learning System

**Goal**: Continuously adapt to new speakers/domains without forgetting previous adaptations.

**Strategy**:
1. **Sequential Adapter Training**: Train new adapters for new domains
2. **Ensemble Inference**: Combine multiple adapters
3. **Adapter Merging**: Merge successful adapters into consolidated versions

```python
# Pseudo-code for ensemble inference
def ensemble_transcribe(audio, adapters):
    predictions = []
    for adapter_path in adapters:
        model = load_adapter(base_model, adapter_path)
        pred = model.transcribe(audio)
        predictions.append(pred)
    
    return consensus_prediction(predictions)
```

### Scenario 3: Production Deployment Pipeline

**Goal**: Efficient deployment with multiple specialized models.

**Architecture**:
```
Base Model (whisper-small, 244M params)
├── Medical Adapter (0.4M params) → Healthcare transcription
├── Legal Adapter (0.4M params) → Legal dictation
├── Meeting Adapter (0.4M params) → Business meetings
└── Phone Adapter (0.4M params) → Phone call transcription
```

**Benefits**:
- **Single base model**: One model handles all domains
- **Minimal storage**: 5× domain coverage with <3× storage
- **Hot swapping**: Change domains without model reload
- **A/B testing**: Easy comparison between adapter versions

## Technical Implementation Details

### Framework Integration

The LoRA implementation integrates seamlessly with the existing framework:

#### Detection Logic
```python
# In scripts/finetune.py
lora_params = ['lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules']
has_lora_config = any(param in profile_config for param in lora_params)

if has_lora_config:
    model_type = "whisper-lora"  # Route to LoRA module
```

#### Checkpoint Format
LoRA checkpoints contain only adapter weights:
```
checkpoint-1000/
├── adapter_model/
│   ├── adapter_config.json    # LoRA configuration
│   ├── adapter_model.bin      # Adapter weights (10-50MB)
│   └── README.md             # Loading instructions
├── optimizer.pt              # Optimizer state
├── scheduler.pt              # Learning rate scheduler
└── trainer_state.json       # Training metrics
```

#### Loading Mechanism
```python
# Automatic loading via PeftModel
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Manual loading for debugging
import torch
adapter_weights = torch.load(f"{checkpoint_path}/adapter_model.bin")
```

### Performance Characteristics

#### Training Speed Comparison (whisper-small on M1 Max)

| Method | Training Time | Memory Usage | Checkpoint Size |
|--------|---------------|--------------|-----------------|
| **Standard** | 100% (baseline) | 20GB | 1GB |
| **LoRA r=32** | 45% | 8GB | 25MB |
| **LoRA r=16** | 35% | 6GB | 15MB |
| **LoRA r=64** | 60% | 12GB | 45MB |

#### Convergence Characteristics

LoRA typically converges faster than standard fine-tuning:
- **Epochs required**: 2-3 (vs 5-8 for standard)
- **Learning rate**: Can use higher rates (1e-4 vs 5e-6)
- **Stability**: More stable training, less prone to catastrophic forgetting

## Best Practices Summary

### Getting Started
1. **Start with defaults**: Use `lora_r=32, lora_alpha=64`
2. **Use small model first**: Test with `whisper-small-lora`
3. **Enable fallback initially**: Set `PYTORCH_ENABLE_MPS_FALLBACK=1`
4. **Monitor memory**: Watch for swapping on Apple Silicon

### Scaling Up
1. **Increase rank for complex tasks**: Try r=64 for technical domains
2. **Adjust alpha proportionally**: Maintain 2:1 alpha:rank ratio
3. **Target specific modules**: Focus on encoder or decoder as needed
4. **Use 8-bit for large models**: Enable for memory-constrained scenarios

### Production Deployment
1. **Modular design**: Train domain-specific adapters
2. **Version control adapters**: Track adapter performance separately
3. **Monitor drift**: Retrain adapters as domains evolve
4. **A/B test adapters**: Compare adapter versions in production

### Troubleshooting Workflow
1. **Verify imports**: Ensure `peft` library is installed
2. **Check detection**: Confirm LoRA parameters are in config
3. **Test small**: Start with minimal rank and batch size
4. **Scale gradually**: Increase complexity after basic training works
5. **Monitor closely**: Watch memory, gradients, and convergence

---

LoRA represents the democratization of large model fine-tuning. By understanding its principles and following these best practices, you can achieve excellent results with minimal computational resources while maintaining the flexibility to adapt to new domains quickly and efficiently.