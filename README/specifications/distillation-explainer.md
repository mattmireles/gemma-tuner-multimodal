# Understanding Whisper Distillation: A Simple Guide

This guide explains how knowledge distillation works in the Whisper Fine-Tuner system, focusing on what it does, why it's powerful, and how you can customize it for your needs.

## What is Knowledge Distillation?

Think of knowledge distillation like a master teacher passing their expertise to a student. In our case:
- The **teacher** is a large, powerful Whisper model (like `whisper-large-v2`)
- The **student** is a smaller model (like `whisper-small`) 
- The goal is to make the student almost as good as the teacher, but much faster

### A Simple Analogy

Imagine you're teaching someone to recognize different bird songs:
- A **professor** (large model) has studied birds for decades and knows every subtle detail
- A **student** (small model) is just learning
- Instead of making the student memorize everything, the professor shares their *intuition* about what matters most
- The student learns not just the right answers, but *how* the professor thinks about bird songs

This is exactly what happens in knowledge distillation - the student learns both from the correct answers AND from the teacher's "thought process" (probability distributions).

## How Does Distillation Work in This System?

### The Two-Model Dance

During distillation training, we run both models simultaneously:

1. **Teacher Model** (frozen, no training):
   - Processes the audio and produces predictions
   - Shares its confidence levels for each possible word
   - Acts like a mentor providing guidance

2. **Student Model** (actively learning):
   - Also processes the same audio
   - Tries to match BOTH:
     - The correct transcription (ground truth)
     - The teacher's confidence patterns

### The Magic: Combined Learning

The student learns from two sources simultaneously:

```
Total Loss = α × (How well student matches teacher) + (1-α) × (How well student matches correct answer)
```

- **KL Divergence Loss**: Measures how different the student's thinking is from the teacher's
- **Cross-Entropy Loss**: Measures how wrong the student is compared to the correct answer
- **α (kl_weight)**: Balances between copying the teacher vs getting the right answer (default: 0.5)

### Temperature: Making Knowledge Transfer Easier

Temperature is like adjusting the contrast on a photo:
- **Low temperature (1.0)**: Sharp, high-contrast - only the most likely words stand out
- **High temperature (2.0-4.0)**: Softer, more nuanced - subtle differences become visible

Higher temperature helps the student learn the teacher's nuanced understanding, not just the obvious answers.

## The Key Insight: Asymmetric Architecture

Here's the breakthrough idea that makes distillation so powerful for Whisper:

### Understanding Whisper's Two Parts

Whisper has two main components:

1. **Encoder** (The Listener):
   - Converts audio into understanding
   - Does the hard work of filtering noise, understanding accents
   - Processes all audio in parallel (fast)
   - **More layers = better audio understanding**

2. **Decoder** (The Writer):
   - Converts understanding into text
   - Generates words one at a time (slow)
   - Like autocomplete on your phone
   - **More layers = better grammar, but MUCH slower**

### The Bottleneck Discovery

The decoder is the speed bottleneck because it must generate words sequentially. If we can make the decoder smaller while keeping a powerful encoder, we get:
- ✅ Excellent audio understanding (from large encoder)
- ✅ Much faster text generation (from small decoder)
- ✅ Minimal accuracy loss (thanks to distillation)

## How to Customize Encoder/Decoder Layers

### Method 1: Use Pre-Defined Models

The simplest approach uses existing Whisper models with different layer counts:

```bash
# Large teacher (32 encoder + 32 decoder layers) teaches Small student (12 + 12 layers)
whisper-tuner finetune distil-small-from-large

# Large teacher teaches Medium student (24 + 24 layers)
whisper-tuner finetune distil-medium-from-large
```

### Method 2: Create Custom Architectures

For ultimate control, create a student with a custom number of decoder layers:

```bash
python -m whisper_tuner.models.distil_whisper.finetune \
  --model_name_or_path openai/whisper-small \
  --teacher_model_name_or_path openai/whisper-large-v2 \
  --student_decoder_layers 2 \
  --dataset_name your-dataset
```

This creates a model with:
- Encoder: 12 layers (from whisper-small)
- Decoder: 2 layers (custom, ultra-lightweight)

### Available Customization Options

- `--student_decoder_layers`: Number of decoder layers (e.g., 2, 4, 6)
- `--student_decoder_attention_heads`: Number of attention heads (optional)
- `--student_decoder_ffn_dim`: Feed-forward network size (optional)

## Practical Examples

### Example 1: Call Center Optimization
**Scenario**: Noisy phone audio, simple vocabulary

```bash
# Use large encoder (excellent noise handling) + tiny decoder (simple language)
python -m whisper_tuner.models.distil_whisper.finetune \
  --model_name_or_path openai/whisper-small \
  --teacher_model_name_or_path openai/whisper-large-v2 \
  --student_decoder_layers 2 \
  --dataset_name call-center-data
```

**Result**: 6-10x faster inference with minimal accuracy loss

### Example 2: Lecture Transcription
**Scenario**: Clear audio, complex academic vocabulary

```bash
# Keep balanced architecture - complex language needs more decoder layers
whisper-tuner finetune distil-medium-from-large
```

**Result**: 2-3x faster with good handling of technical terms

### Example 3: Mobile App Deployment
**Scenario**: Need smallest possible model for on-device use

```bash
# Extreme compression with 2-layer decoder
python -m whisper_tuner.models.distil_whisper.finetune \
  --model_name_or_path openai/whisper-base \
  --teacher_model_name_or_path openai/whisper-large-v2 \
  --student_decoder_layers 2 \
  --dataset_name mobile-app-data \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-5
```

**Result**: ~50MB model suitable for mobile deployment

## Memory Requirements and Performance

### Memory Usage During Training

Distillation requires both models in memory:

| Setup | Memory Required | Recommended Hardware |
|-------|----------------|---------------------|
| Large → Small | ~24GB | M1 Max or better |
| Large → Medium | ~32GB | M1 Max with 64GB |
| Medium → Small | ~16GB | M1 Pro |

### Batch Size Recommendations

Start conservative and increase if memory allows:

```bash
# For 32GB M1 Max
--per_device_train_batch_size 4  # Large → Small
--per_device_train_batch_size 2  # Large → Medium

# Use gradient accumulation for larger effective batch size
--gradient_accumulation_steps 4
```

### Expected Performance Gains

| Teacher → Student | Size Reduction | Speed Improvement | Typical WER Increase |
|-------------------|---------------|-------------------|---------------------|
| Large → Medium | 2x | 2-3x | +1-2% |
| Large → Small | 6x | 4-6x | +2-4% |
| Large → Custom-2L | 10x+ | 8-12x | +3-6% |

## Tips for Success

1. **Start Simple**: Begin with pre-defined model pairs before custom architectures
2. **Monitor Memory**: Use Activity Monitor to watch for swap usage
3. **Temperature Tuning**: Start with 2.0, increase to 3.0-4.0 if student struggles
4. **Learning Rate**: Use lower rates (1e-5) for stability
5. **Validation**: Always evaluate on held-out data to catch overfitting

## Common Issues and Solutions

### "Out of Memory"
- Reduce batch size to 2 or even 1
- Use gradient accumulation to maintain effective batch size
- Consider using a smaller teacher model

### "Student Performance is Poor"
- Increase temperature (try 3.0 or 4.0)
- Increase kl_weight to 0.7-0.8 (more teacher influence)
- Train for more epochs (5-10 instead of 3)
- Check that teacher and student use same tokenizer

### "Training is Unstable"
- Lower learning rate to 5e-6 or 1e-6
- Increase warmup steps to 1000+
- Ensure teacher is in eval mode (should be automatic)

## Summary

Knowledge distillation in this system allows you to:
1. Create smaller, faster models that retain most of the accuracy
2. Customize the architecture for your specific use case
3. Deploy Whisper in resource-constrained environments

The key insight is that you can mix and match encoder/decoder sizes - keeping a powerful encoder for audio understanding while shrinking the decoder for speed. This asymmetric approach, combined with knowledge distillation, enables 5-10x speedups with minimal accuracy loss.

Remember: The encoder understands audio, the decoder generates text. For most applications, a strong encoder + small decoder is the winning combination.