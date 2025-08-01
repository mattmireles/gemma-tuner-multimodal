# Advanced Distillation Guide

This guide provides a deeper technical overview of the knowledge distillation process used in this framework. It is intended for advanced users who want to move beyond standard teacher-student pairs and create highly optimized, custom-architected models.

## The Asymmetric Architecture Strategy

The core principle behind effective Whisper distillation is the intentional creation of an **asymmetric model architecture**. This strategy is based on a fundamental understanding of the separate roles and computational costs of the encoder and decoder.

### The Encoder: The Expert Listener
The encoder's role is to transform a raw audio spectrogram into a high-level, abstract representation of its content. It performs the acoustically difficult task of filtering noise, handling accents, and understanding the core meaning of the speech.

-   **Function**: Audio -> Abstract Meaning
-   **Computational Cost**: High, but highly parallelizable across the entire audio sequence.
-   **Impact of Depth**: More layers improve robustness to noise, accents, and challenging recording conditions. For high-fidelity transcription, a powerful encoder is non-negotiable.

### The Decoder: The Eloquent Speaker
The decoder's role is to translate the encoder's abstract representation into a sequence of text tokens. It is fundamentally a conditional language model.

-   **Function**: Abstract Meaning -> Text Tokens
-   **Computational Cost**: Very high, and bottlenecked by its auto-regressive (sequential, token-by-token) nature.
-   **Impact of Depth**: More layers improve grammatical correctness, vocabulary nuance, and long-range textual cohesion.

### The Strategic Imbalance
For most transcription tasks, the acoustic understanding (encoder) is significantly harder and more crucial than the text generation (decoder). Because the decoder is also the primary performance bottleneck, the optimal strategy is to **pair a large, powerful encoder with a small, fast decoder.** This preserves the model's critical listening ability while dramatically accelerating inference speed.

## Architectural Scenarios & Practical Implementation

This understanding leads to clear choices when designing a model architecture for a specific task.

| Scenario | Recommended Architecture | Why? |
| :--- | :--- | :--- |
| **Noisy Audio, Simple Language** <br/> (e.g., Call centers, factory floors) | **Many Encoder Layers, Few Decoder Layers** <br/> (e.g., `large-v2` encoder, `small` or custom 2-layer decoder) | The main challenge is understanding the audio, so a powerful encoder is crucial. The language is simple, so a small, fast decoder is sufficient. This is the sweet spot for distillation. |
| **Clean Audio, Complex Language** <br/> (e.g., University lectures, legal dictation) | **Balanced Encoder/Decoder Layers** <br/> (e.g., a standard `whisper-medium` or `large-v2` model) | The audio is clean, but the vocabulary and grammar are complex. Shrinking the decoder here could harm the quality of the transcription by sacrificing linguistic nuance. |
| **Resource-Constrained Deployment** <br/> (e.g., On-device mobile app) | **Many Encoder Layers, Minimal Decoder Layers** <br/> (e.g., `large-v2` encoder, custom 2-layer decoder) | The goal is maximum speed and minimum memory. The decoder is the bottleneck, so it should be as small as possible. You rely on distillation to teach this tiny decoder to be "good enough". |

## Advanced Application Strategies

The general scenarios above provide a strong baseline. However, for specific product applications, more sophisticated, multi-stage strategies can yield superior results.

### Strategy 1: Personalized Dictation App (Single *Known* Speaker)

In this scenario, the model only ever needs to transcribe one specific person's voice. This allows for a powerful optimization: **speaker adaptation**.

-   **The Goal:** Create the most accurate and performant model for one individual.
-   **The Strategy:** Fine-tuning first, distillation second.
    1.  **Fine-Tune for Expertise:** Start with a balanced model (e.g., `whisper-medium`). Fine-tune it exclusively on recordings of the target speaker. This creates a smaller, highly specialized "expert" model that can outperform a generic `large` model for that specific voice.
    2.  **Distill for Speed (Optional):** If maximum speed is still required after fine-tuning, you can now distill your new "expert" model. Use your fine-tuned model as the **teacher** and distill it into a student with an even smaller decoder.
-   **Key Takeaway:** For a personalized app, **fine-tuning is the primary optimization tool.** Distillation is a secondary step to further enhance performance.

### Strategy 2: Generalist Dictation App (Any *Single* Speaker)

In this scenario, the model must be a generalist, capable of transcribing for any user, but only one speaker at a time.

-   **The Goal:** Create a robust, general-purpose model that is fast and accurate for dictation.
-   **The Strategy:** Find the "Goldilocks" decoder.
    1.  **Encoder: Maximize Generalization:** Use a powerful, generalist encoder like the one from `openai/whisper-large-v2` to handle the widest possible range of voices.
    2.  **Decoder: Balance Speed and Complexity:** A tiny 2-layer decoder may be too linguistically simple for complex dictation. The optimal choice is a **medium-sized decoder** that is fast but still powerful enough for complex grammar.
-   **Key Takeaway:** The best approach is to distill `openai/whisper-large-v2` (as the teacher) into a student model that uses the **`small` (12-layer) or `medium` (24-layer) decoder**. This provides the best balance of generalist accuracy and dictation-level performance.

## Practical Implementation: Custom Architectures

You can directly implement these custom architectures using the following arguments in `models/distil-whisper/finetune.py`:

-   `--student_decoder_layers`: Specifies the exact number of decoder layers.
-   `--student_decoder_attention_heads`: (Optional) Allows you to adjust the number of attention heads.
-   `--student_decoder_ffn_dim`: (Optional) Allows you to adjust the feed-forward network dimension.

#### Example: Creating an Ultra-Lightweight Model
To build the model for the "Resource-Constrained Deployment" scenario, you would run:

```bash
python models/distil-whisper/finetune.py \
  --model_name_or_path openai/whisper-small \
  --teacher_model_name_or_path openai/whisper-large-v2 \
  --student_decoder_layers 2 \
  --dataset_name your-dataset
  ...
```
This command creates a new student model with a `small` encoder and a custom 2-layer decoder, then begins training it from the `large-v2` teacher.

## Advanced Hyperparameter Tuning

Beyond architecture, two key hyperparameters control the distillation process itself:

-   **`temperature`**: This parameter (default: `2.0`) softens the probability distributions of the teacher's predictions.
    -   **Higher values (2.0-4.0)** create a "softer" target, which helps transfer more nuanced knowledge and can improve the student's ability to generalize.
    -   **Lower values (1.0-2.0)** create a "harder" target, more closely resembling the teacher's top predictions, which can lead to faster convergence.

-   **`kl_weight`**: This parameter (default: `0.5`) balances the loss between the ground-truth data (Cross-Entropy) and the teacher's predictions (KL-Divergence).
    -   **Higher values (0.7-0.9)** force the student to focus more on mimicking the teacher.
    -   **Lower values (0.1-0.3)** force the student to focus more on matching the ground-truth labels.

## Summary of Trade-offs

When designing your distillation process, you are balancing three factors:

1.  **Accuracy**: Primarily driven by the power of the **teacher's encoder** and the quality of your training data.
2.  **Inference Speed**: Primarily driven by the size of the **student's decoder**.
3.  **Model Size**: Driven by the combined size of the student's encoder and decoder.

By using a large encoder and a small, custom-built decoder, you can often achieve significant improvements in speed and size with a surprisingly small drop in accuracy.
