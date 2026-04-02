# Model Evaluation Product Specification

## Executive Summary

The Model Evaluation system is a critical component of the Whisper Fine-Tuner that provides a clear, predictable, and automated measure of model performance. Its primary purpose is to answer the question: "How well is my model learning?" It does this by testing the model against a held-out evaluation dataset at regular intervals during training, using the industry-standard Word Error Rate (WER) metric. This system is essential for tracking progress, preventing overfitting, and automatically saving the best-performing version of the model.

### Key Capabilities
- **Automated In-Training Evaluation**: Run evaluation automatically at the end of each epoch or at a specified step interval.
- **Best Model Selection**: Automatically identify and save the model checkpoint with the lowest Word Error Rate.
- **Consistent Data Handling**: Applies the same data patches and processing to the evaluation set as the training set, ensuring a fair comparison.
- **Standalone Evaluation**: A dedicated command allows for evaluating any trained model against any dataset after training is complete.
- **Clear Reporting**: Evaluation results are logged to the console and saved to `metrics.json` for every run.

## Product Overview

### Purpose
Evaluation is not an optional add-on; it is the compass that guides our training process. Without it, we are flying blind. The system's purpose is to:
1.  Provide a reliable, objective measure of the model's performance on unseen data.
2.  Prevent "overfitting," a state where the model memorizes the training data but fails to generalize.
3.  Automatically select the single best-performing model checkpoint, saving the user from having to guess.
4.  Give the user the necessary feedback to decide if a training run is succeeding or failing.

### Core Value Proposition
"Know exactly how well your model is performing and automatically save the best version. No guesswork."

## Technical Architecture

### In-Training Evaluation Workflow

The evaluation process is integrated directly into the HuggingFace `Seq2SeqTrainer`. It is not random, but triggered by a specific configuration parameter, `evaluation_strategy`.

```
Training Loop
├── Train on Batch 1...N
│
├── Is it time to evaluate? (End of Epoch or `eval_steps` reached)
│   │
│   └── YES → Pause Training
│       │
│       ├── Load Evaluation Dataset (e.g., validation.csv)
│       │   └── Apply same patches as training data
│       │
│       ├── Run Inference on ALL evaluation samples
│       │   └── Use `predict_with_generate=True` for accurate transcription
│       │
│       ├── Compute Metrics
│       │   └── Call `compute_metrics` function with predictions & labels
│       │   └── Calculate Word Error Rate (WER)
│       │
│       ├── Log Results
│       │   └── Print WER and loss to console
│       │   └── Save to metrics files
│       │
│       └── Compare to Best Model
│           └── If current WER is the best so far, save this checkpoint
│
└── Resume Training
```

### Core Components

#### 1. `evaluation_strategy` Parameter
This is the key that controls the evaluation schedule. It lives in `config.ini` and can be set to:
*   `"epoch"`: (Default) Evaluate at the end of every training epoch. This is the simplest and most common strategy.
*   `"steps"`: Evaluate every `eval_steps`. This provides more frequent feedback, which is useful for very large datasets where an epoch can take a long time.
*   `"no"`: Disable in-training evaluation. Not recommended.

#### 2. `compute_metrics` Function
Located in `models/whisper/finetune.py`, this function is the heart of the evaluation logic.
*   **Input**: The raw predictions and labels from the trainer.
*   **Process**:
    1.  Decodes the predicted token IDs into text.
    2.  Decodes the ground-truth label IDs into text, handling special tokens.
    3.  Uses the `evaluate.load("wer")` metric to compare the two sets of text.
*   **Output**: A dictionary containing the calculated WER, e.g., `{"wer": 0.15}`.

#### 3. `load_best_model_at_end`
When this parameter is set to `True` (which is our default), the `Seq2SeqTrainer` will keep track of the evaluation metric you've specified (`metric_for_best_model`, which defaults to `"wer"`). After training is complete, it will automatically load the weights from the checkpoint that achieved the best score on that metric. This ensures that the final model in the output directory is the best one, not just the last one.

### Standalone Evaluation (`whisper-tuner evaluate`) and Gemma Utility
A separate workflow for post-training analysis.
*   **Command**: `whisper-tuner evaluate <path_to_model_dir> --dataset <dataset_name>`
*   **Gemma ASR Utility**: For Gemma 3n audio LoRA runs, a dedicated helper is available:
  ```bash
  python tools/eval_gemma_asr.py \
    --csv data/datasets/<dataset>/validation.csv \
    --model google/gemma-3n-E2B-it \
    --adapters output/<your_run>/ \
    --text-column text \
    --limit 200
  ```
  This uses `jiwer` to compute WER/CER and the model’s `AutoProcessor` to prepare audio+text inputs.
*   **Process**:
    1.  Loads the specified trained model.
    2.  Loads the specified dataset's evaluation split.
    3.  Runs inference on the entire dataset.
    4.  Computes and reports the final WER.

## User Workflow

### Configuring In-Training Evaluation
The user controls evaluation through their profile in `config.ini`:
```ini
[profile:my-profile]
model = whisper-small
dataset = my-dataset
...
evaluation_strategy = epoch  # Or "steps"
eval_steps = 500             # Only used if strategy is "steps"
metric_for_best_model = wer  # Tells the trainer to watch the WER
load_best_model_at_end = true # Automatically save the best model
```

### Monitoring Evaluation
During a training run, the user will see clear output in the console when an evaluation is performed:
```
***** Running Evaluation *****
  Num examples = 500
  Batch size = 8
{'eval_loss': 0.2831, 'eval_wer': 0.1245, 'eval_runtime': 120.4, 'epoch': 1.0}
```

### Final Result
At the end of a successful training run, the user is notified which checkpoint was the best:
```
Loading best model from output/1-my-profile/checkpoint-1500 (score: 0.1245).
```

## Data Requirements for Evaluation Set

The quality of the evaluation is only as good as the quality of the evaluation data.
*   **Must be Held-Out**: The evaluation dataset (`validation.csv`) must contain data that is **not** in the training dataset.
*   **Must be Representative**: It should reflect the same kind of audio (domain, noise levels, accents) that the model will encounter in the real world.
*   **Must be High-Quality**: The transcriptions in the evaluation set must be as close to perfect as possible to provide a reliable score.
*   **Size**: A good evaluation set should have at least 1-2 hours of audio or several hundred samples to be statistically significant.

## Best Practices

1.  **Always Use an Evaluation Set**: Never train without one. It is the only way to know if the model is actually learning.
2.  **Use `"epoch"` Strategy for Most Cases**: It's the simplest and provides a consistent rhythm to the training process. Use `"steps"` only for very large datasets where epochs take many hours.
3.  **Trust the `load_best_model_at_end` Feature**: It is the most reliable way to get the best-performing model from a training run.
4.  **Curate Your Evaluation Set**: Spend time ensuring your `validation.csv` is a clean, representative, and challenging test for your model. This is one of the highest-leverage activities in the entire ML workflow.

This specification makes the evaluation process clear and intentional. It's not a random background task; it is a core, configurable feature that is central to building a high-quality model. 