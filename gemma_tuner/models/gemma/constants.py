"""
Gemma training, validation, and audio-processing constants.

Extracted from gemma/finetune.py so that models/common/collators.py can import
them without creating a circular dependency (finetune -> collators -> finetune).

Imported by:
- models/gemma/finetune.py  (primary consumer)
- models/common/collators.py  (DataCollatorGemmaAudio)
- utils/gemma_dataset_prep.py  (DEFAULT_MODEL_ID)
"""

from gemma_tuner.constants import TrainingDefaults


class GemmaTrainingConstants:
    """Named constants for Gemma training configuration and MPS optimization."""

    DEFAULT_LEARNING_RATE = 2e-4
    DEFAULT_NUM_TRAIN_EPOCHS = 1
    DEFAULT_GRADIENT_ACCUMULATION = 8
    DEFAULT_LOGGING_STEPS = 10
    DEFAULT_SAVE_STRATEGY = "epoch"
    DEFAULT_EVAL_STRATEGY = "epoch"

    # LoRA defaults (safe on consumer hardware)
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    # Attention + text/LLM MLP projections (gate/up/down), matching
    # README/guides/apple-silicon/gemma4-guide.md. Do not use fc1/fc2 — those names
    # are not used in Gemma 3n / Gemma 4 transformers modules; _discover_candidate_target_modules
    # would silently drop them.
    # Audio encoder layers use different suffixes (e.g. Gemma 3n: ffw_layer_1, q_proj under
    # audio blocks). To LoRA audio-tower weights, set lora_target_modules explicitly after
    # inspecting model.named_modules() for your checkpoint.
    # Existing configs that set lora_target_modules explicitly are unaffected; only the
    # implicit fallback in finetune.py uses this constant.
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Default when profile omits base_model (SFT / chat: prefer -it per gemma4-guide.md).
    DEFAULT_BASE_MODEL_ID = "google/gemma-4-E2B-it"

    # Special token handling
    IGNORE_TOKEN_ID = TrainingDefaults.IGNORE_TOKEN_ID


class GemmaValidationConstants:
    """Named constants for Gemma validation and error detection."""

    # Test tensor dimensions for comprehensive bfloat16 validation
    BFLOAT16_TEST_TENSOR_SIZE = 10

    # Shows first 5 problematic sample indices to avoid log spam
    MAX_DISPLAYED_ERROR_SAMPLES = 5


class AudioProcessingConstants:
    """Named constants for audio processing in Gemma multimodal training."""

    # Default sampling rate for Gemma audio tower (USM-based, 16kHz)
    DEFAULT_SAMPLING_RATE = 16000

    # Returns 1.0 second of silence when audio loading fails
    FALLBACK_SILENCE_DURATION_SECONDS = 1.0
