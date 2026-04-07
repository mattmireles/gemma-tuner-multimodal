"""Shared test doubles (minimal processors, etc.) to avoid cross-test imports."""

from __future__ import annotations

import torch


class FakeImageProcessor:
    """Minimal Gemma-like multimodal processor for offline collator tests.

    ``__call__`` returns fixed tensor layouts; it does not exercise real chat templates.
    """

    def __init__(self):
        class Tok:
            pad_token_id = 0
            bos_token_id = 1
            unk_token_id = 3
            start_of_turn_token_id = 7

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                if text == "<start_of_turn>":
                    return [7]
                if text == "model\n":
                    return [20, 21]
                if text.strip() == "Paris":
                    return [200]
                return [99]

            def convert_tokens_to_ids(self, token: str) -> int:
                if token == "<start_of_turn>":
                    return 7
                return self.unk_token_id

        self.tokenizer = Tok()
        self.image_seq_length = 256
        self.boi_token = "<boi>"
        self.eoi_token = "<eoi>"
        self.image_token = "<img>"
        self.full_image_sequence = ""

    def apply_chat_template(self, messages_batch, tokenize=False, add_generation_prompt=False, **kwargs):
        del kwargs
        assert tokenize is False
        batch = len(messages_batch)
        return [f"prompt{i}" for i in range(batch)]

    def __call__(self, text=None, images=None, return_tensors=None, padding=None, **kwargs):
        del kwargs
        assert text is not None and images is not None
        batch = len(text)
        # [bos, …, sot, …, sot, model\\n ids, answer, pad] — answer stays in supervised region
        input_ids = torch.zeros((batch, 9), dtype=torch.long)
        attention_mask = torch.ones((batch, 9), dtype=torch.long)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids[:, 2] = 7
        input_ids[:, 4] = 7
        input_ids[:, 5:7] = torch.tensor([20, 21])
        input_ids[:, 7] = 200
        attention_mask[:, 8] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}
