from __future__ import annotations

from gemma_tuner.scripts.gemma_generate import render_generation_prompts as render_cli_generation_prompts
from tools.eval_gemma_asr import render_generation_prompts as render_eval_generation_prompts


class RecordingProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages_batch, **kwargs):
        self.calls.append((messages_batch, kwargs))
        return ["prompt"]


def test_eval_generation_prompts_request_assistant_turn():
    processor = RecordingProcessor()
    messages_batch = [[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]]

    prompts = render_eval_generation_prompts(processor, messages_batch)

    assert prompts == ["prompt"]
    assert processor.calls == [
        (
            messages_batch,
            {
                "tokenize": False,
                "add_generation_prompt": True,
            },
        )
    ]


def test_cli_generation_prompts_request_assistant_turn():
    processor = RecordingProcessor()
    messages_batch = [[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]]

    prompts = render_cli_generation_prompts(processor, messages_batch)

    assert prompts == ["prompt"]
    assert processor.calls == [
        (
            messages_batch,
            {
                "tokenize": False,
                "add_generation_prompt": True,
            },
        )
    ]
