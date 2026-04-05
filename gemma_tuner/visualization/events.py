from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class TrainingVizEvent:
    step: int
    epoch: float
    loss: float
    gradient_norm: float
    learning_rate: float
    memory_gb: float
    attention: Optional[list[list[float]]]
    token_probs: Optional[dict[str, list[float] | list[int]]]
    mel_spectrogram: Optional[list[list[float]]]
    steps_per_second: float
    total_time: float
    architecture: dict[str, Any]
    optimizer_stats: Optional[dict[str, float]]

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_training_event(
    *,
    step: int,
    epoch: float,
    loss: float,
    gradient_norm: float,
    learning_rate: float,
    memory_gb: float,
    batch: Optional[dict[str, Any]],
    outputs: Optional[Any],
    optimizer: Optional[torch.optim.Optimizer],
    steps_per_second: float,
    total_time: float,
    architecture: dict[str, Any],
) -> TrainingVizEvent:
    return TrainingVizEvent(
        step=step,
        epoch=epoch,
        loss=loss,
        gradient_norm=gradient_norm,
        learning_rate=learning_rate,
        memory_gb=memory_gb,
        attention=_extract_attention(outputs),
        token_probs=_extract_token_probs(outputs),
        mel_spectrogram=_extract_audio_features(batch),
        steps_per_second=steps_per_second,
        total_time=total_time,
        architecture=architecture,
        optimizer_stats=_extract_optimizer_stats(optimizer),
    )


def _extract_attention(outputs: Optional[Any]) -> Optional[list[list[float]]]:
    if not outputs or not hasattr(outputs, "attentions") or not outputs.attentions:
        return None
    last_attention = outputs.attentions[-1]
    if last_attention is None:
        return None
    avg_attention = last_attention.mean(dim=1).detach().cpu().numpy()
    return avg_attention[0, :20, :20].tolist()


def _extract_token_probs(outputs: Optional[Any]) -> Optional[dict[str, list[float] | list[int]]]:
    if not outputs or not hasattr(outputs, "logits"):
        return None
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top5 = torch.topk(probs[0], k=min(5, probs[0].shape[-1]))
    return {
        "values": top5.values.detach().cpu().numpy().tolist(),
        "indices": top5.indices.detach().cpu().numpy().tolist(),
    }


def _extract_audio_features(batch: Optional[dict[str, Any]]) -> Optional[list[list[float]]]:
    if not batch:
        return None
    # Gemma 3n batches use "audio_values"; older Whisper-style batches use "input_features".
    # Try both keys so this function works for all model families.
    raw = batch.get("input_features")
    if raw is None:
        raw = batch.get("audio_values")
    if raw is None:
        return None
    mel = raw[0].detach().cpu().numpy()
    return mel[::10, ::10].tolist()


def _extract_optimizer_stats(optimizer: Optional[torch.optim.Optimizer]) -> Optional[dict[str, float]]:
    if optimizer is None or not optimizer.param_groups:
        return None
    group = optimizer.param_groups[0]
    stats = {"lr": float(group.get("lr", 0.0))}
    if "weight_decay" in group:
        stats["weight_decay"] = float(group["weight_decay"])
    return stats
