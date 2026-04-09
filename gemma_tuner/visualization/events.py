from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


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


def _get_attentions_tuple(outputs: Optional[Any]):
    if not outputs:
        return None
    if hasattr(outputs, "attentions"):
        return outputs.attentions
    if isinstance(outputs, dict) and "attentions" in outputs:
        return outputs["attentions"]
    return None


def _extract_attention(outputs: Optional[Any]) -> Optional[list[list[float]]]:
    try:
        att = _get_attentions_tuple(outputs)
        if not att:
            return None
        last_attention = att[-1]
        if last_attention is None:
            return None
        # Expect (batch, heads, q, k); some stacks return 3D — mean only when 4D.
        t = last_attention
        if t.dim() == 4:
            t = t.mean(dim=1)
        elif t.dim() != 3:
            return None
        avg_attention = t.detach().cpu().numpy()
        return avg_attention[0, :20, :20].tolist()
    except Exception as e:
        logger.debug("Viz attention extract skipped: %s", e)
        return None


def _extract_logits_last(outputs: Optional[Any]):
    if not outputs:
        return None
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get("logits")
    if logits is None:
        return None
    return logits[:, -1, :]


def _extract_token_probs(outputs: Optional[Any]) -> Optional[dict[str, list[float] | list[int]]]:
    try:
        logits = _extract_logits_last(outputs)
        if logits is None or logits.numel() == 0:
            return None
        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs[0], k=min(5, int(probs[0].shape[-1])))
        return {
            "values": top5.values.detach().cpu().numpy().tolist(),
            "indices": top5.indices.detach().cpu().numpy().tolist(),
        }
    except Exception as e:
        logger.debug("Viz token_probs extract skipped: %s", e)
        return None


def _extract_audio_features(batch: Optional[dict[str, Any]]) -> Optional[list[list[float]]]:
    if not batch:
        return None
    try:
        # Gemma 3n batches use "audio_values"; some HF ASR pipelines use "input_features".
        raw = batch.get("input_features")
        if raw is None:
            raw = batch.get("audio_values")
        if raw is not None:
            mel = raw[0].detach().cpu().numpy()
            return mel[::10, ::10].tolist()
        # Image batches: reuse the "listening" panel as a coarse grayscale preview.
        pv = batch.get("pixel_values")
        if pv is not None:
            x = pv[0].detach().cpu().float()
            if x.dim() == 3:
                x = x.mean(dim=0)
            return x[::8, ::8].detach().cpu().numpy().tolist()
        return None
    except Exception as e:
        logger.debug("Viz mel/image preview extract skipped: %s", e)
        return None


def _extract_optimizer_stats(optimizer: Optional[torch.optim.Optimizer]) -> Optional[dict[str, float]]:
    if optimizer is None or not optimizer.param_groups:
        return None
    group = optimizer.param_groups[0]
    stats = {"lr": float(group.get("lr", 0.0))}
    if "weight_decay" in group:
        stats["weight_decay"] = float(group["weight_decay"])
    return stats
