"""Versioned Socket.IO payloads for the training dashboard."""

from __future__ import annotations

from typing import Any

# Bump when adding/removing top-level keys the UI depends on.
VIZ_SCHEMA_VERSION = 1


def finalize_training_payload(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Attach schema version and best-effort panel hints. Safe to call on full
    ``build_training_event().as_payload()`` dicts.
    """
    out = dict(flat)
    out["viz_schema_version"] = VIZ_SCHEMA_VERSION
    out["panels_status"] = {
        "attention": "ok" if flat.get("attention") else "empty",
        "mel_spectrogram": "ok" if flat.get("mel_spectrogram") else "empty",
        "token_probs": "ok" if flat.get("token_probs") else "empty",
    }
    return out


def finalize_control_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Smaller events (epoch_change, validation, …): version only."""
    out = dict(data)
    out.setdefault("viz_schema_version", VIZ_SCHEMA_VERSION)
    return out


def finalize_initial_state_payload(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out["viz_schema_version"] = VIZ_SCHEMA_VERSION
    return out
