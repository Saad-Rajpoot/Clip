"""Vidlore TTS backend registry.

The app asks for a backend by name; each knows how to synthesize (in-process for
legacy edge-tts, or via the 3.11 sidecar for kokoro/chatterbox). Importing this
package is safe on Python 3.9 — heavy ML imports happen only inside the sidecar
worker, never here.
"""
from __future__ import annotations

from .base import SynthItem, SynthResult, TTSBackend, VoicePreset
from .chatterbox_backend import ChatterboxBackend
from .kokoro_backend import KokoroBackend
from .legacy_edge_backend import LegacyEdgeBackend

# name -> class. Order = preference for the fallback chain.
_REGISTRY = {
    "chatterbox": ChatterboxBackend,
    "kokoro": KokoroBackend,
    "edge": LegacyEdgeBackend,
}

#: which backends count as "premium local" vs "fallback" vs "legacy"
PREMIUM = "chatterbox"
FALLBACK_FAST = "kokoro"
LEGACY = "edge"


def get_backend(name: str):
    cls = _REGISTRY.get((name or "").strip().lower())
    return cls() if cls else None


def available_backends() -> dict:
    """name -> "" (ready) | reason. Never raises."""
    out = {}
    for n, cls in _REGISTRY.items():
        try:
            out[n] = cls().is_available()
        except Exception as e:                                 # noqa: BLE001
            out[n] = f"error: {e}"
    return out


def resolve_chain(primary: str, *, allow_legacy: bool = True) -> list:
    """Backend fallback order starting at `primary`:
    premium -> fast fallback -> (legacy only if allowed). De-duped, existing."""
    order = [primary, FALLBACK_FAST]
    if allow_legacy:
        order.append(LEGACY)
    seen, chain = set(), []
    for n in order:
        if n and n not in seen and n in _REGISTRY:
            seen.add(n)
            chain.append(n)
    return chain


__all__ = ["SynthItem", "SynthResult", "TTSBackend", "VoicePreset",
           "get_backend", "available_backends", "resolve_chain",
           "PREMIUM", "FALLBACK_FAST", "LEGACY"]
