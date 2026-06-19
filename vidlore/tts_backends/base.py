"""TTS backend abstraction for Vidlore's premium local voiceover.

Runs in the Python 3.9 app. Concrete backends either run in-process (legacy
edge-tts) or shell out to the Python 3.11 TTS sidecar (kokoro / chatterbox) via
``sidecar.run_worker``. The pipeline only ever talks to this interface, so the
chosen premium model is swappable without touching pipeline/assemble.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SynthItem:
    """One unit of narration to synthesize (a scene, or a sub-chunk)."""
    text: str
    out_wav: Path


@dataclass
class SynthResult:
    out_wav: Path
    duration: float
    sample_rate: int


@dataclass
class VoicePreset:
    """A documentary narrator preset, mapped to per-engine voice config.

    engine_cfg maps an engine name -> its kwargs, e.g.::
        {"chatterbox": {"exaggeration": 0.3, "cfg_weight": 0.5, "ref": "deep_male.wav"},
         "kokoro":     {"voice": "am_onyx", "speed": 0.9},
         "edge":       {"voice": "en-US-GuyNeural", "rate": "+0%"}}
    `ref` (chatterbox) is a filename under the shipped voice-refs dir, or None to
    use the model's built-in voice. NO celebrity / unauthorized clones — only
    built-in, open-licensed, or the user's own reference clips.
    """
    key: str
    label: str
    description: str
    engine_cfg: dict = field(default_factory=dict)

    def cfg_for(self, engine: str) -> dict:
        return dict(self.engine_cfg.get(engine, {}))


class TTSBackend(ABC):
    """Common interface for every voice backend."""

    name: str = "base"
    #: human label shown in the dashboard
    label: str = "Base"
    #: which sidecar engine this backend drives ("" => in-process)
    engine: str = ""

    @abstractmethod
    def is_available(self) -> str:
        """Return "" if ready, else a human reason it is not (missing venv,
        model files, deps). Never raises."""
        raise NotImplementedError

    @abstractmethod
    def synth_batch(self, items, preset, *, device: str = "auto",
                    settings=None):
        """Synthesize each :class:`SynthItem` to its out_wav. Returns a list of
        :class:`SynthResult` (same order). Raises on hard failure so the caller
        can fall back to another backend."""
        raise NotImplementedError

    def synth_one(self, text: str, out_wav, preset, *, device="auto",
                  settings=None):
        res = self.synth_batch([SynthItem(text, Path(out_wav))], preset,
                               device=device, settings=settings)
        return res[0]

    def info(self) -> dict:
        return {"name": self.name, "label": self.label, "engine": self.engine,
                "available": self.is_available() == ""}
