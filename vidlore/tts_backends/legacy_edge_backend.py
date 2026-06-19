"""Legacy basic voice (edge-tts). Runs IN-PROCESS on the 3.9 app (no sidecar).
Demoted to fallback-only; kept so a run never dies if the local stack is absent."""
from __future__ import annotations

import asyncio
import contextlib
import wave
from pathlib import Path

from .base import SynthResult, TTSBackend


def _wav_dur(p) -> float:
    try:
        with contextlib.closing(wave.open(str(p), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:                                          # noqa: BLE001
        return 0.0


class LegacyEdgeBackend(TTSBackend):
    name = "edge"
    label = "Legacy Basic (edge-tts)"
    engine = ""  # in-process

    def is_available(self) -> str:
        try:
            import edge_tts  # noqa: F401
            return ""
        except Exception as e:                                 # noqa: BLE001
            return f"edge-tts not installed: {e}"

    def synth_batch(self, items, preset, *, device="auto", settings=None):
        import edge_tts
        from ..ffmpeg_tool import run
        cfg = preset.cfg_for("edge")
        voice = cfg.get("voice", "en-US-GuyNeural")
        rate = cfg.get("rate", "+0%")
        out = []
        for it in items:
            mp3 = Path(it.out_wav).with_suffix(".mp3")

            async def _go(_mp3=mp3, _text=it.text):
                comm = edge_tts.Communicate(_text, voice, rate=rate)
                with open(_mp3, "wb") as fh:
                    async for ch in comm.stream():
                        if ch["type"] == "audio":
                            fh.write(ch["data"])

            asyncio.run(_go())
            run(["-i", str(mp3), "-ar", "44100", "-ac", "2", str(it.out_wav)])
            out.append(SynthResult(Path(it.out_wav), _wav_dur(it.out_wav), 44100))
        return out
