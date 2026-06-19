"""Step 2: narration.

Provider ladder (Vidlore 'hybrid' parity — free by default, upgrade on key):
  * ELEVENLABS_API_KEY set -> ElevenLabs (Vidlore's actual voice engine)
  * otherwise              -> edge-tts (free, no key, the default)
A per-scene ElevenLabs failure degrades to edge-tts so a run never dies.

Produces one audio file per scene plus word-level timings used for
caption sync and scene durations.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shutil
import wave
from dataclasses import dataclass, field
from pathlib import Path

import edge_tts
import requests

from .cache import scene_key
from .ffmpeg_tool import run
from .script_gen import Script

_EL_URL = "https://api.elevenlabs.io/v1/text-to-speech/"


@dataclass
class WordTiming:
    word: str
    start: float           # seconds, absolute on the final timeline
    end: float


@dataclass
class NarratedScene:
    index: int
    audio: Path            # mp3 for this scene
    duration: float        # seconds (measured from decoded wav)
    words: list[WordTiming] = field(default_factory=list)


@dataclass
class Narration:
    scenes: list[NarratedScene]
    audio: Path            # concatenated narration for the whole video
    reused: int = 0        # scenes restored from cache (skipped TTS)

    @property
    def total(self) -> float:
        return sum(s.duration for s in self.scenes)

    def all_words(self) -> list[WordTiming]:
        out: list[WordTiming] = []
        for s in self.scenes:
            out.extend(s.words)
        return out


async def _synth(text: str, voice: str, rate: str, out_mp3: Path) -> None:
    comm = edge_tts.Communicate(text, voice, rate=rate)
    with open(out_mp3, "wb") as fh:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                fh.write(chunk["data"])


def _synth_elevenlabs(
    text: str, voice_id: str, model: str, api_key: str, out_mp3: Path
) -> None:
    """Vidlore's real voice engine. Raises on any failure so the caller
    can fall back to edge-tts for that scene."""
    r = requests.post(
        _EL_URL + voice_id,
        params={"output_format": "mp3_44100_128"},
        headers={
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        },
        json={
            "text": text,
            "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=120,
    )
    r.raise_for_status()
    if "audio" not in r.headers.get("content-type", "") or len(r.content) < 1200:
        raise RuntimeError("ElevenLabs returned non-audio response")
    out_mp3.write_bytes(r.content)


def _make_scene_mp3(
    *, provider: str, text: str, voice: str, rate: str,
    el_api_key: str, el_model: str, fallback_voice: str, mp3: Path,
) -> None:
    """Route a scene to its TTS provider; ElevenLabs failures degrade to
    edge-tts so the pipeline is resilient."""
    if provider == "elevenlabs":
        try:
            _synth_elevenlabs(text, voice, el_model, el_api_key, mp3)
            return
        except Exception as e:  # noqa: BLE001
            print(
                f"  [tts] ElevenLabs failed ({e}); edge-tts fallback",
                flush=True,
            )
            voice = fallback_voice
    asyncio.run(_synth(text, voice, rate, mp3))


def _spread_words(text: str, start: float, duration: float) -> list[WordTiming]:
    """edge-tts does not reliably emit per-word boundaries, so distribute
    the scene's words across its measured audio duration proportionally to
    word length. Scenes are short, so caption sync stays tight."""
    toks = text.split()
    if not toks or duration <= 0:
        return []
    weights = [len(t) + 1 for t in toks]
    total = float(sum(weights))
    out: list[WordTiming] = []
    t = start
    for tok, w in zip(toks, weights):
        share = duration * (w / total)
        out.append(WordTiming(tok, t, t + share))
        t += share
    return out


def _wav_duration(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def _slice_scene(
    master: Path, start: float, end: float, total: float, wav: Path
) -> None:
    """Cut [start, end] out of ``master`` into ``wav`` — defensively.

    Whisper alignment (or accumulated drift) can hand us a degenerate
    window where ``end <= start`` or ``start`` has reached the end of the
    master audio. ffmpeg aborts with "-to value smaller than -ss" on such
    a slice, killing the whole render. Here we clamp both ends inside the
    real audio and guarantee a positive-length cut; if there genuinely is
    no audio left for this scene, we emit a short silence so the pipeline
    never dies on one bad boundary."""
    total = max(0.0, float(total))
    start = min(max(0.0, float(start)), total)
    end = min(max(float(end), start), total)
    # Need a real, positive window strictly inside the master.
    if total > 0.05 and end - start >= 0.05 and start < total:
        run(["-i", str(master), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-ar", "44100", "-ac", "2", str(wav)])
        # ffmpeg can silently produce an empty file if the window fell at
        # the very tail; treat that like the degenerate case below.
        try:
            if wav.exists() and _wav_duration(wav) >= 0.02:
                return
        except Exception:                                      # noqa: BLE001
            pass
    # Degenerate boundary -> short silence keeps timings/caption sync sane.
    run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-t", "0.20", "-ar", "44100", "-ac", "2", str(wav)])


def narrate_from_file(
    script: Script, audio_path: str, workdir: Path
) -> "Narration":
    """Use the user's OWN uploaded voiceover instead of TTS. The single
    audio file is split per scene proportionally to each scene's word
    count (same proportional model used for edge-tts), so scene
    durations and kinetic captions stay in sync with the real voice."""
    workdir.mkdir(parents=True, exist_ok=True)
    master = workdir / "voiceover_master.wav"
    run(["-i", str(Path(audio_path).resolve()),
         "-ar", "44100", "-ac", "2", str(master)])
    total = _wav_duration(master)
    n = len(script.scenes)

    # --- Exact path: forced-align the real voice -------------------- #
    # Build the flat script-word list + where each scene starts in it,
    # then ask Whisper for the true spoken time of every word. Scene
    # boundaries become the actual end-time of each scene's last word
    # (not a word-count guess), so footage cuts AND on-screen text land
    # exactly on the voice.
    scene_tok: list[list[str]] = [s.narration.split() for s in script.scenes]
    flat: list[str] = [w for toks in scene_tok for w in toks]
    bounds: list[int] = []
    c = 0
    for toks in scene_tok:
        bounds.append(c)
        c += len(toks)
    bounds.append(c)  # sentinel = total word count

    # ── HARD MISMATCH GATE (2026-05-27) ──────────────────────────────
    # If the script is a SUBSET of a much-longer voiceover (e.g. user
    # uploads a 25-min VO but the test script is only 200 words), Whisper
    # alignment will fail to anchor the trailing words and the
    # proportional-split fallback will stretch each scene across the FULL
    # audio — producing a 24-min video from a 2-min script. That happened
    # on the Mossad partial-script test. Refuse to render in that case
    # and force the caller to either trim the voiceover, fix the script,
    # or drop the voiceover (fall through to TTS).
    #
    # Plausible human speech rate: 1.5–4.0 words/sec (90–240 wpm). Allow
    # ±35% slack on top for natural pauses, breath, intro/outro silence.
    # If voiceover sits outside [hard_min, hard_max] we hard-error.
    # Env override `VIDLORE_ALLOW_VO_MISMATCH=1` re-enables the old
    # proportional-stretch behaviour for explicit experiments.
    wc = len(flat)
    if wc >= 40 and total > 1.0 and os.environ.get(
        "VIDLORE_ALLOW_VO_MISMATCH"
    ) != "1":
        exp_lo = wc / 4.0           # very fast speaker
        exp_hi = wc / 1.5           # very slow speaker
        hard_lo = exp_lo * 0.65
        hard_hi = exp_hi * 1.35
        if total < hard_lo or total > hard_hi:
            raise RuntimeError(
                "[narrate_from_file] voiceover/script length mismatch: "
                f"script={wc} words (expects {exp_lo:.0f}-{exp_hi:.0f}s "
                f"at 90-240 wpm), voiceover={total:.1f}s. "
                "Refusing to render — proportional split would stretch "
                "scenes across the full audio and produce a "
                "wrong-duration video. "
                "Fix: trim the voiceover to match the script, expand the "
                "script to match the voiceover, or omit --voiceover to "
                "use TTS. "
                "Override with VIDLORE_ALLOW_VO_MISMATCH=1 if intentional."
            )

    aligned = None
    if flat:
        try:
            from .align import align_script

            aligned = align_script(master, flat)
        except Exception:
            aligned = None

    # DECISIVE plausibility gate. Even when align_script returns a full
    # list, a bad Whisper run can leave it degenerate (one scene's last
    # word lands tens of seconds late while the rest collapse) — that is
    # exactly what produced a ~90s FROZEN segment in testing. Derive the
    # scene durations alignment WOULD give and reject the whole thing if
    # any scene balloons, too many collapse, non-finite times appear, or
    # the audio is poorly covered. Falling back to the proportional
    # split is always safe (uniform, in sync, never freezes).
    if aligned and len(aligned) == len(flat):
        bad = any(
            not (math.isfinite(s) and math.isfinite(e))
            for s, e in aligned
        )
        wcount = [max(1, bounds[i + 1] - bounds[i]) for i in range(n)]
        wsum = float(sum(wcount)) or 1.0
        prev = 0.0
        sdur: list[float] = []
        for i in range(n):
            a, b = bounds[i], bounds[i + 1]
            if i == n - 1:
                end = total
            elif b > a:
                end = aligned[b - 1][1]
            else:
                end = prev + 0.2
            end = min(max(end, prev), total)
            sdur.append(end - prev)
            prev = end
        exp = [total * c / wsum for c in wcount]
        ballooned = any(
            sdur[i] > max(9.0, 3.2 * exp[i]) for i in range(n)
        )
        collapsed = sum(1 for d in sdur if d <= 0.4)
        covered = aligned[-1][1] >= 0.60 * total
        if bad or ballooned or collapsed > max(1, int(0.20 * n)) \
                or not covered:
            print("  [align] alignment implausible "
                  "(degenerate scene durations) — proportional split",
                  flush=True)
            aligned = None

    scenes: list[NarratedScene] = []
    if aligned and len(aligned) == len(flat):
        print("  [align] voiceover word-aligned (Whisper) — exact sync",
              flush=True)
        prev_end = 0.0
        for i, sc in enumerate(script.scenes):
            a, b = bounds[i], bounds[i + 1]
            start = prev_end                       # contiguous: no drift
            if i == n - 1:
                end = total
            elif b > a:
                end = max(aligned[b - 1][1], start + 0.2)
            else:                                  # empty scene
                end = start + 0.2
            end = min(end, total)
            if end <= start:
                end = min(total, start + 0.2)
            w_times = [
                WordTiming(
                    flat[k],
                    min(max(start, aligned[k][0]), end),
                    min(max(aligned[k][0], aligned[k][1]), end),
                )
                for k in range(a, b)
            ]
            dur = max(0.2, end - start)
            wav = workdir / f"scene_{sc.index:03d}.wav"
            _slice_scene(master, start, end, total, wav)
            scenes.append(NarratedScene(sc.index, wav, dur, w_times))
            prev_end = end
        return Narration(scenes=scenes, audio=master, reused=0)

    # --- Fallback: proportional split (alignment unavailable) ------- #
    print("  [align] Whisper unavailable — proportional voiceover split",
          flush=True)
    weights = [max(1, len(s.narration.split())) for s in script.scenes]
    wsum = float(sum(weights)) or 1.0
    timeline = 0.0
    acc = 0.0
    for i, sc in enumerate(script.scenes):
        start = acc / wsum * total
        acc += weights[i]
        end = (acc / wsum * total) if i < n - 1 else total
        dur = max(0.2, end - start)
        wav = workdir / f"scene_{sc.index:03d}.wav"
        _slice_scene(master, start, end, total, wav)
        words = _spread_words(sc.narration, timeline, dur)
        scenes.append(NarratedScene(sc.index, wav, dur, words))
        timeline += dur
    return Narration(scenes=scenes, audio=master, reused=0)


# Re-voice deltas (% speech-rate nudge). edge-tts is deterministic, so a
# different rate yields a different *take* of the same scene; ElevenLabs
# is stochastic so the cache-bust alone already gives a fresh take.
_RATE_DELTAS = (0, -3, 4, -5, 6, -2, 8)


def _jitter_rate(rate: str, n: int) -> str:
    if not n:
        return rate
    m = re.match(r"\s*([+-]?\d+)\s*%\s*$", rate or "")
    base = int(m.group(1)) if m else 0
    return f"{base + _RATE_DELTAS[n % len(_RATE_DELTAS)]:+d}%"


def _to_44100(src: Path, dst: Path, *, trim_head: bool = False) -> None:
    """Resample a backend's native wav to the pipeline format (44100/stereo).
    When trim_head (scene 0) cap leading silence to ~0.15s so the doc opens on
    the voice, not a dead pause (mirrors the edge-tts path / IMP_013)."""
    af = []
    if trim_head:
        af.append("silenceremove=start_periods=1:start_silence=0.15:"
                  "start_threshold=-45dB")
    cmd = ["-i", str(src)]
    if af:
        cmd += ["-af", ",".join(af)]
    cmd += ["-ar", "44100", "-ac", "2", str(dst)]
    run(cmd)


def _premium_settings_sig(preset_key: str, device: str, settings: dict) -> str:
    """Stable cache signature for a premium voice config."""
    items = sorted((settings or {}).items())
    return f"{preset_key}|{device}|" + ";".join(f"{k}={v}" for k, v in items)


def narrate_premium(
    script: Script,
    workdir: Path,
    *,
    preset_key: str,
    backend_name: str = "chatterbox",
    cache_dir: Path | None = None,
    device: str = "auto",
    settings: dict | None = None,
    allow_legacy: bool = True,
    voice_variants: dict[int, int] | None = None,
) -> "Narration":
    """Premium LOCAL narration via the tts_backends sidecar (no paid APIs).

    Per-scene content-addressed cache (re-render skips unchanged scenes), a
    fallback chain (premium -> fast fallback -> legacy, never silent-bad), and
    -16 LUFS post-processing. Returns the same :class:`Narration` shape as
    :func:`narrate` so the rest of the pipeline is unchanged."""
    from . import tts_backends as _tb
    from .voice_presets import get_preset

    workdir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    preset = get_preset(preset_key)
    settings = dict(settings or {})
    voice_variants = voice_variants or {}
    sig = _premium_settings_sig(preset.key, device, settings)
    chain = _tb.resolve_chain(backend_name, allow_legacy=allow_legacy)

    # 1) split cached vs to-synthesize (final 44100 wav is what we cache)
    todo, reused = [], 0
    cache_map: dict[int, tuple] = {}
    for sc in script.scenes:
        wav_p = workdir / f"scene_{sc.index:03d}.wav"
        vv = int(voice_variants.get(sc.index, 0))
        key = scene_key("ttsp", backend_name, preset.key, sig, sc.narration,
                        *(("vv", vv) if vv else ()))
        cw = cache_dir / f"{key}.wav" if cache_dir else None
        cm = cache_dir / f"{key}.json" if cache_dir else None
        if cw and cw.exists() and cm.exists():
            shutil.copyfile(cw, wav_p)
            dur = json.loads(cm.read_text(encoding="utf-8"))["duration"]
            cache_map[sc.index] = (wav_p, dur, True)
            reused += 1
        else:
            todo.append((sc, wav_p, cw, cm))

    # 2) synthesize the uncached scenes through the fallback chain
    if todo:
        raw = {sc.index: workdir / f"scene_{sc.index:03d}_raw.wav"
               for sc, _, _, _ in todo}
        items = [_tb.SynthItem(sc.narration, raw[sc.index])
                 for sc, _, _, _ in todo]
        last_err = None
        used = None
        for name in chain:
            backend = _tb.get_backend(name)
            why = backend.is_available() if backend else "no backend"
            if why:
                print(f"  [tts] backend '{name}' unavailable: {why}", flush=True)
                continue
            try:
                print(f"  [tts] premium voice via '{name}' "
                      f"({len(items)} scenes, preset={preset.key}) ...",
                      flush=True)
                backend.synth_batch(items, preset, device=device,
                                    settings=settings)
                used = name
                break
            except Exception as e:                             # noqa: BLE001
                last_err = e
                print(f"  [tts] backend '{name}' failed ({str(e)[:120]}); "
                      "trying next in chain", flush=True)
        if used is None:
            raise RuntimeError(
                f"all premium TTS backends failed (chain={chain}): {last_err}")
        # 3) post-process each new scene -> 44100/stereo, cache it
        for sc, wav_p, cw, cm in todo:
            _to_44100(raw[sc.index], wav_p, trim_head=(sc.index == 0))
            raw[sc.index].unlink(missing_ok=True)
            dur = _wav_duration(wav_p)
            cache_map[sc.index] = (wav_p, dur, False)
            if cw is not None:
                shutil.copyfile(wav_p, cw)
                cm.write_text(json.dumps({"duration": dur, "backend": used}),
                              encoding="utf-8")

    # 4) timeline + concat (order matters)
    scenes: list[NarratedScene] = []
    timeline = 0.0
    lines: list[str] = []
    for sc in script.scenes:
        wav_p, dur, _ = cache_map[sc.index]
        words = _spread_words(sc.narration, timeline, dur)
        scenes.append(NarratedScene(sc.index, wav_p, dur, words))
        lines.append(f"file '{wav_p.name}'")
        timeline += dur
    concat_list = workdir / "narr_concat.txt"
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw_full = workdir / "narration_full_raw.wav"
    run(["-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(raw_full)])
    # 5) loudness-normalize the whole narration to -16 LUFS (premium polish)
    full = workdir / "narration_full.wav"
    try:
        run(["-i", str(raw_full), "-af",
             "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100:async=1",
             "-ar", "44100", "-ac", "2", str(full)])
        if not full.exists() or _wav_duration(full) < 0.1:
            raise RuntimeError("loudnorm produced empty output")
    except Exception as e:                                     # noqa: BLE001
        print(f"  [tts] loudnorm skipped ({str(e)[:80]}); using raw concat",
              flush=True)
        shutil.copyfile(raw_full, full)
    return Narration(scenes=scenes, audio=full, reused=reused)


def narrate(
    script: Script,
    voice: str,
    workdir: Path,
    *,
    rate: str = "+6%",
    cache_dir: Path | None = None,
    provider: str = "edge",
    el_api_key: str = "",
    el_model: str = "eleven_turbo_v2_5",
    fallback_voice: str = "en-US-GuyNeural",
    voice_variants: dict[int, int] | None = None,
) -> Narration:
    """Synthesize narration via ``provider`` ("edge" | "elevenlabs").
    When ``cache_dir`` is given, a scene whose provider/voice/model/text
    is unchanged is restored from cache instead of being re-synthesized
    (the per-scene-regenerate differentiator). ``voice_variants`` maps
    scene index -> re-voice counter; a bumped counter busts that scene's
    TTS cache and produces a fresh take. ``reused`` counts cache hits."""
    workdir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[NarratedScene] = []
    timeline = 0.0
    reused = 0
    concat_list = workdir / "narr_concat.txt"
    lines: list[str] = []

    voice_variants = voice_variants or {}

    # AUTO LANGUAGE-AWARE VOICE -- if the caller passed the default
    # English voice (en-US-GuyNeural) but the SCRIPT is actually in
    # another language, auto-route to the matching neural voice.  This
    # keeps EN renders byte-identical (default voice unchanged) while
    # JP/KR/AR/UR/HE scripts get native-sounding TTS without the user
    # having to know voice IDs.  Explicit non-default voices are
    # respected (the user picked them on purpose).
    _DEFAULT_VOICES = {"en-US-GuyNeural", "en-US-JennyNeural"}
    if voice in _DEFAULT_VOICES and script.scenes:
        try:
            from . import lang as _lang
            sample = " ".join(s.narration[:60] for s in script.scenes[:3])
            auto_voice = _lang.voice_for(text=sample, default=voice)
            if auto_voice != voice:
                print(f"  [tts] auto-voice: '{voice}' -> '{auto_voice}' "
                      f"(detected script: {_lang.detect_script(sample)})",
                      flush=True)
                voice = auto_voice
        except Exception:                                      # noqa: BLE001
            pass

    # ── PARALLEL TTS (USER-REPORTED RENDER SPEED FIX 2026-05-25) ──
    # Old loop ran each scene's TTS network call + ffmpeg conversion
    # serially → for 14 scenes that meant ~25s of pure waiting on the
    # edge-tts service.  TTS is I/O-bound, so a small ThreadPool brings
    # this down to ~3-5s.  Default 4 workers — Microsoft's edge-tts
    # backend rate-limits aggressive concurrency (8+ tends to 503).
    # Order is preserved by sorting the results dict by scene index
    # before the (serial) timeline pass.
    import os as _os
    import time as _time_tts
    from concurrent.futures import ThreadPoolExecutor as _Pool
    try:
        # V1 safe-tuning: documented VIDLORE_AUDIO_WORKERS alias → legacy
        # VIDLORE_TTS_WORKERS → unchanged default 4 (LEVEL-A: default byte-
        # identical; edge-tts rate-limits ≥8 so keep conservative).
        _tts_workers = max(1, int(
            _os.environ.get("VIDLORE_AUDIO_WORKERS")
            or _os.environ.get("VIDLORE_TTS_WORKERS") or "4"))
    except ValueError:
        _tts_workers = 4

    def _one_tts(sc):
        """Synthesize ONE scene with retry-on-transient-failure.
        Safe for parallel execution. Returns (idx, wav, dur, reused)."""
        wav_p = workdir / f"scene_{sc.index:03d}.wav"
        vv_ = int(voice_variants.get(sc.index, 0))
        eff_rate_ = _jitter_rate(rate, vv_)
        key_ = scene_key(
            "tts", provider, el_model, sc.narration, voice, eff_rate_,
            *(("vv", vv_) if vv_ else ()),
        )
        cw = cache_dir / f"{key_}.wav" if cache_dir else None
        cm = cache_dir / f"{key_}.json" if cache_dir else None

        if cw and cw.exists() and cm.exists():
            shutil.copyfile(cw, wav_p)
            dur_ = json.loads(cm.read_text(encoding="utf-8"))["duration"]
            return (sc.index, wav_p, dur_, True)

        mp3_ = workdir / f"scene_{sc.index:03d}.mp3"
        # RETRY LOOP — edge-tts occasionally returns WSServerHandshakeError
        # 503 / Invalid response status (rate-limit / transient backend).
        # Three tries with exponential backoff so one flaky scene doesn't
        # kill the whole render. Per-scene retries are isolated thanks to
        # the parallel ThreadPool — others keep flowing.
        _last_exc = None
        for _attempt in range(3):
            try:
                _make_scene_mp3(
                    provider=provider, text=sc.narration, voice=voice,
                    rate=eff_rate_, el_api_key=el_api_key,
                    el_model=el_model, fallback_voice=fallback_voice,
                    mp3=mp3_,
                )
                _last_exc = None
                break
            except Exception as e:                          # noqa: BLE001
                _last_exc = e
                if _attempt < 2:
                    _time_tts.sleep(1.0 * (2 ** _attempt))   # 1s, 2s
        if _last_exc is not None:
            raise _last_exc
        run(["-i", str(mp3_), "-ar", "44100", "-ac", "2", str(wav_p)])
        # IMP_013 — narration starts at frame 1. The OPENING scene must not
        # begin with a dead pause: cap any leading silence on scene 0 to a
        # natural ~0.15s micro-breath (threshold-based, so it never clips the
        # first word). Premium docs open with the voice immediately over the
        # first frame — no music-only / silent hold. Measured AFTER the trim
        # so the timeline/caption sync keys off the real (trimmed) duration.
        if sc.index == 0:
            try:
                _hp = wav_p.with_name(wav_p.stem + "_head.wav")
                run(["-i", str(wav_p), "-af",
                     "silenceremove=start_periods=1:start_silence=0.15:"
                     "start_threshold=-45dB",
                     "-ar", "44100", "-ac", "2", str(_hp)])
                if _hp.exists() and _wav_duration(_hp) > 0.3:
                    shutil.move(str(_hp), str(wav_p))
                else:
                    _hp.unlink(missing_ok=True)
            except Exception:                                  # noqa: BLE001
                pass
        dur_ = _wav_duration(wav_p)
        if cw is not None:
            shutil.copyfile(wav_p, cw)
            cm.write_text(json.dumps({"duration": dur_}), encoding="utf-8")
        return (sc.index, wav_p, dur_, False)

    # Submit ALL scenes in parallel; collect results indexed by scene idx
    results: dict[int, tuple] = {}
    with _Pool(max_workers=min(_tts_workers, len(script.scenes))) as pool:
        for r in pool.map(_one_tts, script.scenes):
            results[r[0]] = r

    # Second pass: serial timeline accumulation (order matters) — this
    # part is microseconds, no need to parallelize.
    for sc in script.scenes:
        idx, wav, dur, was_reused = results[sc.index]
        if was_reused:
            reused += 1
        words = _spread_words(sc.narration, timeline, dur)
        scenes.append(NarratedScene(sc.index, wav, dur, words))
        lines.append(f"file '{wav.name}'")
        timeline += dur


    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    full = workdir / "narration_full.wav"
    run(
        [
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy", str(full),
        ]
    )
    return Narration(scenes=scenes, audio=full, reused=reused)
