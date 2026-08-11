"""ClipStudio configuration.

Reads `VIDLORE_CLIPSTUDIO_*` env flags into a typed `ClipConfig`, and bridges to the engine's
own config (`vidlore.config.load_config`) and ffmpeg resolver (`vidlore.ffmpeg_tool`) so the
module shares the engine's API keys and binary discovery instead of re-implementing them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _s(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _cpu_count() -> int:
    """Logical core count of THIS machine (per-machine portable — an M5 Pro reports more than a
    laptop, so worker defaults auto-scale instead of being hard-coded)."""
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


_NCPU = _cpu_count()


def _workers(name: str, normal: int, turbo: int) -> int:
    """Resolve a parallelism knob. An explicit env override (>0) always wins; otherwise pick the
    `turbo` ceiling when VIDLORE_CLIPSTUDIO_MAX_CPU is on (saturate every core for fastest render
    on a powerful box), else the gentler `normal` default (leaves headroom for the rest of the OS).
    Result is floored at 1."""
    v = _i(name, 0)
    if v > 0:
        return max(1, v)
    pick = turbo if _b("VIDLORE_CLIPSTUDIO_MAX_CPU", False) else normal
    return max(1, pick)


def verify_prefetch_workers() -> int:
    """How many verdict warms the render drivers run at once.

    This pool is API-bound, not CPU-bound: it waits on vision answers. Measured on job 218acdfe10
    at 4 workers — 230 verdicts warmed in 286s, i.e. ~1.24s per beat against a ~5s per-call latency
    — while the SERIAL ladder in the same stage spent 63.9s per beat across 64 escalation benches.
    The pool is not the bottleneck; its width is.

    Turbo (VIDLORE_CLIPSTUDIO_MAX_CPU, which the portal and both CLI drivers set) raises it to 12.
    Overshooting degrades safely by design: a warm that fails is simply not cached, the serial loop
    re-asks with its own retry and breaker, and a burst of failures aborts the prefetch early — so
    the worst case is exactly today's behaviour. An explicit env value always wins, including =1
    for the fully-serial path the breaker suites require.
    """
    return _workers("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", 4, 12)


@dataclass
class ClipConfig:
    # --- matching ---
    # Score = w_clip*clip01 + w_trans*trans + w_face*face + w_obj*obj - penalties, clamped [0,1].
    # CLIP is the BASE (a strong visual match alone clears min_confidence); the others are
    # additive BONUSES that strengthen confidence rather than gate it.
    min_confidence: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_MIN_CONFIDENCE", 0.42))
    w_clip: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_W_CLIP", 0.80))
    w_trans: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_W_TRANS", 0.20))
    # DIALOGUE-LOCK: when a beat's iconic quote is actually SPOKEN in a clip's ASR, that clip IS the
    # exact scene — weight it strongly so it beats a merely-similar-looking generic clip.
    w_dialogue: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_W_DIALOGUE", 0.55))
    # Face-ID is a strong identity signal (real recognition, not mere face presence).
    w_face: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_W_FACE", 0.30))
    w_obj: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_W_OBJ", 0.10))
    reuse_penalty: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_REUSE_PENALTY", 0.06))
    # small tie-breaker against frames carrying a burned-in source dialogue subtitle (prefer clean)
    subtitle_penalty: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_SUBTITLE_PENALTY", 0.06))
    # heavy penalty when a beat names character A but the shot CONFIRMS a different main
    # character B (e.g. Robb's face on a "Tywin" line) — large enough to lose to any
    # non-wrong shot, but it stays a last-resort fallback if nothing else exists
    wrongface_penalty: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_WRONGFACE_PENALTY", 0.5))
    # how to handle a source carrying a persistent rival-channel watermark: "crop" = KEEP it but
    # punch-in-zoom the logo off-frame (preserves relevance) | "drop" = exclude the whole source.
    watermark_mode: str = field(default_factory=lambda: _s("VIDLORE_CLIPSTUDIO_WATERMARK_MODE", "crop"))
    # --- AI narration voice (reuses the engine's tts stack — vidlore unchanged) ---
    # "chatterbox" / "kokoro" = LOCAL neural AI voice (no API key, best quality) via
    # narrate_premium; "elevenlabs" = cloud AI (needs key); "edge" = free Microsoft TTS.
    voice_provider: str = field(default_factory=lambda: _s("VIDLORE_CLIPSTUDIO_VOICE_PROVIDER", "kokoro"))
    # voice character for the neural backends (see vidlore/voice_presets.py)
    voice_preset: str = field(default_factory=lambda: _s("VIDLORE_CLIPSTUDIO_VOICE_PRESET", "deep_male_documentary"))
    period_penalty: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_PERIOD_PENALTY", 0.20))
    # CLIP cosine → 0..1 calibration (model-specific; tune if flagging is mis-calibrated).
    clip_cos_lo: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_CLIP_COS_LO", 0.20))
    clip_cos_hi: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_CLIP_COS_HI", 0.34))

    # --- pacing / selection constraints ---
    # Scenes are now CONTENT-DRIVEN, not uniform: each natural clause/sentence becomes one beat, so a
    # punchy 3-word line → ~1.5s fast cut and a long dramatic sentence → a 6-8s hold (expert rhythm).
    target_clip_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_TARGET_CLIP_SEC", 2.5))
    min_clip_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_MIN_CLIP_SEC", 1.2))
    max_clip_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_MAX_CLIP_SEC", 8.0))
    min_scene_words: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_MIN_SCENE_WORDS", 4))
    max_scene_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_MAX_SCENE_SEC", 8.0))
    max_reuse_per_source: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_MAX_REUSE_PER_SOURCE", 40))
    max_reuse_per_shot: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_MAX_REUSE_PER_SHOT", 2))
    max_consecutive_same_source: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_MAX_CONSECUTIVE_SAME_SOURCE", 3))
    # Anti-repetition: a shot AND its visual near-duplicates are penalized if used within the last
    # `recency_cooldown` beats; the penalty decays to 0 across that window, so a scene may return
    # after a long gap but never repeats back-to-back. near_dup_cos = CLIP cosine for "same scene".
    # CRITICAL: two shots from the SAME source within `scene_gap_sec` are the same continuous scene
    # (adjacent cuts look identical to a viewer) — treated as a repeat even if keyframes differ.
    recency_weight: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_RECENCY_WEIGHT", 0.7))
    recency_cooldown: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_RECENCY_COOLDOWN", 48))
    near_dup_cos: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_NEAR_DUP_COS", 0.90))
    scene_gap_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_SCENE_GAP_SEC", 8.0))
    source_recency_weight: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_SOURCE_RECENCY_WEIGHT", 0.3))
    source_recency_window: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_SOURCE_RECENCY_WINDOW", 7))
    candidates_per_segment: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_CANDIDATES_PER_SEGMENT", 6))
    # PER-WINDOW anti-repeat. The penalties above are keyed on the SOURCE, so a window's 2nd airing
    # cost nothing extra below the hard cap and one shot could win 6 beats of a 22-minute cut (audit
    # 2026-07-26: 46% of delivered scenes were visual repeats of another scene, one look aired 9×,
    # several only 2-5s apart across a cut). These key on the WINDOW and on TIMELINE SECONDS — beat
    # index is the wrong clock, because a 272-beat essay and a 189-beat one have the same cooldown in
    # beats but very different runtimes.
    window_reuse_penalty: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_WINDOW_REUSE_PENALTY", 0.25))
    window_reuse_gap_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_WINDOW_REUSE_GAP_SEC", 90.0))
    window_reuse_recency_weight: float = field(
        default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_WINDOW_REUSE_RECENCY_WEIGHT", 0.8))
    # HARD block: the identical window may never return inside this gap (either clock). A viewer reads
    # a sub-20s repeat as a cut that goes nowhere, no matter how well the beat scores.
    window_min_gap_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_WINDOW_MIN_GAP_SEC", 20.0))
    window_min_gap_beats: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_WINDOW_MIN_GAP_BEATS", 3))
    # `relax` (single-scene deep-dive on scarce anchor footage) multiplies the per-shot cap. 3× took
    # the cap to 6 airings of one window, which is never editorially right — 2× is the deep-dive
    # allowance without the loop.
    relax_reuse_mult: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_RELAX_REUSE_MULT", 2))

    # --- index ---
    whisper_model: str = field(default_factory=lambda: _s("VIDLORE_CLIPSTUDIO_WHISPER_MODEL", "base"))
    whisper_compute: str = field(default_factory=lambda: _s("VIDLORE_CLIPSTUDIO_WHISPER_COMPUTE", "int8"))
    # CPU threads for the (single, serial) ASR pass — the heaviest per-source index step. Default
    # uses ~half the cores (turbo: all of them); 0 from env means "let ctranslate2 decide".
    whisper_cpu_threads: int = field(default_factory=lambda: _workers(
        "VIDLORE_CLIPSTUDIO_WHISPER_THREADS", max(2, _NCPU // 2), _NCPU))
    scene_threshold: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_SCENE_THRESHOLD", 27.0))
    min_shot_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_MIN_SHOT_SEC", 1.0))
    detect_faces: bool = field(default_factory=lambda: _b("VIDLORE_CLIPSTUDIO_DETECT_FACES", True))
    detect_ocr: bool = field(default_factory=lambda: _b("VIDLORE_CLIPSTUDIO_DETECT_OCR", True))
    dup_hamming: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DUP_HAMMING", 6))

    # --- ingest / download ---
    # Parallel HD downloads. Kept modest by default (YouTube throttles aggressive fan-out); turbo
    # raises it. Env VIDLORE_CLIPSTUDIO_CONCURRENCY still overrides explicitly.
    download_concurrency: int = field(default_factory=lambda: _workers(
        "VIDLORE_CLIPSTUDIO_CONCURRENCY", min(4, max(2, _NCPU // 4)), min(6, max(2, _NCPU // 2))))
    download_retries: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_RETRIES", 3))
    max_height: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_MAX_HEIGHT", 1080))

    # --- performance / parallelism ---
    # Stage-5 cut runs many INDEPENDENT libx264 (-preset veryfast) trims — pure CPU, scales almost
    # linearly with cores. Default leaves one core free; turbo (VIDLORE_CLIPSTUDIO_MAX_CPU=1) uses
    # all. (The final assemble/encode is the parent engine on Apple VideoToolbox — HW-accelerated,
    # not affected by CPU thread counts — so this is the main CPU lever for render speed.)
    cut_workers: int = field(default_factory=lambda: _workers(
        "VIDLORE_CLIPSTUDIO_CUT_WORKERS", min(12, max(2, _NCPU - 1)), _NCPU))

    # --- discovery ---
    discover_target: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DISCOVER_TARGET", 18))
    discover_per_query: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DISCOVER_PER_QUERY", 6))
    discover_min_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_DISCOVER_MIN_SEC", 15))
    discover_max_sec: float = field(default_factory=lambda: _f("VIDLORE_CLIPSTUDIO_DISCOVER_MAX_SEC", 1500))
    # Quality floor + HD preference. PREFER >= prefer_height (720) in ranking + download, but DON'T
    # hard-reject SD: many specific scene clips on free platforms are only 360p, and a relevant 360p
    # scene beats an irrelevant 1080p trailer. Floor just removes true garbage.
    discover_min_height: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DISCOVER_MIN_HEIGHT", 300))
    discover_prefer_height: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DISCOVER_PREFER_HEIGHT", 720))
    discover_max_per_channel: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DISCOVER_MAX_PER_CHANNEL", 3))
    # guarantee footage coverage for each major character (so a multi-character script doesn't fill
    # up with only the most-clipped pairing). Allows up to this many EXTRA sources beyond target.
    discover_coverage_extra: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DISCOVER_COVERAGE_EXTRA", 6))
    discover_resolve_quality: bool = field(default_factory=lambda: _b("VIDLORE_CLIPSTUDIO_DISCOVER_RESOLVE_QUALITY", True))
    discover_resolve_limit: int = field(default_factory=lambda: _i("VIDLORE_CLIPSTUDIO_DISCOVER_RESOLVE_LIMIT", 28))

    def weights(self) -> dict:
        return {"clip": self.w_clip, "transcript": self.w_trans,
                "face": self.w_face, "object": self.w_obj}


def load_clip_config() -> ClipConfig:
    return ClipConfig()


# --- bridges to the engine (lazy so importing this module is cheap) ---

def engine_config():
    """The engine's typed Config (API keys, model ids, feature flags).
    Always load the CLONE's .env (load_config defaults to cwd/.env, which may be wrong)."""
    import os as _os
    from pathlib import Path as _Path
    from vidlore.config import load_config
    clone_root = _Path(__file__).resolve().parents[2]   # clipstudio/config.py -> clone root
    if (clone_root / ".env").exists():
        return load_config(root=clone_root)
    return load_config()


def ffmpeg_exe() -> str:
    from vidlore.ffmpeg_tool import ffmpeg_exe as _e
    return _e()


def ffprobe_exe() -> str:
    """Best-effort ffprobe path. imageio-ffmpeg ships ffmpeg only, so search next to it then
    common install dirs. PyAV is the primary probe path (see ingest.probe); this is a fallback."""
    import os.path as _op
    import sys as _sys
    exe = ".exe" if _sys.platform.startswith("win") else ""
    cands = [
        os.environ.get("VIDLORE_FFPROBE", "").strip(),     # explicit override always wins
        _op.join(_op.dirname(ffmpeg_exe()), "ffprobe" + exe),
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        # the only ffprobe on the original dev machine (existence-checked; inert elsewhere)
        _op.expanduser("~/pinokio/bin/miniconda/bin/ffprobe"),
    ]
    for c in cands:
        if c and _op.exists(c):
            return c
    return "ffprobe" + exe
