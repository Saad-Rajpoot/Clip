"""Runtime config: loads keys from environment / .env and selects providers.
Mirrors Vidlore's 'hybrid' idea: works free out of the box, upgrades when
keys are present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        # Load from .env when the var is unset OR present-but-blank (some
        # hosts export empty ANTHROPIC_API_KEY etc. which must not shadow
        # the real value the user put in .env). A real value still wins.
        if key and val and not os.environ.get(key, "").strip():
            os.environ[key] = val


@dataclass
class Config:
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    pexels_api_key: str = ""
    shutterstock_api_key: str = ""
    pollinations_api_key: str = ""
    fal_key: str = ""
    fal_model: str = "fal-ai/flux/schnell"
    fal_video_model: str = ""  # set e.g. fal-ai/ltxv-13b-098-distilled
    #                            -> real moving clip per scene (paid, slow)
    aimg_enabled: bool = True
    music_enabled: bool = True
    transitions_enabled: bool = True
    overlays_enabled: bool = True
    sfx_enabled: bool = False  # OFF: the repeated boom bored/annoyed the
    # user every time; the soft music bed is the cinematic sound now.
    # Opt back in only via VIDLORE_SFX=1.
    voice: str = "en-US-GuyNeural"
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_turbo_v2_5"
    elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB"  # "Adam" (deep narrator)
    # ── Premium LOCAL voiceover (self-hosted TTS sidecar; no paid APIs) ──
    # backend: premium_local (chatterbox) | fallback_fast (kokoro) | legacy (edge)
    tts_backend: str = "legacy"            # default stays legacy until user opts in
    tts_model: str = "chatterbox"          # which sidecar engine for premium_local
    tts_voice: str = "deep_male_documentary"  # a voice_presets.py preset key
    tts_device: str = "auto"               # auto | cuda | mps | cpu
    tts_cache: bool = True
    root: Path = Path(".")

    @property
    def has_premium_local(self) -> bool:
        return (self.tts_backend or "").strip().lower() == "premium_local"

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_stock(self) -> bool:
        return bool(self.pexels_api_key)

    @property
    def has_shutterstock(self) -> bool:
        """Shutterstock tier 2 stock-video.  Note: free-trial / search-
        only tokens return WATERMARKED preview clips — fine for dev,
        not for final deliverables.  A license-enabled token returns
        clean previews."""
        return bool(self.shutterstock_api_key)

    @property
    def has_fal(self) -> bool:
        return bool(self.fal_key)

    @property
    def has_fal_video(self) -> bool:
        # USER SPEC: "video nae banani ai sa" -- AI must NEVER generate
        # video clips, only still images.  Hard-disabled at the source so
        # every consumer (fetch_footage AI tier, pipeline banner, etc.)
        # automatically routes through the image-only path.
        # Set VIDLORE_AI_VIDEO=1 to re-enable (intentional opt-in only).
        import os as _os
        if _os.environ.get("VIDLORE_AI_VIDEO", "0").strip() not in (
                "1", "true", "yes", "on"):
            return False
        return bool(self.fal_key and self.fal_video_model)

    @property
    def has_elevenlabs(self) -> bool:
        return bool(self.elevenlabs_api_key)

    @property
    def has_aimg(self) -> bool:
        # Master toggle for real per-scene visuals (Openverse keyless by
        # default; Pollinations AI used instead when a key is set).
        return self.aimg_enabled

    def _voice_desc(self) -> str:
        if self.has_elevenlabs:
            return "ElevenLabs (%s) voice=%s" % (
                self.elevenlabs_model, self.elevenlabs_voice_id
            )
        return "edge-tts (free) voice=%s" % self.voice

    def _footage_desc(self) -> str:
        if self.has_stock:
            fb = ("fal.ai VIDEO" if self.has_fal_video
                  else "fal.ai" if self.has_fal else "AI")
            ss = " + Shutterstock tier 2" if self.has_shutterstock else ""
            return f"real Pexels stock video{ss} ({fb} fallback)"
        if self.has_fal_video:
            return f"fal.ai VIDEO {self.fal_video_model} (LLM-directed)"
        if self.has_fal:
            return f"fal.ai {self.fal_model} (LLM-directed)"
        if self.has_aimg:
            if self.pollinations_api_key:
                return "Pollinations AI images (keyed) -> Openverse fallback"
            return "Openverse CC images (free, keyless)"
        return "themed Ken-Burns slides"

    def describe(self) -> str:
        return (
            "  script:  %s\n"
            "  voice:   %s\n"
            "  footage: %s\n"
            "  music:   %s\n"
            "  trans:   %s\n"
            "  overlay: %s"
            % (
                "Anthropic LLM (%s)" % self.anthropic_model
                if self.has_llm
                else "manual --script required (no ANTHROPIC_API_KEY)",
                self._voice_desc(),
                self._footage_desc(),
                "auto theme bed (synthesized, free)"
                if self.music_enabled
                else "off",
                "scene crossfades" if self.transitions_enabled
                else "hard cuts",
                "title card + subscribe CTA"
                if self.overlays_enabled else "off",
            )
        )


def load_config(root: Path | None = None) -> Config:
    root = Path(root or os.getcwd())
    _load_dotenv(root / ".env")
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip(),
        pexels_api_key=os.environ.get("PEXELS_API_KEY", "").strip(),
        shutterstock_api_key=os.environ.get(
            "SHUTTERSTOCK_API_KEY", "").strip(),
        pollinations_api_key=os.environ.get("POLLINATIONS_API_KEY", "").strip(),
        fal_key=os.environ.get("FAL_KEY", "").strip(),
        fal_model=os.environ.get("FAL_MODEL", "fal-ai/flux/schnell").strip(),
        fal_video_model=os.environ.get("FAL_VIDEO_MODEL", "").strip(),
        aimg_enabled=os.environ.get("VIDLORE_AIMG", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        music_enabled=os.environ.get("VIDLORE_MUSIC", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        transitions_enabled=os.environ.get("VIDLORE_TRANSITIONS", "1")
        .strip().lower() not in ("0", "false", "no", "off"),
        overlays_enabled=os.environ.get("VIDLORE_OVERLAYS", "1")
        .strip().lower() not in ("0", "false", "no", "off"),
        sfx_enabled=os.environ.get("VIDLORE_SFX", "0")
        .strip().lower() in ("1", "true", "yes", "on"),
        voice=os.environ.get("VIDLORE_VOICE", "en-US-GuyNeural").strip(),
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", "").strip(),
        elevenlabs_model=os.environ.get(
            "ELEVENLABS_MODEL", "eleven_turbo_v2_5"
        ).strip(),
        elevenlabs_voice_id=os.environ.get(
            "ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB"
        ).strip(),
        tts_backend=os.environ.get("VIDLORE_TTS_BACKEND", "legacy").strip(),
        tts_model=os.environ.get("VIDLORE_TTS_MODEL", "chatterbox").strip(),
        tts_voice=os.environ.get(
            "VIDLORE_TTS_VOICE", "deep_male_documentary").strip(),
        tts_device=os.environ.get("VIDLORE_TTS_DEVICE", "auto").strip(),
        tts_cache=os.environ.get("VIDLORE_TTS_CACHE", "1")
        .strip().lower() not in ("0", "false", "no", "off"),
        root=root,
    )
