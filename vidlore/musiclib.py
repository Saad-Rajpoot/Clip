"""PROFESSIONAL DOCUMENTARY MUSIC ENGINE.

A real, curated MUSIC LIBRARY layered on top of the existing procedural
bed (which stays the always-works fallback). The engine:

  • SCANS a folder of royalty-free tracks the user curates, reading metadata
    (category, duration, LUFS, bpm, tags, license) into an index;
  • MAPS each stretch of the film to a music CATEGORY from the story arc
    (role + energy + style mode + topic), like a human editor;
  • SELECTS tracks with NO-REPEAT rotation;
  • COMPOSES a SCORE for the whole film — one track per cue, LUFS-matched,
    crossfaded at act breaks, with intro/outro fades and a swell into
    reveals — so the music feels EDITED, not pasted.
  • Falls back to the procedural bed when the library is empty.

LIBRARY LAYOUT (drop royalty-free files here, sorted by category folder):
    vidlore/assets/music/<category>/<track>.mp3|wav|m4a|ogg|flac
    vidlore/assets/music/<category>/<track>.json   (optional metadata)
    vidlore/assets/music/LICENSES.md               (track sources/licenses)

Categories (folder names):
    suspense  mystery  dark_investigation  emotional_piano  ambient
    historical_epic  military_tension  tech_cyber  financial
    survival_urgency  slow_reveal  climax_build  aftermath  neutral
    archive_texture
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .ffmpeg_tool import run

_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

CATEGORIES = (
    "suspense", "mystery", "dark_investigation", "emotional_piano",
    "ambient", "historical_epic", "military_tension", "tech_cyber",
    "financial", "survival_urgency", "slow_reveal", "climax_build",
    "aftermath", "neutral", "archive_texture",
)

# When a category has no tracks, fall back along these related chains so a
# cue always finds *something* fitting before dropping to procedural.
_FALLBACK_CHAIN = {
    "suspense": ("dark_investigation", "mystery", "military_tension",
                 "ambient", "neutral"),
    "mystery": ("suspense", "dark_investigation", "ambient", "neutral"),
    "dark_investigation": ("suspense", "mystery", "tech_cyber", "ambient",
                           "neutral"),
    "emotional_piano": ("aftermath", "ambient", "slow_reveal", "neutral"),
    "ambient": ("neutral", "mystery", "archive_texture"),
    "historical_epic": ("ambient", "slow_reveal", "emotional_piano",
                        "neutral"),
    "military_tension": ("suspense", "survival_urgency", "dark_investigation",
                         "neutral"),
    "tech_cyber": ("dark_investigation", "suspense", "ambient", "neutral"),
    "financial": ("tech_cyber", "neutral", "ambient"),
    "survival_urgency": ("military_tension", "suspense", "climax_build",
                         "neutral"),
    "slow_reveal": ("ambient", "emotional_piano", "mystery", "neutral"),
    "climax_build": ("suspense", "military_tension", "survival_urgency",
                     "neutral"),
    "aftermath": ("emotional_piano", "ambient", "slow_reveal", "neutral"),
    "neutral": ("ambient", "mystery"),
    "archive_texture": ("ambient", "neutral"),
}

# ---- topic keyword -> category bias (read from narration/title) -------- #
_TOPIC = (
    # ORDER MATTERS: first match wins.  Specific / aggressive topics
    # are listed FIRST so a "Cold-War spy" doc lands on
    # dark_investigation, not on the gentler historical bucket.
    (("spy", "intelligence", "espionage", "agent", "kgb", "cia", "covert",
      "classified", "defect", "surveillance"), "dark_investigation"),
    (("murder", "killer", "detective", "crime", "forensic", "case",
      "investigation", "evidence", "suspect"), "dark_investigation"),
    (("army", "war", "battle", "military", "soldier", "invasion", "troops",
      "operation", "combat", "siege"), "military_tension"),
    (("cyber", "hacker", "software", "ai", "computer", "data", "network",
      "digital", "code", "algorithm"), "tech_cyber"),
    (("money", "market", "finance", "economy", "dollar", "bank", "stock",
      "trade", "wealth", "profit"), "financial"),
    # User feedback: "homestead" was on the survival_urgency line which
    # gave 1860s Amish / garden / farm docs aggressive escape music.
    # Genuine survival-disaster only here now.
    (("survival shelter", "doomsday", "prepper", "disaster", "escape",
      "famine", "collapse", "apocalypse", "extinction", "evacuation"),
     "survival_urgency"),
    # GENTLE rural / historical / nature / craft topics -- ambient bed,
    # NOT dark or aggressive. These were missing from the table so they
    # used to fall through to the role default (suspense/ambient) and
    # often got picked up by the "ancient/century" historical_epic.
    (("garden", "pest", "crop", "harvest", "farm", "field", "barn",
      "homestead", "amish", "rural", "country life", "village",
      "ledger", "almanac", "orchard", "vineyard", "ranch", "cattle"),
     "ambient"),
    (("ancient", "empire", "dynasty", "medieval", "pharaoh", "rome",
      "civilization", "kingdom", "silk road", "antiquity"),
     "historical_epic"),
    (("century-old", "centuries-old", "ancestral", "heritage",
      "tradition", "handmade", "craft", "artisan", "old-world",
      "lost art", "forgotten"),
     "slow_reveal"),
    (("nature", "wildlife", "forest", "ocean", "mountain", "river",
      "wilderness", "ecosystem", "biology"),
     "ambient"),
    (("recipe", "food", "cooking", "kitchen", "bake", "brewing"),
     "neutral"),
    (("mystery", "secret", "unknown", "vanished", "disappear", "strange",
      "unexplained"), "mystery"),
)

_ROLE_CAT = {
    "hook": "suspense", "problem": "suspense", "stakes": "suspense",
    "context": "ambient", "escalation": "climax_build",
    "build": "climax_build", "turn": "slow_reveal", "reveal": "slow_reveal",
    "climax": "climax_build", "proof": "dark_investigation",
    "evidence": "dark_investigation", "reaction": "emotional_piano",
    "payoff": "aftermath", "resolution": "aftermath",
}


# ====================================================================== #
#  Library location + scanning
# ====================================================================== #
def library_root(cfg=None) -> Path:
    """The music library folder. Env override -> packaged assets/music."""
    env = (os.environ.get("VIDLORE_MUSIC_DIR") or "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "assets" / "music"


def _probe_duration(path: Path) -> float:
    """Track length in seconds via ffmpeg (no ffprobe dependency)."""
    try:
        import subprocess
        from .ffmpeg_tool import ffmpeg_exe
        r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
        if m:
            return (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                    + float(m.group(3)))
    except Exception:                                      # noqa: BLE001
        pass
    return 0.0


def _use_only_root() -> "Path":
    """The local USE-ONLY YTAL library (git-ignored, never bundled in dist).
    Organised the same way as the bundle library: <category>/<id>.mp3 + sidecar."""
    return (Path(__file__).resolve().parent / "audio_library" / "ytal_cache" / "music")


def _with_use_only(lib: dict, cfg=None) -> dict:
    """Merge the USE-ONLY YTAL tier into a bundle library index so the director
    can SELECT those tracks at render time. Raw files stay in ytal_cache/ and are
    excluded from dist. Env-gated (VIDLORE_YTAL_USE_ONLY=0 → bundle-only, e.g. for
    a strict dist build). Defensive: any failure returns the bundle lib unchanged."""
    import os as _os
    if _os.environ.get("VIDLORE_YTAL_USE_ONLY", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return lib
    try:
        root = _use_only_root()
        if not root.exists():
            return lib
        merged = {k: list(v) for k, v in lib.items()}
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in _AUDIO_EXT:
                continue
            cat = p.parent.name.lower()
            # Prefer the sidecar's stored duration; only probe ffmpeg as a
            # fallback. With hundreds of USE-ONLY tracks, probing every file on
            # every scan would add tens of seconds to each render.
            side = p.with_suffix(".json")
            smeta = {}
            if side.exists():
                try:
                    smeta = json.loads(side.read_text(encoding="utf-8"))
                except Exception:                              # noqa: BLE001
                    smeta = {}
            _d = smeta.get("dur") or smeta.get("duration") or smeta.get("measured_duration")
            try:
                dur = float(_d) if _d else round(_probe_duration(p), 2)
            except Exception:                                  # noqa: BLE001
                dur = round(_probe_duration(p), 2)
            meta = {"path": str(p), "name": p.stem, "category": cat,
                    "bpm": None, "tags": [], "license": "", "source": "youtube_al"}
            meta.update(smeta)
            meta["dur"] = round(float(dur), 2)
            meta["use_only"] = True
            meta["license_tier"] = "use_only"
            if meta["dur"] >= 3.0:
                merged.setdefault(cat, []).append(meta)
        return {k: v for k, v in merged.items() if v}
    except Exception:                                          # noqa: BLE001
        return lib


def scan(cfg=None, *, force: bool = False) -> dict:
    """Build/refresh the library index. Returns
    {category: [track-meta, ...]}. Cheap & cached in _index.json; pass
    force=True to rebuild. A track folder name IS its category; a sidecar
    <track>.json may override/extend metadata.

    The USE-ONLY YTAL tier (ytal_cache/, git-ignored, never bundled) is merged in
    on every scan via _with_use_only() so the director can select those tracks at
    render time without putting raw files in dist."""
    root = library_root(cfg)
    if not root.exists():
        return {}
    idx_path = root / "_index.json"
    files = sorted(p for p in root.rglob("*") if p.suffix.lower()
                   in _AUDIO_EXT)
    sig = hashlib.sha1(
        ("|".join(f"{p.relative_to(root)}:{p.stat().st_mtime_ns}"
                  for p in files)).encode()).hexdigest()[:16]
    if not force and idx_path.exists():
        try:
            cached = json.loads(idx_path.read_text(encoding="utf-8"))
            if cached.get("_sig") == sig:
                return _with_use_only({k: v for k, v in cached.items()
                                       if not k.startswith("_")}, cfg)
        except Exception:                                  # noqa: BLE001
            pass
    lib: dict[str, list] = {c: [] for c in CATEGORIES}
    for p in files:
        try:
            cat = p.parent.name.lower()
        except Exception:                                  # noqa: BLE001
            cat = "neutral"
        if cat not in lib:
            lib.setdefault(cat, [])
        meta = {"path": str(p), "name": p.stem, "category": cat,
                "dur": round(_probe_duration(p), 2),
                "bpm": None, "tags": [], "license": "", "source": ""}
        # bpm from filename token like "_120bpm"
        mb = re.search(r"(\d{2,3})\s*bpm", p.stem, re.I)
        if mb:
            meta["bpm"] = int(mb.group(1))
        side = p.with_suffix(".json")
        if side.exists():
            try:
                meta.update(json.loads(side.read_text(encoding="utf-8")))
            except Exception:                              # noqa: BLE001
                pass
        if meta["dur"] >= 3.0:
            lib[cat].append(meta)
    out = {k: v for k, v in lib.items() if v}
    try:
        idx_path.write_text(json.dumps({**out, "_sig": sig}, indent=1),
                            encoding="utf-8")
    except Exception:                                      # noqa: BLE001
        pass
    return _with_use_only(out, cfg)


def library_available(cfg=None) -> bool:
    return bool(scan(cfg))


def track_count(cfg=None) -> int:
    return sum(len(v) for v in scan(cfg).values())


# ====================================================================== #
#  Category mapping + selection (no-repeat)
# ====================================================================== #
def topic_category(title: str, blob: str) -> str | None:
    t = f"{title} {blob}".lower()
    for words, cat in _TOPIC:
        if any(w in t for w in words):
            return cat
    return None


def category_for(role: str, energy: int, *, style: str = "",
                 topic: str | None = None) -> str:
    """Choose a music CATEGORY for a scene from its arc role + energy +
    the documentary's style mode and topic. The topic biases the overall
    palette; role/energy shape the moment."""
    role = (role or "").strip().lower()
    e = max(1, min(5, energy or 3))
    # role drives the moment
    cat = _ROLE_CAT.get(role, "")
    if not cat:
        cat = "climax_build" if e >= 4 else ("ambient" if e <= 2
                                             else "suspense")
    # high-energy non-reveal beats lean tense
    if e >= 5 and cat in ("ambient", "neutral"):
        cat = "climax_build"
    # topic palette nudges neutral/ambient moments toward the film's world
    if topic and cat in ("ambient", "neutral", "suspense"):
        if role in ("context", "") or cat != "suspense":
            cat = topic
    # style-mode flavour
    sm = (style or "").lower()
    if sm == "epic" and cat in ("suspense", "ambient", "neutral"):
        cat = "historical_epic"
    if sm == "true_crime" and cat in ("ambient", "neutral"):
        cat = "dark_investigation"
    return cat if cat in CATEGORIES else "neutral"


# ---------------------------------------------------------------- #
# PER-SCENE narration-text overrides.  Cinematic editors don't
# pick music by role alone -- they read the SCENE'S MEANING and
# match it.  If the narration is about loss / grief / death we
# pull emotional_piano even if the role bucket said suspense.
# If it's about a secret / hidden / vanished we pull mystery
# even if the role said context.  Etc.
#
# Order matters: most specific (emotional / mystery) before
# broader (investigation / tension).  Each rule is a tuple of
# (category, trigger-substrings).  Only fires if the scene
# narration matches one of the substrings AND the role-derived
# category is in `_OVERRIDABLE_CATS`.
# ---------------------------------------------------------------- #
_OVERRIDABLE_CATS = frozenset((
    "suspense", "ambient", "neutral", "mystery", "context",
    "dark_investigation",
))
_SCENE_OVERRIDES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("emotional_piano", (
        "died", "dead", "death", "killed", "buried", "grave",
        "widow", "widower", "orphan", "alone",
        "loss", "lost everything", "tragedy", "tragic",
        "grief", "mourning", "funeral", "memorial",
        "last letter", "final words", "farewell", "goodbye",
        "never saw", "never returned", "never came back",
        "her son", "his son", "her daughter", "his daughter",
        "mother", "father", "child died",
    )),
    ("mystery", (
        "vanished", "disappeared", "without a trace",
        "no one knows", "nobody knew", "still unknown",
        "the question is", "what happened to",
        "the secret", "hidden away", "a mystery",
        "unexplained", "puzzle", "enigma",
    )),
    ("slow_reveal", (
        "discovered", "uncovered", "revealed",
        "turned out to be", "it turned out", "as it turns out",
        "we now know", "the truth was", "the real reason",
        "centuries later", "years later", "decades later",
    )),
    ("climax_build", (
        "exploded", "burst", "broke out", "erupted",
        "collapsed", "crashed", "stormed",
        "charged in", "moments before", "the moment",
        "everything changed", "tipping point",
    )),
    ("dark_investigation", (
        "covert", "classified", "intercepted", "intelligence",
        "spy", "spies", "espionage", "agent",
        "evidence", "interrogation", "surveillance",
        "wiretap", "double agent",
    )),
    ("military_tension", (
        "battalion", "brigade", "regiment", "platoon",
        "battlefield", "barricade", "trench", "front line",
        "the army", "the troops", "the soldiers",
    )),
    ("aftermath", (
        "in the years that followed", "in the decades since",
        "the aftermath", "the silence after", "what remained",
        "rebuilt", "moved on", "remembered",
    )),
)


def scene_text_override(text: str) -> str:
    """Return a category if the scene narration carries strong cues
    that should override the role-derived bucket. Empty string = no
    override (caller keeps the role-based pick)."""
    if not text:
        return ""
    t = " " + text.lower() + " "
    for cat, hits in _SCENE_OVERRIDES:
        for h in hits:
            if h in t:
                return cat
    return ""


def category_for_scene(role: str, energy: int, narration: str = "", *,
                       style: str = "", topic: str | None = None) -> str:
    """Like ``category_for`` but also reads the SCENE NARRATION for
    emotional / mystery / reveal cues. When the text override fires
    AND the role-derived category is in `_OVERRIDABLE_CATS`, the
    override wins. Otherwise role-derived category stands.

    P2-PLUS — CHANNEL CUE-BUILDER HARD-GATE:
      After the universal category/topic/override logic decides a
      category, if the active Look DNA has a `music.category_bias`
      list with `hard_filter: true`, we REMAP any out-of-bias
      decision to the closest in-bias category by energy.  This
      means Atlas literally cannot REQUEST `historical_epic` in the
      first place — the topic→category map is overridden by channel
      identity at the cue-builder stage, so the LOG line also reads
      true and `select()` doesn't have to fall through to neutral."""
    base = category_for(role, energy, style=style, topic=topic)
    override = scene_text_override(narration)
    if override == "emotional_piano":
        chosen = override
    elif override and base in _OVERRIDABLE_CATS:
        chosen = override
    else:
        chosen = base
    return _channel_remap_category(chosen, energy)


# Mapping from "energy bracket" to the role each bias slot should
# play within a channel.  When a channel declares
#   music.category_bias: [tech_cyber, financial, climax_build, neutral]
# the FIRST entry is treated as the channel's MID-energy default,
# the LAST as the LOW-energy floor, and `climax_build`-ish entries
# (if present) take the HIGH slot.  When no climax-style entry is
# present, the highest-energy-feeling category wins for energy 4-5.
_HIGH_ENERGY_FRIENDS = frozenset((
    "climax_build", "military_tension", "survival_urgency",
    "historical_epic", "tech_cyber",
))
_LOW_ENERGY_FRIENDS = frozenset((
    "neutral", "ambient", "emotional_piano", "aftermath",
    "slow_reveal", "archive_texture",
))


def _channel_remap_category(cat: str, energy: int) -> str:
    """Remap an out-of-bias category to the channel's closest in-bias
    category, biased by energy.  Hard-filter only — when no channel
    or no hard_filter, returns `cat` unchanged.

    Atlas (clinical):  high → climax_build, mid → tech_cyber,
                       low  → neutral
    Amber  (reverent): high → historical_epic, mid → slow_reveal,
                       low  → emotional_piano
    Midnight (tense):  high → suspense, mid → dark_investigation,
                       low  → ambient
    """
    try:
        bias, hard = _channel_music_bias()
    except Exception:                                       # noqa: BLE001
        return cat
    if not bias or not hard:
        return cat
    if cat in bias:
        return cat
    e = max(1, min(5, int(energy or 3)))
    # Pick the most-energy-appropriate bias entry.
    if e >= 4:
        for b in bias:
            if b in _HIGH_ENERGY_FRIENDS:
                return b
    if e <= 2:
        for b in bias:
            if b in _LOW_ENERGY_FRIENDS:
                return b
    # mid energy or no obvious match — first bias entry wins
    return bias[0]


def _channel_music_bias() -> tuple[list[str], bool]:
    """Read Look-DNA `music.category_bias` and `music.hard_filter`.

    Returns (bias, is_hard).
      • bias   — channel-allowed categories.  Empty when no channel
                 or no bias is set (legacy path is fully neutral).
      • is_hard — when True, candidates from non-bias categories are
                 REJECTED instead of merely deprioritised.  Default
                 True when a non-empty bias is set (the user's stated
                 P2 goal: "music identity must become channel-native").
    """
    try:
        from .look_dna import current as _ld_current, look_get
        if _ld_current() is None:
            return [], False
        bias = look_get("music.category_bias", []) or []
        if not isinstance(bias, list) or not bias:
            return [], False
        hard = bool(look_get("music.hard_filter",
                              True if bias else False))
        # normalise
        return [str(b).strip() for b in bias if str(b).strip()], hard
    except Exception:                                       # noqa: BLE001
        return [], False


def select(lib: dict, category: str, *, history: dict | None = None,
           seed: int = 0, cue: dict | None = None,
           usage: dict | None = None) -> dict | None:
    """Pick a track for ``category`` with NO-REPEAT rotation, walking the
    fallback chain when the category (and its relatives) are empty.

    When ``cue`` is supplied (Stage 3), candidates are SCORED against the
    scene context (energy, swell, role, duration) AND against cross-render
    usage from ``usage`` (Stage 4 -- least-used tracks win ties). The
    no-repeat ``history["_used"]`` cooldown still applies as a hard filter
    on top of scoring.

    P2 — CHANNEL HARD-GATING (`look.music.category_bias`):
      • If the active Look DNA declares a `music.category_bias` list,
        the fallback chain is REORDERED so the channel's preferred
        categories come first.
      • With `music.hard_filter: true` (default when bias is set),
        any category not on the bias list is FILTERED OUT entirely —
        Atlas can never accidentally play orchestral_swell, Amber can
        never play tech_cyber.  The chain still falls back through
        bias-only categories, then `neutral` as a last resort.
      • With no channel active, behaviour is byte-identical to legacy.
    """
    history = history if history is not None else {}
    chain = (category,) + _FALLBACK_CHAIN.get(category, ("neutral",))
    # ── P2 hard-gate ────────────────────────────────────────────────
    bias, hard = _channel_music_bias()
    if bias:
        # Reorder so bias categories come first, preserving the rest.
        kept_bias = [c for c in chain if c in bias]
        rest      = [c for c in chain if c not in bias]
        # Add any bias category the chain didn't mention (in order).
        for b in bias:
            if b not in kept_bias and b not in rest:
                kept_bias.append(b)
        if hard:
            # Hard filter — drop everything except bias categories,
            # plus a safety-net 'neutral' tail in case the bias pool
            # is empty (silent video is worse than slightly off-genre).
            chain = tuple(kept_bias) + ("neutral",)
        else:
            chain = tuple(kept_bias + rest)
    used = history.setdefault("_used", [])
    for cat in chain:
        pool = lib.get(cat) or []
        if not pool:
            continue
        avail = [t for t in pool if t["path"] not in used] or pool
        if cue is not None and len(avail) > 1:
            scored = sorted(
                avail,
                key=lambda t: (-_score_track(t, cue, usage or {}),
                               (usage or {}).get(t["path"], 0)),
            )
            pick = scored[0]
        else:
            pick = avail[seed % len(avail)]
        used.append(pick["path"])
        del used[:-6]                                  # cooldown window
        if usage is not None:
            usage[pick["path"]] = usage.get(pick["path"], 0) + 1
        return pick
    return None


# ====================================================================== #
#  Stage 3 -- per-track scoring  (editor-quality selection)
#  Stage 4 -- cross-render usage tracking (anti-repetition)
# ====================================================================== #
# Per-category preferences across the composite editorial scores
# (tension / darkness / orchestral_density / rhythmic_aggression).
# +ve weight = prefer high; -ve weight = prefer low.  Magnitudes are
# small (max ~0.6) because they're additive on top of the structural
# score components below; the goal is a NUDGE not a takeover.
_COMPOSITE_PREFS: dict[str, dict[str, float]] = {
    "emotional_piano":    {"orchestral_density": -0.5, "tension": -0.3,
                           "rhythmic_aggression": -0.6},
    "ambient":            {"orchestral_density": -0.4,
                           "rhythmic_aggression": -0.5,
                           "tension": -0.2},
    "aftermath":          {"orchestral_density": -0.4,
                           "rhythmic_aggression": -0.5,
                           "darkness": -0.2},
    "slow_reveal":        {"orchestral_density": -0.2, "tension": +0.2,
                           "darkness": +0.2},
    "mystery":            {"darkness": +0.4, "tension": +0.2,
                           "rhythmic_aggression": -0.3},
    "suspense":           {"tension": +0.5,
                           "rhythmic_aggression": +0.2,
                           "darkness": +0.3},
    "dark_investigation": {"darkness": +0.6, "tension": +0.3,
                           "orchestral_density": +0.1},
    "military_tension":   {"rhythmic_aggression": +0.5, "tension": +0.4,
                           "orchestral_density": +0.2},
    "climax_build":       {"rhythmic_aggression": +0.6,
                           "orchestral_density": +0.4,
                           "tension": +0.2},
    "historical_epic":    {"orchestral_density": +0.4, "tension": +0.2},
    "tech_cyber":         {"rhythmic_aggression": +0.3,
                           "orchestral_density": +0.2},
    "archive_texture":    {"orchestral_density": -0.3},
}


def _composite_fit(track: dict, cue: dict) -> float:
    """Score how well a track's composite features match the cue category.
    Returns roughly -0.2..+0.2 so it nudges but doesn't overwhelm the
    structural score components in :func:`_score_track`."""
    f = ((track.get("classify") or {}).get("features")) or {}
    cat = (cue.get("category") or "").lower()
    prefs = _COMPOSITE_PREFS.get(cat, {})
    if not prefs:
        return 0.0
    s = 0.0
    for key, weight in prefs.items():
        v = float(f.get(key, 0.5))    # 0.5 = neutral for missing features
        s += weight * (v - 0.5)        # delta from neutral
    return max(-0.25, min(0.25, s * 0.35))


def _score_track(track: dict, cue: dict, usage: dict) -> float:
    """Return a 0..1 fitness score for ``track`` against ``cue``.

    ``track`` is a sidecar-enriched meta dict from :func:`scan` -- which
    means tracks ingested via ``vidlore.music_extract`` carry
    ``classify.features`` (audio descriptors from
    :mod:`vidlore.music_classify`). Tracks without features still score
    against duration + cross-render usage so the new selector degrades
    cleanly on hand-curated libraries."""
    energy = int(cue.get("energy") or 3)
    swell = bool(cue.get("swell"))
    role = (cue.get("role") or "").lower()
    needed_dur = float(cue.get("end", 0) or 0) - float(cue.get("start", 0)
                                                      or 0)
    feats = (((track.get("classify") or {}).get("features")) or {})
    score = 0.5

    # --- energy fit
    is_dynamic = bool(feats.get("is_dynamic"))
    is_sparse = bool(feats.get("is_sparse"))
    if energy >= 4 and is_dynamic:
        score += 0.15
    if energy >= 4 and is_sparse:
        score -= 0.10
    if energy <= 2 and is_sparse:
        score += 0.15
    if energy <= 2 and is_dynamic:
        score -= 0.10

    # --- arc / swell fit
    arc = (feats.get("energy_arc") or "").lower()
    if swell and arc == "rising":
        score += 0.20
    if swell and arc == "falling":
        score -= 0.15
    if role in ("aftermath", "resolution", "payoff") and arc == "falling":
        score += 0.15
    if role in ("hook", "stakes", "problem", "climax") and arc == "rising":
        score += 0.10

    # --- brightness fit (dark cues want darker tracks)
    is_bright = bool(feats.get("is_bright"))
    cat = (cue.get("category") or "").lower()
    if cat in ("dark_investigation", "mystery", "suspense",
               "military_tension"):
        if not is_bright:
            score += 0.10
        else:
            score -= 0.10
    if cat in ("historical_epic", "tech_cyber") and is_bright:
        score += 0.05

    # --- duration fit: prefer tracks long enough to cover the cue without
    #     looping more than once
    tdur = float(track.get("dur") or 0.0)
    if needed_dur > 0 and tdur > 0:
        ratio = tdur / needed_dur
        if 0.9 <= ratio <= 2.5:
            score += 0.10
        elif ratio < 0.5:
            score -= 0.10                 # heavy looping = audible repeat

    # --- Stage 4: prefer least-used across renders
    plays = int(usage.get(track["path"], 0))
    if plays == 0:
        score += 0.15                     # fresh tracks lead
    else:
        # decay: each prior use costs ~0.04, capped at -0.20
        score -= min(0.20, plays * 0.04)

    # --- Editorial composite nudge (uses tension / darkness /
    #     orchestral_density / rhythmic_aggression from Stage 2 audio
    #     analysis -- only meaningful on tracks whose sidecars carry
    #     these new fields).
    score += _composite_fit(track, cue)

    return max(0.0, min(1.0, score))


# --- usage persistence (cross-render anti-repetition; Stage 4) ----------- #
def _usage_path(cfg=None) -> Path:
    return library_root(cfg) / "_usage.json"


def load_usage(cfg=None) -> dict[str, int]:
    """Cross-render usage counters {track_path: play_count}."""
    p = _usage_path(cfg)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in d.items()
                if isinstance(v, (int, float))}
    except Exception:                                      # noqa: BLE001
        return {}


def save_usage(usage: dict[str, int], cfg=None) -> None:
    if not usage:
        return
    try:
        _usage_path(cfg).write_text(
            json.dumps(usage, indent=1), encoding="utf-8")
    except Exception:                                      # noqa: BLE001
        pass


# ====================================================================== #
#  Cue planner — segment the film into music sections (act breaks)
# ====================================================================== #
def _wpm(text: str, dur: float) -> float:
    """Words (or pacing-units) per minute -- drives music density routing
    (fast narration -> quieter bed, slow -> fuller).

    Latin uses real word counts. CJK has no spaces so a sentence-long
    Japanese line would count as 1 word and give a WPM of ~12 -- which
    would push music to climax_build density.  We route through
    `lang.caption_pacing_hint` so CJK is counted in characters (the
    spoken-rate equivalent: ~600 chars/min ≈ 180 Latin wpm).  Same
    helper feeds the subtitle splitter so music + captions agree on
    pacing per language."""
    try:
        from . import lang as _lang
        units = _lang.caption_pacing_hint(text or "")
    except Exception:                                          # noqa: BLE001
        units = len((text or "").split())
    # Normalise: CJK characters are ~3-4x denser per second than Latin
    # words.  Convert to "Latin word-equivalents per minute" so the
    # density bands downstream (175, 195 WPM thresholds) keep working.
    try:
        from . import lang as _lang
        script = _lang.detect_script(text or "")
        if script in ("jp", "kr", "cjk"):
            units = units / 3.5                                # to word-equiv
    except Exception:                                          # noqa: BLE001
        pass
    return units * 60.0 / max(0.1, float(dur or 0.1))


# ====================================================================== #
#  MICRO-EDITORIAL REVEAL DETECTION  (premium-doc rewrite)
#  Real documentary editors don't just dip 1-2 dB on a reveal -- they
#  pull the music *out from under the voice* (often -8 to -12 dB) and
#  cut a tiny SILENCE POCKET right before the punch line so the word
#  lands in a vacuum.  We model this as a 3-TIER system:
#
#      tier 1 (soft pivot)   :  -3.5 dB dip,  0.7s wide
#                               -- gentle "get out of the way"
#      tier 2 (strong word)  :  -12 dB dip,   1.1s wide  +
#                               0.20s silence pocket BEFORE the word
#      tier 3 (climax/turn)  :  -22 dB dip,   1.4s wide  +
#                               0.35s silence pocket BEFORE the word
#                               -- guaranteed on every climax/reveal cue,
#                               so the major story beats ALWAYS feel
#                               dramatic.  Without a phrase match we fall
#                               back to the scene's emphasis word, then
#                               to the punch-line position.
#
#  Each event carries a RICH timing record (silence span + trapezoidal
#  dip span) so compose_score can build a properly-shaped envelope --
#  not a narrow triangular tent that the listener barely hears.
# ====================================================================== #

# Tier-1 soft pivots — gentle attenuation (gets music out of the way of
# a contrast/temporal cue without going silent).
_REVEAL_TIER1_WORDS: tuple[str, ...] = (
    "but", "however", "yet", "still", "though", "although",
    "then", "until", "before", "after", "meanwhile",
    "instead", "despite", "because",
)
# Tier-2 strong reveals — deeper dip + small silence pocket.  These are
# the moments a documentary editor would have manually pulled the music
# down on.
_REVEAL_TIER2_PHRASES: tuple[str, ...] = (
    "suddenly", "abruptly", "unexpectedly", "shocking", "shockingly",
    "the truth was", "the real reason", "what happened next",
    "it turned out", "nobody expected", "no one expected",
    "everything changed", "the secret was", "we now know",
    "in reality", "in the end", "finally", "the result was",
    "the answer was", "as it happens", "as it turns out",
    "and then", "and yet", "but then", "but here",
    "for the first time", "this time", "this is",
    "the answer", "the truth", "the truth is",
)
# Climax-only escalation words — when these appear inside a climax /
# reveal / turn cue, we promote them to tier 3 (the strongest ducking +
# longest silence pocket).
_REVEAL_TIER3_BOOSTERS: tuple[str, ...] = (
    "memory", "truth", "answer", "secret",
    "ninety", "ninety-five", "eighty", "seventy", "hundred",
    "always", "never", "everything", "nothing",
    "remembered", "forgotten", "alone", "hands",
    "decades", "century", "generations",
)
# Derived word/multi splits
_REVEAL_TIER1_SET: frozenset[str] = frozenset(_REVEAL_TIER1_WORDS)
_REVEAL_TIER2_WORDS_SET: frozenset[str] = frozenset(
    p for p in _REVEAL_TIER2_PHRASES if " " not in p)
_REVEAL_TIER2_MULTI: tuple[str, ...] = tuple(
    p for p in _REVEAL_TIER2_PHRASES if " " in p)
_REVEAL_TIER3_BOOSTERS_SET: frozenset[str] = frozenset(_REVEAL_TIER3_BOOSTERS)


def _channel_reveal_extras() -> tuple[frozenset, tuple, frozenset]:
    """Look-DNA channel-specific reveal phrases.

    Editors at different channels REVEAL DIFFERENT THINGS:
      • Atlas Explained — pulls music down on DATA reveals
        ("the data shows", "82 percent", "the study found").
      • Amber Chronicles — pulls down on EMOTIONAL reveals
        ("for the last time", "remembered", "vanished").
      • Midnight Pacific — pulls down on TENSION/EVIDENCE reveals
        ("the evidence shows", "no record", "concealed").

    Returns (extra_tier2_words_set, extra_tier2_multi_tuple,
             extra_tier3_boosters_set).  All empty when no channel.
    """
    try:
        from .look_dna import current as _ld_current, look_get
        if _ld_current() is None:
            return frozenset(), (), frozenset()
        t2 = look_get("reveal.tier2", []) or []
        t3 = look_get("reveal.tier3_boost", []) or []
        if not isinstance(t2, list):
            t2 = []
        if not isinstance(t3, list):
            t3 = []
        t2 = [str(p).strip().lower() for p in t2 if str(p).strip()]
        t3 = [str(p).strip().lower() for p in t3 if str(p).strip()]
        words = frozenset(p for p in t2 if " " not in p)
        multis = tuple(sorted({p for p in t2 if " " in p},
                                key=len, reverse=True))
        boosters = frozenset(t3)
        return words, multis, boosters
    except Exception:                                       # noqa: BLE001
        return frozenset(), (), frozenset()

# Tier -> (dip_magnitude, dip_total_hold, silence_pocket_dur)
#   dip_magnitude is the multiplier subtracted from the volume base.
#   With density ~1.0:
#       -0.35 -> 0.65 linear -> -3.7 dB attenuation
#       -0.78 -> 0.22 linear -> -13.2 dB attenuation
#       -0.92 -> 0.08 linear -> -22.0 dB attenuation
#   dip_hold is the FLOOR width (trapezoid hold) -- the time the music
#   stays at the dipped level.  Ramps in/out add ~0.10 + 0.20s on top.
#   silence_pocket_dur is the IDEAL silence span; the scanner shrinks
#   it to fit the actual gap before the trigger word (never overlaps
#   the previous word's tail).
_TIER_PROFILE: dict[int, tuple[float, float, float]] = {
    # (dip_mag 0..1, dip_hold_s, silence_pocket_pref_s).
    # Editorial pass: TRUE silence before a real reveal is the single
    # cheapest cinematic move in a documentary.  Bumped tier-2 and
    # tier-3 silence prefs so when the natural Whisper gap is long
    # enough we actually USE it — old 0.20/0.35 caps left the dip
    # noticeable but the silence imperceptible.  Tier-1 stays dip-only.
    1: (0.35, 0.55, 0.00),     # soft  -- mild dip only
    2: (0.82, 1.10, 0.55),     # strong -- big dip + breath of silence
    3: (0.95, 1.60, 1.10),     # peak   -- music gets out of the way
}


def _resolve_tier_profile() -> dict:
    """Look DNA override for the silence tier table.  When a channel
    is active, pulls ``look.silence.tier{1,2,3}_pocket_s`` and applies
    those to the silence_pref slot of the tier profile.  Falls back to
    the module constant when no look is set."""
    try:
        from .look_dna import look_get
        cur_t2 = look_get("silence.tier2_pocket_s")
        cur_t3 = look_get("silence.tier3_pocket_s")
        if cur_t2 is None and cur_t3 is None:
            return _TIER_PROFILE
        prof = dict(_TIER_PROFILE)
        if cur_t2 is not None:
            mag, hold, _ = prof[2]
            prof[2] = (mag, hold, float(cur_t2))
        if cur_t3 is not None:
            mag, hold, _ = prof[3]
            prof[3] = (mag, hold, float(cur_t3))
        return prof
    except Exception:                                              # noqa: BLE001
        return _TIER_PROFILE


def _normalize_word(w: str) -> str:
    return re.sub(r"[^\w']", "", (w or "").lower())


def _scan_reveal_events(words, cue_start: float, cue_end: float, *,
                        cue_role: str = "",
                        emphasis_words: tuple[str, ...] = (),
                        scene_spans: list[tuple[float, float]] | None = None,
                        peak_intensity: int = 0
                        ) -> list[dict]:
    """Find tiered reveal events inside this cue.

    Each event:
        {tier, type, time, dip_mag, dip_hold, silence_dur,
         silence_start, silence_end, dip_start, dip_end}
        ALL TIMES ARE CUE-RELATIVE (seconds from cue start).

    - tier 1: soft pivot              -> dip only
    - tier 2: strong reveal phrase    -> dip + 0.20s silence before
    - tier 3: climax/turn/reveal cue  -> dip + 0.35s silence before
              (guaranteed at least once per cue with this role, by
               falling back to emphasis word, then to the cue's
               punch-line position)
    """
    if not words:
        return []
    cw = [w for w in words
          if getattr(w, "start", -1) is not None
          and cue_start <= float(w.start) < cue_end]
    if not cw:
        return []
    norm = [_normalize_word(getattr(w, "word", "")) for w in cw]
    cue_len = max(0.5, cue_end - cue_start)

    events: list[dict] = []
    used_buckets: set[int] = set()   # 0.40s grid for dedup
    role = (cue_role or "").lower()
    is_peak_cue = role in ("climax", "reveal", "turn", "twist")
    # IMP_006 — the most powerful documentary moments (intensity-5 reveals /
    # climaxes / turns) get a GUARANTEED ~0.8s pre-silence pocket, not one
    # that's contingent on the narration happening to leave a gap. When this
    # floor is set we extend the tier-3 pocket (already allowed to over-reach
    # into a soft previous-word tail) up to 0.8s so "the room goes quiet"
    # before the reveal word lands every time it truly matters.
    _force_sil_min = 0.80 if (is_peak_cue and int(peak_intensity or 0) >= 5) else 0.0

    def _gap_before(i: int) -> float:
        """Seconds of silence between previous-word end and word i start.
        For i==0 of a cue, use the cue start as the prior boundary."""
        if i <= 0:
            return max(0.0, float(cw[0].start) - cue_start)
        try:
            prev_end = float(cw[i - 1].end)
            cur_start = float(cw[i].start)
            return max(0.0, cur_start - prev_end)
        except Exception:                                       # noqa: BLE001
            return 0.0

    def _emit(i: int, tier: int) -> bool:
        word_abs = float(cw[i].start)
        t_rel = word_abs - cue_start
        # Don't drop a dip too close to the cue boundaries (xfade region).
        if t_rel < 0.6 or t_rel > cue_len - 0.6:
            return False
        bucket = int(round(t_rel / 0.40))
        if bucket in used_buckets:
            return False

        dip_mag, dip_hold, sil_pref = _resolve_tier_profile()[tier]
        # PRE-ROLL: start the dip slightly before the word.  Bumped
        # tier-3 pre-roll from 0.30 → 0.55 so the music has fully
        # backed off by the time the silence pocket starts — a clean
        # "the room goes quiet" feel before the reveal word lands.
        pre = {1: 0.12, 2: 0.30, 3: 0.55}[tier]
        # Trapezoidal dip: ramp_in + hold + ramp_out.
        # ramp_in is short (snap into the dip), ramp_out is gentler so the
        # music breathes back up under the post-word tail.
        ramp_in = 0.08
        ramp_out = 0.22
        # Hold extends through the word + a little after.
        try:
            word_dur = max(0.20, float(cw[i].end) - float(cw[i].start))
        except Exception:                                       # noqa: BLE001
            word_dur = 0.30
        hold = max(dip_hold, word_dur + 0.50)
        dip_start = t_rel - pre
        dip_end = dip_start + ramp_in + hold + ramp_out

        # SILENCE POCKET: only when the natural gap before the word can
        # absorb it without cutting off the previous word's tail.
        sil_dur = 0.0
        sil_start = 0.0
        sil_end = 0.0
        if sil_pref > 0:
            gap = _gap_before(i)
            # Need at least ~120ms of natural gap to insert a pocket; the
            # pocket is bounded by what the gap can hold (-40ms safety on
            # either side).  Tier-3 can OVER-EXTEND the pocket by up to
            # +0.45s past the natural gap when the previous-word tail is
            # already a soft fade — for the climax we WANT the audience
            # to feel the room hold its breath, not just register a beat.
            usable = max(0.0, gap - 0.08)
            if tier == 3:
                usable += 0.45                    # stretch room for climax
                # IMP_006 — intensity-5 peak: guarantee the pocket reaches
                # ~0.8s (still bounded by the cue start downstream, so it
                # never runs past the scene boundary).
                if _force_sil_min > 0.0:
                    usable = max(usable, _force_sil_min)
            if usable >= 0.12:
                sil_dur = min(sil_pref, usable)
                # End the pocket ~30ms before the trigger word so silence
                # cuts cleanly into voice (no overlap, no tail truncation).
                sil_end = t_rel - 0.03
                sil_start = sil_end - sil_dur
                # If silence overlaps the dip ramp, advance dip_start to
                # right after silence so they don't fight.
                if dip_start < sil_end:
                    dip_start = sil_end
                    dip_end = dip_start + ramp_in + hold + ramp_out

        used_buckets.add(bucket)
        events.append({
            "tier": tier,
            "type": "dip",            # silence is now an attribute on dip
            "time": round(t_rel, 3),
            "word": cw[i].word if hasattr(cw[i], "word") else "",
            "dip_mag": dip_mag,
            "dip_hold": round(hold, 3),
            "dip_start": round(max(0.0, dip_start), 3),
            "dip_end": round(min(cue_len, dip_end), 3),
            "ramp_in": ramp_in,
            "ramp_out": ramp_out,
            "silence_dur": round(sil_dur, 3),
            "silence_start": round(max(0.0, sil_start), 3),
            "silence_end": round(max(0.0, sil_end), 3),
            "source": "phrase",
        })
        return True

    # P-reveal — channel reveal taste: Atlas fires on data words,
    # Amber on emotional words, Midnight on tension/evidence words.
    # Merged with the universal sets so universal triggers still work;
    # channel triggers ADD on top.
    _ch_t2_words, _ch_t2_multi, _ch_t3_boost = _channel_reveal_extras()
    _t2_words   = _REVEAL_TIER2_WORDS_SET   | _ch_t2_words
    _t2_multi   = tuple(sorted(
        set(_REVEAL_TIER2_MULTI) | set(_ch_t2_multi), key=len, reverse=True))
    _t3_boost   = _REVEAL_TIER3_BOOSTERS_SET | _ch_t3_boost

    # --------- 1. multi-word TIER 2 phrases (longest first) ------------ #
    for phrase in _t2_multi:
        toks = phrase.split()
        for i in range(len(norm) - len(toks) + 1):
            if all(norm[i + j] == toks[j] for j in range(len(toks))):
                # If we're in a climax cue AND any token is a tier-3
                # booster, promote to tier 3.
                tier = 2
                if is_peak_cue and any(
                        norm[i + j] in _t3_boost
                        for j in range(len(toks))):
                    tier = 3
                _emit(i, tier)

    # --------- 2. single-word TIER 2 words ----------------------------- #
    for i, w in enumerate(norm):
        if w in _t2_words:
            tier = 3 if (is_peak_cue and w in _t3_boost) else 2
            _emit(i, tier)

    # --------- 3. TIER 1 soft pivots ----------------------------------- #
    for i, w in enumerate(norm):
        if w in _REVEAL_TIER1_SET:
            _emit(i, 1)

    # --------- 4. CLIMAX-ROLE GUARANTEE -------------------------------- #
    #     A climax/reveal/turn cue MUST land at least one tier-3 event,
    #     even when no scripted phrase was found.  Fallback order:
    #         (a) the per-scene emphasis word inside this cue
    #         (b) a tier-3 booster word inside this cue
    #         (c) the cue's punch-line position (70-80% through)
    if is_peak_cue:
        has_t3 = any(e["tier"] == 3 for e in events)
        if not has_t3:
            placed = False
            ews = tuple(_normalize_word(w) for w in (emphasis_words or ()))
            ews = tuple(w for w in ews if w)
            # (a) scripted emphasis word — pick the LAST occurrence (the
            # speaker hits emphasis on the punch, not the setup)
            if ews:
                for i in range(len(norm) - 1, -1, -1):
                    if norm[i] in ews and _emit(i, 3):
                        events[-1]["source"] = "emphasis"
                        placed = True
                        break
            # (b) any tier-3 booster word — including channel additions
            if not placed:
                for i in range(len(norm) - 1, -1, -1):
                    if norm[i] in _t3_boost \
                            and _emit(i, 3):
                        events[-1]["source"] = "booster"
                        placed = True
                        break
            # (c) hard fallback: pick a word in the 65-80% range of the
            # cue (the punch-line zone) that has a real silence gap before
            # it (so we can plant a clean silence pocket).
            if not placed:
                target_lo = cue_start + 0.65 * cue_len
                target_hi = cue_start + 0.85 * cue_len
                best = -1
                best_gap = -1.0
                for i, w in enumerate(cw):
                    ws = float(getattr(w, "start", -1))
                    if ws < target_lo or ws > target_hi:
                        continue
                    g = _gap_before(i)
                    if g > best_gap:
                        best_gap = g
                        best = i
                if best < 0:                           # nothing in window
                    # last-resort: 75% mark, ignore gap
                    for i, w in enumerate(cw):
                        if float(getattr(w, "start", -1)) >= \
                                cue_start + 0.72 * cue_len:
                            best = i
                            break
                if best >= 0:
                    # Force placement even if too-close-to-edge; clamp
                    # later.  We bypass _emit's edge-guard by easing the
                    # rules for the guarantee.
                    saved_buckets = set(used_buckets)
                    if _emit(best, 3):
                        events[-1]["source"] = "fallback_punchline"
                    else:
                        # ease edge constraints and retry
                        used_buckets.clear()
                        used_buckets.update(saved_buckets)
                        t_rel = float(cw[best].start) - cue_start
                        t_rel = max(0.5, min(cue_len - 0.5, t_rel))
                        # synthesize an event by hand
                        dip_mag, dip_hold, sil_pref = _TIER_PROFILE[3]
                        ramp_in, ramp_out = 0.08, 0.22
                        hold = max(dip_hold, 1.0)
                        dip_start = max(0.0, t_rel - 0.30)
                        dip_end = min(cue_len, dip_start + ramp_in + hold
                                      + ramp_out)
                        gap = _gap_before(best)
                        sil_dur = 0.0
                        sil_start = sil_end = 0.0
                        if gap >= 0.20:
                            sil_dur = min(sil_pref, gap - 0.08)
                            sil_end = t_rel - 0.03
                            sil_start = sil_end - sil_dur
                            if dip_start < sil_end:
                                dip_start = sil_end
                                dip_end = dip_start + ramp_in + hold \
                                    + ramp_out
                        events.append({
                            "tier": 3, "type": "dip",
                            "time": round(t_rel, 3),
                            "word": cw[best].word if hasattr(cw[best],
                                                              "word") else "",
                            "dip_mag": dip_mag,
                            "dip_hold": round(hold, 3),
                            "dip_start": round(dip_start, 3),
                            "dip_end": round(dip_end, 3),
                            "ramp_in": ramp_in, "ramp_out": ramp_out,
                            "silence_dur": round(sil_dur, 3),
                            "silence_start": round(max(0.0, sil_start), 3),
                            "silence_end": round(max(0.0, sil_end), 3),
                            "source": "fallback_punchline_forced",
                        })

    events.sort(key=lambda e: e["time"])
    return events


def plan_cues(durs: list[float], roles: list[str], energies: list[int], *,
              texts: list[str] | None = None,
              narr_scenes: list | None = None,
              emphases: list[str] | None = None,
              style: str = "", topic: str | None = None,
              min_cue: float = 28.0) -> list[dict]:
    """Segment the timeline into music CUES. A new cue starts at a real
    music change — the category shifts AND we're past `min_cue` since the
    last change — so the score changes with the story (acts), not every
    scene.

    **Editorial enrichment** (when ``texts`` is supplied):

        * ``density`` (0..1) — how loud the music should sit under this cue.
          Fast narration (high WPM) and emotional cues use lower density so
          the score gets out of the way of the voice.
        * ``breath`` (bool) — mark a 1.5s dip near the cue's tail when the
          NEXT cue starts with a reveal/climax/emotional moment, so the
          audience subconsciously hears the moment coming.
        * ``drop`` (bool) — for very short cues that are clearly a hook /
          title moment, drop music entirely for the impact second.
        * ``swell`` (bool) — kept from the original implementation: cue
          leads into a reveal/climax → ramp up over the last 2.5s.
        * ``role`` carries through unchanged so the selector can score per
          scene context.

    Returns a list of ``{start, end, category, role, energy, density,
    breath, drop, swell, wpm}`` dicts.
    """
    n = len(durs)
    if n == 0:
        return []
    starts, acc = [], 0.0
    for d in durs:
        starts.append(acc)
        acc += d
    total = acc
    # Per-scene category routing -- the narration text now drives
    # category overrides too (emotional/mystery/reveal/etc) so an
    # emotional moment doesn't get suspense music just because its
    # role was "stakes".  Pre-flight cleared scenes (no text) still
    # fall back to the role-derived pick.
    cats = []
    for i in range(n):
        narr = (texts[i] if texts and i < len(texts) else "") or ""
        cats.append(category_for_scene(
            roles[i] if i < len(roles) else "",
            energies[i] if i < len(energies) else 3,
            narr, style=style, topic=topic))
    # per-scene WPM (0 if texts not supplied -- planner will skip the
    # narration-aware nudges)
    wpms = [_wpm(texts[i] if texts and i < len(texts) else "", durs[i])
            for i in range(n)]

    cues: list[dict] = []
    cur = {"start": 0.0, "category": cats[0],
           "role": (roles[0] if roles else ""),
           "energy": (energies[0] if energies else 3),
           "_scene_idxs": [0]}
    for i in range(1, n):
        change = (cats[i] != cur["category"]
                  and (starts[i] - cur["start"]) >= min_cue)
        if change:
            cur["end"] = starts[i]
            cues.append(cur)
            cur = {"start": starts[i], "category": cats[i],
                   "role": (roles[i] if i < len(roles) else ""),
                   "energy": (energies[i] if i < len(energies) else 3),
                   "_scene_idxs": [i]}
        else:
            cur["_scene_idxs"].append(i)
            # carry the most intense category within the cue
            if energies and i < len(energies) and energies[i] >= \
                    cur.get("energy", 3):
                cur["energy"] = energies[i]
    cur["end"] = total
    cues.append(cur)

    # Aggregate WPM + role signals per cue from its contributing scenes
    for cue in cues:
        # keep _scene_idxs accessible -- the reveal-event scanner needs
        # to look up per-scene EMPHASIS words for the climax guarantee.
        idxs = cue.get("_scene_idxs", [])
        if idxs and wpms:
            cue_wpm = sum(wpms[i] for i in idxs) / len(idxs)
        else:
            cue_wpm = 0.0
        cue["wpm"] = round(cue_wpm, 1)

        # Default density 1.0
        density = 1.0
        cat = cue["category"]
        role = (cue.get("role") or "").lower()
        # Fast narration -> get out of the way of the voice
        if cue_wpm and cue_wpm > 175:
            density = min(density, 0.55)
        # Emotional / aftermath / piano cues stay restrained always
        if cat in ("emotional_piano", "aftermath"):
            density = min(density, 0.65)
        if role in ("reaction", "resolution", "payoff", "aftermath"):
            density = min(density, 0.7)
        # context cues with very fast narration drop further
        if role == "context" and cue_wpm and cue_wpm > 195:
            density = min(density, 0.45)
        cue["density"] = round(density, 2)

    # Mark swells + breaths + drops looking at neighbour cues
    for k in range(len(cues) - 1):
        nxt = cues[k + 1]
        nxt_cat = nxt["category"]
        nxt_role = (nxt.get("role") or "").lower()
        # Swell INTO a reveal / climax / slow_reveal (existing behaviour)
        if nxt_cat in ("slow_reveal", "climax_build"):
            cues[k]["swell"] = True
        # Breath BEFORE a reveal / climax / emotional moment -- a tiny
        # 1.5s dip in the LAST 2.5s of this cue gives the audience a beat
        # to brace
        if (nxt_cat in ("slow_reveal", "climax_build", "emotional_piano")
                or nxt_role in ("reveal", "turn", "climax", "twist")):
            cues[k]["breath"] = True

    # Title-card / hook cues that are SHORT (<10s) -> hard drop the music
    # for the impact moment.  These are typically the cold-open snap.
    if cues and (cues[0].get("role") or "").lower() in ("hook", "tease"):
        if cues[0]["end"] - cues[0]["start"] <= 10.0:
            cues[0]["drop"] = True

    # ====== MICRO-EDITORIAL EVENTS (Stage 5 — premium-doc rewrite) ==== #
    # Use Whisper word-timings (when narr_scenes is supplied) to find
    # tiered reveal moments INSIDE each cue.  Each cue's role drives the
    # TIER promotion (climax/reveal/turn cues are guaranteed a tier-3
    # silence pocket + heavy dip), and per-scene EMPHASIS words feed the
    # fallback so even cues without a scripted reveal phrase land their
    # punch-line cleanly.
    if narr_scenes:
        all_words: list = []
        for ns in narr_scenes:
            for w in (getattr(ns, "words", None) or []):
                all_words.append(w)
        emph_list = list(emphases or [])
        for cue in cues:
            idxs = cue.get("_scene_idxs", []) or []
            # role for guarantee: prefer the cue's representative role, but
            # if ANY scene in this cue is climax/reveal/turn, treat the cue
            # as a peak cue too (so a context-tagged cue containing a
            # reveal scene still gets the dramatic ducking)
            cue_role = (cue.get("role") or "").lower()
            # IMP_006 — track the intensity of the PEAK scene driving this
            # cue so the silence pocket can be guaranteed on intensity-5
            # reveal/climax/turn moments.
            _peak_intensity = 0
            for si in idxs:
                if si < len(roles):
                    sr = (roles[si] or "").lower()
                    if sr in ("climax", "reveal", "turn", "twist"):
                        cue_role = sr
                        if si < len(energies):
                            _peak_intensity = max(
                                _peak_intensity, int(energies[si] or 0))
            # collect emphasis words from scenes in this cue
            emws: list[str] = []
            for si in idxs:
                if si < len(emph_list):
                    e = (emph_list[si] or "").strip()
                    if e:
                        emws.append(e)
            evs = _scan_reveal_events(
                all_words, cue["start"], cue["end"],
                cue_role=cue_role,
                emphasis_words=tuple(emws),
                peak_intensity=_peak_intensity,
            )
            cue["events"] = evs
    else:
        for cue in cues:
            cue.setdefault("events", [])

    # _scene_idxs is internal -- strip it now that all consumers are done.
    for cue in cues:
        cue.pop("_scene_idxs", None)

    return cues


# ====================================================================== #
#  Score composer — one edited music track for the whole film
# ====================================================================== #
_BED_LUFS = -25.0          # Netflix-doc spec: voice -16 LUFS, music
#                          # -25 LUFS = 9 dB voice/music ratio before
#                          # sidechain ducks further. Music sits as a
#                          # subconscious bed, never competes with VO.
#                          # (was -19, then -23, now -25 -- user feedback
#                          # "still too loud" after each step.)


def _pair_xfade(prev_cue: dict, next_cue: dict, default: float = 1.2) -> float:
    """Pick a crossfade duration for the boundary between two cues based on
    INTENSITY DELTA + category change (Stage 3 transition smoothing).

      * Same category, similar energy  -> short fade (~0.8s) -- a real
        editor would do a tight cross or even a hard cut here.
      * Going INTO climax_build / slow_reveal -> longer fade (~2.2s) so the
        upcoming track has room to swell underneath the outgoing one.
      * Going INTO aftermath -> longest fade (~2.8s) -- you want the
        outgoing energy to bleed away rather than slap-cut to calm.
      * Big energy drop (>=2 steps) -> long fade (~2.2s) so the drop reads
        as RELIEF, not a mistake.
      * Big energy rise (>=2 steps) into anything else -> medium fade (~1.6s).
    """
    pe = int(prev_cue.get("energy") or 3)
    ne = int(next_cue.get("energy") or 3)
    pc = (prev_cue.get("category") or "").lower()
    nc = (next_cue.get("category") or "").lower()
    if nc == "aftermath":
        return 2.8
    if nc in ("climax_build", "slow_reveal"):
        return 2.2
    delta = ne - pe
    if delta <= -2:                           # big drop
        return 2.2
    if delta >= 2:                            # big rise
        return 1.6
    if pc == nc and abs(delta) <= 1:          # same world, same energy
        return 0.8
    return default                            # 1.2 default


def compose_score(cues: list[dict], total: float, dest: Path, *,
                  cfg=None, cache_dir: Path | None = None,
                  xfade: float = 1.2, niche: str = "",
                  cue_sheet_out: "Path | None" = None,
                  video_id: str = "") -> Path | None:
    """Build ONE edited score wav: a LUFS-matched track per cue, crossfaded
    at the cue boundaries (act breaks), intro fade-in + outro fade-out, and
    a gentle swell on cues that lead into a reveal. Returns the wav path, or
    None when the library has no usable tracks (caller -> procedural bed).

    Stage 3 polish: each boundary's crossfade duration is chosen by
    :func:`_pair_xfade` -- intensity-aware, so a climax doesn't slap-cut
    into an aftermath and a same-category continuation doesn't slop into a
    sluggish 1.2s blur. """
    lib = scan(cfg)
    if not lib or not cues:
        return None
    total = max(1.0, round(float(total), 2))
    history: dict = {}
    # Stage 4: cross-render usage history so the same 10 tracks don't
    # dominate every video. select() decays score with prior plays.
    usage = load_usage(cfg)
    seg_specs = []
    for ci, cue in enumerate(cues):
        seglen = max(2.0, cue["end"] - cue["start"])
        tr = select(lib, cue["category"], history=history,
                    seed=int(cue["start"]) * 7 + ci,
                    cue=cue, usage=usage)
        if not tr:
            return None
        seg_specs.append((cue, tr, seglen))
    # persist updated usage so the NEXT render picks fresher tracks
    try:
        save_usage(usage, cfg)
    except Exception:                                      # noqa: BLE001
        pass

    # AUDIO DIRECTOR (Phase 4) — niche intro envelope + reveal-duck character +
    # chapter-level cue sheet. All defensive: any failure leaves the proven
    # score path byte-identical to before.
    _intro_expr = ""
    _revscale = 1.0
    try:
        from .audio_director import music_director as _md
        if niche:
            _intro_expr = _md.intro_volume_expr(niche)
            _revscale = float(_md.reveal_duck_scale(niche))
            if cue_sheet_out is not None:
                _sheet = _md.build_cue_sheet(seg_specs, total, niche=niche,
                                             video_id=video_id)
                _md.write_cue_sheet(_sheet, cue_sheet_out)
    except Exception:                                      # noqa: BLE001
        _intro_expr, _revscale = "", 1.0

    # Stage 3: per-boundary crossfade durations (intensity-aware).
    # xfades[i] = fade from seg[i] -> seg[i+1]; len(xfades) == N-1.
    xfades: list[float] = []
    for i in range(len(seg_specs) - 1):
        xfades.append(_pair_xfade(seg_specs[i][0], seg_specs[i + 1][0],
                                  default=xfade))

    # cache by the resolved (track, length, per-pair-xfade, editorial+events) plan
    def _evs_sig(evs):
        # tier + time + dip span + silence dur is enough to invalidate the
        # cache when the new tiered scanner changes the envelope shape.
        return "+".join(
            f"t{e.get('tier','?')}/{e.get('time',0):.1f}/"
            f"d{e.get('dip_mag', e.get('magnitude', 0)):.2f}/"
            f"h{e.get('dip_hold', e.get('dur', 0)):.2f}/"
            f"s{e.get('silence_dur', 0):.2f}"
            for e in (evs or []))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(
            ("score5|" + "|".join(   # bumped score4 -> score5 (new schema)
                f"{t['name']}:{round(sl, 1)}:s{c.get('swell', 0)}"
                f":d{c.get('density', 1.0):.2f}"
                f":b{c.get('breath', 0)}:p{c.get('drop', 0)}"
                f":w{c.get('wpm', 0):.0f}"
                f":e[{_evs_sig(c.get('events'))}]"
                for c, t, sl in seg_specs)
             + "|x:" + ",".join(f"{x:.2f}" for x in xfades)
             + f"|nd:{niche}:r{_revscale:.2f}:i{1 if _intro_expr else 0}"
             ).encode()).hexdigest()[:18]
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil
            shutil.copyfile(cached, dest)
            return dest

    # Render each cue segment to its own normalized wav, then crossfade-
    # concatenate. Each segment is (seglen + xfade_out) so the trailing
    # overlap consumed by the next acrossfade preserves total length.
    tmp = (cache_dir or dest.parent)
    seg_files: list[Path] = []
    try:
        for i, (cue, tr, seglen) in enumerate(seg_specs):
            x_out = xfades[i] if i < len(xfades) else 0.0
            ln = seglen + x_out
            sf = tmp / f"_score_seg_{i:03d}.wav"
            af = [
                f"loudnorm=I={_BED_LUFS}:TP=-1.5:LRA=11",
                "aresample=44100",
            ]

            # Editorial: per-cue volume DENSITY (1.0 = full, 0.4 = sit
            # under voice). Applied as a flat volume floor; swell/breath/
            # events layer on top via a time-varying expression.
            density = float(cue.get("density", 1.0))
            breath = bool(cue.get("breath"))
            drop = bool(cue.get("drop"))
            cue_events = cue.get("events") or []

            # Build a SINGLE time-varying volume expression — easier to
            # reason about than stacked filters.
            #
            #   base    = density
            #   swell   : +0.35 ramp over the last 2.5s
            #   breath  : -0.55 dip 1.5s wide before xfade overlap
            #   drop    : full silence in the last 1.0s (impact moment)
            #   events  : per-reveal-phrase dips + silence pockets
            #             (sourced from Whisper word timings)
            #
            # Each event contributes one term to the running sum -- ffmpeg
            # max(0,...) inside each tent naturally clips outside the
            # event window so terms don't interfere with each other.
            parts = [f"{density:.2f}"]
            if cue.get("swell") and ln > 4:
                # Reduced from +0.35 to +0.18 -- the user said music
                # was popping above voice during gap+swell moments.
                # 18% is still a perceptible lift into reveals but
                # cannot push music over the voice.
                sw = max(0.0, ln - 2.5)
                parts.append(f"+0.18*max(0\\,(t-{sw:.2f})/2.5)")
            if breath and ln > 4:
                # Tent dip near cue-end (was -0.55, now -0.40 to match
                # the lower swell magnitude -- dips still register
                # against the new quieter bed)
                b_end = max(0.0, ln - x_out - 0.4)
                b_mid = max(0.75, b_end - 0.75)
                parts.append(
                    f"-0.40*max(0\\,1-abs(t-{b_mid:.2f})/0.75)")

            # ---- Per-reveal-phrase events (tiered, premium-doc) ----- #
            # Two-part envelope per event:
            #   (1) optional SILENCE POCKET in the natural gap before the
            #       trigger word -- music drops to 0 for ~0.2-0.35s so the
            #       reveal word lands in a vacuum.
            #   (2) trapezoidal DIP that snaps in (~80ms), holds at the
            #       attenuated floor through the word + 0.5s tail, then
            #       eases back up (~220ms).  Tier 3 drops ~-22 dB, tier
            #       2 ~-13 dB, tier 1 ~-3.7 dB.
            #
            # Math model (each event contributes ADDITIVE terms to expr):
            #   silence  : -1.0 * between(t, s0, s1)
            #   dip(tri/trap): -mag * clip(min((t-ds)/ri, (de-t)/ro), 0, 1)
            #
            # clip(x,0,1) bounds the trapezoid: <0 outside the window, =1
            # in the hold region, ramps at the edges.  Multiple events do
            # not interfere because each is masked to its own window.
            for ev in cue_events:
                tier = int(ev.get("tier", 0))
                # New-schema fields with backwards-compat fallback to
                # old-schema (legacy 'type'/'magnitude'/'dur' from cached
                # plans).  Brand-new events always carry the new fields.
                if tier and "dip_start" in ev:
                    # niche reveal-duck character: a bounded per-niche scale on
                    # the dip depth (spy/crime deeper, history/business lighter).
                    mag = min(0.98, float(ev.get("dip_mag", 0.4)) * _revscale)
                    ds = float(ev.get("dip_start", 0.0))
                    de = float(ev.get("dip_end", ds + 1.0))
                    ri = max(0.04, float(ev.get("ramp_in", 0.08)))
                    ro = max(0.10, float(ev.get("ramp_out", 0.22)))
                    sil_dur = float(ev.get("silence_dur", 0.0))
                    sil_s = float(ev.get("silence_start", 0.0))
                    sil_e = float(ev.get("silence_end", 0.0))
                else:
                    # Legacy fallback (kept so old caches still play)
                    t = float(ev.get("time", 0.0))
                    if t < 0.5 or t > (ln - 0.6):
                        continue
                    kind = ev.get("type", "dip")
                    mag = float(ev.get("magnitude", 0.4))
                    half = float(ev.get("dur", 0.7))
                    centre = max(0.3, t - 0.20)
                    if kind == "silence":
                        s0 = centre
                        s1 = centre + max(0.3, half)
                        parts.append(
                            f"-1.0*(gte(t\\,{s0:.2f})*lte(t\\,{s1:.2f}))")
                    else:
                        parts.append(
                            f"-{mag:.2f}*max(0\\,1-abs(t-{centre:.2f})/"
                            f"{max(0.30,half):.2f})")
                    continue

                # Clamp to segment bounds (cue may extend slightly via the
                # xfade tail; we only protect the meaningful window)
                if ds < 0.3 or ds > ln - 0.3:
                    continue
                de = min(ln - 0.1, de)
                if de - ds < ri + ro + 0.05:
                    continue
                # SILENCE POCKET (only when silence_dur > 0 -- we already
                # confirmed it fits the natural voice gap upstream)
                if sil_dur > 0.05 and sil_s >= 0.1 and sil_e <= ln - 0.1 \
                        and sil_e > sil_s:
                    parts.append(
                        f"-1.0*between(t\\,{sil_s:.3f}\\,{sil_e:.3f})")
                # TRAPEZOIDAL DIP -- sustained floor for perceptual impact
                parts.append(
                    f"-{mag:.3f}*clip(min((t-{ds:.3f})/{ri:.3f}\\,"
                    f"({de:.3f}-t)/{ro:.3f})\\,0\\,1)")

            expr = "(" + "".join(parts) + ")"
            af.append(f"volume='{expr}':eval=frame")
            if drop and ln > 3:
                d_start = max(0.0, ln - 1.0)
                af.append(
                    f"volume='if(gte(t,{d_start:.2f}),0,1)':eval=frame")

            # intro fade-in on the very first cue; tail handled by xfade
            if i == 0:
                # NICHE INTRO INTELLIGENCE — a louder-then-recede envelope tuned
                # per niche (spy restrained pulse, business confident swell,
                # mystery silence-heavy). Multiplicative + bounded, so it shapes
                # only the opening and never overrides the proven mix/duck.
                if _intro_expr:
                    af.append(f"volume='{_intro_expr}':eval=frame")
                af.append("afade=t=in:d=1.6")
            if i == len(seg_specs) - 1:
                # v14.1: gentler outro tail (Vidlore AI fades ~-0.4 dB/s; our
                # 2 s fade was ~-0.9..-1.2). Stretch to ~3.5 s where the cue allows.
                _ofd = round(min(3.5, max(2.0, ln * 0.35)), 2)
                af.append(f"afade=t=out:st={max(0.0, ln - _ofd):.2f}:d={_ofd:.2f}")
            run([
                "-stream_loop", "-1", "-i", tr["path"],
                "-t", f"{ln:.2f}",
                "-af", ",".join(af),
                "-ac", "2", "-ar", "44100", str(sf),
            ])
            seg_files.append(sf)

        if len(seg_files) == 1:
            import shutil
            shutil.copyfile(seg_files[0], dest)
        else:
            inputs: list[str] = []
            for sf in seg_files:
                inputs += ["-i", sf.name]
            # chain acrossfade pairwise (label per step, per-pair duration)
            parts = []
            prev = "[0:a]"
            for i in range(1, len(seg_files)):
                out = f"[x{i}]"
                d = xfades[i - 1]
                parts.append(
                    f"{prev}[{i}:a]acrossfade=d={d:.2f}:c1=tri:c2=tri"
                    f"{out}")
                prev = out
            run(["-i", seg_files[0].name]
                + sum([["-i", s.name] for s in seg_files[1:]], [])
                + ["-filter_complex", ";".join(parts),
                   "-map", prev, "-ac", "2", "-ar", "44100",
                   str(dest.resolve())],
                cwd=str(tmp))
    except Exception as e:                                 # noqa: BLE001
        print(f"  [music] score compose failed ({e}); procedural fallback",
              flush=True)
        return None
    finally:
        for sf in seg_files:
            try:
                sf.unlink(missing_ok=True)
            except Exception:                              # noqa: BLE001
                pass

    if cache_dir is not None:
        import shutil
        try:
            shutil.copyfile(dest, cached)
        except Exception:                                  # noqa: BLE001
            pass
    return dest


def summary(cfg=None) -> dict:
    lib = scan(cfg)
    return {c: len(lib.get(c, [])) for c in CATEGORIES if lib.get(c)}
