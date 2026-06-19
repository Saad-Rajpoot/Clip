"""Music director — chapter-level soundtrack intelligence (engine-level, reusable).

Sits ON TOP of the mature `musiclib` core (plan_cues / compose_score / select),
which is preserved unchanged. This module adds the spec's missing layers:

  • NICHE-AWARE INTRO INTELLIGENCE — the first 20-35 s gets a louder-then-recede
    envelope tuned per niche (spy = restrained pulse; business = confident swell;
    mystery = silence-heavy). Returns a *relative* gain envelope so it shapes the
    opening without touching the proven absolute mix / sidechain duck.
  • CHAPTER-LEVEL CUE SHEET — emits `music_cue_sheet.json` from the real selected
    tracks (category, track, start/end, editorial flags, reveal tiers, intro/outro).
  • CROSS-VIDEO ANTI-REPETITION — reorders the category fallback chain so a channel
    doesn't lead consecutive videos on the same category (via audio_usage_history).
  • PER-NICHE MIX CHARACTER — bounded bed/atmos/reveal-duck multipliers the mixer
    applies on top of the empirically-tuned base duck (never replaces it).

All editorial values are DATA-DRIVEN (intro_profiles.json / niche_mix.json /
category_semantics.json next to this file), authored + adversarially verified by
the audio-engine design workflow, so they tune without code changes. Every public
function is defensive: a missing/broken config returns a safe neutral default so a
render never breaks.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIB = _HERE.parent / "audio_library"

# bounded safety rails — the director can never push the opening hot enough to
# bury the voice, nor the per-niche character beyond a gentle tilt.
_INTRO_START_MAX = 2.8          # first-cue opening gain ceiling (× body bed)
_INTRO_RECEDE_MIN = 4.0         # never recede faster than this (s)
_INTRO_RECEDE_MAX = 30.0        # v14: allow the lift to HOLD across the hook
#                                 (~20-30 s) then settle, not collapse at 7 s.
_BED_MULT_LO, _BED_MULT_HI = 0.7, 1.3
_ATMOS_MULT_LO, _ATMOS_MULT_HI = 0.6, 1.2
_REVEAL_SCALE_LO, _REVEAL_SCALE_HI = 0.8, 1.4

_BODY_BED = 0.16                # matches assemble._MUSIC_VOL_DUCK rest level

# neutral fallbacks (used when config missing or niche unknown) — these reproduce
# today's behaviour exactly (no intro shaping, no character tilt).
# v14: the DEFAULT now carries a real intro lift too, so EVERY video (not only
# the recognised niches) opens stronger then settles. Bounded + ducked = safe.
# _DEFAULT_INTRO is the UNRECOGNISED / un-profiled-niche fallback. It MUST be a
# byte-identical NO-OP — intro_level == recede_to == body bed => start_mult ==
# end_mult == 1.0 => empty intro_volume_expr — so an unknown/default niche never
# gets an unvalidated intro swell. (2026-06-04 stabilization: intro_level 0.27 /
# recede_to 0.17 had accidentally made the default a 1.688× / 26 s treatment,
# contradicting the "neutral no-op intro" intent; restored to no-op. The stronger
# v14 hold-then-recede stays on the named, validated niche profiles only.)
_DEFAULT_INTRO = {"niche": "default", "duration_s": 26, "intro_level": _BODY_BED,
                  "recede_to": _BODY_BED, "recede_at_s": 26.0, "silence_bias": 0.4,
                  "category_preference": [], "opening_texture": "neutral"}
_DEFAULT_MIX = {"niche": "default", "duck_threshold": 0.030, "duck_ratio": 12,
                "duck_attack_ms": 8, "duck_release_ms": 300, "music_bed_mult": 1.0,
                "atmos_mult": 1.0, "reveal_duck_scale": 1.0}

_cache: dict = {}


def _load(name: str, base: Path) -> dict:
    key = str(base / name)
    if key in _cache:
        return _cache[key]
    try:
        d = json.loads((base / name).read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        d = {}
    _cache[key] = d
    return d


_CANON = ("spy", "crime", "business", "history", "geopolitics", "mystery")
_ALIASES = {"true_crime": "crime", "truecrime": "crime", "mossad": "spy",
            "intelligence": "spy", "espionage": "spy", "war": "geopolitics",
            "geopolitical": "geopolitics", "finance": "business",
            "economics": "business", "biography": "history",
            # brief THEME tokens -> a profiled niche (so they get a real intro
            # envelope + a fitting music character, not just the default)
            "modern": "business", "minimalist": "mystery", "standard": "history"}


def _strict_niche(niche: str) -> str:
    """Canonical niche if the input is RECOGNISED (a canonical name, a known
    alias, or the palette engine maps it); else "" — so callers can fall back to
    a byte-identical neutral default instead of guessing."""
    n = (niche or "").strip().lower()
    if n in _CANON:
        return n
    if n in _ALIASES:
        return _ALIASES[n]
    try:
        from .. import niche_palette
        p = niche_palette.normalize_niche(niche or "")
        if p in _CANON:
            return p
    except Exception:                                          # noqa: BLE001
        pass
    return ""


def normalize_niche(niche: str) -> str:
    """Canonicalize to one of {spy,crime,business,history,geopolitics,mystery}.
    Always returns a definite niche (history is the neutral fallback) — use for
    category routing where a concrete niche is required."""
    return _strict_niche(niche) or "history"


def _clamp(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


# ── intro intelligence ────────────────────────────────────────────────────────
def intro_profile(niche: str) -> dict:
    nn = _strict_niche(niche)
    if not nn:                       # unrecognised niche -> neutral no-op intro
        return dict(_DEFAULT_INTRO)
    cfg = _load("intro_profiles.json", _HERE)
    for p in cfg.get("profiles", []):
        if p.get("niche") == nn:
            return p
    return dict(_DEFAULT_INTRO)


def intro_relative_envelope(niche: str, body_level: float = _BODY_BED) -> dict:
    """Relative first-cue gain envelope: opens at `start_mult` × body bed, recedes
    to `end_mult` × body bed over `recede_s`. Multiplicative so it shapes only the
    opening and leaves the proven absolute mix + sidechain duck intact. Bounded so
    it can never open hot enough to mask narration."""
    p = intro_profile(niche)
    body = max(0.05, float(body_level or _BODY_BED))
    start = _clamp((p.get("intro_level", body) or body) / body, 0.8, _INTRO_START_MAX)
    end = _clamp((p.get("recede_to", body) or body) / body, 0.7, 1.15)
    recede = _clamp(p.get("recede_at_s", 6.0), _INTRO_RECEDE_MIN, _INTRO_RECEDE_MAX)
    return {"start_mult": round(start, 3), "end_mult": round(end, 3),
            "recede_s": round(recede, 2), "silence_bias": p.get("silence_bias", 0.4),
            "texture": p.get("opening_texture", "")}


def intro_volume_expr(niche: str, body_level: float = _BODY_BED) -> str:
    """ffmpeg `volume=` expression (eval=frame) for the FIRST music segment.

    v14: a HOLD-then-RECEDE so the intro music actually FEELS stronger for the
    whole hook (~the first 12-18 s) and then settles smoothly into the body
    bed — instead of the old linear collapse that was back at body level by ~7 s
    (measured: intro = body, so the 'stronger intro' was inaudible). The hold
    keeps the opening energetic and cinematic; the gentle taper is the smooth
    transition into the body music. Still bounded so it never masks narration
    (it's the same relative envelope on the proven absolute mix + sidechain duck)."""
    e = intro_relative_envelope(niche, body_level)
    s, d, r = e["start_mult"], e["end_mult"], e["recede_s"]
    # no meaningful lift -> unity (byte-identical no-op for un-profiled niches)
    if abs(s - 1.0) < 0.05 and abs(d - 1.0) < 0.05:
        return ""
    hold = round(min(r * 0.55, 18.0), 2)          # hold the lift through the hook
    rend = round(max(hold + 6.0, r), 2)           # fully settled into body by here
    span = rend - hold
    return (f"if(lt(t,{hold}),{s:.3f},"
            f"if(lt(t,{rend}),{s:.3f}+({d:.3f}-{s:.3f})*((t-{hold})/{span:.2f}),"
            f"{d:.3f}))")


# ── per-niche mix character (bounded multipliers, never replaces base duck) ────
def niche_mix(niche: str) -> dict:
    nn = _strict_niche(niche)
    if not nn:                       # unrecognised niche -> proven base values
        return dict(_DEFAULT_MIX)
    cfg = _load("niche_mix.json", _HERE)
    prof = None
    for p in cfg.get("profiles", []):
        if p.get("niche") == nn:
            prof = p
            break
    if not prof:
        return dict(_DEFAULT_MIX)
    return {
        "niche": nn,
        "duck_threshold": prof.get("duck_threshold", _DEFAULT_MIX["duck_threshold"]),
        "duck_ratio": prof.get("duck_ratio", _DEFAULT_MIX["duck_ratio"]),
        "duck_attack_ms": prof.get("duck_attack_ms", _DEFAULT_MIX["duck_attack_ms"]),
        "duck_release_ms": prof.get("duck_release_ms", _DEFAULT_MIX["duck_release_ms"]),
        "music_bed_mult": _clamp(prof.get("music_bed_mult", 1.0), _BED_MULT_LO, _BED_MULT_HI),
        "atmos_mult": _clamp(prof.get("atmos_mult", 1.0), _ATMOS_MULT_LO, _ATMOS_MULT_HI),
        "reveal_duck_scale": _clamp(prof.get("reveal_duck_scale", 1.0),
                                    _REVEAL_SCALE_LO, _REVEAL_SCALE_HI),
    }


def bed_mult(niche: str) -> float:
    return niche_mix(niche)["music_bed_mult"]


def atmos_mult(niche: str) -> float:
    return niche_mix(niche)["atmos_mult"]


def reveal_duck_scale(niche: str) -> float:
    return niche_mix(niche)["reveal_duck_scale"]


# ── cross-video anti-repetition ───────────────────────────────────────────────
def bias_category_chain(chain: list[str], niche: str = "", scope: str = "default",
                        cfg=None) -> list[str]:
    """Reorder a category fallback chain so categories used by the last couple of
    videos sink to the back (never dropped). Keeps a channel from leading two
    consecutive videos on the same music category."""
    if not chain:
        return chain
    try:
        from . import audio_usage_history as H
        hist = H.load_history(cfg, scope=scope)
        return H.filter_categories(hist, list(chain), gap=2)
    except Exception:                                          # noqa: BLE001
        return chain


# ── cue sheet emission (real selected tracks) ─────────────────────────────────
def build_cue_sheet(seg_specs: list, total: float, niche: str = "",
                    video_id: str = "") -> dict:
    """seg_specs: list of (cue_dict, track_dict, seglen) from compose_score. Build
    a chapter-level music cue sheet describing the actual edited score."""
    cues_out = []
    used_cats: list[str] = []
    used_tracks: list[str] = []
    attributions: list[dict] = []
    lead_cat = ""
    n = len(seg_specs)
    for i, item in enumerate(seg_specs):
        try:
            cue, tr, seglen = item
        except Exception:                                      # noqa: BLE001
            continue
        tr = tr or {}
        cat = cue.get("category", "")
        name = tr.get("name") or tr.get("title") or ""
        used_cats.append(cat)
        if name:
            used_tracks.append(f"{cat}/{name}")
        # capture attribution for any track that requires credit (CC-BY / YTAL)
        if tr.get("attribution_required") and tr.get("attribution"):
            cred = {"track": tr.get("title") or name, "artist": tr.get("artist", ""),
                    "license": tr.get("license") or tr.get("license_type", ""),
                    "source": tr.get("source", ""),
                    "source_url": tr.get("source_url", ""),
                    "attribution_text": tr.get("attribution", "")}
            if cred not in attributions:
                attributions.append(cred)
        if i == 0:
            lead_cat = cat
        evs = cue.get("events") or []
        cues_out.append({
            "index": i,
            "role": cue.get("role", ""),
            "category": cat,
            "track": name,
            "start_s": round(float(cue.get("start", 0.0)), 2),
            "end_s": round(float(cue.get("end", 0.0)), 2),
            "duration_s": round(float(seglen), 2),
            "is_intro": i == 0,
            "is_outro": i == n - 1,
            "swell": bool(cue.get("swell")),
            "breath": bool(cue.get("breath")),
            "drop": bool(cue.get("drop")),
            "density": round(float(cue.get("density", 1.0)), 2),
            "reveal_tiers": {
                "tier1": sum(1 for e in evs if e.get("tier") == 1),
                "tier2": sum(1 for e in evs if e.get("tier") == 2),
                "tier3": sum(1 for e in evs if e.get("tier") == 3),
            },
        })
    nn = normalize_niche(niche)
    env = intro_relative_envelope(nn)
    return {
        "schema_version": 1,
        "kind": "music_cue_sheet",
        "video_id": video_id,
        "niche": nn,
        "total_s": round(float(total), 2),
        "chapters": len(cues_out),
        "lead_category": lead_cat,
        "categories_used": sorted(set(used_cats)),
        "tracks_used": sorted(set(used_tracks)),
        "intro_profile": {"texture": env["texture"], "start_mult": env["start_mult"],
                          "recede_s": env["recede_s"], "silence_bias": env["silence_bias"]},
        "attributions": attributions,
        "cues": cues_out,
    }


def generate_attribution_file(sheet: dict, dest) -> bool:
    """Write a per-video CREDITS file listing the exact attribution text for every
    selected track that requires credit (CC-BY / YTAL). Returns True if any credit
    was written. Call this whenever a render uses attribution-required music."""
    creds = sheet.get("attributions") or []
    if not creds:
        return False
    lines = ["MUSIC ATTRIBUTION — required credits for this video",
             "(paste into the video description)", "=" * 52, ""]
    for c in creds:
        lines.append(c.get("attribution_text", "").strip())
        meta = " · ".join(x for x in [c.get("artist", ""), c.get("license", ""),
                                      c.get("source", "")] if x)
        if meta:
            lines.append(f"   [{meta}]")
        if c.get("source_url"):
            lines.append(f"   {c['source_url']}")
        lines.append("")
    try:
        Path(dest).write_text("\n".join(lines), encoding="utf-8")
        return True
    except Exception:                                          # noqa: BLE001
        return False


def write_cue_sheet(sheet: dict, dest) -> None:
    try:
        Path(dest).write_text(json.dumps(sheet, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass


def record_history(sheet: dict, *, scope: str = "default", date: str = "",
                   sfx_families=None, cfg=None) -> None:
    """Record this video's audio fingerprint for cross-video anti-repetition."""
    try:
        from . import audio_usage_history as H
        hist = H.load_history(cfg, scope=scope)
        H.record_video(hist, video_id=sheet.get("video_id", ""),
                       niche=sheet.get("niche", ""),
                       categories=sheet.get("categories_used", []),
                       tracks=sheet.get("tracks_used", []),
                       sfx_families=sfx_families or [],
                       lead_category=sheet.get("lead_category", ""), date=date)
        H.save_history(hist, cfg, scope=scope)
    except Exception:                                          # noqa: BLE001
        pass
