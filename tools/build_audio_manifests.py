#!/usr/bin/env python3
"""Deterministically build the Vidlore audio-library manifests.

Reads ground truth (never guesses audio values):
  • vidlore/assets/music/<cat>/<track>.json  — real measured per-track features
  • vidlore/audio_library/category_semantics.json — editorial taxonomy (data-driven)
  • vidlore.sfx PRESETS / _POOL / EVENT_CATEGORY / EVENT_PREFER — synth SFX registry

Writes:
  • vidlore/audio_library/music_manifest.json — every track, full tag + license schema
  • vidlore/audio_library/sfx_manifest.json   — every synthesized preset + routing
  • vidlore/audio_library/dist_exclude.txt     — USE-ONLY raw files to keep out of dist/

License policy (enforced here, see AUDIO_SOURCE_LICENSE_MATRIX.md):
  incompetech / CC0 / CC-BY / synthesized originals -> BUNDLE-OK
  mixkit / pixabay / youtube_al                      -> USE-ONLY (no raw redistribution)

Run:  python3 tools/build_audio_manifests.py [--measure-lufs]
Idempotent: re-running reproduces the same manifests (modulo --measure-lufs cache).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "vidlore" / "assets" / "music"
LIB_DIR = ROOT / "vidlore" / "audio_library"
SEMANTICS = LIB_DIR / "category_semantics.json"
SCHEMA_VERSION = 1

# license source -> (tier, commercial_use, attribution_required, redistribution_allowed)
_LICENSE_TIER = {
    "incompetech":  ("bundle",   True,  True,  True),   # CC BY 4.0
    "cc0":          ("bundle",   True,  False, True),
    "freesound":    ("bundle",   True,  True,  True),   # per-asset CC0/CC-BY
    "synthesized":  ("bundle",   True,  False, True),   # we own it
    "mixkit":       ("use_only", True,  False, False),  # Mixkit Free License
    "pixabay":      ("use_only", True,  False, False),
    "youtube_al":   ("use_only", True,  False, False),
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, x)), 3)


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        return None


def _measure_lufs(mp3: Path) -> float | None:
    """Integrated LUFS via ffmpeg ebur128 (optional, slow ~ a few s/track)."""
    exe = _ffmpeg()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-nostats", "-i", str(mp3),
             "-filter_complex", "ebur128", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120).stderr
        val = None
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("I:") and "LUFS" in s:
                val = float(s.split()[1])
        return val
    except Exception:                                          # noqa: BLE001
        return None


def _loudness_to_energy(mean_db: float | None) -> float:
    if mean_db is None:
        return 0.5
    # -40 dBFS (quiet) .. -8 dBFS (hot) -> 0..1
    return _clamp((mean_db + 40.0) / 32.0)


def _loopability(silence_ratio: float, energy_arc: str, is_dynamic: bool) -> float:
    base = 1.0 - float(silence_ratio or 0.0)
    if (energy_arc or "") in ("rising", "falling"):
        base *= 0.7                                   # directional arc loops worse
    if is_dynamic:
        base *= 0.85                                  # big dynamics seam on loop
    return _clamp(base)


def _section_fit(cat_sem: dict, energy_arc: str) -> dict:
    """Per-track intro/body/climax/outro suitability = category base nudged by arc."""
    intro = cat_sem["intro_suitability"]
    body = cat_sem["body_suitability"]
    climax = cat_sem["climax_suitability"]
    outro = cat_sem["outro_suitability"]
    a = (energy_arc or "steady").lower()
    if a == "rising":
        climax += 0.12; intro -= 0.08; outro -= 0.10
    elif a == "falling":
        outro += 0.12; climax -= 0.08; intro += 0.04
    elif a in ("steady", "flat"):
        intro += 0.06; body += 0.06; climax -= 0.06
    return {"intro": _clamp(intro), "body": _clamp(body),
            "climax": _clamp(climax), "outro": _clamp(outro)}


_DUCK_FADES = {  # ducking_profile -> (fade_in_s, fade_out_s)
    "gentle":     (1.8, 2.6),
    "standard":   (1.4, 2.0),
    "aggressive": (1.0, 1.6),
}


def build_music_manifest(measure_lufs: bool = False) -> dict:
    sem = json.loads(SEMANTICS.read_text(encoding="utf-8"))["categories"]
    # carry forward already-measured LUFS so re-runs don't re-measure
    prev = {}
    out_path = LIB_DIR / "music_manifest.json"
    if out_path.exists():
        try:
            for t in json.loads(out_path.read_text(encoding="utf-8")).get("tracks", []):
                if t.get("lufs") is not None:
                    prev[t["id"]] = t["lufs"]
        except Exception:                                      # noqa: BLE001
            pass

    tracks: list[dict] = []
    per_cat: dict[str, int] = {}
    use_only: list[str] = []
    for sidecar in sorted(MUSIC_DIR.rglob("*.json")):
        if sidecar.name.startswith("_"):
            continue
        try:
            d = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        mp3 = sidecar.with_suffix(".mp3")
        if not mp3.exists():
            continue
        cat = d.get("category") or d.get("classify", {}).get("category") or "neutral"
        cat_sem = sem.get(cat, sem.get("neutral"))
        feats = (d.get("classify") or {}).get("features", {}) or {}
        src = (d.get("source") or "synthesized").lower()
        tier, commercial, attr_req, redistrib = _LICENSE_TIER.get(
            src, ("use_only", True, False, False))
        rel = str(mp3.relative_to(ROOT))
        tid = f"{cat}/{mp3.stem}"
        if tier == "use_only":
            use_only.append(rel)

        mean_db = feats.get("mean_volume_db")
        lufs = prev.get(tid)
        if measure_lufs and lufs is None:
            lufs = _measure_lufs(mp3)

        arc = feats.get("energy_arc", "steady")
        fit = _section_fit(cat_sem, arc)
        fi, fo = _DUCK_FADES.get(cat_sem["ducking_profile"], (1.4, 2.0))
        cinematic = feats.get("orchestral_density")
        cinematic = _clamp(cinematic) if cinematic is not None else cat_sem["cinematic"]

        tracks.append({
            "id": tid,
            "path": rel,
            "title": d.get("title") or mp3.stem.replace("-", " ").title(),
            "category": cat,
            "source": d.get("source"),
            "creator": d.get("channel"),
            "license": d.get("license"),
            "attribution": d.get("attribution"),
            "license_tier": tier,
            "commercial_use": commercial,
            "attribution_required": attr_req,
            "redistribution_allowed": redistrib,
            "bundle_ok": tier == "bundle",
            "download_date": d.get("fetched"),
            "duration": round(float(feats.get("duration", d.get("dur", 0.0)) or 0.0), 2),
            "bpm": feats.get("bpm"),
            "key": None,                       # honest: no pitch detection in extractor
            "loudness_dbfs": mean_db,
            "lufs": lufs,
            "peak_dbfs": feats.get("max_volume_db"),
            "mood": cat_sem["mood"],
            "tags": d.get("tags", []),
            "energy": _loudness_to_energy(mean_db) if mean_db is not None else cat_sem["energy"],
            "tension": _clamp(feats["tension"]) if "tension" in feats else cat_sem["tension"],
            "darkness": _clamp(feats["darkness"]) if "darkness" in feats else cat_sem["darkness"],
            "cinematic": cinematic,
            "loopability": _loopability(feats.get("silence_ratio", 0.0), arc,
                                        bool(feats.get("is_dynamic"))),
            "energy_arc": arc,
            "intro_suitability": fit["intro"],
            "body_suitability": fit["body"],
            "climax_suitability": fit["climax"],
            "outro_suitability": fit["outro"],
            "niches": cat_sem["niches"],
            "ducking_profile": cat_sem["ducking_profile"],
            "fade_in_s": fi,
            "fade_out_s": fo,
            "reuse_gap_videos": cat_sem["reuse_gap_videos"],
            "checksum_sha1": _sha1(mp3),
        })
        per_cat[cat] = per_cat.get(cat, 0) + 1

    bundle_n = sum(1 for t in tracks if t["bundle_ok"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "music",
        "built": date.today().isoformat(),
        "total_tracks": len(tracks),
        "bundle_ok_tracks": bundle_n,
        "use_only_tracks": len(tracks) - bundle_n,
        "categories_total": len(sem),
        "categories_populated": sorted(per_cat),
        "per_category_counts": dict(sorted(per_cat.items())),
        "license_policy": "BUNDLE-OK = redistributable raw file; USE-ONLY = render-time only, excluded from dist (see dist_exclude.txt).",
        "tracks": sorted(tracks, key=lambda t: t["id"]),
    }, use_only


def build_sfx_manifest() -> dict:
    sys.path.insert(0, str(ROOT))
    from vidlore import sfx  # noqa: E402

    _FAMILY = {  # pool -> dominant family (for restraint grouping)
        "text_reveal": "whoosh", "transition": "whoosh", "kinetic": "whoosh",
        "impact": "impact", "map": "map", "ui": "ui", "document": "foley_doc",
        "timeline": "data", "process": "data", "data": "data",
        "surveillance": "surveillance",
    }
    pools = {}
    for pool, ids in sfx._POOL.items():
        pools[pool] = {
            "count": len(ids),
            "family": _FAMILY.get(pool, pool),
            "preset_ids": list(ids),
        }
    presets = {}
    for pid, spec in sfx.PRESETS.items():
        pool = spec.get("cat", "")
        presets[pid] = {
            "pool": pool,
            "family": _FAMILY.get(pool, pool),
            "duration_s": round(float(spec.get("dur", 0.0)), 3),
            "base_volume": spec.get("vol"),
            "synthesis": "ffmpeg-lavfi",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sfx",
        "built": date.today().isoformat(),
        "license_tier": "bundle",
        "license": {
            "source": "synthesized_original",
            "creator": "Vidlore",
            "commercial_use": True,
            "attribution_required": False,
            "redistribution_allowed": True,
            "note": "100% procedurally synthesized (no third-party samples); Vidlore owns all SFX -> always bundle-safe.",
        },
        "synthesis": "ffmpeg lavfi (sine + colored noise + filter chains + envelopes + seeded jitter)",
        "total_presets": len(presets),
        "pool_count": len(pools),
        "pools": pools,
        "presets": dict(sorted(presets.items())),
        "event_routing": {
            "event_to_category": dict(sorted(sfx.EVENT_CATEGORY.items())),
            "event_preferred_preset": dict(sorted(sfx.EVENT_PREFER.items())),
        },
    }


def main(argv: list[str]) -> int:
    measure = "--measure-lufs" in argv
    LIB_DIR.mkdir(parents=True, exist_ok=True)

    music, use_only = build_music_manifest(measure_lufs=measure)
    (LIB_DIR / "music_manifest.json").write_text(
        json.dumps(music, indent=1, ensure_ascii=False), encoding="utf-8")

    sfx_m = build_sfx_manifest()
    (LIB_DIR / "sfx_manifest.json").write_text(
        json.dumps(sfx_m, indent=1, ensure_ascii=False), encoding="utf-8")

    (LIB_DIR / "dist_exclude.txt").write_text(
        "# USE-ONLY raw audio files — keep these OUT of any distributable build.\n"
        "# Licensed for use IN rendered videos, NOT for standalone redistribution.\n"
        "# (Mixkit Free License / Pixabay / YouTube Audio Library.) Auto-generated.\n"
        + "\n".join(sorted(use_only)) + "\n", encoding="utf-8")

    print(f"music_manifest.json : {music['total_tracks']} tracks "
          f"({music['bundle_ok_tracks']} bundle-ok, {music['use_only_tracks']} use-only)"
          f"{' + LUFS measured' if measure else ''}")
    print(f"  categories: {len(music['categories_populated'])} populated / "
          f"{music['categories_total']} defined")
    print(f"sfx_manifest.json   : {sfx_m['total_presets']} presets / "
          f"{sfx_m['pool_count']} pools (all bundle-safe synthesized)")
    print(f"dist_exclude.txt    : {len(use_only)} use-only raw files flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
