#!/usr/bin/env python3
"""Classify + quality-filter + organize ingested YTAL music into the library.

Reads the flat YTAL cache (vidlore/audio_library/ytal_cache/music/<id>.mp3 + .json
from tools/ingest_ytal.py), then for each track:
  • classifies into a musiclib CATEGORY (music_classify.classify_title + features),
  • tags the user's 17 documentary categories + niche compatibility,
  • computes mood + intro/body/climax/outro suitability,
  • scores documentary QUALITY (music_quality.doc_quality) and REJECTS weak filler,
  • dedups (checksum + near-title), and
  • MOVES the mp3 into ytal_cache/music/<category>/<id>.mp3 with a musiclib-format
    sidecar (so musiclib._with_use_only() can SELECT it at render time) carrying the
    FULL provenance + license metadata.

Writes vidlore/audio_library/ytal_music_manifest.json (USE-ONLY tier; raw files
never bundled in dist). Idempotent + re-runnable as more tracks are ingested.

Usage:  python3 tools/merge_ytal_music.py [--min-quality 0.0] [--min-dur 45]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "vidlore" / "audio_library" / "ytal_cache" / "music"

# musiclib category -> (mood, applicable user-17 tags, default role suitabilities)
_CAT_PROFILE = {
    "suspense":           ("taut watchful dread", ["suspense", "spy", "true_crime", "dark_investigation", "geopolitics", "body_bed"], (0.6, 0.9, 0.7, 0.3)),
    "mystery":            ("shadowed intrigue", ["suspense", "true_crime", "spy", "atmospheric", "intro_energy", "body_bed"], (0.8, 0.85, 0.5, 0.5)),
    "dark_investigation": ("grim forensic descent", ["dark_investigation", "true_crime", "spy", "suspense", "geopolitics", "body_bed", "reveal"], (0.5, 0.9, 0.7, 0.35)),
    "emotional_piano":    ("tender human gravity", ["emotional", "reflective", "history", "outro"], (0.6, 0.55, 0.25, 0.95)),
    "ambient":            ("weightless neutral air", ["atmospheric", "intro_energy", "body_bed", "history", "geopolitics"], (0.9, 0.85, 0.2, 0.65)),
    "historical_epic":    ("sweeping grandeur", ["history", "geopolitics", "business", "wealth", "climax", "reveal"], (0.6, 0.75, 0.85, 0.6)),
    "military_tension":   ("martial pressure", ["war_tension", "spy", "geopolitics", "history", "body_bed", "climax"], (0.5, 0.85, 0.8, 0.3)),
    "tech_cyber":         ("cold digital precision", ["spy", "business", "geopolitics", "atmospheric", "body_bed"], (0.6, 0.8, 0.6, 0.35)),
    "financial":          ("calculated momentum", ["business", "wealth", "geopolitics", "body_bed"], (0.6, 0.8, 0.65, 0.45)),
    "survival_urgency":   ("driving crisis", ["war_tension", "geopolitics", "true_crime", "climax"], (0.45, 0.8, 0.9, 0.3)),
    "slow_reveal":        ("patient unfolding", ["suspense", "true_crime", "spy", "reveal", "intro_energy", "body_bed"], (0.7, 0.75, 0.6, 0.55)),
    "climax_build":       ("rising to the peak", ["climax", "reveal", "war_tension", "geopolitics"], (0.3, 0.7, 0.95, 0.3)),
    "aftermath":          ("settled fallout", ["reflective", "emotional", "history", "outro"], (0.5, 0.65, 0.3, 0.9)),
    "neutral":            ("transparent bed", ["body_bed", "business", "history"], (0.6, 0.75, 0.4, 0.5)),
    "archive_texture":    ("aged film grain", ["atmospheric", "history", "true_crime", "intro_energy"], (0.7, 0.7, 0.25, 0.6)),
}
_DEFAULT = ("documentary underscore", ["body_bed"], (0.6, 0.75, 0.5, 0.5))

# Official YouTube Audio Library mood -> documentary niche augmentation.
_YTAL_MOOD_NICHE = {
    "dramatic":      ["suspense", "climax", "reveal", "dark_investigation"],
    "dark":          ["dark_investigation", "true_crime", "spy", "suspense"],
    "angry":         ["war_tension", "climax", "geopolitics"],
    "sad":           ["emotional", "reflective", "history", "outro"],
    "calm":          ["atmospheric", "reflective", "body_bed", "outro"],
    "inspirational": ["reflective", "history", "business", "intro_energy"],
    "happy":         ["intro_energy", "business"],
    "bright":        ["intro_energy", "business"],
    "funky":         ["intro_energy"],
    "romantic":      ["emotional", "reflective", "outro"],
    "sentimental":   ["emotional", "reflective", "outro"],
    "epic":          ["climax", "reveal", "history", "geopolitics"],
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:48] or "track"


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-quality", type=float, default=-1.0,
                    help="reject doc_quality score below this (default off)")
    ap.add_argument("--min-dur", type=float, default=45.0)
    ap.add_argument("--max-dur", type=float, default=600.0,
                    help="reject over-long tracks (films/explainers, not cues)")
    a = ap.parse_args(argv)
    _NARRATED = re.compile(
        r"full movie|explained|episode|podcast|interview|\bvlog\b|hijack|"
        r"documentary film|drone film|\b4k\b|case study|movie explanation|"
        r"\(19\d\d\)|\(20\d\d\)|short film", re.I)

    from vidlore import music_classify as mc
    try:
        from vidlore import music_quality as mq
    except Exception:                                              # noqa: BLE001
        mq = None

    sidecars = [p for p in CACHE.glob("*.json")] if CACHE.exists() else []
    # also re-scan already-organised category folders so re-runs are idempotent
    organised = [p for p in CACHE.rglob("*/*.json")] if CACHE.exists() else []
    seen_titles: set[str] = set()
    out_tracks, rejected, moved = [], [], 0

    def _process(sc: Path, already_organised: bool):
        nonlocal moved
        try:
            d = json.loads(sc.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return
        mp3 = sc.with_suffix(".mp3")
        if not mp3.exists():
            return
        title = d.get("title", "")
        dur = float(d.get("measured_duration") or d.get("duration") or 0)
        tkey = _slug(title)[:32]
        # quality + dedup gates
        blob = f"{title} {' '.join(d.get('tags', []))}"
        reasons = []
        if dur and dur < a.min_dur:
            reasons.append(f"too short ({dur:.0f}s)")
        if dur and dur > a.max_dur:
            reasons.append(f"too long ({dur:.0f}s) — film/explainer, not a cue")
        if _NARRATED.search(title):
            reasons.append("narrated/non-music content")
        # LICENSE GATE (non-negotiable): a CC-BY track without its exact credit
        # text is incomplete and must NOT be used.
        if d.get("attribution_required") and not (d.get("attribution")
                                                  or d.get("attribution_text")):
            reasons.append("CC-BY attribution text missing (incomplete license)")
        dq_score = None
        if mq is not None:
            try:
                dq = mq.doc_quality(blob)
                dq_score = round(dq.score, 3)
                if mq.is_reject(blob) or (a.min_quality > -1 and dq.score < a.min_quality):
                    reasons.append(f"doc_quality {dq.verdict}")
            except Exception:                                      # noqa: BLE001
                pass
        if tkey in seen_titles:
            reasons.append("duplicate title")
        if reasons:
            rejected.append({"title": title, "reasons": reasons})
            # Remove ANY rejected track — flat-staged or already-organised — so the
            # USE-ONLY tier never keeps dup/low-quality leftovers loose in the
            # cache root (where they'd leak in as a bogus "music" category).
            try:
                mp3.unlink(missing_ok=True)
                sc.unlink(missing_ok=True)
            except Exception:                                      # noqa: BLE001
                pass
            return
        seen_titles.add(tkey)
        cat = mc.classify_title(title)
        mood, tags17, roles = _CAT_PROFILE.get(cat, _DEFAULT)
        # Blend in the REAL YouTube Audio Library mood/genre (captured live from
        # the Studio UI). The official YTAL mood taxonomy maps cleanly onto the
        # documentary niches, so selection is no longer limited to title-keyword
        # classification. (No-op for the older CC-BY tracks, which lack these.)
        ymood = (d.get("mood_ytal") or "").strip()
        ygenre = (d.get("genre_ytal") or "").strip()
        if ymood:
            tags17 = sorted(set(list(tags17) + _YTAL_MOOD_NICHE.get(ymood.lower(), [])))
            mood = f"{mood} · {ymood.lower()}"
        arc = d.get("energy_arc", "steady")
        intro, body, climax, outro = roles
        if arc == "rising":
            climax += 0.12; outro -= 0.10
        elif arc == "falling":
            outro += 0.12; climax -= 0.08
        rec = {
            "id": d.get("id"), "name": d.get("id"), "title": title,
            "category": cat, "artist": d.get("artist"),
            "source": "youtube_al", "source_url": d.get("source_url"),
            "license": d.get("license_type"),
            "license_type": d.get("license_type"),
            "attribution_required": d.get("attribution_required"),
            # read both keys (ingest stores attribution_text; recover/merge store
            # attribution) so the credit survives repeated merge cycles.
            "attribution": d.get("attribution") or d.get("attribution_text") or "",
            "attribution_text": d.get("attribution") or d.get("attribution_text") or "",
            "commercial_use": d.get("commercial_use"),
            "download_date": d.get("download_date"),
            "checksum_sha1": d.get("checksum_sha1"),
            "license_tier": "use_only", "use_only": True,
            "dur": round(dur, 2),
            "duration": round(dur, 2),
            "lufs": d.get("lufs"), "peak_dbfs": d.get("peak_dbfs"),
            "bpm": d.get("bpm"),
            "tension": d.get("tension"), "darkness": d.get("darkness"),
            "energy_arc": arc, "silence_ratio": d.get("silence_ratio"),
            "mood": mood, "niches17": tags17,
            "intro_suitability": round(max(0, min(1, intro)), 2),
            "body_suitability": round(max(0, min(1, body)), 2),
            "climax_suitability": round(max(0, min(1, climax)), 2),
            "outro_suitability": round(max(0, min(1, outro)), 2),
            "quality_score": dq_score,
            "duplicate_status": "unique",
            "genre_ytal": ygenre, "mood_ytal": ymood,
            "tags": d.get("tags", []),
        }
        # organise into category folder (musiclib reads folder=category)
        dest_dir = CACHE / cat
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_mp3 = dest_dir / f"{d.get('id')}.mp3"
        if not already_organised:
            try:
                shutil.move(str(mp3), str(dest_mp3))
                sc.unlink(missing_ok=True)
                moved += 1
            except Exception:                                      # noqa: BLE001
                dest_mp3 = mp3
        else:
            dest_mp3 = mp3
        rec["path"] = str(dest_mp3.relative_to(ROOT))
        dest_mp3.with_suffix(".json").write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        out_tracks.append(rec)

    for sc in organised:
        _process(sc, already_organised=True)
    for sc in sidecars:
        _process(sc, already_organised=False)

    # manifest
    by_cat: dict = {}
    by_lic: dict = {}
    for t in out_tracks:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        lt = "attribution_required" if t["attribution_required"] else "no_attribution"
        by_lic[lt] = by_lic.get(lt, 0) + 1
    man = {
        "schema_version": 1, "kind": "ytal_music",
        "distribution_tier": "USE_ONLY (CC-BY/official YTAL; raw files git-ignored, excluded from dist)",
        "total": len(out_tracks),
        "per_category": dict(sorted(by_cat.items())),
        "by_attribution": by_lic,
        "rejected": len(rejected),
        "tracks": sorted(out_tracks, key=lambda t: (t["category"], t["id"] or "")),
        "rejected_sample": rejected[:30],
    }
    (ROOT / "vidlore" / "audio_library" / "ytal_music_manifest.json").write_text(
        json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[merge] organised {len(out_tracks)} YTAL music tracks "
          f"({moved} moved) · per-category {man['per_category']}")
    print(f"[merge] attribution: {by_lic} · rejected {len(rejected)}")
    print(f"[merge] manifest -> vidlore/audio_library/ytal_music_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
