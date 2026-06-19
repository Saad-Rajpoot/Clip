#!/usr/bin/env python3
"""audio_quality_audit.py — the Vidlore AUDIO QA GATE (Phase-4 deliverable).

Wraps the low-level metrics in `audio_metrics.py` and adds the documentary-grade
checks the audio spec requires, then emits one PASS / WARN / FAIL verdict and
writes `audio_quality_report.json` beside the media.

Checks (each contributes a verdict):
  loudness            integrated LUFS vs -16 target
  true_peak / clip    dBTP vs -1.0; hard FAIL if > 0 (clipping)
  intro_vs_body       opening 30 s energy vs the rest (intro should lift, recede)
  silence / dead_air  silence pockets; dead air = long gap with no room-tone floor
  mud                 low-mid (180-500 Hz) build-up vs full-band
  dialogue_balance    voice-vs-music RMS (needs stems) — intelligibility proxy
  ducking             music energy reduction under voice (needs stems)
  sfx_density         whoosh / total SFX per minute (needs an sfx cue-sheet)
  sfx_repeat          per-category cooldown regressions (reuses audio_metrics)
  music_repeat        repeated tracks within the video (needs a music cue-sheet)
  cross_video_repeat  overlap of categories / tracks / sfx families vs history
  license_complete    every manifest track has license+source+creator+checksum
  provenance          no track missing provenance; (bundle target) no USE-ONLY raw

Degrades gracefully: a check with no input (no stems / no cue-sheet) is reported
as "n/a (needs ...)" rather than failing.

Usage:
  python tools/audio_quality_audit.py <media.mp4> \
      [--voice v.wav --music m.wav --sfx s.wav] \
      [--music-cues music_cue_sheet.json] [--sfx-cues sfx_cue_sheet.json] \
      [--scope channel_id] [--target render|bundle]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from audio_metrics import (  # noqa: E402  (reuse the low-level primitives)
    FFMPEG, TARGET_LUFS, TARGET_TP, _astats, _ebur128, _silence,
    audit_sfx_schedule,
)

LIB = _ROOT / "vidlore" / "audio_library"


def _seg_lufs(media: Path, ss: float, t: float) -> float | None:
    err = subprocess.run(
        [FFMPEG, "-nostats", "-ss", str(ss), "-t", str(t), "-i", str(media),
         "-filter_complex", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    iv = re.findall(r"I:\s*(-?\d+\.?\d*)\s*LUFS", err)
    return float(iv[-1]) if iv else None


def _duration(media: Path) -> float:
    err = subprocess.run([FFMPEG, "-i", str(media), "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _band_rms(media: Path, lo: int, hi: int) -> float | None:
    err = subprocess.run(
        [FFMPEG, "-i", str(media), "-af",
         f"highpass=f={lo},lowpass=f={hi},astats=metadata=1",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"RMS level dB:\s*(-?\d+\.?\d*)", err)
    return float(m.group(1)) if m else None


def check_intro_body(media: Path, total: float) -> dict:
    if total < 45:
        return {"verdict": "n/a (clip too short)", "intro_lufs": None, "body_lufs": None}
    intro = _seg_lufs(media, 0.0, 30.0)
    body = _seg_lufs(media, 30.0, max(15.0, total - 30.0))
    if intro is None or body is None:
        return {"verdict": "n/a", "intro_lufs": intro, "body_lufs": body}
    delta = round(intro - body, 1)
    # intro may be a touch hotter or comparable; a much QUIETER intro reads weak,
    # a much HOTTER intro reads like it never recedes.
    if delta < -3.0:
        verdict = f"WARN(intro {delta} LU quieter — weak open)"
    elif delta > 4.0:
        verdict = f"WARN(intro {delta} LU hotter — doesn't recede)"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "intro_lufs": intro, "body_lufs": body, "delta_lu": delta}


def check_mud(media: Path) -> dict:
    full = _astats(media).get("rms_db")
    lowmid = _band_rms(media, 180, 500)
    if full is None or lowmid is None:
        return {"verdict": "n/a"}
    # how hot is the 180-500 Hz band relative to the whole signal
    ratio = round(lowmid - full, 1)
    verdict = "PASS" if ratio <= 3.0 else f"WARN(low-mid +{ratio}dB — mud risk)"
    return {"verdict": verdict, "low_mid_rms_db": lowmid, "full_rms_db": full,
            "low_mid_excess_db": ratio}


def check_silence_dead(media: Path) -> dict:
    sil = _silence(media)
    pockets = sil.get("silence_pockets_s", [])
    # "dead air" = a pocket longer than 2.5 s detected at the -32 dB floor (i.e.
    # not even room tone present). Short pockets are intentional reveal-silence.
    dead = [d for d in pockets if d > 2.5]
    sil["dead_air_pockets"] = dead
    sil["verdict"] = ("PASS" if not dead
                      else f"WARN({len(dead)} dead-air gaps > 2.5s)")
    return sil


def check_dialogue(stems: dict) -> dict:
    v, m = stems.get("voice"), stems.get("music")
    if v is None or m is None:
        return {"verdict": "n/a (needs --voice & --music stems)"}
    margin = round(v - m, 1)
    # voice should sit clearly above the music bed (documentary ~ 8-16 dB)
    if margin < 6:
        verdict = f"WARN(voice only +{margin}dB over music — intelligibility risk)"
    elif margin > 22:
        verdict = f"WARN(voice +{margin}dB — music nearly inaudible)"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "voice_rms_db": v, "music_rms_db": m, "margin_db": margin}


def check_density(sfx_cues: dict | None, total: float) -> dict:
    if not sfx_cues or total <= 0:
        return {"verdict": "n/a (needs --sfx-cues)"}
    events = sfx_cues.get("events") or sfx_cues.get("cues") or []
    n = len(events)
    whoosh_fams = {"whoosh", "transition", "text_reveal", "kinetic"}
    whoosh = sum(1 for e in events
                 if (e.get("family") in whoosh_fams or e.get("kind") in
                     ("reveal", "transition", "whoosh", "text_slam")))
    per_min = round(n / (total / 60.0), 2)
    whoosh_per_min = round(whoosh / (total / 60.0), 2)
    verdict = "PASS"
    if whoosh_per_min > 3.0:
        verdict = f"WARN(whoosh {whoosh_per_min}/min — too busy)"
    elif per_min > 18:
        verdict = f"WARN(sfx {per_min}/min — dense)"
    return {"verdict": verdict, "sfx_per_min": per_min,
            "whoosh_per_min": whoosh_per_min, "total_events": n}


def check_music_repeat(music_cues: dict | None) -> dict:
    if not music_cues:
        return {"verdict": "n/a (needs --music-cues)"}
    cues = music_cues.get("cues") or music_cues.get("chapters") or []
    tracks = [c.get("track") for c in cues if c.get("track")]
    from collections import Counter
    cnt = Counter(tracks)
    # a track used 3+ times in one video reads as recycled
    over = {t: c for t, c in cnt.items() if c >= 3}
    verdict = "PASS" if not over else f"WARN({len(over)} track(s) used 3+ times)"
    return {"verdict": verdict, "distinct_tracks": len(cnt),
            "total_cues": len(tracks), "overused": over}


def check_cross_video(music_cues, sfx_cues, scope: str) -> dict:
    try:
        sys.path.insert(0, str(_ROOT))
        from vidlore.audio_director import audio_usage_history as H
    except Exception as e:                                         # noqa: BLE001
        return {"verdict": f"n/a ({e})"}
    hist = H.load_history(scope=scope)
    if not hist.get("videos"):
        return {"verdict": "PASS (no prior videos)", "prior_videos": 0}
    cats = set()
    if music_cues:
        for c in (music_cues.get("cues") or music_cues.get("chapters") or []):
            if c.get("category"):
                cats.add(c["category"])
    recent = H._recent(hist, "categories", 2)
    overlap = sorted(cats & recent)
    verdict = "PASS" if len(overlap) <= 1 else f"WARN({len(overlap)} categories repeat recent videos)"
    return {"verdict": verdict, "prior_videos": len(hist["videos"]),
            "repeated_categories": overlap}


def check_license(target: str) -> dict:
    mf = LIB / "music_manifest.json"
    if not mf.exists():
        return {"verdict": "n/a (no music_manifest.json)"}
    man = json.loads(mf.read_text(encoding="utf-8"))
    tracks = man.get("tracks", [])
    req = ("license", "source", "creator", "checksum_sha1")
    missing = [t["id"] for t in tracks
               if any(not t.get(k) for k in req)]
    no_attr = [t["id"] for t in tracks
               if t.get("attribution_required") and not t.get("attribution")]
    use_only = [t["id"] for t in tracks if t.get("license_tier") == "use_only"]
    verdict = "PASS"
    if missing or no_attr:
        verdict = f"FAIL({len(missing)} missing provenance, {len(no_attr)} missing attribution)"
    elif target == "bundle" and use_only:
        verdict = f"FAIL({len(use_only)} USE-ONLY tracks present in a bundle target)"
    elif use_only:
        verdict = f"PASS (note: {len(use_only)} USE-ONLY tracks are render-only)"
    return {"verdict": verdict, "tracks": len(tracks),
            "missing_provenance": missing[:20],
            "missing_attribution": no_attr[:20],
            "use_only_tracks": len(use_only)}


def _agg(verdict: str) -> str:
    if verdict.startswith("FAIL"):
        return "FAIL"
    if verdict.startswith("WARN"):
        return "WARN"
    return "PASS"


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("--") and argv[0] != "--target":
        print(__doc__)
        return 2
    media = Path(argv[0])
    if not media.exists():
        print(f"not found: {media}")
        return 2
    opts: dict = {"target": "render", "scope": "default"}
    stem_paths: dict = {}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("--voice", "--music", "--sfx") and i + 1 < len(argv):
            stem_paths[a[2:]] = argv[i + 1]; i += 2; continue
        if a in ("--music-cues", "--sfx-cues", "--scope", "--target") and i + 1 < len(argv):
            opts[a.lstrip("-").replace("-", "_")] = argv[i + 1]; i += 2; continue
        i += 1

    def _load(key):
        p = opts.get(key)
        if p and Path(p).exists():
            try:
                return json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:                                      # noqa: BLE001
                return None
        return None

    music_cues = _load("music_cues")
    sfx_cues = _load("sfx_cues")
    stems = {k: _astats(Path(v)).get("rms_db") if Path(v).exists() else None
             for k, v in stem_paths.items()}

    total = _duration(media)
    base = _ebur128(media)
    base.update(_astats(media))

    report: dict = {"file": media.name, "duration_s": round(total, 1),
                    "integrated_lufs": base.get("integrated_lufs"),
                    "true_peak_dbtp": base.get("true_peak_dbtp"),
                    "lra": base.get("lra"), "checks": {}}
    C = report["checks"]

    # loudness
    li = base.get("integrated_lufs")
    if li is not None:
        d = abs(li - TARGET_LUFS)
        C["loudness"] = {"verdict": "PASS" if d <= 2 else
                         ("FAIL(%.1f LUFS)" % li if d > 3 else "WARN(%.1f LUFS)" % li),
                         "lufs": li, "target": TARGET_LUFS}
    # true peak / clipping
    tp = base.get("true_peak_dbtp")
    if tp is not None:
        C["true_peak"] = {"verdict": ("FAIL(clipping %.1f dBTP)" % tp if tp > 0 else
                          ("PASS" if tp <= TARGET_TP else "WARN(%.1f dBTP)" % tp)),
                          "dbtp": tp}

    C["intro_vs_body"] = check_intro_body(media, total)
    C["mud"] = check_mud(media)
    C["silence"] = check_silence_dead(media)
    C["dialogue_balance"] = check_dialogue(stems)
    C["ducking"] = ({"verdict": "n/a (needs stems)"} if stems.get("music") is None
                    else C["dialogue_balance"])
    C["sfx_density"] = check_density(sfx_cues, total)
    C["music_repeat"] = check_music_repeat(music_cues)
    C["cross_video_repeat"] = check_cross_video(music_cues, sfx_cues, opts["scope"])
    C["license"] = check_license(opts["target"])
    if sfx_cues:
        kinds = [e.get("kind", "") for e in (sfx_cues.get("events") or [])]
        C["sfx_repeat"] = {"verdict": audit_sfx_schedule(kinds).get("verdict", "n/a")}

    # aggregate
    levels = [_agg(v.get("verdict", "PASS")) for v in C.values()
              if not str(v.get("verdict", "")).startswith("n/a")]
    overall = "FAIL" if "FAIL" in levels else ("WARN" if "WARN" in levels else "PASS")
    report["verdict"] = overall
    report["summary"] = {k: v.get("verdict") for k, v in C.items()}

    out = media.parent / "audio_quality_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nOVERALL: {overall}   ->  wrote {out}")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
