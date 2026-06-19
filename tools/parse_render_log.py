#!/usr/bin/env python3
"""Render log -> structured metrics JSON (MNT_7). Pure local, no API/render.

Parses a Vidlore render log into a structured `render_metrics.json` (saved
beside the render): scenes, beats, shot-length, card types, footage source
breakdown, transitions, black-frame repairs, music cues, audio loudness,
stage/render times, and warnings/errors. Supplements from render_meta.json +
script.json + benchmark_scores.json when found beside the render.

Usage:
    python3 tools/parse_render_log.py <log_file_or_render_dir> [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _f(pat, text, grp=1, cast=float):
    m = re.search(pat, text)
    if m:
        try:
            return cast(m.group(grp))
        except (ValueError, TypeError):
            pass
    return None


def parse_log(text: str) -> dict:
    m = {}
    m["scenes"] = _f(r"\[script\]\s+(\d+)\s+scenes", text, 1, int)
    m["script_words"] = _f(r"\[script\][^\n]*?~?(\d+)\s+words", text, 1, int)
    m["beats"] = _f(r"(\d+)\s+beats from\s+\d+\s+scenes", text, 1, int)
    sl = re.search(r"shot len\s+([\d.]+)-([\d.]+)s", text)
    if sl:
        m["shot_len_min_s"] = float(sl.group(1))
        m["shot_len_max_s"] = float(sl.group(2))
    m["narration_audio_s"] = _f(r"narration\s+([\d.]+)s audio", text)
    m["nim_niche"] = (re.search(r"\[NIM\]\s+active niche:\s+([\w/]+)", text)
                      or [None, None])[1] if re.search(
        r"\[NIM\]\s+active niche:", text) else None
    ai = re.search(r"ai usage:\s+(\d+)/(\d+)", text)
    if ai:
        m["ai_images_used"] = int(ai.group(1))
        m["ai_images_cap"] = int(ai.group(2))
    m["graphic_card_images"] = _f(r"graphic art:\s+(\d+)\s+card image", text, 1, int)
    tr = re.search(r"transitions:\s+(\d+)\s+motivated\s*\(([^)]*)\)", text)
    if tr:
        m["transitions_motivated"] = int(tr.group(1))
        m["transitions_detail"] = tr.group(2).strip()
    # black-frame repair: detected then cleaned
    det = re.findall(r"black-frame repair:\s+(\d+)\s+span\(s\) detected", text)
    cln = re.search(r"black-frame repair:\s+clean\s+\((\d+)\s+black spans\)", text)
    m["black_spans_detected"] = sum(int(x) for x in det) if det else 0
    m["black_spans_after_repair"] = int(cln.group(1)) if cln else None
    m["music_cues"] = _f(r"scored music:\s+(\d+)\s+cues", text, 1, int)
    m["footage_wall_s"] = _f(r"footage wall\s+([\d.]+)s", text)
    m["tts_wall_s"] = _f(r"TTS wall\s+([\d.]+)s", text)
    m["assembly_wall_s"] = _f(r"assembly wall\s+([\d.]+)s", text)
    m["wallclock_s"] = (_f(r"wallclock:\s*([\d.]+)s", text)
                        or _f(r"DONE in\s+([\d.]+)s", text))
    # source breakdown (log-marker hit counts)
    m["sources"] = {
        "pexels_stock": len(re.findall(r"\bpexels\b", text, re.I)),
        "ai_fal": len(re.findall(r"\bfal\b|flux|symbolic_concept", text, re.I)),
        "web_footage_selected": _f(r"web-footage\] decisions:\s+(\d+)\s+selected",
                                   text, 1, int) or 0,
        "web_image_selected": _f(r"web-image\] decisions:\s+(\d+)\s+selected",
                                 text, 1, int) or 0,
        "archive_org": len(re.findall(r"archive\.org", text)),
    }
    # warnings / errors (deduped lines)
    flags = []
    for ln in text.splitlines():
        if re.search(r"(?i)\b(error|traceback|exception|failed|warning|"
                     r"black frame|skipped)\b", ln):
            s = ln.strip()
            if s and s not in flags:
                flags.append(s[:160])
    m["warnings_errors"] = flags[:40]
    m["warning_error_count"] = len(flags)
    return m


# QA thresholds mirror tools/benchmark_engine.py so the verdict and the
# benchmark can never disagree (target -16 LUFS, peak <0 dBFS, 3-6s avg shot).
_LUFS_TARGET = -16.0
_LUFS_WARN_DELTA = 3.0       # benchmark drops below 8.0 past this
_LUFS_FAIL_DELTA = 6.0       # benchmark drops below 6.0 past this
_SHOT_AVG_LO, _SHOT_AVG_HI = 2.5, 8.0
# benchmark text-restraint scores 10 at <=35%, -2pts/10pp over; 0.50 == score 7.0
# (the "real concern" line). Healthy renders (~0.39) score ~8.8 and don't warn.
_GFX_DENSITY_WARN = 0.50     # distinct graphic scenes / scenes
_DUR_WARN, _DUR_FAIL = 0.15, 0.30


def qa_verdict(metrics: dict, target_minutes: float | None = None) -> dict:
    """Turn parsed metrics into an actionable PASS/WARN/FAIL with reasons.

    Aligned to the benchmark's constants so an auto-run can flag real defects
    (residual black frames, clipping, off-target loudness/duration, graphic
    clutter, hard errors) without a human reading the log. No API/render.
    """
    fails, warns = [], []

    # ── hard defects ────────────────────────────────────────────────────
    if metrics.get("scenes") is None:
        fails.append("no scene count parsed (log truncated or render aborted)")
    bsr = metrics.get("black_spans_after_repair")
    if isinstance(bsr, int) and bsr > 0:
        fails.append(f"{bsr} black span(s) REMAIN after repair")
    pk = metrics.get("audio", {}).get("true_peak_db")
    if isinstance(pk, (int, float)) and pk >= 0.0:
        fails.append(f"audio clips (true peak {pk:+.1f} dBFS >= 0)")
    # hard errors (tracebacks/exceptions) vs benign fallback 'failed' lines
    hard = [w for w in metrics.get("warnings_errors", [])
            if re.search(r"traceback|exception", w, re.I)]
    if hard:
        fails.append(f"{len(hard)} hard error line(s) (traceback/exception)")

    # ── loudness (target -16 LUFS) ──────────────────────────────────────
    lufs = metrics.get("audio", {}).get("lufs")
    if isinstance(lufs, (int, float)):
        d = abs(lufs - _LUFS_TARGET)
        if d > _LUFS_FAIL_DELTA:
            fails.append(f"loudness {lufs:.1f} LUFS off target by {d:.1f}")
        elif d > _LUFS_WARN_DELTA:
            warns.append(f"loudness {lufs:.1f} LUFS off target by {d:.1f}")

    # ── duration vs requested (optional) ────────────────────────────────
    actual_s = None
    rm = metrics.get("render_meta") or {}
    if isinstance(rm.get("video_seconds"), (int, float)):
        actual_s = rm["video_seconds"]
    elif isinstance(metrics.get("narration_audio_s"), (int, float)):
        actual_s = metrics["narration_audio_s"]
    if target_minutes and actual_s:
        rel = abs(actual_s - target_minutes * 60.0) / (target_minutes * 60.0)
        msg = (f"duration {actual_s/60:.1f}min vs requested "
               f"{target_minutes:.0f}min ({rel*100:.0f}% off)")
        if rel > _DUR_FAIL:
            fails.append(msg)
        elif rel > _DUR_WARN:
            warns.append(msg)

    # ── pacing: avg shot length (render_meta ground truth) ───────────────
    avg = (rm.get("shot_len_s") or {}).get("avg") if isinstance(
        rm.get("shot_len_s"), dict) else None
    if isinstance(avg, (int, float)) and not (_SHOT_AVG_LO <= avg <= _SHOT_AVG_HI):
        warns.append(f"avg shot {avg:.1f}s outside {_SHOT_AVG_LO}-{_SHOT_AVG_HI}s")

    # ── graphics clutter (distinct graphic scenes / scenes) ─────────────
    ct = metrics.get("card_types") or {}
    sc = metrics.get("scenes")
    if ct and isinstance(sc, int) and sc > 0:
        ratio = sum(ct.values()) / sc
        if ratio > _GFX_DENSITY_WARN:
            warns.append(f"graphic density {ratio*100:.0f}% of scenes "
                         f"(>{_GFX_DENSITY_WARN*100:.0f}%)")

    # ── soft warnings (non-error log flags) ─────────────────────────────
    wc = metrics.get("warning_error_count", 0)
    if wc and not hard:
        warns.append(f"{wc} warning line(s) in log (non-fatal)")

    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
    return {"verdict": verdict, "fails": fails, "warns": warns}


def _card_types(render_dir: Path) -> dict:
    sj = render_dir / "script.json"
    if not sj.is_file():
        return {}
    try:
        scenes = json.loads(sj.read_text()).get("scenes", [])
    except Exception:                                          # noqa: BLE001
        return {}
    out = {}
    for s in scenes:
        gk = (s.get("graphic_kind") or "").strip()
        if gk:
            out[gk] = out.get(gk, 0) + 1
    return out


def _loudness(render_dir: Path) -> dict:
    """LUFS/true-peak from benchmark_scores.json (by slug), if scored."""
    for p in (ROOT / "research" / "benchmark_scores.json",
              ROOT / "tools" / "benchmark_scores.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:                                      # noqa: BLE001
            continue
        # store is dict {slug: [history records...]} (+ '_' meta keys)
        rec = data.get(render_dir.name) if isinstance(data, dict) else None
        if isinstance(rec, list) and rec:
            rec = rec[-1]
        if isinstance(rec, dict):
            return {"lufs": rec.get("lufs"),
                    "true_peak_db": rec.get("true_peak_db")}
    return {"lufs": None, "true_peak_db": None}


def main(target: str, out: str | None, target_min: float | None = None) -> int:
    tp = Path(target)
    if tp.is_dir():
        cand = [tp.parent / f"{tp.name}_log.txt",
                ROOT / "output" / f"{tp.name}_log.txt"]
        log_file = next((c for c in cand if c.is_file()), None)
        render_dir = tp
    else:
        log_file = tp
        render_dir = None
    text = log_file.read_text(errors="ignore") if log_file and log_file.is_file() else ""
    metrics = parse_log(text)
    # locate the render dir (for script.json / loudness / render_meta)
    if render_dir is None:
        mv = re.search(r"video:\s+(\S+\.mp4)", text)
        render_dir = Path(mv.group(1)).parent if mv else (
            log_file.parent if log_file else ROOT)
    metrics["card_types"] = _card_types(render_dir)
    metrics["audio"] = _loudness(render_dir)
    rm = render_dir / "render_meta.json"
    if rm.is_file():
        try:
            metrics["render_meta"] = json.loads(rm.read_text())
        except Exception:                                      # noqa: BLE001
            pass
    metrics["_render_dir"] = str(render_dir)
    metrics["_log_file"] = str(log_file) if log_file else None
    metrics["qa"] = qa_verdict(metrics, target_min)

    out_path = Path(out) if out else (render_dir / "render_metrics.json")
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"render_metrics.json -> {out_path}")
    keys = ("scenes", "beats", "script_words", "narration_audio_s",
            "graphic_card_images", "transitions_motivated",
            "black_spans_detected", "black_spans_after_repair", "music_cues",
            "assembly_wall_s", "warning_error_count")
    for k in keys:
        print(f"   {k:24s} {metrics.get(k)}")
    print(f"   {'card_types':24s} {metrics.get('card_types')}")
    print(f"   {'sources':24s} {metrics.get('sources')}")
    print(f"   {'audio':24s} {metrics.get('audio')}")
    qa = metrics["qa"]
    print(f"\n   QA VERDICT: {qa['verdict']}")
    for f in qa["fails"]:
        print(f"     FAIL  {f}")
    for w in qa["warns"]:
        print(f"     warn  {w}")
    # exit non-zero only on a hard FAIL so this can gate CI / auto-runs
    return 2 if qa["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="render log file OR render dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-min", type=float, default=None,
                    help="requested duration in minutes (enables duration QA)")
    a = ap.parse_args()
    sys.exit(main(a.target, a.out, a.target_min))
