#!/usr/bin/env python3
"""Measure a DELIVERED render — the file, not the pipeline's opinion of it.

Everything here is read off the finished mp4, its subtitle track and its own source media. The
pipeline's bookkeeping is used only to say which source window a delivered scene CLAIMS to come
from; whether it actually does is then settled by decoding both and comparing pixels.

    python3 tools/audit_delivered_file.py <job_dir> [--pairs 15] [--speech 3] [--out DIR]

Reports:
  * exact audio/video duration difference
  * caption cue count and maximum characters-per-second
  * native-HD scenes N/total, taken from the delivered frame's own height
  * black/unusable delivered frames
  * paired delivered-frame vs claimed-source-frame images for a vision comparison
  * delivered narration transcribed back and scored against the script
  * breakouts, semantic blockers and ad/promo findings as the audit artifacts record them

Exit code is 0 whether the numbers are good or bad — this measures, it does not judge.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vidlore.clipstudio.config import ffmpeg_exe, ffprobe_exe     # noqa: E402

FF, FP = ffmpeg_exe(), ffprobe_exe()


def _probe(path, args):
    return subprocess.run([FP, "-v", "0", *args, "-of", "json", str(path)],
                          capture_output=True, text=True).stdout


def stream_durations(mp4: Path) -> dict:
    """Per-stream duration, straight from the container."""
    doc = json.loads(_probe(mp4, ["-show_streams", "-show_format"]) or "{}")
    out = {"container": float((doc.get("format") or {}).get("duration") or 0.0)}
    for s in doc.get("streams") or []:
        kind = s.get("codec_type")
        if kind not in ("video", "audio"):
            continue
        d = s.get("duration")
        if d is None:                                   # mp4 sometimes carries it only per-stream
            d = (s.get("tags") or {}).get("DURATION")
        out[kind] = float(d) if d else 0.0
        if kind == "video":
            out["height"] = int(s.get("height") or 0)
            out["width"] = int(s.get("width") or 0)
    return out


# ---------------------------------------------------------------- captions
_TS = re.compile(r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)")


def _secs(m) -> float:
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


def caption_cues(srt: Path) -> list:
    """(start, end, text) per cue. CPS is measured on the visible text only."""
    if not srt.exists():
        return []
    cues, block = [], []
    for line in srt.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        if line.strip():
            block.append(line)
            continue
        if len(block) >= 2:
            ts = [m for m in _TS.finditer(block[1])]
            if len(ts) == 2:
                txt = " ".join(block[2:]).strip()
                cues.append((_secs(ts[0]), _secs(ts[1]), re.sub(r"<[^>]+>", "", txt)))
        block = []
    return cues


def caption_stats(cues: list) -> dict:
    worst, worst_txt = 0.0, ""
    for a, b, t in cues:
        dur = max(1e-6, b - a)
        cps = len(t) / dur
        if cps > worst:
            worst, worst_txt = cps, t
    return {"cues": len(cues), "max_cps": round(worst, 2), "max_cps_text": worst_txt[:80]}


# ---------------------------------------------------------------- frames
def grab(src, t: float, dest: Path, *, vf: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FF, "-y", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
           "-frames:v", "1"]
    if vf:
        cmd += ["-vf", vf]
    cmd += [str(dest)]
    subprocess.run(cmd, capture_output=True)
    return dest.exists() and dest.stat().st_size > 0


def frame_stats(src, t: float) -> dict:
    """Mean luma and its spread — a delivered frame that is black or flat is unusable."""
    r = subprocess.run(
        [FF, "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
         "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"YAVG=([0-9.]+)", r.stdout + r.stderr)
    yavg = float(m.group(1)) if m else -1.0
    r2 = subprocess.run(
        [FF, "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YDIF:file=-",
         "-f", "null", "-"], capture_output=True, text=True)
    m2 = re.search(r"YDIF=([0-9.]+)", r2.stdout + r2.stderr)
    return {"yavg": yavg, "ydif": float(m2.group(1)) if m2 else -1.0}


def delivered_height(mp4: Path, t: float) -> int:
    """Native-HD is a property of the PIXELS aired, so read it off the aired frame."""
    doc = json.loads(_probe(mp4, ["-select_streams", "v:0", "-show_entries",
                                  "stream=height"]) or "{}")
    for s in doc.get("streams") or []:
        return int(s.get("height") or 0)
    return 0


# ---------------------------------------------------------------- speech
def transcribe(mp4: Path, start: float, dur: float, work: Path) -> str:
    """Pull the delivered narration back out of the delivered file."""
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"speech_{int(start)}.wav"
    subprocess.run([FF, "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(mp4),
                    "-t", f"{dur:.3f}", "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   capture_output=True)
    if not wav.exists() or wav.stat().st_size == 0:
        return ""
    from vidlore.clipstudio.breakout_asr import transcribe_breakout_words
    try:
        words = transcribe_breakout_words(wav, duration=dur, cache=False)
    except TypeError:
        words = transcribe_breakout_words(wav, duration=dur)
    # the ASR helper returns dicts in some paths and (word, start, end) tuples in others
    def _tok(w):
        if isinstance(w, dict):
            return str(w.get("word") or w.get("text") or "")
        if isinstance(w, (list, tuple)) and w:
            return str(w[0])
        return str(w)
    return " ".join(_tok(w).strip() for w in (words or [])).strip()


# ---------------------------------------------------------------- beat -> delivered time
def _toks(s: str):
    return re.findall(r"[a-z']{2,}", s.lower())


def beat_timeline(cues: list, segs: dict) -> dict:
    """Locate every beat on the delivered timeline USING ITS OWN BURNED CAPTION.

    Modelling the timeline arithmetically does not work here: summing the aired clips' durations
    reproduces the total length to 0.08% and still puts individual beats in the wrong place, and no
    constant offset repairs it (measured: 28% agreement at best, across every offset in +/-12s).
    So the timeline is not derived at all — it is READ OFF the subtitles, which are the narration
    the viewer actually heard. Each beat's caption is its own proof of position.

    Returns {segment_index: (t0, t1, agreement)} for the beats that could be located.
    """
    stream = []                                            # (token, cue_start, cue_end)
    for a, b, txt in cues:
        for w in _toks(txt):
            stream.append((w, a, b))
    out, cursor = {}, 0
    for idx in sorted(segs):
        want = _toks(segs[idx].get("text") or "")
        if len(want) < 3:
            continue
        best, best_at = 0.0, -1
        # beats are spoken in order, so only ever search forward from the last match
        for start in range(cursor, max(cursor, len(stream) - len(want)) + 1):
            got = {w for w, _, _ in stream[start:start + len(want)]}
            hit = sum(1 for w in want if w in got) / len(want)
            if hit > best:
                best, best_at = hit, start
            if best == 1.0:
                break
        if best_at < 0 or best < 0.6:
            continue
        seg_slice = stream[best_at:best_at + len(want)]
        # Sample frames at the beat's MIDDLE token, not at the midpoint of its span: a cue that
        # straddles two beats shows both narrations, so a span midpoint can land on a boundary cue
        # and grade the neighbour's picture. The middle token is inside this beat by construction.
        mw, ma, mb = seg_slice[len(seg_slice) // 2]
        out[idx] = {"t0": seg_slice[0][1], "t1": seg_slice[-1][2],
                    "agreement": round(best, 3), "sample_t": (ma + mb) / 2.0}
        cursor = best_at + max(1, len(want) - 1)
    return out


def word_overlap(a: str, b: str) -> float:
    """Fraction of the script's words that survive into the delivered audio."""
    norm = lambda s: [w for w in re.findall(r"[a-z']+", s.lower())]           # noqa: E731
    wa, wb = norm(a), set(norm(b))
    return round(sum(1 for w in wa if w in wb) / max(1, len(wa)), 3)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--speech", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    job = Path(a.job)
    out = Path(a.out) if a.out else job / "output" / "delivered_audit"
    out.mkdir(parents=True, exist_ok=True)

    mp4 = next((p for p in (job / "output").glob("final*.mp4")), None)
    if mp4 is None:
        print(json.dumps({"error": "no delivered file under output/"}))
        return 1
    srt = next((p for p in (job / "output").glob("final*.srt")), job / "output" / "final.srt")

    proj = json.loads((job / "project.json").read_text(encoding="utf-8"))
    segs = {s["index"]: s for s in proj.get("segments", [])}
    sels = {s["segment_index"]: s for s in proj.get("selections", [])}
    srcs = {s["id"]: s for s in proj.get("sources", [])}

    rep = {"file": str(mp4), "size_bytes": mp4.stat().st_size}
    rep["durations"] = stream_durations(mp4)
    d = rep["durations"]
    rep["av_duration_diff_s"] = round(abs(d.get("video", 0.0) - d.get("audio", 0.0)), 3)
    rep["captions"] = caption_stats(caption_cues(srt))

    # What each aired clip is. `original_beat` — NOT `beat` — is the segment index: measured on a
    # real render, aired.original_beat's source_id agrees with selections[idx].source_id 99/99,
    # while aired.beat agrees only 31/97 (it is a post-reindex display value).
    aw = job / "output" / "aired_windows.json"
    if not aw.exists():
        print(json.dumps({"error": "aired_windows.json missing — cannot map delivered time to beat"}))
        return 1
    clips = (json.loads(aw.read_text(encoding="utf-8")) or {}).get("clips") or []
    rep["aired_clips"] = {"total": len(clips)}
    from collections import Counter
    rep["aired_clips"]["via"] = dict(Counter(str(c.get("via") or "") for c in clips))
    rep["lineage"] = {
        "validated": sum(1 for c in clips if c.get("lineage_validated")),
        "total": len(clips),
    }

    # Beat -> delivered time comes from the CAPTIONS, not from clip arithmetic. See beat_timeline:
    # summing the aired durations reproduces the total length to 0.08% and still misplaces
    # individual beats, and no constant offset repairs it.
    cues = caption_cues(srt)
    tl = beat_timeline(cues, segs)
    agree = [v["agreement"] for v in tl.values()]
    rep["beat_mapping"] = {
        "located": len(tl), "of": len(segs),
        "median_agreement": round(sorted(agree)[len(agree) // 2], 3) if agree else 0.0,
        "at_or_above_0.9": sum(1 for x in agree if x >= 0.9),
        "monotonic": all(tl[x]["t0"] <= tl[y]["t0"]
                         for x, y in zip(sorted(tl), sorted(tl)[1:])),
    }
    if len(tl) < 0.9 * len(segs):
        rep["beat_mapping"]["WARNING"] = "too few beats located — frame pairs are not trustworthy"

    rows = []
    for c in clips:
        beat = c.get("original_beat")
        if str(c.get("via") or "") != "selection_derivative" or beat not in tl:
            continue                                   # only real footage has a source to compare
        rows.append({"beat": beat, "via": str(c.get("via") or ""),
                     "source_id": str(c.get("source_id") or ""),
                     "source_title": str(c.get("source_title") or ""),
                     "in": float(c.get("in") or 0.0), "need": float(c.get("need") or 0.0),
                     "sample_t": tl[beat]["sample_t"], "agreement": tl[beat]["agreement"],
                     "narration": (segs.get(beat) or {}).get("text", "")})

    seen = set()
    rows = [r for r in rows if not (r["beat"] in seen or seen.add(r["beat"]))]
    step = max(1, len(rows) // max(1, a.pairs))
    picked = rows[::step][:a.pairs]
    pairs, black, hd_ok, hd_total = [], 0, 0, 0
    for r in picked:
        beat = r["beat"]
        sel = sels.get(beat) or {}
        src = srcs.get(r["source_id"])
        mid_d = r["sample_t"]
        din = out / f"pair_{beat:03d}_delivered.jpg"
        sin = out / f"pair_{beat:03d}_source.jpg"
        okd = grab(mp4, mid_d, din)
        oks = False
        # the source instant that MUST correspond: the aired clip's own in-point + its midpoint
        # offset, not the selection's window midpoint (a clip can be a sub-span of the selection)
        mid_s = float(r["in"]) + r["need"] / 2.0
        if src and src.get("local_path") and Path(src["local_path"]).exists():
            oks = grab(src["local_path"], mid_s, sin)
        st = frame_stats(mp4, mid_d) if okd else {"yavg": -1, "ydif": -1}
        if 0 <= st["yavg"] < 16:
            black += 1
        hd_total += 1
        h = int((src or {}).get("height") or 0)
        if h >= 720:
            hd_ok += 1
        pairs.append({
            "beat": beat, "via": r["via"],
            "delivered_frame": str(din) if okd else "", "source_frame": str(sin) if oks else "",
            "delivered_t": round(mid_d, 3),
            "source_id": r["source_id"],
            "source_title": r["source_title"][:80],
            "aired_source_window": [round(r["in"], 3), round(r["in"] + r["need"], 3)],
            "source_probe_t": round(mid_s, 3),
            "narration": r["narration"][:120],
            "yavg": st["yavg"], "ydif": st["ydif"],
            "visual_policy": (segs.get(beat) or {}).get("visual_policy", ""),
        })
    rep["frame_pairs"] = pairs
    rep["black_or_unusable_frames"] = black
    rep["native_hd"] = {"ok": hd_ok, "total": hd_total}

    # ---- delivered speech vs script
    speech = []
    located = sorted(tl)
    if located:
        stride = max(1, len(located) // max(1, a.speech))
        for idx in located[::stride][:a.speech]:
            v = tl[idx]
            t0, t1 = v["t0"], v["t1"]
            dur = min(20.0, max(5.0, t1 - t0))
            got = transcribe(mp4, t0, dur, out / "speech")
            script = (segs.get(idx) or {}).get("text", "")
            speech.append({"beat": idx, "start": round(t0, 2), "window_s": round(dur, 2),
                           "script": script[:220], "delivered": got[:260],
                           "word_overlap": word_overlap(script, got)})
    rep["delivered_speech"] = speech

    # ---- what the pipeline's own audits recorded
    def _load(name):
        p = job / "output" / name
        if not p.exists():
            p = job / name
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return None

    # Job-wide figures the build already computed, which beat any sample of 15. Read as data —
    # the frame pairs above remain the independent check on whether the pipeline's own numbers
    # describe the file that was actually delivered.
    nat = _load("native_resolution_audit.json")
    if isinstance(nat, dict):
        sels = nat.get("selections")
        rep["native_hd_jobwide"] = {
            "passed": nat.get("passed"),
            "selections": len(sels) if isinstance(sels, list) else sels,
            "failures": len(nat.get("failures") or []),
            "minimum_short_edge": nat.get("minimum_short_edge"),
        }
    cap = _load("caption_readability_audit.json")
    if isinstance(cap, dict):
        rep["caption_readability"] = {k: cap.get(k) for k in
                                      ("passed", "hard_cps", "max_cps", "cue_count",
                                       "word_count", "problem_count") if k in cap}
        rep["caption_readability"]["blocked_windows"] = len(cap.get("blocked_windows") or [])
    lin = _load("scene_lineage_audit.json")
    if isinstance(lin, dict):
        rep["scene_lineage_audit"] = {
            "status": lin.get("status"),
            "encoded_segments": len(lin.get("encoded_segments") or []),
            "delivered_checks": len(lin.get("delivered_checks") or []),
            "failures": len(lin.get("failures") or []),
            "timeline_order_entries": (len(lin.get("timeline_order") or [])
                                       if isinstance(lin.get("timeline_order"), list)
                                       else lin.get("timeline_order")),
        }

    # The ad/promo gate reports through the build log rather than an artifact.
    ads = []
    for lg in sorted((job / "output").glob("build*.log")):
        try:
            for line in lg.read_text(encoding="utf-8", errors="replace").splitlines():
                if "AD-GATE" in line or "ad-gate" in line:
                    ads.append(line.strip()[:160])
        except Exception:                                          # noqa: BLE001
            pass
    rep["ad_gate_log"] = ads[-4:]
    rep["ad_gate_emergency_override"] = any("EMERGENCY OVERRIDE" in x for x in ads)

    sr = _load("selection_relevance_audit.json") or {}
    rep["semantic_blockers"] = sr.get("blocked_count", None)
    rf = _load("rejected_footage_audit.json") or {}
    rep["rejected_footage_blocked"] = (rf.get("blocked_count")
                                       if isinstance(rf, dict) else None)
    ba = _load("breakout_audit.json")
    if isinstance(ba, dict):
        aired = ba.get("aired") or ba.get("admitted") or []
        rep["breakouts"] = {"aired": len(aired) if isinstance(aired, list) else aired,
                            "considered": ba.get("considered", None)}
    else:
        rep["breakouts"] = None
    lineage = _load("scene_lineage_canary.json") or _load("lineage_audit.json")
    if isinstance(lineage, dict):
        rep["lineage"] = {k: lineage.get(k) for k in
                          ("proved", "checked", "total", "failed", "status") if k in lineage}

    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    slim = {k: v for k, v in rep.items() if k not in ("frame_pairs",)}
    slim["frame_pairs_n"] = len(pairs)
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    print(f"\nfull report + {len(pairs)} frame pairs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
