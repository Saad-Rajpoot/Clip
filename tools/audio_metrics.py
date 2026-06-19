#!/usr/bin/env python3
"""audio_metrics.py — local audio-QA for a rendered Vidlore video.

Measures the FINAL mix with ffmpeg (no network, no re-render) and writes
`render_audio_metrics.json` beside the input:

  • integrated LUFS + loudness range (LRA)   — documentary target ~ -16 LUFS
  • true peak dBTP                            — should stay < -1.0 dBTP
  • overall RMS / peak                        — astats
  • silence pockets (count + durations)       — silencedetect
  • loudest 3 one-second windows              — from ebur128 momentary
  • simple verdicts (PASS/WARN) per metric

Optional --voice / --music / --sfx stem wavs (the render's intermediate
beds) add per-stem RMS so you can see dialogue-vs-music-vs-SFX balance.

Usage:
  python tools/audio_metrics.py <media.mp4> [--voice v.wav --music m.wav --sfx s.wav]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# allow `from vidlore import sfx` (for the SFX audit) regardless of CWD
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG = get_ffmpeg_exe()
except Exception:                                                  # noqa: BLE001
    FFMPEG = "ffmpeg"

TARGET_LUFS = -16.0
TARGET_TP = -1.0


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True).stderr


def _ebur128(path: Path) -> dict:
    """Integrated LUFS, LRA, true peak + momentary series (for loudest)."""
    err = _run([FFMPEG, "-nostats", "-i", str(path),
                "-filter_complex", "ebur128=peak=true", "-f", "null", "-"])
    out: dict = {}
    # ebur128 prints per-FRAME readings then a final Summary; the integrated
    # 'I' starts near -70 and converges. Take the LAST occurrence (the
    # Summary value), not the first frame.
    iv = re.findall(r"I:\s*(-?\d+\.?\d*)\s*LUFS", err)
    if iv:
        out["integrated_lufs"] = float(iv[-1])
    lr = re.findall(r"LRA:\s*(-?\d+\.?\d*)\s*LU", err)
    if lr:
        out["lra"] = float(lr[-1])
    pk = re.findall(r"Peak:\s*(-?\d+\.?\d*)\s*dBFS", err)
    if pk:
        out["true_peak_dbtp"] = float(pk[-1])
    # momentary loudness series "t: <s> ... M: <lufs>"
    series = []
    for mm in re.finditer(r"t:\s*([\d.]+).*?M:\s*(-?\d+\.?\d*)", err):
        try:
            series.append((float(mm.group(1)), float(mm.group(2))))
        except ValueError:
            pass
    if series:
        top = sorted(series, key=lambda x: -x[1])[:3]
        out["loudest_windows"] = [{"t_s": round(t, 1), "momentary_lufs": v}
                                  for t, v in top]
    return out


def _astats(path: Path) -> dict:
    err = _run([FFMPEG, "-i", str(path), "-filter_complex",
                "astats=metadata=1", "-f", "null", "-"])
    out: dict = {}
    m = re.search(r"RMS level dB:\s*(-?\d+\.?\d*)", err)
    if m:
        out["rms_db"] = float(m.group(1))
    m = re.search(r"Peak level dB:\s*(-?\d+\.?\d*)", err)
    if m:
        out["peak_db"] = float(m.group(1))
    return out


def _silence(path: Path) -> dict:
    err = _run([FFMPEG, "-i", str(path), "-af",
                "silencedetect=noise=-32dB:d=0.5", "-f", "null", "-"])
    durs = [float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", err)]
    return {"silence_pocket_count": len(durs),
            "silence_total_s": round(sum(durs), 2),
            "silence_pockets_s": [round(d, 2) for d in durs[:40]]}


def _stem_rms(path: str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    return _astats(p).get("rms_db")


def audit_sfx_schedule(events: list) -> dict:
    """Per-kind SFX REPEAT audit (anti-rep axis #7). `events` is an ordered
    list of SFX event-kind strings (or {"kind": ...} dicts) — a render's SFX
    schedule. Replays them through `sfx.pick` with a shared history and, per
    SFX CATEGORY, reports the concrete presets chosen, their counts, and the
    CLOSEST spacing (in category-events) between two identical presets.

    `sfx.pick` excludes the last-3 picks per category, so a repeat inside that
    window is only POSSIBLE (and acceptable) when the family pool is <=3 — the
    '3-variant floor'. A repeat inside the window with a pool >3 is a REGRESSION
    (the cooldown failed) and is flagged. No render needed."""
    try:
        from vidlore import sfx
    except Exception as e:                                         # noqa: BLE001
        return {"error": f"cannot import sfx: {e}"}
    kinds = [e if isinstance(e, str) else (e or {}).get("kind", "")
             for e in (events or [])]
    hist: dict = {}
    cat_seq: dict = {}                 # category -> [preset names in order]
    cat_pool: dict = {}                # category -> family/category pool size
    for i, k in enumerate(kinds):
        if not k:
            continue
        cat = sfx.EVENT_CATEGORY.get(k, "text_reveal")
        name, _ = sfx.pick(k, seed=i * 0.37 + 0.11, history=hist)
        cat_seq.setdefault(cat, []).append(name)
        if cat not in cat_pool:
            fam = sfx.EVENT_PREFER.get(k)
            pool = [n for n in sfx._POOL.get(cat, [])
                    if fam and n.startswith(fam)]
            cat_pool[cat] = len(pool) or len(sfx._POOL.get(cat, []))
    by_cat: dict = {}
    regressions: list = []
    for cat, names in cat_seq.items():
        from collections import Counter
        cnt = Counter(names)
        last: dict = {}
        min_gap = None
        for j, nm in enumerate(names):
            if nm in last:
                g = j - last[nm]
                min_gap = g if min_gap is None else min(min_gap, g)
                if g < 3 and cat_pool.get(cat, 1) > 3:
                    regressions.append({"category": cat, "preset": nm,
                                        "gap": g})
            last[nm] = j
        by_cat[cat] = {"events": len(names), "distinct": len(cnt),
                       "pool": cat_pool.get(cat), "counts": dict(cnt),
                       "min_repeat_gap": min_gap}
    return {
        "sfx_events": len([k for k in kinds if k]),
        "categories": len(by_cat),
        "by_category": by_cat,
        "cooldown_regressions": regressions,
        "verdict": "PASS" if not regressions
        else f"WARN({len(regressions)} repeats inside cooldown w/ pool>3)",
    }


# A realistic mixed SFX stream (whoosh-heavy + pins + booms + UI + paper +
# text reveals) to exercise the cooldown when no real render schedule exists.
_SYNTH_SFX_STREAM = (
    ["whoosh"] * 6 + ["map_pin"] * 5 + ["boom"] * 4 + ["keyboard"] * 5
    + ["paper"] * 4 + ["text_reveal"] * 8 + ["stinger"] * 3
    + ["whoosh", "map_pin", "boom", "keyboard", "paper", "text_reveal"] * 4
)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: audio_metrics.py <media> [--voice v --music m --sfx s]")
        print("       audio_metrics.py --sfx-audit [schedule.json]")
        return 2
    # ---- standalone SFX repeat audit (no render needed) -------------------
    if argv[0] == "--sfx-audit":
        if len(argv) > 1 and Path(argv[1]).exists():
            data = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
            events = data.get("events", data) if isinstance(data, dict) else data
            src = argv[1]
        else:
            events = _SYNTH_SFX_STREAM
            src = "synthetic stress stream"
        rep = audit_sfx_schedule(events)
        rep["source"] = src
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("verdict", "").startswith("PASS") else 1
    media = Path(argv[0])
    if not media.exists():
        print(f"not found: {media}")
        return 2
    stems = {}
    i = 1
    while i < len(argv) - 1:
        if argv[i] in ("--voice", "--music", "--sfx"):
            stems[argv[i][2:]] = argv[i + 1]; i += 2; continue
        i += 1

    m = {"file": media.name}
    m.update(_ebur128(media))
    m.update(_astats(media))
    m.update(_silence(media))
    if stems:
        m["stems_rms_db"] = {k: _stem_rms(v) for k, v in stems.items()}

    # per-kind SFX repeat audit — auto-pick a schedule sidecar if the render
    # emitted one (<stem>_sfx_schedule.json), else skip silently.
    for cand in (media.with_name(media.stem + "_sfx_schedule.json"),
                 media.parent / "sfx_schedule.json"):
        if cand.exists():
            try:
                _d = json.loads(cand.read_text(encoding="utf-8"))
                _ev = _d.get("events", _d) if isinstance(_d, dict) else _d
                m["sfx_repeat_audit"] = audit_sfx_schedule(_ev)
            except Exception:                                      # noqa: BLE001
                pass
            break

    # verdicts
    v = {}
    li = m.get("integrated_lufs")
    if li is not None:
        v["loudness"] = ("PASS" if abs(li - TARGET_LUFS) <= 2.0
                         else "WARN(%.1f vs %.0f)" % (li, TARGET_LUFS))
    tp = m.get("true_peak_dbtp")
    if tp is not None:
        v["true_peak"] = "PASS" if tp <= TARGET_TP else "WARN(clip risk %.1f)" % tp
    m["verdicts"] = v

    out = media.parent / "render_audio_metrics.json"
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
