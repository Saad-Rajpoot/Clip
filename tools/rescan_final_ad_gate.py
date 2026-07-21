#!/usr/bin/env python3
"""Re-run the (caption-aware) final-video ad scan on a finished or QUARANTINED render.

A render quarantined as final.FAILED_AD_QA.mp4 by a FALSE-POSITIVE ad-gate hit (e.g. the
narration's own "subscribe…" outro caption) is a complete, good video — re-encoding it for an
hour to re-reach the same gate is pure waste. This tool runs the SAME production scan
(_final_video_ad_scan, with the own-caption whitelist) directly against the file:

    python3 tools/rescan_final_ad_gate.py <project_dir> [--restore]

  <project_dir>   a portal job dir (contains output/) or the output/ dir itself
  --restore       on a CLEAN verdict, restore final.FAILED_AD_QA.mp4 → final.mp4

Whitelist parity with production: the own-caption schedule is loaded ONLY when the project
actually burned captions (project.json → meta.caption_settings.enabled — the same _cap_on the
build's gate uses). final.srt exists even for captions-OFF renders, where screen text can only
come from the footage, so an unconditional whitelist would wrongly clear real promo hits.

NOTE: restoring the file does NOT flip the portal job's status — the job still reads 'failed',
and its Resume button re-runs assembly and OVERWRITES final.mp4. Use the restored file directly.

Exit codes: 0 clean · 2 blocked (real promo hits — printed) · 3 unverified / setup error.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.build import (                          # noqa: E402
    _final_video_ad_scan, _parse_srt_events, _parse_ass_events, _norm_caption_words)


def _captions_burned(root: Path) -> bool:
    """The persisted _cap_on for this project (build.py writes meta.caption_settings on every
    render). Missing/unreadable → False: the conservative default is NO whitelist."""
    try:
        meta = json.loads((root / "project.json").read_text(encoding="utf-8")).get("meta", {})
        return bool((meta.get("caption_settings") or {}).get("enabled"))
    except Exception:
        return False


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    restore = "--restore" in sys.argv[1:]
    if not args:
        print(__doc__)
        return 3
    root = Path(args[0]).expanduser().resolve()
    out = root / "output" if (root / "output").is_dir() else root
    proj_root = out.parent if (out.parent / "project.json").exists() else out
    final = out / "final.mp4"
    quar = out / "final.FAILED_AD_QA.mp4"
    target = final if final.exists() else quar
    if not target.exists():
        print(f"no final.mp4 / final.FAILED_AD_QA.mp4 under {out}")
        return 3

    try:
        from rapidocr_onnxruntime import RapidOCR               # same engine build.py uses
        ocr = RapidOCR()
    except Exception as e:                                      # noqa: BLE001
        print(f"RapidOCR unavailable ({e}) — cannot scan")
        return 3

    # Own-caption schedule — production parity: only when this project burned captions, and
    # explicitly from output/final.srt (the quarantined file's stem would otherwise look for
    # final.FAILED_AD_QA.srt, which never exists).
    own = []
    burned = _captions_burned(proj_root)
    srt = out / "final.srt"
    ass = out / "work" / "breakout_caps.ass"
    if burned:
        evs = []
        if srt.exists():
            evs.extend(_parse_srt_events(srt))
        if ass.exists():
            evs.extend(_parse_ass_events(ass))
        for t0, t1, text in evs:
            ws = _norm_caption_words(text)
            if ws:
                own.append((float(t0), float(t1), ws, text))
    else:
        print("captions were NOT burned on this render (project.json caption_settings) — "
              "no own-caption whitelist, scanning at full strictness")

    # ISOLATED scratch dir: _final_video_ad_scan writes work/_adscan frame JPEGs with fixed
    # names — sharing the project's work/ with a concurrently running render's own ad gate
    # would cross-contaminate both scans.
    scan_work = Path(tempfile.mkdtemp(prefix="adrescan_"))
    print(f"scanning {target.name} ({target.stat().st_size / 1e6:.0f} MB) — "
          f"own-caption schedule: {len(own)} event(s) "
          f"(captions_burned={burned}, srt={'yes' if srt.exists() else 'no'}, "
          f"breakout-ass={'yes' if ass.exists() else 'no'})")

    r = _final_video_ad_scan(target, scan_work, ocr, log=print, own_captions=own or None)
    print(json.dumps({k: r[k] for k in ("status", "frames", "ocr_errors", "reason")}, indent=1))
    for h in r.get("hits", []):
        print(f"  HIT @{h['t']}s token={h['token']!r} text={h.get('text', '')!r}")
    if r["status"] == "clean":
        if restore and target == quar:
            if final.exists():
                print(f"NOT restoring: {final.name} already exists (a newer render appeared "
                      f"during the scan) — refusing to overwrite it. The clean quarantined "
                      f"file remains at {quar.name}.")
                return 0
            quar.rename(final)
            print(f"restored → {final}")
            print("note: the portal job still reads 'failed'; its Resume button re-runs "
                  "assembly and would OVERWRITE this file. Use the restored file directly.")
        elif target == quar:
            print("clean — rerun with --restore to rename it back to final.mp4")
        return 0
    return 2 if r["status"] == "blocked" else 3


if __name__ == "__main__":
    sys.exit(main())
