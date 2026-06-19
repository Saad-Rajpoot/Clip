#!/usr/bin/env python3
"""RC5.1 — post-render QA: game-UI geometry inside an engine CARD WINDOW is
exempt by default (the engine's own redacted_document / classified / dashboard /
diagram cards legitimately carry axis-aligned panel geometry that trips the
ui_geom probe), while game-UI on a FOOTAGE beat OUTSIDE every card window still
FAILs at full sensitivity. VIDLORE_QA_CARD_UI_STRICT=1 restores the old
over-eager behaviour. Fast + deterministic — all heavy seams mocked.

Regression anchor for the Iran-Iraq served render where the classified card at
~116-117s (ui_geom≈0.65, in_card=True) was a QA false positive.
"""
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vidlore.relevance_qa as RQ

IN_WINDOW_TS = 15.0     # inside the mocked card window (10,20)
OUT_WINDOW_TS = 50.0    # footage beat, outside every window
_orig = {}


def _install_mocks():
    _orig["probe"] = RQ._probe_duration
    _orig["build"] = RQ._build_sample_times
    _orig["beats"] = RQ._beat_times
    _orig["mg"] = RQ._resolve_mg_manifest
    _orig["win"] = RQ._card_windows
    _orig["extract"] = RQ._extract_frame
    _orig["sig"] = RQ.VR.graphic_signal

    RQ._probe_duration = lambda *a, **k: 60.0
    RQ._build_sample_times = lambda *a, **k: [IN_WINDOW_TS, OUT_WINDOW_TS]
    RQ._beat_times = lambda *a, **k: []
    RQ._resolve_mg_manifest = lambda *a, **k: {"scenes": [{"scene_index": 0, "primitive": "redacted_document"}]}
    RQ._card_windows = lambda *a, **k: [(10.0, 20.0)]            # window covers IN_WINDOW_TS only

    def _fake_extract(ff, mp4, ts, dest):
        Path(dest).write_bytes(b"\x89PNG\r\n\x1a\n")             # touch a non-empty file
        return True
    RQ._extract_frame = _fake_extract

    # every sampled frame probes as a strong UI-geometry designed graphic
    RQ.VR.graphic_signal = lambda *a, **k: {
        "looks_ui_screenshot": True, "ui_geom": 0.65,
        "graphic_dom": 0.08, "looks_designed": True,
    }


def _restore():
    RQ._probe_duration = _orig["probe"]
    RQ._build_sample_times = _orig["build"]
    RQ._beat_times = _orig["beats"]
    RQ._resolve_mg_manifest = _orig["mg"]
    RQ._card_windows = _orig["win"]
    RQ._extract_frame = _orig["extract"]
    RQ.VR.graphic_signal = _orig["sig"]


def _sweep(strict):
    if strict:
        os.environ["VIDLORE_QA_CARD_UI_STRICT"] = "1"
    else:
        os.environ.pop("VIDLORE_QA_CARD_UI_STRICT", None)
    os.environ["VIDLORE_VISUAL_RELEVANCE"] = "1"
    meta = {"video_seconds": 60.0, "scene_starts": [0.0, 40.0], "scene_durations": [40.0, 20.0]}
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(b"\x00" * 32)
        mp4 = fh.name
    try:
        return RQ.sweep(mp4, meta, expected_sha256=None)
    finally:
        try: os.unlink(mp4)
        except OSError: pass


def _ts_set(res):
    return {round(float(f.get("timestamp")), 1) for f in res.get("flags", [])}


def main():
    _install_mocks()
    failures = []
    try:
        d = _sweep(strict=False)
        s = _sweep(strict=True)
        dts, sts = _ts_set(d), _ts_set(s)

        checks = [
            ("default: in-card UI EXEMPT (15.0 not flagged)", IN_WINDOW_TS not in dts),
            ("default: out-of-card UI FLAGGED (50.0 flagged)", OUT_WINDOW_TS in dts),
            ("default verdict reflects the lone out-of-card flag", d.get("verdict") == "FAIL_RELEVANCE_QA"),
            ("default: exactly one flag (the footage-beat UI)", len(d.get("flags", [])) == 1),
            ("strict: in-card UI FLAGGED (15.0 flagged)", IN_WINDOW_TS in sts),
            ("strict: out-of-card UI FLAGGED (50.0 flagged)", OUT_WINDOW_TS in sts),
            ("strict: both frames flagged", len(s.get("flags", [])) == 2),
            ("card windows resolved (=1)", d.get("card_windows") == 1),
        ]
        for name, ok in checks:
            print(f"  [{'OK' if ok else 'FAIL'}] {name}")
            if not ok:
                failures.append(name)
    finally:
        _restore()
        os.environ.pop("VIDLORE_QA_CARD_UI_STRICT", None)

    print(f"\nRESULT: {'ALL GREEN' if not failures else 'FAIL'}  ({len(failures)} failure(s))")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
