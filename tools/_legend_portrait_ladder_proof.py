#!/usr/bin/env python3
"""V3.0.1 — portrait fallback-ladder PROOF (micro-render).

Exercises footage.resolve_legend_portrait + portrait_legend_reveal.render across
the three required cases and saves a real rendered frame for each:

  A  real portrait available  → real validated portrait used (monogram NOT used)
  B  no portrait available     → premium monogram pedestal, fallback reason logged
  C  bad candidate portrait    → wrong face rejected, falls through, reason logged

Frames + the resolved portrait images land under
research/motion_graphics_expansion/v301_portrait_polish/frames/.
Heavy MP4s are deleted after the frame is extracted (disk-safety); PNGs kept.
"""
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VIDLORE_REAL_PERSON", "1")
os.environ.setdefault("VIDLORE_WIKIMEDIA", "1")

from vidlore import footage                                    # noqa: E402
from vidlore.motion_graphics.biography import portrait_legend_reveal as PLR  # noqa: E402
from vidlore.ffmpeg_tool import ffmpeg_exe                     # noqa: E402
from PIL import Image                                          # noqa: E402
import numpy as np                                             # noqa: E402

OUT = ROOT / "research/motion_graphics_expansion/v301_portrait_polish/frames"
OUT.mkdir(parents=True, exist_ok=True)


def _mid_frame(mp4: Path, png: Path) -> float:
    """Extract the middle frame → png; return mean luma 0..1. Deletes the mp4."""
    ff = ffmpeg_exe()
    subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-ss", "1.4",
                    "-i", str(mp4), "-frames:v", "1", str(png)],
                   capture_output=True)
    mp4.unlink(missing_ok=True)
    if not png.exists():
        return -1.0
    a = np.asarray(Image.open(png).convert("L"), dtype=np.float32) / 255.0
    return round(float(a.mean()), 3)


def _render(case: str, name: str, portrait_path, kicker: str) -> dict:
    td = Path(tempfile.mkdtemp())
    mp4 = td / f"{case}.mp4"
    res = PLR.render(str(mp4), name=name, portrait_path=portrait_path,
                     kicker=kicker, dur=3.0, fps=24, w=1280, h=720, seed=7)
    png = OUT / f"case_{case}.png"
    luma = _mid_frame(mp4, png) if res.get("ok") else -1.0
    return {"ok": res.get("ok"), "luma": luma, "frame": str(png)}


def main():
    print("=" * 64)
    # ── CASE A — real portrait available ──────────────────────────────
    tdA = Path(tempfile.mkdtemp())
    rA = footage.resolve_legend_portrait(
        "John D. Rockefeller", role="industrialist",
        narration="John D. Rockefeller was born in 1839 and founded Standard Oil.",
        dest=tdA / "a.jpg", cache_dir=tdA, allow_ai=False)
    # keep the resolved portrait as proof
    if rA["portrait_path"] and Path(rA["portrait_path"]).exists():
        Image.open(rA["portrait_path"]).save(OUT / "case_A_resolved_portrait.png")
    renA = _render("A", "John D. Rockefeller", rA["portrait_path"],
                   "THE FIRST BILLIONAIRE")
    print(f"[A real]    type={rA['portrait_source_type']:22s} "
          f"score={rA['portrait_validation_score']} "
          f"path={bool(rA['portrait_path'])} | render ok={renA['ok']} "
          f"luma={renA['luma']} | reason={rA['portrait_fallback_reason'] or 'ok'}")
    assert rA["portrait_source_type"] in ("archival_public_domain", "validated_web"), \
        "Case A must use a REAL portrait"
    assert rA["portrait_path"], "Case A must have a portrait_path"

    # ── CASE B — no portrait available (fictional subject), AI off ────
    tdB = Path(tempfile.mkdtemp())
    rB = footage.resolve_legend_portrait(
        "Aldric Thornwald the Third", role="",
        narration="Aldric Thornwald the Third was a wholly fictional emperor of nowhere.",
        dest=tdB / "b.jpg", cache_dir=tdB, allow_ai=False)
    renB = _render("B", "Aldric Thornwald the Third", rB["portrait_path"],
                   "EMPEROR OF NOWHERE")
    print(f"[B none]    type={rB['portrait_source_type']:22s} "
          f"score={rB['portrait_validation_score']} "
          f"path={bool(rB['portrait_path'])} | render ok={renB['ok']} "
          f"luma={renB['luma']} | reason={rB['portrait_fallback_reason']}")
    assert rB["portrait_source_type"] == "monogram_pedestal", "Case B must be monogram"
    assert not rB["portrait_path"], "Case B must NOT have a portrait"
    assert rB["portrait_fallback_reason"], "Case B must log a fallback reason"

    # ── CASE C — bad candidate portrait (wrong face rejected) ─────────
    # Simulate the existing _real_person_image gates rejecting an unrelated face:
    # the resolver must surface the rejection reason and fall through to monogram.
    _orig = footage._real_person_image
    def _reject(name, role, dest, *, gender="unknown", cache_dir=None):
        footage._PORTRAIT_PROVENANCE[name] = {
            "person": name, "source": None,
            "validator": "rejected", "name_match": 0.18,
            "fallback_reason": "validator rejected: wrong face (unrelated_subject)"}
        return None
    footage._real_person_image = _reject
    try:
        tdC = Path(tempfile.mkdtemp())
        rC = footage.resolve_legend_portrait(
            "Marcus Vellan", role="",
            narration="Marcus Vellan — a candidate whose only image hits are an unrelated face.",
            dest=tdC / "c.jpg", cache_dir=tdC, allow_ai=False)
    finally:
        footage._real_person_image = _orig
    renC = _render("C", "Marcus Vellan", rC["portrait_path"], "DISPUTED IDENTITY")
    print(f"[C reject]  type={rC['portrait_source_type']:22s} "
          f"score={rC['portrait_validation_score']} "
          f"path={bool(rC['portrait_path'])} | render ok={renC['ok']} "
          f"luma={renC['luma']} | reason={rC['portrait_fallback_reason']}")
    assert rC["portrait_source_type"] == "monogram_pedestal", "Case C must fall to monogram"
    assert not rC["portrait_path"], "Case C must reject the wrong face (no portrait)"
    assert "reject" in (rC["portrait_fallback_reason"] or "").lower(), \
        "Case C must log the rejection reason"

    print("=" * 64)
    print("LADDER PROOF: all 3 cases PASS. Frames →", OUT)


if __name__ == "__main__":
    main()
