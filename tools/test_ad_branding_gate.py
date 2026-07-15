#!/usr/bin/env python3
"""Final-video ad/branding release gate — end-to-end proof.

Builds a 6s video whose first half is photographic (mandelbrot) and second half is a flat
Max-style promo CARD, then drives the real gate with a MOCK OCR engine (deterministic, no
RapidOCR dependency). Proves the TWO-FACTOR rule: the mock returns promo text on EVERY frame, yet
only the card frames are flagged — the photographic frames are skipped by the card-geometry
pre-filter, so an in-scene sign / caption / small bug would never trip the gate.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.build import (                          # noqa: E402
    _final_video_ad_scan, _final_video_ad_gate, _frame_card_uniformity,
    _PROMO_RX, _branding_probe_offsets)
from vidlore.clipstudio.config import ffmpeg_exe                # noqa: E402

FF = ffmpeg_exe()
PASS = FAIL = 0


def _say(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


class MockOCR:
    """Callable like RapidOCR: returns (results, elapsed). `text` on every frame; `box` controls
    layout (big box = a designed card / overlay; small box = an in-scene sign)."""
    def __init__(self, text, box=None):
        self.text = text
        self.box = box or [[500, 320], [640, 320], [640, 350], [500, 350]]   # small in-scene sign

    def __call__(self, img_path):
        return ([(self.box, self.text, 0.9)], 0.0)


def _build_test_video(dest: Path) -> None:
    td = dest.parent
    scene = td / "_scene.mp4"
    card = td / "_card.mp4"
    # photographic half: detailed fractal (low center uniformity)
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(scene)], check=True,
                   capture_output=True)
    # promo card half: flat blue slate with big centered promo text
    txt = "max  THE ONE TO WATCH"
    vf = (f"drawtext=fontfile=/System/Library/Fonts/Helvetica.ttc:text='{txt}':"
          f"fontsize=64:fontcolor=white:x=(w-tw)/2:y=(h-th)/2")
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=0x1030A0:s=1280x720:rate=30", "-t", "3",
                    "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(card)], check=True,
                   capture_output=True)
    concat = td / "_concat.txt"
    concat.write_text(f"file '{scene.name}'\nfile '{card.name}'\n")
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-c", "copy", str(dest)], check=True, capture_output=True)


def main():
    import tempfile
    # (1) _PROMO_RX matches promo/CTA language, NOT ordinary narration
    for s in ["max PLANS START AT $9.99/MONTH mox.com", "Subscribe for more",
              "only on HBO Max", "www.channel.tv", "Now Streaming", "discover more"]:
        _say(bool(_PROMO_RX.search(s)), f"_PROMO_RX flags promo string: {s!r}")
    for s in ["watch how she moves through the hall", "follow her journey to Braavos",
              "the dragons happened", "he catches her by the throat", "a Lannister always pays"]:
        _say(not _PROMO_RX.search(s), f"_PROMO_RX ignores narration: {s!r}")

    # (2) _branding_probe_offsets covers the TAIL of a clip
    offs = _branding_probe_offsets(8.0)
    _say(any(o >= 6.0 for o in offs) and max(offs) >= 7.0,
         f"branding probe samples the clip TAIL (offsets ...{[o for o in offs if o >= 5][-3:]})")
    _say(len(_branding_probe_offsets(0)) == 5, "unknown-duration falls back to legacy head grid")

    # (3) _frame_card_uniformity: flat card high, photographic low (PIL only, no ffmpeg)
    try:
        from PIL import Image
        import numpy as np
        flat = Image.new("RGB", (1280, 720), (16, 48, 160))
        rnd = Image.fromarray((np.random.RandomState(7).rand(720, 1280, 3) * 255).astype("uint8"))
        fp1, fp2 = Path(tempfile.mktemp(suffix=".png")), Path(tempfile.mktemp(suffix=".png"))
        flat.save(fp1); rnd.save(fp2)
        u_flat, u_rnd = _frame_card_uniformity(fp1), _frame_card_uniformity(fp2)
        fp1.unlink(missing_ok=True); fp2.unlink(missing_ok=True)
        _say(u_flat >= 0.9, f"flat card reads as designed CARD (uniformity {u_flat:.2f} >= 0.9)")
        _say(u_rnd < 0.2, f"photographic frame reads as scene (uniformity {u_rnd:.2f} < 0.2)")
    except Exception as e:
        _say(False, f"card-uniformity PIL test errored: {e}")

    # (4) PATH A (flat card): scene+card video, mock returns promo text in a SMALL box (in-scene
    #     sign). Only the flat-card half is flagged; the photographic half with a small sign is NOT.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        work = td / "output" / "work"
        work.mkdir(parents=True)
        vid = work / "final.mp4"
        _build_test_video(vid)
        ocr_sign = MockOCR("max THE ONE TO WATCH $9.99/month mox.com")   # small box = in-scene sign
        r = _final_video_ad_scan(vid, work, ocr_sign, stride=0.5)
        hits = r["hits"]
        _say(r["status"] == "blocked" and len(hits) >= 3,
             f"path A: flat promo card flagged ({len(hits)} hits, status={r['status']})")
        _say(all(h["t"] >= 2.5 for h in hits),
             f"path A: only the flat-card half flagged, NOT the photographic frame with a small "
             f"in-scene sign (hit times {[h['t'] for h in hits]})")
        raised = False
        try:
            _final_video_ad_gate(vid, work, ocr_sign, log=lambda m: None)
        except RuntimeError:
            raised = True
        _say(raised and not vid.exists()
             and vid.with_name(vid.stem + ".FAILED_AD_QA.mp4").exists(),
             "gate RAISES + QUARANTINES on a surviving promo card")

        # (5) PATH B (image-backed non-uniform card): photographic video + BIG promo text box that
        #     persists → confirmed even though the background is not flat.
        vidb = work / "finalB.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vidb)], check=True,
                       capture_output=True)
        ocr_big = MockOCR("SUBSCRIBE now at www.channel.tv only on HBO Max",
                          box=[[120, 120], [1160, 120], [1160, 300], [120, 300]])   # big overlay
        rb = _final_video_ad_scan(vidb, work, ocr_big, stride=0.5)
        _say(rb["status"] == "blocked" and rb["hits"],
             f"path B: image-backed non-uniform promo overlay flagged ({len(rb['hits'])} hits)")

        # (6) FAIL-CLOSED: no OCR engine → unverified → gate blocks (unless override)
        vid2 = work / "final2.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid2)], check=True,
                       capture_output=True)
        r_none = _final_video_ad_scan(vid2, work, None)
        _say(r_none["status"] == "unverified", "no OCR engine → status 'unverified' (not clean)")
        import os as _os
        blocked = False
        try:
            _final_video_ad_gate(vid2, work, None, log=lambda m: None)
        except RuntimeError:
            blocked = True
        _say(blocked, "fail-closed: no-OCR gate BLOCKS (does not silently pass)")
        # restore the quarantined file, then test the explicit override
        _q = vid2.with_name(vid2.stem + ".FAILED_AD_QA.mp4")
        if _q.exists():
            _q.rename(vid2)
        _os.environ["VIDLORE_CLIPSTUDIO_AD_GATE_OVERRIDE"] = "1"
        try:
            out = _final_video_ad_gate(vid2, work, None, log=lambda m: None)
            _say(out == vid2 and vid2.exists(), "explicit override publishes with a loud warning")
        finally:
            _os.environ.pop("VIDLORE_CLIPSTUDIO_AD_GATE_OVERRIDE", None)

        # (7) a clean video with benign OCR (real card-geometry, benign text) passes untouched
        vid3 = work / "final3.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid3)], check=True,
                       capture_output=True)
        benign = MockOCR("the dragons already proved it",
                         box=[[120, 120], [1160, 120], [1160, 300], [120, 300]])
        out = _final_video_ad_gate(vid3, work, benign, log=lambda m: None)
        _say(out == vid3 and vid3.exists(), "clean video (benign text, no promo token) passes")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
