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
    _PROMO_RX, _branding_probe_offsets, _parse_srt_events, _parse_ass_events,
    _caption_explained, _norm_caption_words, _own_caption_schedule)
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


class MockOCRMulti:
    """Multi-box mock: `items` = [(box, text)] returned with conf 0.9 on every frame."""
    def __init__(self, items):
        self.items = items

    def __call__(self, img_path):
        return ([(box, text, 0.9) for box, text in self.items], 0.0)


# A caption-style box: single, wide, in the caption band — big enough that max_frac >= 0.04
# trips the layout_heavy path exactly like the observed production false positive (a large
# Professional-style caption reading the narration's OWN "subscribe…" outro line).
CAP_BOX = [[220, 600], [1060, 600], [1060, 660], [220, 660]]
CAP_LINE = "— subscribe because that's the story"


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

        # (8) R4-2 TAIL fixture: a promo card ONLY in the FINAL 1.0s must be reached and BLOCKED
        scene5 = td / "_s5.mp4"; card1 = td / "_c1.mp4"; tailvid = work / "tail.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "5",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(scene5)], check=True,
                       capture_output=True)
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=0x1030A0:s=1280x720:rate=30", "-t", "1",
                        "-vf", "drawtext=fontfile=/System/Library/Fonts/Helvetica.ttc:"
                               "text='max THE ONE TO WATCH':fontsize=64:fontcolor=white:"
                               "x=(w-tw)/2:y=(h-th)/2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(card1)], check=True, capture_output=True)
        _lst = td / "_tl.txt"; _lst.write_text(f"file '{scene5.name}'\nfile '{card1.name}'\n")
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(_lst), "-c", "copy", str(tailvid)], check=True, capture_output=True)
        r_tail = _final_video_ad_scan(tailvid, work, MockOCR("max THE ONE TO WATCH"), stride=0.5)
        _say(r_tail["status"] == "blocked" and any(h["t"] >= 4.8 for h in r_tail["hits"]),
             f"promo card in the FINAL 1.0s is reached + BLOCKED (hits {[h['t'] for h in r_tail['hits']]})")

        # ── (9) OWN-CAPTION whitelist — the production false positive, reproduced end-to-end ──
        # The narration's OWN outro line ("…subscribe because that's the story") is burned as a
        # caption; the OCR sees it in a big caption box (max_frac >= 0.04 → layout_heavy). WITHOUT
        # the schedule the gate blocks; WITH the schedule (captions_burned=True + final.srt) it
        # must pass — and a real promo card at the same instant must STILL block.
        vid9 = work / "final9.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid9)], check=True,
                       capture_output=True)
        ocr_cap = MockOCR(CAP_LINE, box=CAP_BOX)
        r_nocap = _final_video_ad_scan(vid9, work, ocr_cap, stride=0.5)
        _say(r_nocap["status"] == "blocked",
             "REGRESSION GUARD: without a caption schedule the 'subscribe' caption still blocks "
             "(whitelist only ever applies when captions were burned)")
        # now provide the schedule: final9.srt covering the whole clip with that exact line
        vid9.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:03,500\n— subscribe because that's the story\n\n",
            encoding="utf-8")
        own9 = _own_caption_schedule(vid9, work)
        _say(len(own9) == 1 and "subscribe" in own9[0][2],
             f"_own_caption_schedule loads the srt next to the video ({own9[0][2] if own9 else []})")
        r_cap = _final_video_ad_scan(vid9, work, ocr_cap, stride=0.5, own_captions=own9)
        _say(r_cap["status"] == "clean",
             f"OWN caption 'subscribe…' line is whitelisted by text+time → clean "
             f"(status={r_cap['status']})")
        out9 = _final_video_ad_gate(vid9, work, ocr_cap, log=lambda m: None, captions_burned=True)
        _say(out9 == vid9 and vid9.exists(),
             "gate with captions_burned=True publishes the own-caption CTA render")

        # (10) same caption text but scheduled at a DIFFERENT time → NOT whitelisted → blocked
        vid10 = work / "final10.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid10)], check=True,
                       capture_output=True)
        vid10.with_suffix(".srt").write_text(
            "1\n00:01:40,000 --> 00:01:43,000\n— subscribe because that's the story\n\n",
            encoding="utf-8")
        r10 = _final_video_ad_scan(vid10, work, ocr_cap, stride=0.5,
                                   own_captions=_own_caption_schedule(vid10, work))
        _say(r10["status"] == "blocked",
             "same text at the WRONG time (no caption scheduled now) still blocks")

        # (11) a REAL promo card while a benign caption is active → blocked (text mismatch)
        vid11 = work / "final11.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid11)], check=True,
                       capture_output=True)
        vid11.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:03,500\nhe rides for the capital at dawn\n\n",
            encoding="utf-8")
        ocr_promo = MockOCR("SUBSCRIBE start your free trial at www.maxx.com", box=CAP_BOX)
        r11 = _final_video_ad_scan(vid11, work, ocr_promo, stride=0.5,
                                   own_captions=_own_caption_schedule(vid11, work))
        _say(r11["status"] == "blocked",
             "third-party promo text over an active (different) caption still blocks")

        # (12) MIXED frame: our caption box + a separate real promo overlay → still blocked
        ocr_mixed = MockOCRMulti([
            (CAP_BOX, CAP_LINE),                                        # our caption (whitelisted)
            ([[150, 100], [1130, 100], [1130, 320], [150, 320]],        # big promo overlay
             "PLANS START AT $9.99/month  only on HBO Max"),
        ])
        r12 = _final_video_ad_scan(vid9, work, ocr_mixed, stride=0.5, own_captions=own9)
        _say(r12["status"] == "blocked",
             "mixed frame: caption whitelisted but the co-present promo overlay still blocks")

        # (13) breakout ASS lines join the schedule (karaoke tags stripped)
        (work / "breakout_caps.ass").write_text(
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:02.50,BK,,0,0,0,,"
            "{\\fad(120,120)}{\\kf46}subscribe {\\kf14}to {\\kf16}nothing, {\\kf16}ser\n",
            encoding="utf-8")
        vid9.with_suffix(".srt").unlink()                # isolate: ASS is now the only schedule
        own13 = _own_caption_schedule(vid9, work)
        _say(len(own13) == 1 and own13[0][2] == ["subscribe", "to", "nothing", "ser"],
             f"breakout ASS parsed into the schedule (tags stripped: {own13[0][2] if own13 else []})")
        ocr_bk = MockOCR("subscribe to nothing, ser", box=CAP_BOX)
        r13 = _final_video_ad_scan(vid9, work, ocr_bk, stride=0.5, own_captions=own13)
        _say(r13["status"] == "clean",
             "breakout word-by-word caption line with a CTA word is whitelisted → clean")
        (work / "breakout_caps.ass").unlink()

        # (14) _caption_explained unit edges: OCR noise tolerated, subset NOT over-matched
        act = [_norm_caption_words("— subscribe because that's the story coming next")]
        _say(_caption_explained("- subscrlbe because thats the story", act),
             "OCR noise ('subscrlbe', dropped apostrophe, '-' for em-dash) still explained")
        _say(not _caption_explained("subscribe now free trial", act),
             "promo phrasing sharing ONE caption word is NOT explained (80% word coverage)")
        _say(not _caption_explained("", act) and not _caption_explained("subscribe", []),
             "empty OCR text / empty schedule never explain")
        # REVERSE coverage (review finding): a box whose words are a SUBSET of the caption's
        # must cover >= min(3, len(event)) event words — a lone 'SUBSCRIBE' end-card button
        # sharing one word with the active caption is NEVER explained.
        _say(not _caption_explained("SUBSCRIBE", act),
             "single-word 'SUBSCRIBE' box is NOT explained by a caption containing the word")
        _say(not _caption_explained("subscribe story", act),
             "two-word subset of the caption is NOT explained (reverse coverage < 3)")
        _say(_caption_explained("subscribe because that's", act),
             "a full caption ROW (3+ event words) IS explained (two-row renders keep working)")
        _say(not _caption_explained("subscribe now story", act),
             "ceil(80%) forward coverage: 2/3 words matched is NOT explained")

        # (15) END-TO-END bypass regression: a persistent full-screen 'SUBSCRIBE' end-card
        # while our own 'subscribe…' caption is active must STILL block — text+time whitelist,
        # not a token amnesty.
        vid15 = work / "final15.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid15)], check=True,
                       capture_output=True)
        vid15.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:03,500\n— subscribe because that's the story\n\n",
            encoding="utf-8")
        ocr_button = MockOCR("SUBSCRIBE",
                             box=[[340, 250], [940, 250], [940, 420], [340, 420]])  # giant button
        r15 = _final_video_ad_scan(vid15, work, ocr_button, stride=0.5,
                                   own_captions=_own_caption_schedule(vid15, work))
        _say(r15["status"] == "blocked",
             "full-screen 'SUBSCRIBE' end-card during our own subscribe-caption still BLOCKS")

        # (16) digit-only caption cue survives SRT parsing (a '1942' caption must stay in the
        # whitelist schedule; only pre-timestamp index lines are skipped)
        srt16 = work / "digits.srt"
        srt16.write_text("1\n00:00:01,000 --> 00:00:02,000\n1942\n\n"
                         "2\n00:00:03,000 --> 00:00:04,000\nthe war begins\n\n", encoding="utf-8")
        ev16 = _parse_srt_events(srt16)
        _say(len(ev16) == 2 and ev16[0][2] == "1942",
             f"digit-only cue text ('1942') is kept by the SRT parser ({[e[2] for e in ev16]})")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
