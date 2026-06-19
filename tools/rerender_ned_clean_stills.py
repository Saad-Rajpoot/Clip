#!/usr/bin/env python3
"""Surgical re-render of the Ned Stark project: the footage match/cut/verify are CORRECT and stay
untouched — only the image-recovery stills were wrong (20 reactor/essay frames leaked into the
still-recovery pool). Clear ALL image assignments, re-run the (now source-gated + OCR-guarded)
image-fallback, and re-assemble. No LLM re-verify (keeps the exact same, audited cut; fast + free).

Usage: python3 tools/rerender_ned_clean_stills.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# image-fallback + build are fully local (CLIP/Face-ID/ffmpeg) — no API needed. Disable the web pass
# (the original render produced 0 web stills anyway; keeps this offline + fast + identical-in-effect).
os.environ["VIDLORE_CLIPSTUDIO_WEB_IMAGE_FALLBACK"] = "0"

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.analyze import ScriptAnalysis
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.build import build_video
from vidlore.clipstudio import faceid as _faceid
from vidlore.clipstudio import ledger, review as _review
from vidlore.clipstudio.orchestrate import _fill_image_fallbacks

PROJ = Path.home() / "Desktop" / "clipstudio_output" / "portal" / "ned_stark_baelor"
VO = "/Users/hussnain/Desktop/Ned Stark's Final Words Hid a Message Nobody Caught.mp3"
TITLE = "Ned Stark's Final Words Hid a Message Nobody Caught"


def log(m):
    print(m, flush=True)


cfg = load_clip_config()
proj = ClipProject.load(PROJ)
proj.ensure_dirs()
analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
segs = proj.segments

# --- clear ALL stale image assignments so every still is re-picked from the gated pool ---
cleared = 0
for s in proj.selections:
    if getattr(s, "image_path", "") or getattr(s, "image_meta", None):
        s.image_path = ""
        s.image_meta = {}
        cleared += 1
log(f"loaded: {len(segs)} beats · {len(proj.selections)} selections · cleared {cleared} stale still(s)")

log("· build Face-ID references")
faceid_obj, refs = None, {}
if _faceid.available():
    faceid_obj = _faceid.FaceID()
    refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                    faceid_obj, progress=log)

log("· image fallback (source-gated + OCR-guarded pool)")
n = _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log)
log(f"· image fallback filled {n} beat(s)")

ledger.finalize(proj, segs, cfg)
_review.write_review(proj, segs)
proj.save()

log("· assemble final video")
out = build_video(proj, segs, cfg, captions=True, title=TITLE,
                  theme_name="history", voiceover=VO, use_tts=True, progress=log)
log(f"DONE → {out}")
