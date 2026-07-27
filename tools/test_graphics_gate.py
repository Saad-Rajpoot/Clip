#!/usr/bin/env python3
"""Non-show graphics gate + freeze-repair donor + verifier hard rule — regression proof.

The leaks this pins against (one real render, 3 beats): a teal sports-news CGI intro aired on an
abstract-effect beat (the vision verifier called it "suitable for a general transition"); the
near-black freeze-repair then PROPAGATED that CGI frame into the next scene; and Jaime/Cersei
fan art aired via an unverified secondary beat window from a 41%-illustrated book-essay source.
Fix layers: (1) tiered per-shot graphics verdict from the persisted CLIP embedding, with a
source-level drop for illustrated/parody uploads; (2) same-scene freeze donors; (3) a non-show
HARD RULE in every verifier rung."""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio import index as IX                       # noqa: E402
from vidlore.clipstudio.match import (                           # noqa: E402
    _shot_graphics_tier, _shot_is_graphics, _graphics_source_verdict, _shot_dirty_reason)
from vidlore.clipstudio.models import Shot                       # noqa: E402

PASS = FAIL = 0


def _say(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def _sh(**kw):
    d = dict(index=0, start=0.0, end=2.0, subs_flag=0, luma_avg=120.0, luma_hi=200.0,
             ocr_text="", corner_masks={}, graphics_flag=-1, keyframe_path="", quality=0.8)
    d.update(kw)
    return types.SimpleNamespace(**d)


def main():
    # Inject tiny deterministic anchor matrices: axes = (graphic, realphoto, photo, art)
    saved = IX._GFX_MATS[0]
    IX._GFX_MATS[0] = (np.array([[1, 0, 0, 0]], "float32"), np.array([[0, 1, 0, 0]], "float32"),
                       np.array([[0, 0, 1, 0]], "float32"), np.array([[0, 0, 0, 1]], "float32"))
    try:
        v_hard = np.array([0.5, 0.4, 0.0, 0.0], "float32")       # gdom 0.10 > 0.036
        v_band_art = np.array([0.5, 0.48, 0.10, 0.30], "float32")  # gdom 0.02; art > photo
        v_band_photo = np.array([0.5, 0.48, 0.30, 0.10], "float32")  # gdom 0.02; photo wins
        v_clean = np.array([0.3, 0.5, 0.4, 0.1], "float32")      # gdom negative

        # (1) tier semantics
        _say(IX.graphics_flag_of(v_hard, 120.0) == 2, "hard graphics → tier 2")
        _say(IX.graphics_flag_of(v_band_art, 120.0) == 1, "band + art-verdict → tier 1")
        _say(IX.graphics_flag_of(v_band_photo, 120.0) == 0, "band + photo-verdict → tier 0")
        _say(IX.graphics_flag_of(v_clean, 120.0) == 0, "clean embedding → tier 0")
        _say(IX.graphics_flag_of(v_hard, 10.0) == 0,
             "near-black luma guard → never judged graphics (unreadable gates own it)")
        _say(IX.graphics_flag_of(None) == -1, "no embedding → -1 (cannot judge)")

        # (2) persisted-first + live fallback
        _say(_shot_graphics_tier(_sh(graphics_flag=2)) == 2, "persisted hard tier wins (no model)")
        _say(_shot_graphics_tier(_sh(graphics_flag=0), v_hard) == 0,
             "persisted 0 short-circuits (no recompute)")
        _say(_shot_graphics_tier(_sh(), v_hard) == 2, "old index + embed → computed live")
        _say(_shot_graphics_tier(_sh()) == -1, "old index + no embed → -1 (fail-open)")
        _say(_shot_is_graphics(_sh(), v_band_art) is False,
             "band tier alone NEVER hard-gates a shot (the drawbridge-aerial live FP)")
        _say(_shot_is_graphics(_sh(), v_hard) is True, "hard tier gates")

        # (3) source-level verdict — calibrated fractions from the real 50-source render
        for name, n, tot, want in (("cartoon 88/93", 88, 93, True), ("parody 51/72", 51, 72, True),
                                   ("book-essay 17/41", 17, 41, True), ("news 4/14", 4, 14, True),
                                   ("clean-worst 3/49", 3, 49, False), ("tiny 1/1", 1, 1, False),
                                   ("two-shots 2/5", 2, 5, False)):
            _say(_graphics_source_verdict(n, tot) is want, f"source verdict {name} → drop={want}")

        # (4) window-QC dirty reason: persisted HARD tier only; subs precedence intact
        _say(_shot_dirty_reason(_sh(graphics_flag=2)) == "graphics",
             "persisted hard tier reads as dirty 'graphics' in window-QC")
        _say(_shot_dirty_reason(_sh(graphics_flag=1)) == "",
             "persisted band tier is NOT dirty without source context")
        _say(_shot_dirty_reason(_sh(graphics_flag=2, subs_flag=1)) == "subs",
             "subs precedence unchanged")
    finally:
        IX._GFX_MATS[0] = saved

    # (5) Shot model round-trip
    s = Shot(source_id="s", index=1, start=0.0, end=2.0, graphics_flag=2)
    _say(Shot.from_dict(s.to_dict()).graphics_flag == 2, "graphics_flag survives round-trip")
    _say(Shot.from_dict({"source_id": "s", "index": 1, "start": 0, "end": 2}).graphics_flag == -1,
         "old shots.json without the field loads as -1 (unknown)")

    # (6) review-hardening behaviors (adversarial-review findings, all confirmed on real code)
    import os
    from vidlore.clipstudio.match import _score_pool  # noqa: F401  (import proves availability)
    # kill switch: GRAPHICS_GATE=0 must disable the persisted-hard dirty reason too
    os.environ["VIDLORE_CLIPSTUDIO_GRAPHICS_GATE"] = "0"
    try:
        _say(_shot_dirty_reason(_sh(graphics_flag=2)) == "",
             "kill switch: GRAPHICS_GATE=0 disables the window-QC graphics dirty reason")
    finally:
        os.environ.pop("VIDLORE_CLIPSTUDIO_GRAPHICS_GATE", None)
    _say(_shot_dirty_reason(_sh(graphics_flag=1), band_graphics=True) == "graphics-band",
         "window-QC: band tier dirties WITH source context (mirrors the pool rule)")
    _say(_shot_dirty_reason(_sh(graphics_flag=1), band_graphics=False) == "",
         "window-QC: band tier stays clean without context")
    # live-fallback luma: a legitimate 0.0 luma_avg must GUARD (not collapse to 'unknown')
    saved2 = IX._GFX_MATS[0]
    IX._GFX_MATS[0] = (np.array([[1, 0, 0, 0]], "float32"), np.array([[0, 1, 0, 0]], "float32"),
                       np.array([[0, 0, 1, 0]], "float32"), np.array([[0, 0, 0, 1]], "float32"))
    try:
        _vh = np.array([0.5, 0.4, 0.0, 0.0], "float32")
        _say(_shot_graphics_tier(_sh(luma_avg=0.0), _vh) == 0,
             "pure-black shot (luma_avg 0.0) is GUARDED, not judged as graphics")
    finally:
        IX._GFX_MATS[0] = saved2

    # (7) wiring fingerprints
    root = Path(__file__).resolve().parents[1] / "vidlore/clipstudio"
    msrc = (root / "match.py").read_text()
    _say("_band_armed = _n_hard >= 3" in msrc,
         "pool: band tier armed only by SOLID hard evidence (>=3 — one title card can't arm it)")
    _say("_graphics_source_verdict(_n_hard, len(shots))" in msrc,
         "pool: source drop counts HARD shots only (band FPs can't kill a real source)")
    _say("dropping illustrated/graphics source" in msrc,
         "pool: illustrated/parody source dropped wholesale with a log line")
    _say("ggate_on and _shot_is_graphics(ps.shot" in msrc,
         "_score_pool mirror keyed on the GRAPHICS gate env (kill switch works)")
    _say("_persist_graphics_flags(proj, src.id, shots)" in msrc,
         "pool: computed tiers written back to shots.json (old-index backfill)")
    bsrc = (root / "build.py").read_text()
    _say("_own_clean[0] if _own_clean else _last_clean_d" in bsrc
         and "PREVIOUS-scene" in bsrc,
         "freeze-repair prefers a SAME-scene donor; cross-scene donation is loud")
    _say("_is_synth" in bsrc and '"_nobrand" in _s or "_nodark" in _s' in bsrc,
         "freeze-repair: synthesized freezes/placeholders are never donors (no silent "
         "cross-scene propagation relabelled as same-scene)")
    _say("_own_ok = [cp for cp, _bf in zip(clips, _brand_flags) if not _bf]" in bsrc,
         "branding-card pass also prefers a same-scene donor")
    _say('int(getattr(sh, "graphics_flag", -1) or -1) >= 2' in bsrc,
         "build replay-walk and breakout gating repeat the graphics verdict")
    isrc = (root / "image_fallback.py").read_text()
    _say('int(getattr(shot, "graphics_flag", -1) or -1) >= 2' in isrc,
         "both source-frame still pickers reject hard-tier graphics (shared guard)")
    osrc = (root / "orchestrate.py").read_text()
    _say('"gatev2-graphics"' in osrc,
         "resume signature folds the gate version (cached pre-gate selections never replay)")
    vsrc = (root / "verify.py").read_text()
    # asserted on the INTENT (_nonshow reaches every rung) rather than the exact concatenation:
    # the prompt gains blocks over time (_look, the instructed-target question, was added between
    # _story and _mf) and a literal match fails on an unrelated, correct change.
    _prompt_line = next((ln for ln in vsrc.splitlines() if "+ _rule + _nonshow" in ln), "")
    _say("HARD RULE" in vsrc and "news/broadcast motion graphics" in vsrc
         and bool(_prompt_line) and "_venue" in _prompt_line,
         "verifier: non-show hard rule in EVERY rung's prompt")
    _say('PROMPT_VERSION = "v7' in vsrc,
         "verifier prompt version bumped (cached verdicts from v6 never reused)")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
