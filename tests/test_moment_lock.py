"""DIALOGUE-BASED MOMENT MATCHING — when the narration talks about a line, the cut must land on it.

Regression this locks in (job 69d80e9dd4, audited 2026-07-26):
  * `find_quote_span` — the primitive that locates a line in a source's CONTINUOUS ASR word stream —
    was wired only into breakout selection. The match path used `_dialogue_match` against per-shot
    transcripts, which index._assign_transcript bins by MIDPOINT, so a line spoken across a cut
    belongs to no shot at all (index.py documents this). Result: w_dialogue is the heaviest weight in
    the scorer at 0.55 and it fired on 20 of 85 quoted beats (23.5%), mean 0.105.
  * Even when the right SHOT was picked, `_trim_window` centred the cut on the shot MIDPOINT, so a
    4.9s shot holding a line at [146.8-147.6] was cut [144.5-147.0] — right shot, wrong instant.
  * `_clean_copy_swap` re-derived the window when moving to a cleaner copy of the scene and dropped
    the moment, putting the cut back on the midpoint of a different source.

    python3 tests/test_moment_lock.py

No network, no model, no ffmpeg.
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import match as M                    # noqa: E402
from vidlore.clipstudio import index as IX                   # noqa: E402
from vidlore.clipstudio.config import ClipConfig             # noqa: E402
from vidlore.clipstudio.models import Shot                   # noqa: E402

FAILS = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


LINE = "I did what I did to protect Sansa"


def _words(text, t0=0.0, step=0.4):
    """Fake a word stream: [(start, end, word)] like index.load_words returns."""
    out, t = [], t0
    for w in text.split():
        out.append((round(t, 2), round(t + step * 0.8, 2), w))
        t += step
    return out


# ---------------------------------------------------------------------------
# the locator
# ---------------------------------------------------------------------------
def test_span_locates_the_line_in_a_word_stream():
    stream = _words("some earlier chatter here " + LINE + " and then later talk", t0=10.0)
    sp = IX.find_quote_span(stream, LINE)
    check("find_quote_span locates the line", sp is not None)
    if sp:
        check("located span starts where the line starts", 11.4 <= sp[0] <= 12.2)
        check("located span reports a high ratio", sp[2] >= 0.9)


def test_span_is_memoized_and_survives_a_missing_source():
    M._QSPAN_CACHE.clear()
    proj = NS(meta={}, index_dir=None, root="/nonexistent")
    got = M.quote_span_in_source(proj, "no_such_source", LINE)
    check("a source with no word stream returns None (never raises)", got is None)
    check("the None result is cached", ("no_such_source", " ".join(M._norm_words(LINE)))
          in M._QSPAN_CACHE)
    check("an empty quote is a no-op", M.quote_span_in_source(proj, "x", "") is None)


# ---------------------------------------------------------------------------
# proximity
# ---------------------------------------------------------------------------
def test_proximity_prefers_the_shot_holding_the_line():
    span = (100.0, 102.0, 0.95)
    on = Shot(source_id="s", index=1, start=98.0, end=105.0)
    just_before = Shot(source_id="s", index=2, start=95.0, end=99.0)
    far = Shot(source_id="s", index=3, start=140.0, end=145.0)
    check("a shot containing the line scores 1.0", M._moment_proximity(on, span) == 1.0)
    check("a shot ending inside the pre-roll still scores 1.0",
          M._moment_proximity(just_before, span) == 1.0)
    check("a far shot scores 0", M._moment_proximity(far, span) == 0.0)
    check("no span -> no score", M._moment_proximity(on, None) == 0.0)


def test_neighbourhood_is_wide_enough_for_a_returning_beat():
    """40 of 85 quoted beats in one essay shared a line, so anti-reuse denies most of them the exact
    seconds. A neighbouring shot of the SAME SCENE must still beat unrelated footage."""
    span = (100.0, 102.0, 0.95)
    same_scene = Shot(source_id="s", index=9, start=108.0, end=111.0)      # ~6s after the line
    unrelated = Shot(source_id="s", index=10, start=400.0, end=404.0)
    a = M._moment_proximity(same_scene, span)
    b = M._moment_proximity(unrelated, span)
    check("another shot of the same scene keeps real credit", a > 0.3)
    check("unrelated footage in the same source gets none", b == 0.0)
    check("the scene neighbour outranks the unrelated shot", a > b)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
def test_window_centres_on_the_line_not_the_shot_midpoint():
    cfg = ClipConfig()
    seg = NS(index=3, est_duration=1.92, text="", expected_visual="", quote=LINE)
    shot = Shot(source_id="s", index=35, start=143.3, end=148.2)
    span = (146.8, 147.6, 0.91)
    a0, b0 = M._trim_window(shot, seg, cfg)                 # no moment -> midpoint
    a1, b1 = M._trim_window(shot, seg, cfg, span)           # moment -> centred on the line
    mid = (span[0] + span[1]) / 2.0
    check("midpoint window MISSES the line (the measured bug)", not (a0 <= mid <= b0))
    check("moment window CONTAINS the line", a1 <= mid <= b1)
    check("moment window stays inside the shot", a1 >= shot.start - 1e-6 and b1 <= shot.end + 1e-6)


def test_window_ignores_a_moment_outside_the_shot():
    cfg = ClipConfig()
    seg = NS(index=3, est_duration=2.0, text="", expected_visual="", quote=LINE)
    shot = Shot(source_id="s", index=35, start=143.3, end=148.2)
    far = (54.2, 55.1, 0.91)                                # same line, DIFFERENT copy of the scene
    check("a moment outside this shot leaves the window alone",
          M._trim_window(shot, seg, cfg, far) == M._trim_window(shot, seg, cfg))


def test_window_opens_on_a_line_longer_than_the_beat():
    cfg = ClipConfig()
    seg = NS(index=1, est_duration=2.0, text="", expected_visual="", quote=LINE)
    shot = Shot(source_id="s", index=1, start=70.0, end=85.0)
    span = (73.2, 82.4, 0.91)                               # 9.2s line, ~2.6s of screen time
    a, b = M._trim_window(shot, seg, cfg, span)
    check("a long line is opened ON, not entered halfway", abs(a - span[0]) <= 1.0)


# ---------------------------------------------------------------------------
# beat -> line resolution
# ---------------------------------------------------------------------------
def test_beat_quote_prefers_the_beats_own_quote():
    seg = NS(quote=" " + LINE + " ", text="something else entirely", expected_visual="")
    check("an explicit quote wins", M._beat_quote(seg, ["unrelated anchor line here"]) == LINE)


def test_beat_quote_falls_back_to_the_anchor_line_it_echoes():
    anchor = ["You stand accused of murder. You stand accused of treason.",
              "I did it to protect the woman I love."]
    seg = NS(quote="", expected_visual="",
             text="He stands accused of murder and accused of treason, and he knows it.")
    got = M._beat_quote(seg, anchor)
    check("a beat that echoes an anchor line resolves to it", got == anchor[0])
    weak = NS(quote="", expected_visual="", text="And that raises the next question.")
    check("a beat that echoes nothing resolves to nothing", M._beat_quote(weak, anchor) == "")
    check("no anchor lines -> no fallback", M._beat_quote(weak, None) == "")


# ---------------------------------------------------------------------------
# wiring / kill switch
# ---------------------------------------------------------------------------
def test_paraphrase_fallback_tries_the_verbatim_anchor_line():
    """The analyzer's quote is a paraphrase as often as a transcription, and find_quote_span scores
    the WHOLE phrase, so extra words sink it. Measured on real data: the aired line is "I did it to
    protect you" and the analyzer wrote it two ways across neighbouring beats —
    "I did it to protect Sansa." located at ratio 0.909, "I did what I did to protect Sansa." not at
    all. So a beat must be allowed a second phrasing: the anchor scene's verbatim line."""
    anchor = ["I did it to protect the woman I love.",
              "You stand accused of murder. You stand accused of treason."]
    seg = NS(quote="I did what I did to protect Sansa.", expected_visual="",
             text="He says he did it to protect the woman he love[d].")
    cands = M.beat_quote_candidates(seg, anchor)
    check("the beat's own quote is tried first", cands and cands[0].startswith("I did what I did"))
    check("a verbatim anchor line is offered as a second phrasing", len(cands) >= 2)
    check("the anchor candidate is the echoed line", anchor[0] in cands)

    only_quote = NS(quote=LINE, expected_visual="", text="nothing in common at all here")
    check("no echo -> only the beat's own quote",
          M.beat_quote_candidates(only_quote, anchor) == [LINE])
    check("no quote and no echo -> nothing",
          M.beat_quote_candidates(NS(quote="", expected_visual="", text="zzz"), anchor) == [])

    dupe = NS(quote=anchor[0], expected_visual="", text=anchor[0])
    check("identical phrasings are not tried twice",
          len(M.beat_quote_candidates(dupe, anchor)) == 1)


def test_locate_tries_every_candidate():
    M._QSPAN_CACHE.clear()
    proj = NS(meta={}, index_dir=None, root="/nonexistent")
    seg = NS(quote=LINE, expected_visual="", text="")
    check("locate_beat_moment returns None when nothing is findable",
          M.locate_beat_moment(proj, "nope", seg, ["some anchor line here now"]) is None)


def test_source_wiring():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "vidlore" / "clipstudio" / "match.py").read_text()
    check("_score_pool consults the located moment",
          "locate_beat_moment(proj, ps.sid, seg, anchor_lines)" in src)
    check("the moment rides on the ranking bonus, not reported confidence",
          "bonus += _mom_bonus" in src)
    check("a located moment is reported in the signals", '"moment_lock"' in src)
    check("the main candidate loop centres the window on the moment",
          "_trim_window(ps.shot, seg, cfg, _cand_mom)" in src)
    check("the clean-copy swap re-locates the line in the new source",
          "_trim_window(ps.shot, seg, cfg, _sw_mom)" in src)
    check("there is a kill switch", "VIDLORE_CLIPSTUDIO_MOMENT_LOCK" in src)
    check("the ratio floor guards a loose phrase match", "_MOMENT_MIN_RATIO" in src)


def test_ratio_floor_is_above_the_locator_floor():
    check("moment decisions need more than find_quote_span's own floor",
          M._MOMENT_MIN_RATIO > 0.72)


TESTS = [
    test_span_locates_the_line_in_a_word_stream,
    test_span_is_memoized_and_survives_a_missing_source,
    test_proximity_prefers_the_shot_holding_the_line,
    test_neighbourhood_is_wide_enough_for_a_returning_beat,
    test_window_centres_on_the_line_not_the_shot_midpoint,
    test_window_ignores_a_moment_outside_the_shot,
    test_window_opens_on_a_line_longer_than_the_beat,
    test_beat_quote_prefers_the_beats_own_quote,
    test_beat_quote_falls_back_to_the_anchor_line_it_echoes,
    test_paraphrase_fallback_tries_the_verbatim_anchor_line,
    test_locate_tries_every_candidate,
    test_source_wiring,
    test_ratio_floor_is_above_the_locator_floor,
]

if __name__ == "__main__":
    for fn in TESTS:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
