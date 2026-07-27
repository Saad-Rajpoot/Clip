"""A breakout caption line the VERIFIED SOURCE LINE corroborates must not be deleted for low
ASR confidence.

Regression (job 69d80e9dd4, audited 2026-07-26): all four breakouts lost burned text to the
0.45 ASR word-confidence floor, leaving 35-71% of the words on screen. The worst case showed
"white winds blow, the lone wolf" and never showed the payoff "but the pack survives" — while
the DELIVERED AUDIO said the whole line verbatim (re-transcribing the delivered mix with
medium.en returned "When the snows fall and the white winds blow, the lone wolf dies, but the
pack survives"). The floor was judging "base" int8 whisper's opinion of movie audio, blind to
the fact that _correct_breakout_words had already matched those exact words to the verified
source line one step earlier.

    python3 tests/test_breakout_caption_source_rescue.py

No network, no model, no ffmpeg.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio.build import _correct_breakout_words          # noqa: E402

FAILS = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


# the real breakout-4 line, with the low confidences whisper actually reported
ARYA = "When the snows fall and the white winds blow, the lone wolf dies, but the pack survives."
WORDS = [(w, i * 0.4, i * 0.4 + 0.35, 0.42 if i >= 9 else 0.80)
         for i, w in enumerate(ARYA.replace(",", "").replace(".", "").split())]


def test_meta_is_opt_in():
    """Default call shape is byte-identical — existing callers and tests must not shift."""
    out = _correct_breakout_words(list(WORDS), ARYA)
    check("default return is still just the word list", isinstance(out, list))
    check("default return preserves the 4-tuple shape",
          all(isinstance(w, tuple) and len(w) == 4 for w in out))
    check("default return preserves ASR timings and probabilities",
          [(w[1], w[2], w[3]) for w in out] == [(w[1], w[2], w[3]) for w in WORDS])


def test_align_and_src_ok_on_a_verbatim_line():
    out, align, src_ok, ops, kw = _correct_breakout_words(list(WORDS), ARYA, return_meta=True)
    check("verbatim line aligns at ~1.0 recall", align >= 0.95)
    check("every word is corroborated by the source line", all(src_ok))
    check("src_ok is parallel to the word list", len(src_ok) == len(WORDS))
    check("the payoff words are corroborated",
          all(src_ok[i] for i, w in enumerate(WORDS) if w[0] in ("pack", "survives")))


def test_drifted_dialogue_is_not_corroborated():
    """The safeguard: a breakout that drifted onto other dialogue must NOT be rescued."""
    other = [(w, i * 0.4, i * 0.4 + 0.35, 0.30)
             for i, w in enumerate("the night is dark and full of terrors my lord".split())]
    out, align, src_ok, ops, kw = _correct_breakout_words(other, ARYA, return_meta=True)
    check("drifted dialogue aligns far below the strong tier", align < 0.80)
    check("drifted dialogue is not corroborated word-by-word", not all(src_ok))


def test_short_or_missing_source_line_is_a_no_op():
    out, align, src_ok, ops, kw = _correct_breakout_words(list(WORDS), "", return_meta=True)
    check("no source line -> zero align (caller keeps today's drop behaviour)", align == 0.0)
    check("no source line -> nothing is corroborated", not any(src_ok))
    out2, align2, src_ok2, _, _ = _correct_breakout_words(list(WORDS), "two words",
                                                          return_meta=True)
    check("sub-3-token source line -> zero align", align2 == 0.0)


def test_rescue_rule_as_the_burner_applies_it():
    """Mirror of the branch in _breakout_caption_ass: a sub-floor group whose every word is
    source-backed is kept; an uncorroborated one is still dropped."""
    from vidlore.clipstudio.build import _BK_CAP_ALIGN_MIN
    _, align, src_ok, _, _ = _correct_breakout_words(list(WORDS), ARYA, return_meta=True)
    tail = [i for i, w in enumerate(WORDS) if w[3] < 0.45]
    strong = bool(ARYA) and align >= _BK_CAP_ALIGN_MIN
    backed = bool(tail) and all(src_ok[i] for i in tail)
    check("the low-confidence payoff group would now be KEPT", strong and backed)

    other = [(w, i * 0.4, i * 0.4 + 0.35, 0.30)
             for i, w in enumerate("the night is dark and full of terrors my lord".split())]
    _, align_o, src_ok_o, _, _ = _correct_breakout_words(other, ARYA, return_meta=True)
    keep_o = bool(ARYA) and align_o >= _BK_CAP_ALIGN_MIN and all(src_ok_o)
    check("an uncorroborated low-confidence group is still DROPPED", not keep_o)


def test_source_grep_wires_the_rescue_into_the_burner():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "vidlore" / "clipstudio" / "build.py").read_text()
    check("burner asks for the alignment metadata",
          "_correct_breakout_words(\n            words, cap.get(\"line\", \"\"), log=log, return_meta=True)" in src
          or "return_meta=True)" in src)
    check("burner logs a source-backed keep",
          "every word matches the verified" in src)
    check("confidence floor is env-tunable", "VIDLORE_CLIPSTUDIO_BK_CAP_CONF_FLOOR" in src)


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
