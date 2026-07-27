"""Cross-source breakout dedup — the same spoken line from two different uploads must dedup.

Regression (job 5462677f95): the S4E4 Olenna confession aired TWICE, 3.4 min apart, from a fan
compilation (seg 192, ASCII apostrophes) and the official HBO clip (seg 234, typographic
U+2019 apostrophes). Two independent defeats:
  * the tokenizer's [a-z'] class split "don’t" into "don"+"t", mangling the Jaccard token set
    below the 0.5 dedup threshold (measured 0.444);
  * the transcripts differ by one ASR mishear ('do you? Well,' vs 'to you? What?'), which
    defeats the substring key.

    python3 tests/test_breakout_crosssource_dedup.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vidlore.clipstudio.build as B                       # noqa: E402

FAILS = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


def _rw(t):
    # mirror of the _select_breakouts closure (apostrophe-normalized tokenizer)
    return re.findall(r"[a-z']+", (t or "").translate(B._BK_APOS_TR).lower())


def _keys(line_a, line_b):
    """Recompute the dedup keys exactly as the pick loop does."""
    cwa = {w for w in _rw(line_a)[:10] if len(w) > 2}
    cwb = {w for w in _rw(line_b)[:10] if len(w) > 2}
    cqa = " ".join(_rw(line_a))
    cqb = " ".join(_rw(line_b))
    substr = bool(cqa and cqb and (cqa in cqb or cqb in cqa))
    tok = (len(cwa & cwb) / max(1, len(cwa | cwb)) >= 0.5) if cwa else False
    sim = B._bk_dedup_same_line(cqa, cqb)
    return substr, tok, sim


L192 = "You don't think I'd let you marry that"                      # fan upload (ASCII ')
L234 = "You don’t really think I’d let you marry that beast, do you?"   # HBO (U+2019)


def test_unicode_apostrophes_no_longer_mangle_tokens():
    toks = _rw(L234)
    check("don’t tokenizes as don't", "don't" in toks)
    check("I’d tokenizes as i'd", "i'd" in toks)
    check("no orphan single-letter fragments", "t" not in toks and "d" not in toks)


def test_the_192_234_double_confession_now_dedups():
    substr, tok, sim = _keys(L192, L234)
    check("at least one dedup key fires on the real pair", substr or tok or sim)
    check("the token-Jaccard key itself recovers (apostrophe fix)", tok)


def test_asr_variant_of_the_same_line_dedups():
    a = "You don't think I'd let you marry that beast, do you? Well,"
    b = "You don't think I'd let you marry that beast to you? What?"
    substr, tok, sim = _keys(a, b)
    check("one-word ASR divergence of the same line is caught", substr or tok or sim)


def test_distinct_dialogue_is_not_deduped():
    pairs = [
        ("The things I do for love.", "He did this! He poisoned my son!"),
        ("You don't think I'd let you marry that beast, do you?",
         "I know you don't want to believe it, but she is. Now,"),
        ("A Lannister always pays his debts.", "Winter is coming, my lord."),
    ]
    for a, b in pairs:
        substr, tok, sim = _keys(a, b)
        check(f"distinct lines stay distinct: {a[:24]!r} vs {b[:24]!r}",
              not (substr or tok or sim))


def test_wired_into_the_pick_loop():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "vidlore", "clipstudio", "build.py")).read()
    check("dedup loop consults _bk_dedup_same_line",
          "_line_sim = _bk_dedup_same_line(_cq, _pq)" in src
          and "_same_src_win or _substr or _tok or _line_sim" in src)
    check("tokenizer applies the apostrophe translation",
          '(t or "").translate(_BK_APOS_TR).lower()' in src)


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
