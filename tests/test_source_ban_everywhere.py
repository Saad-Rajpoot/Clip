"""A source BAN must hold in every pool a frame or a line can reach the timeline from:
match (moving clips), the still/image-fallback pool, and breakout selection (real audio).

Regression: an "ALTERNATE ENDING" AI recreation was banned from match yet still aired — as
stills AND as a 9.9s real-audio BREAKOUT — because those two pools read proj.sources directly.

    python3 tests/test_source_ban_everywhere.py

No network, no model, no ffmpeg.
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import match as M                      # noqa: E402

FAILS = []


def _proj(banned=None, meta_extra=None):
    meta = {"banned_sources": list(banned or [])}
    meta.update(meta_extra or {})
    return NS(meta=meta)


# ---------------------------------------------------------------------------
# the shared reader
# ---------------------------------------------------------------------------
def test_banned_ids_from_project_meta():
    assert M.banned_source_ids(_proj(["a", "b"])) == {"a", "b"}
    assert M.banned_source_ids(_proj([])) == set()
    assert M.banned_source_ids(NS(meta={})) == set()
    assert M.banned_source_ids(NS(meta=None)) == set(), "missing meta must not raise"


def test_banned_ids_from_env_merge():
    old = os.environ.get("VIDLORE_CLIPSTUDIO_BANNED_SOURCES")
    os.environ["VIDLORE_CLIPSTUDIO_BANNED_SOURCES"] = " x , y ,,"
    try:
        got = M.banned_source_ids(_proj(["a"]))
        assert got == {"a", "x", "y"}, got
    finally:
        if old is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_BANNED_SOURCES", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_BANNED_SOURCES"] = old


def test_ids_are_strings_even_if_meta_holds_numbers():
    assert M.banned_source_ids(_proj([1, 2])) == {"1", "2"}


# ---------------------------------------------------------------------------
# every consumer actually consults it (source-level wiring check)
# ---------------------------------------------------------------------------
def _src(path):
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "vidlore", "clipstudio", path), encoding="utf-8").read()


def test_match_pool_consults_the_ban():
    s = _src("match.py")
    assert "banned_source_ids(proj" in s, "match._load_pool must read the shared ban list"
    assert "if src.id in _banned" in s, "match must skip banned sources"
    # _load_pool re-derives its OWN source-level rejections every run, so it must not read its
    # own persisted output back — otherwise the auto-bans go sticky and a gate kill-switch can
    # never re-admit a source.
    assert "include_auto=False" in s, \
        "match._load_pool must exclude auto-rejections from its own ban read"


def test_auto_rejections_hold_in_every_other_pool():
    """A SOURCE-LEVEL rejection made while building the match pool (subtitled copy, static image,
    non-photographic, watermarked, modern talking-head, promo overlay) must also exclude the
    source from the still pool, the breakout pool and build's shot-walk.

    Regression (job 69d80e9dd4): 'How Game of Thrones Filmed Arya And Brienne's Sword Fight' was
    dropped by _load_pool as a subtitled copy and contributed ZERO beats to selections — yet its
    behind-the-scenes stunt-rehearsal footage (modern t-shirts, gym mats, Nike trainers) still
    aired three times, because those other pools read proj.sources directly."""
    s = _src("match.py")
    assert "auto_rejected_sources" in s, "the shared reader must union the auto-rejected list"
    # every source-level drop goes through the one recorder, which adds to _auto_rej AND files the
    # reason. Asserted on the recorder rather than a literal `.add(...)` so the check survives the
    # refactor but still fails if a new gate is added with a bare `continue`.
    assert "_auto_rej.add(sid)" in s, "_load_pool must RECORD its source-level rejections"
    assert s.count("_reject(src.id,") >= 10, \
        "each source-level gate must record its rejection through _reject()"
    assert "_auto_rej.add(src.id)" not in s, \
        "a gate is bypassing _reject() — its rejection would carry no reason for the backfill pass"
    # the shared reader unions them, so the still/breakout/walk consumers inherit them for free
    assert M.banned_source_ids(_proj([], {"auto_rejected_sources": ["bts"]})) == {"bts"}
    assert M.banned_source_ids(_proj(["op"], {"auto_rejected_sources": ["bts"]})) == {"op", "bts"}
    # ...but match's own pass must be able to ignore them
    assert M.banned_source_ids(_proj(["op"], {"auto_rejected_sources": ["bts"]}),
                               include_auto=False) == {"op"}


def test_auto_rejection_list_is_replaced_not_accumulated():
    """It must describe the CURRENT gate configuration, so flipping a gate off re-admits the
    source everywhere instead of leaving a stale ban behind."""
    s = _src("match.py")
    assert "REPLACE, never union" in s, \
        "the auto-rejected list must be replaced each run, not accumulated"


def test_still_pool_consults_the_ban():
    s = _src("orchestrate.py")
    assert "banned_source_ids as _banned_ids" in s, \
        "the image-fallback still pool must import the shared ban list"
    assert "if s.id in _banned_sv" in s, \
        "the still pool must SKIP banned sources (a banned still sits on screen for seconds)"


def test_breakout_selection_consults_the_ban():
    s = _src("build.py")
    assert "banned_source_ids as _banned_bk" in s, \
        "breakout selection must import the shared ban list"
    assert "s.id not in _bk_banned" in s, \
        "a banned source must never open a breakout (it would air its own audio)"


# ---------------------------------------------------------------------------
# behavioural: the filter expression each consumer uses
# ---------------------------------------------------------------------------
def test_filter_semantics_exclude_only_the_banned_id():
    srcs = [NS(id="good1", status="ok"), NS(id="fanfilm", status="ok"), NS(id="good2", status="ok")]
    banned = M.banned_source_ids(_proj(["fanfilm"]))
    kept = [s.id for s in srcs if s.id not in banned]
    assert kept == ["good1", "good2"], kept
    assert "fanfilm" not in kept



def test_rejection_reasons_are_recorded_for_the_backfill_pass():
    """A rejection's REASON decides whether the footage is worth replacing.

    'subtitled copy of the trial' means the right scene arrived in an unusable wrapper — go find a
    clean copy. 'talking-head interview' means footage we never wanted — searching for a cleaner
    copy just buys another interview. Without the reason the backfill cannot tell them apart, and on
    its first live run it spent a search on 'Arya and Bran Stark actors on growing up on the set'."""
    s = _src("match.py")
    assert "auto_rejected_reasons" in s, "_load_pool must persist WHY each source was rejected"
    for code in ("subtitled_copy", "watermarked", "promo_overlay",
                 "reaction", "wrong_show", "graphics"):
        assert f'"{code}"' in s, f"no gate records the {code!r} reason"
    o = _src("orchestrate.py")
    assert "_replaceable" in o and "auto_rejected_reasons" in o, \
        "the backfill pass must filter rejections by reason"


TESTS = [
    test_banned_ids_from_project_meta,
    test_banned_ids_from_env_merge,
    test_ids_are_strings_even_if_meta_holds_numbers,
    test_match_pool_consults_the_ban,
    test_auto_rejections_hold_in_every_other_pool,
    test_auto_rejection_list_is_replaced_not_accumulated,
    test_rejection_reasons_are_recorded_for_the_backfill_pass,
    test_still_pool_consults_the_ban,
    test_breakout_selection_consults_the_ban,
    test_filter_semantics_exclude_only_the_banned_id,
]

if __name__ == "__main__":
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            FAILS.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - len(FAILS)}/{len(TESTS)} passed")
    sys.exit(1 if FAILS else 0)
