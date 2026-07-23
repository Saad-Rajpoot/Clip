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
    assert "banned_source_ids(proj)" in s, "match._load_pool must read the shared ban list"
    assert "if src.id in _banned" in s, "match must skip banned sources"


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


TESTS = [
    test_banned_ids_from_project_meta,
    test_banned_ids_from_env_merge,
    test_ids_are_strings_even_if_meta_holds_numbers,
    test_match_pool_consults_the_ban,
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
