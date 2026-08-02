"""A gate-rejected source must be REPLACED, not silently subtracted from the pool.

Measured on job 69d80e9dd4_v4: 11 of 84 ok sources were dropped by the pool gates (7 burned-caption
re-uploads, 2 promo-card compilations, 1 STAR India screener with a burned timecode, 1 talking-head).
Every one of those rejections is correct — the footage really is unusable. The bug was what came
next: discovery had already spent its budget, so the drops just made the pool thinner, and among the
casualties were the most on-topic upload in the whole pool ("The Trial of Petyr Baelish", 1080p,
23 min) and the only clip of the dagger handover. 207 beats then asked for the trial and were served
scene packs of neighbouring scenes — right character, wrong scene, 135 times over.

These tests pin the replacement pass: it runs before match, searches with the LOST upload's own
title, and cannot spin.
"""
from __future__ import annotations

import types

import pytest

from vidlore.clipstudio import orchestrate as O


class _Src:
    def __init__(self, sid, title, status="ok", url=""):
        self.id, self.title, self.status = sid, title, status
        self.url = url or f"https://y/{sid}"


class _Proj:
    def __init__(self, sources, rejected):
        self.sources = sources
        self.meta = {"auto_rejected_sources": list(rejected)}
        self.saved = 0

    def save(self):
        self.saved += 1


def _wire(monkeypatch, *, cands=None, downloads=None):
    """Stub the network edges; capture what the pass asked for."""
    seen = {"queries": [], "pool_calls": 0, "index_calls": 0, "downloaded": []}
    cands = cands or []
    downloads = downloads if downloads is not None else []

    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "_load_pool",
                        lambda *a, **k: seen.__setitem__("pool_calls", seen["pool_calls"] + 1))

    def _disc(analysis, cfg, *, segments=None, progress=None, extra_queries=None):
        seen["queries"].append(list(extra_queries or []))
        return list(cands)

    def _dl(proj, new, cfg, *, policy=None, limit=None, progress=None, on_ready=None):
        for s in downloads:
            proj.sources.append(s)
        seen["downloaded"].append(len(new))

    def _idx(proj, cfg, **k):
        seen["index_calls"] += 1

    monkeypatch.setattr("vidlore.clipstudio.discover.discover_sources", _disc)
    monkeypatch.setattr("vidlore.clipstudio.download.download_candidates", _dl)
    monkeypatch.setattr("vidlore.clipstudio.index.index_all", _idx)
    return seen


def _run(proj, **kw):
    # refs/faceid must be present: a replacement indexed without cast data cannot compete
    # (w_face is 0.30 of the score), so the pass refuses to run without them.
    kw.setdefault("refs", {"Aidan Gillen": object()})
    kw.setdefault("faceid_obj", object())
    return O._backfill_rejected_sources(
        proj, [], types.SimpleNamespace(movie_title="Game of Thrones", key_scenes=[]),
        types.SimpleNamespace(), roster=None,
        policy="approved_testing", max_sources=8, show_title="Game of Thrones",
        log=kw.pop("log", lambda m: None), **kw)


def test_refuses_to_run_without_face_id_references():
    """A source indexed without Face-ID starts w_face (0.30) behind every incumbent, so it loses
    the very beats it was fetched for and the failure reads as 'the picker ignores good footage'."""
    msgs = []
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    n = O._backfill_rejected_sources(
        proj, [], types.SimpleNamespace(movie_title="Game of Thrones", key_scenes=[]),
        types.SimpleNamespace(), refs={}, faceid_obj=None, roster=None,
        policy="approved_testing", max_sources=8, show_title="Game of Thrones", log=msgs.append)
    assert n == 0
    assert any("no Face-ID references" in m for m in msgs)


def test_kill_switch_disables_the_pass(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", "0")
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("a", "The Trial of Petyr Baelish")], ["a"])
    assert _run(proj) == 0
    assert seen["pool_calls"] == 0, "kill switch must skip before doing any work"


def test_no_rejections_does_nothing(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("a", "clean scene pack")], [])
    assert _run(proj) == 0
    assert not seen["queries"], "nothing was lost, so nothing should be searched for"


def test_searches_with_the_lost_upload_s_own_title(monkeypatch):
    """No beat-derived query reproduces 'the 23-minute trial upload we just threw away'."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish | The One Detail Everyone Misses"),
                  _Src("d", "Bran and LittleFinger / Catspaw Dagger")], ["t", "d"])
    _run(proj)
    q = seen["queries"][0]
    assert any("Trial of Petyr Baelish" in x for x in q)
    assert any("Catspaw Dagger" in x for x in q)


def test_channel_furniture_is_stripped_from_the_query(monkeypatch):
    """'| 4K' / '| English Subtitles' would drag the search back to more re-uploads."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("t", "Littlefinger Trial [HBO] | English Subtitles | 4K")], ["t"])
    _run(proj)
    q = " ".join(seen["queries"][0]).lower()
    assert "littlefinger trial" in q
    assert "subtitle" not in q and "4k" not in q and "hbo" not in q


def test_admits_and_indexes_the_replacement(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    new = _Src("clean", "Littlefinger Trial FULL SCENE", url="https://y/clean")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/clean", title="Littlefinger Trial")],
                 downloads=[new])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    assert _run(proj) == 1
    assert seen["index_calls"] == 1, "a downloaded replacement is useless until it is indexed"


def test_does_not_re_search_the_same_lost_title_twice(monkeypatch):
    """Round 2 must not spend the budget re-asking for a title round 1 already chased."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "3")
    new = _Src("clean", "replacement", url="https://y/clean")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/clean", title="replacement")],
                 downloads=[new])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    _run(proj)
    flat = [q for rnd in seen["queries"] for q in rnd]
    assert len(flat) == len(set(flat)), f"a title was searched twice: {flat}"


def test_no_new_candidate_stops_cleanly(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch, cands=[])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 0
    assert any("no NEW candidate" in m for m in msgs)
    assert seen["downloaded"] == [], "nothing new means nothing to download"


def test_a_failing_edge_degrades_instead_of_killing_the_render(monkeypatch):
    """Backfill is an improvement pass — it must never take the render down with it."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    _wire(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("vidlore.clipstudio.discover.discover_sources", _boom)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 0
    assert any("discovery failed" in m for m in msgs)


def test_audit_is_persisted(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    _wire(monkeypatch, cands=[])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    _run(proj)
    a = proj.meta.get("backfill_audit")
    assert a and a["rounds"], "the pass must record what it looked for and what it found"
    assert a["rounds"][0]["rejected"] == ["t"]


@pytest.mark.parametrize("title,want", [
    ("Arya Cuts Littlefinger's Throat - 1080p", "arya cuts littlefinger's throat"),
    ("Bran Stark Scene Pack (GOT S7)", "bran stark scene pack"),
])
def test_query_cleaning_keeps_the_scene_words(monkeypatch, title, want):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("t", title)], ["t"])
    _run(proj)
    assert want in " ".join(seen["queries"][0]).lower()


# ---------------------------------------------------------------- reason-aware replacement

def _proj_with_reasons(reasons):
    p = _Proj([_Src(sid, f"title of {sid}") for sid in reasons], list(reasons))
    p.meta["auto_rejected_reasons"] = dict(reasons)
    return p


def test_only_quality_rejects_are_replaced(monkeypatch):
    """A subtitled copy of the right scene is worth replacing. An interview is not — searching for
    a cleaner copy of a talking-head just buys another talking-head."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _proj_with_reasons({
        "subbed": "subtitled_copy",          # right footage, unusable copy  -> replace
        "wm": "watermarked",                 # ditto                          -> replace
        "promo": "promo_overlay",            # ditto                          -> replace
        "interview": "talking_head_visual",  # never wanted                   -> skip
        "reaction": "reaction",              # never wanted                   -> skip
        "hotd": "wrong_show",                # never wanted                   -> skip
        "fanart": "graphics",                # never wanted                   -> skip
    })
    _run(proj)
    q = " ".join(seen["queries"][0]).lower()
    for sid in ("subbed", "wm", "promo"):
        assert sid in q, f"{sid} is a quality reject and must be replaced"
    for sid in ("interview", "reaction", "hotd", "fanart"):
        assert sid not in q, f"{sid} is content we never wanted — do not go looking for more"


def test_unknown_reason_defaults_to_replaceable(monkeypatch):
    """An older project has no reason map; treat its rejects as quality rather than silently
    replacing nothing."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("legacy", "The Trial of Petyr Baelish")], ["legacy"])
    proj.meta.pop("auto_rejected_reasons", None)
    _run(proj)
    assert seen["queries"] and "Trial of Petyr Baelish" in " ".join(seen["queries"][0])


def test_all_content_rejects_means_no_search_at_all(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _proj_with_reasons({"a": "reaction", "b": "wrong_show"})
    msgs = []
    assert _run(proj, log=msgs.append) == 0
    assert not seen["queries"]
    assert any("no gate-rejected source to replace" in m for m in msgs)


# ---------------------------------------------------------------- yield-proven replacements

def test_a_replacement_with_no_usable_shots_is_not_counted(monkeypatch):
    """Clearing the SOURCE gates is not the same as being able to air.

    The live run fetched a replacement titled 'Littlefinger gives Catspaw dagger to Bran Stark' —
    precisely the footage 8 beats were asking for — and it was another screener with 'FOR INTERNAL
    VIEWING ONLY' burned into the picture. It passed every source-level gate and then lost all 11
    shots to the shot-level text gate, so it won zero beats while the pass reported '+3 clean
    sources'. A replacement has to prove a usable yield."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    dud = _Src("screener", "Littlefinger gives Catspaw dagger to Bran", url="https://y/screener")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/screener", title="dagger handover")],
                 downloads=[dud])
    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "usable_shot_yield", lambda proj, sid, cfg=None: (0, 11))
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 0, "a 0-yield source is not a replacement"
    assert any("0 usable shots" in m for m in msgs)
    assert "screener" in (proj.meta.get("banned_sources") or []), \
        "an unusable replacement must be banned, not left to pollute the still/breakout pools"


def test_a_replacement_with_usable_shots_counts(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    good = _Src("clean", "Littlefinger Trial FULL SCENE", url="https://y/clean")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/clean", title="trial")],
                 downloads=[good])
    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "usable_shot_yield", lambda proj, sid, cfg=None: (33, 34))
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 1
    assert any("usable source(s) indexed" in m for m in msgs)
    assert "clean" not in (proj.meta.get("banned_sources") or [])


def test_unmeasurable_yield_is_given_the_benefit_of_the_doubt(monkeypatch):
    """A yield probe that throws must not silently ban a good source."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    s = _Src("odd", "some scene", url="https://y/odd")
    _wire(monkeypatch, cands=[types.SimpleNamespace(url="https://y/odd", title="x")], downloads=[s])
    from vidlore.clipstudio import match as M

    def _boom(*a, **k):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(M, "usable_shot_yield", _boom)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    assert _run(proj) == 1
    assert "odd" not in (proj.meta.get("banned_sources") or [])


# ---------------------------------------------------------------- what a replacement may BE

def _title_filter(title):
    """Exercise the real predicate through a one-source backfill run."""
    import re, inspect
    from vidlore.clipstudio import orchestrate as O
    src = inspect.getsource(O._backfill_rejected_sources)
    assert "_title_ok" in src, "the pass must screen replacement TITLES, not just shot yield"
    return src


@pytest.mark.parametrize("title,blocked", [
    # measured on job benjen_v2 — every one of these was ADMITTED into a Benjen Stark essay
    ("Cersei and Jaime Lannister - Game of Thrones - All Scenes Part 3/8", True),
    ("Cersei and Jaime Lannister - Game of Thrones - All Scenes Part 4/8", True),
    ("Games of Thrones - S07E07 - Behind the Scene - Dragon Pit Meeting", True),
    ("A tale of Benjen Stark -  A Game of Thrones fanfiction - Winter", True),
    # ...and these must still get through
    ("Game of Thrones 7x06 - Benjen Saves Jon Snow", False),
    ("Game of Thrones S7E6 - Beyond The Wall | Wight Bear attack", False),
    ("Night King Destroys The Wall - Game of Thrones S07E07", False),
])
def test_a_replacement_must_be_scene_footage_not_an_anthology(monkeypatch, title, blocked):
    """A usable shot yield proves a source CAN air, never that it SHOULD.

    The 13-minute "All Scenes Part 3/8" compilation cleared every per-shot gate with 162 usable
    shots and then fed the breakout miner a Season-1 Cersei/Ned conversation. The pass had searched
    with the REJECTED upload's own title — which is only a good query when that upload was genuinely
    wanted; this one was already marginal, so asking for a cleaner copy of it bought an anthology of
    a different story."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    new = _Src("cand", title, url="https://y/cand")
    _wire(monkeypatch,
          cands=[types.SimpleNamespace(url="https://y/cand", title=title)],
          downloads=[new])
    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "usable_shot_yield", lambda proj, sid, cfg=None: (162, 178))
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    n = _run(proj, log=msgs.append)
    if blocked:
        assert n == 0, f"{title!r} must never be admitted"
        assert any("rejected" in m for m in msgs)
    else:
        assert n == 1, f"{title!r} is legitimate scene footage and must pass"


def test_the_title_screen_runs_before_the_expensive_yield_probe():
    """No point decoding shots for a source we already know we will not use."""
    import inspect
    from vidlore.clipstudio import orchestrate as O
    src = inspect.getsource(O._backfill_rejected_sources)
    assert src.index("_why_bad = _title_ok(") < src.index("usable_shot_yield(proj, s.id, cfg)")
