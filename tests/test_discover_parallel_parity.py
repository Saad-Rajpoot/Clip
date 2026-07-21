"""P1 discovery concurrency: the parallel fan-out must be candidate-for-candidate identical
to the serial loop over a FROZEN result set — same raw sequence, same first-occurrence
dedupe, same ranking, same chosen prefix — and the per-URL subtitle/probe caches must
cache positives only.

    python3 tests/test_discover_parallel_parity.py

No network (searches are stubbed frozen), no LLM, no ffmpeg.
"""
import os
import sys
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import discover as D                     # noqa: E402

FAILS = []


def _frozen_searches(n_queries=24, per=5):
    """Deterministic per-query result sets with heavy cross-query overlap (the real shape:
    the same popular uploads surface for many queries) so first-occurrence dedupe ORDER is
    actually exercised."""
    def yt(query, n):
        qi = int(query.split("#")[1])
        out = []
        for k in range(per):
            vid = f"v{(qi * 3 + k) % (n_queries * 2)}"           # overlapping ids across queries
            out.append(D.SourceCandidate(
                url=f"https://youtube.com/watch?v={vid}", id=vid,
                title=f"Game of Thrones clip {vid}", provider="youtube",
                duration=120.0, channel=f"ch{k % 3}", view_count=1000 + qi, query=query))
        return out

    def ar(query, n):
        qi = int(query.split("#")[1])
        if qi % 4:
            return []
        ident = f"arc{qi}"
        return [D.SourceCandidate(url=f"https://archive.org/details/{ident}", id=ident,
                                  title=f"got archive {ident}", provider="archive",
                                  permission_hint="public_domain", query=query)]
    return yt, ar


def _run_discover(workers, *, jitter=False):
    """Drive ONLY the fan-out + dedupe head of discover_sources via its real code path."""
    yt, ar = _frozen_searches()
    queries = [f"q#{i}" for i in range(24)]
    orig_yt, orig_ar = D._ytsearch_ex, D._archive_search_ex
    orig_bq, orig_aq = D.build_queries, D.anchor_queries
    orig_env = os.environ.get("VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS")

    def yt_j(q, n):
        if jitter:                                       # adversarial completion order
            time.sleep(0.002 * (hash(q) % 7))
        res = yt(q, n)
        return res, (D.STATUS_OK if res else D.STATUS_EMPTY)

    def ar_ex(q, n):
        res = ar(q, n)
        return res, (D.STATUS_OK if res else D.STATUS_EMPTY)

    D._ytsearch_ex, D._archive_search_ex = yt_j, ar_ex
    D.build_queries = lambda analysis, segments=None: list(queries)
    D.anchor_queries = lambda analysis, segments=None: [queries[0], queries[3]]
    os.environ["VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS"] = str(workers)
    try:
        from vidlore.clipstudio.config import ClipConfig
        cfg = ClipConfig()
        cfg.discover_per_query = 5
        cfg.discover_resolve_limit = 0                   # no network probes in this test
        analysis = NS(movie_title="Game of Thrones", video_type="single_scene",
                      anchor_scenes=[], actors=[], characters=[], key_scenes=[], events=[],
                      visual_keywords=[], locations=[], year="", synopsis="",
                      emotional_moments=[], episode_hint="", tone="")
        cands = D.discover_sources(analysis, cfg)
        return cands
    finally:
        D._ytsearch_ex, D._archive_search_ex = orig_yt, orig_ar
        D.build_queries, D.anchor_queries = orig_bq, orig_aq
        if orig_env is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS"] = orig_env


def _sig(cands):
    return [(c.provider, c.id, c.title, c.query, round(c.relevance, 6)) for c in (cands or [])]


def test_parallel_discovery_is_candidate_identical_to_serial():
    serial = _run_discover(1)
    assert serial, "the serial baseline must produce candidates under the frozen stubs"
    par = _run_discover(4, jitter=True)                  # adversarial out-of-order completion
    assert _sig(serial) == _sig(par), \
        "parallel fan-out must reproduce the serial candidate list, order and ranking exactly"


def test_workers_env_one_is_the_serial_loop():
    a = _run_discover(1)
    b = _run_discover(1)
    assert _sig(a) == _sig(b), "the serial path must be deterministic under frozen stubs"


def test_subs_cache_stores_positives_only():
    D._SUBS_TEXT_CACHE.clear()
    calls = {"n": 0}
    orig = D._fetch_subs_text_uncached

    def fake(url, timeout=45):
        calls["n"] += 1
        return "" if "dead" in url else f"subs for {url}"

    D._fetch_subs_text_uncached = fake
    try:
        assert D._fetch_subs_text("https://x/ok") == "subs for https://x/ok"
        assert D._fetch_subs_text("https://x/ok") == "subs for https://x/ok"
        assert calls["n"] == 1, "a positive result must be served from cache"
        assert D._fetch_subs_text("https://x/dead") == ""
        assert D._fetch_subs_text("https://x/dead") == ""
        assert calls["n"] == 3, "a failure must stay retryable (never cached)"
    finally:
        D._fetch_subs_text_uncached = orig
        D._SUBS_TEXT_CACHE.clear()


TESTS = [
    test_parallel_discovery_is_candidate_identical_to_serial,
    test_workers_env_one_is_the_serial_loop,
    test_subs_cache_stores_positives_only,
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
