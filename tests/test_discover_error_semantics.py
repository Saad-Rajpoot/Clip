"""P1.1 — discovery error semantics: typed provider statuses, bounded serial retry of
failed/throttled buckets with partial-result preservation, and order/ranking parity.

    python3 tests/test_discover_error_semantics.py

No network. Providers are stubbed; only classification and orchestration are real.
"""
import os
import socket
import sys
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import discover as D                     # noqa: E402

FAILS = []


# ---------------------------------------------------------------------------
# E1 — classification
# ---------------------------------------------------------------------------
def test_error_classification():
    import requests
    try:
        import yt_dlp.utils as ytu
        dl = ytu.DownloadError
    except Exception:
        dl = RuntimeError
    resp429 = NS(status_code=429)
    err429 = requests.RequestException()
    err429.response = resp429
    cases = [
        (dl("ERROR: HTTP Error 429: Too Many Requests"), D.STATUS_THROTTLED),
        (dl("unable to download: rate limit exceeded"), D.STATUS_THROTTLED),
        (dl("The read operation timed out"), D.STATUS_TIMEOUT),
        (socket.timeout("timed out"), D.STATUS_TIMEOUT),
        (TimeoutError(), D.STATUS_TIMEOUT),
        (requests.Timeout(), D.STATUS_TIMEOUT),
        (err429, D.STATUS_THROTTLED),
        (requests.ConnectionError("dns fail"), D.STATUS_TRANSPORT),
        (dl("ERROR: This video is unavailable"), D.STATUS_TRANSPORT),
        (ValueError("json parse"), D.STATUS_TRANSPORT),
    ]
    for exc, want in cases:
        got = D._classify_net_error(exc)
        assert got == want, f"{type(exc).__name__}({exc}) -> {got}, want {want}"


def test_legacy_wrappers_still_return_bare_lists():
    orig = D._ytsearch_ex
    D._ytsearch_ex = lambda q, n: ([1, 2], D.STATUS_OK)
    try:
        assert D._ytsearch("q", 5) == [1, 2], "back-compat wrapper must return the list"
    finally:
        D._ytsearch_ex = orig


def test_archive_parseable_non_2xx_is_technical_not_empty():
    import requests
    original = requests.get

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {"response": {"docs": []}}

    try:
        for code in (403, 500):
            requests.get = lambda *_a, _code=code, **_kw: Response(_code)
            result, status = D._archive_search_ex("required scene", 4)
            assert result == [] and status == D.STATUS_TRANSPORT
    finally:
        requests.get = original


# ---------------------------------------------------------------------------
# E2 — fan-out orchestration under scripted provider failures
# ---------------------------------------------------------------------------
def _cand(vid, q, provider="youtube"):
    return D.SourceCandidate(url=f"https://x/{vid}", id=vid, title=f"Game of Thrones {vid}",
                             provider=provider, duration=120.0, channel="ch",
                             view_count=100, query=q)


class _Script:
    """Programmable provider: per (query, call_number) -> (results, status)."""

    def __init__(self, plan, default_results):
        self.plan = plan
        self.default = default_results
        self.calls = {}
        self.sleeps = []

    def __call__(self, q, n):
        k = self.calls[q] = self.calls.get(q, 0) + 1
        res, st = self.plan.get((q, k), (None, D.STATUS_OK))
        if res is None:
            res = self.default(q)
        return list(res), st


def _run_fanout(yt, ar, queries, workers, jitter=False, required_queries=None):
    orig = (D._ytsearch_ex, D._archive_search_ex, D.build_queries, D.anchor_queries)
    orig_env = os.environ.get("VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS")
    orig_sleep = time.sleep

    def yt_j(q, n):
        if jitter:
            orig_sleep(0.002 * (hash(q) % 5))
        return yt(q, n)

    D._ytsearch_ex, D._archive_search_ex = yt_j, ar
    D.build_queries = lambda analysis, segments=None: list(queries)
    D.anchor_queries = lambda analysis, segments=None: []
    os.environ["VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS"] = str(workers)
    time.sleep = lambda s: None                          # backoff must not slow the suite
    try:
        from vidlore.clipstudio.config import ClipConfig
        cfg = ClipConfig()
        cfg.discover_per_query = 4
        cfg.discover_resolve_limit = 0
        analysis = NS(movie_title="Game of Thrones", video_type="single_scene",
                      anchor_scenes=[], actors=[], characters=[], key_scenes=[], events=[],
                      visual_keywords=[], locations=[], year="", synopsis="",
                      emotional_moments=[], episode_hint="", tone="")
        return D.discover_sources(
            analysis, cfg, extra_queries=required_queries,
            required_queries=required_queries)
    finally:
        (D._ytsearch_ex, D._archive_search_ex, D.build_queries, D.anchor_queries) = orig
        time.sleep = orig_sleep
        if orig_env is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS"] = orig_env


def _sig(cands):
    return [(c.provider, c.id, c.query) for c in (cands or [])]


def test_failed_provider_bucket_retries_and_recovers_preserving_partials():
    queries = [f"q{i}" for i in range(6)]

    def mk_scripts():
        yt = _Script({("q2", 1): ([], D.STATUS_THROTTLED)},     # q2 yt: throttled 1st, ok on retry
                     lambda q: [_cand(f"yt-{q}-{k}", q) for k in range(3)])
        ar = _Script({}, lambda q: [_cand(f"ar-{q}", q, "archive")])   # archive healthy throughout
        return yt, ar

    yt1, ar1 = mk_scripts()
    healthy = _run_fanout(_Script({}, yt1.default), _Script({}, ar1.default), queries, 1)
    yt2, ar2 = mk_scripts()
    flaky = _run_fanout(yt2, ar2, queries, 4, jitter=True)
    assert _sig(healthy) == _sig(flaky), \
        "a retried throttled bucket must converge to the healthy candidate list, in order"
    assert yt2.calls["q2"] == 2, "exactly one serial retry for the throttled provider"
    assert ar2.calls["q2"] == 1, "the healthy provider of the same query is NOT re-called"


def test_one_provider_hard_down_keeps_other_providers_partial_results():
    queries = ["qa", "qb"]
    yt = _Script({("qa", 1): ([], D.STATUS_TRANSPORT), ("qa", 2): ([], D.STATUS_TRANSPORT),
                  ("qa", 3): ([], D.STATUS_TRANSPORT)},
                 lambda q: [_cand(f"yt-{q}", q)])
    ar = _Script({}, lambda q: [_cand(f"ar-{q}", q, "archive")])
    out = _run_fanout(yt, ar, queries, 4)
    ids = [c.id for c in out]
    assert "ar-qa" in ids, "archive's successful results for qa must survive yt's hard failure"
    assert "yt-qb" in ids and "ar-qb" in ids, "unrelated queries unaffected"
    assert yt.calls["qa"] == 3, "bounded: initial + 2 retries, then give up"


def test_legitimate_empty_is_an_answer_not_a_retry():
    queries = ["qe"]
    yt = _Script({("qe", 1): ([], D.STATUS_EMPTY)}, lambda q: [])
    ar = _Script({("qe", 1): ([], D.STATUS_EMPTY)}, lambda q: [])
    _run_fanout(yt, ar, queries, 4)
    assert yt.calls["qe"] == 1 and ar.calls["qe"] == 1, \
        "a genuine zero-result answer must never be retried"


def test_required_query_partial_provider_answer_is_conclusive():
    yt = _Script({("qr", 1): ([], D.STATUS_TRANSPORT),
                  ("qr", 2): ([], D.STATUS_TRANSPORT),
                  ("qr", 3): ([], D.STATUS_TRANSPORT)}, lambda q: [])
    ar = _Script({("qr", 1): ([], D.STATUS_EMPTY)}, lambda q: [])
    out = _run_fanout(yt, ar, ["qr"], 1, required_queries=["qr"])
    assert out == []
    assert yt.calls["qr"] == 3 and ar.calls["qr"] == 1


def test_required_query_partial_results_with_only_technical_status_is_inconclusive():
    partial = [_cand("partial-old-hit", "qp")]
    yt = _Script({("qp", 1): (partial, D.STATUS_TRANSPORT),
                  ("qp", 2): (partial, D.STATUS_TRANSPORT),
                  ("qp", 3): (partial, D.STATUS_TRANSPORT)}, lambda q: [])
    ar = _Script({("qp", 1): ([], D.STATUS_TIMEOUT),
                  ("qp", 2): ([], D.STATUS_TIMEOUT),
                  ("qp", 3): ([], D.STATUS_TIMEOUT)}, lambda q: [])
    try:
        _run_fanout(yt, ar, ["qp"], 1, required_queries=["qp"])
    except D.TargetedDiscoveryTechnicalError:
        pass
    else:
        raise AssertionError("technical partial hits must not certify required-query exhaustion")


def test_throttled_backoff_is_bounded_and_serial_path_matches_parallel():
    queries = [f"q{i}" for i in range(5)]

    def mk():
        return (_Script({("q1", 1): ([], D.STATUS_THROTTLED),
                         ("q3", 1): ([], D.STATUS_TIMEOUT)},
                        lambda q: [_cand(f"yt-{q}", q)]),
                _Script({}, lambda q: [_cand(f"ar-{q}", q, "archive")]))

    yt_s, ar_s = mk()
    serial = _run_fanout(yt_s, ar_s, queries, 1)
    yt_p, ar_p = mk()
    par = _run_fanout(yt_p, ar_p, queries, 4, jitter=True)
    assert _sig(serial) == _sig(par), "serial and parallel paths must agree under failures"
    assert yt_s.calls == yt_p.calls, "identical per-query attempt counts either way"


TESTS = [
    test_error_classification,
    test_legacy_wrappers_still_return_bare_lists,
    test_archive_parseable_non_2xx_is_technical_not_empty,
    test_failed_provider_bucket_retries_and_recovers_preserving_partials,
    test_one_provider_hard_down_keeps_other_providers_partial_results,
    test_legitimate_empty_is_an_answer_not_a_retry,
    test_required_query_partial_provider_answer_is_conclusive,
    test_required_query_partial_results_with_only_technical_status_is_inconclusive,
    test_throttled_backoff_is_bounded_and_serial_path_matches_parallel,
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
