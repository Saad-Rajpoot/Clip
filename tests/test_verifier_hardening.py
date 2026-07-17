"""Behavioural tests for the verifier: cache identity, circuit breaker, fail-closed.

These drive verify_and_repair with a FAKE vision backend and count real calls — the previous
suite asserted source strings, which cannot tell you whether the breaker actually stops calling.

    python3 tests/test_verifier_hardening.py

No network, no LLM, no ffmpeg.
"""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import verify as V  # noqa: E402

FAILS = []
BASE = dict(src_hash="a", source_id="s1", shot_start=1.0, shot_end=3.0,
            beat_text="Tywin dismisses Joffrey", required_entity="Tywin Lannister",
            required_kind="character", expected_visual="Tywin at the council table",
            scene_query="Game of Thrones small council", era="S03E10",
            visual_policy="exact_scene", is_specific=True,
            faceid_names=["Charles Dance"], multiframe=False, image_id="kf:abc",
            model="gemini:gemini-2.5-flash:apikey")


# ---------------------------------------------------------------------------
# H1 — cache identity. Every prompt-affecting input must change the key.
# ---------------------------------------------------------------------------
def test_every_prompt_affecting_field_invalidates_the_cache():
    fp0 = V.verdict_fingerprint(**BASE)
    assert fp0 == V.verdict_fingerprint(**BASE), "must be deterministic"
    # each of these is interpolated into the prompt, or selects which prompt is sent
    for k, alt in (("src_hash", "b"), ("source_id", "s2"),
                   ("shot_start", 1.5), ("shot_end", 9.0),
                   ("beat_text", "different beat"),
                   ("required_entity", "Joffrey Baratheon"),
                   ("required_kind", "object"),
                   ("expected_visual", "a different storyboard"),
                   ("scene_query", "a different scene"),
                   ("era", "S04E01"),
                   ("visual_policy", "generic_filler"),
                   ("is_specific", False),
                   ("faceid_names", ["Jack Gleeson"]),
                   ("multiframe", True),
                   ("image_id", "kf:zzz"),
                   ("model", "anthropic:claude-3-5-sonnet")):
        d = dict(BASE)
        d[k] = alt
        assert V.verdict_fingerprint(**d) != fp0, f"{k} must change the verdict identity"


def test_faceid_names_are_order_independent_but_content_sensitive():
    a = V.verdict_fingerprint(**{**BASE, "faceid_names": ["Charles Dance", "Lena Headey"]})
    b = V.verdict_fingerprint(**{**BASE, "faceid_names": ["Lena Headey", "charles dance"]})
    c = V.verdict_fingerprint(**{**BASE, "faceid_names": ["Lena Headey"]})
    assert a == b, "reordered/recased faces are the same evidence"
    assert a != c, "a different face set is different evidence"


def test_prompt_and_sheet_versions_are_in_the_key():
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "vidlore", "clipstudio",
                            "verify.py"), encoding="utf-8").read()
    m = re.search(r"def verdict_fingerprint\(.*?return h\.hexdigest", src, re.S)
    assert m and "PROMPT_VERSION" in m.group(0) and "SHEET_VERSION" in m.group(0), \
        "a changed prompt or a changed sheet sampling must invalidate every verdict"


def test_only_schema_valid_successful_verdicts_are_reusable():
    ok = {"verdict": "keep", "confidence": 0.9, "status": "ok"}
    assert V._verdict_schema_ok(ok)
    for bad, why in (({"status": "error"}, "an error stub is not a judgment"),
                     ({"status": "unavailable"}, "unavailable is not a judgment"),
                     ({"confidence": 0.9}, "a missing verdict key would read as falsy = 'not replace'"),
                     ({"verdict": "maybe", "confidence": 0.5}, "unknown verdict value"),
                     ({"verdict": "keep", "confidence": "high"}, "confidence must be numeric"),
                     ("not a dict", "not a dict"), (None, "None")):
        assert not V._verdict_schema_ok(bad), why


def test_source_fingerprint_sees_any_changed_byte_and_is_memoized():
    """Head+size was too weak (a re-encode preserves both while changing every judged frame), and
    head/middle/tail is still blind BETWEEN the sampled windows. It is a full hash, memoized against
    (size, mtime) so the cost is paid once per source, not per run."""
    d = tempfile.mkdtemp()
    try:
        a = os.path.join(d, "a.bin")
        with open(a, "wb") as fh:
            fh.write(b"HEAD" + b"\0" * (4 << 20) + b"TAIL")
        f0 = V._file_fingerprint(a)
        assert V._file_fingerprint(a) == f0, "must be stable"
        assert os.path.exists(a + ".fp.json"), "digest must be memoized beside the media"

        # ONE byte, at an offset a head/middle/tail sampler would have missed entirely
        with open(a, "r+b") as fh:
            fh.seek((1 << 20) + 7)
            fh.write(b"X")
        os.utime(a, (0, 0))                      # force a distinct mtime -> re-hash
        assert V._file_fingerprint(a) != f0, "a single changed byte must change the identity"

        assert V._file_fingerprint(os.path.join(d, "missing.bin")) == "missing"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# H3 — the breaker must actually STOP CALLING. Counts calls, not strings.
# ---------------------------------------------------------------------------
def _mini_project(tmp, n_beats=30):
    from vidlore.clipstudio.models import ClipProject, ScriptSegment, ClipSelection, SourceVideo, Shot
    proj = ClipProject(name="t", root=str(tmp))
    proj.ensure_dirs()
    media = os.path.join(tmp, "src.mp4")
    with open(media, "wb") as fh:
        fh.write(b"\0" * 2048)
    proj.sources = [SourceVideo(id="s1", url="u", title="Game of Thrones S03E10 council",
                                permission="owner", status="ok", local_path=media)]
    segs, shots = [], []
    for i in range(n_beats):
        segs.append(ScriptSegment(index=i, text=f"Tywin dismisses Joffrey {i}",
                                  required_entity="Tywin Lannister", required_kind="character",
                                  visual_policy="exact_scene", is_specific_claim=True))
        kf = os.path.join(tmp, f"kf{i}.jpg")
        with open(kf, "wb") as fh:
            fh.write(b"\xff\xd8\xff")
        shots.append(Shot(source_id="s1", index=i, start=float(i), end=float(i) + 2.0,
                          keyframe_path=kf))
        proj.selections.append(ClipSelection(segment_index=i, source_id="s1", shot_index=i,
                                             in_point=float(i), out_point=float(i) + 2.0,
                                             confidence=0.8))
    proj.meta["analysis"] = {"video_type": "single_scene", "episode_hint": "S03E10",
                             "episode_hint_verified": True, "characters": [], "actors": []}
    return proj, segs, shots


def _run_verify(tmp, *, always_fail=True, n_beats=30, sheet=False):
    """Drive verify_and_repair with a fake vision backend; return (summary, call_count, proj).

    sheet=False by default: the fake media is 2KB, so a real contact-sheet build cannot succeed,
    and the code CORRECTLY refuses to cache a verdict whose sheet prediction didn't hold (a
    single-frame answer must never be stored under a multiframe key). Turning the sheet off makes
    the prediction hold, so the cache path is exercised honestly rather than through that guard."""
    from vidlore.clipstudio.config import ClipConfig
    proj, segs, shots = _mini_project(tmp, n_beats)
    by = {(s.source_id, s.index): s for s in shots}
    calls = {"n": 0}

    def fake_verify_frame(*a, **k):
        calls["n"] += 1
        return None if always_fail else {"verdict": "keep", "confidence": 0.9}

    orig_vf, orig_lookup = V.verify_frame, V._shot_lookup
    orig_env = os.environ.get("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET")
    os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = "1" if sheet else "0"
    V.verify_frame = fake_verify_frame
    V._shot_lookup = lambda p: (lambda sid, ix: by.get((sid, ix)))
    try:
        summ = V.verify_and_repair(proj, segs, ClipConfig(),
                                   NS(anthropic_model="m", anthropic_key="k"), progress=None)
    finally:
        V.verify_frame, V._shot_lookup = orig_vf, orig_lookup
        if orig_env is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = orig_env
    return summ, calls["n"], proj


def test_a_verdict_is_not_cached_when_the_sheet_prediction_did_not_hold():
    """A contact-sheet build can fail and silently fall back to one frame. That answer is to a
    DIFFERENT question than the key claims, so it must not be stored — otherwise a later run gets a
    single-frame judgment back under a multiframe key."""
    tmp = tempfile.mkdtemp()
    try:
        # sheet=True asks for a sheet the 2KB fake media cannot produce -> prediction fails
        summ, n_calls, _ = _run_verify(tmp, always_fail=False, n_beats=3, sheet=True)
        assert summ["verified"] == 3, "the verdicts are still USED"
        assert not os.path.exists(os.path.join(tmp, "verdict_cache.json")), \
            "a mispredicted verdict must not be cached"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_breaker_stops_making_requests():
    tmp = tempfile.mkdtemp()
    try:
        summ, n_calls, proj = _run_verify(tmp, always_fail=True, n_beats=30)
        trip = V.VERIFIER_BREAKER_TRIP
        assert summ["verifier_down"] is True, "breaker must report open"
        # THE point of a breaker: it stops calling. Not "reports that it would have".
        assert n_calls == trip, (
            f"breaker must stop after {trip} consecutive errors; made {n_calls} calls over 30 beats")
        assert summ["breaker_skipped"] == 30 - trip, summ
        assert summ["errored"] == 30, "every beat is unverified"
        assert summ["verified"] == 0, "'verified' counts successes, never attempts"
        # every exact beat is unresolved -> the build gate release-blocks
        assert summ["failed"] == 30, "each exact_scene beat must be unresolved"
        # and each carries an explicit machine-readable reason
        st = {str((s.verifier or {}).get("status")) for s in proj.selections}
        assert st == {"error", "breaker_open"}, st
        n_open = sum(1 for s in proj.selections
                     if (s.verifier or {}).get("status") == "breaker_open")
        assert n_open == 30 - trip, "beats after the trip are marked breaker_open, not merely error"
        assert all(V.FLAG_VERIFIER_UNVERIFIED in (s.flag_reasons or [])
                   for s in proj.selections), "every unverified beat must be flagged"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_healthy_backend_verifies_every_beat_and_caches():
    tmp = tempfile.mkdtemp()
    try:
        summ, n_calls, proj = _run_verify(tmp, always_fail=False, n_beats=10)
        assert summ["verifier_down"] is False
        assert n_calls == 10 and summ["verified"] == 10 and summ["errored"] == 0
        assert summ["failed"] == 0, "a healthy pass must not release-block"
        cache = json.load(open(os.path.join(tmp, "verdict_cache.json")))
        assert len(cache) == 10, "each distinct question caches one verdict"
        # a second run must REUSE, not re-roll — this is what survives a restart
        summ2, n_calls2, _ = _run_verify(tmp, always_fail=False, n_beats=10)
        assert n_calls2 == 0, f"a warm cache must make zero calls, made {n_calls2}"
        assert summ2["reused"] == 10
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_poisoned_cache_entry_is_dropped_not_served():
    tmp = tempfile.mkdtemp()
    try:
        _run_verify(tmp, always_fail=False, n_beats=3)
        p = os.path.join(tmp, "verdict_cache.json")
        cache = json.load(open(p))
        for k in cache:
            cache[k] = {"status": "error"}          # an error stub masquerading as a verdict
        json.dump(cache, open(p, "w"))
        summ, n_calls, _ = _run_verify(tmp, always_fail=False, n_beats=3)
        assert n_calls == 3, "a schema-invalid entry must be re-asked, never served"
        assert summ["reused"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [
    test_every_prompt_affecting_field_invalidates_the_cache,
    test_faceid_names_are_order_independent_but_content_sensitive,
    test_prompt_and_sheet_versions_are_in_the_key,
    test_only_schema_valid_successful_verdicts_are_reusable,
    test_source_fingerprint_sees_any_changed_byte_and_is_memoized,
    test_breaker_stops_making_requests,
    test_healthy_backend_verifies_every_beat_and_caches,
    test_a_poisoned_cache_entry_is_dropped_not_served,
    test_a_verdict_is_not_cached_when_the_sheet_prediction_did_not_hold,
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
