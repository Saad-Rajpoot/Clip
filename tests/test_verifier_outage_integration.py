"""Verifier outage: the render must not publish. Tiny, no-encode integration test.

Reproduces the exact condition that shipped the 15:24 video — the vision backend returns nothing —
and asserts the four things that were all false at the time:

    zero publication · exact beats unverified · the breaker actually stops calling · retryable=false

Deliberately NOT a second video render: the interesting failure happens before a single frame is
encoded, and a render would only make it slower to observe.

    python3 tests/test_verifier_outage_integration.py
"""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILS = []


def _project(tmp, n_beats=20):
    from vidlore.clipstudio.models import (ClipProject, ScriptSegment, ClipSelection,
                                           SourceVideo, Shot)
    proj = ClipProject(name="outage", root=str(tmp))
    proj.ensure_dirs()
    media = os.path.join(tmp, "src.mp4")
    open(media, "wb").write(b"\0" * 4096)
    proj.sources = [SourceVideo(id="s1", url="u", title="Game of Thrones S03E10 small council",
                               permission="owner", status="ok", local_path=media)]
    segs, shots = [], []
    for i in range(n_beats):
        # every beat is exact_scene — the class that must never pass unverified
        segs.append(ScriptSegment(index=i, text=f"Tywin dismisses Joffrey, beat {i}",
                                  required_entity="Tywin Lannister", required_kind="character",
                                  visual_policy="exact_scene", is_specific_claim=True))
        kf = os.path.join(tmp, f"kf{i}.jpg")
        open(kf, "wb").write(b"\xff\xd8\xff")
        shots.append(Shot(source_id="s1", index=i, start=float(i), end=float(i) + 2.0,
                          keyframe_path=kf))
        proj.selections.append(ClipSelection(segment_index=i, source_id="s1", shot_index=i,
                                             in_point=float(i), out_point=float(i) + 2.0,
                                             confidence=0.9))
    proj.meta["analysis"] = {"video_type": "single_scene", "episode_hint": "S03E10",
                             "episode_hint_verified": True, "characters": [], "actors": []}
    return proj, segs, shots


def test_total_vision_outage_blocks_publication():
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio.config import ClipConfig
    tmp = tempfile.mkdtemp()
    try:
        proj, segs, shots = _project(tmp, 20)
        by = {(s.source_id, s.index): s for s in shots}
        calls = {"n": 0}

        def dead_backend(*a, **k):
            calls["n"] += 1
            return None                       # exactly what a vision outage looks like

        o1, o2 = V.verify_frame, V._shot_lookup
        os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = "0"
        V.verify_frame = dead_backend
        V._shot_lookup = lambda p: (lambda sid, ix: by.get((sid, ix)))
        try:
            summ = V.verify_and_repair(proj, segs, ClipConfig(),
                                       NS(anthropic_model="m", anthropic_key="k"), progress=None)
        finally:
            V.verify_frame, V._shot_lookup = o1, o2

        # 1) the summary must not be able to say "nothing wrong". The shipped run reported
        #    "229 checked, 0 replaced, 0 unresolved" and published.
        assert summ["verified"] == 0, "'verified' counts successes; an outage verifies nothing"
        assert summ["errored"] == 20
        assert summ["failed"] == 20, "every exact beat must be UNRESOLVED, not silently accepted"
        assert summ["verifier_down"] is True
        assert summ["verified_frac"] == 0.0

        # 2) the breaker actually stopped calling
        assert calls["n"] == V.VERIFIER_BREAKER_TRIP, (
            f"breaker must stop the backend after {V.VERIFIER_BREAKER_TRIP} consecutive errors; "
            f"made {calls['n']} calls across 20 beats")

        # 3) every beat carries a machine-readable unverified status
        for s in proj.selections:
            assert (s.verifier or {}).get("status") in ("error", "breaker_open"), s.verifier
            assert V.FLAG_VERIFIER_UNVERIFIED in (s.flag_reasons or [])
            assert s.flagged

        # 4) the BUILD gate refuses to publish. Assert the gate's own predicate on this state
        #    rather than paying for an encode: an exact beat, no still, status=error/breaker_open.
        from vidlore.clipstudio import policy as P
        blocked = [s.segment_index for s, seg in zip(proj.selections, segs)
                   if P.verify_strict(seg) and not getattr(s, "image_path", "")
                   and str((s.verifier or {}).get("status")) in ("error", "unavailable",
                                                                 "breaker_open")]
        assert len(blocked) == 20, f"the unverified-exact gate must block all 20, got {len(blocked)}"

        src = open(os.path.join(ROOT, "vidlore", "clipstudio", "build.py"), encoding="utf-8").read()
        assert "UNVERIFIED-EXACT GATE" in src and "NonRetryableBuildError" in src
    finally:
        os.environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_outage_is_reported_as_non_retryable_at_the_job_boundary():
    """Retrying an outage is what published the render: 8 restarts until the API died, at which
    point 0 rejections read as 'nothing wrong'."""
    from vidlore.clipstudio.verify import NonRetryableBuildError
    src = open(os.path.join(ROOT, "tools", "rerender_project.py"), encoding="utf-8").read()
    assert "NonRetryableBuildError" in src and '"retryable": not _content' in src
    assert "_rerender.status.json" in src or "STATUS" in src
    assert issubclass(NonRetryableBuildError, RuntimeError)


def test_a_healthy_backend_still_publishes():
    """The gate must not simply block everything — that would be a different bug wearing the same
    green tick."""
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio.config import ClipConfig
    tmp = tempfile.mkdtemp()
    try:
        proj, segs, shots = _project(tmp, 6)
        by = {(s.source_id, s.index): s for s in shots}
        o1, o2 = V.verify_frame, V._shot_lookup
        os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = "0"
        V.verify_frame = lambda *a, **k: {"verdict": "keep", "confidence": 0.9,
                                          "matches_narration": True, "quality_ok": True}
        V._shot_lookup = lambda p: (lambda sid, ix: by.get((sid, ix)))
        try:
            summ = V.verify_and_repair(proj, segs, ClipConfig(),
                                       NS(anthropic_model="m", anthropic_key="k"), progress=None)
        finally:
            V.verify_frame, V._shot_lookup = o1, o2
        assert summ["failed"] == 0 and summ["verified"] == 6 and summ["verifier_down"] is False
        blocked = [s for s in proj.selections
                   if str((s.verifier or {}).get("status")) in ("error", "breaker_open")]
        assert not blocked, "a healthy pass must leave nothing unverified"
    finally:
        os.environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", None)
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [test_total_vision_outage_blocks_publication,
         test_outage_is_reported_as_non_retryable_at_the_job_boundary,
         test_a_healthy_backend_still_publishes]

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
