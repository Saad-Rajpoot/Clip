"""Sub-HD footage and an ffprobe that did not run are not the same failure.

The native-HD gate collapsed both into one verdict — `passed: False` — and then raised one
NonRetryableBuildError for the lot. Two consequences:

  * The call site in build_video sat in no try block, so in REVIEW mode it killed the build exactly
    as it does in production, and the portal's auto-review resumed straight back into it. A render
    could never deliver the one thing this gate is best placed to show a human: which beats are
    genuinely below 720p.
  * If it HAD been forgiven wholesale, a missing ffprobe, a deleted source file or a selection with
    no source bound would have ridden out inside a review draft as "imperfect footage". That is the
    fail-open class that once hid a dead recovery path for months.

So the classes are separated at the point of measurement:

    measured_sub_hd  real bytes, real numbers, genuinely below the floor  -> CONTENT
    unprobeable      a real file is there and ffprobe told us nothing     -> TECHNICAL
    no_source        nothing to measure at all                            -> TECHNICAL

Only the first may reach a review draft, and only in review mode. No threshold moves, no selection
is swapped, and native_resolution_audit.json names every failing beat either way.

Say plainly what a delivered draft contains: those beats reach the encode, and build._upscale_filter
lanczos-enlarges anything under 1280 wide onto the 1080p canvas. The draft therefore AIRS upscaled
SD. That is not a way to pass the gate — the gate failed, the file is renamed REVIEW_DRAFT, the log
line says so and the audit names the beats — it is the only way to show a human which beats need
better footage. Production still refuses to produce the file at all.
"""
from __future__ import annotations

import inspect
import json

import pytest

from vidlore.clipstudio import quality_contract as Q
from vidlore.clipstudio import release_policy as RP
from vidlore.clipstudio.verify import NonRetryableBuildError


class _Src:
    def __init__(self, sid, path):
        self.id, self.local_path, self.title = sid, str(path), "t"


class _Proj:
    def __init__(self, sources):
        self._s = {s.id: s for s in sources}

    def source(self, sid):
        return self._s.get(sid)


class _Sel:
    def __init__(self, idx, sid):
        self.segment_index, self.source_id, self.image_path = idx, sid, ""


def _run(tmp_path, monkeypatch, probes: dict, files: dict):
    """probes: path -> {'width','height'} or {} ; files: path -> size on disk (0 = absent)."""
    for path, size in files.items():
        if size:
            (tmp_path / path).write_bytes(b"x" * size)
    def fake_probe(p):
        return dict(probes.get(str(p).rsplit("/", 1)[-1], {}))
    monkeypatch.setattr(Q, "probe_native_video_info", fake_probe)
    srcs = [_Src(f"s{i}", tmp_path / name) for i, name in enumerate(files)]
    sels = [_Sel(i, f"s{i}") for i in range(len(srcs))]
    audit = tmp_path / "native_resolution_audit.json"
    return _Proj(srcs), sels, audit


def test_measured_sub_hd_is_a_content_verdict(tmp_path, monkeypatch):
    proj, sels, audit = _run(tmp_path, monkeypatch,
                             {"a.mp4": {"width": 640, "height": 360}}, {"a.mp4": 10})
    with pytest.raises(NonRetryableBuildError) as ei:
        Q.assert_native_hd_selections(proj, sels, audit)
    assert ei.value.kind == "native_resolution"
    assert RP.gate_class(ei.value) == "content"
    assert "640x360" in str(ei.value)
    rows = json.loads(audit.read_text())["failures"]
    assert rows[0]["failure_class"] == Q.MEASURED_SUB_HD


def test_an_unprobeable_file_is_technical_and_fatal_in_both_modes(tmp_path, monkeypatch):
    """A real file that ffprobe would not read. We do not know what it is — so we do not ship it."""
    proj, sels, audit = _run(tmp_path, monkeypatch, {"a.mp4": {}}, {"a.mp4": 10})
    with pytest.raises(NonRetryableBuildError) as ei:
        Q.assert_native_hd_selections(proj, sels, audit)
    assert ei.value.kind == "native_resolution_probe"
    assert RP.gate_class(ei.value) == "technical"
    assert json.loads(audit.read_text())["failures"][0]["failure_class"] == Q.UNPROBEABLE


def test_a_missing_source_file_is_technical(tmp_path, monkeypatch):
    proj, sels, audit = _run(tmp_path, monkeypatch, {"a.mp4": {}}, {"a.mp4": 0})
    with pytest.raises(NonRetryableBuildError) as ei:
        Q.assert_native_hd_selections(proj, sels, audit)
    assert ei.value.kind == "native_resolution_probe"
    assert json.loads(audit.read_text())["failures"][0]["failure_class"] == Q.NO_SOURCE


def test_one_unmeasured_row_poisons_a_batch_of_genuine_sub_hd(tmp_path, monkeypatch):
    """The important ordering: an unmeasured selection must NOT ride out inside a content batch."""
    proj, sels, audit = _run(
        tmp_path, monkeypatch,
        {"a.mp4": {"width": 640, "height": 360}, "b.mp4": {"width": 854, "height": 480},
         "c.mp4": {}},
        {"a.mp4": 10, "b.mp4": 10, "c.mp4": 10})
    with pytest.raises(NonRetryableBuildError) as ei:
        Q.assert_native_hd_selections(proj, sels, audit)
    assert ei.value.kind == "native_resolution_probe", "the unmeasured row must win"
    assert "1 of 3" in str(ei.value)


def test_hd_footage_still_passes(tmp_path, monkeypatch):
    proj, sels, audit = _run(tmp_path, monkeypatch,
                             {"a.mp4": {"width": 1920, "height": 1080}}, {"a.mp4": 10})
    Q.assert_native_hd_selections(proj, sels, audit)          # no raise
    doc = json.loads(audit.read_text())
    assert doc["passed"] is True and not doc["failures"]


def test_a_narrow_raster_is_still_sub_hd(tmp_path, monkeypatch):
    """640x720 clears the height floor and still needs a 2x horizontal enlargement."""
    proj, sels, audit = _run(tmp_path, monkeypatch,
                             {"a.mp4": {"width": 640, "height": 720}}, {"a.mp4": 10})
    with pytest.raises(NonRetryableBuildError) as ei:
        Q.assert_native_hd_selections(proj, sels, audit)
    assert ei.value.kind == "native_resolution"


# ---------------------------------------------------------------- the delivery decision
@pytest.mark.parametrize("mode,kind,expected", [
    ("warn", "native_resolution", True),            # measured sub-HD: a human can see it
    ("warn", "native_resolution_probe", False),     # unmeasured: never
    ("block", "native_resolution", False),          # production delivers nothing imperfect
    ("block", "native_resolution_probe", False),
])
def test_only_measured_sub_hd_reaches_a_review_draft(monkeypatch, mode, kind, expected):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", mode)
    from vidlore.clipstudio import build as B
    assert B.content_defect_is_deliverable(NonRetryableBuildError("x", kind=kind)) is expected


def test_the_build_call_site_reraises_anything_undeliverable():
    from vidlore.clipstudio import build as B
    src = inspect.getsource(B.build_video)
    i = src.index("_assert_native_hd(proj, proj.selections")
    block = src[i:i + 3000]
    assert "if not content_defect_is_deliverable(_hd_exc):" in block
    assert "raise" in block
    assert "_review_draft.append(" in block, "a delivered defect must be reported"
    # and it must not quietly swap or upscale anything — checked on CODE, since the comment above
    # it says the word "upscaled" and matching prose is the mistake this whole session removed
    code = "\n".join(ln for ln in block.splitlines() if not ln.lstrip().startswith("#"))
    for forbidden in ("upscale", "image_path =", "sel.source_id =", "resize"):
        assert forbidden not in code, forbidden


def test_the_audit_is_written_before_either_raise():
    src = inspect.getsource(Q.assert_native_hd_selections)
    assert src.index("os.replace(tmp, audit_path)") < src.index("if failures:")


def test_a_failed_probe_is_never_memoized():
    """One flaky ffprobe must not become a permanent publication verdict. The gate used to cache
    the empty result in a local dict; it now goes through the function that refuses to."""
    src = inspect.getsource(Q.assert_native_hd_selections)
    assert "probe_native_video_info(path)" in src
    assert "cache[path] = {}" not in src
    assert "except Exception" not in src, "no bare swallow around the measurement"


# ------------------------------------------------- one measurement per path, with one retry
def test_a_flaky_probe_cannot_produce_two_verdicts_for_one_file(tmp_path, monkeypatch):
    """Probing per SELECTION instead of per PATH let the same bytes get contradictory rows in one
    audit — a transient miss on beat 0 and a clean 1920x1080 on beat 1 — and the unmeasured row
    wins the raise. One flaky ffprobe would then turn a deliverable draft into a hard failure,
    which is the exact outcome this whole change exists to remove."""
    (tmp_path / "a.mp4").write_bytes(b"x" * 10)
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        return {} if calls["n"] == 1 else {"width": 1920, "height": 1080}

    monkeypatch.setattr(Q, "probe_native_video_info", flaky)
    src = _Src("s0", tmp_path / "a.mp4")
    sels = [_Sel(0, "s0"), _Sel(1, "s0"), _Sel(2, "s0")]
    audit = tmp_path / "native_resolution_audit.json"
    Q.assert_native_hd_selections(_Proj([src]), sels, audit)      # the retry finds the real bytes
    doc = json.loads(audit.read_text())
    assert doc["passed"] is True
    assert {(r["width"], r["height"]) for r in doc["selections"]} == {(1920, 1080)}, \
        "every row for one path must carry the same measurement"


def test_the_retry_is_bounded_to_one_per_path(tmp_path, monkeypatch):
    """A corrupt source bound to many beats must not pay one ffprobe per beat."""
    (tmp_path / "a.mp4").write_bytes(b"x" * 10)
    calls = {"n": 0}

    def dead(p):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(Q, "probe_native_video_info", dead)
    src = _Src("s0", tmp_path / "a.mp4")
    sels = [_Sel(i, "s0") for i in range(9)]
    with pytest.raises(NonRetryableBuildError) as ei:
        Q.assert_native_hd_selections(_Proj([src]), sels, tmp_path / "a.json")
    assert ei.value.kind == "native_resolution_probe"
    assert calls["n"] == 2, f"expected 1 probe + 1 retry for one path, got {calls['n']}"


def test_a_good_file_is_probed_once_for_many_beats(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"x" * 10)
    calls = {"n": 0}

    def ok(p):
        calls["n"] += 1
        return {"width": 1920, "height": 1080}

    monkeypatch.setattr(Q, "probe_native_video_info", ok)
    Q.assert_native_hd_selections(_Proj([_Src("s0", tmp_path / "a.mp4")]),
                                  [_Sel(i, "s0") for i in range(7)], tmp_path / "a.json")
    assert calls["n"] == 1


# ------------------------------------------------- the draft must admit what it contains
def test_the_delivered_draft_says_the_beats_air_upscaled(monkeypatch):
    """build._upscale_filter lanczos-enlarges anything under 1280 wide onto the 1080p canvas, so a
    delivered draft DOES air upscaled SD. Claiming otherwise in the code was wrong; the reviewer
    who opens the draft has to be told, in the log and in the review-draft ledger."""
    from vidlore.clipstudio import build as B
    src = inspect.getsource(B.build_video)
    i = src.index("_assert_native_hd(proj, proj.selections")
    block = src[i:i + 3000]
    assert "UPSCALED" in block, "the review-draft note must say the beats air upscaled"
    assert "_review_draft.append(_msg_nhd)" in block
    # the filter it is telling the truth about really does exist and really does fire
    up = inspect.getsource(B._upscale_filter)
    assert "src_w < 1280" in up and "lanczos" in up


def test_no_comment_claims_nothing_is_upscaled():
    """The adversarial review caught this exact false claim in three files. Keep it dead."""
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    # assembled from parts so this file does not trip its own assertion
    phrase = " ".join(["nothing", "is", "upscaled"])
    for rel in ("vidlore/clipstudio/build.py", "vidlore/clipstudio/quality_contract.py",
                "tests/test_native_hd_failure_classes.py"):
        text = (root / rel).read_text(encoding="utf-8").lower()
        assert phrase not in text, f"{rel} still claims it"
        assert phrase.replace(" is ", " is\n        # ") not in text, rel   # wrapped across lines
