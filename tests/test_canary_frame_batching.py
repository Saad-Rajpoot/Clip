"""The timeline canary must be able to RUN on a long video.

Job d835faa83e died at the final gate after 5.8 hours:

    FATAL: scene-lineage canary failed at timeline_order: 1 violation(s);
      first: cannot decode timeline canaries from video_only.mp4: rc=244, bytes=0/411264
      [Parsed_select_0] Error while parsing expression 'eq(n,0)+eq(n,57)+...'

Nothing was wrong with the video. The check selects frames with one `eq(n,X)` term each, and
ffmpeg's expression parser refuses the list past a fixed size — bisected against that render's own
483 MB intermediate: **100 frames parse, 101 do not**. A 252-beat video asks for far more, so the
decode returned zero bytes, and a canary that cannot decode fails closed. Correctly: it cannot
prove anything. But it was reporting a LINEAGE violation for an ffmpeg limit, and it blamed the
footage for it.

Fail-closed stays. What changes is that the check can now run at any length. Verified on the real
intermediate after the fix: 101 frames in 2s, 200 in 8s, 400 in 25s, 800 in 77s — all previously
impossible past 100.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vidlore import scene_lineage_canary as C


class _Run:
    """Record every ffmpeg invocation and answer with well-formed feature bytes."""

    def __init__(self, fail_on=None):
        self.batches: list[list[int]] = []
        self.fail_on = fail_on

    def __call__(self, args, capture_output=True, timeout=None):
        expr = next(a for a in args if a.startswith("select='"))
        frames = [int(n) for n in re.findall(r"eq\(n\\,(\d+)\)", expr)]
        self.batches.append(frames)
        n = int(args[args.index("-frames:v") + 1])
        if self.fail_on is not None and len(self.batches) == self.fail_on:
            return _Proc(244, b"")
        return _Proc(0, b"\x01" * (n * C._FEATURE_BYTES))


class _Proc:
    def __init__(self, rc, out):
        self.returncode, self.stdout, self.stderr = rc, out, b"boom"


@pytest.fixture
def run(monkeypatch):
    r = _Run()
    monkeypatch.setattr(C.subprocess, "run", r)
    return r


# ---------------------------------------------------------------- the render that died
def test_a_long_video_is_decoded_instead_of_refused(run):
    frames = [i * 57 for i in range(300)]
    out = C._features_by_frames(Path("v.mp4"), frames)
    assert set(out) == set(frames), "every requested canary must come back"
    assert len(run.batches) > 1, "300 frames in one call is the expression ffmpeg refuses"


def test_no_batch_can_exceed_the_measured_ceiling(run):
    C._features_by_frames(Path("v.mp4"), [i for i in range(1000)])
    assert run.batches, "nothing was decoded"
    worst = max(len(b) for b in run.batches)
    assert worst <= C._MAX_SELECT_TERMS <= 100, \
        f"a batch of {worst} terms — ffmpeg parses 100 and refuses 101"


def test_a_short_video_is_untouched(run):
    """The overwhelming majority of renders were never affected and must not change."""
    frames = [1, 2, 3, 9]
    out = C._features_by_frames(Path("v.mp4"), frames)
    assert len(run.batches) == 1, "a small bank must still be one ffmpeg pass"
    assert run.batches[0] == frames
    assert set(out) == set(frames)


def test_frames_are_deduped_and_ordered_across_batches(run):
    out = C._features_by_frames(Path("v.mp4"), [50, 10, 10, 200, 3] + list(range(100)))
    assert set(out) == {50, 10, 200, 3} | set(range(100))
    seen = [n for b in run.batches for n in b]
    assert seen == sorted(set(seen)), "batches must partition the sorted frame list"


# ---------------------------------------------------------------- fail-closed is preserved
def test_one_bad_batch_still_fails_the_whole_canary(monkeypatch):
    """A batch that cannot decode must raise, not be quietly dropped — otherwise batching would
    turn a fail-closed proof into a partial one, which is far worse than the bug it fixes."""
    r = _Run(fail_on=2)
    monkeypatch.setattr(C.subprocess, "run", r)
    with pytest.raises(C.SceneLineageError, match="cannot decode timeline canaries"):
        C._features_by_frames(Path("v.mp4"), list(range(300)))


def test_no_frames_asks_nothing(run):
    assert C._features_by_frames(Path("v.mp4"), []) == {}
    assert run.batches == []


def test_the_ceiling_is_below_ffmpeg_s_own_limit():
    """Bisected at 100/101 on a real render. The constant is deliberately lower, because the limit
    belongs to ffmpeg and may differ by build — and overshooting does not degrade gracefully, it
    returns zero bytes and fails the entire canary."""
    assert C._MAX_SELECT_TERMS < 100
