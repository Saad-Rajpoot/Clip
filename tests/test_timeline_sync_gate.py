"""The timeline sync gate, exercised against REAL encoded files.

Builds tiny a/v fixtures with ffmpeg and asserts the gate raises on drift, PTS skew, a missing
stream, and anything unreadable. Pure gate behaviour — no project, no LLM.

    python3 tests/test_timeline_sync_gate.py

Skips (exit 0) if no ffmpeg is available.
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def _ffmpeg():
    from vidlore.clipstudio.config import ffmpeg_exe
    try:
        exe = ffmpeg_exe()
        subprocess.run([exe, "-version"], capture_output=True, timeout=20)
        return exe
    except Exception:
        return ""


def _mk(exe, d):
    """good / drift / skew / video-only fixtures."""
    def run(*a):
        subprocess.run([exe, "-v", "error", "-y", *a], capture_output=True, timeout=90)
    p = lambda n: os.path.join(d, n)  # noqa: E731
    run("-f", "lavfi", "-i", "color=c=blue:s=320x180:d=3:r=30", "-f", "lavfi", "-i", "sine=f=440:d=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", p("good.mp4"))
    # video 3s, audio 2s → the shape of the 1.2s concat overrun that shipped
    run("-f", "lavfi", "-i", "color=c=red:s=320x180:d=3:r=30", "-f", "lavfi", "-i", "sine=f=440:d=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p("drift.mp4"))
    # audio delayed 0.5s → streams do not START together: silent lip-sync error
    run("-f", "lavfi", "-i", "color=c=green:s=320x180:d=3:r=30", "-f", "lavfi", "-i", "sine=f=440:d=3",
        "-af", "adelay=500|500", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p("skew.mp4"))
    run("-f", "lavfi", "-i", "color=c=black:s=320x180:d=3:r=30", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", p("vonly.mp4"))
    return p


def test_delivered_gate_on_real_files():
    from vidlore.assemble import assert_delivered_av_sync, TimelineSyncError
    exe = _ffmpeg()
    if not exe:
        print("SKIP  no ffmpeg")
        return
    d = tempfile.mkdtemp()
    try:
        p = _mk(exe, d)
        r = assert_delivered_av_sync(p("good.mp4"))
        assert abs(r["video"][0] - r["audio"][0]) <= r["tol_s"], r

        for name, why in (("drift.mp4", "a 1s duration drift must not ship"),
                          ("skew.mp4", "streams that do not start together must not ship"),
                          ("vonly.mp4", "a delivered render missing a stream must not ship")):
            try:
                assert_delivered_av_sync(p(name))
                raise AssertionError(why)
            except TimelineSyncError:
                pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_post_concat_conform_retimes_video_to_audio():
    """The finished concat is retimed to the composed-audio frame count before the invariant — a
    belt-and-suspenders over the beat-level accounting, since a small STABLE residual (measured +4
    frames) entered at the breakout-insertion boundary and survived every beat-level fix. Bounded so
    a gross error still reaches the invariant."""
    from vidlore.assemble import _conform_video_to_audio, _probe_duration, FPS
    from types import SimpleNamespace as NS
    exe = _ffmpeg()
    if not exe:
        print("SKIP  no ffmpeg")
        return
    d = tempfile.mkdtemp()
    try:
        def run(*a):
            subprocess.run([exe, "-v", "error", "-y", *a], capture_output=True, timeout=90)
        run("-f", "lavfi", "-i", "sine=f=440", "-t", "5.0333", "-c:a", "pcm_s16le",
            os.path.join(d, "a.wav"))                       # 151 frames
        for name, frames, tag in (("long.mp4", 155, "trim"), ("short.mp4", 148, "pad")):
            run("-f", "lavfi", "-i", "color=c=blue:s=320x180:r=30", "-frames:v", str(frames),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(d, name))
            _conform_video_to_audio(os.path.join(d, name), NS(audio=os.path.join(d, "a.wav")), d)
            vd = _probe_duration(os.path.join(d, name))
            ad = _probe_duration(os.path.join(d, "a.wav"))
            assert abs(vd - ad) <= 1 / FPS, f"{tag}: video {vd} not conformed to audio {ad}"

        # MULTI-BREAKOUT accumulation: 5 breakouts drifted +13 frames (0.443s, video LONG) on a
        # real render — over the OLD 12-frame cap, so it hard-failed the invariant. The cap now
        # covers it (24 frames / 0.8s): a 164-frame video (13 over the 151-frame audio) must conform.
        run("-f", "lavfi", "-i", "color=c=red:s=320x180:r=30", "-frames:v", "164",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(d, "acc.mp4"))
        _conform_video_to_audio(os.path.join(d, "acc.mp4"), NS(audio=os.path.join(d, "a.wav")), d)
        assert abs(_probe_duration(os.path.join(d, "acc.mp4")) - 5.0333) <= 1 / FPS, \
            "a 13-frame multi-breakout accumulation must now conform, not fail the invariant"

        # a GROSS gap must NOT be papered over — it must reach the invariant
        run("-f", "lavfi", "-i", "color=c=green:s=320x180:r=30", "-frames:v", "300",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(d, "gross.mp4"))
        _conform_video_to_audio(os.path.join(d, "gross.mp4"), NS(audio=os.path.join(d, "a.wav")), d)
        assert abs(_probe_duration(os.path.join(d, "gross.mp4")) - 10.0) < 0.2, \
            "a >max_fix_frames gap must be left for the invariant, not silently trimmed"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ad_scan_coverage_tolerates_the_conform_grid_artifact():
    """The timeline-conform trims the video to an exact frame-length to lock A/V sync, which makes
    the ad gate's fps=1/stride extraction land its LAST sample one grid step short (376 frames of a
    188.233s video, last at 187.50s, 0.08s past tolerance) even though every frame decoded. The
    frame COUNT is the reliable coverage signal — a count within one of expected is a grid artifact,
    not a decode gap. Genuine tail gaps (the count also falls short) must still fail."""
    from vidlore.clipstudio.build import _scan_coverage_reason as cov
    # grid artifact: video conformed to 188.233s, 376/377 frames -> covered
    assert cov(376, 0.5, 188.233) is None, "the conform grid artifact must read as covered"
    # genuine gaps must still fail closed
    assert cov(376, 0.5, 200.0) is not None, "a real 12s tail gap must fail"
    assert cov(197, 0.5, 100.0) is not None, "a real ~2s tail gap must fail (count 197 << 201)"
    assert cov(0, 0.5, 100.0) is not None, "zero frames must fail"
    assert cov(200, 0.5, 0.0) is not None, "unprobeable duration must fail"


def test_gate_fails_closed_when_it_cannot_measure():
    """A check that cannot run has NOT passed. The first cut returned early on a missing narration
    or a failed ffprobe — the same shape as the verifier bug this branch exists to fix, where an
    error looked exactly like an approval."""
    from vidlore.assemble import (assert_delivered_av_sync, _probe_duration,
                                  _assert_video_audio_sync, TimelineSyncError)
    from types import SimpleNamespace as NS
    for fn, arg in ((assert_delivered_av_sync, "/nope/none.mp4"),
                    (_probe_duration, "/nope/none.mp4")):
        try:
            fn(arg)
            raise AssertionError(f"{fn.__name__} must raise on an unreadable file, never pass")
        except TimelineSyncError:
            pass
    # a missing composed narration must raise, not skip
    try:
        _assert_video_audio_sync("/tmp/x.mp4", NS(audio=None), "/tmp")
        raise AssertionError("missing narration audio must raise, not silently skip the check")
    except TimelineSyncError as e:
        assert "cannot be verified" in str(e)


TESTS = [test_delivered_gate_on_real_files, test_post_concat_conform_retimes_video_to_audio, test_ad_scan_coverage_tolerates_the_conform_grid_artifact, test_gate_fails_closed_when_it_cannot_measure]

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
