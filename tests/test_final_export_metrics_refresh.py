"""Regression coverage for ClipStudio's post-assemble export fingerprint.

The generic renderer writes its metrics after mux, but ClipStudio can then
re-encode the file and rename review drafts.  The sidecar must describe the
artifact that build_video actually returns, not the earlier assembly boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vidlore.assemble import _write_export_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_refresh_replaces_the_pre_postpass_sha(tmp_path):
    final = tmp_path / "final.mp4"
    final.write_bytes(b"assembled-boundary-bytes")
    before = _write_export_metrics(final, fps=30)

    # Model an in-place letterbox/caption post-pass without invoking ffmpeg.
    final.write_bytes(b"delivered-postprocessed-bytes")
    after = _write_export_metrics(final, fps=30)
    persisted = json.loads((tmp_path / "render_export_metrics.json").read_text())

    assert before["sha256"] != _sha256(final)
    assert after["sha256"] == _sha256(final)
    assert persisted == after


def test_refresh_records_the_returned_review_draft_path(tmp_path):
    final = tmp_path / "final.mp4"
    final.write_bytes(b"review-draft-bytes")
    _write_export_metrics(final, fps=30)

    draft = tmp_path / "final.REVIEW_DRAFT.mp4"
    final.replace(draft)
    refreshed = _write_export_metrics(draft, fps=30)
    persisted = json.loads((tmp_path / "render_export_metrics.json").read_text())

    assert refreshed["final_video"] == draft.name
    assert refreshed["final_path"] == str(draft.resolve())
    assert refreshed["sha256"] == _sha256(draft)
    assert persisted == refreshed


def test_build_video_refreshes_after_all_mutations_and_before_return():
    src = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py") \
        .read_text(encoding="utf-8")
    body = src.split("def build_video(", 1)[1]

    refresh = body.rindex("_write_final_export_metrics(result, _EXPORT_FPS)")
    assert refresh > body.index("_apply_cinematic_letterbox(result")
    assert refresh > body.index("_burn_breakout_captions(")
    assert refresh > body.index("_final_video_black_gate(result")
    assert refresh > body.index("_verify_delivered_lineage(")
    assert refresh > body.index("_sync = _avsync(result)")
    assert refresh > body.index("result = _draft")
    assert refresh < body.index('log(f"build: done')
