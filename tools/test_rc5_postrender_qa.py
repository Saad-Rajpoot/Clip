"""RC5 — post-render relevance-QA WIRING regression (pipeline.finalize_relevance_qa).

The bounded frame-sweep `vidlore.relevance_qa.sweep` is the last-line backstop
that catches an off-topic / designed-graphic frame which slipped every
in-pipeline selection gate (incl. the MG/portrait/Review-Editor producers that
never hit the selection gate). RC5 WIRES it into the shared render path via
`pipeline.finalize_relevance_qa(video, run_dir)`, which:

  * runs the sweep on the finished MP4 (here the sweep is MONKEYPATCHED so no
    real render / no CLIP / no ffmpeg is needed),
  * enriches each flag from render_meta.json + ASSET_DECISION_MANIFEST.json,
  * writes <run_dir>/render_relevance_qa.json  (schema render_relevance_qa/1),
  * records the verdict into render_metrics.json  ("relevance_qa" block),
  * NEVER raises and NEVER aborts the render (report + flag; the MP4 stands).

Run (no flags needed — the sweep is mocked):

    python3 tools/test_rc5_postrender_qa.py

HARD requirements (any failure = regression):
  1. sweep PASS                 → render_relevance_qa.json verdict PASS_RELEVANCE_QA
  2. sweep FAIL (anime frame)   → report + render_metrics carry FAIL_RELEVANCE_QA
                                  AND the flag detail (timestamp/scene/reason)
  3. game-UI flag               → FAIL (issue_class REJECT_GAME_UI)
  4. multilingual-sign flag     → FAIL (issue_class REJECT_TEXT_HEAVY)
  5. wrong-era flag             → FAIL (issue_class REJECT_OFF_TOPIC)
  6. low-confidence flag        → documented rule: a REAL flag is a FAIL; a sweep
                                  that only NOTES low confidence (no flag) is a
                                  degraded PASS-with-note, never a silent pass
  7. scorer-unavailable (error) → NOT a silent/false clean PASS: error preserved
  8. the report file is ALWAYS written when the sweep runs
Plus: VIDLORE_POSTRENDER_QA=0 disables (skipped, nothing written), and the
finalize call never raises even when the sweep itself raises.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Disable the real CLIP scorer / any heavy import side effects up front; the
# sweep is monkeypatched anyway so this just keeps the import cheap + offline.
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "0")
os.environ["VIDLORE_POSTRENDER_QA"] = "1"            # default ON for the tests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore import pipeline as P          # noqa: E402
from vidlore import relevance_qa as RQA    # noqa: E402

_FAILS = []


def _check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _FAILS.append(msg)


# --------------------------------------------------------------------------- #
# Fixtures — a run dir with a fake MP4, render_meta.json, ASSET_DECISION_MANIFEST.
# --------------------------------------------------------------------------- #
def _make_run_dir(tmp: Path) -> Path:
    run = Path(tempfile.mkdtemp(prefix="rc5qa_", dir=str(tmp)))
    # Fake "final" MP4 (the sweep is mocked so contents don't matter; but the
    # file must EXIST for a realistic finalize path).
    (run / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake-mp4-bytes")
    # render_meta.json — scene_starts so a flagged scene index is resolvable.
    (run / "render_meta.json").write_text(json.dumps({
        "schema": "render_meta/1",
        "scenes": 4,
        "video_seconds": 48.0,
        "scene_starts": [0.0, 12.0, 24.0, 36.0],
        "scene_durations": [12.0, 12.0, 12.0, 12.0],
    }), encoding="utf-8")
    # ASSET_DECISION_MANIFEST.json — per-beat records for enrichment.
    (run / "ASSET_DECISION_MANIFEST.json").write_text(json.dumps({
        "summary": {},
        "beats": [
            {"scene": 0, "beat": 0, "narration": "The empire began quietly.",
             "source": "pexels_city.mp4", "role": "establish",
             "verdict": "ACCEPT", "outcome": "accepted", "confidence": "high"},
            {"scene": 1, "beat": 0, "narration": "A secret network formed.",
             "source": "wikimedia_map.jpg", "role": "broll",
             "verdict": "ACCEPT", "outcome": "accepted", "confidence": "ok"},
            {"scene": 2, "beat": 0, "narration": "Money moved across borders.",
             "source": "webimg_anime_cover.jpg", "role": "broll",
             "verdict": "ACCEPT", "outcome": "accepted", "confidence": "ok"},
            {"scene": 3, "beat": 0, "narration": "The fall came suddenly.",
             "source": "archive_photo.jpg", "role": "broll",
             "verdict": "ACCEPT", "outcome": "accepted", "confidence": "ok"},
        ],
    }), encoding="utf-8")
    return run


def _install_sweep(monkey_result):
    """Monkeypatch BOTH the relevance_qa.sweep symbol and the name pipeline
    imports lazily (`from . import relevance_qa`) — since finalize does the
    import inside the function, patching the module attribute is sufficient.

    RC5.1: the mock signature now accepts `expected_sha256`/`expected_path` (the
    finalize hook passes the served-export hash), and — unless the canned result
    already provides them — DEFAULTS `scanned_sha256` to the echoed
    `expected_sha256` (i.e. simulate the live pipeline where the sweep scans the
    SAME file it was told to expect, so finalize records hash_match=True). A case
    that wants a stale scan supplies its own `scanned_sha256` / verdict /
    stale_output in `monkey_result` to override the default."""
    def _fake_sweep(mp4_path, render_meta, *, ffmpeg=None, manifest=None,
                    interval_s=None, max_frames=None,
                    expected_sha256=None, expected_path=None):
        # Echo back the canned result; prove the wiring passes real args.
        assert Path(str(mp4_path)).name == "video.mp4"
        out = dict(monkey_result)
        out.setdefault("expected_sha256", str(expected_sha256 or ""))
        out.setdefault("expected_path", str(expected_path or ""))
        # Default: the sweep scanned the same file finalize asked it to expect.
        out.setdefault("scanned_sha256", str(expected_sha256 or ""))
        out.setdefault("scanned_path", str(expected_path or mp4_path))
        out.setdefault("scanned_mtime", 1234567.0)
        out.setdefault("stale_output", False)
        return out
    RQA.sweep = _fake_sweep


# Keep a handle to the genuine sweep so we can restore it between cases.
_REAL_SWEEP = RQA.sweep


def _restore_sweep():
    RQA.sweep = _REAL_SWEEP


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def case_pass(tmp):
    print("\n[1] sweep PASS → PASS_RELEVANCE_QA, report written")
    run = _make_run_dir(tmp)
    _install_sweep({"verdict": "PASS", "flags": [], "sampled": 9,
                    "duration_s": 48.0, "error": ""})
    rep = P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()

    rpath = run / "render_relevance_qa.json"
    _check(rpath.exists(), "render_relevance_qa.json was written")
    disk = json.loads(rpath.read_text())
    _check(disk.get("schema") == "render_relevance_qa/1",
           f"schema is render_relevance_qa/1 (got {disk.get('schema')})")
    _check(disk.get("verdict") == "PASS_RELEVANCE_QA",
           f"verdict PASS_RELEVANCE_QA (got {disk.get('verdict')})")
    _check(disk.get("flags") == [], "no flags on a clean sweep")
    _check(disk.get("sampled") == 9, "sampled count carried through")
    _check(disk.get("generated_by") == "post_render_sweep",
           "generated_by == post_render_sweep")
    # render_metrics.json carries the verdict (created since absent).
    mpath = run / "render_metrics.json"
    _check(mpath.exists(), "render_metrics.json created")
    m = json.loads(mpath.read_text())
    _check(m.get("relevance_qa", {}).get("verdict") == "PASS_RELEVANCE_QA",
           "render_metrics relevance_qa.verdict == PASS_RELEVANCE_QA")
    _check(rep.get("verdict") == "PASS_RELEVANCE_QA",
           "returned report verdict matches disk")
    # RC5.1 — provenance: the report records the SERVED export hash, the SCANNED
    # file hash, and hash_match=True (the mock scans the same file finalize asked
    # it to expect, mirroring the live pipeline).
    _check(bool(disk.get("served_sha256")), "report records served_sha256")
    _check(bool(disk.get("scanned_sha256")), "report records scanned_sha256")
    _check(disk.get("served_sha256") == disk.get("scanned_sha256"),
           "served_sha256 == scanned_sha256 (QA verified the served export)")
    _check(disk.get("hash_match") is True,
           f"hash_match True on a matching scan (got {disk.get('hash_match')})")
    _check(disk.get("stale_output") is False, "stale_output False on a clean match")
    _check(str(disk.get("served_path", "")).endswith("video.mp4"),
           "served_path points at the served export")
    # render_metrics carries the provenance too.
    _check(m.get("relevance_qa", {}).get("hash_match") is True,
           "render_metrics relevance_qa.hash_match True")


def _fail_case(tmp, n, title, flag, want_issue):
    print(f"\n[{n}] {title}")
    run = _make_run_dir(tmp)
    _install_sweep({"verdict": "FAIL_RELEVANCE_QA", "flags": [flag],
                    "sampled": 12, "duration_s": 48.0, "error": ""})
    P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()

    rpath = run / "render_relevance_qa.json"
    _check(rpath.exists(), "render_relevance_qa.json was written")
    disk = json.loads(rpath.read_text())
    _check(disk.get("verdict") == "FAIL_RELEVANCE_QA",
           f"verdict FAIL_RELEVANCE_QA (got {disk.get('verdict')})")
    flags = disk.get("flags") or []
    _check(len(flags) == 1, f"exactly one enriched flag (got {len(flags)})")
    if flags:
        f0 = flags[0]
        _check(f0.get("timestamp") == flag["timestamp"],
               f"flag timestamp preserved ({f0.get('timestamp')})")
        _check(f0.get("scene") == flag["scene"],
               f"flag scene preserved ({f0.get('scene')})")
        _check(bool(f0.get("reason")),
               f"flag reason present ({str(f0.get('reason'))[:40]!r})")
        _check(f0.get("issue_class") == want_issue,
               f"issue_class == {want_issue} (got {f0.get('issue_class')})")
        # Enrichment from the manifest (scene 2 → webimg_anime_cover.jpg etc.).
        _check("narration" in f0 and "asset_path" in f0,
               "flag enriched with narration + asset_path keys")
    # render_metrics carries FAIL so the render is NOT silently approved.
    m = json.loads((run / "render_metrics.json").read_text())
    _check(m.get("relevance_qa", {}).get("verdict") == "FAIL_RELEVANCE_QA",
           "render_metrics relevance_qa.verdict == FAIL_RELEVANCE_QA")
    _check(m.get("relevance_qa", {}).get("flags") == 1,
           "render_metrics relevance_qa.flags == 1")


def case_fail_anime(tmp):
    _fail_case(
        tmp, 2, "sweep FAIL (anime-cover frame) → FAIL + flag detail",
        {"timestamp": 26.4, "scene": 2,
         "reason": "designed-graphic/text-board — anime cover illustration "
                   "(graphic_dom=0.21 > 0.036)",
         "suggestion": "replace with real footage / grounded still"},
        "REJECT_CARTOON_ANIME")


def case_fail_game_ui(tmp):
    _fail_case(
        tmp, 3, "game-UI flag → FAIL",
        {"timestamp": 14.1, "scene": 1,
         "reason": "designed-graphic — game HUD / on-screen interface screenshot",
         "suggestion": "swap for a relevant documentary visual"},
        "REJECT_GAME_UI")


def case_fail_multilingual_sign(tmp):
    _fail_case(
        tmp, 4, "multilingual-sign flag → FAIL",
        {"timestamp": 30.8, "scene": 2,
         "reason": "text-board — foreign-language sign / caption text dominates "
                   "the frame",
         "suggestion": "replace with footage of the place, not its signage"},
        "REJECT_TEXT_HEAVY")


def case_fail_wrong_era(tmp):
    _fail_case(
        tmp, 5, "wrong-era flag → FAIL",
        {"timestamp": 8.2, "scene": 0,
         "reason": "off-topic — modern wrong-era vehicle in a pre-1945 scene "
                   "(anachronism)",
         "suggestion": "use period-correct footage for the scene's era"},
        "REJECT_OFF_TOPIC")


def case_low_confidence(tmp):
    print("\n[6] low-confidence — documented rule (flag = FAIL; note-only = "
          "degraded PASS, never silent)")
    # 6a — an ACTUAL low-confidence FLAG is a real finding → FAIL.
    run = _make_run_dir(tmp)
    _install_sweep({"verdict": "FAIL_RELEVANCE_QA",
                    "flags": [{"timestamp": 20.0, "scene": 1,
                               "reason": "low-confidence designed-graphic probe "
                                         "(uncertain)",
                               "suggestion": "human review"}],
                    "sampled": 10, "duration_s": 48.0, "error": ""})
    P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()
    disk = json.loads((run / "render_relevance_qa.json").read_text())
    _check(disk.get("verdict") == "FAIL_RELEVANCE_QA",
           "low-confidence FLAG → FAIL_RELEVANCE_QA")
    if disk.get("flags"):
        _check(disk["flags"][0].get("issue_class") == "REJECT_LOW_CONFIDENCE",
               "low-confidence flag → issue_class REJECT_LOW_CONFIDENCE")

    # 6b — sweep returns NO flag but a low-confidence NOTE in `error`. Per the
    # rule this is a DEGRADED PASS (the pixel check was uncertain) — reported
    # PASS_RELEVANCE_QA but the error/note is preserved (NOT a clean claim).
    run2 = _make_run_dir(tmp)
    _install_sweep({"verdict": "PASS", "flags": [], "sampled": 3,
                    "duration_s": 48.0,
                    "error": "low-confidence: scorer uncertain on 7/10 frames"})
    P.finalize_relevance_qa(run2 / "video.mp4", run2)
    _restore_sweep()
    disk2 = json.loads((run2 / "render_relevance_qa.json").read_text())
    _check(disk2.get("verdict") == "PASS_RELEVANCE_QA",
           "note-only low-confidence → PASS_RELEVANCE_QA (degraded)")
    _check("low-confidence" in (disk2.get("error") or ""),
           "degraded PASS preserves the low-confidence note (not a silent pass)")


def case_scorer_unavailable(tmp):
    print("\n[7] scorer-unavailable (sweep returns error) → NOT a silent clean "
          "PASS")
    run = _make_run_dir(tmp)
    # This mirrors the real sweep's fail-safe degrade: PASS verdict + error note.
    _install_sweep({"verdict": "PASS", "flags": [], "sampled": 0,
                    "duration_s": 48.0, "error": "scorer-unavailable"})
    rep = P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()
    disk = json.loads((run / "render_relevance_qa.json").read_text())
    # Verdict is PASS_RELEVANCE_QA (reporter never blocks), but the error is
    # surfaced so no one can read it as a verified-clean PASS.
    _check(disk.get("verdict") == "PASS_RELEVANCE_QA",
           "degraded sweep still reports a (non-blocking) PASS verdict")
    _check(disk.get("error") == "scorer-unavailable",
           "error note 'scorer-unavailable' preserved in the report")
    _check(disk.get("sampled") == 0,
           "sampled==0 makes the un-run check visible")
    m = json.loads((run / "render_metrics.json").read_text())
    _check(m.get("relevance_qa", {}).get("error") == "scorer-unavailable",
           "render_metrics relevance_qa.error carries the degrade note")
    _check(rep.get("error") == "scorer-unavailable",
           "returned report carries the error (no false PASS claim)")


def case_report_always_written(tmp):
    print("\n[8] report file is ALWAYS written (even with empty meta/manifest)")
    run = Path(tempfile.mkdtemp(prefix="rc5qa_bare_", dir=str(tmp)))
    (run / "video.mp4").write_bytes(b"fake")
    # No render_meta.json, no manifest — finalize must still write the report.
    _install_sweep({"verdict": "PASS", "flags": [], "sampled": 0,
                    "duration_s": 0.0, "error": "no-sample-points"})
    P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()
    _check((run / "render_relevance_qa.json").exists(),
           "render_relevance_qa.json written even with no meta/manifest")


def case_env_disables(tmp):
    print("\n[9] VIDLORE_POSTRENDER_QA=0 disables (skipped, nothing written)")
    run = _make_run_dir(tmp)
    os.environ["VIDLORE_POSTRENDER_QA"] = "0"
    called = {"v": False}

    def _boom(*a, **k):
        called["v"] = True
        raise AssertionError("sweep must NOT run when disabled")
    RQA.sweep = _boom
    rep = P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()
    os.environ["VIDLORE_POSTRENDER_QA"] = "1"
    _check(rep.get("skipped") is True, "returns {'skipped': True} when disabled")
    _check(not called["v"], "sweep was not invoked when disabled")
    _check(not (run / "render_relevance_qa.json").exists(),
           "no report file written when disabled")


def case_never_raises(tmp):
    print("\n[10] finalize never raises even if the sweep itself raises")
    run = _make_run_dir(tmp)

    def _raiser(*a, **k):
        raise RuntimeError("boom inside sweep")
    RQA.sweep = _raiser
    raised = False
    try:
        rep = P.finalize_relevance_qa(run / "video.mp4", run)
    except Exception as e:                                          # noqa: BLE001
        raised = True
        rep = {}
        print(f"      (unexpected raise: {e})")
    _restore_sweep()
    _check(not raised, "finalize_relevance_qa swallowed the sweep exception")
    # It degrades to a non-blocking PASS with an error note, report still written.
    _check((run / "render_relevance_qa.json").exists(),
           "report written even when the sweep raised")
    if rep:
        _check(rep.get("verdict") == "PASS_RELEVANCE_QA",
               "raised sweep → degraded PASS (non-blocking)")
        _check("exception" in (rep.get("error") or "").lower(),
               "error note records the exception")


# --------------------------------------------------------------------------- #
# RC5.1 STEP 2 — stale-output guard WIRING in pipeline.finalize_relevance_qa.
#
# These mock RQA.sweep (the pipeline WIRING is under test, not the sweep's own
# hashing — that is covered by case_stale_output_guard against the REAL sweep).
# We assert the pipeline:
#   (1) computes the SERVED export hash + passes it to the sweep, and on a
#       matching scan records served+scanned hash + hash_match True + PASS;
#   (2) maps a sweep FAIL_STALE_OUTPUT (or any hash mismatch) to the HARD status
#       FAIL_STALE_OUTPUT_HASH_MISMATCH in BOTH render_relevance_qa.json AND
#       render_metrics.json — NEVER collapsing it to PASS;
#   (3) treats a MISSING served MP4 as a hard fail (not a silent pass).
# --------------------------------------------------------------------------- #
def _install_sweep_raw(fn):
    """Install an arbitrary sweep impl (so a case can control scanned_sha256 /
    verdict / stale_output precisely). Restored by the caller via _restore_sweep."""
    RQA.sweep = fn


def case_served_hash_match(tmp):
    print("\n[10b] RC5.1 served hash MATCH → PASS + report carries served+scanned "
          "hash + hash_match True; finalize passed expected_sha256 to the sweep")
    run = _make_run_dir(tmp)
    seen = {}

    def _capture_sweep(mp4_path, render_meta, *, ffmpeg=None, manifest=None,
                       interval_s=None, max_frames=None,
                       expected_sha256=None, expected_path=None):
        # Prove finalize handed us the SERVED export hash + path.
        seen["expected_sha256"] = expected_sha256
        seen["expected_path"] = expected_path
        seen["manifest_passed"] = manifest is not None
        # Simulate the live pipeline: the sweep scanned the very file finalize
        # asked it to expect → scanned_sha256 == expected_sha256.
        return {"verdict": "PASS", "flags": [], "sampled": 7, "duration_s": 48.0,
                "error": "", "scanned_sha256": expected_sha256,
                "scanned_path": str(mp4_path), "scanned_mtime": 222.0,
                "expected_sha256": str(expected_sha256 or ""),
                "stale_output": False}
    _install_sweep_raw(_capture_sweep)
    rep = P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()

    # The served hash finalize computed must be a real, non-empty sha256 and must
    # have been forwarded to the sweep as expected_sha256.
    real_served = RQA._sha256_file(run / "video.mp4")
    _check(bool(real_served), "finalize could hash the served export")
    _check(seen.get("expected_sha256") == real_served,
           "finalize passed the SERVED export sha256 as expected_sha256= to sweep")
    _check(str(seen.get("expected_path", "")).endswith("video.mp4"),
           "finalize passed expected_path= (the served export path)")
    _check(seen.get("manifest_passed") is True,
           "finalize passed manifest= (ASSET_DECISION_MANIFEST) for per-beat cover")

    disk = json.loads((run / "render_relevance_qa.json").read_text())
    _check(disk.get("verdict") == "PASS_RELEVANCE_QA",
           f"matching hash → PASS_RELEVANCE_QA (got {disk.get('verdict')})")
    _check(disk.get("served_sha256") == real_served,
           "report served_sha256 == the real served-export hash")
    _check(disk.get("scanned_sha256") == real_served,
           "report scanned_sha256 == served (the QA verified the served export)")
    _check(disk.get("hash_match") is True, "report hash_match True")
    _check(disk.get("stale_output") is False, "report stale_output False")
    m = json.loads((run / "render_metrics.json").read_text())
    _check(m.get("relevance_qa", {}).get("hash_match") is True,
           "render_metrics relevance_qa.hash_match True")
    _check(m.get("relevance_qa", {}).get("verdict") == "PASS_RELEVANCE_QA",
           "render_metrics verdict PASS_RELEVANCE_QA on a verified match")


def case_stale_output_wiring(tmp):
    print("\n[10c] RC5.1 sweep FAIL_STALE_OUTPUT → pipeline writes "
          "FAIL_STALE_OUTPUT_HASH_MISMATCH (NOT PASS) into report + metrics")
    run = _make_run_dir(tmp)

    def _stale_sweep(mp4_path, render_meta, *, ffmpeg=None, manifest=None,
                     interval_s=None, max_frames=None,
                     expected_sha256=None, expected_path=None):
        # The sweep scanned a DIFFERENT file than the served export → loud stale.
        return {"verdict": "FAIL_STALE_OUTPUT", "flags": [], "sampled": 5,
                "duration_s": 48.0, "error": "", "stale_output": True,
                "scanned_sha256": "0" * 64,            # not the served hash
                "scanned_path": "/scratch/stale_scan.mp4", "scanned_mtime": 9.0,
                "expected_sha256": str(expected_sha256 or "")}
    _install_sweep_raw(_stale_sweep)
    rep = P.finalize_relevance_qa(run / "video.mp4", run)
    _restore_sweep()

    disk = json.loads((run / "render_relevance_qa.json").read_text())
    _check(disk.get("verdict") == "FAIL_STALE_OUTPUT_HASH_MISMATCH",
           f"stale sweep → FAIL_STALE_OUTPUT_HASH_MISMATCH (NOT PASS) "
           f"(got {disk.get('verdict')})")
    _check(disk.get("verdict") != "PASS_RELEVANCE_QA",
           "stale output is NEVER collapsed to PASS")
    _check(disk.get("stale_output") is True, "report stale_output True")
    _check(disk.get("hash_match") is False,
           "report hash_match False (served != scanned)")
    _check(disk.get("scanned_sha256") == "0" * 64,
           "report records the (wrong) scanned hash for a human")
    _check(disk.get("scanned_path") == "/scratch/stale_scan.mp4",
           "report records WHICH wrong file was scanned")
    _check(bool(disk.get("served_sha256")),
           "report still records the served export hash")
    m = json.loads((run / "render_metrics.json").read_text())
    _check(m.get("relevance_qa", {}).get("verdict")
           == "FAIL_STALE_OUTPUT_HASH_MISMATCH",
           "render_metrics records FAIL_STALE_OUTPUT_HASH_MISMATCH")
    _check(m.get("relevance_qa", {}).get("stale_output") is True,
           "render_metrics relevance_qa.stale_output True")
    _check(rep.get("verdict") == "FAIL_STALE_OUTPUT_HASH_MISMATCH",
           "returned report verdict matches disk")

    # Control: a sweep that scanned a hash that simply does NOT equal the served
    # export (without setting verdict/stale_output) is ALSO caught by finalize's
    # own hash_match check → hard stale. This proves the guard does not rely on
    # the sweep self-reporting stale.
    run2 = _make_run_dir(tmp)

    def _mismatch_only(mp4_path, render_meta, *, ffmpeg=None, manifest=None,
                       interval_s=None, max_frames=None,
                       expected_sha256=None, expected_path=None):
        return {"verdict": "PASS", "flags": [], "sampled": 6, "duration_s": 48.0,
                "error": "", "scanned_sha256": "a" * 64,   # != served, no stale flag
                "scanned_path": str(mp4_path), "stale_output": False}
    _install_sweep_raw(_mismatch_only)
    P.finalize_relevance_qa(run2 / "video.mp4", run2)
    _restore_sweep()
    disk2 = json.loads((run2 / "render_relevance_qa.json").read_text())
    _check(disk2.get("verdict") == "FAIL_STALE_OUTPUT_HASH_MISMATCH",
           "finalize's own hash_match catches a mismatch even when the sweep did "
           f"not self-report stale (got {disk2.get('verdict')})")


def case_served_mp4_missing(tmp):
    print("\n[10d] RC5.1 missing served MP4 → hard fail (NOT a silent PASS)")
    run = _make_run_dir(tmp)
    # finalize is called with a path that does NOT exist on disk.
    missing = run / "does_not_exist.mp4"

    def _ok_sweep(mp4_path, render_meta, *, ffmpeg=None, manifest=None,
                  interval_s=None, max_frames=None,
                  expected_sha256=None, expected_path=None):
        # Even if the sweep itself returned a clean PASS, a missing SERVED export
        # must still fail — the pipeline cannot certify a file it cannot find.
        return {"verdict": "PASS", "flags": [], "sampled": 0, "duration_s": 0.0,
                "error": "mp4-not-found", "scanned_sha256": "",
                "scanned_path": str(mp4_path), "stale_output": False}
    _install_sweep_raw(_ok_sweep)
    rep = P.finalize_relevance_qa(missing, run)
    _restore_sweep()
    disk = json.loads((run / "render_relevance_qa.json").read_text())
    _check(disk.get("verdict") == "FAIL_STALE_OUTPUT_HASH_MISMATCH",
           f"missing served MP4 → hard fail (got {disk.get('verdict')})")
    _check(disk.get("verdict") != "PASS_RELEVANCE_QA",
           "a missing served export is NEVER a silent PASS")
    _check("served-mp4-missing" in (disk.get("error") or ""),
           "error note explains the served export was missing")
    _check(disk.get("served_sha256") == "",
           "served_sha256 empty (could not hash a missing file)")
    m = json.loads((run / "render_metrics.json").read_text())
    _check(m.get("relevance_qa", {}).get("verdict")
           == "FAIL_STALE_OUTPUT_HASH_MISMATCH",
           "render_metrics records the missing-served hard fail")


# --------------------------------------------------------------------------- #
# MG-AWARE sweep internals — the motion-graphics false-positive fix.
#
# The cases above mock RQA.sweep entirely (they test the pipeline WIRING). The
# cases below exercise the REAL sweep() against a controlled per-frame
# graphic_dom, to prove the new card-window behaviour:
#   (a) a designed-graphic frame INSIDE an MG-card window is NOT flagged (the
#       engine's own intentional card is exempt), AND
#   (b) a designed-graphic JUNK frame OUTSIDE every card window is STILL flagged.
# `graphic_signal` and `_extract_frame` are stubbed so no real CLIP/ffmpeg is
# needed — `_extract_frame` records the timestamp it was asked to grab, and the
# stubbed `graphic_signal` returns the graphic_dom our scenario assigns to that
# timestamp. This keeps the sweep's real card-window math + threshold logic under
# test while staying fully offline.
# --------------------------------------------------------------------------- #
def _run_real_sweep(render_meta, mg_manifest, dom_for_ts):
    """Drive the genuine RQA.sweep with stubbed extraction + scorer.

    `dom_for_ts(ts) -> float` assigns the graphic_dom each sampled timestamp
    should probe. Returns the sweep result dict."""
    import vidlore.visual_relevance as VR

    ts_by_path = {}

    def _fake_extract(ff, mp4_path, ts, dest):
        # Pretend the frame extracted fine; remember which ts this path holds so
        # the scorer stub can answer for it.
        try:
            Path(dest).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
        except Exception:                                          # noqa: BLE001
            pass
        ts_by_path[str(dest)] = float(ts)
        return True

    def _fake_graphic_signal(path, is_video=False):
        ts = ts_by_path.get(str(path), 0.0)
        gd = float(dom_for_ts(ts))
        # looks_designed mirrors the live gate: gd over the base graphic_max.
        return {"graphic_dom": round(gd, 4),
                "looks_designed": bool(gd > VR._DEFAULT_GRAPHIC_MAX),
                "engine": "clip-onnx"}

    real_available = VR.available
    real_graphic = VR.graphic_signal
    real_extract = RQA._extract_frame
    VR.available = lambda: True
    VR.graphic_signal = _fake_graphic_signal
    RQA._extract_frame = _fake_extract
    try:
        # mp4 path only needs to EXIST for the sweep's early guard; extraction is
        # stubbed. Use this test file itself as a stand-in "file that exists".
        return RQA.sweep(__file__, render_meta, mg_manifest=mg_manifest)
    finally:
        VR.available = real_available
        VR.graphic_signal = real_graphic
        RQA._extract_frame = real_extract


def _run_real_sweep_ex(render_meta, mg_manifest, signal_for_ts, *,
                       manifest=None, sweep_kwargs=None):
    """RC5.1 — like `_run_real_sweep` but `signal_for_ts(ts) -> dict` returns the
    FULL graphic_signal dict (so a scenario can set looks_ui_screenshot / ui_geom),
    captures every timestamp `_extract_frame` was asked to grab (returned as
    `sampled_times`), and forwards extra kwargs (expected_sha256 etc.) to sweep.

    Returns (result_dict, sorted_sampled_times)."""
    import vidlore.visual_relevance as VR

    ts_by_path = {}
    sampled_times = []

    def _fake_extract(ff, mp4_path, ts, dest):
        try:
            Path(dest).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
        except Exception:                                          # noqa: BLE001
            pass
        ts_by_path[str(dest)] = float(ts)
        sampled_times.append(round(float(ts), 2))
        return True

    def _fake_graphic_signal(path, is_video=False):
        ts = ts_by_path.get(str(path), 0.0)
        out = dict(signal_for_ts(ts))
        out.setdefault("engine", "clip-onnx")
        return out

    real_available = VR.available
    real_graphic = VR.graphic_signal
    real_extract = RQA._extract_frame
    VR.available = lambda: True
    VR.graphic_signal = _fake_graphic_signal
    RQA._extract_frame = _fake_extract
    try:
        kw = dict(mg_manifest=mg_manifest, manifest=manifest)
        kw.update(sweep_kwargs or {})
        res = RQA.sweep(__file__, render_meta, **kw)
        return res, sorted(set(sampled_times))
    finally:
        VR.available = real_available
        VR.graphic_signal = real_graphic
        RQA._extract_frame = real_extract


# --------------------------------------------------------------------------- #
# RC5.1 — stale-output guard + per-beat coverage + game-UI hard reject.
# --------------------------------------------------------------------------- #
def case_stale_output_guard(tmp):
    print("\n[15] RC5.1 stale-output guard — expected_sha256 mismatch → "
          "FAIL_STALE_OUTPUT")
    meta = {"video_seconds": 24.0, "scene_starts": [0.0, 12.0],
            "scene_durations": [12.0, 12.0]}
    # A clean (no-junk) frame everywhere: content verdict would be PASS. But the
    # caller's expected final-export hash does NOT match the scanned file → the QA
    # scanned the WRONG file, so the verdict must be the loud FAIL_STALE_OUTPUT.
    clean = lambda ts: {"graphic_dom": -0.05, "looks_designed": False,
                        "ui_geom": 0.0, "looks_ui_screenshot": False}
    res, _ = _run_real_sweep_ex(
        meta, None, clean,
        sweep_kwargs={"expected_sha256": "deadbeef" * 8,   # 64 hex, won't match
                      "expected_path": "/served/final_export.mp4"})
    _check(res.get("verdict") == "FAIL_STALE_OUTPUT",
           f"mismatch → FAIL_STALE_OUTPUT (got {res.get('verdict')})")
    _check(res.get("stale_output") is True, "stale_output flag set True")
    _check(bool(res.get("scanned_sha256")),
           "scanned_sha256 recorded (the file actually inspected)")
    _check(res.get("scanned_sha256") != "deadbeef" * 8,
           "scanned hash differs from the expected hash (that's the point)")
    _check(res.get("expected_sha256") == "deadbeef" * 8,
           "expected_sha256 echoed back in the result")
    _check(bool(res.get("scanned_path")), "scanned_path recorded for a human")
    _check(res.get("scanned_mtime", 0) > 0, "scanned_mtime recorded")
    _check(res.get("expected_path") == "/served/final_export.mp4",
           "expected_path echoed for the report")

    # Control: when expected_sha256 MATCHES the scanned file → no stale verdict.
    real_sha = RQA._sha256_file(__file__)
    res2, _ = _run_real_sweep_ex(
        meta, None, clean, sweep_kwargs={"expected_sha256": real_sha})
    _check(res2.get("verdict") == "PASS",
           f"matching hash → PASS (got {res2.get('verdict')})")
    _check(res2.get("stale_output") is False, "matching hash → stale_output False")

    # No expected hash supplied → guard inert (still records scanned hash).
    res3, _ = _run_real_sweep_ex(meta, None, clean)
    _check(res3.get("verdict") == "PASS", "no expected hash → PASS (guard inert)")
    _check(bool(res3.get("scanned_sha256")),
           "scanned_sha256 ALWAYS recorded even with no expected hash")


def case_per_beat_coverage(tmp):
    print("\n[16] RC5.1 per-beat coverage — every scene boundary sampled on a "
          "31-scene meta")
    # A ~4-min / 31-scene doc: 31 scene starts, 8s each (248s). Every scene
    # boundary must get a sampled frame so a UI flash on any scene cannot pass.
    n = 31
    starts = [round(i * 8.0, 2) for i in range(n)]
    durs = [8.0] * n
    meta = {"video_seconds": round(n * 8.0, 2),
            "scene_starts": starts, "scene_durations": durs}
    clean = lambda ts: {"graphic_dom": -0.05, "looks_designed": False,
                        "ui_geom": 0.0, "looks_ui_screenshot": False}
    res, sampled = _run_real_sweep_ex(meta, None, clean)

    # The sweep samples each boundary at start+0.4 (rounded to ~0.5 buckets). For
    # each scene boundary, assert SOME sampled frame lands within the scene body
    # (start .. start+dur) — i.e. no scene is skipped.
    missed = []
    for i, s in enumerate(starts):
        lo, hi = s, s + durs[i]
        if not any(lo - 0.01 <= t <= hi + 0.01 for t in sampled):
            missed.append(i)
    _check(not missed, f"every one of {n} scenes has a sampled frame "
                       f"(missed scenes: {missed})")
    # The effective cap must have been raised to cover the scenes (>= 2×scenes).
    plan = res.get("sample_plan") or {}
    _check(plan.get("scenes") == n, f"sample_plan records {n} scenes "
                                    f"(got {plan.get('scenes')})")
    _check(plan.get("cap", 0) >= 2 * n,
           f"effective cap raised to >= 2x scenes ({plan.get('cap')} >= {2 * n})")
    _check(plan.get("cap", 0) <= RQA._ABSOLUTE_MAX_FRAMES,
           "effective cap stays bounded by the absolute ceiling")
    _check(res.get("verdict") == "PASS", "clean 31-scene doc → PASS")

    # Per-BEAT: a manifest with 62 beats (2/scene) must contribute beat times so
    # the planner covers beats, not just scene starts.
    beats = []
    for i in range(n):
        beats.append({"scene": i, "beat": 0, "t": round(i * 8.0, 2)})
        beats.append({"scene": i, "beat": 1, "t": round(i * 8.0 + 4.0, 2)})
    manifest = {"beats": beats}
    bt = RQA._beat_times(meta, manifest)
    _check(len(bt) >= n, f"beat-time extractor returns per-beat stamps "
                         f"(got {len(bt)} for {len(beats)} beats)")
    res_b, sampled_b = _run_real_sweep_ex(meta, None, clean, manifest=manifest)
    # Each beat MID (i*8+4) should also be near a sampled frame.
    beat_mid_missed = []
    for i in range(n):
        mid = i * 8.0 + 4.0
        if not any(mid - 1.2 <= t <= mid + 1.2 for t in sampled_b):
            beat_mid_missed.append(i)
    _check(len(beat_mid_missed) <= 2,
           f"per-beat mids covered (<=2 misses; got {len(beat_mid_missed)})")


def case_game_ui_hard_reject(tmp):
    print("\n[17] RC5.1 game-UI hard reject — UI frame OUTSIDE a card window → "
          "FAIL (and EXEMPT engine card still spared)")
    # scene0 = footage beat (0..12s); scene1 = an MG map card (12..24s). Put a
    # GAME-UI frame (looks_ui_screenshot True, with a map-like sub-gate graphic_dom
    # that alone would NOT trip the designed-graphic gate) on the FOOTAGE beat, and
    # a legitimate engine-card-grade graphic on the MG card beat.
    meta = {"video_seconds": 24.0, "scene_starts": [0.0, 12.0],
            "scene_durations": [12.0, 12.0]}
    mg = {"scenes": [
              {"scene_index": 0, "primitive": None, "skipped": True,
               "reason": "footage"},
              {"scene_index": 1, "primitive": "war_map_advance",
               "skipped": False, "reason": "mg"}],
          "motion_graphics_audit": {"summary": {
              "at_scenes": [[1, "war_map_advance"]]}}}

    def _sig(ts):
        if ts < 12.0:
            # FOOTAGE beat → a game-UI screenshot. graphic_dom stays UNDER the
            # designed-graphic gate (a tactical-game map reads partly as "a map"),
            # so ONLY the ui-geometry signal catches it. This is the missed case.
            return {"graphic_dom": 0.01, "looks_designed": False,
                    "ui_geom": 0.74, "looks_ui_screenshot": True}
        # MG card beat → engine-card-grade designed graphic, NOT a UI screenshot.
        return {"graphic_dom": 0.06, "looks_designed": True,
                "ui_geom": 0.05, "looks_ui_screenshot": False}

    res, _ = _run_real_sweep_ex(meta, mg, _sig)
    _check(res.get("card_windows") == 1, "one MG card window resolved")
    _check(res.get("verdict") == "FAIL_RELEVANCE_QA",
           f"game-UI on footage beat → FAIL (got {res.get('verdict')})")
    flags = res.get("flags") or []
    ui_flags = [f for f in flags
                if "ui" in str(f.get("reason", "")).lower()
                or "interface" in str(f.get("reason", "")).lower()]
    _check(len(ui_flags) >= 1, f"at least one game-UI flag (got {len(ui_flags)})")
    # All UI flags must be OUTSIDE the MG card window (scene0 footage beat).
    win = (12.0 - 0.35, 24.0 + 0.35)
    in_win_ui = [f for f in ui_flags
                 if win[0] <= float(f.get("timestamp", -1)) <= win[1]]
    _check(not in_win_ui,
           f"game-UI flags are on the footage beat, not the MG card "
           f"(in-window UI flags: {len(in_win_ui)})")
    # The engine map card (designed but NOT UI) must NOT be flagged.
    card_designed = [f for f in flags
                     if win[0] <= float(f.get("timestamp", -1)) <= win[1]
                     and "ui" not in str(f.get("reason", "")).lower()]
    _check(not card_designed,
           f"engine MG map card spared (no designed-graphic flag in window; "
           f"got {len(card_designed)})")
    # The flag's issue_class via the pipeline mapper must be REJECT_GAME_UI.
    from vidlore import pipeline as _P
    ic = _P._qa_issue_class(ui_flags[0].get("reason", "")) if ui_flags else ""
    _check(ic == "REJECT_GAME_UI",
           f"game-UI reason maps to REJECT_GAME_UI (got {ic})")


def case_ui_geom_signal_units(tmp):
    print("\n[18] RC5.1 UI-geometry signal — synthetic panel grid scores high, "
          "photo/blank scores low")
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:                                          # noqa: BLE001
        _check(True, f"(skipped — numpy/PIL unavailable: {e})")
        return
    import vidlore.visual_relevance as VR
    if not hasattr(VR, "_ui_geom_signal"):
        _check(False, "VR._ui_geom_signal exists")
        return
    # (a) Synthetic UI: a dense grid of rectangular panels with axis-aligned
    # borders + button rows — should score clearly UI-like (> 0.5).
    W, Hh = 320, 200
    img = np.full((Hh, W, 3), 30, dtype="uint8")
    # vertical panel dividers
    for x in range(0, W, 40):
        img[:, max(0, x - 1):x + 1] = 210
    # horizontal bar dividers
    for y in range(0, Hh, 25):
        img[max(0, y - 1):y + 1, :] = 210
    # a top toolbar + bottom status bar (perimeter chrome)
    img[:14, :] = 200
    img[-14:, :] = 200
    ui_score = VR._ui_geom_signal(Image.fromarray(img))
    _check(ui_score > 0.5,
           f"synthetic panel grid scores UI-like (ui_geom={ui_score:.3f} > 0.5)")

    # (b) A smooth gradient "photo" (no straight panel borders) → low UI score.
    grad = np.zeros((Hh, W, 3), dtype="uint8")
    for y in range(Hh):
        grad[y, :, :] = int(255 * y / Hh)
    photo_score = VR._ui_geom_signal(Image.fromarray(grad))
    _check(photo_score < 0.25,
           f"smooth gradient scores low (ui_geom={photo_score:.3f} < 0.25)")

    # (c) A blank flat frame → ~0 (no structure at all).
    blank = np.full((Hh, W, 3), 128, dtype="uint8")
    blank_score = VR._ui_geom_signal(Image.fromarray(blank))
    _check(blank_score < 0.1,
           f"blank frame scores ~0 (ui_geom={blank_score:.3f} < 0.1)")


def case_game_ui_metadata_tokens(tmp):
    print("\n[19] RC5.1 metadata tokens — strategy-game / HUD / dashboard slugs "
          "hard-reject")
    import vidlore.visual_relevance as VR
    cases = [
        ("hearts of iron 4 gameplay screenshot", "hearts of iron / paradox"),
        ("europa universalis tactical map hud", "grand-strategy UI"),
        ("missile command arcade game", "missile command"),
        ("admin dashboard control panel ui", "software dashboard"),
        ("total war rome in-game interface", "total war in-game"),
    ]
    for slug, label in cases:
        is_junk, reason, hits = VR.classify_junk_metadata(slug=slug,
                                                          narration="war history")
        _check(is_junk, f"{label!r} slug is hard-rejected (hits={hits})")
    # On-topic exemption still holds: a doc literally about the dashboard software.
    ok, _, _ = VR.classify_junk_metadata(
        slug="product dashboard demo",
        narration="this episode reviews the new analytics dashboard software")
    _check(not ok,
           "on-topic narration ('dashboard') exempts the dashboard slug (no "
           "false hard-reject)")


def case_mg_window_exempt_and_junk_caught(tmp):
    print("\n[11] MG-aware sweep — in-window card EXEMPT, out-of-window junk "
          "CAUGHT")
    # Two scenes: scene0 = footage beat (0..12s), scene1 = an MG primitive placed
    # (12..24s). Assign EVERY sampled frame a junk-grade graphic_dom (0.21) — the
    # worst case. Correct behaviour: scene1 (in-window) frames are exempt, scene0
    # (footage) frames flag. So we must get >=1 flag, ALL with in_card_window
    # False, and NONE inside the scene1 window.
    meta = {"video_seconds": 24.0, "scene_starts": [0.0, 12.0],
            "scene_durations": [12.0, 12.0]}
    mg = {"scenes": [
              {"scene_index": 0, "primitive": None, "skipped": True,
               "reason": "footage"},
              {"scene_index": 1, "primitive": "territory_advance_arrows",
               "skipped": False, "reason": "mg"}],
          "motion_graphics_audit": {"summary": {
              "at_scenes": [[1, "territory_advance_arrows"]]}}}
    res = _run_real_sweep(meta, mg, lambda ts: 0.21)   # all junk-grade

    _check(res.get("card_windows") == 1,
           f"resolved exactly 1 MG card window (got {res.get('card_windows')})")
    flags = res.get("flags") or []
    _check(res.get("verdict") == "FAIL_RELEVANCE_QA",
           "all-junk render FAILs (out-of-window junk is caught)")
    _check(len(flags) >= 1, f"at least one junk flag on the footage beat "
                            f"(got {len(flags)})")
    # Every flag must be OUTSIDE the card window (the in-window card is exempt).
    win = (12.0 - 0.35, 24.0 + 0.35)
    in_win = [f for f in flags if win[0] <= float(f.get("timestamp", -1)) <= win[1]]
    _check(not in_win,
           f"NO flag falls inside the MG-card window (got {len(in_win)})")
    _check(all(f.get("in_card_window") is False for f in flags),
           "every flag records in_card_window=False")


def case_mg_card_grade_spared_outside_window(tmp):
    print("\n[12] MG-aware sweep — card-GRADE graphic OUTSIDE window is SPARED "
          "in a card-rich doc")
    # Same layout, but now the on-screen graphics probe at the engine's own
    # CARD grade (~0.06 — a legacy/injected card the MG manifest doesn't track).
    # In a card-rich render those must NOT fail (they're the engine's cards), so
    # the raised card-aware ceiling spares them everywhere → 0 flags, PASS.
    meta = {"video_seconds": 24.0, "scene_starts": [0.0, 12.0],
            "scene_durations": [12.0, 12.0]}
    mg = {"scenes": [
              {"scene_index": 0, "primitive": None, "skipped": True,
               "reason": "footage"},
              {"scene_index": 1, "primitive": "chronology_timeline",
               "skipped": False, "reason": "mg"}],
          "motion_graphics_audit": {"summary": {
              "at_scenes": [[1, "chronology_timeline"]]}}}
    res = _run_real_sweep(meta, mg, lambda ts: 0.06)   # engine-card grade
    _check(res.get("verdict") == "PASS",
           "card-grade graphics in a card-rich doc → PASS (engine cards spared)")
    _check(len(res.get("flags") or []) == 0,
           f"no flags for card-grade graphics (got {len(res.get('flags') or [])})")


def case_no_card_doc_keeps_strict_gate(tmp):
    print("\n[13] MG-aware sweep — a NO-CARD doc keeps the strict gate "
          "(card-grade graphic still flags)")
    # No MG manifest, no graphic_kind scenes → not card-rich. A designed-graphic
    # frame is genuinely anomalous here, so the strict threshold applies and even
    # a card-grade (0.06) graphic flags. (This is the documented fallback: only a
    # render that truly has NO engine cards stays strict.)
    meta = {"video_seconds": 18.0, "scene_starts": [0.0, 9.0],
            "scene_durations": [9.0, 9.0]}
    res = _run_real_sweep(meta, None, lambda ts: 0.06)
    _check(res.get("card_windows") == 0, "no card windows in a no-card doc")
    _check(res.get("verdict") == "FAIL_RELEVANCE_QA",
           "no-card doc flags a designed-graphic frame (strict gate retained)")
    _check(len(res.get("flags") or []) >= 1,
           "at least one flag in the strict no-card path")


def case_card_window_builder_units(tmp):
    print("\n[14] MG-aware sweep — card-window builder unit checks")
    meta = {"scene_starts": [0.0, 10.0, 20.0],
            "scene_durations": [10.0, 10.0, 10.0]}
    # MG manifest: at_scenes lists scene 1; scenes[] also marks scene 2 rendered.
    mg = {"scenes": [
              {"scene_index": 0, "primitive": None, "skipped": True,
               "reason": "footage"},
              {"scene_index": 1, "primitive": "classified_stamp_reveal",
               "skipped": False, "reason": "mg"},
              {"scene_index": 2, "primitive": "chronology_timeline",
               "skipped": False, "reason": "mg"}],
          "motion_graphics_audit": {"summary": {
              "at_scenes": [[1, "classified_stamp_reveal"]]}}}
    idxs = RQA._placed_mg_scene_indices(mg)
    _check(idxs == {1, 2}, f"placed MG scene idxs == {{1,2}} (got {sorted(idxs)})")
    wins = RQA._card_windows(meta, mg)
    _check(len(wins) == 2, f"two card windows built (got {len(wins)})")
    _check(RQA._in_card_window(15.0, wins) is True,
           "ts inside scene-1 window is detected")
    _check(RQA._in_card_window(5.0, wins) is False,
           "ts in footage scene-0 is NOT a card window")
    # Legacy graphic_kind path: a render_meta carrying scene records w/ a card.
    meta_legacy = {"scene_starts": [0.0, 10.0], "scene_durations": [10.0, 10.0],
                   "scenes": [
                       {"graphic_kind": "", "graphic_text": ""},
                       {"graphic_kind": "title_card",
                        "graphic_text": "THE WAR"}]}
    leg = RQA._legacy_card_scene_indices(meta_legacy)
    _check(leg == {1}, f"legacy graphic_kind scene detected == {{1}} "
                       f"(got {sorted(leg)})")
    wins2 = RQA._card_windows(meta_legacy, None)
    _check(RQA._in_card_window(13.0, wins2) is True,
           "legacy-card scene yields a card window")
    # Absent manifest + no card metadata → no windows, never raises.
    _check(RQA._card_windows({"scene_starts": [0.0]}, None) == [],
           "no card metadata → empty window list (no raise)")
    _check(RQA._resolve_mg_manifest({"motion_graphics_audit": {}}, None)
           == {"motion_graphics_audit": {}},
           "dict mg_manifest passes through resolver unchanged")


def main():
    print("=" * 72)
    print("RC5 POST-RENDER RELEVANCE-QA WIRING — pipeline.finalize_relevance_qa")
    print("=" * 72)
    with tempfile.TemporaryDirectory(prefix="rc5qa_root_") as _tmp:
        tmp = Path(_tmp)
        case_pass(tmp)
        case_fail_anime(tmp)
        case_fail_game_ui(tmp)
        case_fail_multilingual_sign(tmp)
        case_fail_wrong_era(tmp)
        case_low_confidence(tmp)
        case_scorer_unavailable(tmp)
        case_report_always_written(tmp)
        case_env_disables(tmp)
        case_never_raises(tmp)
        # RC5.1 STEP 2 — stale-output guard WIRING (served-hash match / mismatch /
        # missing served export) through pipeline.finalize_relevance_qa.
        case_served_hash_match(tmp)
        case_stale_output_wiring(tmp)
        case_served_mp4_missing(tmp)
        # MG-aware sweep internals (the false-positive fix).
        case_mg_window_exempt_and_junk_caught(tmp)
        case_mg_card_grade_spared_outside_window(tmp)
        case_no_card_doc_keeps_strict_gate(tmp)
        case_card_window_builder_units(tmp)
        # RC5.1 — stale-output guard + per-beat coverage + game-UI hard reject.
        case_stale_output_guard(tmp)
        case_per_beat_coverage(tmp)
        case_game_ui_hard_reject(tmp)
        case_ui_geom_signal_units(tmp)
        case_game_ui_metadata_tokens(tmp)

    print("\n" + "=" * 72)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} FAILURE(S)")
        for m in _FAILS:
            print("  - " + m)
        return 1
    print("RESULT: ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
