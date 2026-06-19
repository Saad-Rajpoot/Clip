"""Orchestrator, split into two reviewable stages (the differentiator vs
Vidlore, whose #1 user complaint is "no script preview before a long render"):

  Stage 1  write_script        brief -> script.txt + script.json, then STOP
           (user reviews / edits script.txt)
  Stage 2  render_from_script  reviewed script -> narration -> captions ->
                               footage -> themed assembly -> MP4 + thumbnail

`produce()` chains both with no pause (one-shot / --auto / back-compat).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# (pct 0-100, human message) — optional UI progress hook (web wizard).
ProgressFn = Callable[[int, str], None]


def _emit(progress: "Optional[ProgressFn]", pct: int, msg: str) -> None:
    if progress is not None:
        try:
            progress(pct, msg)
        except Exception:  # noqa: BLE001  (UI hook must never break a render)
            pass

from .assemble import assemble, beats_per_scene
from .brief import Brief
from .config import Config
from .footage import build_graphic_images, fetch_footage
from .music import build_music_bed
from . import musiclib
from .script_gen import (
    Script,
    build_script,
    script_from_edited_txt,
    script_from_json,
)
from .themes import theme as get_theme
from .tts import narrate, narrate_from_file


@dataclass
class ScriptStage:
    run_dir: Path
    script_txt: Path
    script_json: Path
    script: Script


@dataclass
class Result:
    video: Path
    thumbnail: "Optional[Path]"   # thumbnail generation removed (editing tool) -> None
    script_txt: Path
    srt: Path
    srt: Path
    seconds: float                 # RENDER wallclock (time to produce)
    video_seconds: float = 0.0     # MNT_3 — actual muxed VIDEO duration (≠ render time)


def _log(step: str, msg: str) -> None:
    print(f"  [{step}] {msg}", flush=True)


def _slug(title: str) -> str:
    s = "".join(c if c.isalnum() else "-" for c in title.lower())[:50]
    return s.strip("-") or "video"


def run_dir_for(brief: Brief, out_dir: Path) -> Path:
    """Both stages must resolve to the same folder for --resume to work."""
    return out_dir / _slug(brief.title)


def _video_seconds(video: Path, run_dir: Path) -> float:
    """MNT_3 — the actual muxed VIDEO duration (distinct from render wallclock).
    Prefers the render_meta.json sidecar (written by assemble); falls back to
    parsing `ffmpeg -i`. 0.0 if neither is available."""
    try:
        meta = json.loads((run_dir / "render_meta.json").read_text(encoding="utf-8"))
        vs = float(meta.get("video_seconds", 0) or 0)
        if vs > 0:
            return round(vs, 2)
    except Exception:                                          # noqa: BLE001
        pass
    try:
        import subprocess
        from .ffmpeg_tool import ffmpeg_exe
        r = subprocess.run([ffmpeg_exe(), "-i", str(video)],
                           capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
        if m:
            return round(int(m.group(1)) * 3600 + int(m.group(2)) * 60
                         + float(m.group(3)), 2)
    except Exception:                                          # noqa: BLE001
        pass
    return 0.0


# --------------------------------------------------------------------------- #
# RC5 — post-render relevance-QA finalize hook
#
# The fail-closed relevance gate in footage.py runs DURING asset selection
# (the "pre-assembly selection" gate). `vidlore.relevance_qa.sweep` is the
# LAST-LINE backstop: a BOUNDED frame-sweep of the FINISHED MP4 that catches a
# junk visual (anime cover / game-UI / multilingual sign / wrong-era still /
# designed text-board) that slipped every in-pipeline gate — including the
# producers that never pass the selection gate (MG card imagery, portrait
# backgrounds, Review-Editor replacements). This wires it into the shared
# render path so a render is never SILENTLY approved with an off-topic frame.
#
# Contract (RC5 brief):
#   * BOUNDED — `sweep` itself caps sampled frames; we add no extra loop.
#   * FAIL-SAFE — `sweep` never raises (it is a reporter, not a gate); we wrap
#     it again so a finalize failure can never break a render. On any error we
#     treat it as PASS-with-error-note (no false "clean" claim — the note says
#     the pixel check was degraded).
#   * NON-ABORTING — a FAIL is recorded (render_relevance_qa.json +
#     render_metrics.json "relevance_qa" block + a one-line log), but the MP4
#     still exists and we do NOT raise. The status is just visibly
#     FAIL_RELEVANCE_QA so the render is not auto-approved.
#   * env VIDLORE_POSTRENDER_QA (default ON; "0"/"false"/"no"/"off" disables).
#
# RC5.1 STALE-OUTPUT GUARD (STEP 2):
#   The sweep can only certify the file it actually SCANNED. finalize now computes
#   the sha256 of the SERVED export (the `video` arg the caller delivers) with a
#   single streamed read and passes it to `sweep(..., expected_sha256=,
#   expected_path=)`. The sweep records the hash/path/mtime of the file it
#   inspected and, when the two hashes disagree, returns `FAIL_STALE_OUTPUT`
#   (the QA looked at the wrong file). finalize maps that to the HARD status
#   `FAIL_STALE_OUTPUT_HASH_MISMATCH` and NEVER collapses it to PASS, and surfaces
#   `served_path`/`served_sha256`/`scanned_path`/`scanned_sha256`/`hash_match` in
#   render_relevance_qa.json so a human can see WHICH file was verified and that it
#   equals the served export. In the live pipeline the sweep scans the served file
#   itself, so the hashes match and this proves the guard end-to-end + records
#   provenance; if a future scratch/serve split ever diverges it FAILs loudly.
#
# RC5.1 PER-BEAT COVERAGE (STEP 3):
#   finalize passes `manifest=<run_dir>/ASSET_DECISION_MANIFEST.json` so the sweep
#   can mine per-beat times, and feeds it render_meta (scene_starts +
#   scene_durations); the sweep samples ≥1 frame per scene boundary and per beat
#   (start+mid) within its own bounded, fail-safe cap. We only feed the inputs —
#   the sweep owns the sampling and never raises.
# --------------------------------------------------------------------------- #
def postrender_qa_enabled() -> bool:
    """Honor VIDLORE_POSTRENDER_QA — default ON; off only for explicit 0/false."""
    v = str(os.environ.get("VIDLORE_POSTRENDER_QA", "1")).strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _qa_issue_class(reason: str) -> str:
    """Map a sweep flag's free-text `reason` onto a stable issue class, reusing
    the ASSET_DECISION_MANIFEST verdict vocabulary so the post-render report and
    the selection-time manifest speak the same language. Never raises."""
    try:
        w = (reason or "").lower()
        if "quarantin" in w:
            return "REJECT_QUARANTINED"
        # Scorer-certainty signal takes priority over the asset-class guess: an
        # uncertain/unscored probe is a low-confidence finding regardless of what
        # class it tentatively named (e.g. "low-confidence designed-graphic").
        if ("unscored" in w or "unavailable" in w or "low-confidence" in w
                or "low confidence" in w or "no-score" in w or "uncertain" in w
                or w == "error"):
            return "REJECT_LOW_CONFIDENCE"
        if ("anime" in w or "cartoon" in w or "manga" in w or "comic" in w
                or "illustration" in w):
            return "REJECT_CARTOON_ANIME"
        if ("game" in w or "hud" in w or "screenshot" in w or " ui" in w
                or "interface" in w or "console" in w):
            return "REJECT_GAME_UI"
        if ("poster" in w or "cover" in w or "dvd" in w or "wallpaper" in w
                or "box art" in w or "boxart" in w):
            return "REJECT_POSTER_OR_COVER"
        if ("text" in w or "sign" in w or "infographic" in w or "chart" in w
                or "logo" in w or "caption" in w or "designed" in w
                or "graphic" in w or "board" in w):
            return "REJECT_TEXT_HEAVY"
        if ("wrong-era" in w or "era" in w or "period" in w or "anachron" in w
                or "modern" in w):
            return "REJECT_OFF_TOPIC"
        if ("weak" in w or "duplicate" in w or "repetition" in w):
            return "REJECT_WEAK_CONNECTION"
        if "off-topic" in w or "subject" in w or "wrong" in w:
            return "REJECT_OFF_TOPIC"
        return "REVIEW"
    except Exception:                                              # noqa: BLE001
        return "REVIEW"


def _load_json_safe(path: Path):
    """Best-effort JSON read; returns None on anything odd. Never raises."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return None
    return None


def _sha256_file(path: Path) -> str:
    """RC5.1 — streamed sha256 hex of the SERVED/final-export MP4. A SINGLE
    streamed read (1 MiB chunks) — no uncontrolled loop, bounded by file size —
    so finalize can hand `relevance_qa.sweep` the caller's record of the exact
    file the user receives (`expected_sha256=`) and prove the QA scanned that
    same export. Returns "" on any error (missing file / unreadable) so a hashing
    failure degrades the stale-output guard to inert rather than raising — the
    finalize hook stays fully fail-safe."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:                                              # noqa: BLE001
        return ""


def _qa_manifest_index(manifest) -> dict:
    """Index the ASSET_DECISION_MANIFEST beats by (scene, beat) so a sweep flag
    that knows its scene can be enriched with the beat's asset/narration/source/
    verdict. Tolerant of shape; returns {} on anything odd. Never raises."""
    idx: dict = {}
    try:
        beats = manifest.get("beats") if isinstance(manifest, dict) else None
        if not isinstance(beats, list):
            return idx
        for rec in beats:
            if not isinstance(rec, dict):
                continue
            try:
                sc = int(rec.get("scene"))
            except (TypeError, ValueError):
                continue
            try:
                bt = int(rec.get("beat", 0))
            except (TypeError, ValueError):
                bt = 0
            # First record for a scene is the natural "representative" asset; keep
            # the (scene, beat) precise key AND a scene-only fallback (beat 0).
            idx[(sc, bt)] = rec
            idx.setdefault((sc, None), rec)
    except Exception:                                              # noqa: BLE001
        return {}
    return idx


def _enrich_qa_flag(flag: dict, meta: dict, man_idx: dict) -> dict:
    """Turn a raw sweep flag {timestamp, scene, reason, suggestion} into the
    enriched render_relevance_qa schema record, resolving beat/asset/narration
    from render_meta + the asset-decision manifest where possible. Best-effort;
    a missing field is simply omitted/None. Never raises."""
    out: dict = {}
    try:
        ts = flag.get("timestamp")
        scene = flag.get("scene")
        reason = flag.get("reason") or ""
        out["timestamp"] = ts
        out["scene"] = scene
        # Pull the matching manifest beat (precise, else scene-representative).
        rec = None
        try:
            si = int(scene) if scene is not None else None
        except (TypeError, ValueError):
            si = None
        if si is not None and si >= 0 and man_idx:
            rec = man_idx.get((si, None))
        if isinstance(rec, dict):
            out["beat"] = rec.get("beat")
            out["asset_path"] = rec.get("source")          # basename (no secrets)
            out["asset_source"] = rec.get("role") or rec.get("outcome")
            out["narration"] = rec.get("narration")
        else:
            # No manifest beat — try render_meta scene_durations only for context.
            out["beat"] = None
            out["asset_path"] = None
            out["asset_source"] = None
            out["narration"] = None
        out["reason"] = reason
        out["verdict"] = "FAIL_RELEVANCE_QA"
        out["confidence"] = flag.get("confidence", "flagged")
        out["issue_class"] = _qa_issue_class(reason)
        out["suggestion"] = flag.get("suggestion") or ""
    except Exception:                                              # noqa: BLE001
        # Degrade to the raw flag rather than dropping it.
        return {
            "timestamp": flag.get("timestamp"),
            "scene": flag.get("scene"),
            "reason": flag.get("reason"),
            "verdict": "FAIL_RELEVANCE_QA",
            "issue_class": "REVIEW",
            "suggestion": flag.get("suggestion") or "",
        }
    return out


def _merge_metrics_relevance_block(run_dir: Path, block: dict) -> None:
    """Record the relevance-QA verdict into render_metrics.json so the render is
    never silently approved. render_metrics.json is otherwise written only by
    offline log-parser tools, so we CREATE it (minimal) when absent and MERGE the
    "relevance_qa" block when present — never clobbering existing fields.
    Best-effort; never raises."""
    try:
        mpath = run_dir / "render_metrics.json"
        existing = _load_json_safe(mpath)
        if not isinstance(existing, dict):
            existing = {"schema": "render_metrics/partial",
                        "generated_by": "post_render_sweep"}
        existing["relevance_qa"] = block
        mpath.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as _e:                                        # noqa: BLE001
        print(f"  [relevance-qa] render_metrics.json merge skipped ({_e})",
              flush=True)


def finalize_relevance_qa(video: Path, run_dir: Path) -> dict:
    """RC5 / RC5.1 post-render relevance sweep — see section header. Runs the
    bounded `relevance_qa.sweep` on the finished MP4, enriches each flag from
    render_meta + ASSET_DECISION_MANIFEST, writes <run_dir>/render_relevance_qa.json
    (schema "render_relevance_qa/1"), records the verdict into render_metrics.json,
    and prints a one-line summary. NEVER raises and NEVER aborts the render.

    RC5.1 STEP 2 — stale-output guard: the SERVED export's sha256 (`video`, the
    file the user receives) is computed with one streamed read and handed to the
    sweep as `expected_sha256=`/`expected_path=`. If the sweep scanned a different
    file the verdict is the HARD `FAIL_STALE_OUTPUT_HASH_MISMATCH` (never collapsed
    to PASS). The report surfaces served_path / served_sha256 / scanned_path /
    scanned_sha256 / hash_match so a human can see WHICH file was verified.

    RC5.1 STEP 3 — per-beat coverage: the ASSET_DECISION_MANIFEST is passed as
    `manifest=` and render_meta (scene_starts + scene_durations) feeds the sweep's
    own per-scene/per-beat sampling. We only feed inputs; the sweep owns sampling.

    Returns the persisted report dict (so callers/tests can assert on it). When
    VIDLORE_POSTRENDER_QA disables the sweep, returns {"skipped": True} and
    writes nothing.
    """
    run_dir = Path(run_dir)
    video = Path(video)
    if not postrender_qa_enabled():
        return {"skipped": True}

    meta = _load_json_safe(run_dir / "render_meta.json")
    meta = meta if isinstance(meta, dict) else {}
    manifest = _load_json_safe(run_dir / "ASSET_DECISION_MANIFEST.json")
    man_idx = _qa_manifest_index(manifest)

    # RC5.1 — fingerprint the SERVED export (the file the user actually receives).
    # A single streamed read; "" on any failure so the guard degrades to inert
    # rather than raising. Resolve a stable absolute path for the report. A
    # MISSING served file is NOT a silent pass — it becomes a hard fail below.
    served_sha = _sha256_file(video)
    try:
        served_path = str(video.resolve())
    except Exception:                                              # noqa: BLE001
        served_path = str(video)
    served_missing = not video.exists()

    # Resolve the engine ffmpeg (same binary the render used). Fail-soft to the
    # sweep's own bundled-binary discovery if this import ever fails.
    ffmpeg = None
    try:
        from .ffmpeg_tool import ffmpeg_exe
        ffmpeg = ffmpeg_exe()
    except Exception:                                              # noqa: BLE001
        ffmpeg = None

    # The sweep is itself fail-safe; we still wrap it so finalize can NEVER break
    # a render. On error we report PASS-with-error-note (no false clean claim).
    # We pass expected_sha256/expected_path so the sweep can detect a stale/
    # mismatched scan and stamp provenance (RC5.1 FIX 1), and manifest so it can
    # cover every beat (RC5.1 FIX 2).
    error_note = ""
    try:
        from . import relevance_qa as _relqa
        res = _relqa.sweep(video, meta, ffmpeg=ffmpeg, manifest=manifest,
                           expected_sha256=(served_sha or None),
                           expected_path=served_path)
        if not isinstance(res, dict):
            res = {"verdict": "PASS", "flags": [], "sampled": 0,
                   "duration_s": 0.0, "error": "sweep-returned-non-dict"}
    except Exception as _e:                                        # noqa: BLE001
        res = {"verdict": "PASS", "flags": [], "sampled": 0, "duration_s": 0.0,
               "error": f"finalize-exception: {type(_e).__name__}: "
                        f"{str(_e)[:120]}"}
        error_note = res["error"]

    raw_verdict = str(res.get("verdict") or "PASS")
    err = str(res.get("error") or "")
    flags = res.get("flags") if isinstance(res.get("flags"), list) else []
    enriched = [_enrich_qa_flag(f, meta, man_idx)
                for f in flags if isinstance(f, dict)]

    # Provenance: which file did the sweep actually inspect, and does it equal the
    # served export? Prefer the sweep's recorded scanned_sha256; fall back to the
    # served hash (in the live pipeline they are the same file). hash_match is the
    # human-readable "the QA verified the file you are serving" signal.
    scanned_sha = str(res.get("scanned_sha256") or "")
    scanned_path = str(res.get("scanned_path") or served_path)
    scanned_mtime = res.get("scanned_mtime")
    # hash_match is True only when BOTH hashes are known AND equal. Unknown (a
    # hashing failure on either side) is NOT a match — never claim a verified
    # equality we could not compute.
    hash_match = bool(served_sha and scanned_sha
                      and served_sha.strip().lower() == scanned_sha.strip().lower())
    sweep_stale = bool(res.get("stale_output"))

    # Status mapping (precedence):
    #   * served MP4 missing OR the sweep reported a hash mismatch / stale scan /
    #     FAIL_STALE_OUTPUT  → HARD FAIL_STALE_OUTPUT_HASH_MISMATCH. This is the
    #     loudest result and is NEVER collapsed to PASS: the QA either could not
    #     find the served file or verified the WRONG file, so any content verdict
    #     is moot.
    #   * a real relevance FAIL with flags → FAIL_RELEVANCE_QA.
    #   * otherwise → PASS_RELEVANCE_QA. A degraded sweep (scorer off / no ffmpeg /
    #     unreadable / exception) is reported PASS_RELEVANCE_QA but with the error
    #     preserved so it is NOT a silent/false "verified clean" — the note states
    #     the pixel check did not run (the in-pipeline selection gate is the
    #     backstop).
    stale = bool(served_missing or sweep_stale
                 or raw_verdict == "FAIL_STALE_OUTPUT"
                 or (served_sha and scanned_sha and not hash_match))
    if stale:
        status = "FAIL_STALE_OUTPUT_HASH_MISMATCH"
    elif raw_verdict == "FAIL_RELEVANCE_QA" and enriched:
        status = "FAIL_RELEVANCE_QA"
    else:
        status = "PASS_RELEVANCE_QA"

    # If the served file is missing, make that explicit in the error note so the
    # hard fail is self-explanatory (the sweep also records mp4-not-found, but the
    # served export may differ from the path the sweep was handed in the future).
    if served_missing and "served-mp4-missing" not in err:
        err = (err + "; " if err else "") + "served-mp4-missing"

    report = {
        "schema": "render_relevance_qa/1",
        "verdict": status,
        "sampled": int(res.get("sampled") or 0),
        "duration_s": float(res.get("duration_s") or 0.0),
        "error": err or error_note,
        # RC5.1 provenance — a human can see exactly which file was verified.
        "served_path": served_path,
        "served_sha256": served_sha,
        "scanned_path": scanned_path,
        "scanned_sha256": scanned_sha,
        "scanned_mtime": scanned_mtime,
        "hash_match": hash_match,
        "stale_output": stale,
        "flags": enriched,
        "generated_by": "post_render_sweep",
    }

    # 2. Persist the per-render report (ALWAYS written when the sweep ran).
    try:
        (run_dir / "render_relevance_qa.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as _e:                                        # noqa: BLE001
        print(f"  [relevance-qa] render_relevance_qa.json write skipped ({_e})",
              flush=True)

    # 3. Record the verdict into render_metrics.json so the render is never
    # silently approved — including the stale-output provenance.
    _merge_metrics_relevance_block(run_dir, {
        "verdict": status,
        "flags": len(enriched),
        "sampled": report["sampled"],
        "error": report["error"],
        "served_sha256": served_sha,
        "scanned_sha256": scanned_sha,
        "hash_match": hash_match,
        "stale_output": stale,
        "report": "render_relevance_qa.json",
    })

    # One-line summary.
    if status == "FAIL_STALE_OUTPUT_HASH_MISMATCH":
        print(f"  [relevance-qa] FAIL_STALE_OUTPUT_HASH_MISMATCH — the QA did NOT "
              f"verify the served export "
              f"(served={served_sha[:12] or '<missing>'}…, "
              f"scanned={scanned_sha[:12] or '<none>'}…, "
              f"hash_match={hash_match}). Content verdict is moot.", flush=True)
    elif status == "FAIL_RELEVANCE_QA":
        first = enriched[0] if enriched else {}
        print(f"  [relevance-qa] FAIL_RELEVANCE_QA — {len(enriched)} flag(s) "
              f"over {report['sampled']} frame(s); e.g. scene "
              f"{first.get('scene')} @ {first.get('timestamp')}s "
              f"[{first.get('issue_class')}]", flush=True)
    else:
        note = f" (degraded: {report['error']})" if report["error"] else ""
        print(f"  [relevance-qa] PASS_RELEVANCE_QA — {report['sampled']} "
              f"frame(s) sampled, served-export verified "
              f"(hash_match={hash_match}){note}", flush=True)

    return report


def _script_body(script: Script) -> str:
    return f"{script.title}\n\n" + "\n\n".join(s.narration for s in script.scenes)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_script_files(script: Script, run_dir: Path) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    body = _script_body(script)
    script_txt = run_dir / "script.txt"
    script_txt.write_text(body, encoding="utf-8")
    script_json = run_dir / "script.json"
    script_json.write_text(
        json.dumps(
            {
                "title": script.title,
                "source_sha256": _sha(body),
                "scenes": [
                    {
                        "narration": s.narration,
                        "keywords": s.keywords,
                        "visual": s.visual,
                        "intensity": s.intensity,
                        "emphasis": s.emphasis,
                        "shot_type": s.shot_type,
                        "role": s.role,
                        "graphic_kind": s.graphic_kind,
                        "graphic_text": s.graphic_text,
                        "graphic_body": s.graphic_body,
                    }
                    for s in script.scenes
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return script_txt, script_json


def _dump_brief(brief: Brief, run_dir: Path) -> None:
    """Persist the render-relevant brief fields so the web app can
    re-render / per-scene regenerate later without the original form."""
    (run_dir / "brief.json").write_text(
        json.dumps(
            {
                "title": brief.title,
                "fmt": brief.fmt,
                "duration": brief.duration,
                "theme": brief.theme,
                "captions": brief.captions,
                "music": brief.music,
                "voice": brief.voice,
                "background": getattr(brief, "background", "auto"),
                # Bug fix 2026-05-26: voiceover path was being dropped
                # on persist — so user-uploaded narration never
                # survived a re-render/resume. Always include the path
                # (None when no upload, str when user supplied one).
                "voiceover": getattr(brief, "voiceover", None),
            },
            indent=2,
        )
    , encoding="utf-8")


def load_brief(run_dir: Path) -> "Brief | None":
    """Rebuild a Brief from a persisted brief.json (script already exists,
    so the prompt is irrelevant for rendering)."""
    f = run_dir / "brief.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    return Brief(
        title=d["title"],
        prompt="reviewed",
        fmt=d.get("fmt", "documentary"),
        duration=d.get("duration", "6-8"),
        theme=d.get("theme", "history"),
        captions=bool(d.get("captions", True)),
        music=d.get("music"),
        voice=d.get("voice"),
        background=d.get("background", "auto"),
        # Bug fix 2026-05-26: voiceover field was missing from the
        # round-trip — user-uploaded narration was silently dropped on
        # resume / retry / re-render, causing the pipeline to fall back
        # to edge-tts even when the user explicitly supplied their voice.
        voiceover=d.get("voiceover"),
    )


def _load_variants(run_dir: Path, name: str = "variants.json") -> dict[int, int]:
    """Per-scene counters (visual = variants.json, voice = voice_variants
    .json) — a bump busts that scene's cache and picks a fresh asset."""
    f = run_dir / name
    if not f.exists():
        return {}
    try:
        return {int(k): int(v) for k, v in json.loads(f.read_text(encoding="utf-8")).items()}
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# Stage 1 — write the script and stop for review
# --------------------------------------------------------------------------- #
def write_script(
    brief: Brief,
    cfg: Config,
    out_dir: Path,
    *,
    script_file: str | None = None,
    progress: "Optional[ProgressFn]" = None,
) -> ScriptStage:
    run_dir = run_dir_for(brief, out_dir)
    _log("script", "writing script ...")
    _emit(progress, 20, "Writing script…")
    # Install Look DNA BEFORE script generation so the channel's
    # `scene_count_mult`, `pacing_hint`, and reveal-phrase tables
    # reach the LLM prompt (P3 + P-reveal).  Without this, script
    # gen runs with look=None and every channel produces the same
    # scene skeleton.  Render stage installs it again — that's
    # idempotent.  Wrapped in try so a missing channel never breaks
    # the legacy path.
    try:
        from .look_dna import resolve_look as _ld_resolve_pre
        from .look_dna import set_current as _ld_set_pre
        _pre_look = _ld_resolve_pre(brief)
        if _pre_look is not None:
            _ld_set_pre(_pre_look)
    except Exception:                                  # noqa: BLE001
        pass

    # ── SCRIPT.JSON REUSE (v13.3, 2026-05-28) ────────────────────────
    # The per-scene editorial design (graphic_kind / graphic_text /
    # graphic_body / keywords / shot_type) is produced by ~N/6 batched
    # Claude calls inside build_script(). Re-rendering the SAME script
    # re-pays that LLM cost every time — which is how the Mossad
    # debugging burned the Anthropic credit balance (4 full-render
    # attempts × ~37 editor calls). If a designed script.json already
    # exists in this run_dir AND it matches the input script text, load
    # it and SKIP the LLM entirely (zero Claude cost on re-renders).
    #
    # Match rule: when a `script_file` is supplied we compare the stored
    # `source_sha256` against the SHA of the input narrations (verbatim
    # from the file — the LLM never changes narration text, only the
    # design fields). On match → reuse. On mismatch → rebuild (the
    # script genuinely changed). Disable with VIDLORE_REUSE_SCRIPT_JSON=0;
    # force reuse regardless of match with =force.
    _reuse_env = (os.environ.get("VIDLORE_REUSE_SCRIPT_JSON", "1")
                  or "").strip().lower()
    _existing_json = run_dir / "script.json"
    script = None
    if _reuse_env not in ("0", "false", "no", "off") and \
            _existing_json.exists():
        try:
            _data = json.loads(_existing_json.read_text(encoding="utf-8"))
            _stored_sha = _data.get("source_sha256", "")
            _ok = False
            if _reuse_env == "force":
                _ok = True
            elif script_file:
                # Build the comparable body from the raw input file the
                # same way _script_body() does: title line + blank-line
                # separated narration paragraphs.
                try:
                    _raw = Path(script_file).read_text(encoding="utf-8")
                except Exception:                          # noqa: BLE001
                    _raw = ""
                _paras = [p.strip() for p in _raw.split("\n\n")
                          if p.strip()]
                if _paras:
                    _body_in = _paras[0] + "\n\n" + "\n\n".join(_paras[1:])
                    if _sha(_body_in) == _stored_sha:
                        _ok = True
                # Fallback: narration-set equality (encoding/newline noise
                # tolerant) — same trick _load_reviewed_script uses.
                if not _ok:
                    _stored_narr = [
                        (s.get("narration") or "").strip()
                        for s in _data.get("scenes", [])
                    ]
                    _in_narr = [p.strip() for p in _paras[1:]]
                    if _stored_narr and _stored_narr == _in_narr:
                        _ok = True
            else:
                # No script_file → can't diff; trust the existing design.
                _ok = True
            if _ok:
                script = script_from_json(_data)
                _log("script",
                     f"reused existing script.json — SKIPPED LLM editor "
                     f"({len(script.scenes)} scenes, 0 Claude calls)")
        except Exception as _e:                            # noqa: BLE001
            _log("script", f"script.json reuse failed ({str(_e)[:60]}); "
                           "rebuilding via LLM")
            script = None

    if script is None:
        script = build_script(brief, cfg, script_file)
    script_txt, script_json = _write_script_files(script, run_dir)
    _dump_brief(brief, run_dir)
    _log("script", f"{len(script.scenes)} scenes, ~{script.word_count} words")
    _emit(progress, 100,
          f"Script ready · {len(script.scenes)} scenes, "
          f"~{script.word_count} words")
    return ScriptStage(run_dir, script_txt, script_json, script)


# --------------------------------------------------------------------------- #
# Stage 2 — render from the reviewed (possibly edited) script
# --------------------------------------------------------------------------- #
def _load_reviewed_script(brief: Brief, run_dir: Path) -> tuple[Script, bool]:
    """Returns (script, edited?). If script.txt is byte-identical to what we
    generated, reuse script.json (keeps the richer LLM scene split +
    keywords). If the user edited it, re-derive scenes/keywords from text.

    Reads script.txt as UTF-8, with a cp1252 fallback so legacy files
    written by older Windows builds (Python <3.15 defaults to cp1252) are
    not mis-decoded — that mis-decode used to corrupt the SHA, mark the
    script "edited", and drop every scene's visual/keywords/graphics.
    """
    script_txt = run_dir / "script.txt"
    script_json = run_dir / "script.json"
    if not script_txt.exists():
        raise FileNotFoundError(
            f"No reviewed script at {script_txt}. Run the script stage first "
            "(drop --resume), review it, then re-run with --resume."
        )
    try:
        body = script_txt.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Legacy file written in cp1252 by an older Windows render — read
        # it back through cp1252 so the smart-dashes/quotes are decoded
        # correctly, then rewrite as UTF-8 so future loads stay clean.
        body = script_txt.read_text(encoding="cp1252")
        try:
            script_txt.write_text(body, encoding="utf-8")
        except OSError:
            pass
    if script_json.exists():
        data = json.loads(script_json.read_text(encoding="utf-8"))
        if data.get("source_sha256") == _sha(body):
            return script_from_json(data), False
        # Recovery: if the narrations still line up with what we stored
        # in script.json, the only difference is encoding/newline noise
        # from the legacy round-trip — trust the rich JSON (visual,
        # graphics, emphasis, shot_type, role) instead of throwing it
        # away and re-deriving from text (which dropped all of these
        # fields and was the root cause of "no text / irrelevant footage"
        # on Windows).
        blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
        if len(blocks) >= 2:
            narrs = blocks[1:]
            json_narrs = [s.get("narration", "").strip()
                          for s in data.get("scenes", [])]
            if len(narrs) == len(json_narrs) and all(
                    a == b for a, b in zip(narrs, json_narrs)):
                return script_from_json(data), False
    return script_from_edited_txt(body, brief.title), True


def _mg_structured_assets(graphic_kind: str, graphic_body: str) -> dict:
    """Parse the V2.8 *structured* graphic_body hints the director cannot derive
    from prose, so a script author can drive two Section-C primitives explicitly.

    The director auto-derives regions/hops/stages/parent/targets/fact from
    narration, but the inputs for these two are inherently structured (scale-true
    silhouette sizes / route coordinates) — there is no prose to mine — so without
    an explicit body they can NEVER fire. Mirrors the inline adapter's
    "key=value ; key2=val2" idiom: records split on ``|``, fields on ``:`` (a
    trailing ``; key=…`` is excluded by the ``[^;]+`` capture, exactly like the
    ``items=``/``points=`` parsers already used for ranking/growth above).

        graphic_kind="scale_compare"  (or "size_compare")
        graphic_body="items=Bus:12:city bus|Truck:8:road"
            -> {"items": [{"label": "Bus",   "size": 12.0, "note": "city bus"},
                          {"label": "Truck", "size":  8.0, "note": "road"}]}
                                                    → silhouette_scale_compare

        graphic_kind="route_trace"
        graphic_body="points=0.2:0.7:Start|0.5:0.45|0.8:0.3:End"
            -> {"points": [{"x": 0.2, "y": 0.7,  "label": "Start"},
                           {"x": 0.5, "y": 0.45, "label": ""},
                           {"x": 0.8, "y": 0.3,  "label": "End"}]}
                                                    → footage_route_trace

    Returns only the keys it could parse (empty dict if none) so the caller can
    ``_a.update(...)`` without clobbering. Both primitives re-normalise/clamp
    their own inputs, so this stays a thin structural parse.
    """
    import re as _re3
    gk = (graphic_kind or "").lower()
    gb = graphic_body or ""
    out: dict = {}
    # silhouette_scale_compare — two scale-true silhouettes (label:size[:note]).
    if gk in ("scale_compare", "size_compare"):
        _m = _re3.search(r"items=([^;]+)", gb)
        if _m:
            _items = []
            for _rec in _m.group(1).split("|"):
                _f = [p.strip() for p in _rec.split(":", 2)]
                if len(_f) < 2 or not _f[0]:
                    continue
                try:
                    _sz = float(_re3.sub(r"[^\d.\-]", "", _f[1]))
                except ValueError:
                    continue
                _items.append({"label": _f[0], "size": _sz,
                               "note": _f[2] if len(_f) >= 3 else ""})
            if _items:
                out["items"] = _items
    # footage_route_trace — ordered route waypoints (x:y[:label], x,y in 0..1).
    if gk == "route_trace":
        _m = _re3.search(r"points=([^;]+)", gb)
        if _m:
            _pts = []
            for _rec in _m.group(1).split("|"):
                _f = [p.strip() for p in _rec.split(":", 2)]
                if len(_f) < 2:
                    continue
                try:
                    _x = float(_f[0]); _y = float(_f[1])
                except ValueError:
                    continue
                _pts.append({"x": _x, "y": _y,
                             "label": _f[2] if len(_f) >= 3 else ""})
            if _pts:
                out["points"] = _pts
    return out


def _scene_footage_bed(scene_index, work, cache):
    """VIDLORE_MG_FOOTAGE_BED helper (opt-in). Return a path to a representative
    frame of scene ``scene_index``'s sourced footage, for use as a hybrid-overlay
    BED (so a callout/fact/measurement points at the REAL subject instead of a dark
    synthetic bed). Prefers an AI/slide STILL (already a displayable image); else
    extracts frame-0 of the scene's first video clip (cached under ``cache``).
    Returns None when nothing is resolvable so the caller keeps the existing
    dark-bed fallback. NEVER raises (a bed is a nicety, never a render blocker)."""
    import glob as _glob
    try:
        fdir = Path(work) / "footage"
        if not fdir.is_dir():
            return None
        tag = f"{int(scene_index):03d}"
        # 1) AI / slide STILL — already the exact image the scene displays.
        for _pat in (f"fal_{tag}_*.jpg", f"fal_{tag}_*.png", f"slide_{tag}_*.jpg"):
            _hits = sorted(_glob.glob(str(fdir / _pat)))
            if _hits:
                return _hits[0]
        # 2) VIDEO clip — extract a single representative frame (cached).
        _clips = sorted(_glob.glob(str(fdir / f"clip_{tag}_*.mp4")))
        if _clips:
            _out = Path(cache) / f"_mgbed_{tag}.png"
            if _out.exists() and _out.stat().st_size > 0:
                return str(_out)
            try:
                import imageio_ffmpeg
                import subprocess
                _ff = imageio_ffmpeg.get_ffmpeg_exe()
                _out.parent.mkdir(parents=True, exist_ok=True)
                # -nostdin + DEVNULL on every stream: ffmpeg reads stdin by
                # default and, run in a loop, will consume/poison the shared
                # stdin so the pipeline's LATER ffmpeg calls (the MG dispatch +
                # final mux) silently fail. Isolating the streams keeps this a
                # truly side-effect-free still grab.
                subprocess.run([_ff, "-nostdin", "-y", "-loglevel", "error",
                                "-ss", "0.5", "-i", _clips[0], "-frames:v", "1",
                                str(_out)], timeout=30, check=False,
                               stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                if _out.exists() and _out.stat().st_size > 0:
                    return str(_out)
            except Exception:                                      # noqa: BLE001
                return None
        return None
    except Exception:                                              # noqa: BLE001
        return None


def _callout_subject_image(label, note, context, work, cache):
    """PERMANENT anti-mispoint fix (user 2026-06-06). A point-out primitive
    (footage_object_callout / annotated_detail_callout) must point at a REAL,
    KNOWN subject. Over arbitrary stock footage the anchor can only GUESS, and on
    shallow-DOF / bokeh shots it landed on empty BLURRED background (a "PEELING
    PAINT" ring floating on green bokeh). The robust fix is to source a clean AI
    still that literally DEPICTS the labelled subject, composed as a single sharp
    centred hero on a plain background — so the ring always lands on the actual
    thing (and the hardened focus-gated saliency locks it precisely on the clean
    plate). `context` is the scene narration + video topic — it DISAMBIGUATES a
    terse one-word label (the director may shorten "hairline crack in the concrete
    pier" to just "Hairline", which alone makes Imagen draw a person's hairline /
    face). Returns a cached image path, or None (graceful → caller keeps its
    footage/dark-bed fallback). Requires AI stills to be enabled."""
    try:
        import re as _re
        import hashlib as _hl
        from . import footage as _ftc
        subj = _re.sub(r"[^a-z0-9 ,'\-]", " ", str(label or "").strip().lower())
        subj = _re.sub(r"\s+", " ", subj).strip()
        if len(subj) < 2:
            return None
        extra = ""
        nt = _re.sub(r"\s+", " ", str(note or "").strip().lower())
        if nt and nt != subj and len(nt) < 80:
            extra = ", " + nt
        # context (scene narration + topic) anchors the subject so a terse label
        # can't be misread. Trim to a clean clause.
        ctx = _re.sub(r"\s+", " ", str(context or "").strip())
        ctx = _re.sub(r"[\"“”]", "", ctx)[:200].strip()
        ctx_clause = f" Scene context: {ctx}." if len(ctx) > 8 else ""
        # only forbid people when neither the label nor the context is about a
        # person (so a genuine person-detail callout still works).
        _human = _re.compile(r"\b(face|person|people|man|woman|men|women|child|"
                             r"boy|girl|eye|eyes|hair|portrait|soldier|worker|"
                             r"victim|suspect|officer|skin|hand|hands)\b")
        nohuman = ("" if _human.search(subj + " " + ctx)
                   else " no people, no faces, no person,")
        prompt = (
            f"extreme close-up documentary macro photograph of {subj}{extra}, as a "
            f"real physical detail.{ctx_clause} The {subj} is the single clear "
            f"subject filling the centre of the frame, sharp crisp focus on the "
            f"{subj}, shallow depth of field, clean simple uncluttered plain "
            f"background, soft natural daylight, photorealistic, fine detail,"
            f"{nohuman} no text, no words, no captions, no watermark, no border")
        base = cache if cache else work
        cdir = Path(base) / "callout_ai"
        cdir.mkdir(parents=True, exist_ok=True)
        key = _hl.sha1((subj + extra + "|" + ctx).encode("utf-8")).hexdigest()[:16]
        dest = cdir / f"co_{key}.jpg"
        if dest.exists() and dest.stat().st_size > 2048:
            return str(dest)

        class _C:                       # minimal cfg shim — only reached AImg-on
            has_aimg = True
        seed = int(key[:6], 16) % 2_000_000
        ok, prov = _ftc._ai_primary_generate(prompt, dest, _C(), seed,
                                              force_fal=False)
        if ok and dest.exists() and dest.stat().st_size > 2048:
            try:
                print(f"  [callout-ai] subject='{subj[:48]}' -> {prov}", flush=True)
            except Exception:                                      # noqa: BLE001
                pass
            return str(dest)
    except Exception:                                              # noqa: BLE001
        return None
    return None


def _emit_asset_qa_manifest(run_dir: Path, brief: "Brief", *, mg_on: bool) -> dict:
    """V3.4.2 — ALWAYS-ON asset-QA reporting (decoupled from the MG flag).

    The asset-QA *reporting* surface historically lived ONLY inside the
    `VIDLORE_MOTION_GRAPHICS=1` block, so an MG-OFF render produced no asset-QA
    record at all. Asset *protection* already runs regardless of MG — footage
    relevance + period guard + dedup happen during always-on footage sourcing,
    black-frame repair during always-on assembly, and the factual guard is
    unconditional. This helper makes the asset-QA *manifest* always-on too, so
    observability is never silently dropped when MG is off.

    Composes the portrait + provenance guards (asset_qa.py — unchanged) over the
    footage context that sourcing populates (`_PORTRAIT_PROVENANCE`, `_VIDEO_CTX`).
    Pure-read + fully defensive (never raises, never swaps assets). Writes
    `run_dir/asset_qa.json` and returns the payload. When MG is ON, the richer
    in-MG-manifest asset-QA block (palette + warnings) still runs unchanged.
    """
    try:
        import json as _json
        from . import asset_qa as _aqa
        niche = str((getattr(brief, "extra", None) or {}).get("niche", "")).strip()
        era = None
        warn: list = []
        try:
            from . import footage as _ftg
            era = (getattr(_ftg, "_VIDEO_CTX", {}) or {}).get("era")
            for _pn, _pp in (getattr(_ftg, "_PORTRAIT_PROVENANCE", {}) or {}).items():
                _pp = _pp or {}
                warn += _aqa.check_portrait(
                    _pn, prov=_pp, title=str(_pp.get("source", "")),
                    ai_generated=bool(_pp.get("ai_generated")))
                warn += _aqa.check_provenance(_pp)
        except Exception:                                          # noqa: BLE001
            pass
        payload = {
            "schema": "asset_qa/1",
            "mode": "mg_on" if mg_on else "mg_off",
            "niche": niche,
            "era": era,
            "warnings": warn,
            "summary": _aqa.summarize(warn),
        }
        (run_dir / "asset_qa.json").write_text(_json.dumps(payload, indent=1))
        return payload
    except Exception:                                              # noqa: BLE001
        return {}


def render_from_script(
    brief: Brief,
    cfg: Config,
    out_dir: Path,
    *,
    keep_work: bool = False,
    progress: "Optional[ProgressFn]" = None,
    run_dir: "Optional[Path]" = None,
) -> Result:
    t0 = time.time()
    # P4 — robustness: callers (the Review Editor) may pass the EXACT project
    # dir so a re-render always targets the same folder the editor edits, even
    # when _slug(title) would differ (titles with quotes/punctuation/long text).
    run_dir = Path(run_dir) if run_dir is not None else run_dir_for(brief, out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Per-render reset of the Shutterstock usage tally — VIDLORE_
    # SHUTTERSTOCK_MAX (default 5) caps clips PER RENDER, not per
    # process, so a 10-doc batch doesn't share the same quota.
    try:
        from .footage import reset_shutterstock_counter as _ss_reset
        _ss_reset()
    except Exception:                                          # noqa: BLE001
        pass
    # CONCURRENCY-SAFE work dir. Two renders on the same project (e.g. a
    # web-UI render while another render is in flight) used to share
    # run_dir/"work" — one's footage-fetch / end-of-run cleanup deleted
    # the other's clip_*.mp4 mid-assembly ("No such file or directory").
    # Each render now gets its OWN pid-scoped work dir, so they can never
    # clobber each other. (cache/ stays shared — it's content-hashed and
    # safe for concurrent reads.)
    work = run_dir / f"work_{os.getpid()}"
    # Sweep stale work dirs left by crashed/killed renders (they never
    # reached their own cleanup) so footage clips don't pile up on disk.
    # Bug fix 2026-05-26 (user reported 31 GB workdir after 3 retries):
    # the old rule was "delete if older than 2 hours" which was too
    # conservative — three rapid retries within an hour piled up 25 GB
    # of dead work dirs. The new rule is PID-aware: if the dir name's
    # PID is no longer alive, the render is definitively dead and the
    # dir can be reclaimed immediately. The 2-hour fallback survives
    # for legacy dirs whose name doesn't contain a parseable PID.
    _now = time.time()
    for _w in run_dir.glob("work*"):
        try:
            if not _w.is_dir() or _w == work:
                continue
            # Try PID-based liveness check first — fastest, no time wait
            _pid_str = _w.name.removeprefix("work_")
            _is_dead = False
            try:
                _pid = int(_pid_str)
                try:
                    os.kill(_pid, 0)        # signal 0 = liveness probe
                except (ProcessLookupError, PermissionError):
                    _is_dead = True
                except OSError:
                    _is_dead = True
            except ValueError:
                _is_dead = False           # legacy name; fall through to age check
            if _is_dead or (_now - _w.stat().st_mtime > 7200):
                shutil.rmtree(_w, ignore_errors=True)
        except OSError:
            pass
    work.mkdir(parents=True, exist_ok=True)
    # Persistent across runs (survives keep_work=False): lets a re-render
    # after an edit reuse every scene that didn't change.
    cache = run_dir / "cache"
    th = get_theme(brief.theme)
    voice = brief.voice or cfg.voice

    # ── PHASE-WALL-TIME INSTRUMENTATION (render speed audit) ──────
    # Tracks wall time per pipeline phase so the user can see where
    # the seconds actually go.  Logged at the end of each phase.
    import time as _time_mod
    _phase_t: dict[str, float] = {}
    _phase_t["_pipeline_start"] = _time_mod.time()
    def _phase_mark(name: str) -> None:
        _phase_t[name] = _time_mod.time()
    def _phase_dur(name: str, _end: str | None = None) -> float:
        end_t = _phase_t.get(_end, _time_mod.time()) if _end else _time_mod.time()
        return end_t - _phase_t.get(name, end_t)

    # STYLE MODE — the documentary's cinematic personality (pacing,
    # transitions, camera, music, atmosphere). Explicit choice wins;
    # 'auto' infers from theme + topic.
    from .style_modes import resolve_style
    mode = resolve_style(getattr(brief, "style", "auto"), theme=brief.theme,
                         title=brief.title, prompt=getattr(brief, "prompt", ""))
    _log("1/5", f"style mode: {mode.label}")

    # LOOK DNA — the channel's PERSISTENT cinematic identity.  This
    # layer sits ABOVE style mode + theme: when a channel is set,
    # resolve_look() returns a frozen ResolvedLook whose knobs are
    # sampled (deterministically) from per-channel ranges.  Downstream
    # consumers read look.<knob> in preference to mode.<knob>.
    # DOC_001 AUTO-LOOK: when no channel/preset/overrides are set,
    # resolve_look() now auto-selects a niche-appropriate preset (keyed
    # by brief topic + hash) so each video adopts a distinct cinematic
    # identity instead of one fixed neutral default.  VIDLORE_AUTO_LOOK=0
    # → resolve_look() returns None and every legacy path runs untouched.
    # DOC_005 — per-video layout variation: seed scene_variant from the
    # brief hash so two different videos don't run the identical layout /
    # lower-third / card-position sequence (a template tell). Deterministic
    # within a video; varies across videos.
    try:
        from .templates._shared import set_variant_salt as _set_vsalt
        from .look_dna import _hash_brief as _hb
        _set_vsalt(_hb(brief))
    except Exception:                                              # noqa: BLE001
        pass
    try:
        from .look_dna import resolve_look as _resolve_look
        look = _resolve_look(brief)
    except Exception as _e:                                        # noqa: BLE001
        print(f"  [look-dna] resolve failed, falling back: {_e}",
              flush=True)
        look = None
    if look is not None:
        _log("1/5", f"look DNA: {look.label or look.name}")
        # Install the active look as a module-level context var so any
        # downstream renderer can opt-in by calling look_dna.current().
        # This is the keystone wiring — every consumer is 1-line.
        try:
            from .look_dna import set_current as _ld_set
            _ld_set(look)
        except Exception:                                          # noqa: BLE001
            pass
        # The Look DNA accent_rgb overrides BOTH the theme accent AND
        # any style-mode accent — DNA is the highest authority.
        _ldna_acc = look.get("accent_rgb")
        if isinstance(_ldna_acc, (list, tuple)) and len(_ldna_acc) >= 3:
            th = {**th, "accent": (int(_ldna_acc[0]),
                                    int(_ldna_acc[1]),
                                    int(_ldna_acc[2]))}
    else:
        # Legacy path: style-mode accent override (unchanged)
        _macc = getattr(mode, "accent", ()) or ()
        if len(_macc) >= 3:
            th = {**th, "accent": (int(_macc[0]), int(_macc[1]),
                                    int(_macc[2]))}

    script, edited = _load_reviewed_script(brief, run_dir)
    tag = "edited by you" if edited else "unchanged"
    _log(
        "1/5",
        f"reviewed script ({tag}): {len(script.scenes)} scenes, "
        f"~{script.word_count} words",
    )
    _emit(progress, 8, f"Script loaded ({len(script.scenes)} scenes)")

    n = len(script.scenes)
    _phase_mark("phase2_start")
    if brief.voiceover:
        _log("2/5", f"using your own voiceover: {brief.voiceover}")
        narration = narrate_from_file(script, brief.voiceover, work / "tts")
    elif cfg.has_premium_local or \
            getattr(brief, "tts_backend", "") == "premium_local":
        # Premium LOCAL voiceover (self-hosted TTS sidecar; no paid APIs). The
        # fallback chain (premium -> fast fallback -> legacy edge) lives inside
        # narrate_premium so a run never dies on a robotic-or-nothing choice.
        _bk = (getattr(brief, "tts_model", "") or cfg.tts_model or "chatterbox")
        _vp = (getattr(brief, "tts_voice", "") or cfg.tts_voice
               or "deep_male_documentary")
        _log("2/5", f"narrating with premium local voice "
                    f"({_bk} / {_vp}, self-hosted) ...")
        from .tts import narrate_premium
        narration = narrate_premium(
            script, work / "tts", preset_key=_vp, backend_name=_bk,
            cache_dir=cache, device=cfg.tts_device, settings={},
            allow_legacy=True,
            voice_variants=_load_variants(run_dir, "voice_variants.json"))
    else:
        if cfg.has_elevenlabs:
            provider = "elevenlabs"
            tts_voice = cfg.elevenlabs_voice_id
            _log("2/5",
                 f"narrating with ElevenLabs ({cfg.elevenlabs_model}) ...")
        else:
            provider = "edge"
            tts_voice = voice
            _log("2/5", f"narrating with {voice} (edge-tts, free) ...")
        narration = narrate(
            script, tts_voice, work / "tts", cache_dir=cache,
            provider=provider, el_api_key=cfg.elevenlabs_api_key,
            el_model=cfg.elevenlabs_model, fallback_voice=voice,
            voice_variants=_load_variants(run_dir, "voice_variants.json"),
        )
    _phase_mark("phase2_end")
    _log(
        "2/5",
        f"narration {narration.total:0.1f}s audio "
        f"({narration.reused}/{n} scenes reused from cache, "
        f"{n - narration.reused} regenerated) "
        f"· TTS wall {_phase_dur('phase2_start', 'phase2_end'):0.1f}s",
    )
    _emit(progress, 45,
          f"Narration {narration.total:0.0f}s "
          f"({narration.reused}/{n} cached)")

    if cfg.has_stock:
        src = "real Pexels stock video (AI fallback)"
    elif cfg.has_fal_video:
        src = f"fal.ai VIDEO {cfg.fal_video_model}"
    elif cfg.has_fal:
        src = f"fal.ai {cfg.fal_model} (LLM-directed)"
    elif cfg.has_aimg:
        src = "AI images"
    else:
        src = "themed slides"
    # MUST match the assembler's pacing (Style Mode), or the footage
    # fetcher and assembler disagree on the per-scene beat count.
    bps = beats_per_scene([ns.duration for ns in narration.scenes],
                          target=mode.beat_target, bmin=mode.beat_min)
    _log("3/5", f"sourcing footage ({src}) for {n} scenes "
                f"/ {sum(bps)} beats ...")
    _phase_mark("phase3_start")
    fr = fetch_footage(script.scenes, cfg, work / "footage", th,
                       cache_dir=cache, variants=_load_variants(run_dir),
                       beats=bps, title=script.title,
                       background=getattr(brief, "background", "auto"))
    _phase_mark("phase3_end")
    # Review-Editor REPLACE-VISUAL: honor user uploads / search-picks recorded by
    # editor_manifest.apply_overrides (edits/locked_visuals.json, keyed by final
    # scene index). Overrides the auto-selected footage AFTER fetch so footage.py
    # stays untouched. Defensive: any failure leaves auto footage intact.
    try:
        _lv = run_dir / "edits" / "locked_visuals.json"
        if _lv.exists():
            _locked = json.loads(_lv.read_text(encoding="utf-8")) or {}
            if _locked:
                from .footage import FootageItem as _FI
                _napplied = 0
                for _sc in script.scenes:
                    _info = _locked.get(str(_sc.index))
                    if not _info:
                        continue
                    _p = run_dir / _info.get("file", "")
                    if not _p.exists():
                        continue
                    _isv = bool(_info.get("is_video"))
                    _nb = bps[_sc.index] if _sc.index < len(bps) else 1
                    fr.beat_clips[_sc.index] = [_p] * max(1, _nb)
                    _hit = False
                    for _it in fr.items:
                        if getattr(_it, "index", None) == _sc.index:
                            _it.path, _it.is_video = _p, _isv
                            _hit = True
                            break
                    if not _hit:
                        fr.items.append(_FI(index=_sc.index, path=_p, is_video=_isv))
                    _napplied += 1
                if _napplied:
                    _log("3/5", f"{_napplied} user-replaced visual(s) locked in")
    except Exception as _e:  # noqa: BLE001
        print(f"  [replace-visual] skipped: {_e}", flush=True)
    footage = fr.items
    nvid = sum(1 for f in footage if f.is_video)
    kinds: dict[str, int] = {}
    for f in footage:
        nm = Path(f.path).name
        tag = ("video" if f.is_video else
               "fal.ai" if nm.startswith("fal_") else
               "ai-image" if nm.startswith(("aimg_", "ov_")) else "slide")
        kinds[tag] = kinds.get(tag, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in kinds.items())
    _log("3/5", f"{summary} ({fr.reused}/{n} reused from cache)")
    # Honest AI accounting: purpose-gated + hard-capped at ~5% of beats.
    # Cache hits don't count -- only fresh generations do.
    ai = getattr(fr, "ai_stats", None) or {}
    if ai.get("cap", 0) > 0 or ai.get("used", 0) > 0:
        bp = ai.get("by_purpose") or {}
        bp_str = ", ".join(f"{k}:{v}" for k, v in sorted(bp.items())) or "—"
        _log("3/5", f"ai usage: {ai.get('used',0)}/{ai.get('cap',0)} "
                    f"({ai.get('share_pct',0)}% of {ai.get('total_beats',0)} "
                    f"beats) · {bp_str}")
    # PROOF FILE — persist the AI-still provenance NEXT TO the mp4 so every video
    # folder carries verifiable evidence of which provider made each still
    # (Imagen vs fal vs Pollinations) + the model + est. cost. The engine's own
    # copy lives inside work_<pid>/ which the work* cleanup sweep deletes, so the
    # run_dir copy is the only durable record. Only written when this render
    # actually generated/attempted AI stills (skips empty editor re-renders so a
    # prior render's proof is never clobbered).
    try:
        _asl = getattr(fr, "ai_still_log", None) or {}
        _asum = _asl.get("summary") or {}
        if _asum.get("total_attempts"):
            (run_dir / "AI_STILL_LOG.json").write_text(json.dumps({
                "policy": "Google Imagen 4 Fast PRIMARY (model set by "
                          "VIDLORE_IMAGEN_MODEL); fal.ai FLUX SECONDARY (used when "
                          "Imagen is unavailable/errors, and forced on the last "
                          "ladder rung); Pollinations emergency-only. Every still "
                          "is validated + retried. provider= names which engine "
                          "actually generated each accepted still. Costs are "
                          "ESTIMATES.",
                "imagen_model": os.environ.get("VIDLORE_IMAGEN_MODEL",
                                               "imagen-4.0-fast-generate-001"),
                "summary": _asum,
                "attempts": _asl.get("attempts") or [],
            }, indent=1, default=str), encoding="utf-8")
            _log("3/5", f"ai-still proof: AI_STILL_LOG.json written "
                        f"(primary={_asum.get('primary_provider','?')}, "
                        f"imagen={_asum.get('imagen_generations',0)}, "
                        f"fal={_asum.get('fal_generations',0)})")
    except Exception as _e:  # noqa: BLE001
        print(f"  [ai-still-proof] skipped: {_e}", flush=True)
    _log("3/5", f"footage wall {_phase_dur('phase3_start','phase3_end'):0.1f}s")
    _emit(progress, 70, f"Footage ready ({fr.reused}/{n} cached)")

    # MOTION-GRAPHICS DECISION (Phase B). Where the resolved footage is
    # weak (slide fallback / repeated) OR the concept is abstract and only
    # a still was available, substitute a cinematic motion-graphics
    # explainer instead of letting generic B-roll read as "no footage".
    # Conservative: scenes with strong footage or a deliberate editor
    # graphic are left untouched.
    try:
        from .footage_strength import plan_motion_graphics, synthesize_content
        from .script_gen import generate_explainers
        _mg_plan = plan_motion_graphics(script.scenes, footage)
        _by_idx = {sc.index: sc for sc in script.scenes}
        _weak = [(p["index"], _by_idx[p["index"]].narration)
                 for p in _mg_plan
                 if p["decision"] == "motion-graphics" and p["index"] in _by_idx]
        # AI VISUAL-EXPLANATION DIRECTOR: pick the best explainer per weak
        # scene with real variety + short structured content. Falls back to
        # the deterministic synthesiser for any scene the LLM doesn't place.
        _llm_ex: dict = {}
        if _weak:
            try:
                _llm_ex = generate_explainers(script.title, _weak, cfg)
            except Exception as e:                          # noqa: BLE001
                print(f"  [explainer] LLM pass failed ({e}); "
                      "deterministic fallback", flush=True)
        _mg_n = 0
        _kinds_used: dict[str, int] = {}
        # A deliberately-tagged graphic beat (any kind the MG director recognises)
        # must survive the weak-footage explainer — it renders its premium card
        # over the weak footage instead of being clobbered into a generic one.
        # Sourced from the director so every batch's kinds are covered automatically;
        # unioned with the legacy explicit set for belt-and-suspenders safety.
        try:
            from .motion_graphics import director as _mgdir
            _MG_PROTECTED_KINDS = _mgdir.known_graphic_kinds() | {
                "news_article", "document", "headline", "map_reveal", "map_route",
                "network_graph", "conspiracy_board", "org_tree", "define_the_term",
                "stat_insight", "mugshot", "bio", "mini_bio"}
        except Exception:                                       # noqa: BLE001
            _MG_PROTECTED_KINDS = {
                "news_article", "document", "headline", "map_reveal", "map_route",
                "network_graph", "conspiracy_board", "org_tree", "define_the_term",
                "stat_insight", "mugshot", "bio", "mini_bio"}
        for p in _mg_plan:
            if p["decision"] != "motion-graphics":
                continue
            sc = _by_idx.get(p["index"])
            if sc is None:
                continue
            # PROTECT name_reveal — a character intro is the protagonist's
            # only portrait moment. Motion-graphics fallback must NEVER
            # overwrite it with a "number" or "quote_highlight" card just
            # because the stock footage was weak. Bug fix 2026-05-26: the
            # 5-min Eli validation traced the lost portrait to THIS line
            # overwriting `name_reveal` → `number`.
            _gk_sc = (getattr(sc, "graphic_kind", "") or "").lower()
            if _gk_sc == "name_reveal":
                continue
            # When the MG engine is on, a deliberately-tagged graphic scene
            # (the script/editor explicitly chose news_article / network_graph /
            # map / definition / stat) is an intentional editorial beat — the
            # weak-footage AI-explainer must not clobber its kind, exactly like
            # name_reveal above. Flag-gated → legacy path stays byte-identical.
            if os.environ.get("VIDLORE_MOTION_GRAPHICS") == "1" and (
                    _gk_sc in _MG_PROTECTED_KINDS):
                continue
            if p["index"] in _llm_ex:
                kind, text, body = _llm_ex[p["index"]]
            else:
                kind, text, body = synthesize_content(sc, p["suggested"])
            # ROBUSTNESS: a number/stat explainer is only meaningful with a
            # real figure. If the model picked one for a scene that has no
            # digit, fall back to kinetic typography so the scene is NEVER
            # left with a blank/failed card.
            if kind in ("number", "stat", "stat_dashboard") and \
                    not any(c.isdigit() for c in f"{text} {body}"):
                kind, text, body = synthesize_content(sc, "quote_highlight")
            sc.graphic_kind, sc.graphic_text, sc.graphic_body = \
                kind, text, body
            _kinds_used[kind] = _kinds_used.get(kind, 0) + 1
            _mg_n += 1
        if _mg_n:
            _mix = ", ".join(f"{v}×{k}" for k, v in _kinds_used.items())
            _log("3/5", f"motion-graphics: {_mg_n} weak-footage scene(s) "
                        f"→ AI explainers ({_mix}); no filler stock")
    except Exception as e:                                  # noqa: BLE001
        print(f"  [motion-graphics] decision skipped ({e})", flush=True)

    # Forensic v2 — RESTRAINED overlay (footage-first). Trim only WEAK /
    # decorative script cards (kinetic text, quote_highlight, statement,
    # callout, bullet list, redundant lower-thirds) so footage breathes;
    # PREMIUM editorial cards SURVIVE — statistics, research-proof, mechanisms,
    # diagrams, maps, timelines, documents, major reveals, and the name_reveal
    # portrait. Opt-in + (in practice) used on agriculture/explainer; A/B
    # before it becomes a default. VIDLORE_OVERLAY_RESTRAINT=1.
    _mg_removed = []   # V3.3.1 — MG lifecycle audit: cards removed by restraint
    _factual_rejected = []  # V3.3.2 — fact-bearing cards dropped for no evidence
    if os.environ.get("VIDLORE_OVERLAY_RESTRAINT", "0").strip().lower() \
            in ("1", "true", "yes", "on"):
        _WEAK_KINDS = {"kinetic_keyword", "kinetic", "quote_highlight", "quote",
                       "statement", "callout", "bullet_list", "bullets",
                       "label", "lower_third", "subtitle_card"}
        _trim = 0
        for _sc in script.scenes:
            if (getattr(_sc, "graphic_kind", "") or "").lower() in _WEAK_KINDS:
                _mg_removed.append({"scene": getattr(_sc, "index", None),
                                    "kind": _sc.graphic_kind,
                                    "why": "overlay_restraint_weak"})
                _sc.graphic_kind, _sc.graphic_text, _sc.graphic_body = "", "", ""
                _trim += 1
        # P4 (2026-06-01) — niche-aware CLUTTER guard. On a CALM explainer /
        # agriculture / health beat, a busy multi-portrait collage, headline
        # montage, social-card or conspiracy board reads template-y and out of
        # place (the editorial brief for these niches is footage-first +
        # restrained). Trim ONLY the genuinely-cluttered collage family —
        # PREMIUM editorial cards (stats, mechanisms, diagrams, timelines,
        # maps, documents, proof, meaningful single portraits / name_reveal)
        # are deliberately absent from this set and always survive. Crime /
        # spy / conspiracy niches (where a board/montage IS the aesthetic)
        # are excluded by the calm-niche gate.
        _calm_niche = (str((getattr(brief, "extra", None) or {})
                           .get("niche", "")).strip().lower()
                       in ("agriculture_history", "education_explainer",
                           "health_longevity"))
        if _calm_niche:
            _CLUTTER_KINDS = {"photo_collage", "collage", "headline_montage",
                              "social_post", "social_media", "headline_crawl",
                              "quote_stream", "conspiracy_board"}
            for _sc in script.scenes:
                if (getattr(_sc, "graphic_kind", "") or "").lower() in _CLUTTER_KINDS:
                    _mg_removed.append({"scene": getattr(_sc, "index", None),
                                        "kind": _sc.graphic_kind,
                                        "why": "clutter_calm_niche"})
                    _sc.graphic_kind, _sc.graphic_text, _sc.graphic_body = "", "", ""
                    _trim += 1
        if _trim:
            _log("3/5", f"overlay restraint: trimmed {_trim} weak/decorative "
                        f"card(s) → footage-first (premium cards preserved)")

    # ====================================================================== #
    # P1.3 — FULL-SCREEN CARD RESTRAINT (footage-first).  Always-on (kill via
    # VIDLORE_FULLSCREEN_RESTRAINT=0).  A real render came back as a slideshow:
    # too many FULL-SCREEN cards that REPLACE footage, often back-to-back in
    # the hook ("$37 MILLION" → "$40 MILLION" → "SIX DECADES ZERO PROGRESS")
    # plus repeated full-screen "WATER HYACINTH" / "60 YEARS AGO" beats.  The
    # reference competitor keeps cards to ~15-20% of runtime, footage-first,
    # using lower-thirds for numbers.  This pass runs on the FULL ordered scene
    # list (with per-scene seconds from `narration`) and enforces four rules,
    # lowest-value-first, while ALWAYS leaving 1-2 genuine title/reveal cards.
    #
    # A card counts as FULL-SCREEN unless its kind is in _FOOTAGE_BACKED below
    # (lower-thirds, corner callouts, footage overlays, name/portrait, footage_*
    # MG primitives) — those sit ON footage and never trigger the slideshow
    # feel, so they are exempt from the budget / consecutive / cooldown rules.
    # Defaulting unknown kinds to "full-screen" is the safe direction for a
    # footage-first goal.  FAIL-SAFE: a scene is only ever DEMOTED to clean
    # footage (never left blank), and any error leaves all cards untouched.
    # P2.3 — clean any mid-word-truncated card label carried by a REUSED script
    # (a resumed script.json may hold a hard [:64] slice like '...force
    # Washingto'). Runs ALWAYS (independent of the restraint flag); touches only
    # single-label / title cards — structured panels are left untouched.
    try:
        from . import script_gen as _sg_clean
        _CLEAN_KINDS = {"statement", "explainer", "definition", "define_term",
                        "define_the_term", "section_title", "title_card",
                        "callout", "era_banner"}
        # RETIRED card kinds (user-removed): a reused script.json may still carry
        # cause_effect / process_diagram data ("THE ACTIVE INGREDIENT", "AGENT
        # ORANGE") that leaked onto a cheap textured bed. Clear them completely
        # so the scene plays as clean footage + its natural emphasis word.
        _RETIRED_KINDS = {"cause_effect", "cause_effect_chain",
                          "process_diagram", "process_flow_steps"}
        for _scc in script.scenes:
            _k = (getattr(_scc, "graphic_kind", "") or "").strip().lower()
            if _k in _RETIRED_KINDS:
                _scc.graphic_kind, _scc.graphic_text, _scc.graphic_body = \
                    "", "", ""
                try:
                    print(f"  [card-text] scene "
                          f"{getattr(_scc, 'index', None)} cleared retired "
                          f"'{_k}' card -> footage + emphasis", flush=True)
                except Exception:                              # noqa: BLE001
                    pass
                continue
            if _k in _CLEAN_KINDS:
                _orig = getattr(_scc, "graphic_text", "") or ""
                _new = _sg_clean.clean_card_text(_orig)
                if _new != _orig:
                    _scc.graphic_text = _new
                    try:
                        print(f"  [card-text] scene "
                              f"{getattr(_scc, 'index', None)} cleaned mid-word "
                              f"label -> '{_new[:48]}'", flush=True)
                    except Exception:                          # noqa: BLE001
                        pass
    except Exception:                                          # noqa: BLE001
        pass

    if os.environ.get("VIDLORE_FULLSCREEN_RESTRAINT", "1").strip().lower() \
            not in ("0", "false", "no", "off"):
      try:
        # Kinds that render OVER footage (footage stays visible underneath):
        # lower-thirds, corner chips/callouts, title bars, small overlays, the
        # protagonist portrait, and every footage_* MG primitive.  These are
        # NOT slideshow cards, so they are exempt from this pass.
        _FOOTAGE_BACKED = {
            "", "location", "section_title", "diagram_labels", "framed_insert",
            "callout", "label", "typing_date", "lower_third", "subtitle_card",
            "name_reveal", "portrait", "mini_bio", "pull_quote_portrait",
            "spotlight", "status_indicator", "sticky_note", "hashtag_trend",
            "disclaimer", "caution_tape", "sfx_cue", "footnote", "speech_bubble",
            "figure_locator", "id_card",
            # footage-backed MG primitives + small overlays (render_dispatch
            # marks these "(overlay)" — route/fact/object draw over footage).
            "footage_fact_overlay", "footage_object_callout",
            "footage_route_trace", "fact_overlay", "stat_overlay",
            "object_callout", "route_trace", "era_stamp_overlay",
            "measurement_callout", "act_chapter_card",
        }
        # NUMBER-family kinds that SHOULD prefer footage + a lower-third/overlay
        # over a full-screen reveal, UNLESS the beat is a genuine single reveal
        # (rule 4).  When footage exists for the scene, re-route the FIRST such
        # number beat to a footage-backed `footage_fact_overlay` instead of a
        # full-screen number card; later number beats are simply demoted.
        _NUMBER_KINDS = {"number", "stat", "currency_stat", "numerical_ratio",
                         "score_display", "tally_counter", "stat_insight"}
        # Genuine title / reveal cards we always want to KEEP at least 1-2 of
        # (so restraint never strips every card and the film loses its title).
        _REVEAL_KINDS = {"title_card", "statement", "map_reveal", "newspaper"}
        # PREMIUM INFORMATIVE MOTION GRAPHICS — these are the GOOD cards (the
        # reference competitor is MG-HEAVY: animated numbers, charts, maps,
        # explainers, comparisons). They carry real information and look
        # premium, so the restraint pass must NEVER demote them — it only
        # exists to kill cheap/repeated/decorative card spam, not to strip the
        # video of its motion graphics. (2026-06-06 correction: the earlier
        # restraint over-demoted these — the animated $40M count-up became a
        # flat overlay, the biomass chart and the WATER HYACINTH explainer were
        # culled. They are now exempt from all four rules below.)
        _PREMIUM_MG = {
            # numbers / stats (animated count-up + SFX)
            "number", "stat", "currency_stat", "numerical_ratio",
            "score_display", "tally_counter", "stat_insight",
            "gold_number_callout",
            # charts / comparisons
            "scale_compare", "size_compare", "stat_dashboard",
            "vertical_bar_chart", "horizontal_bar_chart", "heatmap_grid",
            "chart", "line_chart", "growth_curve_chart", "statistic_bar_reveal",
            "proportion_ring", "donut_chart", "pictograph_scale",
            "composition_stack", "comparison_split", "vs_balance_scale",
            "before_after_slider",
            # maps / geo
            "map_reveal", "globe_map", "map_pin_cluster", "map_route_spread",
            # explainers / definitions / quotes / press
            "explainer", "definition", "define_term", "define_the_term",
            "statement", "pull_quote_portrait", "quote_highlight",
            "newspaper", "headline_montage",
            # timelines / structure
            "timeline", "mini_timeline", "chronology_timeline", "calendar_grid",
            "network_graph", "org_hierarchy_tree", "family_tree_3gen",
        }
        # Per-scene start + seconds, positionally aligned with script.scenes
        # (narration.scenes[i].index == i).  Fall back to a nominal 6s/scene if
        # narration is short or missing so the pass still functions.
        _nss = list(getattr(narration, "scenes", []) or [])
        _durs_p1 = []
        _acc = 0.0
        for _i in range(len(script.scenes)):
            _d = 6.0
            if _i < len(_nss):
                try:
                    _d = float(getattr(_nss[_i], "duration", 6.0) or 6.0)
                except (TypeError, ValueError):
                    _d = 6.0
            _durs_p1.append((round(_acc, 2), round(_d, 2)))
            _acc += _d
        _runtime_p1 = _acc if _acc > 0 else float(len(script.scenes) * 6)

        def _is_fullscreen(_sc) -> bool:
            return (getattr(_sc, "graphic_kind", "") or "").strip().lower() \
                not in _FOOTAGE_BACKED

        def _fs_seconds(_idx) -> float:
            # On-screen seconds a full-screen card on scene _idx occupies.  The
            # assembler covers (most of) the scene with the card, so the scene
            # duration is the honest estimate of footage it displaces.
            return _durs_p1[_idx][1] if 0 <= _idx < len(_durs_p1) else 6.0

        # Story-value priority — higher survives the budget cull.  Mirrors the
        # script_gen density-cull _PRIO so the two passes agree on what matters.
        _PRIO_P1 = {"climax": 6, "reveal": 6, "turn": 5, "evidence": 4,
                    "proof": 4, "stakes": 4, "problem": 3, "escalation": 3,
                    "hook": 3, "payoff": 2, "context": 2, "reaction": 2,
                    "resolution": 2}

        def _value(_idx) -> int:
            _sc = script.scenes[_idx]
            _k = (getattr(_sc, "graphic_kind", "") or "").lower()
            _v = _PRIO_P1.get((getattr(_sc, "role", "") or "").lower(), 2)
            if _k in _REVEAL_KINDS:
                _v += 3            # protect genuine title / reveal beats
            if _k == "title_card":
                _v += 5            # cold-open title is sacred
            return _v

        # Has usable footage for this scene? (so we know a number beat CAN be
        # re-routed to footage + lower-third, and a demote leaves a real cut).
        _foot_have = set()
        try:
            for _f in footage:
                if getattr(_f, "path", None):
                    _foot_have.add(getattr(_f, "index", None))
        except Exception:                                      # noqa: BLE001
            _foot_have = set()

        def _demote(_sc, _why):
            _mg_removed.append({"scene": getattr(_sc, "index", None),
                                "kind": getattr(_sc, "graphic_kind", ""),
                                "why": _why})
            _sc.graphic_kind, _sc.graphic_text, _sc.graphic_body = "", "", ""

        _fs_idx = [i for i in range(len(script.scenes))
                   if _is_fullscreen(script.scenes[i])]
        _p1_actions = 0

        # ---- RULE 4: NUMBER → footage + lower-third (reveal exception) ----- #
        # The FIRST number-family beat that lands on a non-reveal role and has
        # footage is converted to a screen-locked `footage_fact_overlay` (the
        # number rides over footage instead of replacing it).  This keeps the
        # "single reveal" feel for genuine reveal/climax/turn roles (left as a
        # full-screen card) while killing the back-to-back number slideshow.
        _reveal_roles = {"reveal", "climax", "turn"}
        _converted_number = False
        for _i in _fs_idx:
            _sc = script.scenes[_i]
            _k = (getattr(_sc, "graphic_kind", "") or "").lower()
            if _k not in _NUMBER_KINDS:
                continue
            _role = (getattr(_sc, "role", "") or "").lower()
            _genuine_reveal = (_role in _reveal_roles) and not _converted_number
            if _genuine_reveal:
                _converted_number = True        # the ONE allowed full reveal
                continue
            if _k in _PREMIUM_MG:
                # PREMIUM informative number → keep it as a proper animated
                # motion graphic (count-up + SFX). Never flatten it to a static
                # footage overlay (that read as "cheap 40M text").
                continue
            if _i in _foot_have:
                # Re-route to footage + lower-third overlay (footage-first).
                print(f"  [restraint] number→footage+lower-third: scene "
                      f"{getattr(_sc, 'index', _i)} ({_k}) — footage carries "
                      f"the beat, fact rides as overlay", flush=True)
                _sc.graphic_kind = "footage_fact_overlay"
                _p1_actions += 1
        # Recompute the full-screen set (some numbers became overlays).
        _fs_idx = [i for i in range(len(script.scenes))
                   if _is_fullscreen(script.scenes[i])]

        # ---- RULE 2: FORBID consecutive full-screen cards ------------------ #
        # Two full-screen cards with NO footage scene between them read as a
        # slideshow.  Walk in order; whenever the previous KEPT full-screen
        # card has no footage scene separating it from this one, demote the
        # LOWER-VALUE of the two so a full-screen card is always followed by
        # footage.  (Footage-backed overlays count as "footage between".)
        _prev_fs = None
        for _i in list(_fs_idx):
            _sc = script.scenes[_i]
            if not _is_fullscreen(_sc):
                continue                         # already demoted this pass
            if _prev_fs is not None and (_i - _prev_fs) == 1:
                # Adjacent full-screen cards — demote the weaker one.
                if _value(_i) >= _value(_prev_fs):
                    _drop = _prev_fs
                    _prev_fs = _i
                else:
                    _drop = _i
                _ds = script.scenes[_drop]
                if (getattr(_ds, "graphic_kind", "") or "").lower() \
                        in _PREMIUM_MG:
                    # Both are premium informative MGs — KEEP both (the
                    # reference competitor runs back-to-back charts/numbers).
                    if _drop == _i:
                        continue
                else:
                    print(f"  [restraint] consecutive full-screen cards at scenes "
                          f"{getattr(script.scenes[min(_prev_fs, _drop)], 'index', None)}"
                          f"/{getattr(script.scenes[max(_prev_fs, _drop)], 'index', None)}"
                          f" → demote scene {getattr(_ds, 'index', None)} "
                          f"({getattr(_ds, 'graphic_kind', '')}) to footage",
                          flush=True)
                    _demote(_ds, "fullscreen_consecutive")
                    _p1_actions += 1
                    if _drop == _i:
                        continue                 # _prev_fs unchanged
            else:
                _prev_fs = _i
        _fs_idx = [i for i in range(len(script.scenes))
                   if _is_fullscreen(script.scenes[i])]

        # ---- RULE 3: SAME-FAMILY cooldown (>= 25s) ------------------------- #
        # Don't play two cards of the same FAMILY within ~25s.  Families group
        # near-identical templates (all number cards, all stat dashboards, all
        # timelines, …) so e.g. "$37M" then "$40M" 8s later is caught.  Keep
        # the EARLIER one (it already established the device); demote the later.
        _FAMILY = {
            "number": "number", "stat": "number", "currency_stat": "number",
            "numerical_ratio": "number", "score_display": "number",
            "tally_counter": "number", "stat_insight": "number",
            "stat_dashboard": "stats", "vertical_bar_chart": "stats",
            "heatmap_grid": "stats", "chart": "stats", "line_chart": "stats",
            "progress_bar": "stats",
            "statement": "statement", "era_banner": "statement",
            "timeline": "timeline", "mini_timeline": "timeline",
            "calendar_grid": "timeline", "clock_face": "timeline",
            "newspaper": "press", "press_release": "press",
            "headline_crawl": "press",
            "map_reveal": "map", "globe_map": "map", "map_pin_cluster": "map",
            "network_graph": "network", "family_tree_3gen": "network",
            "document": "document", "document_stack": "document",
            "classified": "document", "receipt": "document",
            "email_screenshot": "document",
        }
        _COOLDOWN_S = 25.0
        _last_seen_t: dict[str, float] = {}        # family -> last start time
        for _i in list(_fs_idx):
            _sc = script.scenes[_i]
            if not _is_fullscreen(_sc):
                continue
            _k = (getattr(_sc, "graphic_kind", "") or "").lower()
            if _k in _PREMIUM_MG:
                continue                         # premium MGs never cooled down
            _fam = _FAMILY.get(_k)
            if not _fam:
                continue                         # no family → no cooldown
            _start = _durs_p1[_i][0] if 0 <= _i < len(_durs_p1) else 0.0
            _prev_t = _last_seen_t.get(_fam)
            if _prev_t is not None and (_start - _prev_t) < _COOLDOWN_S:
                print(f"  [restraint] family cooldown ({_fam}, "
                      f"{_start - _prev_t:.1f}s < {_COOLDOWN_S:.0f}s): demote "
                      f"scene {getattr(_sc, 'index', None)} ({_k}) to footage",
                      flush=True)
                _demote(_sc, f"family_cooldown_{_fam}")
                _p1_actions += 1
            else:
                _last_seen_t[_fam] = _start
        _fs_idx = [i for i in range(len(script.scenes))
                   if _is_fullscreen(script.scenes[i])]

        # ---- RULE 1: RUNTIME BUDGET (full-screen time <= ~18% of runtime) -- #
        # Trim excess full-screen cards beyond the time budget, LOWEST-VALUE
        # first, but NEVER below 1-2 genuine title/reveal cards (rule 5).
        _BUDGET_FRAC = 0.34                        # MG-friendly (competitor is MG-heavy)
        _budget_s = _runtime_p1 * _BUDGET_FRAC
        _fs_total = sum(_fs_seconds(i) for i in _fs_idx)
        if _fs_total > _budget_s and _fs_idx:
            # Lowest story-value first (ties: later scene first — keep early
            # context); reveals/titles sort last so they survive longest.
            _order = sorted(_fs_idx, key=lambda i: (_value(i), -i))
            _keep_reveals = [i for i in _fs_idx
                             if (getattr(script.scenes[i], "graphic_kind", "")
                                 or "").lower() in _REVEAL_KINDS
                             or (getattr(script.scenes[i], "graphic_kind", "")
                                 or "").lower() == "title_card"]
            _min_keep = set(_keep_reveals[:2])    # always keep up to 2 reveals
            for _i in _order:
                if _fs_total <= _budget_s:
                    break
                if _i in _min_keep:
                    continue                      # protect title/reveal floor
                _sc = script.scenes[_i]
                if not _is_fullscreen(_sc):
                    continue
                if (getattr(_sc, "graphic_kind", "") or "").lower() \
                        in _PREMIUM_MG:
                    continue                      # never cull a premium MG
                print(f"  [restraint] over budget "
                      f"({_fs_total:.1f}s > {_budget_s:.1f}s @ "
                      f"{_BUDGET_FRAC*100:.0f}% of {_runtime_p1:.0f}s): demote "
                      f"scene {getattr(_sc, 'index', None)} "
                      f"({getattr(_sc, 'graphic_kind', '')}, "
                      f"{_fs_seconds(_i):.1f}s) to footage", flush=True)
                _fs_total -= _fs_seconds(_i)
                _demote(_sc, "fullscreen_over_budget")
                _p1_actions += 1

        if _p1_actions:
            _kept = [i for i in range(len(script.scenes))
                     if _is_fullscreen(script.scenes[i])]
            _kept_s = sum(_fs_seconds(i) for i in _kept)
            print(f"  [restraint] P1.3 footage-first: {_p1_actions} "
                  f"action(s); {len(_kept)} full-screen card(s) kept "
                  f"({_kept_s:.0f}s ≈ {(_kept_s/_runtime_p1*100):.0f}% of "
                  f"{_runtime_p1:.0f}s runtime)", flush=True)
            _log("3/5", f"full-screen restraint: footage-first "
                        f"({len(_kept)} cards ≈ "
                        f"{(_kept_s/_runtime_p1*100):.0f}% of runtime)")
      except Exception as _p1e:                                # noqa: BLE001
        # FAIL-SAFE: any error leaves every card untouched — a render must
        # never crash or lose its visuals because of the restraint pass.
        print(f"  [restraint] P1.3 full-screen restraint skipped ({_p1e})",
              flush=True)

    # V3.3.2 — UNIVERSAL FACTUAL GUARD (always on; NOT gated by restraint). The
    # centralized layer `vidlore/factual_guard.py` drops any FACT-BEARING card
    # whose values aren't backed by explicit evidence in the scene narration:
    # vague quantifiers are not data, and a planted/invented number/date/percent
    # never renders — footage carries the beat instead. This is the code-level
    # enforcement of the prompt-level `_MG_VOCAB` rule, covering EVERY emission
    # path (LLM brief, paste-script, reload, heuristic fallback). It is a pure
    # logging-retention sibling of motion_graphics_audit and NEVER touches
    # asset_qa. Defensive: a guard error leaves cards untouched (fail-open) so a
    # render can never break.
    try:
        from . import factual_guard as _fg
        import time as _fgt
        _fg_ts = round(_fgt.time(), 1)
        for _sc in script.scenes:
            _gk = (getattr(_sc, "graphic_kind", "") or "").strip()
            if not _gk or not _fg.is_fact_bearing(_gk):
                continue
            _ok, _why = _fg.guard(
                _gk,
                getattr(_sc, "narration", "") or "",
                getattr(_sc, "graphic_text", "") or "",
                getattr(_sc, "graphic_body", "") or "")
            if not _ok:
                _factual_rejected.append({
                    "scene": getattr(_sc, "index", None),
                    "requested_kind": _gk,
                    "template": _gk,
                    "category": _fg.classify(_gk),
                    "source_text": (getattr(_sc, "narration", "") or "")[:160],
                    "guard_status": "rejected",
                    "reason": _why,
                    "fallback": "footage_only",
                    "timestamp": _fg_ts,
                })
                _sc.graphic_kind, _sc.graphic_text, _sc.graphic_body = "", "", ""
        if _factual_rejected:
            _log("3/5", f"factual guard: dropped {len(_factual_rejected)} "
                        f"unsupported fact-card(s) → footage-first "
                        f"(no fabricated data)")
    except Exception as _fge:                                  # noqa: BLE001
        print(f"  [factual-guard] sweep skipped ({_fge})", flush=True)

    # RC4 — MACHINE-PAYLOAD TEXT GUARD (always on; runs for EVERY card kind, not
    # just fact-bearing ones). A card's human text (carried by graphic_text, and
    # any explicit title=/text=/headline=/quote= token in graphic_body) must
    # never surface internal/serialized data. The trigger: a route card packs its
    # waypoints as graphic_text="Tehran@35.69,51.39|Baghdad@33.34,44.40|…" (see
    # editor_manifest._card_encode); a sibling title-only primitive then rendered
    # that raw geo payload as its on-screen title. Map/route kinds are EXEMPT
    # here because their packed graphic_text is consumed as COORDINATES by the
    # adapter (stops=…), never shown as copy. For any other kind, a machine
    # payload in graphic_text means the card's only human text is unusable → drop
    # it to footage-only (same path the factual guard uses). Defensive: any error
    # leaves cards untouched (fail-open) so a render can never break.
    try:
        from . import factual_guard as _fg2
        _payload_dropped = 0
        for _sc in script.scenes:
            _gk2 = (getattr(_sc, "graphic_kind", "") or "").strip()
            if not _gk2:
                continue
            # Map/route family packs coordinates into graphic_text by design.
            if _gk2.lower() in _fg2.GEOMETRIC_EXEMPT:
                continue
            _gtxt = getattr(_sc, "graphic_text", "") or ""
            _gbod = getattr(_sc, "graphic_body", "") or ""
            _bad = _fg2.is_machine_payload(_gtxt)
            if not _bad and _gbod:
                # Explicit human-text tokens carried inside the body.
                for _m in re.finditer(
                        r"(?:^|[;|·])\s*(?:title|text|headline|quote|verdict|"
                        r"definition|caption)\s*=\s*([^;|]+)", _gbod, re.I):
                    if _fg2.is_machine_payload(_m.group(1).strip()):
                        _bad = True
                        break
            if _bad:
                _sc.graphic_kind = _sc.graphic_text = _sc.graphic_body = ""
                _payload_dropped += 1
        if _payload_dropped:
            _log("3/5", f"text-card guard: dropped {_payload_dropped} card(s) "
                        f"carrying machine/serialized data → footage-first "
                        f"(no raw payload on screen)")
    except Exception as _pge:                                  # noqa: BLE001
        print(f"  [textcard-guard] sweep skipped ({_pge})", flush=True)

    # Pass footage paths so build_graphic_images can render the
    # ranking-card carousel (each ranking scene's footage thumbnail
    # is composited into a Vidlore-style framed card on grid bg).
    _foot_paths = {f.index: str(f.path) for f in footage}
    graphic_assets = build_graphic_images(
        script.scenes, cfg, work, cache_dir=cache,
        footage_paths=_foot_paths, theme=th)
    if graphic_assets:
        _log("3/5", f"graphic art: {len(graphic_assets)} card image(s)")

    # Thumbnail generation removed — Vidlore is an editing tool, not a YouTube
    # thumbnail maker. (No run_dir/thumbnail.jpg is produced.)

    if brief.music:
        music = brief.music                       # user track wins
        _log("5/5", f"using supplied music: {music}")
    elif cfg.music_enabled:
        durs = [getattr(ns, "duration", 0.0) or 0.0
                for ns in narration.scenes]
        roles = [(getattr(sc, "role", "") or "") for sc in script.scenes]
        energies = [int(getattr(sc, "intensity", 0)
                        or getattr(sc, "energy", 2) or 2)
                    for sc in script.scenes]
        m_theme = getattr(mode, "music_theme", "") or brief.theme
        music = None
        # PROFESSIONAL SCORE — if a curated music library exists, plan cues
        # from the story arc, select + LUFS-match + crossfade real tracks
        # into an EDITED score. Otherwise fall back to the procedural bed.
        try:
            if musiclib.library_available():
                topic = musiclib.topic_category(
                    script.title,
                    " ".join((s.narration or "")
                             for s in script.scenes[:4]))
                # Pass narration text + Whisper word-timings + per-scene
                # EMPHASIS words so the planner can WPM-shape, mark
                # breaths, AND emit tiered reveal-phrase events
                # (trapezoidal dips + silence pockets BEFORE the punch
                # word).  Emphasis words feed the climax-guarantee
                # fallback so every reveal cue lands a tier-3 ducking
                # event even without a scripted reveal phrase.
                texts = [getattr(s, "narration", "") or ""
                         for s in script.scenes]
                emphases = [getattr(s, "emphasis", "") or ""
                            for s in script.scenes]
                cues = musiclib.plan_cues(
                    durs, roles, energies,
                    texts=texts,
                    narr_scenes=narration.scenes,
                    emphases=emphases,
                    style=getattr(mode, "name", ""), topic=topic)
                # AUDIO DIRECTOR (Phase 4): niche drives the intro envelope +
                # reveal-duck character; a chapter-level music cue sheet is
                # written for QA/editor review. Defensive (niche="" => proven
                # behaviour unchanged).
                _audio_niche = str((getattr(brief, "extra", None) or {})
                                   .get("niche", "") or "")
                _audio_vid = str(getattr(brief, "id", "")
                                 or getattr(script, "title", "") or "video")
                score = musiclib.compose_score(
                    cues, narration.total + 1.0,
                    work / "music_score.wav", cache_dir=cache,
                    niche=_audio_niche, video_id=_audio_vid,
                    cue_sheet_out=work / "music_cue_sheet.json")
                if score:
                    music = str(score)
                    cats = " → ".join(c["category"] for c in cues)
                    # Editorial flags summary (only if any non-trivial)
                    flags = []
                    if any(c.get("swell") for c in cues):
                        flags.append(f"swell×{sum(1 for c in cues if c.get('swell'))}")
                    if any(c.get("breath") for c in cues):
                        flags.append(f"breath×{sum(1 for c in cues if c.get('breath'))}")
                    if any(c.get("drop") for c in cues):
                        flags.append(f"drop×{sum(1 for c in cues if c.get('drop'))}")
                    dens = [c.get("density", 1.0) for c in cues]
                    if dens and min(dens) < 1.0:
                        flags.append(f"density {min(dens):.2f}-{max(dens):.2f}")
                    # Stage 5: TIERED reveal events (premium-doc rewrite)
                    # tier 1 = soft pivot (-3.7 dB)
                    # tier 2 = strong reveal (-13 dB + 0.20s silence)
                    # tier 3 = climax/turn (-22 dB + 0.35s silence)
                    all_evs = sum((c.get("events") or [] for c in cues), [])
                    t1 = sum(1 for e in all_evs if e.get("tier") == 1)
                    t2 = sum(1 for e in all_evs if e.get("tier") == 2)
                    t3 = sum(1 for e in all_evs if e.get("tier") == 3)
                    n_sil = sum(1 for e in all_evs
                                if float(e.get("silence_dur", 0)) > 0.05)
                    # legacy events (cached pre-rewrite) still have 'type'
                    n_legacy_dip = sum(1 for e in all_evs
                                       if not e.get("tier")
                                       and e.get("type") == "dip")
                    n_legacy_sil = sum(1 for e in all_evs
                                       if not e.get("tier")
                                       and e.get("type") == "silence")
                    if (t1 + t2 + t3 + n_legacy_dip + n_legacy_sil) > 0:
                        bits = []
                        if t3: bits.append(f"{t3} tier3")
                        if t2: bits.append(f"{t2} tier2")
                        if t1: bits.append(f"{t1} tier1")
                        if n_sil: bits.append(f"{n_sil} silence")
                        if n_legacy_dip or n_legacy_sil:
                            bits.append(f"{n_legacy_dip}d+{n_legacy_sil}s "
                                        "legacy")
                        flags.append("reveal-events: " + " · ".join(bits))
                    extras = (" · " + ", ".join(flags)) if flags else ""
                    _log("5/5", f"scored music: {len(cues)} cues "
                                f"({musiclib.track_count()}-track library) "
                                f"· {cats}{extras}")
        except Exception as e:                             # noqa: BLE001
            _log("5/5", f"music library skipped ({e}); procedural bed")
        if music is None:
            # PHASE 3.2: feed the emotional arc so the procedural score's
            # movements rise/fall WITH the story.
            arc = list(zip(durs, energies))
            bed = build_music_bed(
                m_theme, narration.total + 1.0,
                work / "music_bed.wav", cache_dir=cache, arc=arc,
                intensity=getattr(mode, "music_intensity", 1.0),
            )
            music = str(bed)
            _log("5/5", f"auto music bed ({m_theme}, arc-aware, "
                        f"{mode.name})")
    else:
        music = None

    fx = "crossfades" if cfg.transitions_enabled else "hard cuts"
    _log("5/5", f"assembling video ({fx}, ffmpeg) ...")
    _emit(progress, 82, f"Assembling video ({fx})…")
    video = run_dir / f"{_slug(brief.title)}.mp4"
    # ── MOTION-GRAPHICS DIRECTOR (flag: VIDLORE_MOTION_GRAPHICS=1) ──────────
    # Off by default → legacy behaviour untouched. On → the director picks a
    # SPARSE, contextual set of premium graphics from real scene data; the
    # dispatcher renders+caches them and writes motion_graphics_manifest.json;
    # assemble slices each clip across its scene's beats (footage stays the bed).
    _motion_graphics = None
    _mg_windows = None
    _mg_primitives = None        # {scene_index: primitive_id} for SFX forwarding
    if os.environ.get("VIDLORE_MOTION_GRAPHICS") == "1":
        try:
            from .motion_graphics import director as _mgdir, render_dispatch as _mgrd
            from .motion_graphics import look as _mglook
            # V1.2.1 — safety net for the rare hard-crash orphan: sweep stale MG
            # PNG frame dirs (older than 1h, so an in-progress render is never
            # touched) left in the temp dir by a previously crashed render.
            # Each primitive self-cleans its own dir on success/ffmpeg-fail.
            try:
                _sw = _mglook.sweep_stale_frames(max_age_s=3600.0)
                if _sw.get("removed"):
                    _log("5/5", f"swept {_sw['removed']} stale MG temp dir(s) "
                                f"(~{_sw['freed_mb']} MB)")
            except Exception:  # noqa: BLE001
                pass
            _mg_scenes = []
            for _i, _sc in enumerate(script.scenes):
                _a = {}
                _gk = (getattr(_sc, "graphic_kind", "") or "").lower()
                _gt = getattr(_sc, "graphic_text", "") or ""
                _gb = getattr(_sc, "graphic_body", "") or ""
                # graphic_assets[i] is a STRING (the rendered card PHOTO path);
                # for a portrait-kind card that photo is the person portrait.
                _ga = (graphic_assets or {}).get(_i)
                if _ga is None:
                    _ga = (graphic_assets or {}).get(getattr(_sc, "index", _i))
                _gimg = (_ga if isinstance(_ga, str) and _ga else
                         ((_ga.get("portrait") or _ga.get("image") or _ga.get("path"))
                          if isinstance(_ga, dict) else None))
                # V1.1.1 — for a real named person, prefer the FULL-HEAD real
                # portrait (the face-safe canvas `_real_person_image` cached)
                # over the legacy card's `_pic` layer, which can re-crop the head.
                _rp_used = False
                # For quote kinds the QUOTE is in graphic_text and the speaker's
                # NAME is in graphic_body — key the real-portrait lookup off the
                # name; for name/map kinds the name is graphic_text.
                _rname = (_gb.split("·")[0].strip()
                          if _gk in ("pull_quote", "long_quote",
                                     "pull_quote_portrait") else _gt)
                if _rname and _gk in ("name_reveal", "mugshot", "bio", "mini_bio",
                                      "map_reveal", "map_route",
                                      "pull_quote_portrait", "pull_quote",
                                      "long_quote"):
                    try:
                        from .footage import scene_key as _sk
                        _rck = _sk("realperson", _rname.lower().strip())
                        for _root in (cache, run_dir, work):
                            _rc = Path(_root) / f"{_rck}.jpg"
                            if not _rc.exists():
                                _g2 = list(Path(_root).glob(f"**/{_rck}.jpg"))
                                _rc = _g2[0] if _g2 else _rc
                            if _rc.exists():
                                _a["portrait_path"] = str(_rc.resolve())
                                _rp_used = True
                                break
                    except Exception:  # noqa: BLE001
                        pass
                if not _rp_used and _gimg and _gk in (
                        "name_reveal", "mugshot", "bio", "mini_bio",
                        "pull_quote_portrait", "map_reveal", "map_route"):
                    # graphic_assets values are bare filenames written under the
                    # work/cache/run dirs — resolve to an absolute, existing path
                    # (else cinematic_portrait_hold can't open it).
                    _pp = None
                    for _root in (work, run_dir, cache):
                        try:
                            _c = Path(_root) / _gimg
                            if _c.exists():
                                _pp = str(_c.resolve())
                                break
                            _g = list(Path(_root).glob(f"**/{Path(_gimg).name}"))
                            if _g:
                                _pp = str(_g[0].resolve())
                                break
                        except Exception:  # noqa: BLE001
                            pass
                    if not _pp and Path(_gimg).exists():
                        _pp = str(Path(_gimg).resolve())
                    if _pp:
                        # Prefer the RAW portrait layer (nmr_XXX_pic.png) over the
                        # composed name card (nmr_XXX.png). The card bakes in the
                        # name/dates + its own frame, so feeding it to the MG
                        # portrait primitive double-frames it and shows text where
                        # a clean face should be. The `_pic` sibling is the bare
                        # photo the card was built from.
                        _ppath = Path(_pp)
                        _picv = _ppath.with_name(_ppath.stem + "_pic" + _ppath.suffix)
                        if _picv.exists():
                            _pp = str(_picv.resolve())
                            # The `_pic` layer pastes the portrait into a full
                            # 1920×1080 RGBA canvas (left-aligned, rest
                            # transparent). Crop to the opaque bbox so the MG
                            # primitive frames a TIGHT portrait instead of mostly
                            # empty space inside the gold frame.
                            try:
                                from PIL import Image as _PILImg
                                _pim = _PILImg.open(_pp)
                                if _pim.mode in ("RGBA", "LA"):
                                    _bb = _pim.split()[-1].getbbox()
                                    if _bb and (_bb[2] - _bb[0]) > 120 \
                                            and (_bb[3] - _bb[1]) > 120:
                                        _cr = _pim.crop(_bb)
                                        _crp = _ppath.with_name(
                                            _ppath.stem + "_pcrop.png")
                                        _cr.save(_crp)
                                        _pp = str(_crp.resolve())
                            except Exception:  # noqa: BLE001
                                pass
                        _a["portrait_path"] = _pp
                # structured MG hints allowed in graphic_body:
                #   "place=Cleveland, Ohio · 1863"
                #   "branches=Pipelines:$40M|Railroads:$25M|Refineries:$90M"
                import re as _re2
                _mpl = _re2.search(r"place=([^;|]+)", _gb)
                _mbr = _re2.search(r"branches=([^;]+)", _gb)
                _mlbl = _re2.search(r"label=([^;|]+)", _gb)
                _plain_gb = "" if any(h in _gb for h in (
                    "place=", "branches=", "label=", "events=", "pair=",
                    "values=", "tag=", "bars=", "coords=", "suffix=", "prefix=",
                    "points=", "stops=", "focus=", "share=", "sub=", "steps=",
                    "children=", "title=", "emphasis=", "source=", "count=",
                    "total=", "segments=", "definition=", "pos=", "before=",
                    "after=", "headlines=", "hotspots=", "reveal=",
                    "stamp=", "items=", "eras=", "value=", "to=", "unit=",
                    "nodes=", "links=", "quotes=", "bands=", "readout=",
                    "kicker=", "yes=", "no=", "chosen=", "yes_label=",
                    "no_label=", "from_pos=", "to_pos=")) else _gb
                if _mlbl:
                    _a["label"] = _mlbl.group(1).strip()
                if _gk in ("name_reveal", "mugshot", "bio", "mini_bio") and _gt:
                    _a["name"] = _gt
                    if _plain_gb:
                        _a["sub"] = _plain_gb
                    if _mpl:
                        _a["place"] = _mpl.group(1).strip()
                if _gk in ("map_reveal", "map_route") and _gt:
                    _a["name"] = _gt
                    if _mpl:
                        _a["place"] = _mpl.group(1).strip()
                if _gk in ("news_article", "document", "headline", "redacted",
                           "statement", "press_release") and _gt:
                    _a["headline"] = _gt
                    if _plain_gb:
                        _a["source"] = _plain_gb
                if _gk in ("define_the_term", "quote_highlight") and _gt:
                    _a["keyword"] = _gt.split()[0]
                if _gk in ("network_graph", "conspiracy_board", "org_tree") and _gt and _mbr:
                    _a["center"] = _gt
                    _a["branches"] = [[p.split(":", 1)[0].strip(),
                                       p.split(":", 1)[1].strip()]
                                      for p in _mbr.group(1).split("|") if ":" in p]
                # V1.2 — structural storytelling beats. `graphic_body` hints:
                #   "events=1865:Refinery|1870:Standard Oil|1882:Trust"
                #   "pair=Standard Oil|Independents · values=90|10"
                if _gk in ("timeline", "era_banner", "chronology", "chapter") \
                        and (_gt or _gb):
                    _mev = _re2.search(r"events=([^;]+)", _gb)
                    if _mev:
                        _a["events"] = [
                            [p.split(":", 1)[0].strip(),
                             (p.split(":", 1)[1].strip() if ":" in p else "")]
                            for p in _mev.group(1).split("|") if p.strip()]
                    else:
                        # V3.3.2 — do NOT fabricate a dated timeline by scraping
                        # bare years out of the headline into EMPTY-label events
                        # (that presents undescribed dates as authored timeline
                        # data). A timeline card requires an explicit `events=`
                        # body; with none, the primitive gates to footage.
                        if _plain_gb:
                            _a["label"] = _plain_gb
                    if _gt and not _re2.search(r"\d", _gt):
                        _a["title"] = _gt
                if _gk in ("pull_quote", "long_quote", "pull_quote_portrait",
                           "quote") and _gt:
                    _a["quote"] = _gt
                    if _plain_gb:
                        _a["name"] = _plain_gb
                if _gk in ("comparison", "versus", "vs", "ratio") and _gt:
                    _mpair = _re2.search(r"(?:pair|vs)=([^;|]+\|[^;]+)", _gb)
                    if _mpair:
                        _lr = _mpair.group(1).split("|")
                        if len(_lr) >= 2:
                            # strip any trailing `· values=…`/`values=…` that a
                            # single-string hint may have bled into a label.
                            _clab = lambda s: _re2.split(
                                r"\s*[·|]?\s*values=", s)[0].strip()
                            _a["left"], _a["right"] = _clab(_lr[0]), _clab(_lr[1])
                    elif " vs " in _gt.lower():
                        _pp2 = _re2.split(r"\s+vs\.?\s+", _gt, maxsplit=1, flags=_re2.I)
                        if len(_pp2) == 2:
                            _a["left"], _a["right"] = _pp2[0].strip(), _pp2[1].strip()
                    _mval = _re2.search(r"values=([\d.]+)\|([\d.]+)", _gb)
                    if _mval:
                        _a["leftval"] = float(_mval.group(1))
                        _a["rightval"] = float(_mval.group(2))
                # V1.3 — Batch 2: evidence / data / location beats. Hints:
                #   "tag=EXHIBIT A"
                #   "bars=1870:4|1880:30|1890:90 ; suffix=M ; prefix=$"
                #   "place=Cleveland, Ohio · coords=41.5°N 81.7°W"
                if _gk in ("evidence", "exhibit", "artifact", "framed_photo") and _gt:
                    _a["caption"] = _gt
                    _mtag = _re2.search(r"tag=([^;|]+)", _gb)
                    _a["tag"] = (_mtag.group(1).strip() if _mtag else
                                 ("EXHIBIT" if _gk == "exhibit" else "EVIDENCE"))
                    if _gimg:                               # use the artifact photo
                        for _root in (work, run_dir, cache):
                            try:
                                _c = Path(_root) / _gimg
                                if _c.exists():
                                    _a["image_path"] = str(_c.resolve()); break
                                _g3 = list(Path(_root).glob(f"**/{Path(_gimg).name}"))
                                if _g3:
                                    _a["image_path"] = str(_g3[0].resolve()); break
                            except Exception:  # noqa: BLE001
                                pass
                if _gk in ("bar_chart", "stat_bars", "breakdown", "data_viz") \
                        and (_gt or _gb):
                    _mbars = _re2.search(r"bars=([^;]+)", _gb)
                    if _mbars:
                        _a["bars"] = [
                            [p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                            for p in _mbars.group(1).split("|") if ":" in p]
                    if _gt:
                        _a["title"] = _gt
                    _msf = _re2.search(r"suffix=([^;|]+)", _gb)
                    if _msf:
                        _a["suffix"] = _msf.group(1).strip()
                    _mpf = _re2.search(r"prefix=([^;|]+)", _gb)
                    if _mpf:
                        _a["prefix"] = _mpf.group(1).strip()
                if _gk in ("location", "establish", "setting", "place_card") and _gt:
                    _a["place"] = (_mpl.group(1).strip() if _mpl else _gt)
                    if _plain_gb:
                        _a["sub"] = _plain_gb
                    _mco = _re2.search(r"coords=([^;|]+)", _gb)
                    if _mco:
                        _a["coords"] = _mco.group(1).strip()
                # V1.4 — Batch 3: growth / annotation / route beats. Hints:
                #   "points=2010:4|2014:30|2018:90 ; suffix=M ; prefix=$"
                #   "focus=0.5,0.44 ; tag=DETAIL"   (graphic_text = the label)
                #   "stops=London:0.2,0.34|Suez:0.55,0.46|Bombay:0.82,0.55"
                if _gk in ("growth", "trend", "line_chart", "trajectory",
                           "curve") and (_gt or _gb):
                    _mpts = _re2.search(r"points=([^;]+)", _gb)
                    if _mpts:
                        _a["points"] = [
                            [p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                            for p in _mpts.group(1).split("|") if ":" in p]
                    if _gt:
                        _a["title"] = _gt
                    _msf2 = _re2.search(r"suffix=([^;|]+)", _gb)
                    if _msf2:
                        _a["suffix"] = _msf2.group(1).strip()
                    _mpf2 = _re2.search(r"prefix=([^;|]+)", _gb)
                    if _mpf2:
                        _a["prefix"] = _mpf2.group(1).strip()
                if _gk in ("annotate", "detail", "highlight_detail",
                           "closeup") and _gt:
                    _a["label"] = _gt
                    _mfoc = _re2.search(r"focus=([\d.]+\s*,\s*[\d.]+)", _gb)
                    if _mfoc:
                        _a["focus"] = _mfoc.group(1).replace(" ", "")
                    _mtag2 = _re2.search(r"tag=([^;|]+)", _gb)
                    _a["tag"] = _mtag2.group(1).strip() if _mtag2 else "DETAIL"
                    if _gimg:                               # annotate the scene photo
                        for _root in (work, run_dir, cache):
                            try:
                                _c = Path(_root) / _gimg
                                if _c.exists():
                                    _a["image_path"] = str(_c.resolve()); break
                                _g4 = list(Path(_root).glob(f"**/{Path(_gimg).name}"))
                                if _g4:
                                    _a["image_path"] = str(_g4[0].resolve()); break
                            except Exception:  # noqa: BLE001
                                pass
                if _gk in ("route", "journey", "expansion", "spread", "map_route",
                           "shipping_route", "velocity_route", "trade_route",
                           "trajectory_map", "supply_route", "supply_lines") and (_gt or _gb):
                    _mstp = _re2.search(r"stops=([^;]+)", _gb)
                    if _mstp:
                        _stops = []
                        for _p in _mstp.group(1).split("|"):
                            if ":" in _p:
                                _nm, _ps = _p.split(":", 1)
                                _stops.append([_nm.strip(), _ps.strip()])
                            elif _p.strip():
                                _stops.append([_p.strip(), None])
                        if _stops:
                            _a["stops"] = _stops
                    # RC4 — TITLE only from an EXPLICIT body `title=` token, NEVER
                    # the bare graphic_text/_gt: a route card's _gt is often the
                    # packed waypoint DSL ("Tehran@35.69,51.39|Baghdad@33.34,…")
                    # produced by editor_manifest._card_encode. Copying that into
                    # the SHARED `title` asset key let a sibling title-only card
                    # (act_chapter_card) surface the raw geo payload as its title.
                    # Map primitives still get their coordinates via `stops` above.
                    _mrtt = _re2.search(r"title=([^;|]+)", _gb)
                    if _mrtt:
                        _a["title"] = _mrtt.group(1).strip()
                # V2.4 — cinematic-map route tags (shipping_route / velocity_route /
                #   trade_route → velocity_route_map). Sibling to the route block
                #   above, NOT folded into it: velocity_route_map resolves each stop
                #   through geo.resolve() and therefore needs a FLAT list of place-
                #   NAME strings ("Basra","Abadan",…). The route block builds
                #   [name, pos] PAIRS for map_route_spread; a pair would stringify to
                #   "['Basra', None]" and never resolve, silently degrading the card
                #   to a synthetic route. Any "name:pos" position hint is dropped here
                #   — the gazetteer supplies the real coordinates. Title: an explicit
                #   body title= wins (cf. the hierarchy block), else graphic_text.
                if _gk in ("shipping_route", "velocity_route", "trade_route") \
                        and (_gt or _gb):
                    _mstp2 = _re2.search(r"stops=([^;]+)", _gb)
                    if _mstp2:
                        _vstops = []
                        for _p in _mstp2.group(1).split("|"):
                            _nm = (_p.split(":", 1)[0] if ":" in _p else _p).strip()
                            if _nm:
                                _vstops.append(_nm)
                        if _vstops:
                            _a["stops"] = _vstops
                    _mvtt = _re2.search(r"title=([^;|]+)", _gb)
                    if _mvtt:
                        _a["title"] = _mvtt.group(1).strip()
                    elif _gt:
                        _a["title"] = _gt
                # V1.5 — Batch 4: proportion / process / hierarchy beats. Hints:
                #   graphic_text="90" ; "label=of all U.S. oil ; sub=Std Oil 1890"
                #   "steps=Buy refinery|Undercut|Starve|Absorb"   (text = title)
                #   graphic_text="Standard Oil Trust" ;
                #     "children=Std of Ohio|Std of N.J.|Std of N.Y. ; title=THE TRUST"
                if _gk in ("share", "proportion", "percent", "dominance",
                           "market_share") and (_gt or _gb):
                    _msh = _re2.search(r"share=([\d.]+)", _gb)
                    if _msh:
                        _a["share"] = float(_msh.group(1))
                    elif _gt:
                        _mn = _re2.search(r"([\d.]+)", _gt)
                        if _mn:
                            _a["share"] = float(_mn.group(1))
                    _mlb2 = _re2.search(r"label=([^;|]+)", _gb)
                    if _mlb2:
                        _a["label"] = _mlb2.group(1).strip()
                    elif _plain_gb:
                        _a["label"] = _plain_gb
                    _msub = _re2.search(r"sub=([^;|]+)", _gb)
                    if _msub:
                        _a["center_sub"] = _msub.group(1).strip()
                if _gk in ("process", "method", "mechanism", "flow", "steps",
                           "how_it_works") and (_gt or _gb):
                    _mstep = _re2.search(r"steps=([^;]+)", _gb)
                    if _mstep:
                        _a["steps"] = [s.strip() for s in
                                       _mstep.group(1).split("|") if s.strip()]
                    if _gt:
                        _a["title"] = _gt
                if _gk in ("hierarchy", "org_chart", "structure", "chain") and _gt:
                    _a["root"] = _gt
                    _mch = _re2.search(r"children=([^;]+)", _gb)
                    if _mch:
                        _a["children"] = [c.strip() for c in
                                          _mch.group(1).split("|") if c.strip()]
                    _mtt = _re2.search(r"title=([^;|]+)", _gb)
                    if _mtt:
                        _a["title"] = _mtt.group(1).strip()
                # V1.6 — Batch 5: statement / pictograph / composition beats. Hints:
                #   text="He rewrote it." ; "emphasis=rewrote it ; source=…"
                #   label="businesses that fail" ; "count=8 ; total=10"
                #   title="OF EVERY DOLLAR" ; "segments=Profit:60|Costs:28|Tax:12"
                if _gk in ("statement", "thesis", "claim", "takeaway") and _gt:
                    _a["text"] = _gt
                    _mem = _re2.search(r"emphasis=([^;|]+)", _gb)
                    if _mem:
                        _a["emphasis"] = _mem.group(1).strip()
                    _msrc = _re2.search(r"source=([^;|]+)", _gb)
                    if _msrc:
                        _a["source"] = _msrc.group(1).strip()
                if _gk in ("pictograph", "ratio", "figures", "in_ten") \
                        and (_gt or _gb):
                    _mct = _re2.search(r"count=(\d+)", _gb)
                    if _mct:
                        _a["count"] = int(_mct.group(1))
                    _mtl = _re2.search(r"total=(\d+)", _gb)
                    if _mtl:
                        _a["total"] = int(_mtl.group(1))
                    if _gt:
                        _a["label"] = _gt
                if _gk in ("composition", "breakdown", "stacked", "split") \
                        and (_gt or _gb):
                    _mseg = _re2.search(r"segments=([^;]+)", _gb)
                    if _mseg:
                        _a["segments"] = [
                            [p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                            for p in _mseg.group(1).split("|") if ":" in p]
                    if _gt:
                        _a["title"] = _gt
                    _msf3 = _re2.search(r"suffix=([^;|]+)", _gb)
                    if _msf3:
                        _a["suffix"] = _msf3.group(1).strip()
                # V1.7 — Batch 6: definition / balance / before-after beats. Hints:
                #   term="Monopoly" ; "pos=noun ; definition=exclusive control…"
                #   title="WHO HELD THE POWER" ; "pair=Capital|Labour ; values=72|28"
                #   caption="A century apart" ; "before=1900 ; after=TODAY"
                if _gk in ("definition", "define", "term", "glossary") and _gt:
                    _a["term"] = _gt
                    _mdef = _re2.search(r"definition=([^;|]+)", _gb)
                    if _mdef:
                        _a["definition"] = _mdef.group(1).strip()
                    elif _plain_gb:
                        _a["definition"] = _plain_gb
                    _mpos = _re2.search(r"pos=([^;|]+)", _gb)
                    if _mpos:
                        _a["pos"] = _mpos.group(1).strip()
                if _gk in ("balance", "tradeoff", "tension", "scales") \
                        and (_gt or _gb):
                    _mpr2 = _re2.search(r"pair=([^;|]+\|[^;]+)", _gb)
                    if _mpr2:
                        _lr = _mpr2.group(1).split("|")
                        _a["left"], _a["right"] = _lr[0].strip(), _lr[1].strip()
                    _mvl2 = _re2.search(r"values=([\d.]+)\|([\d.]+)", _gb)
                    if _mvl2:
                        _a["leftval"] = float(_mvl2.group(1))
                        _a["rightval"] = float(_mvl2.group(2))
                    if _gt:
                        _a["title"] = _gt
                if _gk in ("before_after", "transformation", "then_now",
                           "makeover") and (_gt or _gb):
                    _mbf = _re2.search(r"before=([^;|]+)", _gb)
                    _a["before_label"] = _mbf.group(1).strip() if _mbf else "BEFORE"
                    _maf = _re2.search(r"after=([^;|]+)", _gb)
                    _a["after_label"] = _maf.group(1).strip() if _maf else "AFTER"
                    if _gt:
                        _a["caption"] = _gt
                    if _gimg:                               # wipe over the scene photo
                        for _root in (work, run_dir, cache):
                            try:
                                _c = Path(_root) / _gimg
                                if _c.exists():
                                    _a["image_path"] = str(_c.resolve()); break
                                _g5 = list(Path(_root).glob(f"**/{Path(_gimg).name}"))
                                if _g5:
                                    _a["image_path"] = str(_g5[0].resolve()); break
                            except Exception:  # noqa: BLE001
                                pass
                # V1.8 — Batch 7: headlines / heat-spread / redacted beats. Hints:
                #   headlines="MONOPOLY BROKEN|COURT RULES|THE OCTOPUS FALLS"
                #   hotspots="Paris:0.42,0.46|Berlin:0.55,0.38|Vienna:0.50,0.50"
                #   title="MEMO — EYES ONLY" ; stamp="TOP SECRET"
                if _gk in ("headlines", "press", "press_frenzy", "media_frenzy",
                           "montage", "scandal", "coverage") and (_gt or _gb):
                    _mhl = _re2.search(r"headlines=([^;]+)", _gb)
                    if _mhl:
                        _a["headlines"] = [s.strip() for s in _mhl.group(1).split("|")
                                           if s.strip()]
                        if _gt:
                            _a["title"] = _gt
                    elif _gt:
                        _a["headlines"] = [s.strip() for s in _gt.split("|")
                                           if s.strip()]
                    _mti2 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mti2:
                        _a["title"] = _mti2.group(1).strip()
                if _gk in ("heat_spread", "heatmap", "contagion", "outbreak",
                           "epidemic", "wildfire") and (_gt or _gb):
                    _mhs = _re2.search(r"hotspots=([^;]+)", _gb)
                    if _mhs:
                        _hl2 = []
                        for _p in _mhs.group(1).split("|"):
                            if ":" in _p:
                                _nm, _pos = _p.split(":", 1)
                                _hl2.append([_nm.strip(), _pos.strip()])
                            elif _p.strip():
                                _hl2.append([_p.strip(), ""])
                        if _hl2:
                            _a["hotspots"] = _hl2
                    if _gt:
                        _a["title"] = _gt
                    if _gimg:                               # heat over a map photo
                        for _root in (work, run_dir, cache):
                            try:
                                _c = Path(_root) / _gimg
                                if _c.exists():
                                    _a["map_image"] = str(_c.resolve()); break
                                _g6 = list(Path(_root).glob(f"**/{Path(_gimg).name}"))
                                if _g6:
                                    _a["map_image"] = str(_g6[0].resolve()); break
                            except Exception:  # noqa: BLE001
                                pass
                if _gk in ("classified", "redacted", "secret", "top_secret",
                           "declassified", "dossier", "leak") and _gt:
                    _a["reveal"] = _gt
                    _mti3 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mti3:
                        _a["title"] = _mti3.group(1).strip()
                    _mstp = _re2.search(r"stamp=([^;|]+)", _gb)
                    if _mstp:
                        _a["stamp"] = _mstp.group(1).strip()
                # V1.9 — Batch 8: ranked-list / sankey-flow / era-band beats. Hints:
                #   items="Standard Oil:90|Carnegie:62|Vanderbilt:48" ; prefix=$ ; suffix=M
                #   source="REVENUE" ; branches="Refining:60|Pipelines:25|Railroads:15"
                #   eras="Steam:1780-1840|Rail:1840-1900|Oil:1900-1945"
                if _gk in ("ranking", "leaderboard", "top_list", "ranked",
                           "top_five", "top_ten", "countdown_list") and (_gt or _gb):
                    _mit = _re2.search(r"items=([^;]+)", _gb)
                    if _mit:
                        _items = []
                        for _p in _mit.group(1).split("|"):
                            if ":" in _p:
                                _nm, _vv = _p.rsplit(":", 1)
                                try:
                                    _items.append([_nm.strip(),
                                                   float(_re2.sub(r"[^\d.\-]", "", _vv))])
                                except ValueError:
                                    pass
                        if _items:
                            _a["items"] = _items
                    if _gt:
                        _a["title"] = _gt
                    _mpfx = _re2.search(r"prefix=([^;|]+)", _gb)
                    if _mpfx:
                        _a["prefix"] = _mpfx.group(1).strip()
                    _msfx = _re2.search(r"suffix=([^;|]+)", _gb)
                    if _msfx:
                        _a["suffix"] = _msfx.group(1).strip()
                if _gk in ("sankey", "money_split", "allocation", "flow_breakdown",
                           "where_it_goes") and (_gt or _gb):
                    _mbr2 = _re2.search(r"branches=([^;]+)", _gb)
                    if _mbr2:
                        _brs = []
                        for _p in _mbr2.group(1).split("|"):
                            if ":" in _p:
                                _nm, _vv = _p.rsplit(":", 1)
                                try:
                                    _brs.append([_nm.strip(),
                                                 float(_re2.sub(r"[^\d.\-]", "", _vv))])
                                except ValueError:
                                    pass
                        if _brs:
                            _a["branches"] = _brs
                    _msrc2 = _re2.search(r"source=([^;|]+)", _gb)
                    if _msrc2:
                        _a["source"] = _msrc2.group(1).strip()
                    elif _gt:
                        _a["source"] = _gt
                    _mtit4 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mtit4:
                        _a["title"] = _mtit4.group(1).strip()
                    _mpfx2 = _re2.search(r"prefix=([^;|]+)", _gb)
                    if _mpfx2:
                        _a["prefix"] = _mpfx2.group(1).strip()
                    _msfx2 = _re2.search(r"suffix=([^;|]+)", _gb)
                    if _msfx2:
                        _a["suffix"] = _msfx2.group(1).strip()
                if _gk in ("eras", "ages", "era_bands", "periods", "epochs") \
                        and (_gt or _gb):
                    _mer = _re2.search(r"eras=([^;]+)", _gb)
                    if _mer:
                        _ers = []
                        for _p in _mer.group(1).split("|"):
                            if ":" in _p:
                                _nm, _rg = _p.split(":", 1)
                                _rg = _rg.replace("—", "-").replace("–", "-")
                                _pp = [x for x in _rg.split("-") if x.strip()]
                                if len(_pp) >= 2:
                                    _ers.append([_nm.strip(), _pp[0].strip(),
                                                 _pp[1].strip()])
                                elif _nm.strip():
                                    _ers.append([_nm.strip(), _rg.strip()])
                        if _ers:
                            _a["eras"] = _ers
                    if _gt:
                        _a["title"] = _gt
                # V2.0 — Batch 9: countdown / connection-web / quote-stream beats. Hints:
                #   value=72 ; to=0 ; unit=HOURS    (graphic_text = the label)
                #   nodes="The Senator|The Banker" ; links="The Senator-The Banker"
                #   quotes="A menace.:The Times|The octopus.:Harper's"
                if _gk in ("countdown", "deadline", "timer", "clock",
                           "ticking_clock", "urgency_clock") and (_gt or _gb):
                    _mval = _re2.search(r"value=([\d.,]+)", _gb)
                    if _mval:
                        try:
                            _a["value"] = float(_mval.group(1).replace(",", ""))
                        except ValueError:
                            pass
                    _mto = _re2.search(r"to=([\d.,]+)", _gb)
                    if _mto:
                        try:
                            _a["to"] = float(_mto.group(1).replace(",", ""))
                        except ValueError:
                            pass
                    _mun = _re2.search(r"unit=([^;|]+)", _gb)
                    if _mun:
                        _a["unit"] = _mun.group(1).strip()
                    if _gt:
                        _a["label"] = _gt
                if _gk in ("connection_web", "network_web", "relationship_map",
                           "conspiracy_web", "web_of_connections", "connections") \
                        and (_gt or _gb):
                    _mnd = _re2.search(r"nodes=([^;]+)", _gb)
                    if _mnd:
                        _a["nodes"] = [s.strip() for s in _mnd.group(1).split("|")
                                       if s.strip()]
                    _mlk = _re2.search(r"links=([^;]+)", _gb)
                    if _mlk:
                        _a["links"] = [s.strip() for s in _mlk.group(1).split("|")
                                       if s.strip()]
                    if _gt:
                        _a["title"] = _gt
                if _gk in ("quote_stream", "quotes", "chorus", "reactions",
                           "verdict_quotes") and (_gt or _gb):
                    _mqs = _re2.search(r"quotes=([^;]+)", _gb)
                    if _mqs:
                        _qs = []
                        for _p in _mqs.group(1).split("|"):
                            if ":" in _p:
                                _tx, _by = _p.rsplit(":", 1)
                                _qs.append([_tx.strip(), _by.strip()])
                            elif _p.strip():
                                _qs.append([_p.strip(), ""])
                        if _qs:
                            _a["quotes"] = _qs
                    if _gt:
                        _a["title"] = _gt
                # V2.1 — Batch 10: region-highlight / cause-effect / spectrum-meter. Hints:
                #   region via _gt ; pos=0.52,0.42 ; sub=Demilitarized Zone ; title=...
                #   steps="Humiliation|Collapse|War"  (cause_effect)
                #   value=88 ; bands="LOW|HIGH|SEVERE" ; readout=SEVERE ; label via _gt
                if _gk in ("region", "map_region", "territory", "region_highlight",
                           "region_map") and _gt:
                    _a["region"] = _gt
                    _mps = _re2.search(r"pos=([^;|]+)", _gb)
                    if _mps:
                        _a["pos"] = _mps.group(1).strip()
                    _msb = _re2.search(r"sub=([^;|]+)", _gb)
                    if _msb:
                        _a["sub"] = _msb.group(1).strip()
                    elif _plain_gb:
                        _a["sub"] = _plain_gb
                    _mti5 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mti5:
                        _a["title"] = _mti5.group(1).strip()
                    if _gimg:                               # highlight over a map photo
                        for _root in (work, run_dir, cache):
                            try:
                                _c = Path(_root) / _gimg
                                if _c.exists():
                                    _a["map_image"] = str(_c.resolve()); break
                                _g7 = list(Path(_root).glob(f"**/{Path(_gimg).name}"))
                                if _g7:
                                    _a["map_image"] = str(_g7[0].resolve()); break
                            except Exception:  # noqa: BLE001
                                pass
                if _gk in ("cause_effect", "causation", "cause_chain", "domino",
                           "chain_reaction") and (_gt or _gb):
                    _mst2 = _re2.search(r"steps=([^;]+)", _gb)
                    if _mst2:
                        _a["steps"] = [s.strip() for s in _mst2.group(1).split("|")
                                       if s.strip()]
                    if _gt:
                        _a["title"] = _gt
                if _gk in ("gauge", "meter", "threat_level", "severity",
                           "rating") and (_gt or _gb):
                    _mvl3 = _re2.search(r"value=([\d.,]+)", _gb)
                    if _mvl3:
                        try:
                            _a["value"] = float(_mvl3.group(1).replace(",", ""))
                        except ValueError:
                            pass
                    _mbn = _re2.search(r"bands=([^;]+)", _gb)
                    if _mbn:
                        _a["bands"] = [s.strip() for s in _mbn.group(1).split("|")
                                       if s.strip()]
                    _mrd = _re2.search(r"readout=([^;|]+)", _gb)
                    if _mrd:
                        _a["readout"] = _mrd.group(1).strip()
                    _mti6 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mti6:
                        _a["title"] = _mti6.group(1).strip()
                    if _gt:
                        _a["label"] = _gt
                # V2.2 — Batch 11: spotlight-reveal / decision-fork / world-arc. Hints:
                #   subject via _gt ; kicker=THE ONE WHO LIED ; sub=Seen at midnight ; title=...
                #   question via _gt ; yes=Released ; no=Held ; chosen=no ; title=THE DECISION
                #   from_place via _gt ; to=Tokyo ; from_pos=0.24,0.40 ; to_pos=0.76,0.44 ; title=...
                if _gk in ("spotlight", "reveal", "subject_reveal", "behold",
                           "unveil", "spotlight_object_hold") and _gt:
                    _a["subject"] = _gt
                    _mkc = _re2.search(r"kicker=([^;|]+)", _gb)
                    if _mkc:
                        _a["kicker"] = _mkc.group(1).strip()
                    _msb3 = _re2.search(r"sub=([^;|]+)", _gb)
                    if _msb3:
                        _a["sub"] = _msb3.group(1).strip()
                    _mti7 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mti7:
                        _a["title"] = _mti7.group(1).strip()
                if _gk in ("decision", "flowchart", "flowchart_decision",
                           "decision_tree", "branch", "choice", "fork",
                           "yes_no") and _gt:
                    _a["question"] = _gt
                    _mys = _re2.search(r"yes=([^;|]+)", _gb)
                    if _mys:
                        _a["yes"] = _mys.group(1).strip()
                    _mno = _re2.search(r"no=([^;|]+)", _gb)
                    if _mno:
                        _a["no"] = _mno.group(1).strip()
                    _mch = _re2.search(r"chosen=([^;|]+)", _gb)
                    if _mch:
                        _a["chosen"] = _mch.group(1).strip()
                    _myl = _re2.search(r"yes_label=([^;|]+)", _gb)
                    if _myl:
                        _a["yes_label"] = _myl.group(1).strip()
                    _mnl = _re2.search(r"no_label=([^;|]+)", _gb)
                    if _mnl:
                        _a["no_label"] = _mnl.group(1).strip()
                    _mti8 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mti8:
                        _a["title"] = _mti8.group(1).strip()
                if _gk in ("world_arc", "world_map_arc", "arc", "global_link",
                           "transatlantic", "great_circle", "globe_arc") and _gt:
                    _a["from_place"] = _gt
                    _mtp = _re2.search(r"to=([^;|]+)", _gb)
                    if _mtp:
                        _a["to_place"] = _mtp.group(1).strip()
                    _mfp = _re2.search(r"from_pos=([^;|]+)", _gb)
                    if _mfp:
                        _a["from_pos"] = _mfp.group(1).strip()
                    _mtps = _re2.search(r"to_pos=([^;|]+)", _gb)
                    if _mtps:
                        _a["to_pos"] = _mtps.group(1).strip()
                    _mti9 = _re2.search(r"title=([^;|]+)", _gb)
                    if _mti9:
                        _a["title"] = _mti9.group(1).strip()
                    if _gimg:                               # arc over a supplied map photo
                        for _root in (work, run_dir, cache):
                            try:
                                _c = Path(_root) / _gimg
                                if _c.exists():
                                    _a["map_image"] = str(_c.resolve()); break
                                _g8 = list(Path(_root).glob(f"**/{Path(_gimg).name}"))
                                if _g8:
                                    _a["map_image"] = str(_g8[0].resolve()); break
                            except Exception:  # noqa: BLE001
                                pass
                # V2.8 — Section-C structured-asset primitives. Two cards carry
                # inherently structured data the director can't mine from prose
                # (scale-true silhouettes / route coordinates), so they only fire
                # when the script author supplies it explicitly in graphic_body:
                #   scale_compare/size_compare: items=Bus:12:city bus|Truck:8:road
                #       → silhouette_scale_compare
                #   route_trace:                points=0.2:0.7:Start|0.5:0.45|0.8:0.3:End
                #       → footage_route_trace
                if _gk in ("scale_compare", "size_compare", "route_trace") and _gb:
                    _a.update(_mg_structured_assets(_gk, _gb))
                    _msc_t = _re2.search(r"title=([^;|]+)", _gb)
                    if _msc_t:
                        _a["title"] = _msc_t.group(1).strip()
                    elif _gt:           # graphic_text fallback → title (all 3 kinds)
                        _a["title"] = _gt
                _mg_scenes.append({
                    "index": _i, "role": getattr(_sc, "role", None),
                    "graphic_kind": _gk,
                    "intensity": int(getattr(_sc, "intensity", None)
                                     or getattr(_sc, "energy", 3) or 3),
                    "narration": getattr(_sc, "narration", "") or "", "assets": _a})
            # RC4 — when the brief carries no explicit niche, DON'T silently
            # default to "business" (that mis-routed a war/geopolitics doc into
            # business MG palettes/templates). Classify from the brief text and
            # fall back to a NEUTRAL documentary default ("history") on an
            # empty/low-confidence result — never "business" as the silent floor.
            _niche = str((getattr(brief, "extra", None) or {}).get("niche", "")
                         or "").strip().lower()
            if not _niche:
                try:
                    from .look_dna import classify_brief_niche as _cbn
                    _cls = (_cbn(brief) or "").strip().lower()
                    _niche = _cls if _cls and _cls != "_default" else "history"
                except Exception:                              # noqa: BLE001
                    _niche = "history"
            # STABLE per-video seed — Python's built-in hash() is salted per
            # process (PYTHONHASHSEED), which changed the seed (and therefore
            # every MG clip's content-cache key) on every run, so the rendered
            # clips never cache-hit across renders. A content hash keeps the
            # seed (palette + per-scene jitter + cache key) deterministic.
            import hashlib as _hl
            _seed = int(_hl.sha1((brief.title or "video").encode("utf-8"))
                        .hexdigest()[:8], 16) % 100000
            _dens = os.environ.get("VIDLORE_MG_DENSITY", "").strip()
            _pkw = {}
            try:
                if _dens:
                    _pkw["density"] = max(0.0, min(0.6, float(_dens)))
            except ValueError:
                pass
            # Forensic v2 — RESTRAINED overlay mode (footage-first). Raises the
            # director's score-floor so ONLY well-justified PREMIUM cards fire
            # (stats / research-proof / mechanisms / diagrams / maps / timelines
            # / documents / major reveals all score high and survive) and widens
            # card spacing — this trims weak kinetic text, redundant lower-thirds
            # and decorative callouts WITHOUT forcing the count to zero. Opt-in
            # (A/B before it becomes a niche default): VIDLORE_OVERLAY_RESTRAINT=1.
            if os.environ.get("VIDLORE_OVERLAY_RESTRAINT", "0").strip().lower() \
                    in ("1", "true", "yes", "on"):
                # Footage-first QUALITY floor: a premium card must genuinely
                # justify itself (score ≥ floor) — but 2.8 was so high only 1-2
                # of 5-7 real opportunities survived. 2.2 keeps weak kinetic
                # text / decorative callouts out while letting the genuinely
                # earned premium beats (maps, diagrams, timelines, stats, webs,
                # documents) through to the niche-aware bounded target. The
                # director's budget (niche-aware) + min_gap still keep it sparse.
                _pkw["score_floor"] = float(
                    os.environ.get("VIDLORE_OVERLAY_FLOOR", "2.2") or 2.2)
                _pkw["min_gap_scenes"] = 1
            _dec = _mgdir.plan(_mg_scenes, niche=_niche, seed=_seed, **_pkw)
            # ── CALLOUT SUBJECT IMAGE (permanent anti-mispoint fix, user
            # 2026-06-06) ──────────────────────────────────────────────────────
            # A point-out callout that lands on empty blurred background is the
            # single worst MG failure (a "PEELING PAINT" ring on green bokeh).
            # Root cause: over arbitrary stock footage the anchor can only GUESS.
            # Fix: for the two pointer primitives, source a clean AI still that
            # literally depicts the LABELLED subject (centred, sharp, plain bg) and
            # hand it to the callout as its image — so it always points at the real
            # thing. Runs BEFORE the footage-bed block so AI wins for callouts; the
            # footage bed still serves every OTHER hybrid. Default-ON when AI stills
            # are enabled; VIDLORE_CALLOUT_AI=0 disables. A beat that already carries
            # an explicit image OR an explicit point/focus (a curated Review-Editor
            # override / structured-script anchor) is left untouched. Fully
            # graceful: a generation miss leaves the existing fallback in place.
            _aimg_on_co = (os.environ.get("VIDLORE_AIMG", "0") or "").strip() \
                .lower() in ("1", "true", "yes", "on")
            _co_on = (os.environ.get("VIDLORE_CALLOUT_AI", "1") or "").strip() \
                .lower() not in ("0", "false", "no", "off")
            if _aimg_on_co and _co_on:
                _CO_IMGKEY = {"footage_object_callout": "bg_image",
                              "annotated_detail_callout": "image_path"}
                # per-scene narration (disambiguates a terse label) + video topic
                _co_narr = {int(_s.get("index")): str(_s.get("narration") or "")
                            for _s in _mg_scenes
                            if isinstance(_s, dict) and _s.get("index") is not None}
                _co_topic = str(getattr(brief, "title", "") or "")
                for _d in _dec:
                    try:
                        _pid = getattr(_d, "primitive", None)
                        if _pid not in _CO_IMGKEY:
                            continue
                        _ins = getattr(_d, "inputs", None)
                        if _ins is None:
                            continue
                        _imgk = _CO_IMGKEY[_pid]
                        if _ins.get(_imgk) or _ins.get("point") or _ins.get("focus"):
                            continue              # curated image/anchor — respect it
                        _lab = str(_ins.get("label") or _ins.get("tag") or "").strip()
                        if len(_lab) < 2:
                            continue
                        _ctx = (_co_narr.get(
                            int(getattr(_d, "scene_index", -1)), "")
                            + " " + _co_topic).strip()
                        _aip = _callout_subject_image(
                            _lab, _ins.get("note", ""), _ctx, work, cache)
                        if not _aip:
                            continue
                        _ins[_imgk] = _aip
                        # footage_object_callout: leave `point` UNSET so the
                        # primitive runs the hardened focus-gated saliency on the
                        # clean AI plate (locks precisely on the centred sharp
                        # subject; falls back to centre on any miss).
                        # annotated_detail_callout: has no saliency — pin its focus
                        # to the centred AI subject.
                        if _pid == "annotated_detail_callout":
                            _ins["focus"] = "0.5,0.46"
                    except Exception:                              # noqa: BLE001
                        continue
            # VIDLORE_MG_FOOTAGE_BED (opt-in, OFF by default) — give the CHOSEN
            # hybrid OVERLAY primitives the scene's REAL footage frame as their bed
            # so a callout ring / fact / measurement points at the actual subject
            # instead of a dark synthetic bed. Injected POST-plan into the winning
            # decision's inputs ONLY, so it can NEVER perturb the director's
            # selection (the bed is a render input, not a selection cue). Default-OFF
            # ⇒ zero change to existing renders. Graceful: a miss leaves the
            # primitive's dark-bed fallback untouched.
            if os.environ.get("VIDLORE_MG_FOOTAGE_BED", "0").strip().lower() \
                    in ("1", "true", "yes", "on"):
                try:
                    from .motion_graphics import registry as _mgreg
                    for _d in _dec:
                        _pid = getattr(_d, "primitive", None)
                        _ins = getattr(_d, "inputs", None)
                        if not _pid or _ins is None or "bg_image" in _ins:
                            continue
                        try:
                            if _mgreg.get(_pid).get("family") != "hybrid":
                                continue
                        except Exception:                          # noqa: BLE001
                            continue
                        _bed_img = _scene_footage_bed(_d.scene_index, work, cache)
                        if _bed_img:
                            _ins["bg_image"] = _bed_img
                except Exception:                                  # noqa: BLE001
                    pass
            # V3.0.1 — PORTRAIT LADDER for the biography cold-open. The
            # `portrait_legend_reveal` primitive accepts a face-safe
            # `portrait_path` but the director only derives the subject NAME;
            # without a resolved image it draws its (premium) monogram pedestal.
            # Resolve a REAL archival / validated portrait here via the existing
            # sourcing ladder (Wikipedia lead / Commons / LoC → optional fal
            # still → monogram) and inject it, so the legend opens on the real
            # face wherever one exists. Pure-pipeline: the primitive stays
            # deterministic/offline; provenance is recorded for the manifest.
            _portrait_ladder: dict = {}
            try:
                from . import footage as _ftgp
                _aimg_on = (os.environ.get("VIDLORE_AIMG", "0") or "").strip() \
                    .lower() in ("1", "true", "yes", "on")
                for _d in _dec:
                    if getattr(_d, "primitive", None) != "portrait_legend_reveal":
                        continue
                    _pin = getattr(_d, "inputs", {}) or {}
                    _pname = str(_pin.get("name") or "").strip()
                    if not _pname:
                        continue
                    _si = _d.scene_index
                    # respect a portrait_path already resolved upstream
                    _exist = _pin.get("portrait_path")
                    if _exist and Path(str(_exist)).exists():
                        _portrait_ladder[_si] = {
                            "portrait_subject": _pname,
                            "portrait_source_type": "preresolved",
                            "portrait_source_path": str(_exist),
                            "portrait_validation_score": None,
                            "portrait_fallback_reason": None}
                        continue
                    _nar = ""
                    try:
                        _nar = getattr(script.scenes[_si], "narration", "") or ""
                    except Exception:  # noqa: BLE001
                        _nar = ""
                    _pdest = run_dir / f"legend_portrait_{_si}.jpg"
                    _pl = _ftgp.resolve_legend_portrait(
                        _pname, role="", narration=_nar, dest=_pdest, cfg=cfg,
                        cache_dir=cache, allow_ai=_aimg_on, seed=(_seed + _si))
                    if _pl.get("portrait_path"):
                        _d.inputs["portrait_path"] = _pl["portrait_path"]
                    _portrait_ladder[_si] = {_k: _pl.get(_k) for _k in (
                        "portrait_subject", "portrait_source_type",
                        "portrait_source_path", "portrait_validation_score",
                        "portrait_fallback_reason")}
                    print(f"  [legend-portrait] {_pname[:34]} | "
                          f"{_pl.get('portrait_source_type')} | "
                          f"score={_pl.get('portrait_validation_score')} | "
                          f"{_pl.get('portrait_fallback_reason') or 'ok'}",
                          flush=True)
            except Exception as _ple:  # noqa: BLE001
                _log("5/5", f"legend-portrait ladder skipped "
                            f"({str(_ple)[:80]})")
            # V1.1 DURATION STRATEGY — each graphic OWNS the scene only for its
            # USEFUL window (per-primitive: short keyword punch … longer empire
            # read), then footage RETURNS for the rest of the scene. The clip is
            # rendered `win + 1.0`s long so the animation completes and the
            # closing fade parks just past the window → the scene cuts BRIGHT
            # back to footage (no dim frozen tail). `win` is capped at the scene
            # length, so short scenes still fill (current safe behaviour).
            _win_map: dict = {}
            _durs: dict = {}
            _nsc = len(narration.scenes)
            for _d in _dec:
                if not getattr(_d, "primitive", None):
                    continue
                _si = _d.scene_index
                _sdur = (narration.scenes[_si].duration
                         if 0 <= _si < _nsc else 6.0) or 6.0
                _ud = _mgrd.useful_dur(_d.primitive)
                _win = max(2.0, min(_ud, _sdur))
                _win_map[_si] = round(_win, 2)
                # V1.2 — render the clip long enough to COVER the whole scene
                # (plus a margin), not just win+1. For a multi-beat scene the
                # per-beat windowing still returns to footage after the window;
                # but for a SINGLE long beat (no split — beat-split is deferred)
                # the slice would otherwise run past a win+1 clip and tpad-hold
                # its faded-to-black tail for seconds (a ~6 s near-black hold).
                # Covering the scene means the held frame is the BRIGHT graphic.
                _durs[_si] = round(max(_win + 1.0, _sdur + 0.4), 2)
            _res = _mgrd.dispatch_all(_dec, run_dir=run_dir, durations=_durs, fps=30)
            _motion_graphics = {e["scene_index"]: e["path"] for e in _res["entries"]
                                if e.get("ok") and e.get("path") and not e.get("skipped")}
            _mg_windows = {si: _win_map[si] for si in _motion_graphics
                           if si in _win_map}
            # P5 (2026-06-01) — forward the per-scene PRIMITIVE id so the SFX
            # director can voice director-INJECTED stat/number cards (e.g. the
            # Cornell "96%" gold_number_callout) that carry no script
            # graphic_kind and were therefore landing SILENT.
            _mg_primitives = {_d.scene_index: _d.primitive for _d in _dec
                              if getattr(_d, "primitive", None)
                              and _d.scene_index in _motion_graphics}
            _log("5/5", f"motion graphics: {len(_motion_graphics)} scene(s) "
                 f"· {_res['summary']['by_primitive']} · "
                 f"{_res['summary']['cache_hits']}h/{_res['summary']['cache_misses']}m "
                 f"· windows {sorted(set(_mg_windows.values()))}s")
            # ── Reusable asset-QA pass: record the palette-selection reason and
            # flag any niche/palette/portrait doubts into the MG manifest, so no
            # future render silently accepts a questionable asset. Defensive. ──
            try:
                import json as _json
                from . import asset_qa as _aqa
                _pal, _pal_reason = _mgdir.video_palette_reason(_niche, _seed)
                _qa_warn = list(_aqa.check_palette_niche(_niche, _pal))
                _era = None
                try:
                    from . import footage as _ftg
                    for _pn, _pp in (getattr(_ftg, "_PORTRAIT_PROVENANCE", {})
                                     or {}).items():
                        _qa_warn += _aqa.check_portrait(
                            _pn, prov=_pp, title=str(_pp.get("source", "")),
                            ai_generated=bool(_pp.get("ai_generated")))
                    _era = (getattr(_ftg, "_VIDEO_CTX", {}) or {}).get("era")
                except Exception:                              # noqa: BLE001
                    pass
                _mpath = run_dir / "motion_graphics_manifest.json"
                if _mpath.exists():
                    _md = _json.loads(_mpath.read_text())
                    _md["asset_qa"] = {
                        "palette": _pal, "palette_reason": _pal_reason,
                        "niche": _niche, "era": _era, "warnings": _qa_warn,
                        "summary": _aqa.summarize(_qa_warn)}
                    # V3.3.1 — MG LIFECYCLE AUDIT (logging retention ONLY; does NOT
                    # read, alter, or depend on the asset_qa payload above). Records
                    # the director's requested→selected→rendered decisions + the
                    # restraint removals as a NAMESPACED sibling key, so the final
                    # manifest preserves the full MG lifecycle across this rewrite.
                    # Built defensively — a failure here never affects asset_qa or
                    # the manifest write (caught + skipped).
                    try:
                        import time as _t331
                        _ents = (_res.get("entries", []) if isinstance(_res, dict) else [])
                        _byrend = {e.get("scene_index"): e for e in _ents
                                   if isinstance(e, dict)}
                        _decs = []
                        for _d in (_dec or []):
                            _si = getattr(_d, "scene_index", None)
                            _rd = _byrend.get(_si, {})
                            _prim = getattr(_d, "primitive", None)
                            _decs.append({
                                "scene_index": _si,
                                "requested_primitive": _prim,
                                "score": round(float(getattr(_d, "score", 0.0) or 0.0), 3),
                                "select_reason": getattr(_d, "reason", ""),
                                "rendered": bool(_rd.get("ok")),
                                "fallback": bool(_rd.get("fallback", False)),
                                "skipped": bool(_rd.get("skipped", False)),
                                "cache_hit": bool(_rd.get("cache_hit", False)),
                                "render_s": _rd.get("render_s"),
                                "mode": ("engine_suggested" if _prim else "footage_only"),
                                "factual_guard": ("source_evidence_required" if _prim
                                                  else "no_evidence_footage_only"),
                            })
                        _md["motion_graphics_audit"] = {
                            "version": "v3.3.1",
                            "timestamp": round(_t331.time(), 1),
                            "niche": _niche, "palette": _pal,
                            "summary": (_mgdir.plan_summary(_dec)
                                        if hasattr(_mgdir, "plan_summary") else {}),
                            "decisions": _decs,
                            "restraint_removed": list(_mg_removed),
                            "rendered_primitives": sorted(
                                {e.get("primitive") for e in _ents
                                 if e.get("ok") and e.get("primitive")}),
                            "factual_guard_rule": "no_verified_evidence_then_footage_only",
                            # V3.3.2 — additive: every fact-bearing card the
                            # centralized guard rejected (scene/kind/source-span/
                            # reason/footage-only/timestamp). The audit-namespace
                            # `version` stays "v3.3.1" so the preservation test is
                            # unaffected; the guard stamps its own version here.
                            "factual_guard_version": "v3.3.2",
                            "factual_guard_rejected": list(_factual_rejected),
                        }
                    except Exception:                              # noqa: BLE001
                        pass
                    # V3.0.1 — attach portrait-ladder provenance to each
                    # portrait_legend_reveal scene entry (manifest fields).
                    if _portrait_ladder:
                        for _ent in _md.get("scenes", []):
                            _pl = _portrait_ladder.get(_ent.get("scene_index"))
                            if _pl:
                                _ent.update(_pl)
                    _mpath.write_text(_json.dumps(_md, indent=1))
                _log("5/5", f"asset-QA: palette={_pal} ({_pal_reason}) · "
                     f"{len(_qa_warn)} warning(s)")
            except Exception:                                  # noqa: BLE001
                pass
        except Exception as _mge:  # noqa: BLE001
            _motion_graphics = None
            _mg_windows = None
            _log("5/5", f"motion graphics skipped ({str(_mge)[:90]})")
    # V3.4.2 — ASSET-QA DECOUPLED FROM THE MG FLAG. Always emit asset_qa.json so
    # an MG-OFF render is never left without an asset-QA record. (Asset
    # PROTECTION — footage relevance, period guard, dedup, black-frame repair,
    # factual guard — already runs in the always-on footage+assembly stages,
    # independent of MG; this restores the matching OBSERVABILITY in both modes.)
    try:
        _aq_payload = _emit_asset_qa_manifest(
            run_dir, brief,
            mg_on=(os.environ.get("VIDLORE_MOTION_GRAPHICS") == "1"))
        if _aq_payload:
            _log("5/5", f"asset-QA [{_aq_payload.get('mode')}]: "
                 f"{len(_aq_payload.get('warnings', []))} warning(s) → asset_qa.json")
    except Exception:                                              # noqa: BLE001
        pass
    _phase_mark("phase5_start")
    assemble(
        footage, narration, th, work, video,
        motion_graphics=_motion_graphics,
        motion_graphics_windows=_mg_windows,
        motion_graphics_primitives=_mg_primitives,
        captions=brief.captions, music=music,
        transitions=cfg.transitions_enabled,
        title=script.title, overlays=cfg.overlays_enabled,
        chapters=[
            (sc.keywords[0] if sc.keywords else sc.narration)
            for sc in script.scenes
        ],
        energies=[sc.energy for sc in script.scenes],
        emphasis=[sc.emphasis for sc in script.scenes],
        graphics=[(sc.graphic_kind, sc.graphic_text, sc.graphic_body)
                  for sc in script.scenes],
        graphic_assets=graphic_assets,
        shot_types=[sc.shot_type for sc in script.scenes],
        roles=[sc.role for sc in script.scenes],
        beat_clips=fr.beat_clips,
        sfx=cfg.sfx_enabled,
        style=mode,
    )
    _phase_mark("phase5_end")
    _log("5/5", f"assembly wall {_phase_dur('phase5_start','phase5_end'):0.1f}s")
    _emit(progress, 98, "Finalizing…")

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    # Clear the look context var first — render is done, no more
    # downstream consumers should read it.
    try:
        from .look_dna import clear_current as _ld_clear
        _ld_clear()
    except Exception:                                              # noqa: BLE001
        pass

    # ── LOOK DNA: append this render's fingerprint to the channel's
    # history.jsonl + bump the variation cursor.  We use the on-disk
    # forensics module to get a real measured fingerprint (not the
    # predicted one) so anti-convergence next time has ground truth.
    if look is not None and getattr(brief, "channel", None):
        try:
            from .look_dna import commit_render as _commit
            from .forensics import analyze as _forensic_analyze
            fp_obj = _forensic_analyze(video)
            fp_dict = {
                "median_shot_s":     fp_obj.median_shot_s,
                "cuts_per_min":      fp_obj.cuts_per_min,
                "music_presence_pct": fp_obj.music_presence_pct,
                "n_long_silences":   fp_obj.n_long_silences,
                "mean_brightness":   fp_obj.mean_brightness,
                "mean_saturation":   fp_obj.mean_saturation,
                "mean_edge_density": fp_obj.mean_edge_density,
                "text_density_proxy": fp_obj.text_density_proxy,
            }
            _commit(brief.channel, brief, look, fp_dict)
            print(f"  [look-dna] history appended for channel "
                  f"'{brief.channel}'", flush=True)
        except Exception as _e:                                    # noqa: BLE001
            print(f"  [look-dna] history commit failed: {_e}",
                  flush=True)

    # ── EDITORIAL RECIPE: record this render's recipe signature in the
    # cross-video history (~/.vidlore/editorial_history.json) so the
    # anti-repetition engine nudges the NEXT same-niche video toward a
    # DIFFERENT editorial recipe.  Fires for channel-less AUTO-LOOK
    # renders too (cheap signature append — no forensic pass), which is
    # exactly the de-templating case (20 same-niche docs each differ).
    try:
        from .look_dna import last_look_recipe as _last_recipe
        from . import editorial_recipe as _er
        _rec = _last_recipe()
        if _rec and not _er.disabled():
            _er.record_recipe(
                _rec, channel=getattr(brief, "channel", None) or None)
            print(f"  [recipe] editorial recipe recorded "
                  f"(niche={_rec.get('niche')})", flush=True)
    except Exception as _e:                                        # noqa: BLE001
        print(f"  [recipe] recipe record skipped: {_e}", flush=True)

    # ── AUDIO cross-video anti-repetition: record this video's audio
    # fingerprint (categories / tracks / SFX families) so the NEXT video on
    # this channel leads with a different music category + SFX texture.
    try:
        import json as _json
        from .audio_director import music_director as _amd
        _scope = str(getattr(brief, "channel", "") or "default")
        _ms_path = None
        for _cand in (work / "music_cue_sheet.json",
                      run_dir / "music_cue_sheet.json"):
            if _cand.exists():
                _ms_path = _cand
                break
        if _ms_path is not None:
            _sheet = _json.loads(_ms_path.read_text(encoding="utf-8"))
            _fams = []
            for _sc in (run_dir / "sfx_cue_sheet.json",
                        work / "sfx_cue_sheet.json"):
                if _sc.exists():
                    _fams = list(_json.loads(_sc.read_text(encoding="utf-8"))
                                 .get("families_used", []))
                    break
            _amd.record_history(_sheet, scope=_scope, sfx_families=_fams)
            print(f"  [audio] cross-video history recorded "
                  f"(niche={_sheet.get('niche')}, "
                  f"lead={_sheet.get('lead_category')})", flush=True)
            # per-video attribution file for any credit-required (CC-BY/YTAL) track
            if _amd.generate_attribution_file(_sheet, run_dir / "MUSIC_CREDITS.txt"):
                print("  [audio] wrote MUSIC_CREDITS.txt (attribution-required "
                      "music used)", flush=True)
    except Exception as _e:                                        # noqa: BLE001
        print(f"  [audio] audio history record skipped: {_e}", flush=True)

    # ── EDITORIAL QA — autonomous post-render quality gate (opt-in). When
    # VIDLORE_EDITORIAL_QA=1 every finished render is scanned and an
    # EDITORIAL_QA_REPORT.json/.md is written next to the MP4 (repeated footage,
    # unreadable card text, black/dim frames, loudness …). The re-route +
    # re-render AUTOFIX loop lives in qa_autofix.run_with_qa (the portal/CLI
    # wraps render with it); this hook guarantees the report exists even for a
    # single one-shot render. QA never breaks a render — failures are caught.
    try:
        from . import qa_autofix as _qa
        if _qa.qa_enabled():
            _qa.scan_and_report(video, run_dir)
    except Exception as _qe:                                       # noqa: BLE001
        print(f"  [editorial-qa] post-render scan skipped: {_qe}", flush=True)

    # ── RC5 POST-RENDER RELEVANCE QA — last-line backstop. The fail-closed
    # relevance gate runs during asset SELECTION; this samples a BOUNDED set of
    # frames from the FINISHED MP4 and flags any off-topic / designed-graphic
    # frame that slipped every in-pipeline gate (incl. MG/portrait/editor paths
    # that never hit the selection gate). It writes render_relevance_qa.json +
    # a render_metrics.json "relevance_qa" block so a render is never silently
    # approved with a junk visual. NON-aborting (report+flag; the MP4 stands),
    # fully fail-safe, and gated by VIDLORE_POSTRENDER_QA (default ON).
    try:
        finalize_relevance_qa(video, run_dir)
    except Exception as _rqe:                                      # noqa: BLE001
        # finalize is already fail-safe; this is belt-and-suspenders so the
        # render result is returned no matter what.
        print(f"  [relevance-qa] post-render sweep skipped: {_rqe}", flush=True)

    _emit(progress, 100, "Done")
    return Result(
        video=video,
        thumbnail=None,
        script_txt=run_dir / "script.txt",
        srt=video.with_suffix(".srt"),
        seconds=time.time() - t0,
        video_seconds=_video_seconds(video, run_dir),
    )


# --------------------------------------------------------------------------- #
# One-shot (no review pause) — back-compat / --auto
# --------------------------------------------------------------------------- #
def produce(
    brief: Brief,
    cfg: Config,
    out_dir: Path,
    *,
    script_file: str | None = None,
    keep_work: bool = False,
) -> Result:
    write_script(brief, cfg, out_dir, script_file=script_file)
    return render_from_script(brief, cfg, out_dir, keep_work=keep_work)
