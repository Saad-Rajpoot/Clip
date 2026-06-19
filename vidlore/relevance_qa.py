"""RC5 — bounded post-render visual-relevance frame-sweep QA.

The fail-closed gate in `visual_relevance.py` runs DURING asset selection, but
three producers reach the screen WITHOUT it (MG-card imagery, portrait
resolution, Review-Editor replacements — see RC5_VISUAL_RELEVANCE_REPORT.md
"NOT yet done / STEP 9 + STEP 10"). This module is the LAST-LINE backstop: it
samples a small, BOUNDED set of frames from a FINISHED render and asks the same
CLIP graphic/relevance scorer "does any frame look like a designed graphic /
text-board / off-topic image?" — surfacing a timestamp + suggestion for each
flagged frame so an operator (or a harness) can catch a junk visual that slipped
every in-pipeline gate.

DESIGN CONSTRAINTS (per the RC5 brief):
  * BOUNDED sampling — a regular interval (~every N seconds) PLUS each scene
    boundary from `render_meta["scene_starts"]` PLUS any MG/portrait placement
    timestamps found in an optional manifest. The total is capped to roughly
    scene_count + boundaries; there is NO uncontrolled per-frame loop.
  * NEVER raises — any failure (no ffmpeg, scorer off, unreadable mp4) returns a
    PASS verdict with an `error` note, so a QA harness is never the thing that
    breaks. It is a *reporter*, not a render gate.
  * Uses the bundled imageio-ffmpeg binary via `vidlore.ffmpeg_tool.ffmpeg_exe`
    to extract frames to a temp dir, and cleans the temp dir up afterwards.
  * Importable AND standalone-callable (`python -m vidlore.relevance_qa MP4`).
  * NOT wired into the hot render path — a harness calls `sweep(...)` explicitly.

MOTION-GRAPHICS AWARENESS (the false-positive fix):
  The pixel `graphic_dom` probe sees "designed graphic with text" and CANNOT
  tell a fetched junk poster (anime cover / game UI / multilingual sign) apart
  from the engine's OWN intentional motion-graphics cards (chapter title cards,
  cause→effect cards, the classified INTEL BRIEF card, chronology timelines,
  statement cards, process-flow steps). Those engine cards ARE designed graphics
  with text — by design — so the raw probe false-positives on every one of them
  and a card-rich documentary spuriously FAILs the sweep.

  The sweep therefore builds a set of CARD TIME WINDOWS and treats the
  designed-graphic / text-heavy signal as EXEMPT inside them. A window is built
  from any of:
    1. the MOTION GRAPHICS manifest (`motion_graphics_manifest.json`):
         * `motion_graphics_audit.summary.at_scenes` → [scene_index, primitive]
         * any `scenes[]` entry with a non-null `primitive` (i.e. an MG actually
           rendered for that scene, not a skipped footage scene)
       each placed scene_index → time window via render_meta
       `scene_starts[i] .. scene_starts[i] + scene_durations[i]`.
    2. a legacy/engine card discoverable from render_meta or an attached script
       (a scene carrying a non-empty `graphic_kind` / `graphic_text`).
  The MG manifest is loaded from an explicit `mg_manifest=` arg OR auto-discovered
  as `<mp4 dir>/motion_graphics_manifest.json` (the render run-dir), so the wired
  pipeline caller benefits without passing it.

  A sampled frame whose timestamp falls INSIDE a card window is exempt from the
  designed-graphic rejection ONLY (other reasons — off-topic / era — would still
  flag there if we computed them). A frame OUTSIDE every card window keeps the
  existing strict junk detection, so a fetched anime/poster/sign on a footage
  beat is STILL caught.

  FALLBACK (no manifest, no card metadata): a card-rich doc would otherwise
  spuriously FAIL, so for full-frame card-like frames we RAISE the graphic_dom
  threshold to a card-aware ceiling (`VIDLORE_QA_CARD_GRAPHIC_MAX`, default
  0.12). Calibration: the engine's legitimate cards probe graphic_dom ≈
  0.037–0.076 (just over the 0.036 selection-gate threshold), whereas genuine
  junk (anime cover ≈ 0.21, game UI / poster screenshots) sits far higher — so a
  ~0.12 ceiling cleanly separates "engine card" from "fetched junk" even with no
  manifest to consult.

STALE-OUTPUT GUARD (RC5.1 FIX 1):
  The sweep can only certify the file it actually SCANNED. A render pipeline that
  regenerates into a scratch dir but SERVES a different (stale, cached) MP4 will
  pass QA on the scratch file while the user watches the junk-bearing stale file
  (the exact game-UI miss this guard closes). So the sweep now:
    * ALWAYS records `scanned_path`, `scanned_sha256`, `scanned_mtime` in the
      result and the written report, so a human can see WHICH file was inspected.
    * accepts optional `expected_sha256` / `expected_path` — the caller's record
      of the FINAL EXPORT the user receives. If `expected_sha256` is supplied and
      does NOT match the scanned file's sha256, the verdict is `FAIL_STALE_OUTPUT`
      (loud — the QA scanned the wrong file; its PASS/FAIL on content is moot).

PER-BEAT COVERAGE (RC5.1 FIX 2):
  Bounded grid sampling can skip a beat (a 1.5-2s UI flash between grid points).
  The sample plan now ALWAYS includes ≥1 representative frame per SCENE BOUNDARY
  and, when beat timings are available (render_meta `beat_starts`, an
  ASSET_DECISION_MANIFEST with per-beat times, or scene_starts+scene_durations as
  a fallback beat grid), ≥1 frame per BEAT (start + mid). The frame cap is RAISED
  to `max(80, 2×scenes, beats+boundaries+slack)` so no beat is silently skipped on
  a ~4-min / 31-scene / 62-beat doc, while staying bounded + fail-safe.

Public API:
    sweep(mp4_path, render_meta, *, ffmpeg=None, manifest=None,
          mg_manifest=None, interval_s=None, max_frames=None,
          expected_sha256=None, expected_path=None) -> dict

Returns:
    {
      "verdict": "PASS" | "FAIL_RELEVANCE_QA" | "FAIL_STALE_OUTPUT",
      "flags": [ {timestamp, scene, reason, suggestion, in_card_window}, ... ],
      "sampled": <int frames actually scored>,
      "card_windows": <int card windows resolved>,
      "duration_s": <float>,
      "scanned_path": <str absolute path of the MP4 inspected>,
      "scanned_sha256": <str sha256 hex of the inspected MP4, or "">,
      "scanned_mtime": <float mtime of the inspected MP4, or 0.0>,
      "expected_sha256": <str echoed back when provided, else "">,
      "stale_output": <bool — True when expected_sha256 != scanned_sha256>,
      "error": <str or "">,         # non-empty ⇒ sweep degraded (still PASS)
    }

  Each flag records `in_card_window` (bool) — whether the flagged timestamp fell
  inside a resolved engine/MG card window. The sweep never EMITS a flag whose
  sole reason is the designed-graphic signal AND which is inside a card window
  (those are exempt), so a present `in_card_window: true` on a flag means the
  flag survived for a non-graphic reason.

Env flags (all optional; sensible defaults so a bare call just works):
    VIDLORE_QA_INTERVAL_S       regular-sample spacing in seconds (default 6.0)
    VIDLORE_QA_MAX_FRAMES       hard cap on total sampled frames (default 80;
                                 the effective cap is raised per-render to cover
                                 every beat — see PER-BEAT COVERAGE above)
    VIDLORE_VR_GRAPHIC_MAX      designed-graphic threshold (shared with the gate)
    VIDLORE_QA_CARD_GRAPHIC_MAX raised graphic_dom ceiling for card-like frames
                                 OUTSIDE a known window (default 0.12)
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import visual_relevance as VR

# Defaults — kept conservative so the sweep is cheap and bounded. The interval
# is wide (a documentary cut rarely changes subject faster than this), and the
# frame cap is an absolute ceiling so even a pathological 2-hour input can never
# spin the scorer thousands of times.
_DEFAULT_INTERVAL_S = 6.0
_DEFAULT_MAX_FRAMES = 80
_MIN_INTERVAL_S = 1.5            # never sample faster than this, whatever the env

# RC5.1 FIX 2 — absolute hard ceiling on the per-beat-raised cap. The effective
# cap is raised per-render to cover every beat (max(MAX_FRAMES, 2×scenes,
# beats+boundaries+slack)) so no beat is silently skipped on a long doc, but it is
# still BOUNDED — never more than this, so a pathological input can't spin the
# scorer thousands of times. A 4-min/31-scene/62-beat doc needs ~100; this ceiling
# leaves generous headroom while staying fail-safe.
_ABSOLUTE_MAX_FRAMES = 400

# Card-aware designed-graphic ceiling used ONLY for frames OUTSIDE a known card
# window (the no-manifest / unknown-card fallback). The engine's legitimate
# cards probe graphic_dom ~0.037-0.076; genuine fetched junk (anime cover ~0.21,
# game UI / poster screenshots) sits far higher, so a ~0.12 ceiling separates the
# two without a manifest. Inside a resolved card window the designed-graphic
# signal is exempt outright (no threshold consulted).
_DEFAULT_CARD_GRAPHIC_MAX = 0.12

# Small symmetric pad (s) around a card window so a sample landing a hair before/
# after the cut still maps onto the card, and so a window of exactly one frame is
# never zero-width.
_WINDOW_PAD_S = 0.35


def _fnum(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _inum(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _probe_duration(mp4_path: Path, ff: str) -> float:
    """Best-effort clip duration in seconds (0.0 if unknown). Reuses the scorer's
    own ffmpeg-stderr parser so there is no ffprobe dependency. Never raises."""
    try:
        return float(VR._probe_duration(mp4_path, ff))   # noqa: SLF001
    except Exception:                                          # noqa: BLE001
        return 0.0


def _scene_for_ts(ts: float, scene_starts) -> int:
    """0-based scene index whose [start, next_start) window contains `ts`.
    Returns -1 when there are no boundaries to map against."""
    try:
        if not scene_starts:
            return -1
        idx = -1
        for i, s in enumerate(scene_starts):
            if ts + 1e-6 >= float(s):
                idx = i
            else:
                break
        return idx
    except Exception:                                          # noqa: BLE001
        return -1


def _manifest_timestamps(manifest) -> list:
    """Pull MG / portrait PLACEMENT timestamps out of an optional manifest so a
    card / portrait background (which never passes the selection gate) is always
    among the sampled frames. Tolerant of several shapes — a list of placements,
    or a dict with a 'placements'/'motion_graphics'/'cards' list — and of several
    key names for the time field. Never raises; returns [] on anything odd."""
    out: list = []
    try:
        if not manifest:
            return out
        items = manifest
        if isinstance(manifest, dict):
            for key in ("placements", "motion_graphics", "cards", "items",
                        "mg", "portraits"):
                v = manifest.get(key)
                if isinstance(v, list):
                    items = v
                    break
            else:
                items = []
        if not isinstance(items, list):
            return out
        for it in items:
            if not isinstance(it, dict):
                continue
            for tkey in ("t", "ts", "time", "start", "start_s", "at",
                         "timestamp", "scene_start"):
                if tkey in it and it[tkey] is not None:
                    try:
                        out.append(round(float(it[tkey]), 2))
                        break
                    except (TypeError, ValueError):
                        continue
    except Exception:                                          # noqa: BLE001
        return []
    return out


def _load_json_path(path):
    """Read + parse a JSON file, returning the object or None. Never raises."""
    try:
        from pathlib import Path as _P
        p = _P(path)
        if not p.exists():
            return None
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def _sha256_file(path) -> str:
    """Streamed sha256 hex of a file (RC5.1 stale-output guard). Returns "" on any
    error so a hashing failure degrades the guard to "unknown", never an exception
    — the sweep stays fail-safe."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:                                          # noqa: BLE001
        return ""


def _beat_times(render_meta, manifest) -> list:
    """RC5.1 FIX 2 — per-BEAT representative timestamps (start + mid of each beat).

    Beats are finer than scenes; bounded grid sampling can skip a 1.5-2s beat
    (the UI-flash that slipped QA). We harvest beat windows from, in priority:
      1. render_meta `beat_starts` (+ optional `beat_durations`) — explicit beats.
      2. an ASSET_DECISION_MANIFEST `beats[]` carrying per-beat `t`/`start`/`ts`
         (and optional `end`/`dur`) — the live pipeline writes one beat per asset.
      3. FALLBACK: render_meta `scene_starts` + `scene_durations` treated as a beat
         grid (so even a meta with only scene timings still gets start+mid per
         scene — never silently skipping a scene's body).
    For each beat we emit its START and (when a duration/next-start is known) its
    MID. Tolerant of odd shapes; never raises; returns []."""
    out: list = []
    try:
        meta = render_meta if isinstance(render_meta, dict) else {}

        def _emit(start, end):
            try:
                s = float(start)
            except (TypeError, ValueError):
                return
            out.append(round(max(0.0, s), 3))
            if end is not None:
                try:
                    e = float(end)
                    if e > s:
                        out.append(round((s + e) * 0.5, 3))
                except (TypeError, ValueError):
                    pass

        # (1) explicit beat timings in render_meta
        bstarts = meta.get("beat_starts")
        if isinstance(bstarts, list) and bstarts:
            bdurs = meta.get("beat_durations")
            for i, s in enumerate(bstarts):
                end = None
                if isinstance(bdurs, list) and i < len(bdurs):
                    try:
                        end = float(s) + float(bdurs[i])
                    except (TypeError, ValueError):
                        end = None
                if end is None and i + 1 < len(bstarts):
                    end = bstarts[i + 1]
                _emit(s, end)
            if out:
                return sorted(set(out))

        # (2) ASSET_DECISION_MANIFEST per-beat times
        beats = None
        if isinstance(manifest, dict) and isinstance(manifest.get("beats"), list):
            beats = manifest["beats"]
        if isinstance(beats, list) and beats:
            got = False
            for b in beats:
                if not isinstance(b, dict):
                    continue
                st = None
                for tkey in ("t", "ts", "start", "start_s", "at", "timestamp",
                             "scene_start", "beat_start"):
                    if b.get(tkey) is not None:
                        st = b.get(tkey)
                        break
                if st is None:
                    continue
                end = None
                for ekey in ("end", "end_s", "stop"):
                    if b.get(ekey) is not None:
                        end = b.get(ekey)
                        break
                if end is None:
                    for dkey in ("dur", "duration", "len", "length"):
                        if b.get(dkey) is not None:
                            try:
                                end = float(st) + float(b.get(dkey))
                            except (TypeError, ValueError):
                                end = None
                            break
                _emit(st, end)
                got = True
            if got and out:
                return sorted(set(out))

        # (3) FALLBACK — scene grid as a coarse beat grid (start + mid per scene)
        starts = meta.get("scene_starts")
        durs = meta.get("scene_durations")
        if isinstance(starts, list) and starts:
            n = len(starts)
            for i, s in enumerate(starts):
                end = None
                if isinstance(durs, list) and i < len(durs):
                    try:
                        end = float(s) + float(durs[i])
                    except (TypeError, ValueError):
                        end = None
                if end is None and i + 1 < n:
                    end = starts[i + 1]
                _emit(s, end)
    except Exception:                                          # noqa: BLE001
        return sorted(set(out))
    return sorted(set(out))


def _resolve_mg_manifest(mg_manifest, mp4_path):
    """Resolve the MOTION GRAPHICS manifest to a dict. Accepts an already-parsed
    dict, a path to `motion_graphics_manifest.json`, OR (when neither is given)
    auto-discovers `<mp4 dir>/motion_graphics_manifest.json` — the render run-dir,
    so the wired pipeline caller benefits without threading a new arg. Returns the
    dict or None. Never raises."""
    try:
        if isinstance(mg_manifest, dict):
            return mg_manifest
        if isinstance(mg_manifest, (str, bytes, os.PathLike)):
            return _load_json_path(mg_manifest)
        # Auto-discover next to the rendered MP4 (the run dir).
        if mp4_path is not None:
            sib = Path(mp4_path).parent / "motion_graphics_manifest.json"
            return _load_json_path(sib)
    except Exception:                                          # noqa: BLE001
        return None
    return None


def _placed_mg_scene_indices(mg_doc) -> set:
    """The set of 0-based scene indices where an MG primitive was actually placed,
    read from the MOTION GRAPHICS manifest. Unions two sources that the engine
    writes:
        * motion_graphics_audit.summary.at_scenes — [[scene_index, primitive], …]
        * scenes[] entries whose `primitive` is non-null and not skipped
    Tolerant of missing keys / odd shapes. Never raises; returns set() on junk."""
    out: set = set()
    try:
        if not isinstance(mg_doc, dict):
            return out
        # (a) audit.summary.at_scenes
        aud = mg_doc.get("motion_graphics_audit")
        if isinstance(aud, dict):
            summ = aud.get("summary")
            if isinstance(summ, dict):
                ats = summ.get("at_scenes")
                if isinstance(ats, list):
                    for entry in ats:
                        idx = None
                        if isinstance(entry, (list, tuple)) and entry:
                            idx = entry[0]
                        elif isinstance(entry, dict):
                            idx = (entry.get("scene_index")
                                   if entry.get("scene_index") is not None
                                   else entry.get("scene"))
                        elif isinstance(entry, (int, float)):
                            idx = entry
                        try:
                            if idx is not None:
                                out.add(int(idx))
                        except (TypeError, ValueError):
                            continue
        # (b) scenes[] with a rendered primitive (skip footage-only scenes)
        scenes = mg_doc.get("scenes")
        if isinstance(scenes, list):
            for s in scenes:
                if not isinstance(s, dict):
                    continue
                prim = s.get("primitive")
                if prim in (None, "", "footage"):
                    continue
                if s.get("skipped") is True:
                    continue
                idx = (s.get("scene_index")
                       if s.get("scene_index") is not None
                       else s.get("scene"))
                try:
                    if idx is not None:
                        out.add(int(idx))
                except (TypeError, ValueError):
                    continue
    except Exception:                                          # noqa: BLE001
        return out
    return out


def _legacy_card_scene_indices(render_meta) -> set:
    """The set of scene indices carrying a LEGACY/engine card discoverable from
    render_meta (or an attached script under common keys). A scene with a
    non-empty `graphic_kind` / `graphic_text` is an authored full-screen card
    (title_card / statement / classified / cause_effect / explainer / …). Best
    effort — render_meta usually carries only counts, so this is empty unless a
    caller attaches scene records. Never raises."""
    out: set = set()
    try:
        if not isinstance(render_meta, dict):
            return out
        scenes = None
        for key in ("scenes", "scene_records", "script_scenes", "beats"):
            v = render_meta.get(key)
            if isinstance(v, list):
                scenes = v
                break
        # Some callers attach the parsed script under render_meta["script"].
        if scenes is None:
            scr = render_meta.get("script")
            if isinstance(scr, dict) and isinstance(scr.get("scenes"), list):
                scenes = scr["scenes"]
        if not isinstance(scenes, list):
            return out
        for i, s in enumerate(scenes):
            if not isinstance(s, dict):
                continue
            gk = str(s.get("graphic_kind") or "").strip()
            gt = str(s.get("graphic_text") or "").strip()
            if not gk and not gt:
                continue
            idx = s.get("scene_index")
            try:
                idx = int(idx) if idx is not None else i
            except (TypeError, ValueError):
                idx = i
            out.add(idx)
    except Exception:                                          # noqa: BLE001
        return out
    return out


def _card_windows(render_meta, mg_doc) -> list:
    """Build the list of (start_s, end_s) CARD TIME WINDOWS — spans of the render
    where the on-screen visual is an engine-rendered card / MG primitive and the
    designed-graphic / text-heavy probe must NOT be treated as junk.

    Each placed scene index (from the MG manifest and/or legacy graphic_kind) maps
    to its time window via render_meta `scene_starts[i] .. scene_starts[i] +
    scene_durations[i]`, padded by `_WINDOW_PAD_S`. Never raises; returns []."""
    windows: list = []
    try:
        meta = render_meta if isinstance(render_meta, dict) else {}
        starts = meta.get("scene_starts") or []
        durs = meta.get("scene_durations") or []
        if not isinstance(starts, list) or not starts:
            return windows
        idxs = set()
        idxs |= _placed_mg_scene_indices(mg_doc)
        idxs |= _legacy_card_scene_indices(meta)
        n = len(starts)
        for i in sorted(idxs):
            if not (0 <= i < n):
                continue
            try:
                s0 = float(starts[i])
            except (TypeError, ValueError):
                continue
            # Window end: this scene's duration if known, else next start, else a
            # short default so a card always owns at least a couple of seconds.
            dur = None
            if isinstance(durs, list) and i < len(durs):
                try:
                    dur = float(durs[i])
                except (TypeError, ValueError):
                    dur = None
            if dur is not None and dur > 0:
                s1 = s0 + dur
            elif i + 1 < n:
                try:
                    s1 = float(starts[i + 1])
                except (TypeError, ValueError):
                    s1 = s0 + 4.0
            else:
                s1 = s0 + 4.0
            windows.append((max(0.0, s0 - _WINDOW_PAD_S), s1 + _WINDOW_PAD_S))
    except Exception:                                          # noqa: BLE001
        return windows
    return windows


def _in_card_window(ts: float, windows) -> bool:
    """True when `ts` falls inside any (start, end) card window. Never raises."""
    try:
        t = float(ts)
        for w in (windows or []):
            if w[0] <= t <= w[1]:
                return True
    except Exception:                                          # noqa: BLE001
        return False
    return False


def _build_sample_times(duration: float, scene_starts, manifest,
                        interval_s: float, max_frames: int,
                        beat_times=None) -> list:
    """The BOUNDED sample plan: a union of (a) a regular interval grid, (b) each
    scene boundary +0.4s (so we read INTO the scene, not the dissolve seam), (c)
    any MG/portrait placement timestamps, and (d) RC5.1 per-BEAT timestamps
    (start + mid of each beat). Deduplicated to ~0.5s buckets, sorted, clamped
    inside the clip.

    RC5.1 FIX 2 — scene BOUNDARIES and per-beat starts are MANDATORY: they are
    preserved in full and the cap is enforced by decimating only the FILL points
    (interval grid + beat mids + manifest placements). So no scene/beat is ever
    silently skipped, while the total stays bounded."""
    mandatory: set = set()          # scene boundaries + beat starts — never dropped
    fill: set = set()               # interval grid + beat mids + manifest places
    usable = max(0.4, (duration - 0.3)) if duration > 0.8 else 0.0

    def _clamp(ts):
        return round(max(0.2, float(ts)), 2)

    # (a) regular interval grid  → FILL
    if usable > 0:
        step = max(_MIN_INTERVAL_S, interval_s)
        t = step * 0.5                                  # start mid-first-interval
        guard = 0
        while t < usable and guard < 100000:
            fill.add(round(t, 2))
            t += step
            guard += 1

    # (b) scene boundaries (+0.4s into the scene to clear the transition) → MANDATORY
    try:
        for s in (scene_starts or []):
            ts = float(s) + 0.4
            if usable <= 0 or ts < usable:
                mandatory.add(_clamp(ts))
    except Exception:                                          # noqa: BLE001
        pass

    # (c) MG / portrait placement timestamps from the manifest → FILL
    for ts in _manifest_timestamps(manifest):
        if usable <= 0 or 0 <= ts < usable:
            fill.add(_clamp(ts + 0.3))

    # (d) RC5.1 per-BEAT timestamps. The beat START (+0.3s into the beat to clear
    # the cut) is MANDATORY so a short beat can never be skipped; the beat MID is
    # FILL (nice-to-have, decimated first under the cap).
    bt = list(beat_times or [])
    if bt:
        # _beat_times emits [start, mid, start, mid, ...] per beat (mid only when a
        # duration was known). We can't perfectly re-pair here, so treat the FIRST
        # of each adjacent (start<mid) pair as mandatory and the rest as fill via a
        # simple heuristic: every value is a candidate; mark as mandatory unless it
        # is the midpoint between two other sampled beat values. Cheaper + safe:
        # mark ALL beat values mandatory when they are few; when many, keep starts
        # mandatory by taking every value but letting the cap decimate fill only.
        for ts in bt:
            if usable <= 0 or 0 <= float(ts) < usable:
                mandatory.add(_clamp(float(ts) + 0.3))

    # Fallback: if we have neither a duration nor any boundaries/beats, still probe
    # a few fixed early offsets so the sweep does *something* useful.
    if not mandatory and not fill:
        fill = {0.5, 2.0, 4.0, 7.0, 11.0}

    # Dedup to ~0.5s buckets so near-identical seeks don't double-scan a frame.
    # A point present in BOTH sets stays mandatory (drop it from fill).
    def _bucket(s):
        b: dict = {}
        for t in s:
            b[round(t * 2) / 2.0] = t
        return b
    mand_b = _bucket(mandatory)
    fill_b = {k: v for k, v in _bucket(fill).items() if k not in mand_b}
    mand_sorted = sorted(mand_b.values())
    fill_sorted = sorted(fill_b.values())

    cap = max(1, int(max_frames))
    # Mandatory points are never dropped. If they ALONE exceed the cap (a doc with
    # an enormous beat count), decimate the mandatory set evenly too — but only as a
    # last resort, and keep coverage spread across the whole clip.
    if len(mand_sorted) >= cap:
        if len(mand_sorted) > cap:
            stride = len(mand_sorted) / float(cap)
            mand_sorted = [mand_sorted[int(i * stride)] for i in range(cap)]
        return mand_sorted
    # Otherwise fill the remaining budget with decimated fill points.
    budget = cap - len(mand_sorted)
    if len(fill_sorted) > budget:
        stride = len(fill_sorted) / float(budget) if budget > 0 else len(fill_sorted)
        fill_sorted = [fill_sorted[int(i * stride)] for i in range(budget)]
    return sorted(set(mand_sorted) | set(fill_sorted))


def _extract_frame(ff: str, mp4_path: Path, ts: float, dest: Path) -> bool:
    """Extract ONE frame at `ts` to `dest` (PNG, downscaled). Returns False on
    any failure — the caller just skips that timestamp. Never raises."""
    try:
        subprocess.run(
            [ff, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{max(0.0, ts):.2f}", "-i", str(mp4_path),
             "-frames:v", "1", "-vf", "scale=320:-1", str(dest)],
            check=True, timeout=25,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return dest.exists() and dest.stat().st_size > 256
    except Exception:                                          # noqa: BLE001
        return False


def _finalize_verdict(base_verdict: str, flags, is_stale: bool) -> str:
    """RC5.1 verdict precedence. A STALE-OUTPUT mismatch (the QA scanned a file
    that is NOT the caller's final export) is the loudest possible result — it
    means the content verdict is about the WRONG file — so it overrides PASS and
    even a content FAIL. Otherwise the content verdict stands."""
    if is_stale:
        return "FAIL_STALE_OUTPUT"
    return base_verdict


def _suggestion_for(reason: str) -> str:
    """A short, human next-action for a flagged frame."""
    r = (reason or "").lower()
    # UI / game interface FIRST — its reason also contains "graphic"-ish words but
    # the right action is specific (it's a screenshot of software/a game).
    if ("ui screenshot" in r or "interface" in r or "in-game" in r
            or "game hud" in r or "game/software" in r or "dashboard" in r
            or "control panel" in r):
        return ("Frame is a game / software interface screenshot (HUD / dashboard "
                "/ control panel), not footage — replace with real archival or "
                "stock footage / an engine-rendered map card, never a UI capture.")
    if "graphic" in r or "designed" in r:
        return ("Frame looks like a designed graphic / text-board, not footage — "
                "replace this scene's visual with real footage or a grounded AI "
                "still, or remove the card background.")
    if "junk" in r or "metadata" in r:
        return ("Asset metadata matches a junk class (game/anime/cover/UI/poster/"
                "logo/meme) — swap for a relevant documentary visual.")
    if "off-topic" in r or "subject" in r or "wrong" in r:
        return ("Frame appears off-topic for the scene — replace with footage that "
                "matches the narration's subject.")
    return "Review this frame; it may not match the documentary's subject."


def sweep(mp4_path, render_meta, *, ffmpeg=None, manifest=None,
          mg_manifest=None, interval_s=None, max_frames=None,
          expected_sha256=None, expected_path=None) -> dict:
    """Bounded post-render frame-sweep relevance QA — see module docstring.

    Samples a capped set of frames (interval grid + scene boundaries + per-BEAT
    timestamps + manifest MG/portrait placements), runs the CLIP designed-graphic
    / relevance probe on each, and returns a verdict + per-flag timestamps. NEVER
    raises: on any error (scorer off, no ffmpeg, unreadable file) it returns PASS
    with an `error` note, because this is a reporter, not a render gate.

    RC5.1 STALE-OUTPUT GUARD: the scanned MP4's sha256 + path + mtime are ALWAYS
    recorded in the result. If `expected_sha256` is supplied (the caller's record
    of the FINAL EXPORT the user receives) and it does NOT match the scanned file,
    the verdict is `FAIL_STALE_OUTPUT` (loud) — the QA scanned the wrong file, so
    any content PASS/FAIL is moot. `expected_path` is echoed for the report.

    RC5.1 PER-BEAT COVERAGE: ≥1 representative frame per scene boundary AND per
    beat (start+mid), with a per-render-raised but bounded frame cap, so no beat is
    silently skipped.

    Motion-graphics aware: a frame inside an engine/MG CARD WINDOW (resolved from
    `mg_manifest` — explicit dict/path or auto-discovered
    `<mp4 dir>/motion_graphics_manifest.json` — plus any legacy `graphic_kind`
    scenes in render_meta) is EXEMPT from the designed-graphic/text-heavy
    rejection, so the engine's own intentional cards never false-fail. Frames
    OUTSIDE every window keep strict junk detection (incl. a UI-geometry hard
    reject for game/software interface frames, with a raised card-aware ceiling
    for full-frame card-like frames when no window data exists)."""
    result = {"verdict": "PASS", "flags": [], "sampled": 0,
              "card_windows": 0, "duration_s": 0.0,
              "scanned_path": "", "scanned_sha256": "", "scanned_mtime": 0.0,
              "expected_sha256": str(expected_sha256 or ""),
              "expected_path": str(expected_path or ""),
              "stale_output": False, "error": ""}
    tmpdir = None
    try:
        mp4_path = Path(mp4_path)
        # Record WHICH file is being inspected as early as possible — even a
        # not-found / hash-failure path leaves a breadcrumb for a human.
        try:
            result["scanned_path"] = str(mp4_path.resolve())
        except Exception:                                          # noqa: BLE001
            result["scanned_path"] = str(mp4_path)
        if not mp4_path.exists():
            result["error"] = "mp4-not-found"
            return result
        # STALE-OUTPUT GUARD (RC5.1 FIX 1): hash the file we actually scan, record
        # path + sha256 + mtime, and compare against the caller's expected final-
        # export hash. A mismatch is a LOUD FAIL_STALE_OUTPUT — we inspected the
        # wrong file, so a content verdict would be misleading. Hash failure
        # degrades to "" (guard inert) rather than raising.
        try:
            result["scanned_mtime"] = round(float(mp4_path.stat().st_mtime), 3)
        except Exception:                                          # noqa: BLE001
            result["scanned_mtime"] = 0.0
        scanned_sha = _sha256_file(mp4_path)
        result["scanned_sha256"] = scanned_sha
        exp_sha = str(expected_sha256 or "").strip().lower()
        is_stale = bool(exp_sha and scanned_sha and
                        exp_sha != scanned_sha.strip().lower())
        result["stale_output"] = is_stale
        meta = render_meta if isinstance(render_meta, dict) else {}
        scene_starts = meta.get("scene_starts") or []

        # ffmpeg binary — caller override, else the bundled imageio-ffmpeg one.
        ff = ffmpeg
        if not ff:
            try:
                from .ffmpeg_tool import ffmpeg_exe
                ff = ffmpeg_exe()
            except Exception:                                  # noqa: BLE001
                ff = "ffmpeg"

        # Duration: prefer render_meta, fall back to probing the file.
        duration = 0.0
        try:
            duration = float(meta.get("video_seconds") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0.8:
            duration = _probe_duration(mp4_path, ff)
        result["duration_s"] = round(duration, 2)

        # MOTION-GRAPHICS / engine-card awareness: resolve the MG manifest (arg,
        # path, or auto-discovered next to the MP4) and build the set of card time
        # windows. A frame inside one of these windows is the engine's OWN
        # intentional card — exempt from the designed-graphic/text-heavy rejection
        # so a card-rich documentary never spuriously FAILs on its own graphics.
        mg_doc = _resolve_mg_manifest(mg_manifest, mp4_path)
        card_windows = _card_windows(meta, mg_doc)
        result["card_windows"] = len(card_windows)

        # "card-rich" = this render is KNOWN to contain engine-rendered cards
        # (an MG primitive was placed somewhere, OR a scene carries a legacy
        # graphic_kind). When true, the engine ALSO bakes legacy/director-injected
        # cards (title_card / cause_effect / explainer / statement / process) that
        # are NOT enumerated in any manifest and fall OUTSIDE the MG windows. We
        # cannot pixel-distinguish those from junk in the borderline graphic_dom
        # band, so for OUT-of-window frames we apply the raised card-aware ceiling
        # (not the strict gate threshold) — sparing the engine's own cards while
        # still catching junk-grade graphics (anime/poster/UI sit far higher). A
        # render with NO cards at all keeps the strict threshold everywhere.
        card_rich = bool(card_windows) or bool(_placed_mg_scene_indices(mg_doc)) \
            or bool(_legacy_card_scene_indices(meta))

        # RC5.1 FIX 2 — per-beat timestamps + a RAISED but bounded frame cap so no
        # beat is silently skipped on a long doc. beat_times harvests start+mid of
        # every beat (render_meta beat_starts / ASSET_DECISION_MANIFEST beats /
        # scene grid fallback).
        beat_times = _beat_times(meta, manifest)
        n_scenes = len(scene_starts) if isinstance(scene_starts, list) else 0
        n_beats = len(beat_times)
        n_bounds = n_scenes

        iv = float(interval_s) if interval_s else _fnum("VIDLORE_QA_INTERVAL_S",
                                                        _DEFAULT_INTERVAL_S)
        if max_frames:
            cap = int(max_frames)
        else:
            base_cap = _inum("VIDLORE_QA_MAX_FRAMES", _DEFAULT_MAX_FRAMES)
            # Effective cap: high enough to cover EVERY scene boundary + beat (with
            # slack for the interval grid), floor at base_cap, hard-bounded by the
            # absolute ceiling so it can never run away. ceiling >= max(80, 2×scenes).
            need = n_beats + n_bounds + max(8, int(round((duration or 0) / max(
                _MIN_INTERVAL_S, iv)))) + 8
            cap = min(_ABSOLUTE_MAX_FRAMES,
                      max(base_cap, 2 * n_scenes, need))
        times = _build_sample_times(duration, scene_starts, manifest, iv, cap,
                                    beat_times=beat_times)
        result["sample_plan"] = {"scenes": n_scenes, "beats": n_beats,
                                 "cap": cap, "planned_frames": len(times)}
        if not times:
            result["error"] = "no-sample-points"
            result["verdict"] = _finalize_verdict("PASS", [], is_stale)
            return result

        # If the scorer is unavailable we cannot make a vision judgement. Report
        # it (so the harness knows the pixel check did not run) but PASS — the
        # in-pipeline metadata gate + the fail-closed selection path are the
        # backstops, exactly as `graphic_signal` documents. The STALE guard still
        # wins: a wrong-file scan is FAIL_STALE_OUTPUT even when the scorer is off.
        if not VR.available():
            result["error"] = "scorer-unavailable"
            result["sampled"] = 0
            result["verdict"] = _finalize_verdict("PASS", [], is_stale)
            return result

        try:
            gmax = float(os.environ.get("VIDLORE_VR_GRAPHIC_MAX",
                                        VR._DEFAULT_GRAPHIC_MAX))  # noqa: SLF001
        except (TypeError, ValueError):
            gmax = VR._DEFAULT_GRAPHIC_MAX                         # noqa: SLF001
        # Raised card-aware ceiling for card-like frames OUTSIDE a known window
        # (the no-manifest fallback). See module docstring for the calibration.
        card_gmax = _fnum("VIDLORE_QA_CARD_GRAPHIC_MAX",
                          _DEFAULT_CARD_GRAPHIC_MAX)

        tmpdir = Path(tempfile.mkdtemp(prefix="vidlore_relqa_"))
        sampled = 0
        flags = []
        for i, ts in enumerate(times):
            png = tmpdir / f"f{i:04d}.png"
            if not _extract_frame(ff, mp4_path, ts, png):
                continue
            # Per-frame designed-graphic / text-board probe (keyword-independent)
            # + RC5.1 UI-geometry signal (game/software interface screenshot).
            g = VR.graphic_signal(str(png), is_video=False)
            sampled += 1
            looks_ui = bool(g.get("looks_ui_screenshot"))
            if g.get("looks_designed") or looks_ui:
                gd = g.get("graphic_dom")
                ui_geom = g.get("ui_geom")
                in_card = _in_card_window(ts, card_windows)
                # (1) Inside a resolved engine/MG card window → EXEMPT (both the
                #     designed-graphic AND the ui-geometry signals).
                #     RC5.1 CALIBRATION FIX: a card window is a time range the
                #     engine fills with its OWN generated card. A *fetched* asset —
                #     the only way a real game/software/HUD screenshot can enter a
                #     render — is placed on FOOTAGE beats, never on a card window,
                #     and any fetched card BACKGROUND already passed the fetch-time
                #     relevance gate. The earlier assumption that engine cards are
                #     "clean cartographic renders with no interface chrome" was
                #     empirically WRONG: redacted_document / classified dossier /
                #     evidence-board / stat-dashboard / diagram cards legitimately
                #     carry axis-aligned panel geometry (redaction bars, document
                #     borders, stamps, node grids) that trips the ui_geom probe
                #     (~0.6) — observed on the Iran-Iraq classified card at ~116s.
                #     A genuine game-UI leak still surfaces on a footage beat
                #     OUTSIDE every window, where full ui_geom sensitivity is kept
                #     below. VIDLORE_QA_CARD_UI_STRICT=1 restores the old
                #     (over-eager) behaviour where ui_geom fails even in a card.
                _ui_strict = os.environ.get("VIDLORE_QA_CARD_UI_STRICT", "0") == "1"
                if in_card and (not looks_ui or not _ui_strict):
                    try:
                        png.unlink()
                    except Exception:                          # noqa: BLE001
                        pass
                    continue
                # (2a) UI-GEOMETRY HARD REJECT (RC5.1 FIX 3). A game / software /
                #      tactical interface screenshot is wrong footage for any
                #      documentary beat and is caught regardless of the card-aware
                #      ceiling — its give-away is the dense axis-aligned panel
                #      geometry, not its (map-like) CLIP semantics. By default it
                #      fires only OUTSIDE a card window (engine cards are exempted
                #      at (1) above); with VIDLORE_QA_CARD_UI_STRICT=1 it also
                #      fires inside a window on the UI signal.
                if looks_ui:
                    reason = (f"game/software UI screenshot — interface geometry "
                              f"(ui_geom={ui_geom}, graphic_dom={gd}) "
                              f"[{'in' if in_card else 'outside'} card window]")
                    flags.append({
                        "timestamp": round(float(ts), 2),
                        "scene": _scene_for_ts(ts, scene_starts),
                        "reason": reason,
                        "suggestion": _suggestion_for(reason),
                        "in_card_window": bool(in_card),
                    })
                    try:
                        png.unlink()
                    except Exception:                          # noqa: BLE001
                        pass
                    continue
                # (2b) Designed-graphic, outside every card window. Pick the
                #      effective ceiling:
                #       * card-rich render ⇒ legacy/injected engine cards (not in
                #         any manifest, outside the MG windows) live here too, so
                #         use the raised card-aware ceiling: only clearly junk-
                #         grade frames (graphic_dom well above any engine card)
                #         fail; the engine's own cards (~0.04-0.08) are spared.
                #       * no cards anywhere in the render ⇒ a designed-graphic
                #         frame is genuinely anomalous, so keep the strict gate
                #         threshold (full junk sensitivity on a pure-footage doc).
                eff_max = card_gmax if card_rich else gmax
                try:
                    gd_val = float(gd)
                except (TypeError, ValueError):
                    gd_val = None
                # `looks_designed` already means gd>gmax. Re-apply the effective
                # ceiling so the raised-fallback path can spare card-like frames.
                if gd_val is not None and gd_val <= eff_max:
                    try:
                        png.unlink()
                    except Exception:                          # noqa: BLE001
                        pass
                    continue
                reason = (f"designed-graphic/text-board (graphic_dom={gd} > "
                          f"{eff_max}) [outside card window]")
                flags.append({
                    "timestamp": round(float(ts), 2),
                    "scene": _scene_for_ts(ts, scene_starts),
                    "reason": reason,
                    "suggestion": _suggestion_for(reason),
                    "in_card_window": False,
                })
            try:
                png.unlink()
            except Exception:                                  # noqa: BLE001
                pass

        result["sampled"] = sampled
        result["flags"] = flags
        base_verdict = "FAIL_RELEVANCE_QA" if flags else "PASS"
        result["verdict"] = _finalize_verdict(base_verdict, flags, is_stale)
        return result
    except Exception as e:                                     # noqa: BLE001
        # Reporter, never a gate: degrade to PASS + note. BUT a stale-output
        # mismatch (recorded before the exception) still wins — a wrong-file scan
        # must never be hidden behind a degraded PASS.
        result["verdict"] = _finalize_verdict(
            "PASS", [], bool(result.get("stale_output")))
        result["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return result
    finally:
        if tmpdir is not None:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:                                  # noqa: BLE001
                pass


def _main(argv) -> int:
    """Standalone CLI: `python -m vidlore.relevance_qa <mp4> [render_meta.json]
    [manifest.json] [motion_graphics_manifest.json]
    [--expected-sha256=HEX] [--expected-path=PATH]`. Prints the verdict + flags
    as JSON. Exit code 0 = PASS, 2 = FAIL_RELEVANCE_QA / FAIL_STALE_OUTPUT (so a
    CI harness can gate on either), 0 on a degraded/errored sweep. The MG manifest
    is also auto-discovered next to the MP4 when not given, so a bare `<mp4>` call
    is motion-graphics aware out of the box.

    --expected-sha256 is the FINAL-EXPORT hash the user receives; if the scanned
    MP4 does not match it the sweep returns FAIL_STALE_OUTPUT (RC5.1)."""
    import json
    # Pull the optional --expected-* flags out first, leaving positionals.
    expected_sha256 = None
    expected_path = None
    positionals = []
    for a in argv:
        s = str(a)
        if s.startswith("--expected-sha256="):
            expected_sha256 = s.split("=", 1)[1].strip() or None
        elif s.startswith("--expected-path="):
            expected_path = s.split("=", 1)[1].strip() or None
        else:
            positionals.append(s)
    argv = positionals
    if not argv:
        print("usage: python -m vidlore.relevance_qa <mp4> [render_meta.json] "
              "[manifest.json] [motion_graphics_manifest.json] "
              "[--expected-sha256=HEX] [--expected-path=PATH]")
        return 1
    mp4 = argv[0]
    meta = {}
    manifest = None
    mg_manifest = None
    # Auto-discover a sibling render_meta.json when not given explicitly.
    if len(argv) > 1 and argv[1]:
        try:
            meta = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            meta = {}
    else:
        sib = Path(mp4).parent / "render_meta.json"
        if sib.exists():
            try:
                meta = json.loads(sib.read_text(encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                meta = {}
    if len(argv) > 2 and argv[2]:
        try:
            manifest = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            manifest = None
    # Optional explicit MG manifest path (4th arg); otherwise sweep() auto-finds
    # it next to the MP4.
    if len(argv) > 3 and argv[3]:
        mg_manifest = argv[3]
    res = sweep(mp4, meta, manifest=manifest, mg_manifest=mg_manifest,
                expected_sha256=expected_sha256, expected_path=expected_path)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 2 if res.get("verdict") in ("FAIL_RELEVANCE_QA",
                                       "FAIL_STALE_OUTPUT") else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
