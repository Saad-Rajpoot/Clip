#!/usr/bin/env python3
"""
Vidlore Autonomous Documentary Intelligence Improvement Loop
=============================================================

Self-running system that:
  1. Studies premium documentary YouTube videos
  2. Extracts editing/pacing/audio DNA patterns
  3. Compares against Vidlore's own renders
  4. Finds weaknesses → generates improvements
  5. Safely implements + tests them
  6. Reports results

Usage:
    # Run a full daily cycle (analyze + score + report)
    python tools/autonomous_research_loop.py --mode daily

    # Analyze a single YouTube video and update DNA library
    python tools/autonomous_research_loop.py --mode single_video --video VIDEO_ID

    # Score an existing render
    python tools/autonomous_research_loop.py --mode benchmark --mp4 path/to/out.mp4 --script path/to/script.json

    # Run the next pending improvement (safe, with backup+revert)
    python tools/autonomous_research_loop.py --mode improve

    # Pilot study: analyze 3 reference videos + generate first report
    python tools/autonomous_research_loop.py --mode pilot

    # Run continuously: analyze → improve → report → sleep → repeat (NEVER stops except for critical blockers)
    python tools/autonomous_research_loop.py --mode continuous
    # Or via wrapper:  ./run_continuous_improvement_loop.sh

Environment variables:
    ANTHROPIC_API_KEY               required for LLM analysis
    RESEARCH_MAX_VIDEOS             max new videos to analyze per run  (default: 3)
    RESEARCH_MAX_VIDEOS_PER_CYCLE   max videos per continuous cycle    (default: 3)
    RESEARCH_MAX_IMPROVEMENTS_PER_CYCLE  max improvements per cycle    (default: 1)
    RESEARCH_SLEEP_BETWEEN_CYCLES_MIN    sleep between cycles (min)    (default: 10)
    RESEARCH_DAILY_BUDGET_MODE      safe | normal (default: safe)
    RESEARCH_CACHE=1                use cached analysis only (no network/LLM calls)
    VIDLORE_PORT                    portal port (default: 5050)

Continuous mode stop conditions (ONLY these stop the loop):
    - research/STOP file exists   (touch research/STOP to halt gracefully)
    - Disk < 2 GB free
    - ANTHROPIC_API_KEY missing   (skips LLM work, continues with cached/metadata)
    - Critical: render reliability < 6 on baseline (logs warning, skips improvement)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Research directories ──────────────────────────────────────────────────────
_RESEARCH = _ROOT / "research"
_REPORTS = _RESEARCH / "reports"
_DAILY_REPORTS = _RESEARCH / "daily_reports"
_REF_MANIFEST = _RESEARCH / "documentary_references" / "reference_manifest.json"
_DNA_LIBRARY = _RESEARCH / "documentary_dna_library.json"
_BACKLOG = _RESEARCH / "improvement_backlog.json"
_SCORES = _RESEARCH / "benchmark_scores.json"
_STATE_FILE = _RESEARCH / "loop_state.json"
_STOP_FILE = _RESEARCH / "STOP"               # touch this file to halt the loop

# ── Reference video queue (our study corpus) ──────────────────────────────────
# Premium documentary channels — mix of styles: narrative, investigative,
# historical, nature. The loop skips any ID already in the manifest, so the
# 3 confirmed-good IDs below (analyzed in the pilot) make Phase A a clean
# no-op until NEW, verified IDs are appended here. We deliberately list only
# IDs we have confirmed resolve (the earlier placeholder IDs 404'd via yt-dlp
# and produced empty analyses) — append fresh verified IDs to grow the corpus.
_REFERENCE_CHANNELS: list[dict] = [
    {
        "channel": "Johnny Harris",
        "style": "narrative_investigative",
        "seed_videos": [
            "2r5eKpptixo",  # JFK — cold open 112s, document highlights, 4:3 (analyzed)
        ],
        "notes": "Best-in-class cold opens, map overlays, silence-as-reveal",
    },
    {
        "channel": "Vox",
        "style": "explainer_cinematic",
        "seed_videos": [
            "1ZKBaRsP1gY",  # Xi Jinping — title delayed 99s, photo call-outs (analyzed)
        ],
        "notes": "Graphics restraint, strong narration, motivated maps",
    },
    {
        "channel": "LEMMiNO",
        "style": "mystery_atmospheric",
        "seed_videos": [
            "I2O7blSSzpI",  # Cicada 3301 — constant drift, UI sim, silence beats (analyzed)
        ],
        "notes": "Best pacing variance, atmospheric SFX, music arc",
    },
    {
        "channel": "Veritasium",
        "style": "cinematic_science",
        "seed_videos": [
            "IgF3OX8nT0w",  # The Most Powerful Computers You've Never Heard Of
            #               13M views, 4K+CC — distinct VISUAL genre (field
            #               cinematography + clean science animation + lab
            #               demos), absent from the corpus, transferable to
            #               Vidlore education/science niches.
        ],
        "notes": "Cinematic science: field shoots + clean explanatory "
                 "animation + demonstration; curiosity-gap hooks; restrained "
                 "on-screen text. A distinct visual genre from the "
                 "investigative/economics/crime corpus.",
    },
    # NOTE: RealLifeLore / True Crime Daily entries removed for now — their
    # placeholder IDs did not resolve. Append channel blocks with VERIFIED
    # video IDs to widen the study corpus (the loop auto-skips analyzed ones).
]

# ── Cost controls ─────────────────────────────────────────────────────────────
_MAX_VIDEOS_PER_RUN = int(os.environ.get("RESEARCH_MAX_VIDEOS", "3"))
_CACHE_ONLY = os.environ.get("RESEARCH_CACHE", "0") == "1"

# ── Continuous mode controls (all overridable via env) ────────────────────────
_C_MAX_VIDEOS     = int(os.environ.get("RESEARCH_MAX_VIDEOS_PER_CYCLE", "3"))
_C_MAX_IMPROVE    = int(os.environ.get("RESEARCH_MAX_IMPROVEMENTS_PER_CYCLE", "1"))
_C_SLEEP_MIN      = int(os.environ.get("RESEARCH_SLEEP_BETWEEN_CYCLES_MIN", "10"))
_C_BUDGET_MODE    = os.environ.get("RESEARCH_DAILY_BUDGET_MODE", "safe")
_C_MIN_DISK_GB    = 2.0   # stop if less than 2 GB free before a render
_C_MIN_RELIABILITY = 6.0  # skip improvement if baseline render reliability < this


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_manifest() -> dict:
    return _load_json(_REF_MANIFEST, {"videos": []})


def _already_analyzed(video_id: str) -> bool:
    """Check if video_id is already in the reference manifest."""
    manifest = _load_manifest()
    return any(v.get("video_id") == video_id for v in manifest.get("videos", []))


def _load_backlog() -> list:
    return _load_json(_BACKLOG, [])


def _save_backlog(items: list) -> None:
    _save_json(_BACKLOG, items)


# ── Phase 1: Research — Analyze reference videos ──────────────────────────────

def analyze_reference_videos(max_videos: int = _MAX_VIDEOS_PER_RUN) -> list[dict]:
    """Analyze up to max_videos reference docs. Returns list of analysis notes."""
    from tools.research_engine import analyze_youtube_video, update_dna_library

    # Load config for Anthropic key
    cfg = _load_cfg()

    # Flatten seed video list, skip already-analyzed
    queue: list[tuple[str, dict]] = []
    for ch in _REFERENCE_CHANNELS:
        for vid_id in ch.get("seed_videos", []):
            if not _already_analyzed(vid_id):
                queue.append((vid_id, ch))

    if not queue:
        print("[research] All reference videos already analyzed. Nothing new to study.")
        return []

    results = []
    analyzed_count = 0

    for video_id, channel_info in queue:
        if analyzed_count >= max_videos:
            print(f"[research] Budget reached ({max_videos} videos). Stopping.")
            break

        print(f"\n[research] === Analyzing: {video_id} ({channel_info['channel']}) ===")
        try:
            notes = analyze_youtube_video(video_id, cfg)
            if notes:
                # Merge channel context into notes
                notes["channel"] = channel_info["channel"]
                notes["channel_style"] = channel_info["style"]
                notes["channel_notes"] = channel_info.get("notes", "")

                # Update DNA library with extracted patterns
                try:
                    from tools.research_engine import extract_editing_patterns
                    patterns = extract_editing_patterns(notes)
                    update_dna_library(patterns, _DNA_LIBRARY)
                    print(f"[research] DNA library updated with patterns from {video_id}")
                except Exception as e:
                    print(f"[research] Warning: could not extract patterns: {e}")

                # Update reference manifest
                _add_to_manifest(video_id, notes, channel_info)

                results.append(notes)
                analyzed_count += 1
            else:
                print(f"[research] No notes returned for {video_id} — skipping")
        except Exception as e:
            print(f"[research] Error analyzing {video_id}: {e}")
            continue

    print(f"\n[research] Analyzed {analyzed_count} new reference videos.")
    return results


def _add_to_manifest(video_id: str, notes: dict, channel_info: dict) -> None:
    """Add analyzed video to the reference manifest."""
    manifest = _load_manifest()
    entry = {
        "video_id": video_id,
        "channel": channel_info["channel"],
        "style": channel_info["style"],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "title": notes.get("title", ""),
        "duration_s": notes.get("duration_s", 0),
        "key_patterns": notes.get("key_patterns", []),
        "top_strengths": notes.get("top_strengths", []),
        "notes_file": f"video_notes/{video_id}.json",
    }
    videos = manifest.get("videos", [])
    # Replace if already exists
    videos = [v for v in videos if v.get("video_id") != video_id]
    videos.append(entry)
    manifest["videos"] = videos
    manifest["total_analyzed"] = len(videos)
    manifest["_last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_json(_REF_MANIFEST, manifest)
    print(f"[research] Manifest updated: {len(videos)} total videos analyzed")


# ── Phase 2: Benchmark — Score a Vidlore render ─────────────────────────────

def score_render(mp4_path: Path, script_json: Path, work_dir: Path | None = None) -> dict:
    """Score a Vidlore render and save results."""
    from tools.benchmark_engine import score_render as _score
    work_dir = work_dir or mp4_path.parent
    print(f"\n[benchmark] Scoring: {mp4_path.name}")
    result = _score(mp4_path, script_json, work_dir)
    scores = result.get("scores", {})
    overall = scores.get("overall_quality", 0)

    print("\n┌─────────────────────────────────────────────────┐")
    print("│  Vidlore Render Quality Report                  │")
    print("├─────────────────────────────────────────────────┤")
    _dims = [
        ("footage_relevance",   "Footage relevance"),
        ("scene_script_match",  "Scene/script match"),
        ("documentary_realism", "Documentary realism"),
        ("pacing",              "Pacing"),
        ("visual_variation",    "Visual variation"),
        ("motion_quality",      "Motion quality"),
        ("graphics_quality",    "Graphics quality"),
        ("text_restraint",      "Text restraint"),
        ("audio_mix",           "Audio mix"),
        ("sfx_restraint",       "SFX restraint"),
        ("music_psychology",    "Music psychology"),
        ("no_template_feel",    "No template feel"),
        ("render_reliability",  "Render reliability"),
    ]
    for key, label in _dims:
        val = scores.get(key, 0)
        bar_n = int(round(val))
        bar = "█" * bar_n + "░" * (10 - bar_n)
        print(f"│  {label:<22} {val:>4.1f}  {bar}  │")
    print("├─────────────────────────────────────────────────┤")
    print(f"│  ★ OVERALL QUALITY          {overall:>4.1f}               │")
    print("└─────────────────────────────────────────────────┘\n")

    return result


# ── Phase 3: Improvement — Find and apply next improvement ────────────────────

def run_next_improvement() -> dict:
    """Find the next pending improvement, apply it safely, test, keep or revert."""
    from tools.improvement_engine import run_improvement_cycle

    backlog = _load_backlog()
    pending = [i for i in backlog if i.get("status") == "pending"]

    if not pending:
        print("[improve] No pending improvements in backlog.")
        return {"status": "no_pending"}

    # Sort by priority (lowest number = highest priority)
    pending.sort(key=lambda x: x.get("priority", 99))
    item = pending[0]
    print(f"\n[improve] Running improvement: {item['id']} — {item['title']}")
    print(f"[improve] Hypothesis: {item['hypothesis']}")
    print(f"[improve] Expected: {item['expected_improvement']}")

    # NOTE: run_improvement_cycle() picks the next pending item itself via
    # get_next_improvement(); its first positional arg is `dry_run`, so we
    # MUST call it with no args (passing `item` would silently dry-run).
    result = run_improvement_cycle()
    return result


# ── Phase 4: Daily Report ─────────────────────────────────────────────────────

def generate_daily_report() -> str:
    """Generate a daily report of research findings, scores, and improvements."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = _REPORTS / f"daily_{today}.md"
    _REPORTS.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    backlog = _load_backlog()
    scores_data = _load_json(_SCORES, {})

    # Recent video analyses
    recent_videos = sorted(
        manifest.get("videos", []),
        key=lambda v: v.get("analyzed_at", ""),
        reverse=True,
    )[:5]

    # Backlog summary
    pending = [i for i in backlog if i.get("status") == "pending"]
    implemented = [i for i in backlog if i.get("status") == "implemented"]
    in_progress = [i for i in backlog if i.get("status") == "in_progress"]

    # Latest benchmark score
    latest_score = None
    latest_slug = None
    if scores_data:
        # Handle both: keyed-by-slug format and flat {renders:[]} schema
        all_entries = []
        for slug, entries in scores_data.items():
            if slug.startswith("_"):
                continue        # schema metadata key
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict):
                        all_entries.append((e.get("scored_at", ""), slug, e))
            elif slug == "renders" and isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict):
                        all_entries.append((e.get("scored_at", ""), "render", e))
        if all_entries:
            all_entries.sort(reverse=True)
            latest_score = all_entries[0][2]
            latest_slug = all_entries[0][1]

    # Build report
    lines = [
        f"# Vidlore Documentary Intelligence Report — {today}",
        "",
        "## Executive Summary",
        "",
        f"- Reference videos analyzed: **{manifest.get('total_analyzed', 0)}**",
        f"- Improvements implemented: **{len(implemented)}**",
        f"- Improvements pending: **{len(pending)}**",
        f"- Improvements in progress: **{len(in_progress)}**",
        "",
    ]

    if latest_score:
        sc = latest_score.get("scores", {})
        overall = sc.get("overall_quality", 0)
        lines += [
            f"## Latest Render Score (slug: `{latest_slug}`)",
            "",
            f"| Dimension | Score |",
            f"|-----------|-------|",
        ]
        for dim in ["footage_relevance", "documentary_realism", "pacing",
                    "audio_mix", "no_template_feel", "overall_quality"]:
            lines.append(f"| {dim} | {sc.get(dim, 'N/A')} |")
        lines.append("")

    if recent_videos:
        lines += [
            "## Recently Analyzed Reference Videos",
            "",
        ]
        for v in recent_videos:
            lines.append(f"### {v.get('channel')} — `{v.get('video_id')}`")
            if v.get("title"):
                lines.append(f"**Title:** {v['title']}")
            if v.get("key_patterns"):
                lines.append(f"**Key patterns:** {', '.join(v['key_patterns'])}")
            if v.get("top_strengths"):
                lines.append(f"**Strengths:** {', '.join(v['top_strengths'])}")
            lines.append("")

    if implemented:
        lines += [
            "## Implemented Improvements",
            "",
        ]
        for imp in implemented:
            lines.append(f"- **{imp['id']}** — {imp['title']}")
            if imp.get("result"):
                lines.append(f"  - Result: {imp['result'][:200]}")
        lines.append("")

    if pending:
        lines += [
            "## Pending Improvements (by priority)",
            "",
        ]
        for imp in sorted(pending, key=lambda x: x.get("priority", 99)):
            expected = imp.get("expected_improvement", {})
            lines.append(
                f"- **{imp['id']}** (P{imp.get('priority', '?')}) — {imp['title']}  "
                f"*Expected: {expected.get('dimension', '?')} {expected.get('delta', '?')}*"
            )
        lines.append("")

    # DNA Library snapshot
    dna = _load_json(_DNA_LIBRARY, {})
    if dna:
        lines += [
            "## Documentary DNA Library",
            "",
            f"Last updated: {dna.get('_last_updated', 'unknown')}",
            "",
        ]
        for section_key in ["hook_patterns", "pacing_patterns", "graphics_patterns",
                             "sound_patterns", "cinematography_patterns"]:
            section = dna.get(section_key, {})
            entries = section.get("entries", [])
            implemented_pcts = [
                e for e in entries
                if e.get("vidlore_status", "").startswith("implemented")
            ]
            lines.append(
                f"- **{section_key}**: {len(entries)} patterns, "
                f"{len(implemented_pcts)} implemented"
            )
        lines.append("")

    lines += [
        "---",
        f"*Generated automatically by Vidlore Autonomous Research Loop — {today}*",
    ]

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\n[report] Daily report saved → {report_path}")
    print("\n" + "=" * 60)
    print(report_content[:2000])
    if len(report_content) > 2000:
        print(f"... (truncated, full report at {report_path})")
    print("=" * 60)

    return str(report_path)


# ── Continuous loop — state, safety, daily reports ────────────────────────────

def _load_state() -> dict:
    """Load the persistent loop state (cycle counters, daily-report date)."""
    return _load_json(_STATE_FILE, {
        "cycle_count": 0,
        "started_at": None,
        "last_cycle_at": None,
        "last_daily_report_date": None,
        "videos_analyzed_total": 0,
        "improvements_implemented_total": 0,
        "improvements_reverted_total": 0,
        "improvements_skipped_total": 0,
        "last_status": "init",
        "history": [],          # short rolling log of cycle outcomes
    })


def _save_state(state: dict) -> None:
    _save_json(_STATE_FILE, state)


def _disk_free_gb(path: Path = _ROOT) -> float:
    """Free disk space in GB at `path` (used as a render safety gate)."""
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except Exception:
        return 999.0          # if we can't tell, don't block on it


def _check_stop_conditions() -> tuple[bool, str]:
    """Return (should_stop, reason). ONLY hard blockers stop the loop."""
    # 1. Explicit STOP sentinel — user touched research/STOP
    if _STOP_FILE.exists():
        return True, "STOP sentinel file present (research/STOP)"
    # 2. Disk space — we render video, need headroom
    free = _disk_free_gb()
    if free < _C_MIN_DISK_GB:
        return True, f"Low disk space: {free:.1f} GB free (< {_C_MIN_DISK_GB} GB)"
    return False, ""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Continuous daily report (richer than the legacy reports/ one) ─────────────

def generate_continuous_daily_report(state: dict, cycle_results: list[dict]) -> str:
    """Write research/daily_reports/YYYY-MM-DD.md summarising the day's work."""
    _DAILY_REPORTS.mkdir(parents=True, exist_ok=True)
    today = _today()
    report_path = _DAILY_REPORTS / f"{today}.md"

    manifest = _load_manifest()
    backlog = _load_backlog()
    pending = sorted(
        [i for i in backlog if i.get("status") == "pending"],
        key=lambda x: x.get("priority", 99),
    )
    implemented = [i for i in backlog if i.get("status") == "implemented"]
    reverted = [i for i in backlog if i.get("status") == "reverted"]

    # Pull what happened in TODAY's cycles from state history.
    todays = [h for h in state.get("history", []) if h.get("at", "").startswith(today)]

    lines = [
        f"# Vidlore Autonomous Loop — Daily Report {today}",
        "",
        "## Loop status",
        "",
        f"- Total cycles run: **{state.get('cycle_count', 0)}**",
        f"- Cycles today: **{len(todays)}**",
        f"- Reference videos analyzed (all-time): **{manifest.get('total_analyzed', 0)}**",
        f"- Improvements implemented (all-time): **{state.get('improvements_implemented_total', 0)}**",
        f"- Improvements reverted (all-time): **{state.get('improvements_reverted_total', 0)}**",
        f"- Improvements skipped (all-time): **{state.get('improvements_skipped_total', 0)}**",
        f"- Budget mode: `{_C_BUDGET_MODE}` "
        f"(max {_C_MAX_VIDEOS} videos / {_C_MAX_IMPROVE} improvement(s) per cycle, "
        f"sleep {_C_SLEEP_MIN} min)",
        "",
    ]

    # This cycle batch
    if cycle_results:
        lines += ["## This run's cycles", ""]
        for r in cycle_results:
            lines.append(
                f"- Cycle **#{r.get('cycle')}** — "
                f"videos: {r.get('videos_analyzed', 0)}, "
                f"improvement: {r.get('improvement_id', '—')} "
                f"→ **{r.get('improvement_status', 'none')}** "
                f"(Δ {r.get('improvement_delta', '—')})"
            )
        lines.append("")

    # Videos analyzed (recent)
    recent_videos = sorted(
        manifest.get("videos", []),
        key=lambda v: v.get("analyzed_at", ""),
        reverse=True,
    )[:6]
    if recent_videos:
        lines += ["## Recently analyzed reference videos", ""]
        for v in recent_videos:
            lines.append(
                f"- **{v.get('channel', '?')}** `{v.get('video_id')}` — "
                f"{v.get('title', '(untitled)')}"
            )
            if v.get("key_patterns"):
                lines.append(f"  - patterns: {', '.join(v['key_patterns'][:6])}")
        lines.append("")

    # Implemented improvements
    if implemented:
        lines += ["## Implemented improvements", ""]
        for imp in implemented:
            res = imp.get("result")
            note = ""
            if isinstance(res, dict):
                note = f" (Δ {res.get('delta', '?')})"
            elif isinstance(res, str):
                note = f" — {res[:120]}"
            lines.append(f"- **{imp['id']}** — {imp['title']}{note}")
        lines.append("")

    if reverted:
        lines += ["## Reverted (did not improve score)", ""]
        for imp in reverted:
            lines.append(f"- **{imp['id']}** — {imp['title']}")
        lines.append("")

    # Next up
    lines += ["## Next improvements planned (by priority)", ""]
    if pending:
        for imp in pending[:8]:
            exp = imp.get("expected_improvement", {})
            lines.append(
                f"- **{imp['id']}** (P{imp.get('priority', '?')}) — {imp['title']} "
                f"*[{exp.get('dimension', '?')} {exp.get('delta', '?')}]*"
            )
    else:
        lines.append("- _Backlog empty — loop will keep studying references and "
                     "auto-generating new improvement ideas._")
    lines.append("")

    # Anything needed from the user
    blockers = state.get("pending_user_input", [])
    lines += ["## Needs from user", ""]
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- Nothing — loop is running autonomously. "
                     "Touch `research/STOP` to halt gracefully.")
    lines.append("")

    lines += [
        "---",
        f"*Vidlore Autonomous Research Loop — continuous mode — {today}*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[daily-report] → {report_path}")
    return str(report_path)


# ── One continuous cycle ──────────────────────────────────────────────────────

def run_one_cycle(state: dict) -> dict:
    """Execute exactly one research→improve cycle. Returns a result summary.

    Each cycle:
      1. analyze up to _C_MAX_VIDEOS new reference videos (skips done ones)
      2. update DNA library (handled inside analyze_reference_videos)
      3. run up to _C_MAX_IMPROVE pending improvements (safe: backup→test→keep/revert)
      4. return a per-cycle summary
    Never raises — any failure is captured and returned so the loop survives.
    """
    cycle_no = state.get("cycle_count", 0) + 1
    result = {
        "cycle": cycle_no,
        "at": datetime.now(timezone.utc).isoformat(),
        "videos_analyzed": 0,
        "improvement_id": None,
        "improvement_status": "none",
        "improvement_delta": None,
        "errors": [],
    }

    # ── Phase A: study references ─────────────────────────────────────────────
    print(f"\n[cycle {cycle_no}] Phase A — analyzing reference videos "
          f"(max {_C_MAX_VIDEOS})...")
    try:
        analyzed = analyze_reference_videos(max_videos=_C_MAX_VIDEOS)
        result["videos_analyzed"] = len(analyzed)
        state["videos_analyzed_total"] = (
            state.get("videos_analyzed_total", 0) + len(analyzed)
        )
    except Exception as e:                                       # noqa: BLE001
        msg = f"video analysis error: {e}"
        print(f"[cycle {cycle_no}] {msg}")
        result["errors"].append(msg)

    # ── Phase B: improvement (safe, with backup + before/after + revert) ──────
    backlog = _load_backlog()
    pending = sorted(
        [i for i in backlog if i.get("status") == "pending"],
        key=lambda x: x.get("priority", 99),
    )
    # An item is machine-applicable only if it carries an exact code diff
    # (change_description_structured = {"old":..., "new":...} or {"content":...}).
    # Items with only a natural-language description are DEFERRED — flagged for
    # a human/agent to author the structured diff — never destructively skipped.
    def _is_applicable(it: dict) -> bool:
        cds = it.get("change_description_structured") or {}
        return bool(cds.get("content") or (cds.get("old") and cds.get("new")))

    applicable = [i for i in pending if _is_applicable(i)]
    deferred = [i for i in pending if not _is_applicable(i)]

    if deferred:
        notes = state.get("pending_user_input", [])
        for it in deferred[: _C_MAX_IMPROVE + 2]:
            msg = (f"{it['id']} ({it['title']}) needs a structured code diff "
                   f"(change_description_structured) before the loop can auto-apply it.")
            if msg not in notes:
                notes.append(msg)
        state["pending_user_input"] = notes[-12:]

    if not applicable:
        print(f"[cycle {cycle_no}] Phase B — no machine-applicable improvements "
              f"({len(deferred)} deferred awaiting structured diffs). "
              "Studying references only this cycle.")
    else:
        improved_this_cycle = 0
        for item in applicable:
            if improved_this_cycle >= _C_MAX_IMPROVE:
                break
            iid = item["id"]
            print(f"[cycle {cycle_no}] Phase B — improvement {iid}: {item['title']}")

            # Pre-render disk safety
            free = _disk_free_gb()
            if free < _C_MIN_DISK_GB:
                msg = f"skipping improvement {iid}: low disk {free:.1f}GB"
                print(f"[cycle {cycle_no}] {msg}")
                result["errors"].append(msg)
                break

            try:
                from tools.improvement_engine import run_improvement_cycle
                cyc = run_improvement_cycle()       # picks highest-priority pending
            except Exception as e:                                  # noqa: BLE001
                msg = f"improvement {iid} crashed: {e}"
                print(f"[cycle {cycle_no}] {msg}")
                result["errors"].append(msg)
                state["improvements_skipped_total"] = (
                    state.get("improvements_skipped_total", 0) + 1
                )
                break

            status = cyc.get("status", "unknown")
            result["improvement_id"] = cyc.get("item_id", iid)
            result["improvement_status"] = status
            ev = cyc.get("evaluation", {})
            if ev:
                result["improvement_delta"] = ev.get("delta")

            if status == "implemented":
                state["improvements_implemented_total"] = (
                    state.get("improvements_implemented_total", 0) + 1
                )
            elif status == "reverted":
                state["improvements_reverted_total"] = (
                    state.get("improvements_reverted_total", 0) + 1
                )
            else:
                state["improvements_skipped_total"] = (
                    state.get("improvements_skipped_total", 0) + 1
                )

            improved_this_cycle += 1
            # Whatever the outcome, the item is no longer "pending" (the
            # engine marked it implemented/reverted/skipped), so the next
            # cycle naturally moves on — true resume support.
            break

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    state["cycle_count"] = cycle_no
    state["last_cycle_at"] = result["at"]
    state["last_status"] = result["improvement_status"]
    hist = state.get("history", [])
    hist.append({
        "cycle": cycle_no,
        "at": result["at"],
        "videos_analyzed": result["videos_analyzed"],
        "improvement_id": result["improvement_id"],
        "improvement_status": result["improvement_status"],
        "improvement_delta": result["improvement_delta"],
    })
    state["history"] = hist[-50:]    # keep last 50 cycles
    _save_state(state)
    return result


# ── Continuous mode entry point ───────────────────────────────────────────────

def run_continuous(max_cycles: int | None = None) -> None:
    """Run the research→improve→report loop forever (or `max_cycles` times).

    Stops ONLY for critical blockers:
      - research/STOP sentinel present
      - disk < 2 GB free
    Everything else (a failed render, a reverted improvement, a missing
    transcript) is logged and the loop continues to the next cycle.
    """
    print("\n" + "=" * 64)
    print("  Vidlore Autonomous Research Loop — CONTINUOUS MODE")
    print(f"  start: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  budget: {_C_BUDGET_MODE} | {_C_MAX_VIDEOS} vids/cycle | "
          f"{_C_MAX_IMPROVE} improve/cycle | sleep {_C_SLEEP_MIN}min")
    print(f"  stop: touch {_STOP_FILE.relative_to(_ROOT)}  (or Ctrl-C)")
    print("=" * 64)

    state = _load_state()
    if not state.get("started_at"):
        state["started_at"] = datetime.now(timezone.utc).isoformat()

    # Soft warning (NOT a stop) if no API key — we degrade to cached/metadata.
    cfg = _load_cfg()
    if not cfg.get("anthropic_api_key"):
        print("[continuous] NOTE: no ANTHROPIC_API_KEY — video analysis will be "
              "metadata/cached only; improvement tests still run.")
        notes = state.get("pending_user_input", [])
        if "Set ANTHROPIC_API_KEY for full LLM video analysis." not in notes:
            notes.append("Set ANTHROPIC_API_KEY for full LLM video analysis.")
        state["pending_user_input"] = notes
        _save_state(state)

    this_run_cycles: list[dict] = []
    n = 0
    try:
        while True:
            # Hard stop gate (checked every cycle)
            stop, reason = _check_stop_conditions()
            if stop:
                print(f"\n[continuous] STOP: {reason}")
                state["last_status"] = f"stopped: {reason}"
                _save_state(state)
                if _STOP_FILE.exists():
                    print("[continuous] (remove research/STOP to resume.)")
                break

            # Run one full cycle (never raises)
            res = run_one_cycle(state)
            this_run_cycles.append(res)
            n += 1

            print(f"[continuous] cycle {res['cycle']} done — "
                  f"videos:{res['videos_analyzed']} "
                  f"improvement:{res['improvement_id']}→{res['improvement_status']}")
            if res["errors"]:
                for e in res["errors"]:
                    print(f"[continuous]   note: {e}")

            # Daily report — once per UTC day
            if state.get("last_daily_report_date") != _today():
                generate_continuous_daily_report(state, this_run_cycles)
                state["last_daily_report_date"] = _today()
                _save_state(state)

            # Bounded run support (smoke tests)
            if max_cycles is not None and n >= max_cycles:
                print(f"\n[continuous] reached max_cycles={max_cycles} — exiting.")
                # always emit a fresh report on bounded exit
                generate_continuous_daily_report(state, this_run_cycles)
                break

            # Sleep between cycles (respect cost controls). Re-check STOP often.
            sleep_total = max(0, _C_SLEEP_MIN * 60)
            print(f"[continuous] sleeping {_C_SLEEP_MIN} min before next cycle "
                  f"(touch research/STOP to halt)...")
            slept = 0
            while slept < sleep_total:
                if _STOP_FILE.exists():
                    break
                step = min(15, sleep_total - slept)
                time.sleep(step)
                slept += step
    except KeyboardInterrupt:
        print("\n[continuous] interrupted by user (Ctrl-C). State saved.")
        state["last_status"] = "interrupted"
        _save_state(state)

    print("\n[continuous] loop ended. "
          f"Cycles this run: {n}. Total: {state.get('cycle_count', 0)}.")


# ── Phase 5: Pilot — First 3 reference videos ────────────────────────────────

def run_pilot() -> None:
    """Analyze 3 seed reference videos + generate first daily report.

    This is the entry point for the very first run. Uses budget-controlled
    LLM calls (max 3 videos, transcript-only if no API key).
    """
    print("\n" + "=" * 60)
    print("  Vidlore Documentary Intelligence — PILOT RUN")
    print("  Analyzing 3 reference videos to seed DNA library")
    print("=" * 60 + "\n")

    cfg = _load_cfg()
    if not cfg.get("anthropic_api_key"):
        print("[pilot] WARNING: No ANTHROPIC_API_KEY set. Analysis will be metadata-only.")
        print("[pilot] Set ANTHROPIC_API_KEY in .env or environment for full LLM analysis.\n")

    # Step 1: Analyze reference videos
    print("[pilot] STEP 1: Analyzing reference videos...")
    results = analyze_reference_videos(max_videos=3)
    print(f"[pilot] Analyzed {len(results)} videos.\n")

    # Step 2: Generate report
    print("[pilot] STEP 2: Generating daily report...")
    report_path = generate_daily_report()
    print(f"[pilot] Report saved: {report_path}\n")

    # Step 3: Show next actions
    backlog = _load_backlog()
    pending = [i for i in backlog if i.get("status") == "pending"]
    pending.sort(key=lambda x: x.get("priority", 99))

    print("\n[pilot] STEP 3: Next recommended actions:")
    print(f"  - {len(pending)} improvements pending")
    if pending:
        top = pending[0]
        print(f"  - Top priority: {top['id']} — {top['title']}")
        print(f"    Run: python tools/autonomous_research_loop.py --mode improve")
    print(f"\n[pilot] Pilot complete. Run --mode daily to continue the improvement loop.")


# ── Daily mode ────────────────────────────────────────────────────────────────

def run_daily() -> None:
    """Full daily cycle: research → (optional improve) → report."""
    print("\n" + "=" * 60)
    print("  Vidlore Documentary Intelligence — DAILY CYCLE")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60 + "\n")

    # Step 1: Study reference videos
    print("[daily] STEP 1: Analyzing reference videos...")
    results = analyze_reference_videos(max_videos=_MAX_VIDEOS_PER_RUN)
    print(f"[daily] Studied {len(results)} new videos.\n")

    # Step 2: Generate report
    print("[daily] STEP 2: Generating daily report...")
    generate_daily_report()


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    """Load Anthropic API key from .env or environment."""
    env_file = _ROOT / ".env"
    cfg = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")

    api_key = (
        cfg.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    model = (
        cfg.get("ANTHROPIC_MODEL")
        or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    )

    return {
        "anthropic_api_key": api_key,
        "anthropic_model": model,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Vidlore Autonomous Documentary Intelligence Improvement Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["daily", "single_video", "benchmark", "improve", "pilot",
                 "report", "continuous"],
        default="pilot",
        help="Operating mode (default: pilot)",
    )
    p.add_argument("--max-cycles", type=int, default=None,
                   help="Stop continuous mode after N cycles (default: forever)")
    p.add_argument("--video", metavar="VIDEO_ID",
                   help="YouTube video ID (for --mode single_video)")
    p.add_argument("--mp4", metavar="PATH",
                   help="MP4 path to score (for --mode benchmark)")
    p.add_argument("--script", metavar="PATH",
                   help="script.json path (for --mode benchmark)")
    p.add_argument("--max-videos", type=int, default=_MAX_VIDEOS_PER_RUN,
                   help=f"Max videos to analyze per run (default: {_MAX_VIDEOS_PER_RUN})")
    args = p.parse_args()

    if args.mode == "pilot":
        run_pilot()

    elif args.mode == "continuous":
        run_continuous(max_cycles=args.max_cycles)

    elif args.mode == "daily":
        run_daily()

    elif args.mode == "single_video":
        if not args.video:
            print("Error: --video VIDEO_ID required for --mode single_video")
            return 1
        cfg = _load_cfg()
        from tools.research_engine import analyze_youtube_video, extract_editing_patterns, update_dna_library
        print(f"[single] Analyzing: {args.video}")
        notes = analyze_youtube_video(args.video, cfg)
        if notes:
            patterns = extract_editing_patterns(notes)
            update_dna_library(patterns, _DNA_LIBRARY)
            print(json.dumps(notes, indent=2, ensure_ascii=False)[:3000])
        else:
            print("[single] No notes returned.")

    elif args.mode == "benchmark":
        if not args.mp4:
            print("Error: --mp4 PATH required for --mode benchmark")
            return 1
        mp4 = Path(args.mp4)
        script = Path(args.script) if args.script else mp4.parent / "script.json"
        score_render(mp4, script)

    elif args.mode == "improve":
        result = run_next_improvement()
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "report":
        generate_daily_report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
