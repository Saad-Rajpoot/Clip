"""Incident advisor — when a render dies on an UNEXPECTED technical error, ask the reasoning
LLM (DeepSeek primary) what to do next, with full context — but only ever execute one of a
FIXED, safe action menu. The LLM classifies; deterministic code acts.

Why the menu is closed: an open-ended "LLM decides anything" would be unauditable and could
erode quality silently (skip a gate here, fudge an env there). Here the model can only pick:

    retry            — transient failure: re-run from stage checkpoints immediately
    retry_after_wait — throttle/network flavored: wait, then re-run from checkpoints
    abort            — real defect or unsafe to continue: fail loudly (the default)

Everything is logged to output/incident_report.json (context sent, model's answer, action
taken), capped at VIDLORE_CLIPSTUDIO_INCIDENT_MAX (2) interventions per render, and the
whole layer is bypassed by VIDLORE_CLIPSTUDIO_INCIDENT_ADVISOR=0 or an unreachable LLM —
in both cases behavior is exactly today's: the exception propagates.

CONTENT problems (wrong footage, blocked beats, caption issues) are NOT this module's job —
the verifier, QA gates and the self-heal loop own those with their own bounded machinery.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path

ACTIONS = ("retry", "retry_after_wait", "abort")


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _context(stage: str, exc: BaseException, proj=None, log_tail: str = "") -> dict:
    ctx = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc)[:600],
        "traceback_tail": "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))[-1200:],
        "log_tail": (log_tail or "")[-1200:],
    }
    try:
        if proj is not None:
            ctx["project"] = {
                "sources_ok": sum(1 for s in proj.sources if s.status == "ok"),
                "selections": len(proj.selections or []),
                "beats_with_stills": sum(1 for s in (proj.selections or [])
                                         if getattr(s, "image_path", "")),
            }
    except Exception:                                    # noqa: BLE001
        pass
    return ctx


def advise(stage: str, exc: BaseException, *, proj=None, log_tail: str = "",
           log=print) -> dict:
    """One advisory verdict: {"action", "wait_s", "why", "served"}. abort on any doubt."""
    verdict = {"action": "abort", "wait_s": 0, "why": "advisor unavailable", "served": ""}
    if not _env_on("VIDLORE_CLIPSTUDIO_INCIDENT_ADVISOR", "1"):
        verdict["why"] = "advisor disabled"
        return verdict
    ctx = _context(stage, exc, proj=proj, log_tail=log_tail)
    try:
        from . import llm
        out, meta = llm.complete_ex(
            system=(
                "You are the incident triager for an automated video-render pipeline. "
                "Given a technical failure's context, choose EXACTLY ONE action:\n"
                "  retry            — transient (network blip, race, flaky subprocess)\n"
                "  retry_after_wait — rate limit / throttle / service hiccup; include wait_s "
                "(30-300)\n"
                "  abort            — a code defect, bad input, disk/permission problem, or "
                "anything where retrying cannot help\n"
                "Reply ONLY JSON: {\"action\": ..., \"wait_s\": <int>, \"why\": \"<one line>\"}. "
                "When unsure, abort."),
            messages=[{"role": "user", "content": json.dumps(ctx, indent=1)}],
            max_tokens=120)
        m = re.search(r"\{.*\}", out or "", re.S)
        v = json.loads(m.group(0)) if m else {}
        action = str(v.get("action", "")).strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"non-menu action {action!r}")
        verdict = {"action": action,
                   "wait_s": max(0, min(300, int(v.get("wait_s", 0) or 0))),
                   "why": str(v.get("why", ""))[:200],
                   "served": str((meta or {}).get("served", ""))}
    except Exception as e:                               # noqa: BLE001 — no advice → abort
        verdict["why"] = f"advisor error: {str(e)[:120]}"
    _wait = f" (wait {verdict['wait_s']}s)" if verdict["action"] == "retry_after_wait" else ""
    log(f"incident-advisor [{stage}]: {verdict['action']}{_wait} — {verdict['why']}")
    _record(proj, {"context": ctx, "verdict": verdict})
    return verdict


def _record(proj, entry: dict) -> None:
    try:
        if proj is None:
            return
        p = Path(proj.output_dir) / "incident_report.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        hist = []
        if p.exists():
            hist = json.loads(p.read_text() or "[]")
        entry["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        hist.append(entry)
        p.write_text(json.dumps(hist, indent=1))
    except Exception:                                    # noqa: BLE001
        pass


def interventions_used(proj) -> int:
    try:
        p = Path(proj.output_dir) / "incident_report.json"
        if not p.exists():
            return 0
        return sum(1 for e in json.loads(p.read_text() or "[]")
                   if (e.get("verdict") or {}).get("action") in ("retry", "retry_after_wait"))
    except Exception:                                    # noqa: BLE001
        return 0
