#!/usr/bin/env python3
"""Run ClipStudio's analyze stage into a crash-resumable scratch artifact.

This intentionally calls the production analyzer and policy finalizer without touching the
source project.  Text-model replies are content-addressed and appended before they are returned
to the analyzer, so an interrupted audit replays completed calls and resumes at the first missing
batch instead of buying the whole analysis again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _call_key(*, system: str, messages, max_tokens: int, model: str) -> str:
    raw = json.dumps(
        {
            "system": system,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "model": str(model or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _valid_json_reply(text: str) -> bool:
    """Only replay replies the production analyzer could actually parse.

    A non-empty but truncated array must reach analyze.py's live retry. Replaying it merely because
    transport succeeded would turn the response cache into a behavior change.
    """
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, count=1, flags=re.I)
    object_at, array_at = value.find("{"), value.find("[")
    starts = [item for item in ((object_at, "{"), (array_at, "[")) if item[0] >= 0]
    if not starts:
        return False
    start, root = min(starts)
    pattern = r"\{.*\}" if root == "{" else r"\[.*\]"
    match = re.search(pattern, value[start:], re.S)
    if not match:
        return False
    value = match.group(0)
    try:
        return isinstance(json.loads(value), (dict, list))
    except Exception:
        return False


def _load_calls(path: Path) -> dict[str, tuple[str, dict]]:
    cached: dict[str, tuple[str, dict]] = {}
    if not path.is_file():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            key = str(row.get("key", ""))
            text = str(row.get("text", ""))
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            if len(key) == 64 and _valid_json_reply(text):
                cached[key] = (text, meta)
        except Exception:
            continue
    return cached


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--movie-hint", required=True)
    args = parser.parse_args()

    from vidlore.clipstudio import llm, policy
    from vidlore.clipstudio.analyze import analyze_script
    from vidlore.clipstudio.config import engine_config, load_clip_config

    job = args.job.expanduser().resolve()
    out = args.out.expanduser().resolve()
    calls_path = out.with_name(out.stem + ".calls.jsonl")
    script_path = job / "script.txt"
    script = script_path.read_text(encoding="utf-8")
    cached = _load_calls(calls_path)
    original_complete_ex = llm.complete_ex
    started = time.time()
    served = 0
    replayed = 0

    def checkpointed_complete_ex(*, system: str = "", messages, max_tokens: int = 1024,
                                 eng_cfg=None, model: str = "") -> tuple:
        nonlocal served, replayed
        key = _call_key(system=system, messages=messages, max_tokens=max_tokens, model=model)
        if key in cached:
            replayed += 1
            text, meta = cached[key]
            print(f"analyze-audit: replayed call {key[:12]}", flush=True)
            return text, dict(meta)
        text, meta = original_complete_ex(
            system=system, messages=messages, max_tokens=max_tokens,
            eng_cfg=eng_cfg, model=model)
        text = str(text or "")
        meta = dict(meta or {})
        row = {
            "schema_version": 1,
            "key": key,
            "text": text,
            "meta": meta,
            "recorded_at_epoch": time.time(),
        }
        calls_path.parent.mkdir(parents=True, exist_ok=True)
        with calls_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        # Do not add a newly served reply to the startup replay map. If analyze.py rejects its JSON
        # and immediately retries the same request, that retry must remain a real provider call.
        # A later process may replay the row only after _load_calls validates its JSON structure.
        served += 1
        print(f"analyze-audit: checkpointed call {key[:12]} served={meta.get('served', '')}",
              flush=True)
        return text, meta

    # Keep ``llm.complete`` itself untouched. Its production wrapper resolves ``complete_ex`` at
    # call time, while complete_ex's legacy test seam continues to see the original complete.
    llm.complete_ex = checkpointed_complete_ex

    def progress(message) -> None:
        print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    analysis, segments = analyze_script(
        script,
        topic=args.topic,
        movie_hint=args.movie_hint,
        eng_cfg=engine_config(),
        cfg=load_clip_config(),
        progress=progress,
    )
    tally = policy.finalize_beats(segments)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "job": str(job),
        "script_path": str(script_path),
        "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "topic": args.topic,
        "movie_hint": args.movie_hint,
        "elapsed_seconds": round(time.time() - started, 3),
        "response_calls_served": served,
        "response_calls_replayed": replayed,
        "analysis": analysis.to_dict(),
        "segments": [segment.to_dict() for segment in segments],
        "policy_tally": tally,
    }
    # This is deliberately the first operation after the production stage returns.
    _atomic_json(out, payload)
    print(f"analyze-audit: SAVED {out} beats={len(segments)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
