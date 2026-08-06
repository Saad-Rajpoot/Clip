"""Prove — or disprove — that a beat's footage is absent from the pool, by looking at it.

The specificity ladder (exact -> character -> abstract) is gated behind evidence that the beat's
authored target is not in the pool. That gate is right, and it is why an honest footage gap used to
kill a render: the only producer of that evidence was a human writing prose, and nothing in the
pipeline ever wrote one.

This module is the machine producer. It does NOT impersonate the human review — it declares
``audit_kind: machine_exhaustive_strict_verifier`` and stands on a different claim: not "a person
looked and ruled out pipeline bugs", but "every eligible window in a recorded, countable universe
was put through the SAME judge the publication gate uses, and none passed".

That claim is only as good as three properties, each enforced below rather than asserted:

  * ONE JUDGE. Windows are decided by `verify.strict_window_verdict`, the function
    `_try_promote_inner` itself calls. There is no second implementation of the bar to drift.
  * CLIP MAY ORDER, NEVER EXCLUDE. Retrieval rank decides what is examined FIRST and nothing else.
    Beat 134's answer sat at CLIP rank 563 of 4942; an auditor that trusted the bench would have
    called it a gap and been wrong. Exclusions are structural and each one is recorded with a
    reason.
  * A CALL THAT DID NOT ANSWER IS NOT A REJECTION. `strict_window_verdict` returns
    ``status="incomplete"`` on a transport error or an exception. One such window anywhere in the
    universe makes the whole audit ``audit_incomplete``, which authorizes nothing.

A run that finds a passing window is the MORE valuable outcome: the beat is not a footage gap, the
pipeline missed footage it had, and the audit says so and names the window.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

GAP_AUDIT_SCHEMA = 1
GAP_AUDIT_KIND = "machine_exhaustive_strict_verifier"

# Verdicts are memoised per (window bytes, beat contract, judge identity). Nothing here may be
# reused across a change to any of those — see `_verdict_key`.
_VERDICT_CACHE_SCHEMA = 1


def _sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _sha_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def beat_contract(seg) -> dict:
    """Everything about the beat that can change which windows satisfy it."""
    return {
        "index": int(getattr(seg, "index", -1)),
        "text": str(getattr(seg, "text", "") or ""),
        "visual_policy": str(getattr(seg, "visual_policy", "") or ""),
        "required_kind": str(getattr(seg, "required_kind", "") or ""),
        "required_entity": str(getattr(seg, "required_entity", "") or ""),
        "scene_query": str(getattr(seg, "scene_query", "") or ""),
        "expected_visual": str(getattr(seg, "expected_visual", "") or ""),
        "quote": str(getattr(seg, "quote", "") or ""),
        "is_specific_claim": bool(getattr(seg, "is_specific_claim", True)),
    }


def judge_identity(eng) -> dict:
    """The verifier's identity. A different model or prompt is a different judgement."""
    from . import verify as _V
    return {
        "strict_window_verdict_src": hashlib.sha256(
            __import__("inspect").getsource(_V.strict_window_verdict).encode("utf-8")
        ).hexdigest()[:16],
        "vision_model": str(getattr(eng, "vision_model", "") or getattr(eng, "model", "") or ""),
        "cache_schema": _VERDICT_CACHE_SCHEMA,
    }


def eligible_universe(proj) -> tuple[list, list]:
    """(windows, exclusions). Every SOURCE_OK indexed window, minus STRUCTURAL exclusions only.

    A structural exclusion is one that makes the window impossible or impermissible to air, and
    every one is recorded with its reason: the pool gate already rejected the source (subtitled
    copy, sub-native-HD, promo overlay, non-live-action...), the media is gone, or the window has no
    keyframe to judge. Nothing is excluded for ranking low — that is the mistake this exists to
    avoid.
    """
    from .index import load_shots
    meta = getattr(proj, "meta", {}) or {}
    rejected = {str(k): str(v) for k, v in (meta.get("auto_rejected_reasons") or {}).items()}
    banned = {str(x) for x in (meta.get("banned_sources") or [])}
    windows, exclusions = [], []
    for src in getattr(proj, "sources", None) or []:
        sid = str(getattr(src, "id", "") or "")
        if str(getattr(src, "status", "") or "") != "ok":
            exclusions.append({"source_id": sid, "reason": "source_status_not_ok",
                               "windows": None})
            continue
        if sid in rejected:
            exclusions.append({"source_id": sid, "reason": f"pool_gate:{rejected[sid]}",
                               "windows": None})
            continue
        if sid in banned:
            exclusions.append({"source_id": sid, "reason": "pool_gate:banned_source",
                               "windows": None})
            continue
        lp = str(getattr(src, "local_path", "") or "")
        if not lp or not Path(lp).exists():
            exclusions.append({"source_id": sid, "reason": "media_missing", "windows": None})
            continue
        try:
            shots = load_shots(proj, sid) or []
        except Exception as exc:                          # noqa: BLE001 — unreadable index
            exclusions.append({"source_id": sid,
                               "reason": f"index_unreadable:{type(exc).__name__}",
                               "windows": None})
            continue
        kept = 0
        for sh in shots:
            kf = str(getattr(sh, "keyframe_path", "") or "")
            if not kf or not Path(kf).exists():
                exclusions.append({"source_id": sid, "shot_index": int(getattr(sh, "index", -1)),
                                   "reason": "keyframe_missing", "windows": 1})
                continue
            windows.append((src, sh))
            kept += 1
    return windows, exclusions


def _verdict_key(kf_sha: str, contract: dict, judge: dict) -> str:
    return _sha_obj({"schema": _VERDICT_CACHE_SCHEMA, "keyframe_sha256": kf_sha,
                     "contract": contract, "judge": judge})


class _VerdictCache:
    """Memo for strict window verdicts, keyed on the window's BYTES and the whole contract+judge.

    Only JUDGED results are stored. An incomplete verdict is never cached, because a cached
    incomplete would silently become a rejection on the next pass — the one substitution this
    audit's whole claim depends on not happening.
    """

    def __init__(self, path: Path):
        self.path = path
        self.hit = self.miss = 0
        try:
            self._data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except Exception:                                 # noqa: BLE001 — absent/corrupt == empty
            self._data = {}

    def get(self, key):
        got = self._data.get(key)
        if isinstance(got, dict) and got.get("status") == "judged":
            self.hit += 1
            return got
        self.miss += 1
        return None

    def put(self, key, decision: dict) -> None:
        if decision.get("status") != "judged":
            return
        self._data[key] = decision

    def flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:                                 # noqa: BLE001 — a memo fault costs time
            pass


def exhaustive_gap_audit(proj, seg, cfg, eng, *, log=print, cap: int = 0,
                         order_by_rank=None) -> dict:
    """Judge every eligible window against this beat and report what was found.

    `cap` exists for exploration only: a capped scan can never report exhaustion, and the returned
    status says so. `order_by_rank`, if given, may REORDER the universe (cheapest-first) and must
    never remove anything from it.
    """
    from . import verify as _V
    from . import selfheal as _SH

    contract = beat_contract(seg)
    judge = judge_identity(eng)
    windows, exclusions = eligible_universe(proj)
    universe_n = len(windows)
    if order_by_rank is not None:
        try:
            ordered = list(order_by_rank(windows))
            if len(ordered) == universe_n:
                windows = ordered
        except Exception:                                 # noqa: BLE001 — ordering is optional
            pass

    # QUESTION PARITY — the unresolved half, and the reason this cannot authorize anything yet.
    #
    # cb0dcaf made the DECISION callable, and this module uses it. But the pipeline does not ask
    # `verify_frame` the bare question this module asks: `_verify_ctx` builds a 15/50/85 contact
    # sheet of the SELECTED WINDOW rather than judging the shot's keyframe, and it passes the
    # shot's Face-ID names, the beat's era hint, its deictic must-see target and the exact-cast
    # warning. Withholding those asks a WEAKER question, and a weaker question says yes too often:
    # measured on beat 134, this scan accepted an Oberyn-fight window for a beat that requires
    # Shae, with an empty evidence block (faceid_names None, model None). Right conclusion, wrong
    # window — which is luck, not a proof.
    #
    # An audit that cannot show it asked the IDENTICAL question has no business authorizing the
    # loss of a beat's specificity, so it does not. Until the question construction is extracted
    # the same way the decision was, `authorizes_softening` is pinned False and the reason is
    # recorded in the artifact. A "not a footage gap" finding remains fully usable — it only ever
    # says the pipeline had footage it did not use, and a weaker question cannot invent one.
    question_parity = False
    cache = _VerdictCache(Path(getattr(proj, "index_dir", ".")) / "strict_window_verdicts.json")
    pol = contract["visual_policy"]
    exact, character = pol == "exact_scene", pol == "character_specific"
    try:
        must_see = _V._must_see(seg) if hasattr(_V, "_must_see") else ""
    except Exception:                                     # noqa: BLE001
        must_see = ""
    c2a = _V._project_char2actor(proj)

    examined = incomplete = 0
    passes, incomplete_rows = [], []
    scan = windows[:cap] if cap else windows
    for src, sh in scan:
        kf = str(getattr(sh, "keyframe_path", "") or "")
        try:
            kf_sha = _sha_file(kf)
        except Exception as exc:                          # noqa: BLE001
            incomplete += 1
            incomplete_rows.append({"source_id": src.id, "shot_index": getattr(sh, "index", -1),
                                    "reason": f"keyframe_unreadable:{type(exc).__name__}"})
            continue
        key = _verdict_key(kf_sha, contract, judge)
        decision = cache.get(key)
        if decision is None:
            av = _V.verify_frame(
                kf, contract["text"], contract["required_entity"], contract["required_kind"],
                [], eng, is_specific=contract["is_specific_claim"],
                expected_visual=contract["expected_visual"],
                scene_query=contract["scene_query"])
            alt = type("_W", (), {"source_id": src.id,
                                  "shot_index": int(getattr(sh, "index", -1)),
                                  "in_point": float(getattr(sh, "start", 0.0) or 0.0),
                                  "out_point": float(getattr(sh, "end", 0.0) or 0.0)})()
            decision = _V.strict_window_verdict(av, seg, alt, proj, cfg, c2a, downgrade=False,
                                                exact=exact, character=character,
                                                must_see=must_see)
            cache.put(key, decision)
        examined += 1
        if decision.get("status") != "judged":
            incomplete += 1
            incomplete_rows.append({"source_id": src.id, "shot_index": getattr(sh, "index", -1),
                                    "reason": decision.get("reason", "incomplete")})
            continue
        if decision.get("accept"):
            passes.append(decision)
            break                                         # one pass disproves absence
    cache.flush()

    covered = examined + incomplete >= universe_n and not cap
    complete = covered and incomplete == 0
    if passes:
        classification, status = "not_a_footage_gap", "complete"
    elif complete:
        classification, status = "footage_gap", "complete"
    else:
        classification, status = "undetermined", "audit_incomplete"

    pool_fp, pool_n = _SH._gap_pool_fingerprint(proj)
    return {
        "schema_version": GAP_AUDIT_SCHEMA,
        "audit_kind": GAP_AUDIT_KIND,
        "status": status,
        "classification": classification,
        "beat_index": contract["index"],
        "beat_fingerprint": _SH._gap_beat_fingerprint(seg),
        "contract": contract,
        "contract_fingerprint": _sha_obj(contract),
        "judge": judge,
        "pool_fingerprint": pool_fp,
        "pool_source_count": pool_n,
        "universe_size": universe_n,
        "universe_fingerprint": _sha_obj(
            sorted(f"{s.id}#{getattr(sh, 'index', -1)}" for s, sh in windows)),
        "windows_examined": examined,
        "windows_incomplete": incomplete,
        "incomplete_windows": incomplete_rows[:20],
        "exclusions": exclusions[:200],
        "exclusion_count": len(exclusions),
        "passing_window": passes[0] if passes else None,
        "verdict_cache": {"hit": cache.hit, "miss": cache.miss},
        # The claim, stated as what it is. Absence is only authorized by a COMPLETE scan of a
        # recorded universe in which every call answered and none passed.
        "question_parity_with_gate": question_parity,
        "question_parity_gap": (
            "" if question_parity else
            "verify_frame is called without the selected-window contact sheet, Face-ID names, "
            "era hint, must-see target or exact-cast warning that _verify_ctx supplies"),
        "authorizes_softening": bool(
            classification == "footage_gap" and status == "complete" and question_parity),
    }
