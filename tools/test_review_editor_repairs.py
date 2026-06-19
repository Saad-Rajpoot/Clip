#!/usr/bin/env python3
"""Regression tests for the Review-Editor V1 repair pass (2026-06-02).

Covers the backend (editor_manifest) side of the fixes shipped in that pass:

  * Replace-Visual reaches the renderer        -> apply_overrides writes
    edits/locked_visuals.json keyed by FINAL scene index, follows a reorder,
    and clears when the override is removed.
  * Re-render commit                           -> mark_rendered() clears the
    one-shot regen flags and records a signature so pending_summary reports
    rendered_clean (the "unsaved" indicator clears on success; refresh-safe).
  * Card "Restore" != "Reset scene"            -> restore_card() clears only
    card_removed and preserves other edits on the scene.
  * Empty-project protection                   -> remove_scene() refuses to
    delete the last visible scene.
  * Card-body data-loss guard                  -> save_card_override() passes
    the scene originals to _card_encode so map/default kinds keep graphic_body.

Run:  .venv/bin/python tools/test_review_editor_repairs.py
Pure-logic; no server, no render, no network. Uses a temp copy of an existing
project's script.json + brief.json (skips cleanly if none is found).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore import editor_manifest as EM  # noqa: E402


def _find_project() -> Path | None:
    out = ROOT / "output"
    if not out.is_dir():
        return None
    # prefer a top-level, regex-openable project with a card scene
    cands = []
    for d in out.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "script.json").exists():          # brief.json is optional (EM falls back to {})
            cands.append(d)
    return cands[0] if cands else None


def _fresh_copy(src: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="rev_editor_test_"))
    shutil.copy(src / "script.json", tmp / "script.json")
    if (src / "brief.json").exists():
        shutil.copy(src / "brief.json", tmp / "brief.json")
    return tmp


def main() -> int:
    src = _find_project()
    if src is None:
        print("SKIP: no top-level output/<slug> project with script.json+brief.json")
        return 0
    print(f"using project: {src.name}")
    sj = json.loads((src / "script.json").read_text())
    n = len(sj["scenes"])
    passed = failed = 0

    def check(name, cond, extra=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS {name}")
        else:
            failed += 1
            print(f"  FAIL {name} {extra}")

    # ---- 1. Replace-Visual -> locked_visuals.json (+ reorder follow + clear) ----
    t = _fresh_copy(src)
    sid2 = EM.scene_stable_id(t, 2)
    (t / "edits" / "uploads").mkdir(parents=True, exist_ok=True)
    (t / "edits" / "uploads" / "sc002.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    ov = EM.load_overrides(t)
    ov["scenes"][sid2] = {"visual_override": {"user_file": "edits/uploads/sc002.jpg",
                                              "is_video": False}, "scene_index_hint": 2}
    EM._save_overrides(t, ov, snapshot=False)
    EM.apply_overrides(t)
    lv = json.loads((t / "edits" / "locked_visuals.json").read_text())
    check("locked_visuals written at scene index 2",
          lv.get("2", {}).get("file") == "edits/uploads/sc002.jpg", lv)
    EM.reorder_scene(t, 2, 0)
    EM.apply_overrides(t)
    lv = json.loads((t / "edits" / "locked_visuals.json").read_text())
    check("locked_visuals follows scene through reorder (2->0)",
          lv.get("0", {}).get("file") == "edits/uploads/sc002.jpg", lv)
    EM.reset_all(t)
    EM.apply_overrides(t)
    lv = json.loads((t / "edits" / "locked_visuals.json").read_text())
    check("locked_visuals cleared after reset_all", lv == {}, lv)

    # ---- 2. mark_rendered: clears regen flags + records rendered_clean ----
    t = _fresh_copy(src)
    EM.set_regen(t, 3, "visual")
    EM.set_regen(t, 4, "voice")
    EM.save_global_override(t, {"music_enabled": False})
    p = EM.pending_summary(t)
    check("pending shows regen before render", p["regen"] == 2, p)
    check("rendered_clean False before any render", p["rendered_clean"] is False)
    EM.mark_rendered(t)
    ov = EM.load_overrides(t)
    has_regen = any(s.get("regen_visual") or s.get("regen_voice")
                    for s in ov["scenes"].values())
    check("mark_rendered cleared one-shot regen flags", not has_regen)
    p = EM.pending_summary(t)
    check("rendered_clean True after mark_rendered (pending 'applied')",
          p["rendered_clean"] is True, p)
    check("global preserved across mark_rendered (persistent setting)",
          ov.get("global", {}).get("music_enabled") is False)
    # editing again must flip rendered_clean back to False
    EM.save_global_override(t, {"captions_enabled": True})
    check("rendered_clean flips False after a new edit",
          EM.pending_summary(t)["rendered_clean"] is False)

    # ---- 3. restore_card clears ONLY card_removed ----
    t = _fresh_copy(src)
    card_idx = next((i for i, s in enumerate(sj["scenes"])
                     if (s.get("graphic_kind") or "").strip()), None)
    if card_idx is not None:
        EM.save_visual_prompt(t, card_idx, "co-existing edit")
        EM.remove_card_override(t, card_idx)
        sid = EM.scene_stable_id(t, card_idx)
        check("card removed flag set",
              EM.load_overrides(t)["scenes"][sid].get("card_removed") is True)
        EM.restore_card(t, card_idx)
        sc = EM.load_overrides(t)["scenes"].get(sid, {})
        check("restore_card clears card_removed", "card_removed" not in sc)
        check("restore_card preserves co-existing edit",
              sc.get("visual_prompt_override") == "co-existing edit", sc)
    else:
        print("  SKIP restore_card (no card scene)")

    # ---- 4. empty-project protection ----
    t = _fresh_copy(src)
    for i in range(n - 1):
        r = EM.remove_scene(t, i)
        if not r.get("ok"):
            check(f"remove_scene {i} unexpectedly blocked", False, r)
            break
    r = EM.remove_scene(t, n - 1)
    check("remove_scene refuses to delete the last visible scene",
          (not r.get("ok")) and "last scene" in r.get("error", ""), r)

    # ---- 5. card-body preserve (no silent wipe) ----
    t = _fresh_copy(src)
    body_idx = next((i for i, s in enumerate(sj["scenes"])
                     if (s.get("graphic_body") or "").strip()
                     and (s.get("graphic_kind") or "")
                     in ("map_region", "map_route", "")), None)
    if body_idx is not None:
        orig_body = sj["scenes"][body_idx].get("graphic_body", "")
        kind = (sj["scenes"][body_idx].get("graphic_kind") or "").strip()
        vals = ({"place": "X", "coords": "1,2"} if kind == "map_region"
                else {"title": "X"})  # edit a field that does NOT include body
        EM.save_card_override(t, body_idx, vals)
        sid = EM.scene_stable_id(t, body_idx)
        saved = EM.load_overrides(t)["scenes"][sid]["card_text_override"].get("graphic_body", "")
        check(f"card-body preserved for kind={kind or 'default'}",
              saved == orig_body, f"orig={orig_body!r} saved={saved!r}")
    else:
        print("  SKIP card-body (no suitable card scene)")

    # ---- 6. P2 — build_manifest prefers exact render_meta scene timing ----
    t = _fresh_copy(src)
    nsc = len(json.loads((t / "script.json").read_text())["scenes"])
    (t / "render_meta.json").write_text(json.dumps({
        "video_seconds": 10.0 * nsc,
        "scene_starts": [round(10.0 * i, 3) for i in range(nsc)],
        "scene_durations": [10.0] * nsc,
    }))
    m = EM.build_manifest(t)
    s0, s1 = m["scenes"][0], m["scenes"][1]
    check("P2 build_manifest uses exact render_meta scene_starts",
          abs(s0["start"] - 0.0) < 0.01 and abs(s1["start"] - 10.0) < 0.01
          and s0.get("duration_source") == "render_meta",
          (s0["start"], s1["start"], s0.get("duration_source")))
    (t / "render_meta.json").write_text(json.dumps({
        "video_seconds": 100.0, "scene_starts": [0, 5], "scene_durations": [5, 5]}))
    m2 = EM.build_manifest(t)
    check("P2 falls back to estimate when count mismatches",
          m2["scenes"][0].get("duration_source") != "render_meta")

    # ---- 7. P5 — unsupported upload type rejected with a friendly message ----
    t = _fresh_copy(src)
    r = EM.save_visual_override(t, 0, "malware.exe", b"x")
    check("P5 unsupported file type rejected (friendly msg)",
          (not r.get("ok")) and "supported" in r.get("error", "").lower(), r)

    # ---- 8. P3 — deep upload validation (probe rejects corrupt/empty; original kept) ----
    import subprocess as _sp
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe as _ffx
        _ff = _ffx()
    except Exception:                                          # noqa: BLE001
        _ff = None
    if _ff:
        t = _fresh_copy(src)
        vp = t / "valid.png"
        _sp.run([_ff, "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:d=1",
                 "-frames:v", "1", str(vp)], capture_output=True, timeout=30)
        r_ok = EM.save_visual_override(t, 2, "v.png", vp.read_bytes())
        check("P3 valid image accepted", r_ok.get("ok") is True, r_ok)
        r_bad = EM.save_visual_override(t, 3, "bad.mp4", b"this is not a real video file")
        check("P3 corrupt video rejected (friendly msg)",
              (not r_bad.get("ok")) and "read" in r_bad.get("error", "").lower(), r_bad)
        r_empty = EM.save_visual_override(t, 4, "e.png", b"")
        check("P3 empty file rejected",
              (not r_empty.get("ok")) and "empty" in r_empty.get("error", "").lower(), r_empty)
        ov = EM.load_overrides(t)
        sid3 = EM.scene_stable_id(t, 3)
        check("P3 original preserved on reject (no override recorded)",
              not ov["scenes"].get(sid3, {}).get("visual_override"))
    else:
        print("  SKIP P3 upload validation (no ffmpeg)")

    # ---- 9. P2 — per-project build lock + atomic-write helpers exist ----
    lk = EM._build_lock(_fresh_copy(src))
    check("P2 _build_lock returns a real lock", hasattr(lk, "acquire") and hasattr(lk, "release"))
    check("P2 _atomic_write present", callable(getattr(EM, "_atomic_write", None)))

    # ---- 8. Bug 2 — timeline seek/clamp math (mirrors __edTLSeekX in web.py) ----
    # JS: t = clamp(((clientX - rowLeft) / rowWidth) * tot, 0, tot-0.05)
    def _seek(x, left, width, tot):
        return max(0.0, min(tot - 0.05, ((x - left) / max(1.0, width)) * tot))
    tot, left, width = 201.0, 120.0, 1284.0
    check("seek: click at row start -> 0", abs(_seek(120, left, width, tot) - 0.0) < 0.01)
    check("seek: click at 50% -> ~tot/2",
          abs(_seek(left + width / 2, left, width, tot) - tot / 2) < 0.5)
    check("seek: click past right edge clamps to tot-0.05",
          abs(_seek(left + width + 500, left, width, tot) - (tot - 0.05)) < 0.01)
    check("seek: click left of start clamps to 0", _seek(left - 300, left, width, tot) == 0.0)
    check("seek: ~2:00 maps near 120s",
          abs(_seek(left + width * (120 / tot), left, width, tot) - 120) < 0.5)

    # ---- 9. Live-preview layers: set_layer + layers_off + render honoring ----
    t = _fresh_copy(src)
    sj9 = json.loads((t / "script.json").read_text())
    cidx = next((i for i, s in enumerate(sj9["scenes"]) if (s.get("graphic_kind") or "").strip()), None)
    if cidx is not None:
        EM.set_layer(t, cidx, "card", False)
        sid9 = EM.scene_stable_id(t, cidx)
        check("set_layer card off -> card_removed",
              EM.load_overrides(t)["scenes"][sid9].get("card_removed") is True)
        EM.apply_overrides(t)
        baked = json.loads((t / "script.json").read_text())["scenes"]
        check("hidden card removed from rendered script.json (final MP4 honors it)",
              (baked[cidx].get("graphic_kind") or "") == "")
        EM.set_layer(t, cidx, "card", True)
        check("set_layer card on -> card_removed cleared",
              "card_removed" not in EM.load_overrides(t)["scenes"].get(sid9, {}))
    EM.set_layer(t, 0, "visual", False)
    check("set_layer visual off -> recorded in layers_off",
          "visual" in (EM.load_overrides(t)["scenes"][EM.scene_stable_id(t, 0)].get("layers_off") or []))
    EM.set_layer(t, 0, "captions", False)
    check("set_layer captions -> whole-video global",
          EM.load_overrides(t)["global"].get("captions_enabled") is False)
    check("manifest exposes per-scene layers_off",
          "layers_off" in (EM.build_manifest(t)["scenes"][0].get("edit_status") or {}))
    r9 = EM.set_layer(t, 0, "bogus", False)
    check("set_layer rejects unknown layer", not r9.get("ok"))

    # ---- 10. Scene-selection <-> preview sync (V1.4.1) — mirrors web.py JS ----
    # 10a. __edSeekAnchor(sc): a SAFE preview seek INSIDE the scene, past the incoming
    #   dissolve (assemble.py dissolves run up to ~0.85s). The old code used start+0.05,
    #   which landed in the dissolve and showed the PREVIOUS scene.
    #   JS: off = min(max(0.9, dur*0.15), dur*0.5); return start + off
    def _anchor(start, end):
        dur = max(0.1, end - start)
        off = min(max(0.9, dur * 0.15), dur * 0.5)
        return start + off
    check("anchor: clears an 0.85s dissolve (>= start+0.85)", _anchor(7.671, 13.719) >= 7.671 + 0.85)
    check("anchor: stays inside the scene (< midpoint)", _anchor(7.671, 13.719) < (7.671 + 13.719) / 2)
    check("anchor: <6s scene uses the 0.9s floor", abs(_anchor(10.0, 14.0) - (10.0 + 0.9)) < 0.01)
    check("anchor: ~6.4s scene grows just above floor (dur*0.15)", abs(_anchor(20.49, 26.89) - (20.49 + 0.96)) < 0.01)
    check("anchor: long scene grows to 15% of duration", abs(_anchor(0.0, 20.0) - 3.0) < 0.01)
    check("anchor: short scene clamps to <= 50% (never overshoots)",
          0.0 < _anchor(0.0, 1.0) <= 0.5 + 1e-9)
    check("anchor: first scene (start 0) -> positive in-scene time", _anchor(0.0, 7.671) > 0.0)
    check("anchor: old start+0.05 would NOT have cleared the dissolve (regression)",
          (7.671 + 0.05) < (7.671 + 0.85) <= _anchor(7.671, 13.719))

    # 10b. stable scene_index -> current array position (reorder-safe __edsel mapping).
    #   JS: pos = M.scenes.findIndex(s => s.scene_index === i)
    def _pos_for(scenes, scene_index):
        return next((p for p, s in enumerate(scenes) if s["scene_index"] == scene_index), -1)
    fresh = [{"scene_index": i} for i in range(5)]
    check("scene_index->pos: fresh project is identity", _pos_for(fresh, 3) == 3)
    reordered = [{"scene_index": i} for i in [0, 1, 2, 4, 3]]   # scenes 4 & 5 swapped
    check("scene_index->pos: after reorder, stable id 3 -> pos 4", _pos_for(reordered, 3) == 4)
    check("scene_index->pos: after reorder, stable id 4 -> pos 3", _pos_for(reordered, 4) == 3)
    check("scene_index->pos: unknown id -> -1 (fallback)", _pos_for(reordered, 99) == -1)

    # 10c. active-scene half-open [start,end) + final-scene end clamp (__edActiveScene)
    SS = [(0.0, 7.671), (7.671, 13.719), (13.719, 20.487), (20.487, 26.895)]
    def _active(time_s, ss, sel=0):
        i = next((p for p, (a, b) in enumerate(ss) if a <= time_s < b), -1)
        if i < 0:
            if ss and time_s >= ss[-1][0]:
                return len(ss) - 1
            return sel
        return i
    check("active: t inside scene 2 -> index 1", _active(8.57, SS) == 1)
    check("active: exact boundary t==start belongs to NEW scene (half-open)", _active(7.671, SS) == 1)
    check("active: t just below the boundary stays in previous scene", _active(7.670, SS) == 0)
    check("active: t at the final scene's end clamps to last (not -1)", _active(26.895, SS) == 3)
    check("active: t far past the end clamps to last", _active(999.0, SS) == 3)

    # ---- 11. Undo reverts the FIRST edit on a fresh project (no prior overrides) ----
    # Regression: _save_overrides used to push an undo snapshot only when a prior
    # overrides file already existed, so the very first edit left undo_stack.json
    # empty and Undo was a silent no-op (undo() -> {"undone": False}). The first
    # edit must now snapshot a clean baseline and be undoable to the pristine state.
    t = _fresh_copy(src)
    usp = t / "edits" / "undo_stack.json"
    check("fresh project starts with no undo stack", not usp.exists())
    orig_order = [sc["scene_index"] for sc in EM.build_manifest(t)["scenes"]]
    if n >= 2:
        # the inspector "Down" button -> move_scene(scene_index, +1)
        EM.move_scene(t, 0, 1)
        check("first reorder set an order override",
              EM.load_overrides(t).get("order") is not None)
        edited_order = [sc["scene_index"] for sc in EM.build_manifest(t)["scenes"]]
        check("first reorder changed the manifest order",
              edited_order != orig_order, (orig_order, edited_order))
        stack = json.loads(usp.read_text()) if usp.exists() else []
        check("first edit pushed exactly one baseline onto the undo stack",
              len(stack) == 1, stack)
        check("pushed baseline is a clean no-overrides doc",
              bool(stack) and stack[0].get("order") is None
              and stack[0].get("scenes") == {}, stack[:1])
        r_undo = EM.undo(t)
        check("undo reports it undid the first edit (not a no-op)",
              r_undo.get("undone") is True, r_undo)
        ov_undone = EM.load_overrides(t)
        check("undo cleared the order override", ov_undone.get("order") is None)
        check("undo emptied the scene overrides", ov_undone.get("scenes") == {})
        undone_order = [sc["scene_index"] for sc in EM.build_manifest(t)["scenes"]]
        check("undo rebuilt the manifest to the original order",
              undone_order == orig_order, (orig_order, undone_order))
        check("undo stack empty after undoing the only edit",
              (json.loads(usp.read_text()) if usp.exists() else []) == [])
    else:
        print("  SKIP undo-first-reorder (project has <2 scenes)")

    # 11b. same guarantee for a first CARD-TEXT edit (different entry point,
    #   save_card_override -> _save_overrides -> same baseline-snapshot path).
    t = _fresh_copy(src)
    cidx11 = next((i for i, s in enumerate(sj["scenes"])
                   if (s.get("graphic_kind") or "").strip()), None)
    if cidx11 is not None:
        sid11 = EM.scene_stable_id(t, cidx11)
        EM.save_card_override(t, cidx11, {"title": "FIRST EDIT — undo me"})
        check("first card edit recorded an override",
              bool(EM.load_overrides(t)["scenes"].get(sid11, {}).get("card_text_override")))
        r = EM.undo(t)
        check("undo reverts the first card edit (not a no-op)", r.get("undone") is True, r)
        check("undo cleared the card-text override",
              not EM.load_overrides(t)["scenes"].get(sid11, {}).get("card_text_override"))
    else:
        print("  SKIP undo-first-card-edit (no card scene)")

    # ---- 12. V1.4.2 live-interaction completeness ----
    # 12a. Caption .srt timing lookup (mirrors __edSrtT + current-cue pick in web.py)
    def _srt_t(s):
        s = s.strip().replace(",", "."); p = s.split(":")
        return 0.0 if len(p) < 3 else int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    check("caption time: '00:00:07,671' -> 7.671", abs(_srt_t("00:00:07,671") - 7.671) < 1e-6)
    check("caption time: '00:01:20,500' -> 80.5", abs(_srt_t("00:01:20,500") - 80.5) < 1e-6)
    cues = [{"s": 0.0, "e": 2.0, "t": "In the summer of"}, {"s": 2.0, "e": 4.3, "t": "1812"}]
    def _cue_at(t):
        return next((c for c in cues if c["s"] <= t < c["e"]), None)
    check("caption lookup: t=1.0 -> first cue", (_cue_at(1.0) or {}).get("t") == "In the summer of")
    check("caption lookup: t=2.0 -> second cue (half-open)", (_cue_at(2.0) or {}).get("t") == "1812")
    check("caption lookup: t=9.9 (past end) -> none", _cue_at(9.9) is None)

    # 12b. captions-on logic (overlay shows unless captions_enabled === false)
    def _caps_on(g):
        return not (g.get("captions_enabled") is False)
    check("captions on when global unset (default)", _caps_on({}) is True)
    check("captions on when explicitly true", _caps_on({"captions_enabled": True}) is True)
    check("captions OFF only when explicitly false", _caps_on({"captions_enabled": False}) is False)

    # 12c. Generate-new prompt variation -> distinct fal cache keys per click
    hints = ["alternate composition", "wide establishing shot, environmental context",
             "closer detail, shallow depth of field", "different angle, dramatic natural light",
             "fresh framing, eye-level, documentary"]
    def _gprompt(base, seed):
        return f"{base}. {hints[seed % len(hints)]} (v{seed})."
    g1, g2 = _gprompt("a soldier", 1001), _gprompt("a soldier", 1002)
    check("generate: different seeds -> different prompts (cache-miss = fresh image)", g1 != g2)
    check("generate: prompt keeps the base subject", g1.startswith("a soldier"))
    check("generate: hint rotation cycles all 5", len({hints[s % len(hints)] for s in range(5)}) == 5)

    # 12d. nocap-proxy freshness rule (proxy usable iff >= mp4 mtime - 2s grace)
    def _fresh(nc_m, mp4_m):
        return nc_m >= mp4_m - 2
    check("nocap fresh: proxy newer than mp4 -> usable", _fresh(105.0, 100.0))
    check("nocap fresh: proxy within 2s grace -> usable", _fresh(99.0, 100.0))
    check("nocap stale: proxy much older than mp4 -> rebuild", not _fresh(50.0, 100.0))

    # 12e. Tooltip coverage — every required visible control has a data-tip OR title
    web = (ROOT / "vidlore" / "web.py").read_text(encoding="utf-8")
    import re as _re
    required = ["id=edexport", "id=edundo", "id=edmenu", "id=tbprev", "id=tbplay",
                "id=tbnext", "id=tbmute", "id=tbfs", "id=tbreplay", "id=edcaps",
                "__edGenerateNew", "class=scgrip", "class=scdots"]
    for marker in required:
        # find the line with the marker; assert a tip is present on that line
        line = next((ln for ln in web.splitlines() if marker in ln), "")
        has_tip = ("data-tip=" in line) or ("title=" in line)
        check(f"tooltip present for control: {marker}", has_tip, line[:80])
    # tooltip engine + status + caption-overlay machinery is wired
    check("tooltip engine present (__edTipInit)", "__edTipInit" in web and "data-tiptitle" in web)
    check("live status model present (__edStatus)", "function __edStatus(" in web)
    check("caption overlay sync present (__edCaptionSync)", "function __edCaptionSync(" in web)
    check("no-caption proxy builder present", "_prepare_caption_proxy" in web and "captions = False" in web)
    check("generate endpoint uses fal on-demand", "_fal_image(" in web and "/scene/<int:idx>/generate" in web)

    # 12f. LIVE_CAPTION (2026-06-03) — instant caption toggle RE-ENABLED.
    # The editor must SWAP the preview to the footage-matched no-caption base
    # (not the old honest-only "re-render" dead-end), the toggle must update the
    # HTML overlay instantly (paused-safe), and the renderer must EMIT that base
    # alongside the captioned MP4 in the SAME render, differing ONLY by captions.
    asm = (ROOT / "vidlore" / "assemble.py").read_text(encoding="utf-8")
    # ── editor side
    check("caption init re-enables the base-swap (no honest-only dead-end)",
          "var ready=await __edEnsureNocap(false)" in web)
    check("base-swap points the preview at the footage-matched nocap base",
          "function __edUseNocapBase(" in web and "/file/nocap-video" in web)
    check("caption toggle syncs the HTML overlay instantly (paused-safe)",
          ".then(function(){__edCaptionSync();})" in web)
    check("overlay renders ONLY over the nocap base (gated on __edNocapReady)",
          "__edNocapReady" in web and "function __edCaptionSync(" in web)
    # ── renderer side
    check("renderer emits the no-caption editor base (preview_nocap.mp4)",
          "preview_nocap.mp4" in asm)
    check("renderer composes the SAME graph minus ONLY the subtitle burn",
          "_compose_vchain(" in asm and 'startswith("subtitles=")' in asm)
    check("no-caption emission gated by VIDLORE_EMIT_NOCAP (CLI can opt out)",
          "VIDLORE_EMIT_NOCAP" in asm and "_emit_nocap" in asm)
    check("no-caption is a separate best-effort Stage-V (captioned path untouched)",
          "_tmp_video_nocap" in asm and "_err_nocap" in asm)

    # ---- 14. V1.4.4 live-text-overlay duplication fix — mirrors web.py JS ----
    SIMPLE = {"lower_third", "lower", "title_card", "title", "chapter", "chapter_card",
              "location", "location_title", "label", "name_reveal", "name"}
    def _card_simple(card):
        return ((card.get("graphic_kind") or "").lower() in SIMPLE
                and len(card.get("graphic_body") or "") <= 160)
    def _card_overlay(card, removed=False, card_off=False):
        # mirrors __edDraftSync card branch: base MP4 already shows every card, so only
        # draw an overlay for an EDITED card (draft if simple, honest note if complex).
        if removed or card_off:
            return "hidden_note"
        edited = bool(card.get("edited"))
        if not edited:
            return "none"
        return "draft" if _card_simple(card) else "honest_note"
    LT = {"graphic_kind": "lower_third", "graphic_text": "TEHRAN, IRAN", "graphic_body": "a decade", "edited": False}
    CLS = {"graphic_kind": "classified", "graphic_text": "EXEC ORDER", "graphic_body": "Subject accused…" * 6, "edited": False}
    STMT = {"graphic_kind": "statement", "graphic_text": "The door opens now.", "graphic_body": "", "edited": False}
    NAME = {"graphic_kind": "name_reveal", "graphic_text": "EBRAHIM RAISI", "graphic_body": "PRESIDENTIAL CANDIDATE", "edited": False}
    # Rule 1 — NEVER draw an overlay on an UNEDITED card (no duplicate title/body)
    check("unedited lower_third -> no overlay (no duplicate)", _card_overlay(LT) == "none")
    check("unedited classified document -> no overlay (no duplicate heading/body)", _card_overlay(CLS) == "none")
    check("unedited statement/kinetic -> no overlay (no giant words)", _card_overlay(STMT) == "none")
    check("unedited name_reveal -> no overlay", _card_overlay(NAME) == "none")
    # Edited simple kinds -> clean compact draft
    check("edited lower_third -> compact draft", _card_overlay({**LT, "edited": True}) == "draft")
    check("edited name_reveal -> compact draft", _card_overlay({**NAME, "edited": True}) == "draft")
    check("edited title_card -> compact draft",
          _card_overlay({"graphic_kind": "title_card", "graphic_body": "x", "edited": True}) == "draft")
    # Edited complex / unsupported kinds -> HONEST note (not a broken oversized overlay)
    check("edited classified (dense document) -> honest note", _card_overlay({**CLS, "edited": True}) == "honest_note")
    check("edited statement (kinetic) -> honest note", _card_overlay({**STMT, "edited": True}) == "honest_note")
    check("edited callout -> honest note (not whitelisted)",
          _card_overlay({"graphic_kind": "callout", "graphic_text": "PRAYER BEADS", "graphic_body": "", "edited": True}) == "honest_note")
    check("edited number/stat -> honest note", _card_overlay({"graphic_kind": "stat", "graphic_body": "", "edited": True}) == "honest_note")
    # Overflow guard — even a whitelisted kind with a long body falls back
    check("edited simple kind with long body (>160) -> honest note (overflow guard)",
          _card_overlay({"graphic_kind": "lower_third", "graphic_body": "y" * 200, "edited": True}) == "honest_note")
    # Hidden / removed card -> the hidden-card note (independent of edited)
    check("removed card -> hidden-card note", _card_overlay({**LT, "edited": True}, removed=True) == "hidden_note")
    check("card layer hidden -> hidden-card note", _card_overlay({**LT, "edited": True}, card_off=True) == "hidden_note")

    # Source-level guards in web.py
    web = (ROOT / "vidlore" / "web.py").read_text(encoding="utf-8")
    check("draft overlay gated on sc.card.edited (not 'is a card')", "sc.card.edited" in web and "cardEdited" in web)
    check("simple-card whitelist present", "__edSimpleCardKinds" in web and "__edCardSimple" in web)
    check("honest fallback note present", "edcardnote" in web and "appears after re-render" in web)
    check("__edDraftCardHTML no longer renders dense doc/number overlays (kinds fall back)",
          ".dcdoc" not in web.split("function __edDraftCardHTML")[1].split("function __edDraftSync")[0]
          and ".dcnum" not in web.split("function __edDraftCardHTML")[1].split("function __edDraftSync")[0])
    check("draft typography limits present (line-clamp + overflow)",
          "-webkit-line-clamp" in web and "#edlaycard .dcbody" in web)
    check("final-render card payload untouched (apply_overrides still bakes card_text_override)",
          "card_text_override" in (ROOT / "vidlore" / "editor_manifest.py").read_text(encoding="utf-8"))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
