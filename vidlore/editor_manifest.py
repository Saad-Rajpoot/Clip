"""Vidlore Review Editor — manifest generator (Phase 1, read-only).

Builds ``editor_manifest.json`` (the editor's durable read-model) from the REAL
on-disk render package — script.json + render_meta.json + render_metrics.json +
brief.json + the .srt + the templates registry. Pure (no Flask), Python-3.9
safe, NEVER consumed by the render pipeline (so it can't break it). All user
edits live in a SEPARATE overrides file (Phase 2); this module only projects.

Honest degradations (see the architecture risks):
 * per-scene durations are NOT stored anywhere -> derived word-proportional and
   scaled so the sum equals render_meta.video_seconds (duration_source flags it).
 * per-beat asset/source maps are unrecoverable from a normal saved render
   (opaque content-hash cache names, work dir deleted) -> we report the
   video-level source MIX + one self-extracted poster frame per scene.
 * most card art lives in the deleted work dir -> the editor shows the decoded
   card TEXT (always in script.json) and a "preview needs re-render" note.

Public API:
    build_manifest(run_dir) -> dict
    write_manifest(run_dir) -> Path           # + timeline.json + poster frames
    extract_scene_thumbs(run_dir, manifest)   # ffmpeg poster frames
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

SCHEMA = "editor_manifest/1"
TIMELINE_SCHEMA = "editor_timeline/1"
_WORDS_PER_SECOND = 3.0

# P2 — per-project build lock: manifest.json + timeline.json are fetched in
# PARALLEL on first open (and two tabs can open the same project), which used to
# race on duplicate ffmpeg thumbnail extraction + non-atomic manifest writes.
# A lock per run_dir serializes the build; thumbs are content-cached (skip-if-
# fresh) so the second caller returns near-instantly.
_BUILD_LOCKS: dict = {}
_BUILD_LOCKS_GUARD = threading.Lock()


def _build_lock(run_dir):
    key = str(Path(run_dir).resolve())
    with _BUILD_LOCKS_GUARD:
        lk = _BUILD_LOCKS.get(key)
        if lk is None:
            lk = _BUILD_LOCKS[key] = threading.Lock()
    return lk


def _atomic_write(p: Path, text: str) -> None:
    """Write via a temp file + os.replace so a concurrent reader never sees a
    half-written manifest/timeline."""
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def _probe_media(path) -> dict:
    """P3 — confirm an uploaded image/video is READABLE by ffmpeg and exposes a
    real video/image stream with sane dimensions. Returns {ok, codec, w, h} or
    {ok:False, reason}. The renderer re-encodes everything, so ffmpeg-readability
    + a valid stream is the meaningful gate (catches corrupt / empty / truncated /
    unsupported files); reason 'unreadable' covers corrupt + unknown-codec."""
    try:
        from .ffmpeg_tool import ffmpeg_exe
        r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, timeout=30)
        err = r.stderr or ""
    except Exception:                                          # noqa: BLE001
        return {"ok": False, "reason": "probe_failed"}
    m = re.search(r"Stream #\d+:\d+[^\n]*: Video: ([A-Za-z0-9_]+)[^\n]*?(\d{2,5})x(\d{2,5})", err)
    if not m:
        return {"ok": False, "reason": "unreadable"}
    codec, w, h = m.group(1).lower(), int(m.group(2)), int(m.group(3))
    if w < 16 or h < 16 or w > 8192 or h > 8192:
        return {"ok": False, "reason": "dimensions", "w": w, "h": h}
    return {"ok": True, "codec": codec, "w": w, "h": h}


# ── small IO helpers ─────────────────────────────────────────────────────
def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def _stable_id(narration: str, index: int) -> str:
    h = hashlib.sha256(f"{index}:{(narration or '')[:120]}".encode("utf-8"))
    return "sc-" + h.hexdigest()[:6]


def _mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


# ── caption (.srt) parsing ───────────────────────────────────────────────
def _srt_t(s: str) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", s.strip())
    if not m:
        return 0.0
    h, mi, se, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + se + ms / 1000.0


def _parse_srt(path: Path) -> list:
    if not path or not path.is_file():
        return []
    cues = []
    block = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
        if line.strip() == "":
            if len(block) >= 2 and "-->" in block[1]:
                a, b = block[1].split("-->")
                cues.append({"start": round(_srt_t(a), 2), "end": round(_srt_t(b), 2),
                             "text": " ".join(block[2:]).strip()})
            block = []
        else:
            block.append(line)
    return cues


# ── card packed-DSL <-> friendly EDITABLE fields ─────────────────────────
# Each field: {key, label, value, multiline}. _card_fields decodes the packed
# graphic_text/graphic_body into friendly inputs; _card_encode rebuilds them
# (round-trip safe; coords preserved so a name edit can't corrupt a map).
def _card_fields(kind: str, text: str, body: str) -> list:
    text = text or ""
    body = body or ""
    out = []

    def f(key, label, value, multiline=False):
        out.append({"key": key, "label": label, "value": value, "multiline": multiline})

    if kind == "map_region":
        place, _, coords = text.partition("@")
        f("place", "Place", place)
        f("coords", "Coordinates (lat,lng)", coords)
    elif kind == "map_reveal":
        f("place", "Place", text)
        f("coords", "Coordinates (lat,lng)", body)
    elif kind == "map_route":
        wps = []
        for wp in text.split("|"):
            nm, _, co = wp.partition("@")
            wps.append(nm.strip() + (" | " + co if co else ""))
        f("route", "Route stops (one per line: Place | lat,lng)", "\n".join(wps), True)
    elif kind == "news_article":
        mast, _, date = text.partition("|")
        head, _, exc = body.partition("|")
        f("masthead", "Masthead", mast)
        f("date", "Date", date)
        f("headline", "Headline", head)
        f("excerpt", "Highlighted excerpt", exc, True)
    elif kind == "cause_effect":
        f("title", "Title", text)
        f("pairs", "Cause→Effect (one per line: Label | Text)",
          "\n".join(p.replace("|", " | ") for p in body.split(";;") if p.strip()), True)
    elif kind in ("bullet_list", "conspiracy_board"):
        f("title", "Title", text)
        f("items", "Items (one per line)",
          "\n".join(b.strip() for b in body.split("|") if b.strip()), True)
    elif kind == "pull_quote_portrait":
        spk, _, ctx = text.partition("|")
        f("speaker", "Speaker", spk)
        f("context", "Context", ctx)
        f("quote", "Quote", body, True)
    else:
        f("title", "Title / Headline", text)
        f("body", "Body / Subtitle", body, True)
    return out


def _card_encode(kind: str, vals: dict, orig_text: str = "", orig_body: str = "") -> dict:
    def g(k, d=""):
        v = vals.get(k)
        return d if v is None else str(v)

    if kind == "map_region":
        place, coords = g("place"), g("coords").strip()
        return {"graphic_text": place + ("@" + coords if coords else ""),
                "graphic_body": orig_body}
    if kind == "map_reveal":
        return {"graphic_text": g("place"), "graphic_body": g("coords")}
    if kind == "map_route":
        stops = []
        for line in g("route").splitlines():
            line = line.strip()
            if not line:
                continue
            nm, _, co = line.partition("|")
            nm, co = nm.strip(), co.strip()
            stops.append(nm + ("@" + co if co else ""))
        return {"graphic_text": "|".join(stops), "graphic_body": orig_body}
    if kind == "news_article":
        return {"graphic_text": g("masthead") + "|" + g("date"),
                "graphic_body": g("headline") + "|" + g("excerpt")}
    if kind == "cause_effect":
        pairs = []
        for line in g("pairs").splitlines():
            line = line.strip()
            if not line:
                continue
            lab, _, val = line.partition("|")
            pairs.append(lab.strip() + "|" + val.strip())
        return {"graphic_text": g("title"), "graphic_body": ";;".join(pairs)}
    if kind in ("bullet_list", "conspiracy_board"):
        items = [x.strip() for x in g("items").splitlines() if x.strip()]
        return {"graphic_text": g("title"), "graphic_body": "|".join(items)}
    if kind == "pull_quote_portrait":
        return {"graphic_text": g("speaker") + "|" + g("context"),
                "graphic_body": g("quote")}
    return {"graphic_text": g("title"), "graphic_body": g("body", orig_body)}


# ── non-destructive overrides (Phase 2) ──────────────────────────────────
OVR_SCHEMA = "editor_overrides/1"
HIST_SCHEMA = "editor_history/1"
_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


def overrides_path(run_dir) -> Path:
    return Path(run_dir) / "edits" / "user_overrides.json"


def load_overrides(run_dir) -> dict:
    d = _load_json(overrides_path(run_dir))
    if not isinstance(d, dict):
        d = {}
    d.setdefault("schema", OVR_SCHEMA)
    d.setdefault("slug", Path(run_dir).name)
    d.setdefault("global", {})
    d.setdefault("scenes", {})
    d.setdefault("order", None)
    return d


def _clean_overrides(run_dir) -> dict:
    """The pristine, no-edits overrides doc — mirrors load_overrides() on a fresh
    project (before any user_overrides.json exists). Pushed as the undo baseline
    for the FIRST edit so that edit is undoable back to the no-overrides state."""
    return {"schema": OVR_SCHEMA, "slug": Path(run_dir).name,
            "global": {}, "scenes": {}, "order": None}


def _save_overrides(run_dir, ov: dict, *, snapshot: bool = True) -> None:
    p = overrides_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if snapshot:                       # push the pre-write state for undo
        prev = _load_json(p)
        # On the FIRST edit there is no prior overrides file (prev is None); push a
        # clean no-overrides baseline so undo() restores the pristine pre-edit state
        # instead of being a silent no-op (undo_stack.json was left empty before).
        _push_undo(run_dir, prev if prev is not None else _clean_overrides(run_dir))
    p.write_text(json.dumps(ov, indent=2, ensure_ascii=False), encoding="utf-8")


def _push_undo(run_dir, state: dict) -> None:
    sp = Path(run_dir) / "edits" / "undo_stack.json"
    stack = _load_json(sp) or []
    stack.append(state)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(stack[-25:]), encoding="utf-8")


def undo(run_dir) -> dict:
    """Revert the last edit (restore the previous overrides snapshot)."""
    sp = Path(run_dir) / "edits" / "undo_stack.json"
    stack = _load_json(sp) or []
    if not stack:
        return {"ok": True, "undone": False}
    prev = stack.pop()
    sp.write_text(json.dumps(stack), encoding="utf-8")
    overrides_path(run_dir).write_text(
        json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8")
    _history_append(run_dir, {"field": "undo", "render_class": "none"})
    return {"ok": True, "undone": True, "remaining": len(stack)}


def clear_visual(run_dir, idx: int) -> dict:
    """Remove ONLY the uploaded/searched visual replacement for a scene, then
    safely tidy up: delete the orphaned scratch upload (if nothing else uses it)
    and restore the original poster thumbnail. Originals are never touched."""
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    sc = ov["scenes"].get(sid)
    removed_file = None
    if sc:
        removed_file = (sc.get("visual_override") or {}).get("user_file")
        sc.pop("visual_override", None)
        if not sc:
            ov["scenes"].pop(sid, None)
        _save_overrides(run_dir, ov)
    cleaned = False
    if removed_file:
        cleaned = cleanup_orphan_upload(run_dir, removed_file, ov)
    restore_thumb(run_dir, idx)
    return {"ok": True, "cleaned": cleaned}


def _history_append(run_dir, entry: dict) -> None:
    hp = Path(run_dir) / "edits" / "edit_history.json"
    h = _load_json(hp)
    if not isinstance(h, dict):
        h = {"schema": HIST_SCHEMA, "entries": []}
    h.setdefault("entries", []).append(entry)
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")


def scene_stable_id(run_dir, idx: int) -> str:
    script = _load_json(Path(run_dir) / "script.json") or {}
    scs = script.get("scenes", [])
    if 0 <= idx < len(scs):
        return _stable_id(scs[idx].get("narration", ""), idx)
    return f"idx-{idx}"


def _scene_kind(run_dir, idx: int) -> str:
    script = _load_json(Path(run_dir) / "script.json") or {}
    scs = script.get("scenes", [])
    if 0 <= idx < len(scs):
        return (scs[idx].get("graphic_kind") or "").strip()
    return ""


def _scene_entry(ov: dict, sid: str, idx: int) -> dict:
    sc = ov["scenes"].setdefault(sid, {})
    sc["scene_index_hint"] = idx
    return sc


def save_card_override(run_dir, idx: int, field_vals: dict) -> dict:
    """Save edited card text (friendly fields) as a non-destructive override."""
    kind = _scene_kind(run_dir, idx)
    if not kind:
        return {"ok": False, "error": "this scene has no card to edit"}
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    # Preserve the scene's original text/body so card kinds that don't surface
    # every sub-field in the editor (map_region / map_route keep body via
    # orig_body; the default kind keeps body if untouched) don't silently wipe it.
    _scs = (_load_json(Path(run_dir) / "script.json") or {}).get("scenes", [])
    _orig = _scs[idx] if 0 <= idx < len(_scs) else {}
    enc = _card_encode(kind, field_vals or {},
                       orig_text=_orig.get("graphic_text", ""),
                       orig_body=_orig.get("graphic_body", ""))
    sc = _scene_entry(ov, sid, idx)
    sc["card_text_override"] = {"graphic_kind": kind, **enc}
    sc.pop("card_removed", None)
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "card_text", "render_class": "card_rebake"})
    return {"ok": True}


def remove_card_override(run_dir, idx: int) -> dict:
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    sc = _scene_entry(ov, sid, idx)
    sc["card_removed"] = True
    sc.pop("card_text_override", None)
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "card_removed", "render_class": "card_rebake"})
    return {"ok": True}


def restore_card(run_dir, idx: int) -> dict:
    """Undo ONLY a card removal (clears the card_removed flag) without touching
    any other edit on the scene — distinct from reset_scene, which discards the
    whole scene's overrides."""
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    sc = ov["scenes"].get(sid)
    if sc:
        sc.pop("card_removed", None)
        if not sc:
            ov["scenes"].pop(sid, None)
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "card_restored", "render_class": "card_rebake"})
    return {"ok": True}


def save_visual_override(run_dir, idx: int, filename: str, data: bytes) -> dict:
    """Store an uploaded replacement clip/image + record the override."""
    ext = (Path(filename).suffix or ".bin").lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp") + _VIDEO_EXTS:
        return {"ok": False,
                "error": "That file type isn't supported. Use a JPG, PNG, or WEBP "
                         "image, or an MP4 / MOV / WEBM video."}
    if not data:
        return {"ok": False, "error": "That file is empty. Your original scene is still safe."}
    updir = Path(run_dir) / "edits" / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    dest = updir / f"sc{idx:03d}{ext}"
    dest.write_bytes(data)
    # P3 — deep validation: probe the written file. On failure, delete it and do
    # NOT record the override, so the scene's ORIGINAL visual is preserved intact.
    probe = _probe_media(dest)
    if not probe.get("ok"):
        try:
            dest.unlink()
        except Exception:                                      # noqa: BLE001
            pass
        msg = {
            "unreadable": ("This file could not be read — it may be corrupt or use an "
                           "unsupported codec. Please upload an MP4 encoded with H.264, "
                           "or a JPG/PNG image. Your original scene is still safe."),
            "dimensions": "This file's dimensions aren't valid. Your original scene is still safe.",
            "probe_failed": "This file could not be checked right now. Your original scene is still safe.",
        }.get(probe.get("reason"), "This file could not be used. Your original scene is still safe.")
        return {"ok": False, "error": msg}
    rel = f"edits/uploads/{dest.name}"
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    sc = _scene_entry(ov, sid, idx)
    sc["visual_override"] = {"user_file": rel, "is_video": ext in _VIDEO_EXTS}
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "visual_upload", "to": rel,
                              "render_class": "scene_footage"})
    # refresh the editor poster from the upload so the left thumbnail updates now
    try:
        regen_thumb_from_upload(run_dir, idx, rel, ext in _VIDEO_EXTS)
    except Exception:                                          # noqa: BLE001
        pass
    return {"ok": True, "path": rel, "is_video": ext in _VIDEO_EXTS}


def reset_scene(run_dir, idx: int) -> dict:
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    popped = ov["scenes"].pop(sid, None)
    existed = popped is not None
    _save_overrides(run_dir, ov)
    # If the scene had an uploaded/searched visual, drop the orphan scratch file
    # and restore the original poster so the left-list thumbnail matches the
    # reset state (clear_visual already did this; reset_scene used not to).
    uf = (popped or {}).get("visual_override", {}).get("user_file") if existed else None
    if uf:
        try:
            cleanup_orphan_upload(run_dir, uf, ov)
            restore_thumb(run_dir, idx)
        except Exception:  # noqa: BLE001
            pass
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "reset_scene", "render_class": "none"})
    return {"ok": True, "existed": existed}


def reset_all(run_dir) -> dict:
    ov = load_overrides(run_dir)
    ov["scenes"] = {}
    ov["global"] = {}
    ov["order"] = None
    _save_overrides(run_dir, ov)
    revert_to_baseline(run_dir)
    _history_append(run_dir, {"field": "reset_all", "render_class": "none"})
    return {"ok": True}


def revert_to_baseline(run_dir) -> dict:
    """Restore the pristine script.json from script.baseline.json (if present)."""
    run_dir = Path(run_dir)
    base = run_dir / "script.baseline.json"
    if base.exists():
        shutil.copyfile(base, run_dir / "script.json")
        return {"ok": True, "reverted": True}
    return {"ok": True, "reverted": False}


def _sha_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def save_global_override(run_dir, patch: dict) -> dict:
    """Project-level audio/caption settings (captions_enabled, music_enabled,
    music_volume 0..1, duck_strength 'light'|'normal'|'heavy')."""
    ov = load_overrides(run_dir)
    for k, v in (patch or {}).items():
        ov["global"][k] = v
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"field": "global", "to": patch,
                              "render_class": "audio_only"})
    return {"ok": True, "global": ov["global"]}


def remove_scene(run_dir, idx: int) -> dict:
    ov = load_overrides(run_dir)
    # Empty-project protection: never let the last visible scene be deleted.
    try:
        base_scenes = (_load_json(Path(run_dir) / "script.json") or {}).get("scenes", [])
        _sids, _b, visible = _effective_order(base_scenes, ov)
        if len(visible) <= 1:
            return {"ok": False,
                    "error": "Can't delete the last scene — a video needs at least one scene."}
    except Exception:                                          # noqa: BLE001
        pass
    sid = scene_stable_id(run_dir, idx)
    _scene_entry(ov, sid, idx)["scene_removed"] = True
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "scene_removed", "render_class": "timeline_reflow"})
    return {"ok": True}


def save_visual_prompt(run_dir, idx: int, prompt: str) -> dict:
    """Edit the AI-image prompt (Scene.visual) for a scene → busts that scene's
    AI-footage cache on next render."""
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    _scene_entry(ov, sid, idx)["visual_prompt_override"] = (prompt or "").strip()
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "visual_prompt", "render_class": "scene_footage"})
    return {"ok": True}


def set_layer(run_dir, idx: int, name: str, visible: bool) -> dict:
    """Toggle a scene LAYER's visibility for the live draft + (where applicable) the
    final render. 'card' -> card_removed (already honored by apply_overrides +
    build_manifest, so a hidden card is excluded from the exported MP4). 'captions'
    -> whole-video setting (global.captions_enabled). 'visual'/'text' -> recorded in
    layers_off for the draft preview (the replacement still renders via
    visual_override). Stored stably by scene stable_id; Undo-snapshotted."""
    name = (name or "").strip().lower()
    if name not in ("visual", "card", "text", "captions"):
        return {"ok": False, "error": "unknown layer"}
    if name == "captions":
        ov = load_overrides(run_dir)
        ov["global"]["captions_enabled"] = bool(visible)
        _save_overrides(run_dir, ov)
        _history_append(run_dir, {"field": "layer_captions", "to": bool(visible),
                                  "render_class": "audio_only"})
        return {"ok": True, "layer": "captions", "visible": bool(visible), "scope": "whole_video"}
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    sc = _scene_entry(ov, sid, idx)
    if name == "card":
        if visible:
            sc.pop("card_removed", None)
        else:
            sc["card_removed"] = True
    off = set(sc.get("layers_off") or [])
    off.discard(name) if visible else off.add(name)
    if off:
        sc["layers_off"] = sorted(off)
    else:
        sc.pop("layers_off", None)
    if not sc or list(sc.keys()) == ["scene_index_hint"]:
        ov["scenes"].pop(sid, None)
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": "layer_" + name, "to": bool(visible),
                              "render_class": "card_rebake" if name == "card" else "none"})
    return {"ok": True, "layer": name, "visible": bool(visible)}


def set_regen(run_dir, idx: int, kind: str) -> dict:
    """Mark a scene's visual or voice to regenerate (variant bump at render)."""
    key = {"visual": "regen_visual", "voice": "regen_voice"}.get(kind)
    if not key:
        return {"ok": False, "error": "kind must be visual|voice"}
    ov = load_overrides(run_dir)
    sid = scene_stable_id(run_dir, idx)
    _scene_entry(ov, sid, idx)[key] = True
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"stable_id": sid, "scene_index": idx,
                              "field": key, "render_class": "scene_footage"})
    return {"ok": True}


def _full_order(sids, ov):
    """The complete ordered sid list (incl. removed), deduped, valid-only."""
    valid = set(sids)
    seen, full = set(), []
    for s in (ov.get("order") or sids):
        if s in valid and s not in seen:
            seen.add(s)
            full.append(s)
    for s in sids:               # append any missing (e.g. new baseline scene)
        if s not in seen:
            seen.add(s)
            full.append(s)
    return full


def move_scene(run_dir, scene_index: int, delta: int) -> dict:
    """Move the scene with baseline `scene_index` by `delta` (±1) among the
    currently-visible scenes (keyed by stable id, reorder-safe)."""
    base = _load_json(Path(run_dir) / "script.baseline.json") \
        or _load_json(Path(run_dir) / "script.json") or {}
    scenes = base.get("scenes", [])
    if not (0 <= scene_index < len(scenes)):
        return {"ok": False, "error": "bad index"}
    sids = [_stable_id(s.get("narration", ""), i) for i, s in enumerate(scenes)]
    target = sids[scene_index]
    ov = load_overrides(run_dir)
    scn = ov.get("scenes", {})
    full = _full_order(sids, ov)
    visible = [s for s in full if not scn.get(s, {}).get("scene_removed")]
    if target not in visible:
        return {"ok": True}
    cur = visible.index(target)
    j = cur + delta
    if j < 0 or j >= len(visible):
        return {"ok": True}
    pa, pb = full.index(target), full.index(visible[j])
    full[pa], full[pb] = full[pb], full[pa]
    ov["order"] = full
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"field": "reorder", "render_class": "timeline_reflow"})
    return {"ok": True}


def reorder_scene(run_dir, scene_index: int, to_visible: int) -> dict:
    """Drag-reorder: move the scene with baseline `scene_index` to visible
    position `to_visible` (true arbitrary-position move, not a bubble-swap).
    Non-destructive — only rewrites `ov['order']` (stable-id keyed); original
    assets are never touched. Removed scenes keep their absolute slots in the
    full order; only the visible scenes are permuted."""
    base = _load_json(Path(run_dir) / "script.baseline.json") \
        or _load_json(Path(run_dir) / "script.json") or {}
    scenes = base.get("scenes", [])
    if not (0 <= scene_index < len(scenes)):
        return {"ok": False, "error": "bad index"}
    sids = [_stable_id(s.get("narration", ""), i) for i, s in enumerate(scenes)]
    target = sids[scene_index]
    ov = load_overrides(run_dir)
    scn = ov.get("scenes", {})
    full = _full_order(sids, ov)
    visible = [s for s in full if not scn.get(s, {}).get("scene_removed")]
    if target not in visible:
        return {"ok": True}
    cur = visible.index(target)
    # clamp destination into the visible range (after removing target)
    vis_wo = [s for s in visible if s != target]
    to_visible = max(0, min(int(to_visible), len(vis_wo)))
    if to_visible == cur:
        return {"ok": True}                       # no-op
    new_visible = vis_wo[:to_visible] + [target] + vis_wo[to_visible:]
    # rebuild full: removed sids stay pinned, visible slots refilled in new order
    it = iter(new_visible)
    new_full = [s if scn.get(s, {}).get("scene_removed") else next(it) for s in full]
    ov["order"] = new_full
    _save_overrides(run_dir, ov)
    _history_append(run_dir, {"field": "reorder", "render_class": "timeline_reflow"})
    return {"ok": True}


def _effective_order(base_scenes, ov):
    """(sids, base_by_sid, ordered_visible_sids) honoring order + scene_removed.
    Deduped + valid-only so a stale/duplicated order can't multiply scenes."""
    sids = [_stable_id(s.get("narration", ""), i) for i, s in enumerate(base_scenes)]
    base_by_sid = {sid: i for i, sid in enumerate(sids)}
    scn = ov.get("scenes", {})
    full = _full_order(sids, ov)
    visible = [s for s in full if not scn.get(s, {}).get("scene_removed")]
    return sids, base_by_sid, visible


def _rekey_variants(run_dir, name, eff_sids, base_by_sid, scn_ov, regen_key):
    old = _load_json(Path(run_dir) / name) or {}
    new = {}
    for newidx, sid in enumerate(eff_sids):
        val = int(old.get(str(base_by_sid.get(sid, -1)), 0))
        if scn_ov.get(sid, {}).get(regen_key):
            val += 1
        if val:
            new[str(newidx)] = val
    (Path(run_dir) / name).write_text(json.dumps(new), encoding="utf-8")


def _rendered_sig_path(run_dir) -> Path:
    return Path(run_dir) / "edits" / "rendered_sig.txt"


def _overrides_sig(ov: dict) -> str:
    """Stable hash of the edit-bearing parts of the overrides (scenes + order +
    global) so the editor can tell whether the current edits are ALREADY baked
    into the latest render (-> 'all edits applied' vs '● N unsaved')."""
    payload = {"scenes": ov.get("scenes", {}), "order": ov.get("order"),
               "global": ov.get("global", {})}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def mark_rendered(run_dir) -> dict:
    """Call AFTER a successful re-render. (1) Clears the one-shot regen_visual /
    regen_voice flags so a later re-export does NOT silently re-roll already-
    regenerated visuals/voices (the variant counter is already bumped + cached).
    (2) Records a signature of the now-rendered edits so the unsaved indicator
    reads 'all edits applied' until the user changes something. Non-destructive
    to baseline/script.json; snapshot=False so it never pollutes the undo stack."""
    ov = load_overrides(run_dir)
    cleared = False
    for sid in list(ov.get("scenes", {}).keys()):
        sc = ov["scenes"][sid]
        if sc.pop("regen_visual", None) is not None:
            cleared = True
        if sc.pop("regen_voice", None) is not None:
            cleared = True
        if not sc:
            ov["scenes"].pop(sid, None)
    if cleared:
        _save_overrides(run_dir, ov, snapshot=False)
    p = _rendered_sig_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_overrides_sig(ov), encoding="utf-8")
    return {"ok": True, "cleared_regen": cleared}


def refresh_render_metrics(run_dir) -> dict:
    """P3 — after a successful re-render, compute a SAFE subset of real QA
    metrics (sustained black frames + integrated loudness, in ONE ffmpeg pass)
    and write render_metrics.json so the editor's QA status is honest + useful
    (verdict PASS / WARN / FAIL). Best-effort: on ANY failure it writes NOTHING
    (the editor keeps showing 'quality check not available' rather than stale or
    fake data). Deliberately omits card_types so the card 'rendered' display
    keeps its safe default."""
    run_dir = Path(run_dir)
    mp4 = next(iter(sorted(run_dir.glob("*.mp4"))), None)
    if mp4 is None:
        return {"ok": False, "error": "no mp4"}
    try:
        import re as _re
        import subprocess
        from .ffmpeg_tool import ffmpeg_exe
        r = subprocess.run(
            [ffmpeg_exe(), "-nostats", "-i", str(mp4),
             "-vf", "blackdetect=d=0.5:pic_th=0.98", "-af", "ebur128",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=900)
        err = r.stderr or ""
        black_spans = err.count("black_start")
        _lufs = _re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", err)
        lufs = round(float(_lufs[-1]), 1) if _lufs else None
        problems = []
        if black_spans > 0:
            problems.append(f"{black_spans} black gap(s)")
        if lufs is not None and not (-20.0 <= lufs <= -12.0):
            problems.append(f"loudness {lufs} LUFS (target ~-16)")
        verdict = "FAIL" if black_spans > 0 else ("WARN" if problems else "PASS")
        metrics = {
            "schema": "render_metrics/editor-1",
            "generated_by": "editor_rerender",
            "black_frames": black_spans,
            "audio": {"lufs": lufs},
            "qa": {"verdict": verdict,
                   "summary": ("; ".join(problems) if problems
                               else "No issues found in the final video.")},
        }
        (run_dir / "render_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
        return {"ok": True, "verdict": verdict, "black": black_spans, "lufs": lufs}
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "error": str(e)}


def pending_summary(run_dir) -> dict:
    """Human summary of pending edits + the heaviest render class (for the
    'Apply & re-render' confirmation). render_class: card_rebake (cheap, no LLM)
    < scene_footage < full."""
    ov = load_overrides(run_dir)
    scn = ov["scenes"]
    g = ov.get("global", {})
    n_card = n_remove = n_visual = n_prompt = n_scene_rm = n_regen = 0
    for sc in scn.values():
        n_card += bool(sc.get("card_text_override"))
        n_remove += bool(sc.get("card_removed"))
        n_visual += bool(sc.get("visual_override"))
        n_prompt += bool(sc.get("visual_prompt_override"))
        n_scene_rm += bool(sc.get("scene_removed"))
        n_regen += bool(sc.get("regen_visual") or sc.get("regen_voice"))
    reordered = bool(ov.get("order"))
    if n_scene_rm or reordered:
        rclass = "timeline_reflow"
    elif n_visual or n_prompt or n_regen:
        rclass = "scene_footage"
    elif n_card or n_remove:
        rclass = "card_rebake"
    elif g:
        rclass = "audio_only"
    else:
        rclass = "none"
    try:
        _stored = _rendered_sig_path(run_dir).read_text(encoding="utf-8").strip()
        rendered_clean = bool(_stored) and (_stored == _overrides_sig(ov))
    except Exception:                                          # noqa: BLE001
        rendered_clean = False
    return {"edited_scenes": len(scn), "card_edits": n_card, "cards_removed": n_remove,
            "visual_uploads": n_visual, "prompt_edits": n_prompt,
            "scenes_removed": n_scene_rm, "regen": n_regen, "reordered": reordered,
            "global": g, "render_class": rclass, "llm_cost": False,
            "rendered_clean": rendered_clean}


def apply_overrides(run_dir) -> dict:
    """Fold ALL non-destructive overrides into render-ready script.json/script.txt,
    keeping a pristine script.baseline.json. Recomputes source_sha256 from the
    (possibly reordered/trimmed) narration so the pipeline reuse-gate matches and
    SKIPS the LLM (no Claude cost). Re-keys variant counters across reorder/remove
    and bumps them for regen. Idempotent (always from baseline). Globals returned
    for the caller to apply to cfg/env."""
    run_dir = Path(run_dir)
    script_p = run_dir / "script.json"
    base_p = run_dir / "script.baseline.json"
    if not script_p.exists():
        return {"ok": False, "error": "no script.json"}
    if not base_p.exists():
        shutil.copyfile(script_p, base_p)
    base = _load_json(base_p) or {}
    title = base.get("title", "")
    base_scenes = base.get("scenes", [])
    ov = load_overrides(run_dir)
    scn = ov.get("scenes", {})
    sids, base_by_sid, visible = _effective_order(base_scenes, ov)

    eff, changed, uploads = [], [], []
    for sid in visible:
        s = dict(base_scenes[base_by_sid[sid]])
        osc = scn.get(sid, {})
        cto = osc.get("card_text_override")
        if cto:
            s["graphic_text"] = cto.get("graphic_text", s.get("graphic_text", ""))
            s["graphic_body"] = cto.get("graphic_body", s.get("graphic_body", ""))
            changed.append("card_text")
        if osc.get("card_removed"):
            s["graphic_kind"] = s["graphic_text"] = s["graphic_body"] = ""
            changed.append("card_removed")
        if osc.get("visual_prompt_override"):
            s["visual"] = osc["visual_prompt_override"]
            changed.append("prompt")
        if osc.get("visual_override"):
            uploads.append({"sid": sid, **osc["visual_override"]})
        eff.append((sid, s))

    body = f"{title}\n\n" + "\n\n".join(s.get("narration", "") for _, s in eff)
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    out = {"title": title, "source_sha256": _sha_body(body),
           "scenes": [s for _, s in eff]}
    script_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    eff_sids = [sid for sid, _ in eff]
    _rekey_variants(run_dir, "variants.json", eff_sids, base_by_sid, scn, "regen_visual")
    _rekey_variants(run_dir, "voice_variants.json", eff_sids, base_by_sid, scn, "regen_voice")
    # Review-Editor REPLACE-VISUAL: record user uploads / search-picks against
    # their FINAL (post-reorder/trim) scene index so the renderer uses them
    # directly. Read back in pipeline.py right after fetch_footage (footage.py
    # stays untouched). Written every time (empty {} clears a removed override).
    locked = {}
    for new_idx, (sid, _s) in enumerate(eff):
        vo = (scn.get(sid, {}) or {}).get("visual_override") or {}
        uf = vo.get("user_file")
        if uf:
            locked[str(new_idx)] = {"file": uf, "is_video": bool(vo.get("is_video"))}
    lv_p = run_dir / "edits" / "locked_visuals.json"
    lv_p.parent.mkdir(parents=True, exist_ok=True)
    lv_p.write_text(json.dumps(locked), encoding="utf-8")
    return {"ok": True, "scenes": len(eff), "changed": changed,
            "uploads": uploads, "global": ov.get("global", {}),
            "removed": len(base_scenes) - len(eff)}


# ── main builder ─────────────────────────────────────────────────────────
def build_manifest(run_dir) -> dict:
    run_dir = Path(run_dir)
    script = _load_json(run_dir / "script.json") or {}
    meta = _load_json(run_dir / "render_meta.json") or {}
    metrics = _load_json(run_dir / "render_metrics.json") or {}
    brief = _load_json(run_dir / "brief.json") or {}
    scenes_in = script.get("scenes", [])

    # registry (human labels, banned set) — defensive import
    try:
        from . import templates as T
        _get, _banned = T.get, set()
        try:
            from .script_gen import _BANNED_TEMPLATES as _banned
        except Exception:                                      # noqa: BLE001
            _banned = {"era_banner", "comparison", "stat_dashboard"}
    except Exception:                                          # noqa: BLE001
        _get, _banned = (lambda k: None), set()

    # video file + mtime
    mp4 = next(iter(sorted(run_dir.glob("*.mp4"))), None)
    srt = next(iter(sorted(run_dir.glob("*.srt"))), None)
    video_seconds = float(meta.get("video_seconds") or 0.0) or None

    # ── timing: word-proportional, scaled to the true total ──────────────
    words = [max(1, len((s.get("narration") or "").split())) for s in scenes_in]
    total_words = float(sum(words)) or 1.0
    if not video_seconds:
        video_seconds = round(total_words / _WORDS_PER_SECOND, 2)
        dur_src = "estimated"
    else:
        dur_src = "estimated_scaled"
    # P2 — prefer the EXACT per-scene start/duration that assemble emits into
    # render_meta.json (ground truth) over the word-proportional estimate, when
    # it matches the current scene count (else fall back gracefully).
    _meta_starts = meta.get("scene_starts")
    _meta_durs = meta.get("scene_durations")
    _accurate_timing = (
        isinstance(_meta_starts, list) and isinstance(_meta_durs, list)
        and len(scenes_in) > 0
        and len(_meta_starts) == len(scenes_in) == len(_meta_durs)
    )
    if _accurate_timing:
        dur_src = "render_meta"
    srt_cues = _parse_srt(srt) if srt else []

    # ── card "rendered vs skipped" heuristic from render_metrics.card_types ─
    card_types = dict((metrics.get("card_types") or {}))
    metrics_present = bool(metrics)
    kind_running: dict = {}

    overrides = load_overrides(run_dir)["scenes"]

    scenes = []
    t = 0.0
    for i, s in enumerate(scenes_in):
        if _accurate_timing:
            start = round(float(_meta_starts[i]), 3)
            dur = round(float(_meta_durs[i]), 3)
            end = round(start + dur, 3)
        else:
            dur = round(video_seconds * (words[i] / total_words), 3)
            start, end = round(t, 3), round(t + dur, 3)
            t = end
        sid = _stable_id(s.get("narration", ""), i)
        osc = overrides.get(sid) or next(
            (v for v in overrides.values() if v.get("scene_index_hint") == i), {})
        dirty_classes = []

        kind = (s.get("graphic_kind") or "").strip()
        card = None
        card_removed = bool(osc.get("card_removed"))
        if kind:
            tpl = _get(kind)
            seen = kind_running.get(kind, 0)
            avail = card_types.get(kind)
            rendered = (seen < avail) if (metrics_present and avail is not None) else True
            kind_running[kind] = seen + 1
            # apply card-text override (effective values shown in the editor)
            cto = osc.get("card_text_override") or {}
            eff_text = cto.get("graphic_text", s.get("graphic_text", ""))
            eff_body = cto.get("graphic_body", s.get("graphic_body", ""))
            if cto or card_removed:
                dirty_classes.append("card_rebake")
            card = {
                "graphic_kind": kind,
                "title_label": getattr(tpl, "title", None) or kind.replace("_", " ").title(),
                "uses_body": bool(getattr(tpl, "uses_body", True)),
                "is_banned": kind in _banned,
                "cap": getattr(tpl, "max_per_video", None),
                "min_gap_scenes": getattr(tpl, "min_gap_scenes", None),
                "graphic_text": eff_text,
                "graphic_body": eff_body,
                "fields": _card_fields(kind, eff_text, eff_body),
                "rendered": bool(rendered) and not card_removed,
                "removed": card_removed,
                "edited": bool(cto),
                "authored_but_skipped": bool(metrics_present and avail is not None and not rendered),
            }

        # visual / prompt / regen / scene-removal overrides
        vo = osc.get("visual_override") or {}
        if vo:
            dirty_classes.append("scene_footage")
        prompt_edited = bool(osc.get("visual_prompt_override"))
        if prompt_edited:
            dirty_classes.append("prompt")
        regen = bool(osc.get("regen_visual") or osc.get("regen_voice"))
        if regen:
            dirty_classes.append("scene_footage")
        scene_removed = bool(osc.get("scene_removed"))
        if scene_removed:
            dirty_classes.append("scene_removed")
        eff_visual = osc.get("visual_prompt_override") or s.get("visual", "")
        source_badge = ("Upload" if vo else ("Card" if (card and card["rendered"]) else "Footage"))

        cues = [c for c in srt_cues if start <= (c["start"] + c["end"]) / 2 < end]
        inten = int(s.get("intensity") or 0)
        energy = inten if 1 <= inten <= 5 else 2
        scenes.append({
            "scene_index": i,
            "stable_id": sid,
            "role": s.get("role", ""),
            "shot_type": s.get("shot_type", ""),
            "narration": s.get("narration", ""),
            "keywords": s.get("keywords", []),
            "visual": eff_visual,
            "visual_prompt_edited": prompt_edited,
            "intensity": inten,
            "energy": energy,
            "emphasis": s.get("emphasis", ""),
            "start": start, "end": end, "duration": dur,
            "duration_source": dur_src,
            "card": card,
            "source_badge": source_badge,
            "removed_scene": scene_removed,
            "captions": {"enabled": bool(brief.get("captions")),
                         "preview_cues": cues[:6]},
            "edit_status": {"visual_variant": 0, "voice_variant": 0,
                            "has_override": bool(osc),
                            "dirty": bool(dirty_classes),
                            "dirty_classes": dirty_classes,
                            "visual_upload": vo.get("user_file") if vo else None,
                            "prompt_edited": prompt_edited, "regen": regen,
                            "layers_off": list(osc.get("layers_off") or []),
                            "removed_scene": scene_removed},
        })

    # variant counters (existing per-scene regen state)
    variants = _load_json(run_dir / "variants.json") or {}
    vvariants = _load_json(run_dir / "voice_variants.json") or {}
    for sc in scenes:
        sc["edit_status"]["visual_variant"] = int(variants.get(str(sc["scene_index"]), 0))
        sc["edit_status"]["voice_variant"] = int(vvariants.get(str(sc["scene_index"]), 0))

    # structural overrides reflected LIVE in the editor (reorder + remove) so the
    # scene list/timeline match what the next render will produce
    _ovr = load_overrides(run_dir)
    _order = _ovr.get("order")
    _n_before = len(scenes)
    if _order:
        _by = {sc["stable_id"]: sc for sc in scenes}
        _oset = set(_order)
        scenes = [_by[s] for s in _order if s in _by] + \
                 [sc for sc in scenes if sc["stable_id"] not in _oset]
    scenes = [sc for sc in scenes if not sc.get("removed_scene")]
    _t = 0.0
    for sc in scenes:
        sc["start"] = round(_t, 3)
        sc["end"] = round(_t + sc["duration"], 3)
        _t = sc["end"]
    structure_edited = bool(_order) or len(scenes) != _n_before

    render = {
        "video_seconds": video_seconds, "fps": meta.get("fps"),
        "scenes": meta.get("scenes", len(scenes)), "beats": meta.get("beats"),
        "cuts": meta.get("cuts"), "transitions_motivated": meta.get("transitions_motivated"),
        "shot_len_s": meta.get("shot_len_s"),
        "metrics_present": metrics_present,
        "script_words": metrics.get("script_words"),
        "ai_images_used": metrics.get("ai_images_used"),
        "ai_images_cap": metrics.get("ai_images_cap"),
        "music_cues": metrics.get("music_cues"),
        "graphic_card_images": metrics.get("graphic_card_images"),
        "sources": metrics.get("sources") or {},
        "card_types": card_types,
        "audio": metrics.get("audio") or {},
        "qa": metrics.get("qa") or {},
    }

    kind_catalog = []
    try:
        from . import templates as T
        for name in sorted(T.all_names()):
            tpl = T.get(name)
            kind_catalog.append({
                "kind": name, "title": getattr(tpl, "title", name),
                "uses_body": bool(getattr(tpl, "uses_body", True)),
                "is_banned": name in _banned})
    except Exception:                                          # noqa: BLE001
        pass

    _g = _ovr.get("global", {})
    warnings = []
    if _g.get("captions_enabled") is False:
        warnings.append("Captions OFF")
    if _g.get("music_enabled") is False:
        warnings.append("Music OFF")
    if _g.get("music_volume") is not None and _g.get("music_volume") != 1:
        warnings.append(f"Music volume {int(float(_g['music_volume'])*100)}%")
    _n_removed = sum(1 for v in _ovr.get("scenes", {}).values() if v.get("scene_removed"))
    if _n_removed:
        warnings.append(f"{_n_removed} scene(s) deleted")
    if _ovr.get("order"):
        warnings.append("Scenes reordered")
    _qa = render.get("qa", {})
    if _qa.get("verdict") == "FAIL":
        warnings.append("QA: render issues — re-render recommended")
    elif _qa.get("verdict") == "WARN":
        warnings.append("QA: minor warnings")
    if not render.get("metrics_present"):
        warnings.append("Quality check not available for this preview yet")

    return {
        "schema": SCHEMA,
        "slug": run_dir.name,
        "title": script.get("title", run_dir.name),
        "source_sha256": script.get("source_sha256", ""),
        "video_present": bool(mp4),
        "video_mtime": (mp4.stat().st_mtime if mp4 else None),
        "duration_pretty": _mmss(video_seconds or 0),
        "assets": {"video": mp4.name if mp4 else None,
                   "srt": srt.name if srt else None,
                   "thumbnail": "thumbnail.jpg" if (run_dir / "thumbnail.jpg").exists() else None},
        "render": render,
        "kind_catalog": kind_catalog,
        "global": _ovr.get("global", {}),
        "editor_signature": meta.get("editor_signature") or {},  # DOC_012
        # Editorial RECIPE (Layer 2) — the per-video pick within the niche
        # envelope.  Powers the Review-Editor recipe chip + 'Lock Look'.
        "editorial_recipe": meta.get("editorial_recipe") or {},
        "editorial_recipe_summary": meta.get("editorial_recipe_summary") or "",
        "structure_edited": structure_edited,
        "warnings": warnings,
        "scenes": scenes,
    }


def _derive_timeline(m: dict) -> dict:
    total = m["render"].get("video_seconds") or 0.0
    fps = m["render"].get("fps") or 30
    visual, card, caption, voice = [], [], [], []
    for sc in m["scenes"]:
        visual.append({
            "scene_index": sc["scene_index"], "stable_id": sc["stable_id"],
            "start": sc["start"], "end": sc["end"],
            "label": (sc["narration"][:48] + "…") if len(sc["narration"]) > 48 else sc["narration"],
            "thumb": f"editor_cache/thumbs/sc{sc['scene_index']:03d}.jpg",
            "badges": [b for b in [sc["shot_type"], sc["role"]] if b],
            "source_badge": sc["source_badge"]})
        voice.append({"scene_index": sc["scene_index"], "start": sc["start"], "end": sc["end"]})
        if sc["card"] and sc["card"]["rendered"]:
            card.append({"scene_index": sc["scene_index"],
                         "kind_label": sc["card"]["title_label"],
                         "graphic_text": sc["card"]["graphic_text"],
                         "start": round(sc["start"] + 0.35, 2),
                         "end": round(max(sc["start"] + 0.5, sc["end"] - 0.2), 2)})
        for c in sc["captions"]["preview_cues"]:
            caption.append({**c, "scene_index": sc["scene_index"]})
    n_cues = m["render"].get("music_cues")
    music = ([{"start": 0.0, "end": total,
               "label": f"Score · {n_cues} cue{'s' if (n_cues or 0) != 1 else ''}"}]
             if total else [])
    return {"schema": TIMELINE_SCHEMA, "slug": m["slug"], "total": total, "fps": fps,
            "tracks": {"visual": visual, "card": card, "caption": caption,
                       "music": music, "voice": voice},
            "transitions_motivated": m["render"].get("transitions_motivated")}


def extract_scene_thumbs(run_dir, manifest, *, width: int = 320) -> int:
    """One poster frame per scene seeked into the final MP4. Best-effort."""
    run_dir = Path(run_dir)
    vid = manifest["assets"].get("video")
    if not vid or not (run_dir / vid).exists():
        return 0
    try:
        from .ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        return 0
    tdir = run_dir / "editor_cache" / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    # P2 — cache reuse, but only if the thumb is NEWER than the mp4 (else it is
    # stale after a re-render and must be regenerated). Uploaded-poster thumbs
    # (regen_thumb_from_upload) are also newer-than-mp4 so they survive.
    try:
        _mp4_mtime = (run_dir / vid).stat().st_mtime
    except Exception:                                          # noqa: BLE001
        _mp4_mtime = 0
    made = 0
    for sc in manifest["scenes"]:
        out = tdir / f"sc{sc['scene_index']:03d}.jpg"
        if out.exists() and out.stat().st_size > 200 and out.stat().st_mtime >= _mp4_mtime:
            made += 1
            continue
        ss = max(0.0, sc["start"] + min(0.6, sc["duration"] * 0.3))
        try:
            r = subprocess.run([ff, "-y", "-ss", f"{ss:.2f}", "-i", str(run_dir / vid),
                                "-frames:v", "1", "-vf", f"scale={width}:-1",
                                str(out)], capture_output=True, timeout=60)
            if r.returncode == 0 and out.exists():
                made += 1
        except Exception:                                      # noqa: BLE001
            pass
    return made


def _ed_thumb_path(run_dir, idx: int) -> Path:
    return Path(run_dir) / "editor_cache" / "thumbs" / f"sc{idx:03d}.jpg"


def regen_thumb_from_upload(run_dir, idx: int, src_rel: str,
                            is_video: bool = False, *, width: int = 320) -> bool:
    """Regenerate ONE scene's editor poster from an uploaded image/video so the
    left scene-row thumbnail reflects the replacement immediately (no re-render).
    Best-effort. Backs up the original poster once (scNNN.orig.jpg) so a later
    Reset Visual can restore it."""
    try:
        from .ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        return False
    src = Path(run_dir) / src_rel
    if not src.exists():
        return False
    out = _ed_thumb_path(run_dir, idx)
    out.parent.mkdir(parents=True, exist_ok=True)
    orig = out.with_suffix(".orig.jpg")
    if out.exists() and not orig.exists():
        try:
            shutil.copy2(out, orig)
        except Exception:                                      # noqa: BLE001
            pass
    cmd = [ff, "-y"]
    if is_video:
        cmd += ["-ss", "0.5"]
    cmd += ["-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-1", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        return r.returncode == 0 and out.exists()
    except Exception:                                          # noqa: BLE001
        return False


def restore_thumb(run_dir, idx: int) -> bool:
    """Restore a scene's original poster after Reset Visual (if it was backed up)."""
    out = _ed_thumb_path(run_dir, idx)
    orig = out.with_suffix(".orig.jpg")
    try:
        if orig.exists():
            shutil.copy2(orig, out)
            orig.unlink()
            return True
    except Exception:                                          # noqa: BLE001
        pass
    return False


def cleanup_orphan_upload(run_dir, user_file: str, ov: dict) -> bool:
    """Delete an uploaded scratch replacement that no override references any more.
    Hard safety: only ever removes files under edits/uploads/ (original generated
    assets live elsewhere and are never touched); skips files still referenced by
    another scene's override."""
    rel = str(user_file or "")
    if not rel.startswith("edits/uploads/"):
        return False
    for sc in (ov.get("scenes") or {}).values():
        if (sc.get("visual_override") or {}).get("user_file") == rel:
            return False                                       # still in use
    p = Path(run_dir) / rel
    try:
        if p.exists() and p.is_file():
            p.unlink()
            return True
    except Exception:                                          # noqa: BLE001
        pass
    return False


def write_manifest(run_dir, *, thumbs: bool = True) -> Path:
    run_dir = Path(run_dir)
    mp = run_dir / "editor_manifest.json"
    with _build_lock(run_dir):
        m = build_manifest(run_dir)
        _atomic_write(mp, json.dumps(m, indent=2, ensure_ascii=False))
        ec = run_dir / "editor_cache"
        ec.mkdir(parents=True, exist_ok=True)
        _atomic_write(ec / "timeline.json",
                      json.dumps(_derive_timeline(m), indent=2, ensure_ascii=False))
        if thumbs:
            try:
                extract_scene_thumbs(run_dir, m)
            except Exception:                                  # noqa: BLE001
                pass
    return mp


if __name__ == "__main__":
    import sys
    p = write_manifest(sys.argv[1], thumbs="--no-thumbs" not in sys.argv)
    print("wrote", p)
