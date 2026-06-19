#!/usr/bin/env python3
"""Black-frame deleted-scene-reflow regression harness (cost-locked, $0).

Drives 7 edit scenarios through the REAL editor export path
(apply_overrides -> render_from_script), reusing the 732 MB content-hashed
cache (LLM/TTS/footage all skipped). For each scenario it records:
  - apply_overrides result (scenes kept / removed / changed)
  - the final MP4's black spans (repair detector d=0.30 AND stricter d=0.15)
  - the render_black_frame_metrics.json sidecar (pre-repair count, per-span
    classification, repair_method, anchor luma/direction, resolved, result)
  - audio (mean/max dB), stream presence, duration
The project is restored byte-identical from /tmp/relsnap_full afterwards.
"""
import os, json, shutil, time, sys, subprocess, re, traceback
from pathlib import Path

# ───────── COST LOCK (no paid API; cache-only) ─────────
os.environ["VIDLORE_AIMG"] = "0"
os.environ["FAL_KEY"] = ""
os.environ["VIDLORE_SHUTTERSTOCK"] = "0"
os.environ["WEB_FOOTAGE_ENGINE"] = "0"
os.environ["WEB_IMAGE_ENGINE"] = "0"
os.environ["VIDLORE_TTS_BACKEND"] = "legacy"
os.environ["VIDLORE_MUSIC_VOLUME"] = "0.5"
os.environ.setdefault("VIDLORE_ENCODE_WORKERS", "6")

from vidlore import editor_manifest as EM
from vidlore import assemble as A
from vidlore.config import load_config
from vidlore.pipeline import load_brief, render_from_script

SLUG = "pablo-escobar--the-rise-and-fall-of-the-king-of-co"
OUT = Path("output")
d = OUT / SLUG
SNAP = Path("/tmp/relsnap_full")
RESULTS = Path("/tmp/blackreg"); RESULTS.mkdir(exist_ok=True)
GLOBALS = {"captions_enabled": False, "music_volume": 0.5, "look_preset": "auto"}
FF = ("/Users/hussnain/Library/Python/3.9/lib/python/site-packages/"
      "imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
MP4 = d / (SLUG + ".mp4")


def log(m): print(m, flush=True)


def write_overrides(scenes):
    ov = {"schema": "editor_overrides/1", "slug": SLUG,
          "global": dict(GLOBALS), "scenes": scenes, "order": None}
    (d / "edits" / "user_overrides.json").write_text(
        json.dumps(ov, indent=2), encoding="utf-8")


# SIDs MUST come from the STABLE baseline (scene_stable_id reads the current
# render-ready script.json, whose indices shift after a deletion). apply_overrides
# matches override keys against baseline SIDs computed exactly this way.
_BASE_SCENES = json.loads((d / "script.baseline.json").read_text())["scenes"]


def sid(idx):
    return EM._stable_id(_BASE_SCENES[idx].get("narration", ""), idx)


def audio_audit(mp4):
    r = subprocess.run([FF, "-i", str(mp4), "-af", "volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    se = r.stderr
    dm = re.search(r"Duration: (\d+):(\d+):([\d.]+)", se)
    durs = (int(dm.group(1)) * 3600 + int(dm.group(2)) * 60
            + float(dm.group(3))) if dm else 0.0
    mean = re.search(r"mean_volume: ([\-\d.]+) dB", se)
    mx = re.search(r"max_volume: ([\-\d.]+) dB", se)
    return {"duration_s": round(durs, 2),
            "mean_db": float(mean.group(1)) if mean else None,
            "max_db": float(mx.group(1)) if mx else None,
            "has_video": "Video:" in se, "has_audio": "Audio:" in se}


def restore():
    for f in SNAP.glob("*"):
        if f.name == "_edits_user_overrides.json":
            shutil.copy2(f, d / "edits" / "user_overrides.json")
        else:
            shutil.copy2(f, d / f.name)
    (d / "render_black_frame_metrics.json").unlink(missing_ok=True)
    for w in d.glob("work_*"):
        if w.is_dir():
            shutil.rmtree(w, ignore_errors=True)


# ───────── scenarios (executed REPRO-first to de-risk) ─────────
def ov_repro():
    return {
        sid(1): {"scene_index_hint": 1, "card_text_override": {
            "graphic_kind": "name_reveal", "graphic_text": "EL PATRÓN",
            "graphic_body": "THE BOSS OF BOSSES"}},
        sid(4): {"scene_index_hint": 4, "card_removed": True},
        sid(35): {"scene_index_hint": 35, "scene_removed": True},
    }


SCENARIOS = [
    ("4_delete_late_sc35_REPRO", ov_repro),
    ("1_baseline_no_deletion", lambda: {}),
    ("2_delete_early_sc3",
     lambda: {sid(3): {"scene_index_hint": 3, "scene_removed": True}}),
    ("3_delete_middle_sc19",
     lambda: {sid(19): {"scene_index_hint": 19, "scene_removed": True}}),
    ("5_delete_two_adjacent_sc19_20",
     lambda: {sid(19): {"scene_index_hint": 19, "scene_removed": True},
              sid(20): {"scene_index_hint": 20, "scene_removed": True}}),
    ("6_reorder_no_deletion", "REORDER"),
    ("7_reset_all_edits", lambda: {}),
]

restore()                     # clean start (undo any pre-flight residue)
base_sha_before = (d / "script.baseline.json").read_bytes()
summary = []
t_all = time.time()
try:
    for name, builder in SCENARIOS:
        log(f"\n===== SCENARIO {name} =====")
        t0 = time.time()
        rec = {"scenario": name}
        try:
            if builder == "REORDER":
                write_overrides({})
                EM.apply_overrides(d)           # establish baseline script.json
                EM.reorder_scene(d, 5, 10)      # visible indices now == baseline
            else:
                write_overrides(builder())
            ap = EM.apply_overrides(d)
            rec["apply"] = {"ok": ap.get("ok"), "scenes": ap.get("scenes"),
                            "removed": ap.get("removed"),
                            "changed": ap.get("changed")}
            brief = load_brief(d)
            cfg = load_config()
            g = ap.get("global", {})
            if g.get("captions_enabled") is not None:
                brief.captions = bool(g["captions_enabled"])
            if g.get("music_volume") is not None:
                os.environ["VIDLORE_MUSIC_VOLUME"] = str(float(g["music_volume"]))
            lp = (g.get("look_preset") or "").strip()
            if lp and lp.lower() != "auto":
                try:
                    brief.look_preset = lp
                except Exception:
                    pass
            (d / "render_black_frame_metrics.json").unlink(missing_ok=True)
            render_from_script(brief, cfg, OUT)
            rec["mp4_bytes"] = MP4.stat().st_size if MP4.exists() else 0
            rec["final_black_spans_d0.30"] = [
                (round(s, 2), round(e, 2))
                for s, e in A._detect_black_spans(MP4, min_d=0.30, pix_th=0.10)]
            rec["final_black_spans_d0.15"] = [
                (round(s, 2), round(e, 2))
                for s, e in A._detect_black_spans(MP4, min_d=0.15, pix_th=0.10)]
            mp = d / "render_black_frame_metrics.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                rec["repair"] = {
                    "before": m.get("before_scan_span_count"),
                    "after": m.get("after_scan_span_count"),
                    "result": m.get("result"),
                    "unresolved": m.get("unresolved_repair_count"),
                    "preserved": m.get("preserved_count"),
                    "spans": [{k: sp.get(k) for k in (
                        "start_s", "end_s", "duration_s", "classification",
                        "repair_method", "anchor_direction", "anchor_luma",
                        "anchor_is_dark", "resolved")}
                        for sp in m.get("spans", [])]}
                shutil.copy2(mp, RESULTS / f"{name}_metrics.json")
            else:
                rec["repair"] = {"note": "0 spans detected — repair not invoked"}
            rec["audio"] = audio_audit(MP4)
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = str(e)[:300]
            rec["trace"] = traceback.format_exc()[-1200:]
            log("  !! SCENARIO FAILED: " + str(e)[:200])
        rec["wall_s"] = round(time.time() - t0, 1)
        summary.append(rec)
        rp = rec.get("repair", {})
        log(f"  -> ok={rec.get('ok')} "
            f"final_d0.30={rec.get('final_black_spans_d0.30')} "
            f"final_d0.15={rec.get('final_black_spans_d0.15')} "
            f"repair={rp.get('result') if isinstance(rp, dict) else None} "
            f"wall={rec['wall_s']}s")
        (RESULTS / "regression_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
finally:
    base_sha_after = (d / "script.baseline.json").read_bytes()
    log(f"\nbaseline untouched: {base_sha_before == base_sha_after}")
    restore()
    log(f"===== ALL DONE in {round(time.time() - t_all, 1)}s — "
        "project restored pristine =====")
    (RESULTS / "regression_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log("RESULTS -> /tmp/blackreg/regression_summary.json")
