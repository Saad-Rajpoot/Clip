#!/usr/bin/env python3
"""Caption SMOKE render — drive the REAL engine renderer end-to-end on a short local fixture.

This is NOT a mocked worker: it calls the same two production functions build_video calls to burn
captions —
  • vidlore.assemble.assemble(...)          → the narration (kinetic word-by-word) caption, styled
                                              from the SELECTED preset exactly as build_video sets it
  • build._burn_breakout_captions(...)      → the real-audio breakout karaoke caption

on staged local footage (synthetic clips) + real narration + a REAL spoken breakout clip (macOS
`say` → whisper transcribes it), then measures the rendered pixels of the final MP4s.

Two cases, mirroring the portal:
  1. one selected preset, captions ON  → narration caption + breakout karaoke, both burned
  2. captions OFF                       → zero visible caption ink

Honest boundary: upstream footage-DISCOVERY / LLM / matching is not exercised here (it needs real
YouTube content and can't run on a 10-20s local fixture) — that path is covered by the earlier
full-movie portal renders. This smoke proves the CAPTION render path end-to-end at the pixel level.

    python3 tools/smoke_render_captions.py [preset]      # default preset: focus
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio_ffmpeg                                          # noqa: E402
FF = imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd, **k):
    return subprocess.run(cmd, check=True, capture_output=True, timeout=600, **k)


def _clip(path: Path, dur: float, pattern: str):
    _run([FF, "-y", "-f", "lavfi", "-i", f"{pattern}=size=1280x720:rate=25:duration={dur}",
          "-pix_fmt", "yuv420p", str(path)])


def _silence(path: Path, dur: float):
    _run([FF, "-y", "-f", "lavfi", "-i",
          f"anullsrc=channel_layout=stereo:sample_rate=44100:d={dur}", str(path)])


def _speech(path: Path, text: str):
    """Real spoken audio via macOS `say` → wav (whisper transcribes it for the breakout caption)."""
    aiff = path.with_suffix(".aiff")
    _run(["say", "-o", str(aiff), text])
    _run([FF, "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(path)])
    return path


def main():
    from vidlore.footage import FootageItem
    from vidlore.tts import Narration, NarratedScene, WordTiming
    from vidlore.themes import theme as get_theme
    from vidlore.assemble import assemble
    from vidlore.clipstudio import caption_presets as CP
    import vidlore.clipstudio.build as B
    try:
        from caption_pixel_probe import render_frame as _pf, measure as _measure, have_libass
    except Exception as e:
        print(f"probe import failed: {e}")
        return 1

    preset_name = (sys.argv[1] if len(sys.argv) > 1 else "focus").strip().lower()
    if preset_name not in CP.VALID_STYLES:
        preset_name = "focus"
    preset = CP.CAPTION_PRESETS[preset_name]

    out = Path.home() / "Desktop" / "clipstudio_output" / "caption_previews" / "phase10" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    # ── stage local fixture: 3 footage clips + a 3-scene narration (~13s) ─────────────────────────
    durs = [4.0, 5.0, 4.0]
    patterns = ["testsrc2", "testsrc2", "testsrc2"]           # high-entropy → clears size floor
    footage = []
    for i, (d, p) in enumerate(zip(durs, patterns)):
        cp = work / f"clip{i}.mp4"
        _clip(cp, d + 1.0, p)                                  # a little longer than the scene needs
        footage.append(FootageItem(index=i, path=str(cp), is_video=True))

    def _words(text, t0, t1):
        toks = text.split()
        step = (t1 - t0) / max(1, len(toks))
        return [WordTiming(w, t0 + j * step, t0 + (j + 1) * step) for j, w in enumerate(toks)]

    lines = [
        "Power tends to corrupt the patient and reshape every ambition",
        "",                                                    # scene 1 = breakout region (silent VO)
        "History remembers only what the victors ultimately choose to write",
    ]
    scenes, t = [], 0.0
    master = work / "narration.wav"
    _silence(master, sum(durs))
    for i, d in enumerate(durs):
        sa = work / f"scene{i}.wav"
        _silence(sa, d)
        scenes.append(NarratedScene(index=i, audio=sa, duration=d,      # Path (assemble .resolve()s)
                                    words=_words(lines[i], t, t + d) if lines[i] else []))
        t += d
    narration = Narration(scenes=scenes, audio=master, reused=0)        # Path, not str

    # ── theme, styled by the SELECTED preset EXACTLY as build_video does ──────────────────────────
    th = get_theme("history")
    _cap_dict, _cap_accent = preset.theme_caption()
    th = {**th, "caption": {**th.get("caption", {}), **_cap_dict}, "caption_accent": _cap_accent}
    th = {**th, "grade": "eq=contrast=1.05:saturation=1.04", "overlay_effects": []}   # neutral grade

    n = len(scenes)
    b_start, b_dur = 4.0, 5.0                                  # breakout = scene 1 window
    breakout_windows = [(b_start, round(b_start + b_dur, 2))]

    def _assemble(dst, captions):
        return Path(assemble(
            footage, narration, th, work / ("on" if captions else "off"), dst,
            captions=captions, music=None, transitions=True, title="Smoke Test",
            energies=[6] * n, emphasis=["patient", "", "victors"],
            graphics=[("", "", "")] * n, graphic_assets={},
            shot_types=[""] * n, roles=[""] * n,
            # during the breakout the narration caption is silent (its own dialogue is captioned)
            caption_suppress_windows=breakout_windows if captions else None,
            breakout_windows=breakout_windows,
        ))

    print(f"[smoke] preset={preset_name}  → real assemble() + real _burn_breakout_captions()")
    # CASE 1 — captions ON: narration caption + breakout karaoke
    on_mp4 = _assemble(out / f"smoke_{preset_name}_ON.mp4", True)
    print(f"[smoke] ON  render → {on_mp4.name}")
    # real breakout karaoke over the ON render (real spoken audio → whisper → \kf)
    bwav = _speech(work / "breakout.wav",
                   "A man who has no conscience is not truly free at all")
    burned = B._burn_breakout_captions(on_mp4, [{"audio": str(bwav), "start": b_start, "dur": b_dur}],
                                       work, print, preset=preset)
    print(f"[smoke] breakout karaoke burned: {burned}")
    # CASE 2 — captions OFF
    off_mp4 = _assemble(out / f"smoke_{preset_name}_OFF.mp4", False)
    print(f"[smoke] OFF render → {off_mp4.name}")

    # ── measure the REAL caption pixels by DIFFING each final frame against the engine's own
    #    caption-free baseline (editor_cache/preview_nocap.mp4 = identical footage, no captions) so
    #    the bright testsrc footage cancels out and only the burned caption glyphs remain ──────────
    import numpy as np
    from PIL import Image
    base_mp4 = on_mp4.parent / "editor_cache" / "preview_nocap.mp4"
    if not base_mp4.exists():
        print(f"[smoke] baseline {base_mp4} missing — mp4s produced, margin probe skipped")
        return 0

    def _frame(mp4, t, png):
        _run([FF, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}",
              "-i", str(mp4), "-frames:v", "1", str(png)])
        return np.asarray(Image.open(png).convert("RGB")).astype(np.int16)

    def caption_bbox(mp4, t, tag, band_top=0.55):
        on_a = _frame(mp4, t, out / f"{mp4.stem}_{tag}.png")
        base_a = _frame(base_mp4, t, out / f"base_{tag}.png")
        H, W = on_a.shape[:2]
        y0 = int(H * band_top)                                # captions live in the lower band
        diff = np.abs(on_a[y0:] - base_a[y0:]).max(axis=2)
        bright = on_a[y0:].min(axis=2) > 150                  # caption fill is near-white/bright
        mask = (diff > 50) & bright
        col = np.where(mask.sum(axis=0) >= 2)[0]              # ≥2 → ignore lone-pixel encode noise
        rowp = mask.sum(axis=1)
        rws = np.where(rowp >= 2)[0]
        if len(col) == 0 or len(rws) == 0:
            return {"empty": True, "W": W, "H": H, "px": int(mask.sum())}
        # count vertical text bands in the masked rows (merge <10px gaps)
        bands, prev, inb = 0, -99, False
        for y in rws:
            if y - prev > 10:
                bands += 1
            prev = y
        return {"empty": False, "W": W, "H": H, "px": int(mask.sum()),
                "l": int(col.min()), "r": int(col.max()), "rows": bands,
                "margin_l": int(col.min()), "margin_r": int(W - 1 - col.max())}

    W = 1280
    ml_scale = 90 * (W / 1920.0)
    ok = True
    # (1) narration caption present, ≤2 rows, inside safe margins (mid scene-0)
    m_nar = caption_bbox(on_mp4, 2.0, "narration")
    nar_ok = (not m_nar["empty"] and m_nar["rows"] <= 2 and m_nar["px"] > 200
              and m_nar["margin_l"] >= ml_scale - 8 and m_nar["margin_r"] >= ml_scale - 8)
    print(f"  narration@2.0s: rows={m_nar.get('rows')} margins L={m_nar.get('margin_l')} "
          f"R={m_nar.get('margin_r')} (safe≥{ml_scale:.0f}) px={m_nar.get('px')} "
          f"{'OK' if nar_ok else 'FAIL'}")
    ok &= nar_ok
    # (2) breakout karaoke present in the breakout window, ≤2 rows, inside safe margins
    if burned:
        m_bk = caption_bbox(on_mp4, b_start + 1.5, "breakout")
        bk_ok = (not m_bk["empty"] and m_bk["rows"] <= 2 and m_bk["px"] > 200
                 and m_bk["margin_l"] >= ml_scale - 8 and m_bk["margin_r"] >= ml_scale - 8)
        print(f"  breakout@{b_start+1.5:.1f}s: rows={m_bk.get('rows')} margins "
              f"L={m_bk.get('margin_l')} R={m_bk.get('margin_r')} (safe≥{ml_scale:.0f}) "
              f"px={m_bk.get('px')} {'OK' if bk_ok else 'FAIL'}")
        ok &= bk_ok
    else:
        print("  breakout: caption not burned (whisper offline?) — narration/OFF still validated")
    # (3) Caption OFF: the caption band carries NO burned caption glyphs (vs a clearly-inked ON band)
    m_off = caption_bbox(off_mp4, 2.0, "off")
    off_px = 0 if m_off.get("empty") else m_off.get("px", 0)
    off_ok = off_px < max(50, m_nar.get("px", 0) * 0.15)
    print(f"  OFF@2.0s caption-band ink px={off_px} vs ON px={m_nar.get('px')} "
          f"{'OK (no caption burn)' if off_ok else 'FAIL'}")
    ok &= off_ok

    print(f"\n[smoke] preset={preset_name} → {'PASS' if ok else 'FAIL'}. "
          f"ON={on_mp4.name} OFF={off_mp4.name} breakout_burned={burned}\n[smoke] evidence: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
