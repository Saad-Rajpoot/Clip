#!/usr/bin/env python3
"""Vidlore Windows RTX benchmark — isolates the two GPU-accelerated paths and
proves SPEED + QUALITY/VERDICT PARITY before GPU mode is trusted as default.

The four production modes are the product of two independent choices:
    encode : libx264 (CPU)   vs  h264_nvenc (RTX)
    CLIP   : CPU provider     vs  DirectML  (RTX)
  Mode A = libx264 + CPU      Mode C = libx264 + DirectML
  Mode B = nvenc   + CPU      Mode D = nvenc   + DirectML
So benchmarking the encode axis and the CLIP axis INDEPENDENTLY measures all
four modes while isolating the variable — more rigorous than four redundant
full renders, and it needs NO network / API keys / cloud (deterministic).

Measures, per axis:
  encode : wall time, output integrity (decodes, duration/res/fps preserved),
           black-frame count, file size/bitrate, peak NVIDIA GPU + encoder util
  CLIP   : wall time, active backend, and PARITY vs CPU (decision flips +
           max score delta) — a flip is a relevance regression and is flagged

Run on the actual Windows machine via tools\\run_windows_gpu_benchmark.bat.
Writes research/final_release/windows_gpu/BENCHMARK_RESULT.json + prints a table.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "research" / "final_release" / "windows_gpu"
WORK = OUT_DIR / "_bench_work"


def ff():
    from vidlore.ffmpeg_tool import ffmpeg_exe
    return ffmpeg_exe()


def _run(cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


# ── NVIDIA sampling ─────────────────────────────────────────────────────────
def smi_sample():
    """One nvidia-smi reading: (gpu_util%, enc_util%, mem_used_MiB) or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.encoder,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        first = r.stdout.strip().splitlines()[0]
        g, e, m = [x.strip() for x in first.split(",")]
        return (float(g), float(e), float(m))
    except Exception:                                          # noqa: BLE001
        return None


class SMIMonitor:
    """Background nvidia-smi poller capturing peak GPU + encoder utilisation."""
    def __init__(self, hz=5):
        self._stop = threading.Event()
        self._t = None
        self.hz = hz
        self.samples = []

    def _loop(self):
        while not self._stop.is_set():
            s = smi_sample()
            if s:
                self.samples.append(s)
            time.sleep(1.0 / self.hz)

    def __enter__(self):
        if smi_sample() is not None:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def peak(self):
        if not self.samples:
            return {"gpu_pct": None, "enc_pct": None, "vram_mib": None}
        return {"gpu_pct": max(s[0] for s in self.samples),
                "enc_pct": max(s[1] for s in self.samples),
                "vram_mib": max(s[2] for s in self.samples)}


# ── integrity ───────────────────────────────────────────────────────────────
def black_frames(path):
    rc, out = _run([ff(), "-hide_banner", "-i", str(path),
                    "-vf", "blackdetect=d=0.05:pix_th=0.10", "-an",
                    "-f", "null", "-"])
    return out.count("black_start")


def probe_dims(path):
    rc, out = _run([ff(), "-hide_banner", "-i", str(path)])
    import re
    dur = res = fps = None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out)
    if m:
        h, mi, s = m.groups()
        dur = int(h) * 3600 + int(mi) * 60 + float(s)
    m = re.search(r"(\d{3,4})x(\d{3,4})", out)
    if m:
        res = f"{m.group(1)}x{m.group(2)}"
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", out)
    if m:
        fps = float(m.group(1))
    return dur, res, fps


def make_test_clip(path, seconds=50):
    """Deterministic 1080p30 synthetic clip — motion + detail so the encoder
    actually works. No network."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ff(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i",
           f"testsrc2=size=1920x1080:rate=30:duration={seconds}",
           "-f", "lavfi", "-i",
           f"mandelbrot=size=1920x1080:rate=30",
           "-filter_complex", "[0:v][1:v]blend=all_mode=average,format=yuv420p[v]",
           "-map", "[v]", "-t", str(seconds),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(path)]
    rc, out = _run(cmd, timeout=300)
    if rc != 0 or not path.exists():
        raise RuntimeError(f"test clip build failed: {out[-400:]}")
    return path


# ── encode axis ─────────────────────────────────────────────────────────────
def venc_args(force):
    """Engine's encoder args for a forced VIDLORE_VENC value."""
    from vidlore import assemble as A
    os.environ["VIDLORE_VENC"] = force
    try:
        A._pick_video_encoder.cache_clear()
    except Exception:                                          # noqa: BLE001
        pass
    return A._venc("20"), A._pick_video_encoder()


def bench_encode(src, force, label):
    out = WORK / f"enc_{label}.mp4"
    args, chosen = venc_args(force)
    cmd = [ff(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           *args, str(out)]
    with SMIMonitor() as mon:
        t0 = time.time()
        try:
            rc, log = _run(cmd, timeout=600)
            err = "" if rc == 0 else log[-300:]
        except Exception as e:                                  # noqa: BLE001
            rc, err = 1, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
    peak = mon.peak()
    res = {"mode": label, "forced": force, "chosen_encoder": chosen,
           "ok": rc == 0 and out.exists(), "seconds": round(dt, 2),
           "error": err, "gpu_peak": peak}
    if res["ok"]:
        dur, dims, fps = probe_dims(out)
        res.update({"out_size_mb": round(out.stat().st_size / 1e6, 2),
                    "duration_s": round(dur or 0, 2), "resolution": dims,
                    "fps": fps, "black_frames": black_frames(out)})
        if dur:
            res["bitrate_mbps"] = round(out.stat().st_size * 8 / 1e6 / dur, 2)
    return res


# ── CLIP axis ───────────────────────────────────────────────────────────────
def _reset_vr():
    import vidlore.visual_relevance as vr
    vr._load_tried = False
    vr._load_ok = False
    vr._vis_sess = None
    vr._txt_sess = None
    vr._vr_is_dml = False
    vr._text_emb_cache = {}
    vr._asset_cache = {}
    return vr


def make_test_images(n=10):
    """Deterministic test frames spanning subjects so accept/reject is non-trivial."""
    imgs = []
    subjects = [
        ("testsrc2=size=640x360:rate=1", "color bars"),
        ("mandelbrot=size=640x360:rate=1", "fractal"),
        ("color=c=gray:size=640x360", "flat gray"),
        ("color=c=black:size=640x360", "black"),
    ]
    for i in range(n):
        lav, _ = subjects[i % len(subjects)]
        p = WORK / f"clip_img_{i}.png"
        _run([ff(), "-y", "-hide_banner", "-loglevel", "error",
              "-f", "lavfi", "-i", lav, "-frames:v", "1", str(p)], timeout=60)
        if p.exists():
            imgs.append(p)
    return imgs


def bench_clip(images, provider, label):
    os.environ["VIDLORE_CLIP_PROVIDER"] = provider
    os.environ["VIDLORE_VISUAL_RELEVANCE"] = "1"
    vr = _reset_vr()
    loaded = vr._try_load()
    st = vr.vr_status()
    scores, decisions = {}, {}
    with SMIMonitor() as mon:
        t0 = time.time()
        for p in images:
            try:
                ok, sc, _reason = vr.accept(
                    str(p), False,
                    expected=["a clear photograph of a city street", "documentary footage"],
                    objects=(), place="", period="", modern_risk=False)
                scores[p.name] = round(float(sc.get("score", 0.0)), 5)
                decisions[p.name] = bool(ok)
            except Exception as e:                              # noqa: BLE001
                scores[p.name] = None
                decisions[p.name] = f"err:{type(e).__name__}"
        dt = time.time() - t0
    return {"mode": label, "provider": provider, "loaded": loaded,
            "backend": st.get("backend"), "is_directml": st.get("is_directml"),
            "device_id": st.get("device_id"), "gpu_name": st.get("gpu_name"),
            "seconds": round(dt, 2), "gpu_peak": mon.peak(),
            "scores": scores, "decisions": decisions}


def clip_parity(cpu, other):
    flips, deltas = [], []
    for k, dec in cpu["decisions"].items():
        od = other["decisions"].get(k)
        if isinstance(dec, bool) and isinstance(od, bool) and dec != od:
            flips.append(k)
        sc, oc = cpu["scores"].get(k), other["scores"].get(k)
        if isinstance(sc, float) and isinstance(oc, float):
            deltas.append(abs(sc - oc))
    return {"decision_flips": flips, "n_flips": len(flips),
            "max_score_delta": round(max(deltas), 5) if deltas else 0.0,
            "mean_score_delta": round(statistics.mean(deltas), 5) if deltas else 0.0,
            "verdict_parity": len(flips) == 0}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    secs = int(os.environ.get("VIDLORE_BENCH_SECONDS", "50"))
    print("VIDLORE — WINDOWS RTX BENCHMARK")
    print(f"building deterministic {secs}s 1080p30 test clip (no network)…")
    src = make_test_clip(WORK / "src.mp4", secs)

    print("\n[1/2] ENCODE axis  (libx264 CPU  vs  h264_nvenc RTX)")
    enc_cpu = bench_encode(src, "x264", "A_cpu_libx264")
    enc_nv = bench_encode(src, "nvenc", "B_rtx_nvenc")
    for r in (enc_cpu, enc_nv):
        print(f"  {r['mode']:18s} enc={r['chosen_encoder']:18s} ok={r['ok']} "
              f"{r['seconds']}s  black={r.get('black_frames','?')} "
              f"size={r.get('out_size_mb','?')}MB encPeak={r['gpu_peak'].get('enc_pct')}%")
    speedup = (round(enc_cpu["seconds"] / enc_nv["seconds"], 2)
               if enc_nv["ok"] and enc_nv["seconds"] else None)
    print(f"  => NVENC speedup vs libx264: {speedup}x"
          if speedup else "  => NVENC unavailable/failed (stays on libx264 — safe)")

    print("\n[2/2] CLIP axis  (CPU provider  vs  DirectML RTX)")
    imgs = make_test_images(10)
    clip_cpu = bench_clip(imgs, "cpu", "A_cpu")
    clip_dml = bench_clip(imgs, "directml", "C_directml")
    parity = clip_parity(clip_cpu, clip_dml)
    for r in (clip_cpu, clip_dml):
        print(f"  {r['mode']:12s} backend={r['backend']:28s} "
              f"dml={r['is_directml']} {r['seconds']}s gpuPeak={r['gpu_peak'].get('gpu_pct')}%")
    cspeedup = (round(clip_cpu["seconds"] / clip_dml["seconds"], 2)
                if clip_dml["seconds"] and clip_dml["is_directml"] else None)
    print(f"  => DirectML CLIP speedup vs CPU: {cspeedup}x"
          if cspeedup else "  => DirectML unavailable (stays on CPU CLIP — safe)")
    print(f"  => VERDICT PARITY: {parity['verdict_parity']}  "
          f"(flips={parity['n_flips']}, max_score_delta={parity['max_score_delta']})")
    if not parity["verdict_parity"]:
        print(f"     ⚠ DECISION FLIPS on {parity['decision_flips']} — relevance "
              f"regression; do NOT enable DirectML CLIP until investigated.")

    result = {"test_seconds": secs, "encode": [enc_cpu, enc_nv],
              "encode_speedup_x": speedup, "clip": [clip_cpu, clip_dml],
              "clip_speedup_x": cspeedup, "clip_parity": parity}
    (OUT_DIR / "BENCHMARK_RESULT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'BENCHMARK_RESULT.json'}")
    print("Paste this file (or the table above) back so the production default "
          "can be set on EVIDENCE, not assumption.")


if __name__ == "__main__":
    main()
