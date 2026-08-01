"""Does the burned caption say what the viewer HEARS, at the moment they hear it?

The pipeline's captions are generated from the voiceover's own word timings, so every internal
check of them is circular: the words agree with the timings because they came from the timings.
The only non-circular test is to listen to the DELIVERED file. This transcribes windows of the
rendered mp4 and scores each against the .srt text for the same window — catching both a wrong
transcript and a constant offset (the cold-open desync shipped ~4.5s of lag that every internal
check passed).

    python3 research/eval/caption_sync_probe.py <job_dir> [--windows 6] [--len 25]

Reports, per window, the word-overlap with the SRT at zero shift and the shift (-6..+6s) that
maximises it. A healthy render peaks at 0.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.eval.scene_audit_dataset import ffmpeg_exe, parse_srt     # noqa: E402

WORD = re.compile(r"[a-z']+")


def words(s: str) -> list:
    return WORD.findall((s or "").lower())


def overlap(a: list, b: list) -> float:
    """Bag-of-words F1 — robust to ASR dropping or inventing a word, unlike exact match."""
    if not a or not b:
        return 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    p, r = inter / len(b), inter / len(a)
    return 0.0 if not (p + r) else 2 * p * r / (p + r)


def main() -> int:
    job = Path(sys.argv[1])
    nwin = int(sys.argv[sys.argv.index("--windows") + 1]) if "--windows" in sys.argv else 6
    wlen = float(sys.argv[sys.argv.index("--len") + 1]) if "--len" in sys.argv else 25.0
    vid = sorted(job.glob("output/final*.mp4"))[0]
    cues = parse_srt(next(job.glob("output/final*.srt")))
    total = cues[-1][1]

    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

    ff = ffmpeg_exe()
    rows = []
    for k in range(nwin):
        #  spread across the runtime, skipping the very start and end
        t0 = total * (k + 0.5) / nwin - wlen / 2
        t0 = max(2.0, min(t0, total - wlen - 2))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav = tf.name
        subprocess.run([ff, "-nostdin", "-v", "error", "-ss", f"{t0:.2f}", "-t", f"{wlen:.2f}",
                        "-i", str(vid), "-ac", "1", "-ar", "16000", "-y", wav],
                       check=False, capture_output=True)
        segs, _ = model.transcribe(wav, language="en", beam_size=1)
        heard = words(" ".join(s.text for s in segs))
        Path(wav).unlink(missing_ok=True)

        best, best_sh = -1.0, 0.0
        at_zero = 0.0
        for sh in [x / 2 for x in range(-12, 13)]:
            a, z = t0 + sh, t0 + wlen + sh
            shown = words(" ".join(c[2] for c in cues if c[1] > a and c[0] < z))
            sc = overlap(shown, heard)
            if abs(sh) < 1e-9:
                at_zero = sc
            if sc > best:
                best, best_sh = sc, sh
        rows.append({"t0": round(t0, 1), "f1_at_0": round(at_zero, 3),
                     "best_f1": round(best, 3), "best_shift_s": best_sh,
                     "heard_words": len(heard)})
        print(f"  t={t0:7.1f}s  F1@0={at_zero:.3f}  best={best:.3f} @ shift {best_sh:+.1f}s  "
              f"({len(heard)} words heard)")

    bad = [r for r in rows if abs(r["best_shift_s"]) >= 1.0 and r["best_f1"] - r["f1_at_0"] > 0.08]
    print(json.dumps({"windows": rows, "desynced_windows": len(bad)}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
