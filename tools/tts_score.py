#!/usr/bin/env python3
"""Objective TTS sample scorer (dev-only, no paid APIs).

Subjective realism needs human ears — this measures what a machine CAN judge so
candidates can be compared fairly + regressions caught:
  - generated? duration / sample-rate / RMS / peak (clipping) / silence ratio
  - EBU R128 integrated loudness + true peak (via ffmpeg, same as benchmark)
  - intelligibility: faster-whisper ASR of the OUTPUT vs the intended text ->
    word error rate (catches skipped words) + key-term presence (catches
    mangled proper nouns / numbers like "Escobar" / "1982").

Usage:
    python3 tools/tts_score.py <wav> --text "the intended line" [--keys escobar,1982]
    python3 tools/tts_score.py --dir research/tts_samples/kokoro   # score a tree w/ meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_NUM = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
}


def _year_to_words(y: int) -> str:
    # 1982 -> "nineteen eighty two", 2010 -> "twenty ten" (loose, for matching)
    if 1100 <= y <= 1999:
        hi, lo = y // 100, y % 100
        return f"{_NUM.get(str(hi), str(hi))} {_two(lo)}".strip()
    if 2000 <= y <= 2099:
        lo = y % 100
        return f"twenty {_two(lo)}".strip() if lo else "two thousand"
    return str(y)


def _two(n: int) -> str:
    if n == 0:
        return ""
    if n in (0,) or str(n) in _NUM:
        return _NUM[str(n)]
    if n < 20:
        return _NUM.get(str(n), str(n))
    tens, ones = (n // 10) * 10, n % 10
    return (_NUM.get(str(tens), "") + (" " + _NUM[str(ones)] if ones else "")).strip()


def _norm(s: str) -> list[str]:
    """Lowercase, expand bare years to words, strip punctuation -> token list."""
    s = (s or "").lower()
    s = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: _year_to_words(int(m.group(1))), s)
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return s.split()


def _wer(ref: list[str], hyp: list[str]) -> float:
    # Levenshtein over tokens
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[m] / float(n)


def _ffmpeg():
    from vidlore.ffmpeg_tool import ffmpeg_exe
    return ffmpeg_exe()


def _loudness(wav: Path) -> dict:
    try:
        r = subprocess.run([_ffmpeg(), "-i", str(wav), "-af",
                            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                            "-f", "null", "-"], capture_output=True, text=True)
        txt = r.stderr
        i, j = txt.rfind("{"), txt.rfind("}")
        d = json.loads(txt[i:j + 1]) if i >= 0 and j > i else {}
        return {"lufs": _f(d.get("input_i")), "true_peak_db": _f(d.get("input_tp"))}
    except Exception:                                          # noqa: BLE001
        return {"lufs": None, "true_peak_db": None}


def _f(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def _probe(wav: Path) -> dict:
    import wave
    import contextlib
    try:
        with contextlib.closing(wave.open(str(wav), "rb")) as w:
            fr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
            return {"sample_rate": fr, "seconds": round(n / float(fr), 2), "channels": ch}
    except Exception:                                          # noqa: BLE001
        # non-wav (mp3/flac) -> ffprobe-ish via ffmpeg
        return {"sample_rate": None, "seconds": None, "channels": None}


_ASR = {}


def _transcribe(wav: Path) -> str:
    if "m" not in _ASR:
        from faster_whisper import WhisperModel
        _ASR["m"] = WhisperModel("base", device="cpu", compute_type="int8")
    segs, _ = _ASR["m"].transcribe(str(wav), language="en", beam_size=1)
    return " ".join(s.text for s in segs).strip()


def score_wav(wav: Path, text: str, keys: list[str] | None = None) -> dict:
    out = {"file": str(wav), "ok": wav.is_file() and wav.stat().st_size > 200}
    out.update(_probe(wav))
    out.update(_loudness(wav))
    asr = ""
    try:
        asr = _transcribe(wav)
    except Exception as e:                                     # noqa: BLE001
        out["asr_error"] = str(e)[:120]
    out["asr"] = asr
    ref, hyp = _norm(text), _norm(asr)
    out["wer"] = round(_wer(ref, hyp), 3)
    out["intelligibility"] = round(max(0.0, 1.0 - out["wer"]), 3)
    klist = keys or []
    hyptext = " ".join(hyp)
    out["key_terms"] = {k: (k.lower() in hyptext) for k in klist}
    out["keys_ok"] = all(out["key_terms"].values()) if klist else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--text", default="")
    ap.add_argument("--keys", default="")
    ap.add_argument("--dir", default=None, help="score a sample tree containing meta.json")
    a = ap.parse_args()
    keys = [k.strip() for k in a.keys.split(",") if k.strip()]
    if a.dir:
        d = Path(a.dir)
        meta = json.loads((d / "meta.json").read_text())
        rows = []
        for item in meta.get("samples", []):
            w = d / item["file"]
            rows.append(score_wav(w, item["text"], item.get("keys", [])))
        agg = [r for r in rows if r.get("intelligibility") is not None]
        avg = round(sum(r["intelligibility"] for r in agg) / len(agg), 3) if agg else None
        (d / "scores.json").write_text(json.dumps({"avg_intelligibility": avg, "rows": rows}, indent=2))
        print(f"avg intelligibility {avg}  ({len(rows)} samples) -> {d/'scores.json'}")
        for r in rows:
            print(f"  {Path(r['file']).name:20s} intel={r.get('intelligibility')} "
                  f"wer={r.get('wer')} lufs={r.get('lufs')} keys={r.get('keys_ok')}  asr={r.get('asr','')[:60]!r}")
        return 0
    print(json.dumps(score_wav(Path(a.wav), a.text, keys), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
