"""Breakout caption WORD-LEVEL REPAIR — the ASR-confidence gate must not delete a quote's
payload for one bad word.

Regressions reproduced from job 5462677f95 (audited 2026-07-28):
  * seg 192/234 — 'marry that beast, do you? Well,' dropped (min conf 0.33): every word is
    source-backed except the trailing reply 'Well,' — the essay's own cited quote vanished,
    twice. Repair: trim the unbacked fringe, keep the backed payload.
  * seg 89 — 'go drink until it feels right' dropped (conf 0.34): past the truncated
    cap['line'] so no source backing, but the selection-time aired_transcript heard the same
    words. Repair: aired corroboration (+ non-hopeless acoustics) keeps it.
  * seg 174 — 'Please don't go away. He poisoned' (conf 0.01) is real garble and must STAY
    dropped; the surviving orphan 'my son.' (25% of spoken words) must be suppressed too.

    python3 tests/test_breakout_caption_word_repair.py

No network, no real model, no ffmpeg.
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


try:
    import faster_whisper as _fw
except Exception:
    print("faster_whisper unavailable — cannot exercise the burner; skipping")
    sys.exit(0)

import vidlore.clipstudio.build as B                       # noqa: E402
from vidlore.clipstudio import caption_presets as CP       # noqa: E402


class _W:
    def __init__(self, word, start, end, probability):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class _Seg:
    def __init__(self, words):
        self.words = words


_canned = {"segs": []}


class _FakeModel:
    def __init__(self, *a, **k):
        pass

    def transcribe(self, audio, **k):
        return list(_canned["segs"]), types.SimpleNamespace()


def _words(text, conf_map=None, conf=0.9):
    conf_map = conf_map or {}
    out = []
    for i, w in enumerate(text.split()):
        out.append(_W(w, i * 0.4, i * 0.4 + 0.35, conf_map.get(w, conf)))
    return out


def _burn(cap, words, audit_entries=None):
    _pre = CP.CAPTION_PRESETS["professional"]
    tmp = Path(tempfile.mkdtemp(prefix="bkrep_"))
    if audit_entries is not None:
        (tmp / "breakout_audit.json").write_text(json.dumps({"accepted": audit_entries}))
    _canned["segs"] = [_Seg(words)]
    cap = dict(cap, audio=str(tmp / "a.wav"))
    _orig, _fw.WhisperModel = _fw.WhisperModel, _FakeModel
    try:
        out = tmp / "bk.ass"
        B._breakout_caption_ass([cap], out, log=lambda m: print("   ·", m), preset=_pre)
        if not out.exists():
            return ""
        return "\n".join(l for l in out.read_text().splitlines() if l.startswith("Dialogue"))
    finally:
        _fw.WhisperModel = _orig


LINE_192 = "You don't think I'd let you marry that beast, do you?"


def test_fringe_trim_keeps_the_cited_quote():
    # ASR heard the quote fine but continued into Margaery's reply; whisper tanked one word.
    words = _words("You don't think I'd let you marry that beast, do you? Well,",
                   conf_map={"beast,": 0.33, "Well,": 0.33})
    dlg = _burn({"start": 10.0, "dur": 9.0, "seg_index": 192, "line": LINE_192}, words)
    check("payload 'marry that beast' survives", "beast" in dlg)
    check("payload 'do you?' survives", "do" in dlg and "you?" in dlg)
    check("unbacked fringe 'Well,' is trimmed", "Well" not in dlg)


def test_aired_corroboration_rescues_past_quote_words():
    line_89 = "I know you don't want to believe it, but she is. Now,"
    aired = ("I know you don't want to believe it but she is. Now, go drink until it "
             "feels right you did the right thing.")
    words = _words("go drink until it feels right",
                   conf_map={"drink": 0.34, "feels": 0.34})
    dlg = _burn({"start": 10.0, "dur": 6.3, "seg_index": 89, "line": line_89}, words,
                audit_entries=[{"seg_index": 89, "aired_transcript": aired}])
    check("aired-corroborated words are kept", "drink" in dlg and "feels" in dlg)


def test_real_garble_stays_dropped_and_orphan_suppressed():
    line_174 = "He did this! He poisoned my son!"
    aired = "Please don't go away. He poisoned my son."
    words = _words("Please don't go away. He poisoned my son.",
                   conf_map={"Please": 0.01, "don't": 0.01, "go": 0.01, "away.": 0.01,
                             "He": 0.01, "poisoned": 0.01})
    dlg = _burn({"start": 10.0, "dur": 12.6, "seg_index": 174, "line": line_174}, words,
                audit_entries=[{"seg_index": 174, "aired_transcript": aired}])
    check("hopeless-confidence garble is NOT resurrected by aired corroboration",
          "Please" not in dlg and "away" not in dlg)
    check("orphan fragment 'my son.' is suppressed (coverage floor)", "son" not in dlg)


def test_kill_switch_restores_old_behaviour():
    words = _words("You don't think I'd let you marry that beast, do you? Well,",
                   conf_map={"beast,": 0.33, "Well,": 0.33})
    os.environ["VIDLORE_CLIPSTUDIO_BK_CAP_REPAIR"] = "0"
    try:
        dlg = _burn({"start": 10.0, "dur": 9.0, "seg_index": 192, "line": LINE_192}, words)
    finally:
        os.environ.pop("VIDLORE_CLIPSTUDIO_BK_CAP_REPAIR", None)
    check("repair off -> the sub-floor group drops as before", "beast" not in dlg)


def test_high_confidence_lines_untouched():
    words = _words("The things I do for love.", conf=0.95)
    dlg = _burn({"start": 10.0, "dur": 2.0, "seg_index": 113,
                 "line": "The things I do for love."}, words)
    check("high-confidence line burns unchanged", "love." in dlg)
    check("no spurious trimming on a clean line", "things" in dlg)


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
