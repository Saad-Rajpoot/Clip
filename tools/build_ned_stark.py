#!/usr/bin/env python3
"""Build the Ned Stark execution ('Baelor', S1E9) video from the user's script + uploaded voiceover,
with ALL latest fixes: unified visual policy, source-frame-only images (no AI/web-generic), breakout
era/recap/Face-ID gates, burned-text (no-space) gate, black-frame floor, recap/fan-edit source gate.
Engine untouched."""
import html
import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_envf = Path(__file__).resolve().parent.parent / ".env"
for _line in _envf.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _v:
            os.environ[_k] = _v

from vidlore.clipstudio.orchestrate import produce_auto
from vidlore.clipstudio import llm as _llm
from vidlore.config import load_config as _cfg

DOCX = "/Users/hussnain/Desktop/Ned Stark's Final Words Hid a Message Nobody Caught.docx"
VO = "/Users/hussnain/Desktop/Ned Stark's Final Words Hid a Message Nobody Caught.mp3"


def docx_text(p):
    try:
        import docx
        return "\n".join(par.text for par in docx.Document(p).paragraphs)
    except Exception:
        xml = zipfile.ZipFile(p).read("word/document.xml").decode("utf-8", "ignore")
        # join each paragraph's <w:t> runs; paragraphs split on </w:p>
        paras = re.split(r"</w:p>", xml)
        out = []
        for pa in paras:
            ts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", pa, re.S)
            if ts:
                out.append(html.unescape("".join(ts)))
        return "\n".join(out)


script = docx_text(DOCX).strip()
P = Path.home() / "Desktop" / "clipstudio_output" / "portal" / "ned_stark_baelor"
P.mkdir(parents=True, exist_ok=True)


def log(m):
    print(m, flush=True)


log(f"[diag] brain={_llm.active_provider(_cfg())} · script_words={len(script.split())} · "
    f"VO={Path(VO).exists()}")
log(f"[diag] script head: {script[:120]!r}")
log(f"[diag] script tail: {script[-120:]!r}")

res = produce_auto(
    str(P),
    topic="Ned Stark's Final Words Hid a Message Nobody Caught",
    script_text=script,
    movie_hint="Game of Thrones",
    policy="approved_testing",
    max_sources=10,                 # auto-scales with the script
    theme="history",
    captions=True,
    voiceover=VO,                   # user's real voice — chunked word-synced captions
    use_tts=True,                   # fallback only
    verify=True,
    do_build=True,
    progress=log,
)
log(f"OUTPUT → {res.get('output')}")
log(f"SUMMARY → {res.get('summary')}")
