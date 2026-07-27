#!/usr/bin/env python3
"""Short end-to-end test render for the P1-P4 fix pass (worktree code).

Exercises, in ~80 seconds of video: deictic look-targets (P1: "watch the chalice",
"keep your eye on Olenna's hands", a bare proper-noun "watch Joffrey"), the HD downloader
under real YouTube (P2), a quotable line likely to be mined from two uploads (P3B dedup +
P3A caption repair), and the new source screening on a listicle-rich topic (P4).

Run FROM the worktree:  python3 tools/build_shorttest_p1.py
Worktree prerequisites are set below (music dir, HD venv/pot, main .env).
"""
import os
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
sys.path.insert(0, str(WORKTREE))

# main-checkout assets the worktree lacks (gitignored): keys, music, HD venv, pot server
for _line in (MAIN / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))
os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")

from vidlore.clipstudio.orchestrate import produce_auto            # noqa: E402
from vidlore import musiclib                                       # noqa: E402

SCRIPT = """\
At the Purple Wedding, the whole court watched the king die. But almost nobody watched the cup.
Watch the chalice itself. It moves from hand to hand all afternoon.
Joffrey drinks from it, insults his uncle, and drinks again.
Keep your eye on Olenna's hands while she talks to Sansa.
She fusses with Sansa's hair, and the necklace loses a stone.
Nobody at the table reacts. Nobody is meant to.
Watch Joffrey. He laughs, he chokes, and the laughter around him dies.
Olenna had already told Margaery the truth in the garden.
"You don't think I'd let you marry that beast, do you?"
That single line is the whole confession, hidden in plain sight.
The Queen of Thorns poisoned a king at his own wedding, and walked away untouched.
"""

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/shorttest_olenna_p1")
P.mkdir(parents=True, exist_ok=True)
(P / "script.txt").write_text(SCRIPT, encoding="utf-8")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


cats = musiclib.scan()
log(f"musiclib: {len(cats)} categories / "
    f"{sum(len(v) for v in cats.values())} tracks (need 11/118)")

t0 = time.time()
res = produce_auto(
    str(P),
    topic="How Olenna Tyrell poisoned Joffrey at the Purple Wedding",
    script_text=SCRIPT,
    movie_hint="Game of Thrones",
    policy="approved_testing",
    max_sources=6,
    theme="history",
    captions=True,
    verify=True,
    do_build=True,
)
log(f"produce_auto done in {(time.time() - t0) / 60:.1f} min → {res}")
