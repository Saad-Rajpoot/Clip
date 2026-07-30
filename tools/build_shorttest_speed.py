#!/usr/bin/env python3
"""2-minute end-to-end SMOKE render for the merged speed pass — the full produce_auto
portal path on a fresh job, so every new piece runs live: index prewarmer during the
download window, mono-decode flags, OCR pool (opt-in set here like the portal does),
verify phase-1+2 prefetch, early release-gate, parallel probes/sweeps, sleep watcher.

Run FROM the worktree:  python3 tools/build_shorttest_speed.py
"""
import os
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
sys.path.insert(0, str(WORKTREE))

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
# the portal (web.py) sets this for every job; this driver is __main__-guarded so it is
# equally spawn-safe — set it the same way the portal does
os.environ["VIDLORE_CLIPSTUDIO_OCR_POOL_OK"] = "1"

SCRIPT = """\
Everyone remembers the swing of the sword. Almost nobody remembers the lie that came first.
Watch Ned Stark on the steps of the Great Sept of Baelor.
He confesses to a treason he never committed.
He does it to save his daughters. Look at Sansa, standing behind the king.
She still believes a confession means mercy.
Watch Joffrey's face while the crowd jeers.
He was supposed to grant mercy. Cersei expected it. Sansa was promised it.
"Ser Ilyn, bring me his head!"
One sentence, and every plan in King's Landing dies with Ned.
Watch Arya in the crowd, on the statue of Baelor.
Yoren finds her before the sword falls, and pulls her face into his chest.
Cersei panics. Sansa screams. Joffrey smiles.
The boy king just started a war because it felt good.
And the man who warned Ned about mercy, Lord Varys, can only watch.
Winter did not kill Ned Stark. A child's cruelty did.
"""

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/shorttest_speed")
P.mkdir(parents=True, exist_ok=True)
(P / "script.txt").write_text(SCRIPT, encoding="utf-8")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from vidlore.clipstudio.orchestrate import produce_auto            # noqa: E402
from vidlore import musiclib                                       # noqa: E402


def main():
    cats = musiclib.scan()
    log(f"musiclib: {len(cats)} categories / "
        f"{sum(len(v) for v in cats.values())} tracks (need 11/118)")
    assert len(cats) == 11, "music library incomplete — symlink main assets first"
    t0 = time.time()
    res = produce_auto(
        str(P),
        topic="The lie Ned Stark told before he died at the Sept of Baelor",
        script_text=SCRIPT,
        movie_hint="Game of Thrones",
        policy="approved_testing",
        max_sources=10,
        theme="history",
        captions=True,
        verify=True,
        do_build=True,
        resume=("--resume" in sys.argv),
        progress=log,
    )
    log(f"produce_auto done in {(time.time() - t0) / 60:.1f} min → {res.get('output')}")
    s = res.get("summary") or {}
    log(f"flags: {s.get('flag_breakdown')} | sources: {s.get('sources_used')} | "
        f"conf: {s.get('mean_confidence')}")


if __name__ == "__main__":
    main()
