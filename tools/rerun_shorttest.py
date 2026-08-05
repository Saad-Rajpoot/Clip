#!/usr/bin/env python3
"""Re-run the short multi-beat end-to-end test on the existing job, under current code.

The sources are already downloaded and indexed, so only the named stages are invalidated and
re-run. Everything a code change can touch — the quote branch decided at match, the cut contract,
the verifier's rescue ladder, the build — is re-derived; discovery and download are not.

    python3 tools/rerun_shorttest.py [stage ...]      # default: match cut verify recover backfill

A gate that blocks is a RESULT. This never lowers a bar to obtain a file.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
sys.path.insert(0, str(REPO))

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

from vidlore.clipstudio.orchestrate import produce_auto           # noqa: E402
from vidlore import musiclib                                      # noqa: E402

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/shorttest_olenna_p1")
STAGES = sys.argv[1:] or ["match", "cut", "verify", "recover", "backfill"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


pj = P / "project.json"
doc = json.loads(pj.read_text(encoding="utf-8"))
stages = doc.get("meta", {}).get("pipeline", {}).get("stages", {})
dropped = [s for s in STAGES if s in stages]
for s in dropped:
    stages.pop(s, None)
tmp = pj.with_suffix(".json.tmp")
tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp, pj)
log(f"invalidated: {dropped or '(none present)'} | kept: {sorted(stages)}")

cats = musiclib.scan()
log(f"musiclib: {len(cats)} categories / {sum(len(v) for v in cats.values())} tracks")

t0 = time.time()
try:
    res = produce_auto(
        str(P),
        topic="How Olenna Tyrell poisoned Joffrey at the Purple Wedding",
        script_path=str(P / "script.txt"),
        movie_hint="Game of Thrones",
        policy="approved_testing",
        max_sources=6,
        theme="history",
        captions=True,
        verify=True,
        do_build=True,
        resume=True,
        progress=log,
    )
except Exception as e:                                            # noqa: BLE001
    log(f"BLOCKED/FAILED after {(time.time() - t0) / 60:.1f} min: {type(e).__name__}: {e}")
    raise
log(f"done in {(time.time() - t0) / 60:.1f} min -> {res}")
