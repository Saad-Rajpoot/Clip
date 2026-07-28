#!/usr/bin/env python3
"""Round 2 gap fill: beat 127 (Pycelle seized, S2E3) + beat 133 (Littlefinger/Sansa ship
cabin, S4E3 opening). Same targeted discover→download→index as round 1."""
import os
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
J = Path("/Users/hussnain/Desktop/clipstudio_output/portal/olenna_v2_allfixes")
sys.path.insert(0, str(WORKTREE))
for _line in (MAIN / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))
os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))

from vidlore.clipstudio.models import ClipProject          # noqa: E402
from vidlore.clipstudio.config import ClipConfig           # noqa: E402
from vidlore.clipstudio import discover as D               # noqa: E402
from vidlore.clipstudio.download import download_candidates  # noqa: E402
from vidlore.clipstudio import index as I                  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


proj = ClipProject.load(str(J))
cfg = ClipConfig()
seg127 = next(s for s in proj.segments if s.index == 127)
seg133 = next(s for s in proj.segments if s.index == 133)
seg127.scene_query = "Game of Thrones Tyrion arrests Pycelle scene S02E03 Bronn beard"
seg133.scene_query = "Game of Thrones Littlefinger Sansa ship cabin scene 4x03 Breaker of Chains"

import dataclasses as _dc
cfg_r = _dc.replace(cfg, discover_target=24)
ana = type("A", (), dict(
    movie_title="Game of Thrones", year="2011",
    topic="Pycelle seized by Tyrion; Littlefinger and Sansa on the ship",
    anchor_scenes=[
        {"name": "Tyrion has Pycelle seized",
         "query": "Game of Thrones Tyrion arrests Pycelle S2E3 scene",
         "episode": "S02E03 What Is Dead May Never Die",
         "dialogue": ["Take him away.", "I am your loyal servant!"]},
        {"name": "Littlefinger and Sansa on the ship",
         "query": "Game of Thrones Littlefinger rescues Sansa ship scene S04E03",
         "episode": "S04E03 Breaker of Chains",
         "dialogue": ["You're safe now.", "Dontos... I had to pay him."]}],
    characters=[{"name": "Grand Maester Pycelle", "actor": "Julian Glover"},
                {"name": "Tyrion Lannister", "actor": "Peter Dinklage"},
                {"name": "Petyr Baelish", "actor": "Aidan Gillen"},
                {"name": "Sansa Stark", "actor": "Sophie Turner"}],
    actors=["Julian Glover", "Peter Dinklage", "Aidan Gillen", "Sophie Turner"],
    events=[], key_scenes=[], locations=["King's Landing", "the ship"],
    visual_keywords=["ship cabin", "arrest"],
    episode_hint="", episode_hint_verified=False, video_type="multi_scene"))()

cands = D.discover_sources(ana, cfg_r, segments=[seg127, seg133], progress=None) or []
have = {(s.url or "").strip() for s in proj.sources}
new = [c for c in cands if (c.url or "").strip() and (c.url or "").strip() not in have]


def _rel(c):
    t = (c.title or "").lower()
    return (sum(w in t for w in ("pycelle", "arrests", "arrested", "beard")) * 2
            + sum(w in t for w in ("littlefinger", "baelish", "ship", "sansa", "escape")))


new.sort(key=_rel, reverse=True)
new = [c for c in new if _rel(c) >= 2][:5]
log(f"round-2 discovery: {len(cands)} candidates → taking {len(new)}:")
for c in new:
    log(f"  + {(c.title or '')[:84]}")
if new:
    download_candidates(proj, new, cfg, policy="approved_testing", progress=log)
    for sv in proj.sources:
        if sv.url in {c.url for c in new} and sv.status == "ok" and sv.local_path:
            I.index_source(proj, sv, cfg, progress=None)
            log(f"indexed {sv.id} @ {sv.height}p")
proj.save()
log("round-2 gap fixes staged")
