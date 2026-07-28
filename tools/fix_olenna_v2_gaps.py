#!/usr/bin/env python3
"""Resolve the 3 pre-assembly blockers of olenna_v2_allfixes, then the caller resumes.

- Beats 10/11 (Ser Dontos gives Sansa the necklace, S4E1 godswood): the scene exists on
  YouTube but discovery never pulled a clean copy — run a TARGETED discover→download→index
  round for exactly that scene.
- Beat 116 asks for BEHIND-THE-SCENES footage of the showrunners — content our own gates
  forbid by design (BTS/interview titles never air). The handover's own remedy applies:
  a beat demanding unobtainable footage becomes an abstract/filler beat (a 2-second meta
  line does not justify failing the whole video). Applied at the data level, logged here.
"""
import os
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
J = Path("/Users/hussnain/Desktop/clipstudio_output/portal/olenna_v2_allfixes")
sys.path.insert(0, str(WORKTREE))

for _line in (MAIN / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
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

# ── beat 116: abstract meta-beat (BTS footage is gate-forbidden by design) ──
seg116 = next(s for s in proj.segments if s.index == 116)
log(f"beat 116 before: policy={seg116.visual_policy!r} entity={seg116.required_entity!r}")
seg116.visual_policy = "abstract_effect"
seg116.required_entity = ""
seg116.required_kind = ""
seg116.is_specific_claim = False
seg116.expected_visual = ("Neutral Game of Thrones production imagery — the court, the set-like "
                          "grandeur of the throne room; a beat of visual rest under a meta line.")
seg116.scene_query = "Game of Thrones throne room wide shot"
log("beat 116 now: abstract_effect (meta line — BTS ask is unfillable by our own gates)")

# ── beats 10/11: targeted acquisition for the Dontos necklace scene ──
seg10 = next(s for s in proj.segments if s.index == 10)
seg11 = next(s for s in proj.segments if s.index == 11)
seg10.scene_query = "Game of Thrones Sansa Ser Dontos necklace godswood scene 4x01"
seg11.scene_query = "Game of Thrones Ser Dontos gives Sansa necklace scene S04E01"

import dataclasses as _dc
cfg_r = _dc.replace(cfg, discover_target=24)
ana = type("A", (), dict(
    movie_title="Game of Thrones", year="2011",
    topic="Ser Dontos gives Sansa Stark the necklace",
    anchor_scenes=[{"name": "Dontos gives Sansa the necklace",
                    "query": "Game of Thrones Ser Dontos gives Sansa the necklace godswood S4E1",
                    "episode": "S04E01 Two Swords",
                    "dialogue": ["Wear it, wear it for me."]}],
    characters=[{"name": "Ser Dontos Hollard", "actor": "Tony Way"},
                {"name": "Sansa Stark", "actor": "Sophie Turner"}],
    actors=["Tony Way", "Sophie Turner"], events=[], key_scenes=[],
    locations=["the godswood", "King's Landing"], visual_keywords=["necklace"],
    episode_hint="S04E01", episode_hint_verified=False, video_type="multi_scene"))()

cands = D.discover_sources(ana, cfg_r, segments=[seg10, seg11], progress=log) or []
have = {(s.url or "").strip() for s in proj.sources}
new = [c for c in cands if (c.url or "").strip() and (c.url or "").strip() not in have]
new.sort(key=lambda c: sum(w in (c.title or "").lower()
                           for w in ("dontos", "necklace", "sansa")), reverse=True)
new = new[:4]
log(f"targeted discovery: {len(cands)} candidates, taking {len(new)} new:")
for c in new:
    log(f"  + {(c.title or '')[:80]}")
if new:
    download_candidates(proj, new, cfg, policy="approved_testing", progress=log)
    for sv in proj.sources:
        if sv.url in {c.url for c in new} and sv.status == "ok" and sv.local_path:
            n = I.index_source(proj, sv, cfg, progress=log)
            log(f"indexed {sv.id}: {n if n is not None else '?'} shots @ {sv.height}p")
proj.save()
log("gap fixes staged — caller resumes produce_auto")
