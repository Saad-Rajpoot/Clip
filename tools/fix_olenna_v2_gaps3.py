#!/usr/bin/env python3
"""Round 3: Cressen poisoning (S2E1), Tyrion-trial Pycelle testimony (S4E6), necklace/crystal
close-up (S4E1) + soften beat 254 (direct-address meta line, same class as 116)."""
import os, sys, time
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
os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))
from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio import discover as D
from vidlore.clipstudio.download import download_candidates
from vidlore.clipstudio import index as I

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

proj = ClipProject.load(str(J))
cfg = ClipConfig()
seg254 = next(s for s in proj.segments if s.index == 254)
seg254.visual_policy = "abstract_effect"
seg254.required_entity = ""
seg254.required_kind = ""
seg254.is_specific_claim = False
seg254.expected_visual = "A slow, thoughtful pan over the empty feast tables — visual rest under a direct-address line."
log("beat 254 softened to abstract (direct-address meta line, same class as 116)")

s36 = next(s for s in proj.segments if s.index == 36)
s41 = next(s for s in proj.segments if s.index == 41)
s128 = next(s for s in proj.segments if s.index == 128)
s206 = next(s for s in proj.segments if s.index == 206)
s36.scene_query = "Game of Thrones Joffrey Widow's Wail sword cuts pie Purple Wedding"
s41.scene_query = "Game of Thrones Maester Cressen poisons wine Melisandre S02E01"
s128.scene_query = "Game of Thrones Tyrion trial Pycelle testimony S04E06"
s206.scene_query = "Game of Thrones Sansa necklace purple crystal close up strangler poison"
import dataclasses as _dc
cfg_r = _dc.replace(cfg, discover_target=28)
ana = type("A", (), dict(
    movie_title="Game of Thrones", year="2011",
    topic="Cressen poisoning, Tyrion trial testimony, the Strangler crystal",
    anchor_scenes=[
        {"name": "Cressen tries to poison Melisandre",
         "query": "Game of Thrones Maester Cressen poison wine Melisandre death S2E1",
         "episode": "S02E01 The North Remembers", "dialogue": ["The Lord of Light protects his own."]},
        {"name": "Pycelle testifies at Tyrion's trial",
         "query": "Game of Thrones Tyrion trial witnesses Pycelle testimony S4E6",
         "episode": "S04E06 The Laws of Gods and Men",
         "dialogue": ["The Strangler.", "poison most rare"]},
        {"name": "Sansa's necklace with the crystals",
         "query": "Game of Thrones Sansa Stark necklace amethyst crystal S4E1 close up",
         "episode": "S04E01 Two Swords", "dialogue": []}],
    characters=[{"name": "Maester Cressen", "actor": "Oliver Ford Davies"},
                {"name": "Melisandre", "actor": "Carice van Houten"},
                {"name": "Grand Maester Pycelle", "actor": "Julian Glover"},
                {"name": "Tyrion Lannister", "actor": "Peter Dinklage"}],
    actors=["Oliver Ford Davies", "Carice van Houten", "Julian Glover", "Peter Dinklage"],
    events=[], key_scenes=[], locations=["Dragonstone", "King's Landing"],
    visual_keywords=["poison", "crystal", "trial"],
    episode_hint="", episode_hint_verified=False, video_type="multi_scene"))()
cands = D.discover_sources(ana, cfg_r, segments=[s41, s128, s206, s36], progress=None) or []
have = {(s.url or "").strip() for s in proj.sources}
new = [c for c in cands if (c.url or "").strip() and (c.url or "").strip() not in have]

def _rel(c):
    t = (c.title or "").lower()
    return (sum(w in t for w in ("cressen", "melisandre", "trial", "testimony", "strangler")) * 2
            + sum(w in t for w in ("pycelle", "necklace", "poison", "tyrion", "laws of gods")))

new.sort(key=_rel, reverse=True)
new = [c for c in new if _rel(c) >= 2][:6]
log(f"round-3 discovery: {len(cands)} candidates → taking {len(new)}")
for c in new:
    log(f"  + {(c.title or '')[:84]}")
if new:
    download_candidates(proj, new, cfg, policy="approved_testing", progress=log)
    for sv in proj.sources:
        if sv.url in {c.url for c in new} and sv.status == "ok" and sv.local_path:
            I.index_source(proj, sv, cfg, progress=None)
            log(f"indexed {sv.id} @ {sv.height}p")
proj.save()
log("round-3 staged")
