#!/usr/bin/env python3
"""DeepSeek vs Claude: run the SAME script through clipstudio's analyze_script on each provider and
compare the analysis (the LLM's whole job in clipstudio). No render — this is the meaningful quality
test. Loads .env for keys."""
import os, sys, time, json
from pathlib import Path

# load .env (Anthropic + DeepSeek keys)
envf = Path(__file__).resolve().parent.parent / ".env"
for line in envf.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ[k] = v

from vidlore.clipstudio import llm as _llm
from vidlore.clipstudio.analyze import analyze_script
from vidlore.clipstudio.config import load_clip_config
from vidlore.config import load_config as engine_config

SCRIPT = """A man is skinning a dead stag. That's how we meet Tywin Lannister.
In Game of Thrones, Season 1, Tywin guts the animal with his bare hands while lecturing his son Jaime about the family legacy.
He never raises his voice. "A lion doesn't concern himself with the opinions of sheep," he says.
Jaime stands there in his golden armor, the favorite son, being told he's not enough.
It is the coldest father-son scene the show ever made, and Tywin never even looks up from the carcass."""

cfg = load_clip_config(); eng = engine_config()

def run(provider):
    os.environ["VIDLORE_CLIPSTUDIO_LLM_PROVIDER"] = provider
    print(f"\n===== provider={provider} · active={_llm.active_provider(eng)} =====", flush=True)
    t0 = time.time()
    analysis, segs = analyze_script(SCRIPT, topic="Tywin skins a stag — the coldest scene",
                                    movie_hint="Game of Thrones", eng_cfg=eng, cfg=cfg,
                                    progress=lambda m: None)
    dt = time.time() - t0
    print(f"time {dt:.1f}s · source={analysis.source}")
    print(f"movie={analysis.movie_title!r} type={analysis.video_type} ep={analysis.episode_hint!r}")
    print(f"actors={analysis.actors}")
    print(f"characters={[(c.get('name'),c.get('actor')) for c in (analysis.characters or [])]}")
    for sc in (analysis.anchor_scenes or []):
        print(f"anchor: {sc.get('name')!r} | query={sc.get('query')!r}")
        for d in (sc.get('dialogue') or [])[:3]: print(f"   line: {d[:60]!r}")
    print(f"beats={len(segs)} · sample scene_queries:")
    for s in segs[:5]:
        print(f"   [{s.index}] {(s.text or '')[:38]!r} -> {(getattr(s,'scene_query','') or '')[:50]!r}")
    return analysis, segs, dt

for prov in ("deepseek", "anthropic"):
    try:
        run(prov)
    except Exception as e:
        print(f"  {prov} FAILED: {type(e).__name__}: {e}")
