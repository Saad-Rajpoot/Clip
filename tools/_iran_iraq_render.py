#!/usr/bin/env python3
"""REAL-PIPELINE QA RENDER — "How the Iran-Iraq War Reshaped the Middle East".

A portal-equivalent ~60-120s documentary rendered through the SHARED Vidlore
pipeline (vidlore.pipeline.render_from_script), mirroring the established
real-pipeline pattern in tools/multiniche_validate.py + tools/fal_niche_sweep.py.

GOAL of this render: exercise SEVERAL DIFFERENT map primitives naturally
(war front Iraq/Iran, the 1980 invasion advance into Khuzestan, Soviet arms
supply routes, a Gulf shipping/oil chokepoint, Baghdad/Tehran capitals) while
staying FOOTAGE-FIRST (most beats are pure footage). Narration is phrased to
fire the right maps per vidlore/motion_graphics/director.py cue vocabularies
(_WARFRONT / _ADVANCEW / _SUPPLYW / _REGION / _ROUTE / _PLACE) and the
_GK_AFFINITY graphic_kind tags. Place names are chosen to resolve in the bundled
Natural-Earth gazetteer (geo.py) so maps render REAL coastlines/borders.

Env is set BEFORE importing vidlore so all safeguards + Motion Graphics are ON
and AI VIDEO is OFF. FAL stills are budget-capped to 0 (no spend) — we are
auditing maps/footage/audio, not paying for generated stills.

Usage:  .venv/bin/python tools/_iran_iraq_render.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── SAFEGUARDS ON · MG ON · AI-VIDEO OFF · FAL stills 0 (no spend) ───────────
# Mirror the exact flag names used by tools/fal_niche_sweep.py, but force
# FAL_MAX_IMAGES to a hard 0 ceiling so no generated stills are paid for. All
# editorial QA / relevance / period-guard / restraint flags stay at their
# production defaults (we do NOT disable any safeguard).
os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_SUBJECT_FLOOR", "1")
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")            # MG ON
os.environ.setdefault("VIDLORE_STOCK_FILMABLE", "1")
os.environ.setdefault("VIDLORE_OVERLAY_RESTRAINT", "1")         # footage-first
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")                  # AI VIDEO OFF
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")         # strict relevance
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE_CACHE", "0")   # fresh re-score
os.environ.setdefault("VIDLORE_FAL_IMAGE_BUDGET_MODE", "quality_first")
os.environ.setdefault("VIDLORE_FAL_MAX_IMAGES", "0")           # 0 = NO ceiling per sweep…
os.environ["VIDLORE_FAL_MAX_IMAGES"] = "0"                     # …but we force 0 spend
os.environ.setdefault("VIDLORE_FAL_MAX_ATTEMPTS", "4")
os.environ.setdefault("VIDLORE_ALLOW_DEGRADED_TEXT_SLIDES", "0")
os.environ.setdefault("VIDLORE_MAX_TEXT_SLIDE_RATIO", "0.10")
os.environ.setdefault("VIDLORE_EDITORIAL_QA", "1")
# leave period guard / graphic_dom / dedupe / black-frame repair / subtitles /
# music / sfx at their defaults — do NOT disable.

TITLE = "How the Iran-Iraq War Reshaped the Middle East"
NICHE = "geopolitics"
THEME = "modern"
PROMPT = ("How the 1980-1988 war between Iraq and Iran redrew the map of the "
          "modern Middle East")

# Each scene: (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
# Most scenes are PURE FOOTAGE (graphic_kind=""). FIVE tagged scenes carry a
# genuine, DIFFERENT map opportunity, deliberately SPACED ≥3 scenes apart so the
# director's footage-first same-family adjacency rule (director.py plan(): a
# graphic within 2 scenes of another map is dropped) lets each one survive while
# footage stays the foundation. Narration cue-words (_WARFRONT / _ADVANCEW /
# _SUPPLYW / _REGION / _ROUTE) + gazetteer place names make the director derive
# REAL geography (geo.py resolves Baghdad/Tehran/Basra/Abadan/Dubai/Muscat/
# Khuzestan-area cities + Iraq/Iran/Russia boundaries from Natural Earth).
SCENES = [
    # 0 — hook (FOOTAGE) — no hard year so it doesn't grab a timeline at the hook
    ("Two neighbours, two ancient capitals, and one disputed river: this is the "
     "story of the longest conventional war of the twentieth century.",
     4, "hook", "", "", ""),
    # 1 — MAP 1 · the war front Iraq/Iran (parchment_war_map):
    #     _WARFRONT cues + two belligerent proper nouns (Iraq, Iran)
    ("For eight brutal years Iraq was at war with Iran, two armies locked along "
     "a single contested front where the border between them barely moved.",
     4, "warfront", "war_map",
     "THE FRONT", "side_a=Iraq ; side_b=Iran ; title=THE CONTESTED FRONT"),
    # 2 — setup (FOOTAGE)
    ("A revolution had just convulsed Iran, and the rulers in Baghdad believed "
     "their neighbour was weak, divided, and ripe for the taking.",
     3, "setup", "", "", ""),
    # 3 — stakes (FOOTAGE)
    ("What began as a border dispute would soon pull in the great powers and "
     "reshape the balance of the entire Middle East.",
     3, "stakes", "", "", ""),
    # 4 — MAP 2 · the 1980 invasion advance into Khuzestan
    #     (territory_advance_arrows): _ADVANCEW cues + Iraq→Iran countries
    ("In September 1980 Iraqi divisions invaded across the frontier and stormed "
     "into Iran, the invasion driving hard toward the oil-rich province of "
     "Khuzestan in a lightning offensive.",
     5, "invasion", "advance",
     "OFFENSIVE 1980", "title=THE 1980 OFFENSIVE"),
    # 5 — siege (FOOTAGE)
    ("At the port city of Abadan the advance finally stalled, and the war "
     "settled into a grinding stalemate of trenches, shelling, and human "
     "waves.",
     4, "siege", "", "", ""),
    # 6 — human cost (FOOTAGE)
    ("Both armies bled in the marshes and the mountains, and a generation of "
     "young men was fed into a conflict that neither side could win.",
     4, "cost", "", "", ""),
    # 7 — MAP 3 · the disputed oil province on the map (map_region_highlight):
    #     _REGION cues ("the territory of", "borderland") + Khuzestan place
    ("The whole struggle turned on the territory of Khuzestan, the contested "
     "borderland on the Gulf whose oil fields and refineries made it the "
     "richest prize of the war.",
     4, "region", "territory",
     "Khuzestan", "pos=0.58,0.52 ; sub=The disputed oil province ; title=THE PRIZE"),
    # 8 — superpowers (FOOTAGE)
    ("Behind the fighting, the superpowers quietly chose sides, each pouring "
     "money and weapons into a war they did not want to lose.",
     3, "powers", "", "", ""),
    # 9 — proxies (FOOTAGE)
    ("It became a proxy struggle, with foreign hands stacking the odds from "
     "thousands of miles away while two exhausted nations did the dying.",
     3, "proxy", "", "", ""),
    # 10 — MAP 4 · Soviet arms supply routes (supply_route_dashes):
    #      _SUPPLYW cues ("arms shipments", "supplied", "funnelling arms") + source→dest
    ("The Soviet Union supplied Iraq on a vast scale, funnelling arms shipments "
     "from Moscow to Baghdad — tanks, jets, and shells that kept the offensive "
     "alive year after year.",
     4, "supply", "supply_route",
     "THE ARMS LIFELINE", "title=SOVIET ARMS TO BAGHDAD"),
    # 11 — sea war (FOOTAGE)
    ("Then the war spilled out of the desert and onto the water, as both sides "
     "began hunting the oil tankers that were the lifeblood of the Gulf.",
     4, "seawar", "", "", ""),
    # 12 — escalation (FOOTAGE)
    ("Burning hulks drifted in the shipping lanes, and foreign navies were "
     "drawn in to escort the convoys through increasingly dangerous waters.",
     4, "escalate", "", "", ""),
    # 13 — MAP 5 · Gulf shipping / oil chokepoint (velocity_route_map):
    #      _ROUTE cues + gazetteer ports tracing the Strait-of-Hormuz chokepoint
    ("Every cargo had to run the same deadly shipping route, the voyage from "
     "Basra past Abadan and Dubai to the narrow chokepoint at Muscat, the only "
     "gateway out of the Gulf.",
     4, "tankers", "shipping_route",
     "THE TANKER WAR", "stops=Basra|Abadan|Dubai|Muscat ; title=THE GULF CHOKEPOINT"),
    # 14 — the darkest turn (FOOTAGE)
    ("On the front itself the fighting grew darker still, with chemical weapons "
     "unleashed on the battlefield and missiles falling on distant cities.",
     5, "dark", "", "", ""),
    # 15 — the toll (FOOTAGE)
    ("When the guns finally fell silent, perhaps a million people were dead, "
     "and the borders stood exactly where they had begun.",
     4, "toll", "", "", ""),
    # 16 — thesis / outro (statement card — FOOTAGE-backed)
    ("Yet the war had quietly remade the region: it hardened the rift between "
     "Sunni and Shia, drew the world's powers permanently into the Gulf, and "
     "lit the fuse for the conflicts still to come.",
     4, "thesis", "statement",
     "THE WAR THAT REMADE A REGION", ""),
]


def build_script() -> dict:
    scenes = []
    for i, (nar, inten, role, gk, gt, gb) in enumerate(SCENES):
        # portal-equivalent per-scene keywords (production scripts always carry
        # LLM keywords). Derive concrete subject keywords so the footage
        # relevance path is exercised like a real portal render.
        kws = _derive_keywords(nar, gt)
        scenes.append({
            "narration": nar, "keywords": kws, "visual": "",
            "intensity": inten, "emphasis": "",
            "shot_type": "wide" if i % 3 == 0 else "medium",
            "role": role, "graphic_kind": gk, "graphic_text": gt,
            "graphic_body": gb,
        })
    body = TITLE + "\n\n" + "\n\n".join(s["narration"] for s in scenes)
    return {"title": TITLE,
            "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "scenes": scenes}


import re as _re


def _derive_keywords(narration: str, graphic_text: str) -> list:
    """Subject keywords from explicit graphic subject + narration proper nouns +
    concrete lowercase nouns (mirrors fal_niche_sweep._derive_keywords)."""
    kws = []
    if graphic_text and not graphic_text.isupper():
        kws.append(graphic_text.strip())
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", narration or "")
    _stop = {"which", "their", "there", "where", "would", "could", "should",
             "about", "these", "those", "after", "before", "every", "while",
             "first", "single", "nation", "people", "years", "world",
             "september", "twenty"}
    for w in _re.findall(r"\b[a-z]{5,}\b", (narration or "").lower()):
        if w not in _stop and w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        kl = k.lower()
        if kl not in seen and kl not in {"the", "and", "iran", "iraq"} or kl in {"iran", "iraq"}:
            if kl not in seen:
                seen.add(kl)
                out.append(k)
    return out[:5]


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for, render_from_script
    from vidlore.motion_graphics import director as mgdir

    cfg = load_config()
    # FAL configured? (report yes/no WITHOUT printing the key)
    fal_key = ""
    try:
        fal_key = (getattr(cfg, "fal_key", "") or getattr(cfg, "fal_api_key", "")
                   or os.environ.get("FAL_KEY", "") or "")
    except Exception:                                              # noqa: BLE001
        fal_key = os.environ.get("FAL_KEY", "")
    if not fal_key:
        # config may stash providers in a dict
        try:
            d = cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", {})
            blob = json.dumps(d, default=str).lower()
            fal_key = "present" if ("fal" in blob and "key" in blob) else ""
        except Exception:                                          # noqa: BLE001
            pass
    print(f"FAL configured: {'yes' if fal_key else 'no'}", flush=True)

    out = ROOT / "output"
    brief = Brief(title=TITLE, prompt=PROMPT, fmt="documentary",
                  duration="6-8", theme=THEME, captions=True,
                  background="auto", extra={"niche": NICHE})
    run_dir = run_dir_for(brief, out)
    run_dir.mkdir(parents=True, exist_ok=True)

    script = build_script()
    body = TITLE + "\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2),
                                         encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    print(f"scenes={len(script['scenes'])} "
          f"map_opportunities={sum(1 for s in script['scenes'] if s['graphic_kind'])}",
          flush=True)
    pal = mgdir.video_palette(NICHE, int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000)
    print(f"niche={NICHE} palette={pal}", flush=True)

    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    wall = round(time.time() - t0, 1)
    from vidlore.pipeline import _slug
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={wall}s video_exists={vid.exists()} path={vid}",
          flush=True)


if __name__ == "__main__":
    main()
