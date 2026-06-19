#!/usr/bin/env python3
"""REAL-PIPELINE QA RENDER — "The Cambridge Spy Who Vanished" (Kim Philby).

A portal-equivalent ~6-8 min spy/intelligence documentary rendered through the
SHARED Vidlore pipeline (vidlore.pipeline.render_from_script), mirroring the
established pattern in tools/_iran_iraq_render.py + tools/multiniche_validate.py.

GOAL: exercise the NEW Section-B investigation family (V2.6) naturally through
the real director — witness_testimony_card, suspect_profile_card,
investigation_location_map (real geo), route_comparison, evidence_connection_board,
classified_stamp_reveal — while staying FOOTAGE-FIRST (most beats pure footage).
Narration is phrased to fire the right cards via director.py cue vocabularies
(_TESTIMONYW / _SUSPECTW / _SITESW / _ROUTECMP / _NETWORK / _DECLASSW) and the
_GK_AFFINITY graphic_kind tags. Place names (Vienna/Berlin/Moscow/Beirut/London/
Washington) resolve in the bundled Natural-Earth gazetteer so the location map
renders REAL geography. Subject is a long-public historical figure.

Env is set BEFORE importing vidlore so all safeguards + Motion Graphics are ON
and AI VIDEO is OFF. FAL stills are budget-capped to 0 (no spend).

Usage:  .venv/bin/python tools/_secB_spy_render.py
"""
import hashlib
import json
import os
import re as _re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── SAFEGUARDS ON · MG ON · AI-VIDEO OFF · FAL stills 0 (no spend) ───────────
os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_SUBJECT_FLOOR", "1")
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")            # MG ON
os.environ.setdefault("VIDLORE_STOCK_FILMABLE", "1")
os.environ.setdefault("VIDLORE_OVERLAY_RESTRAINT", "1")         # footage-first
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")                  # AI VIDEO OFF
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")         # strict relevance
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE_CACHE", "0")
os.environ.setdefault("VIDLORE_FAL_IMAGE_BUDGET_MODE", "quality_first")
os.environ.setdefault("VIDLORE_FAL_MAX_IMAGES", "0")
os.environ["VIDLORE_FAL_MAX_IMAGES"] = "0"                     # force 0 spend
os.environ.setdefault("VIDLORE_FAL_MAX_ATTEMPTS", "4")
os.environ.setdefault("VIDLORE_ALLOW_DEGRADED_TEXT_SLIDES", "0")
os.environ.setdefault("VIDLORE_MAX_TEXT_SLIDE_RATIO", "0.10")
os.environ.setdefault("VIDLORE_EDITORIAL_QA", "1")

TITLE = "The Cambridge Spy Who Vanished"
NICHE = "spy"
THEME = "modern"
PROMPT = ("How Kim Philby, the most successful double agent of the Cold War, "
          "betrayed British intelligence for the Soviet Union and vanished")

# Each scene: (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
# Most scenes are PURE FOOTAGE (graphic_kind=""). SIX tagged scenes carry a
# genuine, DIFFERENT investigation opportunity, deliberately SPACED ≥3 scenes
# apart so the director's footage-first adjacency rule lets each survive while
# footage stays the foundation. Narration cue-words make the director DERIVE the
# card inputs (a quote, a suspect name + alias, resolvable sites, an official-vs-
# actual route, a connected network, a declassified line).
SCENES = [
    # 0 — hook (FOOTAGE)
    ("For thirty years, the man who hunted Soviet spies for British intelligence "
     "was himself the deepest Soviet spy of them all.",
     4, "hook", "", "", ""),
    # 1 — WITNESS TESTIMONY · a colleague's quote, attributed + recalled
    ("\"He was charming, brilliant, and utterly trusted — no one ever suspected "
     "the man at the very top,\" one MI6 officer recalled years later, under oath.",
     3, "testimony", "testimony",
     "TESTIMONY", "case_no=MI6 / 1963"),
    # 2 — background (FOOTAGE)
    ("Recruited at Cambridge in the nineteen-thirties, he rose quietly through "
     "the ranks of British secret intelligence.",
     3, "setup", "", "", ""),
    # 3 — stakes (FOOTAGE)
    ("By the height of the Cold War he sat at the very heart of the West's war "
     "against Moscow, with access to its most guarded secrets.",
     3, "stakes", "", "", ""),
    # 4 — SUSPECT PROFILE · the subject, code-named, alias, real name
    ("To his Soviet handlers the prime suspect was a double agent code-named "
     "Stanley; his real name was Harold Adrian Russell Philby, the man the world "
     "would know as Kim.",
     4, "suspect", "suspect",
     "PERSON OF INTEREST", "status=DOUBLE AGENT ; case_no=KGB / 5601"),
    # 5 — method (FOOTAGE)
    ("Week after week he photographed documents and passed them to a courier, "
     "feeding the Kremlin a stream of Western secrets.",
     3, "method", "", "", ""),
    # 6 — betrayal (FOOTAGE)
    ("Agents he exposed were rolled up and executed; the blood of dozens of men "
     "was on his hands, yet his reputation only grew.",
     4, "betrayal", "", "", ""),
    # 7 — LOCATION MAP · the network's real sites (geo bed)
    ("His secret world stretched across the safe houses of a divided continent — "
     "the dead drops of Vienna, the back streets of Berlin, and the handlers "
     "waiting in Moscow.",
     3, "sites", "evidence_map",
     "THE NETWORK", ""),
    # 8 — suspicion (FOOTAGE)
    ("Slowly, inside the service, a few quiet officers began to suspect that a "
     "traitor was hiding in plain sight among them.",
     3, "suspicion", "", "", ""),
    # 9 — interrogation (FOOTAGE)
    ("He was questioned, again and again, but each time his easy confidence and "
     "powerful friends let him slip the noose.",
     4, "interro", "", "", ""),
    # 10 — ROUTE COMPARISON · official account vs what really happened
    ("The official account said he was a loyal servant cleared of all suspicion, "
     "but the real timeline showed a very different journey from London to a "
     "waiting Soviet freighter.",
     4, "discrepancy", "route_comparison",
     "THE OFFICIAL STORY", "start=London ; end=Moscow"),
    # 11 — flight (FOOTAGE)
    ("In January nineteen-sixty-three he simply vanished from Beirut, stepping "
     "off the map and out of the reach of British intelligence forever.",
     4, "flight", "", "", ""),
    # 12 — reappearance (FOOTAGE)
    ("Months later he surfaced in Moscow, paraded by the Soviets as the spy who "
     "had beaten the West at its own game.",
     3, "reappear", "", "", ""),
    # 13 — EVIDENCE BOARD · the connected network of the ring
    ("Only later did the full ring come into focus: it was all connected — the "
     "recruiter, the courier, the forger and the handler were every one of them "
     "linked to the same Soviet controller.",
     4, "network", "connection_board",
     "THE RING", ""),
    # 14 — fallout (FOOTAGE)
    ("The scandal tore through the intelligence services on both sides of the "
     "Atlantic and poisoned the trust between London and Washington for years.",
     4, "fallout", "", "", ""),
    # 15 — legacy (FOOTAGE)
    ("He lived out his days in Moscow, a decorated colonel of the very service "
     "he had served in secret for the whole of his adult life.",
     3, "legacy", "", "", ""),
    # 16 — CLASSIFIED REVEAL · the declassified verdict
    ("Decades on, the declassified files, stamped top secret for half a century, "
     "finally revealed how completely he had betrayed the country that raised him.",
     4, "classified", "classified",
     "EYES ONLY", "header=SECRET INTELLIGENCE SERVICE · FILE 1963"),
    # 17 — thesis / outro (statement — FOOTAGE-backed)
    ("In the end his greatest weapon was not a cipher or a camera, but the simple "
     "refusal of decent men to believe that one of their own could lie so well.",
     4, "thesis", "statement",
     "THE PERFECT TRAITOR", ""),
]


def _derive_keywords(narration: str, graphic_text: str) -> list:
    kws = []
    if graphic_text and not graphic_text.isupper():
        kws.append(graphic_text.strip())
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", narration or "")
    _stop = {"which", "their", "there", "where", "would", "could", "should",
             "about", "these", "those", "after", "before", "every", "while",
             "first", "single", "people", "years", "world", "january"}
    for w in _re.findall(r"\b[a-z]{5,}\b", (narration or "").lower()):
        if w not in _stop and w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(k)
    return out[:5]


def build_script() -> dict:
    scenes = []
    for i, (nar, inten, role, gk, gt, gb) in enumerate(SCENES):
        scenes.append({
            "narration": nar, "keywords": _derive_keywords(nar, gt), "visual": "",
            "intensity": inten, "emphasis": "",
            "shot_type": "wide" if i % 3 == 0 else "medium",
            "role": role, "graphic_kind": gk, "graphic_text": gt,
            "graphic_body": gb,
        })
    body = TITLE + "\n\n" + "\n\n".join(s["narration"] for s in scenes)
    return {"title": TITLE,
            "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "scenes": scenes}


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    from vidlore.motion_graphics import director as mgdir

    cfg = load_config()
    fal_key = os.environ.get("FAL_KEY", "") or ""
    try:
        fal_key = fal_key or getattr(cfg, "fal_key", "") or getattr(cfg, "fal_api_key", "")
    except Exception:                                              # noqa: BLE001
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
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)

    # dry-run the director so we can SEE which investigation cards it will fire
    mg = [{"index": i, "role": s["role"], "graphic_kind": s["graphic_kind"],
           "intensity": s["intensity"], "narration": s["narration"],
           "emphasis": s.get("emphasis", ""), "assets": {}}
          for i, s in enumerate(script["scenes"])]
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=0.40)
    fired = [(d.scene_index, d.primitive) for d in dec if d.primitive]
    pal = mgdir.video_palette(NICHE, seed)
    print(f"niche={NICHE} palette={pal} scenes={len(script['scenes'])} "
          f"opportunities={sum(1 for s in script['scenes'] if s['graphic_kind'])}",
          flush=True)
    print("DIRECTOR_FIRED " + json.dumps(fired), flush=True)

    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    wall = round(time.time() - t0, 1)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={wall}s video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
