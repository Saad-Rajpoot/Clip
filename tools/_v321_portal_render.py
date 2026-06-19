#!/usr/bin/env python3
"""V3.2.1 STEP 4+5 — focused portal-equivalent renders (4 samples) + metrics +
disk-safe cleanup. Each sample ~18 scenes (~4-5 min) through the REAL shared
pipeline (strict relevance, period guard, fal stills, AI-video OFF, subtitles,
music/SFX, editorial QA, anti-repetition). Structured beats use TAUGHT/known
graphic_kinds (respected via _MG_PROTECTED_KINDS); the rest are footage-first.

Usage: python tools/_v321_portal_render.py A   (or B / C / D)
After render: writes metrics JSON + contact sheet + proof frames, then DELETES
the heavy raw MP4 + work dir (keeps proof). Honors the storage limit.
"""
import hashlib, json, os, re, subprocess, sys, time, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {"VIDLORE_AIMG": "1", "VIDLORE_SUBJECT_FLOOR": "1",
             "VIDLORE_MOTION_GRAPHICS": "1", "VIDLORE_STOCK_FILMABLE": "1",
             "VIDLORE_OVERLAY_RESTRAINT": "1", "VIDLORE_AI_VIDEO": "0",
             "VIDLORE_REAL_PERSON": "1", "VIDLORE_WIKIMEDIA": "1",
             "VIDLORE_VISUAL_RELEVANCE": "1", "VIDLORE_VISUAL_RELEVANCE_CACHE": "0",
             "VIDLORE_FAL_IMAGE_BUDGET_MODE": "quality_first",
             "VIDLORE_MG_DENSITY": "0.42", "VIDLORE_EDITORIAL_QA": "1"}.items():
    os.environ.setdefault(k, v)

# Each scene = (narration, intensity, role, graphic_kind, graphic_body).
# graphic_kind set ONLY on intentional structured beats (taught kinds); the rest
# are footage-first ("") and may earn a prose-auto card from the director.
SAMPLES = {
 "A": {"title": "The Eastern Front: How Supply Lines Decided a War",
   "niche": "geopolitics", "fmt": "documentary",
   "prompt": "How logistics and supply lines, more than battles, decided the Eastern Front.",
   "scenes": [
    ("In the summer of 1941, the largest invasion in history rolled east across a thousand-mile front.", 4, "hook", "", ""),
    ("The campaign opened at the border city of Brest, where the first shots were fired before dawn.", 3, "context", "location", "Brest|Eastern Front border"),
    ("Three army groups drove toward three prizes: Leningrad in the north, Moscow in the centre, Kiev in the south.", 4, "context", "map_route", "Brest:0.2,0.5|Kiev:0.45,0.7|Moscow:0.6,0.35|Leningrad:0.5,0.2"),
    ("By autumn the front had advanced six hundred kilometres, faster than any army had ever moved.", 4, "turn", "", ""),
    ("But every kilometre east stretched the supply lines thinner.", 3, "context", "", ""),
    ("Fuel, ammunition and food all travelled the same overloaded rail lines from the west.", 3, "context", "", ""),
    ("A single tank consumed two hundred litres of fuel for every hundred kilometres it moved.", 4, "proof", "", ""),
    ("The timeline of the campaign tells the real story.", 3, "context", "timeline", "Jun 1941|Invasion begins;;Sep 1941|Kiev encircled;;Dec 1941|Halt at Moscow;;Nov 1942|Stalingrad trap"),
    ("As winter closed in, the advance that had covered six hundred kilometres slowed to a crawl.", 4, "turn", "", ""),
    ("Temperatures fell to minus forty, and engines froze where they stood.", 4, "context", "", ""),
    ("The industrial heartland the invaders sought lay just beyond their reach.", 3, "context", "map_region", "Donbas|Industrial heartland"),
    ("A captured logistics report admitted the trucks were wearing out faster than they could be replaced.", 3, "proof", "document", "WEHRMACHT LOGISTICS REPORT, 1941|Trucks unfit for service: 38 percent"),
    ("Behind the front, partisans cut the rail lines that fed the entire advance.", 3, "context", "", ""),
    ("Each severed line meant a division went a day without fuel.", 4, "proof", "", ""),
    ("By the second winter the initiative had passed to the defenders.", 4, "turn", "", ""),
    ("Logistics, not tactics, had set the limit of the advance.", 4, "thesis", "statement", "The front stopped where the supply lines ran dry.||"),
    ("The lesson would be studied by every army that came after.", 3, "context", "", ""),
    ("In the end, the war in the east was won and lost on the railways.", 4, "thesis", "", ""),
   ]},
 "B": {"title": "The Cambridge Ring: Britain's Deepest Betrayal",
   "niche": "spy", "fmt": "documentary",
   "prompt": "How five recruits passed Britain's deepest secrets for two decades undetected.",
   "scenes": [
    ("For two decades, Britain's most guarded secrets flowed quietly to a foreign power.", 4, "hook", "", ""),
    ("The leak began not in a back alley, but in the lecture halls of a famous university.", 3, "context", "location", "Cambridge|The recruiting ground"),
    ("Five young men were recruited as students, told to bury their politics and climb.", 3, "context", "", ""),
    ("One by one they entered the Foreign Office, the intelligence service, and the embassy in Washington.", 4, "context", "", ""),
    ("The network connected handlers, couriers and dead drops across two continents.", 3, "context", "network_graph", "The Handler|Courier:Embassy:Dead drop:Analyst"),
    ("A clerk later testified that one man received visitors at odd hours and paid always in cash.", 3, "proof", "", ""),
    ("Counter-intelligence opened a file, but every lead dissolved into respectability.", 3, "turn", "", ""),
    ("The first hard evidence was a decrypted cable naming a source inside the embassy.", 4, "proof", "evidence", "Decrypted cable, 1949|EXHIBIT A"),
    ("The hunt narrowed to a short list of officers with access to the leaked documents.", 3, "context", "", ""),
    ("When the net finally closed, two of the ring slipped away to the east overnight.", 4, "turn", "", ""),
    ("The defection made headlines no censor could contain.", 3, "context", "document", "MISSING DIPLOMATS, 1951|Two officials vanish"),
    ("A classified review concluded the damage was, in its own word, incalculable.", 4, "proof", "classified", "DAMAGE ASSESSMENT|CONFIRMED"),
    ("Some of the most sensitive findings were struck from the public record entirely.", 3, "context", "redacted", "The committee found the service had been penetrated at the highest level."),
    ("It took years to identify the man who had warned the others to run.", 3, "turn", "", ""),
    ("He had hidden in plain sight, trusted to the very end.", 4, "context", "", ""),
    ("The verdict of history remains divided.", 4, "thesis", "statement", "Traitors to some, true believers to others.||"),
    ("What is certain is the cost: networks blown, agents lost, trust shattered.", 4, "proof", "", ""),
    ("The Cambridge ring became the measure against which every later betrayal was judged.", 4, "thesis", "", ""),
   ]},
 "C": {"title": "Andrew Carnegie: The Man Who Forged Steel",
   "niche": "biography", "fmt": "documentary",
   "prompt": "From a penniless immigrant to the richest man in the world, and the empire of steel he built.",
   "scenes": [
    ("He arrived in America with nothing, and left it the richest man in the world.", 4, "hook", "", ""),
    ("Andrew Carnegie was born in a weaver's cottage in Scotland in 1835.", 3, "context", "name_reveal", "Andrew Carnegie|1835 – 1919"),
    ("His family emigrated when the looms fell silent, settling in Pennsylvania.", 3, "context", "", ""),
    ("At thirteen he worked a twelve-hour day in a cotton mill for a little over a dollar a week.", 4, "proof", "", ""),
    ("His life turned on a series of milestones, each one a rung higher.", 3, "context", "timeline", "1848|Mill boy;;1853|Railroad clerk;;1865|Iron works;;1892|Steel empire"),
    ("A mentor lent him books, and taught him to invest his first spare dollars.", 3, "context", "", ""),
    ("He saw that the future belonged not to iron, but to a stronger metal: steel.", 4, "turn", "", ""),
    ("He built the largest steelworks the world had ever seen on the rivers of Pittsburgh.", 4, "context", "", ""),
    ("His fortune climbed from thousands to millions to a sum without precedent.", 4, "proof", "", ""),
    ("By 1901 he sold his empire for four hundred and eighty million dollars.", 4, "proof", "", ""),
    ("He stood among a small circle of partners and rivals who reshaped American industry.", 3, "context", "", ""),
    ("Carnegie ranked the causes he would fund, and put libraries at the very top.", 3, "context", "ranking", "Libraries|2509;;Education|20;;Peace|10;;Science|8"),
    ("He gave away nine-tenths of his wealth before he died.", 4, "proof", "", ""),
    ("His own creed was blunt: the man who dies rich, he wrote, dies disgraced.", 4, "proof", "pull_quote", "The man who dies rich dies disgraced.|Andrew Carnegie"),
    ("Critics remembered the bitter strike that broke his workers' union.", 4, "turn", "", ""),
    ("Admirers remembered the thousands of libraries that bore no name but the town's.", 3, "context", "", ""),
    ("History holds both truths at once.", 4, "thesis", "statement", "A ruthless industrialist and a radical philanthropist.||"),
    ("The boy from the weaver's cottage had remade a nation, for better and for worse.", 4, "thesis", "", ""),
   ]},
 "D": {"title": "How GPS Knows Exactly Where You Are",
   "niche": "technology", "fmt": "documentary",
   "prompt": "The science of how a constellation of satellites pinpoints your position to a few metres.",
   "scenes": [
    ("Right now, a device in your pocket knows where you are to within a few metres.", 4, "hook", "", ""),
    ("It does this by listening to a constellation of satellites twenty thousand kilometres overhead.", 3, "context", "", ""),
    ("The principle has a name: trilateration, measuring distance from several known points.", 4, "explain", "define_the_term", "Trilateration|Fixing a position from distances to known points"),
    ("Each satellite carries an atomic clock accurate to a billionth of a second.", 4, "proof", "", ""),
    ("It broadcasts a single message: my position, and the exact time I sent this.", 3, "context", "", ""),
    ("Your receiver compares that time to its own clock to learn how far the signal travelled.", 4, "explain", "", ""),
    ("The signal travels at the speed of light, so a millionth of a second equals three hundred metres.", 4, "proof", "", ""),
    ("One satellite places you on a sphere; a second narrows it to a circle.", 3, "explain", "", ""),
    ("A third shrinks it to two points, and a fourth fixes you in space and time.", 4, "proof", "", ""),
    ("The whole journey, from orbit to pocket, is a chain of precise hand-offs.", 3, "context", "route_trace", "0.15,0.25:Satellite|0.5,0.5:Atmosphere|0.85,0.75:Receiver"),
    ("But the atmosphere bends the signal, adding tiny errors along the way.", 3, "turn", "", ""),
    ("Engineers correct for it with ground stations that constantly measure the drift.", 3, "context", "", ""),
    ("Even gravity matters: clocks in orbit tick faster than clocks on the ground.", 4, "proof", "", ""),
    ("Relativity, once pure theory, is now corrected for in every phone on Earth.", 4, "turn", "", ""),
    ("Compare the scale: a satellite the size of a car, guiding ships the length of a city block.", 3, "context", "scale_compare", "Satellite:5:5 m|Cargo ship:300:300 m"),
    ("Without these corrections, your position would drift by ten kilometres a day.", 4, "proof", "", ""),
    ("A system built for missiles now guides farmers, pilots and pedestrians alike.", 3, "context", "", ""),
    ("Invisible, silent, and everywhere: the quiet machine that always knows where you stand.", 4, "thesis", "", ""),
   ]},
}


def _kw(nar):
    kws = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    for w in re.findall(r"\b[a-z]{5,}\b", (nar or "").lower()):
        if w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        if k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out[:5]


def build_script(s):
    scenes = []
    for i, (nar, inten, role, gk, gb) in enumerate(s["scenes"]):
        scenes.append({"narration": nar, "keywords": _kw(nar), "visual": "",
                       "intensity": inten, "emphasis": "",
                       "shot_type": "wide" if i % 3 == 0 else "medium",
                       "role": role, "graphic_kind": gk, "graphic_text": "",
                       "graphic_body": gb})
    return {"title": s["title"], "source_sha256": hashlib.sha256(
        (s["title"] + "".join(x["narration"] for x in scenes)).encode()).hexdigest(),
        "scenes": scenes}


def main(sid):
    s = SAMPLES[sid]
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    cfg = load_config()
    out = ROOT / "output"
    brief = Brief(title=s["title"], prompt=s["prompt"], fmt=s["fmt"], duration="18-20",
                  theme="modern", captions=True, background="auto",
                  extra={"niche": s["niche"]})
    run_dir = run_dir_for(brief, out); run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script(s)
    body = s["title"] + "\n\n" + "\n\n".join(x["narration"] for x in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(s['title'])}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s video_exists={vid.exists()} path={vid}", flush=True)
    print(f"SAMPLE {sid} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
