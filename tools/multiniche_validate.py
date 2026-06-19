#!/usr/bin/env python3
"""MULTI-NICHE REAL VALIDATION — 5 short documentary samples through the real
Vidlore pipeline with the V2.2 (39-primitive) motion-graphics library.

Niches: spy / crime / business / history / geopolitics. Each is an ORIGINAL
~2.5-3 min script (not copied from any reference) with realistic graphic
OPPORTUNITIES tagged on the scenes where a premium graphic genuinely fits — most
scenes are pure footage. Graphics are NOT forced; the Motion Graphics Director
chooses naturally within a restrained density budget (default 0.35), so it skips
weaker opportunities (which the audit then inspects).

  python3 tools/multiniche_validate.py <id>            # author script.json + FREE dry-run
  python3 tools/multiniche_validate.py <id> --render   # + real cost-tracked render
  python3 tools/multiniche_validate.py all             # dry-run all 5 (no render)

  <id> in {spy, crime, business, history, geopolitics}

The 3 newest primitives are exercised in varied niches: world_map_arc (spy +
geopolitics), flowchart_decision (crime + geopolitics), spotlight_object_hold
(history). Portraits (name_reveal) appear where a real public figure has good
public-domain imagery, to test likeness + crop safety.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTROOT = ROOT / "research/motion_graphics_qa/multiniche"

# Each scene: (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
SAMPLES = {
    "spy": {
        "title": "Our Man in Damascus: The Spy Who Reached the Heart of an Enemy State",
        "niche": "geopolitics", "theme": "modern",
        "prompt": "How one deep-cover operative shifted the balance of a conflict",
        "scenes": [
            ("In the early 1960s, a single man walked into the most guarded circles "
             "of a hostile capital — and for years, not one of them suspected a thing.",
             4, "hook", "", "", ""),
            ("His name was Eli Cohen, an Egyptian-born Israeli who became his "
             "nation's most valuable pair of eyes behind enemy lines.",
             3, "intro", "name_reveal", "Eli Cohen", ""),
            ("Recruited and trained in secret, he built a false identity so complete "
             "that it could survive a long dinner seated beside generals.",
             3, "setup", "", "", ""),
            ("From a quiet room in Tel Aviv, his handlers ran the connection across "
             "the border, a fragile thread reaching all the way into Damascus.",
             3, "reach", "world_arc", "TEL AVIV",
             "to=DAMASCUS ; from_pos=0.28,0.56 ; to_pos=0.64,0.40 ; title=THE LONG REACH"),
            ("Posing as a wealthy returning Syrian, he threw lavish parties for the "
             "political elite, who spoke freely while he simply listened.",
             3, "cover", "", "", ""),
            ("Night after night he transmitted the classified secrets of a state — "
             "troop positions, fortifications, plans — tapped out in careful code.",
             4, "intel", "redacted", "GOLAN DEFENCES — FULL FORWARD LAYOUT",
             "title=INTERCEPT FORTY-ONE ; stamp=TOP SECRET"),
            ("What he sent home would later help shape the outcome of a war before a "
             "single shot of it had been fired.",
             4, "impact", "", "", ""),
            ("He moved easily among the men who ran the country — ministers and "
             "officers, each a thread in a web of quiet, well-placed contacts.",
             3, "network", "connection_web", "THE DAMASCUS CIRCLE",
             "nodes=The Minister|The General|The Party Man|The Broadcaster|The Agent ; "
             "links=The Agent-The Minister|The Agent-The General|The Agent-The Party Man|"
             "The Minister-The General"),
            ("But every broadcast carried a risk, and the enemy had begun to hunt for "
             "a signal that did not belong on their airwaves.",
             4, "danger", "", "", ""),
            ("Foreign experts had arrived to help hunt the signal, the net was "
             "tightening around him, and time was running out.",
             4, "clock", "countdown", "TIME REMAINING",
             "value=21 ; to=0 ; unit=DAYS"),
            ("They traced the transmissions to a single apartment and broke down the "
             "door while his hand was still resting on the key.",
             5, "caught", "", "", ""),
            ("From recruitment to capture, the entire mission had played out across "
             "just a handful of short, extraordinary years.",
             3, "timeline", "timeline", "THE OPERATION",
             "events=Recruited|Into Damascus|The Inner Circle|Exposed"),
            ("His loss was felt as a national tragedy, yet the intelligence he had "
             "already sent home would outlast the man himself by decades.",
             4, "loss", "", "", ""),
            ("One operative, planted deep inside the enemy's own house, had quietly "
             "shifted the balance of an entire conflict.",
             4, "thesis", "statement", "ONE MAN REDREW THE MAP", ""),
            ("And his name is still spoken today with a rare mix of grief and awe — "
             "the spy who got closer than anyone believed was possible.",
             3, "outro", "", "", ""),
        ],
    },
    "crime": {
        "title": "The Untouchable: How They Finally Caught Al Capone",
        "niche": "crime", "theme": "crime",
        "prompt": "How the most powerful gangster in America was finally brought down",
        "scenes": [
            ("For a few violent years, one man owned an American city — its bars, its "
             "police, its politicians, and the fear that held it all together.",
             4, "hook", "", "", ""),
            ("He was Al Capone, a Brooklyn-born racketeer who turned Prohibition into "
             "an empire built on beer, blood, and bribes.",
             3, "intro", "name_reveal", "Al Capone", "place=Chicago, Illinois"),
            ("Rivals who challenged him tended to disappear, most infamously on a "
             "single morning in a cold North Side garage.",
             4, "violence", "", "", ""),
            ("From the South Side of Chicago, his organisation ran an illegal trade "
             "that stretched across the city and far beyond its limits.",
             3, "territory", "territory", "Chicago's South Side",
             "pos=0.46,0.50 ; sub=Headquarters of the operation ; title=THE TERRITORY"),
            ("The money was staggering. At its height the operation pulled in tens of "
             "millions of dollars a year, almost none of it ever taxed.",
             4, "money", "", "", ""),
            ("For all his power, investigators could never make a violent charge "
             "stick — every witness was too frightened, or too dead, to testify.",
             4, "wall", "", "", ""),
            ("So the prosecutors faced a choice that would define the case — chase "
             "the murders they could not prove, or the money he could not hide.",
             4, "decision", "decision", "Chase the murders, or the money?",
             "yes=Pursue the killings ; no=Prosecute the hidden income ; chosen=no ; "
             "title=THE GAMBLE"),
            ("A small team of agents began quietly assembling the one thing the mob "
             "could not intimidate — a paper trail.",
             3, "build", "", "", ""),
            ("Inside a seized ledger they found it: columns of untaxed profit, "
             "written in plain ink, tying the empire straight back to its boss.",
             4, "evidence", "redacted", "NET PROFIT — UNDECLARED FOR SIX YEARS",
             "title=EXHIBIT — THE LEDGER ; stamp=EVIDENCE"),
            ("The charge, when it came, was almost an insult to a man of his "
             "reputation — not murder, not extortion, but tax evasion.",
             4, "charge", "", "", ""),
            ("From his rise to his fall, the empire had burned bright and fast, and "
             "then collapsed in a single courtroom.",
             3, "timeline", "timeline", "THE RECKONING",
             "events=The Rise|The Peak|The Ledger|The Verdict"),
            ("In the autumn of nineteen thirty-one the jury returned, and the most "
             "feared man in America was led away to a federal cell.",
             4, "verdict", "", "", ""),
            ("The press, once terrified, now spoke freely, and the headlines told the "
             "story of a kingdom undone by an accountant's pen.",
             3, "press", "quote_stream", "THE VERDICT",
             "quotes=Brought down by arithmetic.:The Tribune|The empire of fear is "
             "over.:The Daily News|Gangland's god is mortal.:The Examiner"),
            ("He would never run the city again, and the men who took him down had "
             "rewritten the rules for fighting organised crime.",
             3, "outro", "", "", ""),
        ],
    },
    "business": {
        "title": "The Richest Man in the World: Andrew Carnegie and the Empire of Steel",
        "niche": "business", "theme": "standard",
        "prompt": "How a penniless immigrant built the largest steel empire on earth",
        "scenes": [
            ("He arrived in America with almost nothing, and he would leave it as the "
             "richest man the modern world had yet produced.",
             4, "hook", "", "", ""),
            ("His name was Andrew Carnegie, a Scottish weaver's son who saw, earlier "
             "than almost anyone, that the future would be forged in steel.",
             3, "intro", "name_reveal", "Andrew Carnegie", "place=Pittsburgh"),
            ("He bet everything on a new process that could turn iron into steel "
             "faster and cheaper than the world had ever seen.",
             3, "bet", "", "", ""),
            ("As the railroads and cities of a growing nation devoured every ton he "
             "could make, his output climbed year after year without pause.",
             4, "growth", "growth", "STEEL OUTPUT",
             "points=1875:38|1885:160|1895:480|1900:1400 ; suffix=K TONS ; title=TONS PER YEAR"),
            ("He was ruthless about cost and obsessive about efficiency, undercutting "
             "any rival until they sold out or simply shut their doors.",
             3, "method", "", "", ""),
            ("To control his costs completely, he bought the entire chain of "
             "production, from the raw ore in the ground to the finished beam.",
             3, "vertical", "org_tree", "CARNEGIE STEEL",
             "children=Ore Mines|Coke Works|The Mills|Rail & Shipping"),
            ("Nothing was left to chance, and nothing was wasted — not a furnace, not "
             "a freight car, not a single hour of a working day.",
             3, "control", "", "", ""),
            ("Every dollar of profit was poured straight back into the machine — into "
             "mills, into mines, into the ships and rails that fed them.",
             4, "reinvest", "sankey", "WHERE THE PROFITS WENT",
             "branches=New Mills:46|Mines & Ore:24|Rail & Ships:18|Reserve:12 ; "
             "source=PROFIT ; title=EVERY DOLLAR REINVESTED"),
            ("Then, at the height of his power, he did something no one expected — he "
             "sold the entire empire and walked away.",
             4, "sale", "", "", ""),
            ("The sale left only one fortune in America that could rival his own, a "
             "rival empire built on oil rather than steel.",
             3, "rival", "comparison", "TWO TITANS",
             "pair=Carnegie · Steel|Rockefeller · Oil ; values=480|520 ; "
             "title=THE GREAT FORTUNES"),
            ("But what he did next was, to him, the real point — he set out to give "
             "the entire fortune away before he died.",
             4, "give", "", "", ""),
            ("He believed that great wealth was a trust to be spent on the public "
             "good, and that to die rich was, in his words, to die disgraced.",
             4, "thesis", "statement", "THE MAN WHO DIED TO GIVE IT AWAY", ""),
            ("Libraries, concert halls, and universities still carry his name, funded "
             "by the empire of steel that one immigrant built from nothing.",
             3, "outro", "", "", ""),
        ],
    },
    "history": {
        "title": "The Road to Moscow: Napoleon's Catastrophe of 1812",
        "niche": "history", "theme": "history",
        "prompt": "How the largest army in European history marched to ruin",
        "scenes": [
            ("In the summer of 1812, the most powerful man in Europe led the largest "
             "army the continent had ever seen toward a single, distant city.",
             4, "hook", "", "", ""),
            ("At its head rode Napoleon Bonaparte, a commander who had not lost a "
             "major campaign in over a decade.",
             3, "intro", "name_reveal", "Napoleon Bonaparte", "place=Moscow"),
            ("He crossed into Russia with a force drawn from half of Europe, "
             "confident the war would be won in a single, decisive summer.",
             3, "invade", "", "", ""),
            ("His route carried that vast army eastward across hundreds of miles of "
             "open country, deeper and deeper toward Moscow.",
             3, "route", "map_route", "THE MARCH EAST",
             "stops=Vilnius|Smolensk|Borodino|Moscow ; title=THE ADVANCE"),
            ("But the Russians refused to be drawn into the decisive battle he "
             "needed, retreating and burning everything behind them as they went.",
             4, "retreat", "", "", ""),
            ("This was a deliberate strategy with an old and brutal name — leave the "
             "advancing enemy nothing at all to live on.",
             3, "define", "definition", "Scorched Earth",
             "definition=Destroying all food and shelter so an invader cannot supply itself"),
            ("By the time he reached Moscow, the city was nearly empty, and within a "
             "day much of it was in flames.",
             4, "burning", "", "", ""),
            ("And then the real enemy arrived — not an army, but the Russian winter, "
             "against which no European force had ever been prepared.",
             5, "winter", "reveal", "GENERAL WINTER",
             "kicker=THE ENEMY NO ARMY COULD BEAT ; sub=Temperatures fell beyond forty "
             "below ; title=THE LONG RETREAT"),
            ("The retreat became a massacre by cold and hunger, the roadside lined "
             "with the men and horses of a dying army.",
             5, "massacre", "", "", ""),
            ("The numbers tell the whole catastrophe — of more than 600,000 who "
             "marched in, barely 27,000 men ever marched home.",
             5, "numbers", "bar_chart", "THE ARMY THAT VANISHED",
             "bars=Marched In:610|Returned:27 ; suffix=K MEN ; title=SIX HUNDRED THOUSAND TO TWENTY-SEVEN"),
            ("Stragglers froze where they fell, and the long road back to the border "
             "became a frozen trail of the dead and the dying.",
             5, "aftermath", "", "", ""),
            ("The disaster had not been caused by one mistake but by a chain of them, "
             "each one feeding the next until nothing was left.",
             4, "cause", "cause_effect", "HOW IT UNRAVELLED",
             "steps=A scorched land|No supplies|The early winter|The army destroyed"),
            ("Napoleon would survive, but the spell of his invincibility was broken, "
             "and within three years his empire was gone.",
             4, "outro", "", "", ""),
        ],
    },
    "geopolitics": {
        "title": "Thirteen Days: The World on the Edge of Nuclear War",
        "niche": "geopolitics", "theme": "modern",
        "prompt": "How the Cuban Missile Crisis brought the world to the brink",
        "scenes": [
            ("For thirteen days in the autumn of 1962, the entire world held its "
             "breath as two superpowers stood a single order away from nuclear war.",
             4, "hook", "", "", ""),
            ("At the centre stood the young president, John F. Kennedy, just two "
             "years into office and facing the gravest decision of the century.",
             3, "intro", "name_reveal", "John F. Kennedy", ""),
            ("The crisis began when spy planes photographed something impossible — "
             "Soviet nuclear missiles being assembled ninety miles from Florida.",
             4, "discovery", "", "", ""),
            ("Those weapons had been shipped in secret across the ocean, all the way "
             "from the Soviet Union to the island of Cuba.",
             4, "shipment", "world_arc", "MOSCOW",
             "to=CUBA ; from_pos=0.64,0.34 ; to_pos=0.30,0.56 ; title=THE SECRET CARGO"),
            ("From Cuba, those missiles could reach most major American cities within "
             "minutes, erasing any warning the country had relied on.",
             4, "threat_geo", "", "", ""),
            ("Military planners raised the nation's alert to a level it had almost "
             "never reached, one step short of all-out war.",
             5, "defcon", "threat_level", "ALERT LEVEL",
             "value=80 ; bands=CALM|TENSION|ALERT|DEFCON 2|WAR ; readout=DEFCON 2 ; "
             "title=ONE STEP FROM WAR"),
            ("Inside the White House, a small group of advisers argued for days over "
             "how to respond without igniting the very war they feared.",
             4, "debate", "", "", ""),
            ("In the end the President faced a choice between two stark options, and "
             "every man in the room knew that each one was terrifying.",
             4, "decision", "decision", "Strike the missiles, or blockade the island?",
             "yes=Bomb the sites at once ; no=Blockade and negotiate ; chosen=no ; "
             "title=THE DECISION"),
            ("He chose the blockade — a ring of warships around Cuba — buying time "
             "and a way out that did not begin with the first shot.",
             4, "blockade", "", "", ""),
            ("Behind the standoff sat three men whose every move was read by the "
             "others — in Washington, in Moscow, and in Havana.",
             3, "players", "connection_web", "THE THREE CAPITALS",
             "nodes=Washington|Moscow|Havana|The U.N.|The Navy ; "
             "links=Washington-Moscow|Moscow-Havana|Washington-The U.N.|Washington-The Navy"),
            ("For days the ships approached the line, and the world waited to see "
             "whether the Soviet captains would stop or sail straight through.",
             5, "approach", "", "", ""),
            ("At the last moment they stopped, and a secret deal began to take shape "
             "behind the public confrontation.",
             4, "deal", "", "", ""),
            ("From the first photograph to the final agreement, the whole crisis had "
             "lasted just under two weeks.",
             3, "timeline", "timeline", "THIRTEEN DAYS",
             "events=Day 1:Discovery|Day 3:Blockade|Day 9:The Brink|Day 13:Resolved"),
            ("Both sides stepped back, but the memory of how close they had come "
             "would shape the rest of the Cold War.",
             4, "outro", "", "", ""),
        ],
    },
    "science": {
        "title": "The Invisible Killer: How John Snow Mapped a Cholera Outbreak",
        "niche": "science", "theme": "history",
        "prompt": "How a doctor traced a deadly epidemic to a single water pump",
        "scenes": [
            ("In the late summer of 1854, a deadly sickness swept through a London "
             "neighbourhood, killing people within hours of their first symptom.",
             4, "hook", "", "", ""),
            ("One physician refused the common belief that the disease spread through "
             "foul air — his name was John Snow.",
             3, "intro", "name_reveal", "John Snow", "place=London"),
            ("He suspected the true cause was not the air at all, but something "
             "hidden in the water the neighbourhood drank every day.",
             3, "theory", "", "", ""),
            ("So he did something almost no one had done before — he walked the "
             "streets and marked every single death on a map.",
             3, "method", "", "", ""),
            ("A clear pattern emerged: the deaths clustered tightly around one public "
             "water pump on Broad Street.",
             4, "cluster", "territory", "Broad Street Pump",
             "pos=0.50,0.48 ; sub=Centre of the outbreak ; title=THE CLUSTER"),
            ("The numbers were stark — homes nearest the pump suffered far more "
             "deaths than those that drew water elsewhere.",
             4, "numbers", "bar_chart", "DEATHS BY WATER SOURCE",
             "bars=Broad St Pump:127|Other Pumps:18 ; suffix= DEATHS ; title=ONE PUMP, MOST OF THE DEAD"),
            ("Cholera, he reasoned, was spread when waste from the sick contaminated "
             "the drinking supply — a route through water, not air.",
             3, "define", "definition", "Waterborne Disease",
             "definition=An illness spread by drinking water contaminated with a pathogen"),
            ("He took his evidence to the local officials and made one simple, "
             "radical request — remove the handle from the pump.",
             4, "action", "", "", ""),
            ("With the pump disabled, the neighbourhood could no longer draw its "
             "tainted water, and the wave of new cases began to fall away.",
             4, "result", "", "", ""),
            ("From the first death to the silenced pump, the investigation had "
             "unfolded in a matter of weeks.",
             3, "timeline", "timeline", "THE INVESTIGATION",
             "events=The Outbreak|The Map|The Pump|The Handle Removed"),
            ("His map became one of the founding documents of a new science — the "
             "study of how disease moves through a population.",
             4, "thesis", "statement", "A MAP THAT BIRTHED EPIDEMIOLOGY", ""),
            ("Today every outbreak investigation still follows the path he walked — "
             "find the pattern, find the source, and shut it off.",
             3, "outro", "", "", ""),
        ],
    },
    "technology": {
        "title": "The Machine That Won the War: Cracking Enigma at Bletchley Park",
        "niche": "technology", "theme": "history",
        "prompt": "How a team of codebreakers built a machine to defeat an unbreakable cipher",
        "scenes": [
            ("For years, an enemy cipher scrambled every order, every position, every "
             "secret — and it was believed to be utterly unbreakable.",
             4, "hook", "", "", ""),
            ("The man who would break it was a brilliant, awkward mathematician named "
             "Alan Turing.",
             3, "intro", "name_reveal", "Alan Turing", "place=Bletchley Park"),
            ("The cipher came from a machine called Enigma, whose rotating wheels "
             "could scramble a message in more ways than there are stars in the sky.",
             4, "scale", "", "", ""),
            ("Every day at midnight the settings changed, wiping out any progress and "
             "forcing the codebreakers to start again from nothing.",
             4, "problem", "", "", ""),
            ("Turing's insight was that no human team could ever search the settings "
             "fast enough — so he would build a machine to do it instead.",
             3, "insight", "", "", ""),
            ("His device, the Bombe, tested thousands of possible settings in the "
             "time it took a human to check just one.",
             4, "machine", "growth", "SETTINGS TESTED PER HOUR",
             "points=Human:1|Early Bombe:600|Refined Bombe:4200 ; title=THE SPEED LEAP"),
            ("The breakthrough relied on a fatal flaw — the machine could never "
             "encode a letter as itself, and that crack was enough.",
             4, "flaw", "", "", ""),
            ("The codebreaking flowed through a careful chain, from intercepted "
             "signal to decoded order on a commander's desk.",
             3, "pipeline", "cause_effect", "FROM SIGNAL TO SECRET",
             "steps=Intercept the signal|Run the Bombe|Find the settings|Read the orders"),
            ("With Enigma readable, Allied commanders could see the enemy's plans "
             "almost as quickly as the enemy's own officers could.",
             4, "impact", "", "", ""),
            ("Historians later estimated the work shortened the war by years and "
             "saved countless lives on every front.",
             4, "scale2", "statement", "A MACHINE THAT SHORTENED A WAR", ""),
            ("And in solving an impossible puzzle, Turing had sketched the blueprint "
             "for something far larger — the modern computer.",
             4, "legacy", "", "", ""),
            ("The cipher that could not be broken had been beaten not by a man, but "
             "by the first glimpse of the machines that would follow.",
             3, "outro", "", "", ""),
        ],
    },
}


def _dryrun_assets(gk: str, gt: str, gb: str) -> dict:
    """Build assets sufficient for the DIRECTOR to select correctly (required
    inputs + key fields). The real pipeline.py adapter does the full hint parse
    at render time; this only needs to validate natural SELECTION cheaply."""
    a = {}
    gb = gb or ""

    def lst(key):
        m = re.search(rf"{key}=([^;]+)", gb)
        return [s.strip() for s in m.group(1).split("|") if s.strip()] if m else None

    def num(key):
        m = re.search(rf"{key}=([\d.,]+)", gb)
        return float(m.group(1).replace(",", "")) if m else None

    def pairs(key):
        m = re.search(rf"{key}=([^;]+)", gb)
        out = []
        if m:
            for p in m.group(1).split("|"):
                if ":" in p:
                    nm, vv = p.rsplit(":", 1)
                    try:
                        out.append([nm.strip(), float(re.sub(r"[^\d.\-]", "", vv))])
                    except ValueError:
                        pass
        return out or None

    if gk in ("name_reveal", "mugshot", "bio", "mini_bio") and gt:
        a["portrait_path"] = "/tmp/_d.png"
        a["name"] = gt
        m = re.search(r"place=([^;|]+)", gb)
        if m:
            a["place"] = m.group(1).strip()
    if gk in ("spotlight", "reveal", "subject_reveal", "behold", "unveil") and gt:
        a["subject"] = gt
    if gk in ("world_arc", "world_map_arc", "arc", "transatlantic", "great_circle") and gt:
        a["from_place"] = gt
        m = re.search(r"to=([^;|]+)", gb)
        if m:
            a["to_place"] = m.group(1).strip()
    if gk in ("decision", "flowchart", "branch", "fork", "choice", "yes_no") and gt:
        a["question"] = gt
    if gk in ("connection_web", "network_web", "relationship_map", "conspiracy_web",
              "connections") and lst("nodes"):
        a["nodes"] = lst("nodes")
    if gk in ("countdown", "deadline", "timer", "ticking_clock") and num("value") is not None:
        a["value"] = num("value")
    if gk in ("timeline", "era_banner", "chronology", "chapter") and gb:
        m = re.search(r"events=([^;]+)", gb)
        if m:
            a["events"] = [[p.split(":", 1)[0].strip(),
                            (p.split(":", 1)[1].strip() if ":" in p else "")]
                           for p in m.group(1).split("|") if p.strip()]
    if gk in ("region", "map_region", "territory", "region_highlight") and gt:
        a["region"] = gt
    if gk in ("gauge", "meter", "threat_level", "severity", "rating") and num("value") is not None:
        a["value"] = num("value")
    if gk in ("cause_effect", "causation", "cause_chain", "domino") and lst("steps"):
        a["steps"] = lst("steps")
    if gk in ("sankey", "money_split", "allocation", "flow_breakdown") and pairs("branches"):
        a["branches"] = pairs("branches")
    if gk in ("hierarchy", "org_chart", "structure", "org_tree") and lst("children"):
        a["root"] = gt or "ROOT"
        a["children"] = lst("children")
    if gk in ("comparison", "versus", "vs", "ratio") and gt:
        m = re.search(r"pair=([^;|]+\|[^;]+)", gb)
        if m:
            lr = m.group(1).split("|")
            a["left"], a["right"] = lr[0].strip(), lr[1].strip()
    if gk in ("growth", "trend", "line_chart", "trajectory", "curve") and pairs("points"):
        a["points"] = pairs("points")
    if gk in ("bar_chart", "stat_bars", "breakdown", "data_viz") and pairs("bars"):
        a["bars"] = pairs("bars")
    if gk in ("map_route", "route", "journey", "expansion") and lst("stops"):
        a["stops"] = lst("stops")
    if gk in ("classified", "redacted", "secret", "top_secret", "declassified",
              "dossier", "leak") and gt:
        a["reveal"] = gt
    if gk in ("quote_stream", "quotes", "chorus", "reactions") and gb:
        m = re.search(r"quotes=([^;]+)", gb)
        if m:
            qs = []
            for p in m.group(1).split("|"):
                if ":" in p:
                    tx, by = p.rsplit(":", 1)
                    qs.append([tx.strip(), by.strip()])
                elif p.strip():
                    qs.append([p.strip(), ""])
            if qs:
                a["quotes"] = qs
    if gk in ("statement", "thesis", "claim", "takeaway") and gt:
        a["text"] = gt
    if gk in ("definition", "define", "term", "glossary") and gt:
        a["term"] = gt
        m = re.search(r"definition=([^;|]+)", gb)
        if m:
            a["definition"] = m.group(1).strip()
    if gk in ("ranking", "leaderboard", "top_list", "ranked") and pairs("items"):
        a["items"] = pairs("items")
    return a


def build_script(sample: dict) -> dict:
    scenes = []
    for i, (nar, inten, role, gk, gt, gb) in enumerate(sample["scenes"]):
        scenes.append({
            "narration": nar, "keywords": [], "visual": "",
            "intensity": inten, "emphasis": "",
            "shot_type": "wide" if i % 3 == 0 else "medium",
            "role": role, "graphic_kind": gk, "graphic_text": gt,
            "graphic_body": gb,
        })
    body = sample["title"] + "\n\n" + "\n\n".join(s["narration"] for s in scenes)
    return {"title": sample["title"],
            "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "scenes": scenes}


def run_one(sid: str, do_render: bool):
    sample = SAMPLES[sid]
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for
    from vidlore.motion_graphics import director as mgdir, registry as mgreg

    outdir = OUTROOT / sid
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    brief = Brief(title=sample["title"], prompt=sample["prompt"], fmt="documentary",
                  duration="6-8", theme=sample["theme"], captions=True,
                  background="auto", extra={"niche": sample["niche"]})
    out = ROOT / "output"
    run_dir = run_dir_for(brief, out)
    run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script(sample)
    body = sample["title"] + "\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (outdir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    print(f"[{sid}] RUN_DIR {run_dir}", flush=True)

    mg = []
    for i, sc in enumerate(script["scenes"]):
        gk = (sc["graphic_kind"] or "").lower()
        a = _dryrun_assets(gk, sc["graphic_text"] or "", sc["graphic_body"] or "")
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"], "assets": a})
    seed = int(hashlib.sha1(sample["title"].encode()).hexdigest()[:8], 16) % 100000
    dens = float(os.environ.get("VIDLORE_MG_DENSITY", "0.35"))
    dec = mgdir.plan(mg, niche=sample["niche"], seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    pal = mgdir.video_palette(sample["niche"], seed)
    n_opps = sum(1 for s in script["scenes"] if s["graphic_kind"])
    print(f"[{sid}] niche={sample['niche']} palette={pal} density={dens} "
          f"scenes={len(script['scenes'])} opportunities={n_opps}", flush=True)
    print(f"[{sid}] DIRECTOR: " + json.dumps(summ), flush=True)
    (outdir / "dryrun.json").write_text(
        json.dumps({"summary": summ, "palette": pal, "opportunities": n_opps,
                    "decisions": [{"i": d.scene_index, "primitive": d.primitive,
                                   "score": d.score} for d in dec if d.primitive]},
                   indent=2), encoding="utf-8")

    if do_render:
        from vidlore.pipeline import render_from_script
        t0 = time.time()
        render_from_script(brief, cfg, out, keep_work=True)
        vid = run_dir / f"{run_dir.name}.mp4"
        print(f"[{sid}] RENDER_DONE wall={time.time()-t0:.1f}s video={vid.exists()} "
              f"path={vid}", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_render = "--render" in sys.argv
    sid = args[0] if args else "all"
    if sid == "all":
        for s in SAMPLES:
            run_one(s, False)
    elif sid in SAMPLES:
        run_one(sid, do_render)
    else:
        print(f"unknown sample '{sid}' — choose from {list(SAMPLES)} or 'all'")
        sys.exit(1)


if __name__ == "__main__":
    main()
