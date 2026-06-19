"""Motion-graphics DIRECTOR — the editorial brain.

Given the scene list (narration · role · intensity · niche · available assets)
and the per-video editorial recipe, it decides *which* premium motion graphic
(if any) each scene should receive, and with what inputs. Footage remains the
foundation: most scenes get NO graphic. Selection is deterministic per video
(seeded), respects per-primitive caps + cooldowns, enforces a global density
target, and never repeats a primitive/family back-to-back.

`plan(scenes, niche=...) -> [Decision]` is pure + testable (no rendering). The
pipeline integration calls `plan()` then renders the chosen primitives via the
registry, baking them as per-scene overlays.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import registry

# narration cue lexicons (cheap, deterministic)
_MONEY_RE = re.compile(r"\$\s?\d[\d,\.]*\s?(?:million|billion|trillion|k|m|bn)?", re.I)
_PCT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%")
_NUM_RE = re.compile(r"\b\d{2,}(?:,\d{3})*\b")
_EVIDENCE = ("court", "ruling", "guilty", "indict", "lawsuit", "headline",
             "newspaper", "report", "document", "testif", "convict", "trial",
             "dissolv", "verdict", "act ", "law ", "investigat")
_EMPIRE = ("empire", "monopoly", "controlled", "owned", "ownership", "trust",
           "conglomerate", "subsidiar", "holdings", "network of", "stake",
           "acquir", "merge", "rockefeller", "standard oil")
_PLACE = ("cleveland", "new york", "ohio", "pennsylvania", "texas", "russia",
          "london", "europe", "america", "city", "region", "country", "border")
# V1.2 cues — time anchors, quotations, A-vs-B contrasts
_YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20[0-2]\d)\b")
_QUOTE_RE = re.compile(r"[\"“']([^\"”']{12,150})[\"”']")
_VS_RE = re.compile(r"\b(?:vs\.?|versus|compared (?:to|with)|pitted against)\b", re.I)
_QSAY = ("famously said", "once said", "declared", "wrote", "remarked",
         "in his words", "put it", "proclaimed")
# V1.3 cues — a physical artifact / exhibit, and a numeric breakdown
_ARTIFACT = ("exhibit", "artifact", "letter", "note", "cheque", "receipt",
             "ledger", "telegram", "contract", "photograph", "manuscript",
             "blueprint", "passport", "diary", "patent", "the document")
# V1.4 cues — a trajectory over time, a route across space, a detail to examine
_GROWTH = ("grew", "growth", "rose", "rising", "climbed", "surged", "soared",
           "skyrocketed", "doubled", "tripled", "quadrupled", "exploded",
           "exponential", "over the years", "year after year", "decade after",
           "trajectory", "kept climbing", "ballooned", "multiplied")
_ROUTE = ("route", "journey", "voyage", "travelled", "traveled", "sailed",
          "marched", "expanded", "expansion", "spread across", "swept across",
          "supply line", "trade route", "advanced toward", "pushed into",
          "all the way to", "from .* to ", "crossed the", "set out for")
_DETAIL = ("look closer", "look closely", "notice", "in this photo",
           "in this photograph", "the figure", "circled", "zoom in",
           "if you look", "can be seen", "pictured here", "standing here",
           "this detail", "examine", "look at the")
# V1.5 cues — a share of a whole, an ordered process, a top-down hierarchy
_SHARE = ("controlled", "control of", "dominated", "monopoly", "monopolised",
          "monopolized", "share of", "of the market", "of all", "of the world's",
          "majority of", "two-thirds", "three-quarters", "the bulk of",
          "the lion's share", "cornered", "percent of", "per cent of")
_PROCESS = ("the process", "how it worked", "the method", "the scheme",
            "the playbook", "the formula", "the recipe", "step by step",
            "first,", "then,", "next,", "finally,", "the strategy was",
            "it worked like this", "the blueprint")
_HIERARCHY = ("parent company", "subsidiary", "subsidiaries", "holding company",
              "chain of command", "reported to", "under the control of",
              "a division of", "owned by", "beneath", "answered to",
              "the structure", "sat at the top", "the umbrella")
# V1.6 cues — an editorial claim, a countable ratio, a 100% composition
_CLAIM = ("the truth is", "the reality is", "make no mistake", "simply put",
          "here is the thing", "here's the thing", "in truth", "the real story",
          "but here is what", "but here's what", "the bottom line",
          "let that sink in", "the point is")
_RATIO_RE = re.compile(r"\b(\d{1,3})\s+(?:in|out of)\s+(\d{1,3})\b", re.I)
_PICTO = ("in every", "out of every", "one in", "two in", "three in", "for every",
          "nine out of", "one out of")
_COMPOSITION = ("every dollar", "breaks down", "broken down", "made up of",
                "consists of", "divided into", "where the money", "split between",
                "for every dollar", "allocated", "carved up", "portioned")
# V1.7 cues — a defined term, a balance of forces, a before/after transformation
_DEFINE = ("is defined as", "by definition", "the term", "refers to",
           "means simply", "what is a", "what exactly is", "in other words",
           "put simply", "is a term for", "literally means")
_BALANCE = ("outweighed", "tipped the balance", "at the expense of", "traded off",
            "weighed against", "balance of power", "tipped in favour",
            "tipped in favor", "counterweight", "offset by",
            "came at the cost of", "outweighing")
_TRANSFORM = ("before and after", "then and now", "transformed", "a century apart",
              "used to be", "once was", "what it became", "transformation",
              "from ruin", "rebuilt", "would become", "decades later")
# V1.8 cues — a press frenzy, a heat spread across geography, a classified file
_PRESS = ("the headlines", "made headlines", "front pages", "the papers",
          "newspapers", "the press", "splashed across", "news broke",
          "headlines screamed", "every paper", "the story broke", "press erupted",
          "media frenzy", "column inches", "ran the story", "plastered across")
_HEAT = ("spread across", "swept across", "swept through", "engulfed", "took hold",
         "spread like", "fanned out", "broke out across", "gripped the",
         "epidemic", "contagion", "outbreak", "wildfire", "spread rapidly",
         "rippled across", "tore through", "spread throughout", "consumed the")
_SECRET = ("classified", "top secret", "redacted", "censored", "blacked out",
           "declassified", "the files showed", "eyes only", "for your eyes only",
           "kept secret", "covered up", "buried the", "sealed records",
           "confidential memo", "the memo", "the dossier", "leaked documents")
# V1.9 cues — a ranking/leaderboard, a proportional flow split, an era timeline
_RANK = ("ranked", "the top five", "the top ten", "top three", "number one",
         "the biggest", "the largest", "the richest", "the deadliest",
         "the most powerful", "in order of", "from smallest to", "leaderboard",
         "the rankings", "came in at number", "the worst", "the greatest")
_FLOW = ("flowed to", "flowed into", "splits into", "split into three",
         "channelled into", "channeled into", "fed into", "poured into",
         "directed toward", "for every dollar", "where it all went",
         "broke down as", "funnelled", "funneled", "divided among", "allocated to")
_ERA = ("the age of", "the era of", "ages of", "eras of", "three ages",
        "four eras", "golden age", "dark ages", "the epoch", "successive ages",
        "each era", "periods of", "across the centuries", "from one age to")
# V2.0 cues — a deadline countdown, a relationship web, a chorus of quotes
_COUNTDOWN = ("counting down", "the clock was ticking", "time was running out",
              "with hours to", "with days to", "hours to go", "days to go",
              "the deadline", "race against time", "ticking clock", "the countdown",
              "hours remained", "days remained", "running out of time",
              "before it was too late", "the final hours", "the final days")
_NETWORK = ("connected to", "a web of", "the network of", "linked to",
            "tied together", "knew each other", "a circle of", "connections between",
            "everyone was connected", "the players", "a tangle of", "a network of",
            "moved in the same", "all linked back to", "the web of")
_CHORUS = ("critics said", "the press called", "many called it", "they called it",
           "reactions poured", "everyone agreed", "one after another", "chorus of",
           "condemned it as", "praised it as", "the verdict was", "voices rose",
           "papers declared", "commentators", "the reviews")
# V2.1 cues — a region on the map, a causal chain, a qualitative gauge/level
_REGION = ("the region of", "the territory of", "the province of", "the zone",
           "annexed", "occupied the", "the borderlands", "the frontier", "ceded",
           "the heartland of", "across the region", "controlled the territory",
           "the disputed", "the contested region", "the area of")
_CAUSAL = ("led to", "which led to", "set off a chain", "triggered", "in turn",
           "caused", "which caused", "the domino", "one thing led to", "spiralled into",
           "cascaded", "knock-on", "as a result", "consequence was", "snowballed",
           "set in motion")
_GAUGE = ("the threat level", "the risk was", "rated as", "off the charts",
          "the severity", "the alert level", "danger level", "on a scale",
          "the assessment", "deemed", "classified as severe", "confidence was",
          "the odds", "the likelihood", "rated the")
# V2.2 cues — a dramatic subject reveal, a yes/no decision fork, a global arc
_REVEAL = ("behold", "meet the", "say hello to", "step into the light",
           "introducing", "the man who", "the woman who", "the one who",
           "this is the", "allow me to introduce", "the figure at the centre",
           "the figure at the center", "the key player", "take a good look",
           "here he is", "here she is", "the star of", "enter the")
_DECISION = ("a decision", "the decision", "two choices", "two options",
             "the choice was", "faced a choice", "a fork in the road",
             "at a crossroads", "the crossroads", "yes or no", "the dilemma",
             "had to decide", "make the call", "decision point",
             "the only question was", "to choose between", "either path")
_ARC = ("across the atlantic", "across the pacific", "across the ocean",
        "transatlantic", "transpacific", "undersea cable", "across the world",
        "halfway around the world", "thousands of miles", "from one continent",
        "spanned the globe", "around the globe", "the link between",
        "the connection across", "wired across", "the flight from",
        "the voyage from", "a world apart", "across the seas")
# V2.4 cues — mapped-conflict family (war front, advance, supply, diplomacy)
_WARFRONT = ("front line", "the front", "battlefield", "frontline", "two armies",
             "stalemate", "trench", "no man's land", "held the line", "the war between",
             "war broke out between", "at war with", "the battle of", "besieged",
             "occupied", "held the territory", "the border between", "contested")
_ADVANCEW = ("advanced", "advance on", "offensive", "pushed into", "stormed",
             "swept into", "marched on", "marched into", "drove toward",
             "broke through", "overran", "blitz", "spearhead", "thrust into",
             "rolled into", "the invasion", "invaded", "counteroffensive")
_SUPPLYW = ("supplied", "supplying", "arms shipments", "weapons to", "funnelled arms",
            "funneled arms", "shipments to", "backed by", "bankrolled", "financed by",
            "sent aid", "military aid", "supply line", "supply lines", "arming",
            "propped up", "lifeline of", "smuggled", "channelled funds", "channeled funds")
_DIPLOW = ("signed a treaty", "the treaty of", "an alliance", "allied with",
           "alliance with", "non-aggression", "pact with", "the pact", "an accord",
           "signed an agreement", "negotiated with", "diplomatic ties",
           "broke off relations", "severed ties", "declared war on", "betrayed",
           "double-crossed", "the deal collapsed", "joined forces with")
_DIPLO_BAD = ("broke off", "severed", "collapsed", "betray", "double-cross",
              "declared war", "fell apart", "torn up", "reneged", "broke down",
              "turned on", "stab")
# changed-border markers → modern Natural Earth borders are NOT historically exact
# (HISTORICAL_MAP_POLICY.md). On these, maps go reference/coastline-safe.
_CHANGED_BORDERS = ("empire", "ottoman", "soviet union", "the ussr", "colonial",
                    "colony", "colonies", "kingdom", "dynasty", "tsarist", "imperial",
                    "prussia", "austro-hungar", "byzantine", "caliphate", "the reich",
                    "yugoslavia", "czechoslovakia", "world war", "wwi", "ww1",
                    "great war", "napoleon", "medieval", "antiquity", "ancient",
                    "mesopotamia", "gaul", "persia", "partition", "the crusades")
# V2.6 cues — investigation / spy / crime / evidence family
_TESTIMONYW = ("recalled", "testified", "in his own words", "in her own words",
               "told investigators", "later admitted", "would later say",
               "said later", "would say", "remembered", "the witness", "a witness",
               "the defector", "the informant", "under oath", "gave evidence",
               "in the interrogation", "the deposition", "confided", "swore that")
_SUSPECTW = ("the suspect", "person of interest", "prime suspect", "the fugitive",
             "the accused", "double agent", "the mole", "code-named", "codenamed",
             "an alias", "went by the name", "also known as", "most wanted",
             "on the run", "a wanted man", "the prime suspect", "the perpetrator",
             "rap sheet", "his real name", "her real name")
_SITESW = ("the crime scene", "a safe house", "safe houses", "dead drop", "the hideout",
           "multiple sites", "across the city", "each location", "the addresses",
           "the stakeout", "under surveillance", "where it happened", "the locations",
           "two locations", "three locations", "several sites", "the meeting points")
_SIGHTW = ("line of sight", "line of fire", "the trajectory", "the angle of",
           "fired from", "the vantage point", "the sniper", "the bullet came",
           "ballistic", "from the window", "the shot came from", "the escape route",
           "fled toward", "approached from", "the rooftop", "the grassy knoll",
           "the firing position", "the kill shot", "the sightline")
_ROUTECMP = ("official account", "the official story", "official version",
             "actually took", "what really happened", "the discrepancy",
             "doesn't add up", "the missing hour", "the missing minutes",
             "unaccounted for", "claimed to have", "the alibi", "supposedly went",
             "two versions", "cctv showed", "the footage showed", "but in reality")
_DECLASSW = ("classified", "top secret", "declassified", "the redacted", "redacted",
             "censored", "eyes only", "the leaked memo", "secret memo",
             "the intercepted", "leaked documents", "the dossier revealed",
             "the file revealed", "the files showed", "covered up", "the cover-up",
             "the suppressed", "buried the report", "the sealed file")
# V2.8 cues — systems / mechanism / business / hybrid (reference-grounded)
_SYSTEMW = ("the system", "the architecture", "the pipeline", "passes through",
            "flows through", "routes through", "moves through", "travels through",
            "each component", "the components", "the stack", "front end",
            "back end", "the server", "the client", "the database", "the network",
            "end to end", "under the hood", "how the system", "the request",
            "data flows", "the data path", "step by step through",
            # business value-chain system language (system_planview_flow)
            "value chain", "every step", "each step", "every stage", "each stage",
            "control every", "from the wells", "the whole chain")

# per-video palette identity (seed-picked, niche-biased) so two same-script
# videos still look different
_NICHE_PAL = {
    "business": ["amber_gold", "parchment_sepia"],
    "crime": ["ember_red", "amber_gold"],
    "history": ["parchment_sepia", "amber_gold"],
    "biography": ["parchment_sepia", "amber_gold"],
    "geopolitics": ["cold_steel", "amber_gold"],
    "tech": ["cold_steel", "amber_gold"],
}


def video_palette(niche: str, seed: int) -> str:
    """Per-video palette identity. Routes through the reusable niche-aware
    weighted selector (so e.g. crime leans ember_red, never warm business gold,
    while keeping per-video variation); falls back to the legacy flat table if
    that module is unavailable."""
    try:
        from .. import niche_palette as _np
        return _np.select_palette(niche, seed)[0]
    except Exception:                                          # noqa: BLE001
        opts = _NICHE_PAL.get(niche, ["amber_gold"])
        return opts[seed % len(opts)]


def video_palette_reason(niche: str, seed: int) -> tuple[str, str]:
    """(palette, human-readable reason) for manifest/audit logging."""
    try:
        from .. import niche_palette as _np
        return _np.select_palette(niche, seed)
    except Exception:                                          # noqa: BLE001
        opts = _NICHE_PAL.get(niche, ["amber_gold"])
        p = opts[seed % len(opts)]
        return p, f"legacy table pick {p}"


@dataclass
class Decision:
    scene_index: int
    primitive: str | None
    inputs: dict = field(default_factory=dict)
    score: float = 0.0
    reason: str = ""
    variant: dict = field(default_factory=dict)   # V3.4 — visual-variant evidence


def _cues(text: str) -> dict:
    t = (text or "").lower()
    return {
        "money": bool(_MONEY_RE.search(text or "")),
        "pct": bool(_PCT_RE.search(text or "")),
        "bignum": bool(_NUM_RE.search(text or "")),
        "evidence": any(k in t for k in _EVIDENCE),
        "empire": any(k in t for k in _EMPIRE),
        "place": any(k in t for k in _PLACE),
        "year": bool(_YEAR_RE.search(text or "")),
        "quote": bool(_QUOTE_RE.search(text or "")) or any(k in t for k in _QSAY),
        "versus": bool(_VS_RE.search(text or "")),
        "artifact": any(k in t for k in _ARTIFACT),
        "growth": any(re.search(k, t) for k in _GROWTH),
        "route": any(re.search(k, t) for k in _ROUTE),
        "detail": any(k in t for k in _DETAIL),
        "share": any(k in t for k in _SHARE),
        "process": any(k in t for k in _PROCESS),
        "hierarchy": any(k in t for k in _HIERARCHY),
        "claim": any(k in t for k in _CLAIM),
        "picto": bool(_RATIO_RE.search(text or "")) or any(k in t for k in _PICTO),
        "composition": any(k in t for k in _COMPOSITION),
        "define": any(k in t for k in _DEFINE),
        "balance": any(k in t for k in _BALANCE),
        "transform": any(k in t for k in _TRANSFORM),
        "press": any(k in t for k in _PRESS),
        "heat": any(k in t for k in _HEAT),
        "secret": any(k in t for k in _SECRET),
        "rank": any(k in t for k in _RANK),
        "flow": any(k in t for k in _FLOW),
        "era": any(k in t for k in _ERA),
        "countdown": any(k in t for k in _COUNTDOWN),
        "network": any(k in t for k in _NETWORK),
        "chorus": any(k in t for k in _CHORUS),
        "region": any(k in t for k in _REGION),
        "causal": any(k in t for k in _CAUSAL),
        "gauge": any(k in t for k in _GAUGE),
        "reveal": any(k in t for k in _REVEAL),
        "decision": any(k in t for k in _DECISION),
        "arc": any(k in t for k in _ARC),
        "warfront": any(k in t for k in _WARFRONT),
        "advancew": any(k in t for k in _ADVANCEW),
        "supplyw": any(k in t for k in _SUPPLYW),
        "diplo": any(k in t for k in _DIPLOW),
        "testimony": any(k in t for k in _TESTIMONYW),
        "suspectw": any(k in t for k in _SUSPECTW),
        "sites": any(k in t for k in _SITESW),
        "sightline": any(k in t for k in _SIGHTW),
        "routecmp": any(k in t for k in _ROUTECMP),
        "declass": any(k in t for k in _DECLASSW),
        "systemw": any(k in t for k in _SYSTEMW),
    }


# graphic_kind (from the LLM's per-scene card tag) → primitive affinity. This
# is the strongest real-data signal — role labels vary too much to filter on.
_GK_AFFINITY = {
    "name_reveal": {"cinematic_portrait_hold": 2.0, "portrait_name_over_map": 1.4},
    "mugshot": {"cinematic_portrait_hold": 2.0},
    "mini_bio": {"cinematic_portrait_hold": 1.8, "portrait_name_over_map": 1.2},
    "bio": {"cinematic_portrait_hold": 1.8},
    # V1.2 — structural storytelling beats
    "pull_quote_portrait": {"pull_quote_portrait": 2.2, "cinematic_portrait_hold": 0.8},
    "pull_quote": {"pull_quote_portrait": 2.2},
    "long_quote": {"pull_quote_portrait": 2.0},
    "timeline": {"chronology_timeline": 2.2},
    "chronology": {"chronology_timeline": 2.2},
    "era_banner": {"chronology_timeline": 1.8},
    "chapter": {"chronology_timeline": 1.4},
    "comparison": {"comparison_split": 2.2},
    "versus": {"comparison_split": 2.2},
    "ratio": {"comparison_split": 1.8, "gold_number_callout": 0.8},
    "news_article": {"headline_document_reveal": 2.2},
    "breaking_news": {"headline_document_reveal": 1.8},
    "document": {"headline_document_reveal": 2.2},
    "press_release": {"headline_document_reveal": 1.8},
    "redacted": {"redacted_document": 2.2, "headline_document_reveal": 0.8},
    "statement": {"headline_document_reveal": 1.4, "kinetic_keyword": 1.2},
    "stat_insight": {"gold_number_callout": 2.2},
    "stat_dashboard": {"gold_number_callout": 1.6},
    "map_reveal": {"portrait_name_over_map": 1.8},
    "map_route": {"portrait_name_over_map": 1.6},
    "conspiracy_board": {"money_flow_empire": 1.8},
    "network_graph": {"money_flow_empire": 2.0},
    "org_tree": {"money_flow_empire": 1.6},
    "define_the_term": {"kinetic_keyword": 1.6},
    "quote_highlight": {"kinetic_keyword": 1.4},
    # V1.3 — Batch 2: evidence / data / location beats
    "evidence": {"framed_evidence_spotlight": 2.2, "headline_document_reveal": 0.8},
    "exhibit": {"framed_evidence_spotlight": 2.2},
    "artifact": {"framed_evidence_spotlight": 2.0},
    "framed_photo": {"framed_evidence_spotlight": 1.8},
    "bar_chart": {"statistic_bar_reveal": 2.2},
    "stat_bars": {"statistic_bar_reveal": 2.2},
    "breakdown": {"statistic_bar_reveal": 2.0},
    "data_viz": {"statistic_bar_reveal": 1.8},
    "location": {"location_establish_card": 2.2},
    "establish": {"location_establish_card": 2.0},
    "setting": {"location_establish_card": 1.6},
    "place_card": {"location_establish_card": 1.8},
    # V1.4 — Batch 3: growth / annotation / route beats
    "growth": {"growth_curve_chart": 2.2},
    "trend": {"growth_curve_chart": 2.2},
    "line_chart": {"growth_curve_chart": 2.0},
    "trajectory": {"growth_curve_chart": 2.0},
    "curve": {"growth_curve_chart": 1.8},
    "annotate": {"annotated_detail_callout": 2.2},
    "detail": {"annotated_detail_callout": 2.2},
    "highlight_detail": {"annotated_detail_callout": 2.0},
    "closeup": {"annotated_detail_callout": 1.6},
    "figure_locator": {"annotated_detail_callout": 1.6},
    "route": {"map_route_spread": 2.2},
    "journey": {"map_route_spread": 2.2},
    "expansion": {"map_route_spread": 2.0, "map_heat_spread": 0.8},
    "spread": {"map_heat_spread": 2.0, "map_route_spread": 1.0},
    "map_route": {"map_route_spread": 2.2},
    # V1.5 — Batch 4: proportion / process / hierarchy beats
    "share": {"proportion_ring": 2.2},
    "proportion": {"proportion_ring": 2.2},
    "percent": {"proportion_ring": 1.8, "gold_number_callout": 0.6},
    "dominance": {"proportion_ring": 2.0},
    "market_share": {"proportion_ring": 2.2},
    "hierarchy": {"org_hierarchy_tree": 2.2},
    "org_chart": {"org_hierarchy_tree": 2.2},
    "structure": {"org_hierarchy_tree": 1.8},
    "chain": {"org_hierarchy_tree": 1.6},
    # V1.6 — Batch 5: statement / pictograph / composition beats
    "statement": {"statement_card": 2.2},
    "thesis": {"statement_card": 2.2},
    "claim": {"statement_card": 2.0},
    "takeaway": {"statement_card": 1.8},
    "pictograph": {"pictograph_scale": 2.2},
    "ratio": {"pictograph_scale": 2.0},
    "figures": {"pictograph_scale": 1.8},
    "in_ten": {"pictograph_scale": 2.0},
    "composition": {"composition_stack": 2.2},
    "breakdown": {"composition_stack": 2.0, "statistic_bar_reveal": 0.6},
    "stacked": {"composition_stack": 2.0},
    "split": {"composition_stack": 1.6},
    # V1.7 — Batch 6: definition / balance / before-after beats
    "definition": {"definition_card": 2.2},
    "define": {"definition_card": 2.2},
    "term": {"definition_card": 1.8},
    "glossary": {"definition_card": 2.0},
    "balance": {"vs_balance_scale": 2.2},
    "tradeoff": {"vs_balance_scale": 2.0},
    "scales": {"vs_balance_scale": 2.0},
    "tension": {"vs_balance_scale": 1.6},
    "before_after": {"before_after_slider": 2.2},
    "transformation": {"before_after_slider": 2.2},
    "then_now": {"before_after_slider": 2.0},
    "makeover": {"before_after_slider": 1.8},
    # V1.8 — Batch 7: headlines / heat-spread / redacted beats
    "headlines": {"headline_montage": 2.2},
    "headline_montage": {"headline_montage": 2.2},
    "press": {"headline_montage": 2.2},
    "press_frenzy": {"headline_montage": 2.2},
    "media_frenzy": {"headline_montage": 2.0},
    "montage": {"headline_montage": 1.8},
    "scandal": {"headline_montage": 1.6},
    "coverage": {"headline_montage": 1.6},
    "heat_spread": {"map_heat_spread": 2.2},
    "heatmap": {"map_heat_spread": 2.0},
    "contagion": {"map_heat_spread": 2.2},
    "outbreak": {"map_heat_spread": 2.0},
    "epidemic": {"map_heat_spread": 2.0},
    "wildfire": {"map_heat_spread": 1.8},
    "classified": {"redacted_document": 2.2},
    "redacted_document": {"redacted_document": 2.2},
    "top_secret": {"redacted_document": 2.2},
    "secret": {"redacted_document": 1.8},
    "declassified": {"redacted_document": 2.0},
    "dossier": {"redacted_document": 1.8},
    "leak": {"redacted_document": 1.6},
    # V1.9 — Batch 8: ranked-list / sankey-flow / era-band beats
    "ranking": {"ranked_list_countdown": 2.2},
    "ranked_list_countdown": {"ranked_list_countdown": 2.2},
    "leaderboard": {"ranked_list_countdown": 2.2},
    "top_list": {"ranked_list_countdown": 2.0},
    "ranked": {"ranked_list_countdown": 2.0},
    "top_five": {"ranked_list_countdown": 2.0},
    "top_ten": {"ranked_list_countdown": 2.0},
    "countdown_list": {"ranked_list_countdown": 1.8},
    "sankey": {"sankey_flow": 2.2},
    "sankey_flow": {"sankey_flow": 2.2},
    "money_split": {"sankey_flow": 2.2},
    "allocation": {"sankey_flow": 2.0},
    "flow_breakdown": {"sankey_flow": 2.0},
    "where_it_goes": {"sankey_flow": 1.8},
    "eras": {"era_band_timeline": 2.2},
    "era_band_timeline": {"era_band_timeline": 2.2},
    "ages": {"era_band_timeline": 2.2},
    "era_bands": {"era_band_timeline": 2.2},
    "periods": {"era_band_timeline": 2.0},
    "epochs": {"era_band_timeline": 1.8},
    # V2.0 — Batch 9: countdown-clock / connection-web / quote-stream beats
    "countdown": {"countdown_clock": 2.2},
    "countdown_clock": {"countdown_clock": 2.2},
    "deadline": {"countdown_clock": 2.2},
    "timer": {"countdown_clock": 2.0},
    "ticking_clock": {"countdown_clock": 2.0},
    "urgency_clock": {"countdown_clock": 1.8},
    "connection_web": {"connection_web": 2.2},
    "network_web": {"connection_web": 2.2},
    "relationship_map": {"connection_web": 2.0},
    "conspiracy_web": {"connection_web": 2.2},
    "web_of_connections": {"connection_web": 2.0},
    "connections": {"connection_web": 1.8},
    "quote_stream": {"quote_stream": 2.2},
    "quotes": {"quote_stream": 2.2},
    "chorus": {"quote_stream": 2.0},
    "reactions": {"quote_stream": 2.0},
    "verdict_quotes": {"quote_stream": 1.8},
    # V2.1 — Batch 10: region-highlight / cause-effect / spectrum-meter beats
    "region": {"map_region_highlight": 2.2},
    "map_region": {"map_region_highlight": 2.2},
    "map_region_highlight": {"map_region_highlight": 2.2},
    "territory": {"map_region_highlight": 2.0},
    "region_highlight": {"map_region_highlight": 2.0},
    "region_map": {"map_region_highlight": 1.8},
    "cause_effect": {"cause_effect_chain": 2.2},
    "cause_effect_chain": {"cause_effect_chain": 2.2},
    "causation": {"cause_effect_chain": 2.0},
    "cause_chain": {"cause_effect_chain": 2.0},
    "domino": {"cause_effect_chain": 2.0},
    "chain_reaction": {"cause_effect_chain": 1.8},
    "gauge": {"spectrum_meter": 2.2},
    "meter": {"spectrum_meter": 2.2},
    "spectrum_meter": {"spectrum_meter": 2.2},
    "threat_level": {"spectrum_meter": 2.2},
    "severity": {"spectrum_meter": 2.0},
    "rating": {"spectrum_meter": 1.8},
    # V2.2 — Batch 11: spotlight-reveal / decision-fork / world-arc beats
    "spotlight": {"spotlight_object_hold": 2.2},
    "spotlight_object_hold": {"spotlight_object_hold": 2.2},
    "reveal": {"spotlight_object_hold": 2.0},
    "subject_reveal": {"spotlight_object_hold": 2.0},
    "behold": {"spotlight_object_hold": 1.8},
    "unveil": {"spotlight_object_hold": 1.8},
    "decision": {"flowchart_decision": 2.2},
    "flowchart": {"flowchart_decision": 2.2},
    "flowchart_decision": {"flowchart_decision": 2.2},
    "decision_tree": {"flowchart_decision": 2.0},
    "branch": {"flowchart_decision": 2.0},
    "choice": {"flowchart_decision": 1.8},
    "fork": {"flowchart_decision": 1.8},
    "yes_no": {"flowchart_decision": 1.8},
    "world_arc": {"world_map_arc": 2.2},
    "world_map_arc": {"world_map_arc": 2.2},
    "arc": {"world_map_arc": 2.0},
    "global_link": {"world_map_arc": 2.0},
    "transatlantic": {"world_map_arc": 2.0},
    "great_circle": {"world_map_arc": 1.8},
    "globe_arc": {"world_map_arc": 1.8},
    # V2.4 — MG-expansion Batch 1A: cinematic-map family
    "status_banner": {"map_status_banner": 2.2},
    "date_banner": {"map_status_banner": 2.0},
    "war_phase": {"map_status_banner": 1.8, "parchment_war_map": 0.8},
    "war_map": {"parchment_war_map": 2.2},
    "battle_map": {"parchment_war_map": 2.2},
    "front_line": {"parchment_war_map": 2.0},
    "front": {"parchment_war_map": 1.8},
    "supply_route": {"supply_route_dashes": 2.2},
    "supply_lines": {"supply_route_dashes": 2.0},
    "influence_map": {"supply_route_dashes": 1.8},
    "advance": {"territory_advance_arrows": 2.2},
    "offensive": {"territory_advance_arrows": 2.2},
    "invasion": {"territory_advance_arrows": 2.0, "parchment_war_map": 0.8},
    "troop_movement": {"territory_advance_arrows": 2.0},
    "treaty": {"diplomatic_link": 2.2},
    "alliance": {"diplomatic_link": 2.2},
    "pact": {"diplomatic_link": 2.0},
    "diplomacy": {"diplomatic_link": 1.8},
    "rupture": {"diplomatic_link": 1.8},
    # V3.5 — icon/logo link beam: product/tech/business "these two connect" beats
    "integration": {"logo_link_beam": 2.2},
    "partnership": {"logo_link_beam": 2.0},
    "collaboration": {"logo_link_beam": 1.8},
    "interface": {"logo_link_beam": 1.6},
    "plugin": {"logo_link_beam": 1.8},
    "actor_badge": {"map_badge_node": 2.2},
    "leader_pin": {"map_badge_node": 2.0},
    "flag_pin": {"map_badge_node": 2.0},
    "faction_badge": {"map_badge_node": 1.8},
    "velocity_route": {"velocity_route_map": 2.2},
    "shipping_route": {"velocity_route_map": 2.2},
    "trade_route": {"velocity_route_map": 2.0, "supply_route_dashes": 0.8},
    "trajectory_map": {"velocity_route_map": 1.8},
    # V2.6 — investigation / spy / crime / evidence family
    "testimony": {"witness_testimony_card": 2.4},
    "witness": {"witness_testimony_card": 2.4},
    "statement": {"witness_testimony_card": 1.6, "statement_card": 0.8},
    "suspect": {"suspect_profile_card": 2.4},
    "mugshot": {"suspect_profile_card": 2.2},
    "dossier": {"suspect_profile_card": 1.8, "classified_stamp_reveal": 1.0},
    "person_of_interest": {"suspect_profile_card": 2.2},
    "evidence_map": {"investigation_location_map": 2.4},
    "location_map": {"investigation_location_map": 1.6, "location_establish_card": 1.0},
    "scene_map": {"investigation_location_map": 2.0},
    "sightline": {"sightline_trajectory": 2.4},
    "trajectory": {"sightline_trajectory": 2.2},
    "line_of_fire": {"sightline_trajectory": 2.4},
    "line_of_sight": {"sightline_trajectory": 2.2},
    "route_comparison": {"route_comparison": 2.4},
    "two_routes": {"route_comparison": 2.2},
    "discrepancy": {"route_comparison": 1.8},
    "connection_board": {"evidence_connection_board": 2.4},
    "evidence_board": {"evidence_connection_board": 2.4},
    "conspiracy_map": {"evidence_connection_board": 2.0, "connection_web": 0.8},
    "classified": {"classified_stamp_reveal": 2.4},
    "declassified": {"classified_stamp_reveal": 2.4},
    "top_secret": {"classified_stamp_reveal": 2.2},
    "redacted_reveal": {"classified_stamp_reveal": 2.0, "redacted_document": 0.8},
    # V2.8 — systems / mechanism / business / hybrid (grounded)
    "system": {"system_planview_flow": 2.2},
    "architecture": {"system_planview_flow": 2.4},
    "plan_view": {"system_planview_flow": 2.4},
    "system_diagram": {"system_planview_flow": 2.2},
    "pipeline": {"system_planview_flow": 2.0},
    "data_path": {"system_planview_flow": 2.0},
    "request_path": {"system_planview_flow": 2.2},
    "packet_route": {"packet_path_trace": 2.4},
    "packet_trace": {"packet_path_trace": 2.4},
    "data_flow": {"packet_path_trace": 2.0, "system_planview_flow": 0.6},
    "kill_chain": {"exploit_chain": 2.4},
    "attack_chain": {"exploit_chain": 2.2},
    "vulnerability": {"exploit_chain": 1.8},
    "measurement": {"measurement_callout": 2.4},
    "dimension": {"measurement_callout": 2.0},
    "scale_compare": {"silhouette_scale_compare": 2.4},
    "size_compare": {"silhouette_scale_compare": 2.2},
    "acquisition": {"acquisition_timeline": 2.4},
    "acquisitions": {"acquisition_timeline": 2.4},
    "takeover": {"acquisition_timeline": 2.0},
    "supply_chain": {"supply_chain_network": 2.4},
    "value_chain": {"supply_chain_network": 2.2},
    "fact_overlay": {"footage_fact_overlay": 2.2},
    "stat_overlay": {"footage_fact_overlay": 2.0},
    "object_callout": {"footage_object_callout": 2.4},
    "route_trace": {"footage_route_trace": 2.4},
    "exploit_chain": {"exploit_chain": 2.4},
}


def known_graphic_kinds() -> set:
    """Every graphic_kind tag that maps to a premium primitive (a director
    trigger). The pipeline uses this to protect a deliberately-tagged editorial
    beat from being clobbered by the weak-footage AI-explainer — so a scene the
    script tagged `classified` / `press` / `contagion` / etc. keeps its kind and
    renders its premium card instead of a generic fallback when footage is weak.
    Self-maintaining: every batch's new affinity keys are protected automatically.
    """
    return set(_GK_AFFINITY)


def _best_for_scene(scene: dict, niche: str) -> tuple[str | None, float, dict]:
    """Highest-relevance eligible primitive for one scene + the inputs it needs.
    Eligibility = niche + intensity + satisfiable inputs (NOT role — real scene
    roles don't match the primitive vocabulary). Relevance = narration cues +
    graphic_kind affinity + role bonus."""
    assets = scene.get("assets", {}) or {}
    role = scene.get("role")
    gk = (scene.get("graphic_kind") or "").lower()
    intensity = int(scene.get("intensity", 3))
    cues = _cues(scene.get("narration", ""))
    have = set(assets)
    # Forensic v2 (vs Vidlore, beat b13 "Cornell ... 96% reduction"): an in-PROSE
    # statistical proof — a %/number backed by a source or finding word — must
    # route to an ANIMATED stat primitive instead of falling through to a static
    # paragraph card held for the whole beat. Cue-driven, so it fires WITHOUT an
    # explicit bar_chart/stat_bars graphic_kind tag (which prose proof never has).
    _proof_stat = bool((cues["pct"] or cues["bignum"]) and (
        cues["evidence"] or any(w in (scene.get("narration", "") or "").lower()
        for w in ("study", "confirmed", "university", "research", "researcher",
                  "scientists", "found that", "measured", "reduction", "percent",
                  "survey", "experiment", "clinical", "data show"))))
    # synthesize inputs the director can derive from narration
    derived = {}
    mm = _MONEY_RE.search(scene.get("narration", "") or "")
    if mm and "value" not in assets:
        raw = re.sub(r"[^\d.]", "", mm.group(0))
        if raw:
            derived["value"] = float(raw)
            derived["prefix"] = "$"
            _mag = mm.group(0).lower()
            if "billion" in _mag or "bn" in _mag:
                derived["suffix"] = " BILLION"
            elif "million" in _mag or re.search(r"\d\s?m\b", _mag):
                derived["suffix"] = " MILLION"
            have.add("value")
    elif cues["pct"] and "value" not in assets:
        # Prefer the scene's EMPHASIS when it is itself a percentage. A narration
        # line frequently carries MORE THAN ONE percent ("not 40, not 60, but
        # 96%" / "rose from 60% to 96%"); a naive first-match grabbed the wrong
        # one (this is the 96%-shown-as-64% class). The emphasis marks the ONE
        # figure the beat lands on, so it wins; otherwise fall back to the first
        # percentage in the narration.
        _emph = scene.get("emphasis")
        _emph = _emph if isinstance(_emph, str) else ""
        pm = _PCT_RE.search(_emph) or _PCT_RE.search(scene.get("narration", "") or "")
        if pm:
            derived["value"] = float(re.sub(r"[^\d.]", "", pm.group(0)))
            derived["suffix"] = "%"
            have.add("value")
    # V1.5 — derive a share % so proportion_ring is eligible on percent scenes
    if "share" not in assets:
        _pm2 = _PCT_RE.search(scene.get("narration", "") or "")
        if _pm2:
            try:
                derived["share"] = float(re.sub(r"[^\d.]", "", _pm2.group(0)))
                have.add("share")
            except ValueError:
                pass
    # V1.6 — derive an "N in M" ratio so pictograph_scale is eligible
    if "count" not in assets:
        _rm = _RATIO_RE.search(scene.get("narration", "") or "")
        if _rm:
            try:
                _cn, _tl = int(_rm.group(1)), int(_rm.group(2))
                if 0 < _cn <= _tl <= 100:
                    derived["count"], derived["total"] = _cn, _tl
                    have.add("count")
            except ValueError:
                pass
    # V1.2 — derive time anchors + quotations from narration (assets win)
    if "events" not in assets:
        _yrs = _YEAR_RE.findall(scene.get("narration", "") or "")
        _seen = set()
        _yrs = [y for y in _yrs if not (y in _seen or _seen.add(y))]
        if _yrs:
            derived["events"] = [[y, ""] for y in _yrs[:4]]
            have.add("events")
    if "quote" not in assets:
        _qm = _QUOTE_RE.search(scene.get("narration", "") or "")
        if _qm:
            derived["quote"] = _qm.group(1).strip()
            have.add("quote")
    # V2 weak-keyword resilience (2026-06-03): derive a RELATIONSHIP WEB from the
    # named entities in the narration so connection_web is eligible even with
    # empty keywords AND no graphic hint. GATED to avoid garbage webs: only when
    # the scene is genuinely about a network/relationship (network language) AND
    # ≥3 distinct MULTI-WORD proper-noun entities are present (single capitalised
    # tokens are too noisy — sentence starts). Conservative by design: a weak
    # signal stays footage, never a meaningless 2-node card.
    _ntext = scene.get("narration", "") or ""
    _net_lang = any(w in _ntext.lower() for w in (
        "network", "web of", "connected", "connection", "circle of", "ring of",
        "ties to", "linked", "relationship", "associates", "inner circle",
        "between", "among", "reported to", "answered to", "worked with"))
    if "nodes" not in assets and _net_lang:
        _ents = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", _ntext)
        _stop = {"The", "A", "An", "In", "On", "At", "But", "And", "For", "From",
                 "With", "When", "What", "Who", "Why", "How", "This", "That",
                 "Then", "Now", "One", "Two", "By", "To", "Of", "As", "If", "It",
                 "He", "She", "They", "His", "Her"}
        _clean = []
        for _e in _ents:
            _w = _e.split()
            if _w and _w[0] in _stop:
                _w = _w[1:]
            _e2 = " ".join(_w)
            if len(_e2) > 3 and (_e2.split()[0] not in _stop if _e2 else False):
                _clean.append(_e2)
        _uniq = list(dict.fromkeys(_clean))
        if len(_uniq) >= 3:
            derived["nodes"] = _uniq[:6]
            have.add("nodes")
    # list fallback (evidence_connection_board / connection_web): the ROLE NOUNS
    # in an enumeration under network language ("the recruiter, the courier, the
    # forger…"). Common-noun heads are a CRIME/SPY idiom — gate to those niches so
    # it never mines abstract nouns out of a thesis/summary beat ("the rift between
    # …, the world's powers, the fuse" → War/Region/Rift/World/Gulf was a false
    # positive that stole geopolitics statement beats). Other niches keep the
    # proper-noun-only derivation above.
    if ("nodes" not in assets and "nodes" not in derived and _net_lang
            and str(niche).lower() in ("spy", "crime", "true_crime",
                                       "intelligence", "intel")):
        try:
            from .maps import geo as _geo3
        except Exception:                                          # noqa: BLE001
            _geo3 = None
        _adj = {"full", "same", "very", "real", "only", "whole", "entire", "single",
                "first", "last", "next", "other", "such", "main", "key", "top",
                "best", "worst", "true", "final", "great", "good", "bad", "new", "old"}
        _cands = re.findall(r"\b(?:the|a|an)\s+([a-z]{3,18})\b", _ntext, flags=re.I)
        _items = []
        for _c in _cands:
            _cl = _c.lower()
            if _cl in _adj or re.fullmatch(r"were|was|is|are|had|have|been", _cl):
                continue
            if _geo3 and _geo3.resolve(_c):       # places belong to the location map
                continue
            _items.append(_cl[:1].upper() + _cl[1:])
        _items = list(dict.fromkeys(_items))
        if len(_items) >= 3:
            derived["nodes"] = _items[:5]
            have.add("nodes")
    # V2.4 — cinematic-map family derivation (weak-keyword resilience). Each is
    # CONSERVATIVE: needs both the right LANGUAGE and concrete proper-noun entities
    # (or a year + a clean editorial phrase), else the scene stays footage. The
    # primary input is gated (so the primitive only becomes eligible when real
    # content exists); secondary inputs are forwarded below and stripped by the
    # signature-filter in dispatch for any non-map winner.
    def _proper_entities(txt):
        # Allow an optional middle INITIAL ("D.") inside a name so
        # "John D. Rockefeller" reads as ONE entity, not "John" + "Rockefeller".
        # A truncated "John" misses the real archival portrait and forces an AI
        # still — fabrication we avoid when a real image exists. Strictly more
        # accurate for every consumer (names/people/sides); war/place pairs with
        # no initials are unaffected.
        ents = re.findall(
            r"\b([A-Z][a-z]+(?:\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+){0,2})\b", txt or "")
        stop = {"The", "A", "An", "In", "On", "At", "But", "And", "For", "From",
                "With", "When", "What", "Who", "Why", "How", "This", "That",
                "Then", "Now", "One", "Two", "By", "To", "Of", "As", "If", "It",
                "He", "She", "They", "His", "Her", "We", "Our", "Their"}
        out = []
        for _e in ents:
            _w = _e.split()
            if _w and _w[0] in stop:
                _w = _w[1:]
            _e2 = " ".join(_w)
            if len(_e2) >= 3:
                out.append(_e2)
        return list(dict.fromkeys(out))
    _n2 = scene.get("narration", "") or ""
    # parchment_war_map: two belligerents + war/front language
    if "side_a" not in assets and cues["warfront"]:
        _e = _proper_entities(_n2)
        if len(_e) >= 2:
            derived["side_a"], derived["side_b"] = _e[0], _e[1]
            have.add("side_a")
    # territory_advance_arrows: advance language → an offensive label + theatre region
    if cues["advancew"]:
        _yr = _YEAR_RE.search(_n2)
        derived.setdefault("label", ("OFFENSIVE " + _yr.group(0)) if _yr else "ADVANCE")
        if _yr:
            derived.setdefault("year", _yr.group(0))
        _ea = _proper_entities(_n2)
        if _ea:
            # two belligerent COUNTRIES → origin (attacker) + target (defender);
            # one → target only; none → last entity. Real region focus either way.
            try:
                from .maps import geo as _geo
                _co = [e for e in _ea if _geo.country_entry(e)]
            except Exception:                                      # noqa: BLE001
                _co = []
            # RC4: forward the RESOLVED origin/target COUNTRY NAMES (e.g. Iran /
            # Iraq) so the renderer — which draws geo labels and tints each side
            # when the region resolves — gets real labels instead of an abstract
            # theatre. When nothing resolves we deliberately leave `region` unset
            # unless the last entity itself resolves as a place, so a non-place
            # token (a person) can never leak into the geo-label path; the renderer
            # then uses its existing safe synthetic fallback. No invented borders.
            if len(_co) >= 2:
                derived.setdefault("origin_region", _co[0])
                derived.setdefault("target_region", _co[1])
                derived.setdefault("region", _co[1])
            elif _co:
                derived.setdefault("target_region", _co[0])
                derived.setdefault("region", _co[0])
            else:
                try:
                    _last_ok = bool(_geo.resolve(_ea[-1]) or _geo.country_entry(_ea[-1]))
                except Exception:                                  # noqa: BLE001
                    _last_ok = False
                if _last_ok:
                    derived.setdefault("region", _ea[-1])
    # supply_route_dashes: supply language + ≥2 entities (source → dests)
    if "source" not in assets and cues["supplyw"]:
        _e = _proper_entities(_n2)
        if len(_e) >= 2:
            derived["source"], derived["dests"] = _e[0], _e[1:4]
            have.add("source")
    # diplomatic_link: treaty/alliance language + two actors (+ outcome)
    if not {"a", "b"} & set(assets) and cues["diplo"]:
        _e = _proper_entities(_n2)
        if len(_e) >= 2:
            derived["a"], derived["b"] = _e[0], _e[1]
            derived["outcome"] = ("reject" if any(k in _n2.lower() for k in _DIPLO_BAD)
                                  else "accept")
            have.add("a"); have.add("b")
    # historical-border safety: when borders materially changed, route maps to
    # the reference/coastline-safe treatment instead of exact modern borders.
    _yrs_hb = _YEAR_RE.findall(_n2)
    _old_year = any(int(y) < 1945 for y in _yrs_hb) if _yrs_hb else False
    if any(k in _n2.lower() for k in _CHANGED_BORDERS) or _old_year:
        derived.setdefault("reference", True)
        derived.setdefault("borders", False)
    # map_status_banner: a year + a clean editorial phrase (the scene emphasis)
    if "event" not in assets:
        _yr = _YEAR_RE.search(_n2)
        _emph = scene.get("emphasis")
        _emph = _emph.strip() if isinstance(_emph, str) else ""
        if _yr and 4 <= len(_emph) <= 48:
            derived["event"], derived["year"] = _emph, _yr.group(0)
            have.add("event")
    # V2.6 — investigation / spy / crime / evidence family derivation. Each is
    # CONSERVATIVE: needs the right LANGUAGE + concrete entities/lines, else the
    # scene stays footage. Primary inputs are gated into `have` (so the primitive
    # only becomes eligible with real content); secondaries forward below and are
    # stripped by dispatch's signature-filter for any non-matching winner.
    # witness_testimony_card: a quotation (already derived) + an attributed speaker.
    if derived.get("quote") and cues["testimony"]:
        _e = _proper_entities(_n2)
        if _e and "name" not in assets:
            derived.setdefault("name", _e[0])
            derived.setdefault("witness", True)        # flag: a testimony beat
    # suspect_profile_card: suspect/identity language + a named subject.
    if "name" not in assets and cues["suspectw"]:
        _e = _proper_entities(_n2)
        if _e:
            derived["name"] = _e[0]
            have.add("name")
            _al = re.search(r'(?:alias|known as|went by(?: the name)?|code-?named)\s+'
                            r'["“]?([A-Z][A-Za-z .’-]{2,28})', _n2)
            if _al:
                derived.setdefault("alias", _al.group(1).strip(' "”'))
    # investigation_location_map: site language + ≥2 resolvable real places.
    if "pins" not in assets and (cues["sites"] or (cues["place"] and cues["network"])):
        try:
            from .maps import geo as _geo2
            _pl = [e for e in _proper_entities(_n2) if _geo2.resolve(e)]
        except Exception:                                          # noqa: BLE001
            _pl = []
        _pl = list(dict.fromkeys(_pl))
        if len(_pl) >= 2:
            derived["pins"] = [{"place": p, "label": ""} for p in _pl[:3]]
            have.add("pins")
    # sightline_trajectory: sightline/ballistic language + a from→to pair. Prefer
    # proper-noun entities; else lift the "from the X … to the Y" nouns (forensic
    # narration usually names the position by common noun — rooftop, window, nest).
    if not {"origin", "target"} & set(assets) and cues["sightline"]:
        _e = _proper_entities(_n2)
        def _title(s): return " ".join(w.capitalize() for w in s.split())
        _from = re.search(r"\bfrom (?:the |a |an |his |her |its )?"
                          r"([a-z][a-z'’ -]{2,22}?)(?:[,.;:]| to | below| above|$)", _n2.lower())
        _to = re.search(r"\b(?:to|toward|towards|at|into) (?:the |a |an |his |her )?"
                        r"([a-z][a-z'’ -]{2,22}?)(?:[,.;:]| below| above|$)", _n2.lower())
        if len(_e) >= 2:
            derived["origin"], derived["target"] = _e[0], _e[1]
            have.add("origin"); have.add("target")
        elif _from or _to:
            derived["origin"] = _e[0] if _e else (_title(_from.group(1).strip())
                                                  if _from else "Origin")
            derived["target"] = _title(_to.group(1).strip()) if _to else "Target"
            have.add("origin"); have.add("target")
        elif _e:
            derived["origin"], derived["target"] = _e[0], "Target"
            have.add("origin"); have.add("target")
    # route_comparison: discrepancy language → two contrasted route labels.
    if not {"route_a", "route_b"} & set(assets) and cues["routecmp"]:
        derived["route_a"] = {"label": "Official account"}
        derived["route_b"] = {"label": "What the evidence shows", "highlight": True}
        have.add("route_a"); have.add("route_b")
        _e = _proper_entities(_n2)
        if len(_e) >= 2:
            derived.setdefault("start", _e[0]); derived.setdefault("end", _e[1])
    # evidence_connection_board shares the derived `nodes` (ring-chain layout when
    # no explicit centre); scoring below routes crime/spy network beats to it.
    # classified_stamp_reveal: STRONG classified language → the suppressed line.
    if "reveal" not in assets and cues["declass"]:
        _emphc = scene.get("emphasis")
        _emphc = _emphc.strip() if isinstance(_emphc, str) else ""
        _line = _emphc if 6 <= len(_emphc) <= 84 else ""
        if not _line:
            _sent = re.split(r"(?<=[.!?])\s+", _n2.strip())
            _line = next((s for s in _sent if 14 <= len(s) <= 110), "")
        if not _line and len(_n2.strip()) >= 14:   # one long sentence → truncate to a word
            _line = _n2.strip()[:84].rsplit(" ", 1)[0].rstrip(",;:") + "…"
        if _line:
            derived["reveal"] = _line
            have.add("reveal")
    # V2.8 — system_planview_flow: system language + ≥2 named components → regions.
    # Niche-gated to tech/science/business (the same gate that fixed the board
    # false-positive — abstract noun lists elsewhere stay footage). Excludes the
    # whole-system nouns themselves (system/architecture/pipeline/stack).
    if ("regions" not in assets and cues["systemw"]
            and str(niche).lower() in ("tech", "technology", "science", "business",
                                       "education_explainer", "explainer")):
        _adjx = {"same", "whole", "entire", "very", "real", "main", "key", "first",
                 "next", "last", "final", "only", "single", "new", "old", "other"}
        _whole = {"system", "architecture", "pipeline", "stack", "component",
                  "components", "process", "way", "thing", "part", "parts"}
        _regs = []
        for _r in re.findall(r"\b(?:the|a|an|each)\s+([a-z]{3,18})\b", _n2, flags=re.I):
            _rl = _r.lower()
            if _rl in _adjx or _rl in _whole:
                continue
            _regs.append(_rl[:1].upper() + _rl[1:])
        _regs = list(dict.fromkeys(_regs))
        if len(_regs) >= 2:
            derived["regions"] = [{"label": r} for r in _regs[:5]]
            have.add("regions")
    # V2.8 — Section-C derivation (conservative, niche-gated; structured-asset
    # primitives silhouette_scale_compare/footage_route_trace stay graphic_kind-only).
    _nl2 = _n2.lower()

    def _role_list(txt):
        _out = []
        for _r in re.findall(r"\b(?:the|a|an|each)\s+([a-z]{3,18})\b", txt, flags=re.I):
            _rl = _r.lower()
            if _rl in ("same", "whole", "first", "next", "last", "only", "very",
                       "real", "main", "other", "single"):
                continue
            _out.append(_rl[:1].upper() + _rl[1:])
        return list(dict.fromkeys(_out))
    # measurement_callout: a real dimension stated in measurement language
    if "value" not in assets:
        _mz = re.search(r"\b\d[\d,.]*\s?(?:nanomet\w+|nm|microns?|micromet\w+|"
                        r"mm|cm|km|metres|meters|miles|mile|ft|feet|"
                        r"kg|tonnes?|tons?|lbs?|gw|mw|kw|mph|bytes?|kb|mb|gb|tb|bits?|m)\b",
                        _n2, re.I)
        if _mz and any(k in _nl2 for k in ("measur", "wide", "long", "tall", "deep",
                       "distance", "span", "diameter", "length", "height", "weigh",
                       "size", "across", "high", "thick", "stretches", "stands",
                       "just", "only", "tiny", "memory", "buffer")):
            derived["value"] = _mz.group(0).strip()
            have.add("value")
    # footage_fact_overlay: a TIGHT number+magnitude+noun fact (restrained default)
    if "fact" not in assets and str(niche).lower() in (
            "tech", "technology", "science", "business", "geopolitics", "history",
            "crime", "spy", "true_crime"):
        _fm = re.search(r"\b(\d[\d,.]*)\s*(%|percent|million|billion|trillion|thousand)?"
                        r"\s+([a-z]{3,14})\b", _n2, re.I)
        if _fm and (_fm.group(2) or _fm.group(1)):
            _num = _fm.group(1) + ((" " + _fm.group(2)) if _fm.group(2) else "")
            _ft = (_num + " " + _fm.group(3)).upper()
            if 6 <= len(_ft) <= 40 and _fm.group(3).lower() not in (
                    "and", "the", "but", "for", "with", "that", "year", "years",
                    "different", "teams", "decade"):
                derived["fact"] = _ft
                have.add("fact")
    # exploit_chain: explicit attack-STAGE language (not just "attacker")
    if "stages" not in assets and any(k in _nl2 for k in ("breach", "exploit",
            "kill chain", "payload", "intrusion", "reconnaissance", "escalation",
            "escalat", "exfiltrat", "the attack unfolded", "attack began")):
        _st = _role_list(_n2)
        if len(_st) >= 2:
            derived["stages"] = [{"label": s} for s in _st[:5]]
            have.add("stages")
    # supply_chain_network: chain language + a stage list
    elif "stages" not in assets and any(k in _nl2 for k in ("supply chain",
            "value chain", "raw material", "refiner", "refining", "factory",
            "distribution", "from raw", "manufactur", "the assembly", "shipped to",
            "crude", "oil field", "oilfield", "rail line", "flowed from",
            "pipeline", "the wells")):
        _st = _role_list(_n2)
        if len(_st) >= 2:
            derived["stages"] = [{"label": s} for s in _st[:5]]
            have.add("stages")
    # acquisition_timeline: "X acquired/bought A, B and C"
    if not {"parent", "targets"} & set(assets) and any(k in _nl2 for k in
            ("acquired", "acquisition", "bought out", "took over", "absorbed",
             "merged with", "snapped up", "swallowed")):
        _ents = _proper_entities(_n2)
        if len(_ents) >= 2:
            derived["parent"] = _ents[0]
            derived["targets"] = [{"label": e} for e in _ents[1:5]]
            have.add("parent"); have.add("targets")
    # packet_path_trace: network-path language + an ordered node list
    if "hops" not in assets and any(k in _nl2 for k in ("packet", "request travels",
            "data travels", "through the network", "routed through", "hops",
            "across the network", "server to", "the signal travels")):
        _hp = _proper_entities(_n2) or _role_list(_n2)
        if len(_hp) >= 2:
            derived["hops"] = [{"label": h} for h in _hp[:5]]
            have.add("hops")
    # footage_object_callout: "notice/look at the X"
    if "label" not in assets and re.search(
            r"\b(?:notice|look at|look closely|spot the|see the|highlighted|"
            r"circled|marked|the small|the tiny|that unassuming)\b", _nl2):
        _ob = _role_list(_n2)
        if _ob:
            derived["label"] = _ob[0]
            have.add("label")
    # V3.0 — Biography / Character derivation. Niche-gated to person-story niches
    # so life-arc graphics are EARNED punctuation, never fired on ordinary
    # explainers. Name / year / wealth / verdict are mined from prose; the
    # structured roster + milestone spine require ≥2 named people / ≥2 distinct
    # years (else a script author supplies them explicitly).
    # RC4 (2026-06-05): "history" dropped from this gate. The wealth/value
    # derivation + biography boost below are business/biography-story devices; on
    # generic history docs (e.g. an Iran-Iraq War beat) they fired ungrounded
    # money/relationship cards. registry.eligible() already blocks the off-theme
    # business primitives for history/war/geopolitics; dropping "history" here
    # stops the prose-mined value/roster from ever being derived for them.
    if str(niche).lower() in ("biography", "crime", "true_crime",
                              "spy", "business"):
        _role_l = str(role).lower()
        _person_cue = any(k in _nl2 for k in ("was born", "born in", "grew up",
            "the young", "the man who", "the woman who", "the boy", "as a child",
            "the son of", "the daughter of", "his life", "her life", "his story",
            "her story"))
        _yrs_b = re.findall(r"\b(1[5-9]\d\d|20[0-4]\d)\b", _n2)
        # portrait_legend_reveal — a named subject on an opening/legend beat
        if ("name" not in assets and "name" not in derived
                and _role_l in ("hook", "intro", "reveal", "legend", "context")
                and _person_cue):
            _ppl = _proper_entities(_n2)
            if _ppl:
                derived["name"] = _ppl[0]; have.add("name")
        # era_stamp_overlay — a real year orienting an archival/biography scene
        if ("year" not in assets and "year" not in derived and _yrs_b and any(
                k in _nl2 for k in ("in 1", "in 2", "by 1", "by 2", "that year",
                "the year", "century", "decade", "back in"))):
            derived["year"] = _yrs_b[0]; have.add("year")
        # wealth_arc_counter — a wealth/value figure on a money beat.
        # NUMERIC GROUNDING (RC4): only ever emit `value` when an ACTUAL number is
        # present in the reviewed narration. We deliberately do NOT invent a figure
        # from the money KEYWORDS alone — if the prose says "wealth"/"fortune" with
        # no digit, no `value` is set, the primitive stays ineligible, and the
        # renderer never fabricates an axis number that would imply false data.
        if ("value" not in assets and "value" not in derived and any(
                k in _nl2 for k in ("net worth", "fortune", "richest", "wealth",
                "billion", "million", "his worth", "her worth", "$"))):
            _mv = re.search(r"\$?\s?\d[\d,.]*\s?(?:billion|million|trillion|thousand)?",
                            _n2, re.I)
            if _mv and any(c.isdigit() for c in _mv.group(0)):
                derived["value"] = _mv.group(0).strip(); have.add("value")
        # verdict_duality_card — a finale legacy/judgment line
        if ("verdict" not in assets and "verdict" not in derived
                and _role_l in ("thesis", "legacy", "resolution", "payoff",
                                "verdict", "climax")
                and any(k in _nl2 for k in ("legacy", "remembered", "history will",
                    "for better or worse", "admired", "critici", "genius", "monster",
                    "hero", "villain", "both a", "judged", "his name", "her name"))):
            _v = " ".join(_n2.split())[:90]
            if 14 <= len(_v) <= 90:
                derived["verdict"] = _v; have.add("verdict")
        # relationship_roster — ≥2 named PEOPLE with relationship language.
        # PEOPLE-ONLY (RC4): `_proper_entities` makes no person/place distinction,
        # so on a war/geopolitics beat it mixed a person (Saddam) with places
        # (Khuzestan, Gulf, Iran). Drop any entity that resolves as a place via the
        # same geo backbone used for territory_advance_arrows above; never feed a
        # place in as a "person". If <2 real people remain it was mostly places —
        # skip the roster entirely and leave the beat to the map / footage.
        if ("people" not in assets and "people" not in derived
                and any(k in _nl2 for k in ("rival", "partner", "mentor", "ally",
                    "allies", "brother", "father", "family", "married", "friend",
                    "enemy", "alongside", "together with", "co-founder", "cofounder"))):
            _ppl2 = _proper_entities(_n2)
            try:
                from .maps import geo as _geo_r
                _ppl2 = [p for p in _ppl2
                         if not (_geo_r.country_entry(p) or _geo_r.resolve(p))]
            except Exception:                                      # noqa: BLE001
                pass
            if len(_ppl2) >= 2:
                derived["people"] = [{"name": p, "role": ""} for p in _ppl2[:5]]
                have.add("people")
        # life_milestone_spine — ≥2 distinct years on a life-arc beat
        if ("milestones" not in assets and "milestones" not in derived
                and len(set(_yrs_b)) >= 2
                and any(k in _nl2 for k in ("over the years", "throughout his",
                    "throughout her", "by the time", "his life", "her life",
                    "career", "decades", "across his", "across her"))):
            _ys = list(dict.fromkeys(_yrs_b))[:6]
            derived["milestones"] = [{"year": y, "label": ""} for y in _ys]
            have.add("milestones")
    # V3.1 — Science / Engineering Explainer derivation. Niche-gated so the
    # cutaway is EARNED on explainer beats only. The cross-section needs labelled
    # internal parts; auto-derive ONLY from an explicit component LIST
    # ("consists of / made up of / contains A, B and C") so we never invent
    # internals — otherwise a script author supplies `parts` in graphic_body.
    if str(niche).lower() in ("science", "tech", "technology", "engineering"):
        _xs_cue = any(k in _nl2 for k in ("cross-section", "cross section",
            "cutaway", "inside the", "how it works", "at the heart", "the core",
            "consists of", "made up of", "made of", "comprises", "contains",
            "components", "anatomy of", "internal"))
        if "parts" not in assets and "parts" not in derived and _xs_cue:
            _xm = re.search(r"(?:consists of|made up of|made of|comprises|"
                            r"contains)\s+(.+?)(?:\.|$)", _n2, re.I)
            _xi = []
            if _xm:
                for _seg in re.split(r",|\band\b|;", _xm.group(1)):
                    _w = " ".join(_seg.split()).strip(" .")
                    # drop a leading article so chips read "electron gun" not
                    # "an electron gun" (content unchanged, just cleaner labels)
                    _w = re.sub(r"^(?:a|an|the)\s+", "", _w, flags=re.I)
                    if 2 <= len(_w) <= 32 and _w.lower() not in (
                            "etc", "more", "others", "so on", "it"):
                        _xi.append(_w)
            if len(_xi) >= 2:
                derived["parts"] = [{"label": _w} for _w in _xi[:4]]
                have.add("parts")
    gka = _GK_AFFINITY.get(gk, {})
    cands = registry.eligible(niche=niche, intensity=intensity, have_inputs=have)
    # RC4: "history" dropped here too (mirrors the derivation gate above) so the
    # biography scoring boosts never run on plain history docs.
    _bio_n = str(niche).lower() in ("biography", "crime",
                                    "true_crime", "spy", "business")
    _sci_n = str(niche).lower() in ("science", "tech", "technology", "engineering")
    best, best_s, best_in = None, 0.0, {}
    for e in cands:
        s = 0.6
        pid = e["id"]
        s += gka.get(pid, 0.0)                       # graphic_kind affinity
        if pid == "gold_number_callout" and (cues["money"] or cues["pct"] or cues["bignum"]):
            s += 2.0
        if pid == "money_flow_empire" and cues["empire"]:
            s += 2.4
        if pid == "headline_document_reveal" and cues["evidence"]:
            s += 2.0
        if pid == "portrait_name_over_map" and cues["place"]:
            s += 1.6
        if pid == "kinetic_keyword" and intensity >= 4:
            s += 1.2
        if pid == "chronology_timeline" and cues["year"]:
            s += 1.4
        if pid == "pull_quote_portrait" and cues["quote"]:
            s += 1.8
        if pid == "comparison_split" and cues["versus"]:
            s += 1.6
        if pid == "framed_evidence_spotlight" and cues["artifact"]:
            s += 1.8
        if pid == "location_establish_card" and cues["place"]:
            s += 1.4
        if pid == "statistic_bar_reveal" and cues["bignum"]:
            s += 0.8
        if _proof_stat and pid in ("gold_number_callout", "statistic_bar_reveal"):
            s += 1.8                              # prose stat-proof → animated stat card
        if pid == "growth_curve_chart" and cues["growth"]:
            s += 1.8
        if pid == "map_route_spread" and cues["route"]:
            s += 1.8
        if pid == "annotated_detail_callout" and cues["detail"]:
            s += 1.8
        # V3.0 — biography primitives win when their (niche-gated) input is
        # derived; gated to person-story niches so they never beat the data/
        # number cards on tech/science/geopolitics explainers.
        if _bio_n:
            if pid == "portrait_legend_reveal" and "name" in have:
                s += 2.4
            if pid == "wealth_arc_counter" and "value" in have:
                s += 2.3
            if pid == "life_milestone_spine" and "milestones" in have:
                s += 2.2
            if pid == "relationship_roster" and "people" in have:
                s += 2.1
            if pid == "verdict_duality_card" and "verdict" in have:
                s += 2.2
            if pid == "act_chapter_card" and "title" in have:
                s += 1.6
            if pid == "era_stamp_overlay" and "year" in have:
                s += 1.6
        # V3.1 — the labeled cross-section wins on explainer beats when its
        # (niche-gated) parts are derived/supplied; gated to science/tech so it
        # never beats the data/number cards on other niches.
        if _sci_n and pid == "labeled_cross_section" and "parts" in have:
            s += 2.3
        if pid == "proportion_ring" and cues["share"]:
            s += 1.8
        if pid == "org_hierarchy_tree" and cues["hierarchy"]:
            s += 1.6
        if pid == "statement_card" and cues["claim"]:
            s += 1.6
        if pid == "pictograph_scale" and cues["picto"]:
            s += 1.8
        if pid == "composition_stack" and cues["composition"]:
            s += 1.8
        if pid == "definition_card" and cues["define"]:
            s += 1.6
        if pid == "vs_balance_scale" and cues["balance"]:
            s += 1.8
        if pid == "before_after_slider" and cues["transform"]:
            s += 1.8
        if pid == "headline_montage" and cues["press"]:
            s += 1.8
        if pid == "map_heat_spread" and cues["heat"]:
            s += 1.8
        if pid == "redacted_document" and cues["secret"]:
            s += 1.8
        if pid == "ranked_list_countdown" and cues["rank"]:
            s += 1.8
        if pid == "sankey_flow" and cues["flow"]:
            s += 1.8
        if pid == "era_band_timeline" and cues["era"]:
            s += 1.8
        if pid == "countdown_clock" and cues["countdown"]:
            s += 1.8
        if pid == "connection_web" and cues["network"]:
            s += 1.8
        if pid == "quote_stream" and cues["chorus"]:
            s += 1.8
        if pid == "map_region_highlight" and cues["region"]:
            s += 1.8
        if pid == "cause_effect_chain" and cues["causal"]:
            s += 1.8
        if pid == "spectrum_meter" and cues["gauge"]:
            s += 1.8
        if pid == "spotlight_object_hold" and cues["reveal"]:
            s += 1.8
        if pid == "flowchart_decision" and cues["decision"]:
            s += 1.8
        if pid == "world_map_arc" and cues["arc"]:
            s += 1.8
        if pid == "parchment_war_map" and cues["warfront"]:
            s += 1.8
        if pid == "territory_advance_arrows" and cues["advancew"]:
            s += 1.8
        if pid == "supply_route_dashes" and cues["supplyw"]:
            s += 1.8
        if pid == "diplomatic_link" and cues["diplo"]:
            s += 1.8
        if pid == "map_status_banner" and cues["year"] and cues["place"]:
            s += 1.4
        if pid == "velocity_route_map" and cues["route"]:
            s += 1.2
        if pid == "map_badge_node" and cues["place"]:
            s += 1.2
        # V2.6 — investigation / spy / crime / evidence family scoring
        if pid == "witness_testimony_card" and cues["testimony"] and derived.get("quote"):
            s += 2.0
        if pid == "suspect_profile_card" and cues["suspectw"]:
            s += 2.0
        if pid == "investigation_location_map" and cues["sites"]:
            s += 1.8
        if pid == "sightline_trajectory" and cues["sightline"]:
            s += 2.0
        if pid == "route_comparison" and cues["routecmp"]:
            s += 1.8
        if (pid == "evidence_connection_board" and (cues["network"] or _net_lang)
                and str(niche).lower() in ("spy", "crime", "true_crime",
                                           "intelligence", "intel", "geopolitics")):
            s += 2.0                                  # crime/spy → board over web
        if pid == "classified_stamp_reveal" and cues["declass"]:
            s += 2.0
        if pid == "system_planview_flow" and cues["systemw"]:
            s += 1.8
        if pid == "packet_path_trace" and gk in ("packet_route", "packet_trace", "data_flow"):
            s += 1.6
        if pid == "exploit_chain" and (gk in ("exploit_chain", "kill_chain",
                "attack_chain", "vulnerability") or "stages" in derived):
            s += 1.4
        if pid == "measurement_callout" and "value" in derived:
            s += 1.4
        if pid == "silhouette_scale_compare" and gk in ("scale_compare", "size_compare"):
            s += 1.6
        if pid == "acquisition_timeline" and "parent" in derived:
            s += 1.6
        if pid == "supply_chain_network" and (gk in ("supply_chain", "value_chain")
                or ("stages" in derived and any(k in _nl2 for k in ("supply", "chain", "refinery")))):
            s += 1.4
        if pid == "footage_fact_overlay" and "fact" in derived:
            s += 0.7
        if pid == "footage_object_callout" and ("label" in derived or gk == "object_callout"):
            s += 1.0
        if pid == "footage_route_trace" and gk == "route_trace":
            s += 1.6
        if role and role in e["roles"]:
            s += 1.0                                  # role bonus (not a filter)
        s += 0.2 * (intensity - 3)                  # stronger scenes lean graphic
        # assemble inputs (assets win, else derived)
        ins = {k: assets.get(k, derived.get(k)) for k in e["required_inputs"]}
        for k in ("name", "prefix", "suffix", "label", "sub", "source",
                  "highlight", "context", "palette_name", "map_image",
                  "portrait_path", "side", "title", "year", "leftval",
                  "rightval", "leftsub", "rightsub", "vs", "decimals",
                  "caption", "tag", "image_path", "bars", "coords", "bg_image",
                  "points", "stops", "focus", "place", "map_image",
                  "share", "steps", "root", "children", "center_sub",
                  "text", "emphasis", "count", "total", "segments",
                  "term", "definition", "pos", "before_label", "after_label",
                  "headlines", "hotspots", "reveal", "stamp",
                  "to", "unit", "links", "bands", "readout",
                  "subject", "kicker", "question", "yes", "no", "yes_label",
                  "no_label", "chosen", "from_place", "to_place", "from_pos",
                  "to_pos"):
            if k in assets:
                ins[k] = assets[k]
        if "prefix" in derived and "prefix" not in ins:
            ins["prefix"] = derived["prefix"]
        if "suffix" in derived and "suffix" not in ins:
            ins["suffix"] = derived["suffix"]
        if "total" in derived and "total" not in ins:        # pictograph_scale
            ins["total"] = derived["total"]
        # V2.4 — forward derived secondary map inputs (primary flows via the
        # required_inputs comprehension; dispatch's signature-filter strips any
        # key a non-map winner doesn't declare, so this is always safe)
        for _dk in ("side_b", "dests", "outcome", "year", "label", "region",
                    "reference", "borders", "origin_region", "target_region", "event",
                    # V2.6 — investigation family secondaries
                    "name", "alias", "start", "end"):
            if _dk in derived and _dk not in ins:
                ins[_dk] = derived[_dk]
        if s > best_s:
            best, best_s, best_in = pid, s, ins
    return best, best_s, best_in


# ── intelligent niche-aware density (2026-06-03) ────────────────────────────
# Bounded editorial targets, NOT a blind fixed count. For a 2-4 min doc (~12-15
# scenes) this lands ~3-5 premium MG beats (min ~2-3, upper 5-7) WHEN the script
# offers that many genuine opportunities — the score floor + per-scene scoring
# still gate quality, and the budget never forces a card that doesn't score.
# Explainer/narrative-dense niches (science, tech, spy/intel, geopolitics) lean
# slightly higher; footage-first niches lower. The old flat 0.16 produced
# round(13×0.16)=2 — the real reason the 7-niche sweep rendered only 1-2.
_NICHE_DENSITY = {
    "tech": 0.36, "technology": 0.36, "science": 0.36, "education_explainer": 0.36,
    "explainer": 0.36, "health_longevity": 0.34, "health": 0.34,
    "geopolitics": 0.34, "spy": 0.34, "intel": 0.34, "intelligence": 0.34,
    "business": 0.33, "finance": 0.33, "economics": 0.33,
    "crime": 0.33, "true_crime": 0.33, "history": 0.32, "biography": 0.30,
    "agriculture_history": 0.24, "agriculture": 0.24,
}
_MG_MIN_TARGET = 2      # when valid opportunities exist
_MG_MAX_TARGET = 7      # explainer-heavy ceiling; footage-first otherwise


def _density_for(niche) -> float:
    return _NICHE_DENSITY.get(str(niche or "").strip().lower(), 0.30)


def plan(scenes: list[dict], *, niche: str = "business", recipe: dict | None = None,
         seed: int = 0, density: float | None = None, min_gap_scenes: int = 1,
         score_floor: float = 1.8, mg_min: int = _MG_MIN_TARGET,
         mg_max: int = _MG_MAX_TARGET) -> list[Decision]:
    """Return a Decision per scene. `density` ≈ fraction of scenes that get a
    graphic (footage stays the foundation). When None, a niche-aware default is
    used. The budget is a BOUNDED editorial target — it never forces a card that
    doesn't clear the score floor, so weak/opportunity-less scripts still stay
    sparse. Deterministic for a given seed."""
    n = len(scenes)
    if n == 0:
        return []
    if density is None:
        density = _density_for(niche)
    rng_base = (seed * 2654435761) & 0xFFFFFFFF
    # 1) score every scene's best candidate
    scored = []
    for sc in scenes:
        pid, s, ins = _best_for_scene(sc, niche)
        # tiny deterministic jitter so ties resolve differently per video
        jit = (((rng_base ^ (sc.get("index", 0) * 40503)) & 0xFFFF) / 0xFFFF - 0.5) * 0.3
        scored.append((sc, pid, s + (jit if pid else 0), ins))
    # 2) global budget from density — bounded editorial target. mg_min only
    # raises the cap, never forces cards (the score floor still gates), so an
    # opportunity-poor script stays footage-only; mg_max keeps it footage-first.
    budget = max(1, round(n * density))
    if recipe and isinstance(recipe.get("graphics_density"), (int, float)):
        budget = max(1, round(n * float(recipe["graphics_density"])))
    budget = max(mg_min, min(mg_max, budget))
    # 3) forward pass with caps / cooldowns / anti-repeat / spacing
    used_count: dict[str, int] = {}
    last_scene_used: dict[str, int] = {}
    placed = 0
    last_graphic_idx = -10
    last_family = None
    last_pid = None
    # rank scenes by score to know which clear the bar within budget
    order = sorted(range(n), key=lambda i: -scored[i][2])
    chosen_threshold = scored[order[min(budget, n) - 1]][2] if budget <= n else score_floor
    thr = max(score_floor, chosen_threshold)
    vpal = video_palette(niche, seed)
    # V3.4 — one seeded visual-variant selector per video (deterministic + anti-repeat)
    from . import variants as _variants
    _vsel = _variants.VariantSelector(
        project_seed=seed, niche=niche,
        channel_dna=str((recipe or {}).get("channel_dna", "")))
    decisions = [Decision(scenes[i].get("index", i), None, {}, 0.0, "footage") for i in range(n)]
    for i in range(n):
        sc, pid, s, ins = scored[i]
        idx = sc.get("index", i)
        if pid is None or s < thr or placed >= budget:
            continue
        e = registry.get(pid)
        if used_count.get(pid, 0) >= e["per_video_cap"]:
            continue
        # cooldown (approx scenes from seconds at ~scene length)
        cd_scenes = max(min_gap_scenes, int(e["repeat_cooldown_s"] / 12))
        if pid in last_scene_used and (i - last_scene_used[pid]) < cd_scenes:
            continue
        if (i - last_graphic_idx) < (min_gap_scenes + 1):
            continue
        # incompatible-adjacent / same-family checks ONLY apply near a recent
        # graphic (graphics are sparse, so `last_pid` lingers across footage).
        near = (i - last_graphic_idx) <= 2
        if near and last_pid and registry.incompatible(pid, last_pid):
            continue
        if near and e["family"] == last_family:
            continue
        if not e["required_inputs"].issubset({k for k, v in ins.items() if v is not None}):
            continue
        # per-video palette + deterministic per-scene seed (videos look distinct)
        ins.setdefault("palette_name", vpal)
        ins.setdefault("seed", (seed * 131 + idx) & 0xFFFF)
        # V3.4 — seeded anti-repeat visual-variant selection. Sets inputs["layout"]
        # to a tasteful skin id; this is a VISUAL choice only and never touches a
        # value / label / route / coordinate / date / scene order (those are fixed
        # upstream). A primitive whose render ignores `layout` simply uses its
        # default; the manifest still records the choice as evidence.
        _ev = _vsel.select(pid, e.get("layout_variants", []),
                           family=e.get("family", ""), scene_index=idx)
        if _ev.get("visual_variant_id") and "layout" not in ins:
            ins["layout"] = _ev["visual_variant_id"]
        decisions[i] = Decision(idx, pid, ins, round(s, 2),
                                f"{pid} (cues→score {s:.1f})", variant=_ev)
        used_count[pid] = used_count.get(pid, 0) + 1
        last_scene_used[pid] = i
        last_graphic_idx = i
        last_family = e["family"]
        last_pid = pid
        placed += 1
    return decisions


def plan_summary(decisions: list[Decision]) -> dict:
    used = [d for d in decisions if d.primitive]
    by = {}
    for d in used:
        by[d.primitive] = by.get(d.primitive, 0) + 1
    return {"scenes": len(decisions), "graphics": len(used),
            "density": round(len(used) / max(1, len(decisions)), 3),
            "by_primitive": by,
            "at_scenes": [(d.scene_index, d.primitive) for d in used]}
