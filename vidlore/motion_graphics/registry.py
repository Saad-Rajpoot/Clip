"""Motion-graphics primitive REGISTRY.

Single source of truth the director queries: each entry binds a primitive's
SPEC (roles, niches, intensity/duration ranges, easing, audio cue, cap,
cooldown, fallback, cost, layout/palette variants) to its render() callable,
plus selection metadata (required inputs, incompatible combos, QA checks).

Adding a primitive = import it here and (optionally) add REQUIRED_INPUTS. No
other wiring needed; the director picks it up automatically.
"""
from __future__ import annotations

from .annotations import annotated_detail_callout
from .charts import (composition_stack, growth_curve_chart, money_flow_empire,
                     pictograph_scale, proportion_ring, ranked_list_countdown,
                     sankey_flow, statistic_bar_reveal)
from .clocks import countdown_clock
from .comparison import comparison_split
from .diagrams import (cause_effect_chain, connection_web, flowchart_decision,
                       logo_link_beam, org_hierarchy_tree)
# RC5.1 — the cheap four-box "STEP 1 → STEP 2" numbered template was fully
# removed from production (renderer deleted): it read as cheap/AI-generated.
# Its use-cases fall back to the director's normal choices (footage-first /
# cause_effect_chain / chronology_timeline / restrained map).
from .documents import headline_document_reveal, redacted_document
from .evidence import framed_evidence_spotlight
from .investigation import (classified_stamp_reveal, evidence_connection_board,
                            investigation_location_map, route_comparison,
                            sightline_trajectory, suspect_profile_card,
                            witness_testimony_card)
from .maps import (diplomatic_link, location_establish_card, map_badge_node,
                   map_heat_spread, map_region_highlight, map_route_spread,
                   map_status_banner, parchment_war_map, portrait_name_over_map,
                   supply_route_dashes, territory_advance_arrows,
                   velocity_route_map, world_map_arc)
from .media import headline_montage
from .meters import spectrum_meter
from .numbers import gold_number_callout
from .portraits import cinematic_portrait_hold
from .quotes import pull_quote_portrait, quote_stream
from .reveals import before_after_slider, spotlight_object_hold
from .scales import vs_balance_scale
from .statements import definition_card, statement_card
from .business import acquisition_timeline, supply_chain_network
from .hybrid import (footage_fact_overlay, footage_object_callout,
                     footage_route_trace)
from .mechanism import measurement_callout, silhouette_scale_compare
from .systems import exploit_chain, packet_path_trace, system_planview_flow
from .timelines import chronology_timeline, era_band_timeline
from .typography import kinetic_keyword
from .biography import (portrait_legend_reveal, relationship_roster,
                       wealth_arc_counter, life_milestone_spine,
                       verdict_duality_card, act_chapter_card, era_stamp_overlay)
from .science import labeled_cross_section

_MODULES = [
    gold_number_callout, cinematic_portrait_hold, headline_document_reveal,
    portrait_name_over_map, kinetic_keyword, money_flow_empire,
    # V1.2 — structural storytelling beats (timeline / quote / comparison)
    chronology_timeline, pull_quote_portrait, comparison_split,
    # V1.3 — Batch 2: evidence / data / location beats
    framed_evidence_spotlight, statistic_bar_reveal, location_establish_card,
    # V1.4 — Batch 3: growth / annotation / route beats
    growth_curve_chart, annotated_detail_callout, map_route_spread,
    # V1.5 — Batch 4: proportion / process / hierarchy beats
    # RC5.1 — cheap four-box step template removed from production.
    proportion_ring, org_hierarchy_tree,
    # V1.6 — Batch 5: statement / pictograph / composition beats
    statement_card, pictograph_scale, composition_stack,
    # V1.7 — Batch 6: definition / balance / before-after beats
    definition_card, vs_balance_scale, before_after_slider,
    # V1.8 — Batch 7: headlines / heat-spread / redacted beats
    headline_montage, map_heat_spread, redacted_document,
    # V1.9 — Batch 8: ranked-list / sankey-flow / era-band beats
    ranked_list_countdown, sankey_flow, era_band_timeline,
    # V2.0 — Batch 9: countdown-clock / connection-web / quote-stream beats
    countdown_clock, connection_web, quote_stream,
    # V2.1 — Batch 10: region-highlight / cause-effect / spectrum-meter beats
    map_region_highlight, cause_effect_chain, spectrum_meter,
    # V2.2 — Batch 11: spotlight-reveal / decision-fork / world-arc beats
    spotlight_object_hold, flowchart_decision, world_map_arc,
    # V2.4 — MG-expansion Batch 1A: cinematic-map family (maps weakest-family fill)
    map_status_banner, parchment_war_map, supply_route_dashes,
    territory_advance_arrows, diplomatic_link, map_badge_node, velocity_route_map,
    # V2.6 — MG-expansion Batch 1B: investigation / spy / crime / evidence family
    witness_testimony_card, suspect_profile_card, investigation_location_map,
    sightline_trajectory, route_comparison, evidence_connection_board,
    classified_stamp_reveal,
    # V2.8 — MG-expansion Batch 1C: systems / mechanism / business / hybrid
    # (reference-grounded; provenance in MG_PRIMITIVE_PROVENANCE_REGISTRY.md)
    system_planview_flow, packet_path_trace, exploit_chain,
    measurement_callout, silhouette_scale_compare,
    acquisition_timeline, supply_chain_network,
    footage_fact_overlay, footage_object_callout, footage_route_trace,
    # V3.0 — Biography / Character family (reference-grounded; proof
    # frames in biography_family/frame_sequences/)
    portrait_legend_reveal, relationship_roster, wealth_arc_counter,
    life_milestone_spine, verdict_duality_card, act_chapter_card,
    era_stamp_overlay,
    # V3.1 — Science / Engineering Explainer family (reference-grounded; the
    # ONE genuine gap = labeled cross-section; see science_explainer_family/)
    labeled_cross_section,
    # V3.5 — icon/logo link beam (two icons fly in + travelling beam + connect);
    # the logo-first, map-free cousin of diplomatic_link (integration/partnership)
    logo_link_beam,
]

# what each primitive needs to render (the director must be able to supply these
# from the scene/script/assets, else the primitive is ineligible)
REQUIRED_INPUTS = {
    "gold_number_callout":     {"value"},                      # a numeric stat
    "cinematic_portrait_hold": {"portrait_path"},              # a person image
    "headline_document_reveal": {"headline"},                  # an evidence line
    "portrait_name_over_map":  {"name", "place"},              # person + geography
    "kinetic_keyword":         {"keyword"},                    # one concept word
    "money_flow_empire":       {"center", "branches"},         # entity + ≥2 branches
    "chronology_timeline":     {"events"},                     # ≥1 year/event
    "pull_quote_portrait":     {"quote"},                      # a quotation (name opt)
    "comparison_split":        {"left", "right"},              # two contrasted sides
    "framed_evidence_spotlight": {"caption"},                  # what the exhibit is
    "statistic_bar_reveal":    {"bars"},                       # ≥1 [label,value]
    "location_establish_card": {"place"},                      # a place name
    "growth_curve_chart":      {"points"},                     # ≥2 [label,value]
    "annotated_detail_callout": {"label"},                     # what to point at
    "map_route_spread":        {"stops"},                      # ≥2 waypoints
    "proportion_ring":         {"share"},                      # a 0-100 percentage
    # RC5.1 — four-box step template intentionally omitted (removed).
    "org_hierarchy_tree":      {"root", "children"},           # root + ≥1 child
    "statement_card":          {"text"},                       # the claim itself
    "pictograph_scale":        {"count"},                      # N lit (total opt.)
    "composition_stack":       {"segments"},                   # ≥1 [label,value]
    "definition_card":         {"term", "definition"},         # word + its meaning
    "vs_balance_scale":        {"left", "right"},              # two weighed forces
    "before_after_slider":     {"after_label"},                # names the after-state
    "headline_montage":        {"headlines"},                  # ≥1 headline
    "map_heat_spread":         {"hotspots"},                   # ≥1 hotspot
    "redacted_document":       {"reveal"},                     # the one legible line
    "ranked_list_countdown":   {"items"},                      # ≥2 ranked items
    "sankey_flow":             {"branches"},                   # ≥2 flow branches
    "era_band_timeline":       {"eras"},                       # ≥2 era bands
    "countdown_clock":         {"value"},                      # a number to count down
    "connection_web":          {"nodes"},                      # ≥2 named nodes
    "quote_stream":            {"quotes"},                     # ≥2 quotations
    "map_region_highlight":    {"region"},                     # a region/place name
    "cause_effect_chain":      {"steps"},                      # ≥2 causal steps
    "spectrum_meter":          {"value"},                      # a 0-100 level
    "spotlight_object_hold":   {"subject"},                    # the word/name to reveal
    "flowchart_decision":      {"question"},                   # the yes/no question
    "world_map_arc":           {"from_place", "to_place"},     # origin + destination
    # V2.4 — Batch 1A: cinematic-map family
    "map_status_banner":       {"event"},                      # a dated map moment
    "parchment_war_map":       {"side_a"},                     # ≥1 named territory
    "supply_route_dashes":     {"source"},                     # a supply/influence origin
    "territory_advance_arrows": set(),                         # role/affinity-gated only
    "diplomatic_link":         {"a", "b"},                     # two actors in a tie
    "map_badge_node":          {"place"},                      # an actor's location
    "velocity_route_map":      {"stops"},                      # ≥2 journey stops
    # V2.6 — Batch 1B: investigation / spy / crime / evidence family
    "witness_testimony_card":  {"quote"},                      # a person's statement
    "suspect_profile_card":    {"name"},                       # the subject's name
    "investigation_location_map": {"pins"},                    # ≥1 pinned site
    "sightline_trajectory":    {"origin", "target"},           # a from→to line
    "route_comparison":        {"route_a", "route_b"},         # two contrasted routes
    "evidence_connection_board": {"nodes"},                    # ≥2 evidence nodes
    "classified_stamp_reveal": {"reveal"},                     # the suppressed line
    # V2.8 — Batch 1C: systems / mechanism / business / hybrid
    "system_planview_flow":    {"regions"},                    # ≥2 system zones
    "packet_path_trace":       {"hops"},                       # ≥3 network hops
    "exploit_chain":           {"stages"},                     # ≥3 attack stages
    "measurement_callout":     {"value"},                      # a measured span
    "silhouette_scale_compare": {"items"},                     # 2 scale-true items
    "acquisition_timeline":    {"parent", "targets"},          # acquirer + targets
    "supply_chain_network":    {"stages"},                     # ≥3 chain stages
    "footage_fact_overlay":    {"fact"},                       # a screen-locked fact
    "footage_object_callout":  {"label"},                      # the object pointed at
    "footage_route_trace":     {"points"},                     # ≥2 route waypoints
    # V3.0 — Biography / Character family
    "portrait_legend_reveal":  {"name"},                       # the subject's name
    "relationship_roster":     {"people"},                     # ≥2 named people
    "wealth_arc_counter":      {"value"},                      # a metric to count
    "life_milestone_spine":    {"milestones"},                 # ≥2 dated milestones
    "verdict_duality_card":    {"verdict"},                    # the finale thesis line
    "act_chapter_card":        {"title"},                      # the act/chapter title
    "era_stamp_overlay":       {"year"},                       # the era/year
    # V3.1 — science / engineering explainer
    "labeled_cross_section":   {"parts"},                      # ≥1 labelled part
}

# pairs that should not fire back-to-back (visually too similar)
INCOMPATIBLE_ADJACENT = {
    ("cinematic_portrait_hold", "portrait_name_over_map"),  # both portrait-led
    ("pull_quote_portrait", "cinematic_portrait_hold"),     # both portrait-led
    ("pull_quote_portrait", "portrait_name_over_map"),      # both portrait-led
    ("location_establish_card", "portrait_name_over_map"),  # both map/place-led
    ("framed_evidence_spotlight", "headline_document_reveal"),  # both document-led
    ("statistic_bar_reveal", "gold_number_callout"),        # both number-led
    # V3.0 — biography family adjacency guards
    ("portrait_legend_reveal", "cinematic_portrait_hold"),  # both portrait-led
    ("portrait_legend_reveal", "pull_quote_portrait"),      # both portrait-led
    ("portrait_legend_reveal", "portrait_name_over_map"),   # both portrait-led
    ("relationship_roster", "connection_web"),              # both relationship maps
    ("relationship_roster", "org_hierarchy_tree"),          # both relationship maps
    ("relationship_roster", "evidence_connection_board"),   # both node clusters
    ("wealth_arc_counter", "gold_number_callout"),          # both number-led
    ("wealth_arc_counter", "growth_curve_chart"),           # both metric arcs
    ("wealth_arc_counter", "statistic_bar_reveal"),         # both data charts
    ("life_milestone_spine", "chronology_timeline"),        # both timelines
    ("life_milestone_spine", "mini_timeline"),              # both timelines
    ("verdict_duality_card", "statement_card"),             # both full-screen theses
    ("verdict_duality_card", "definition_card"),            # both full-screen theses
    ("act_chapter_card", "statement_card"),                 # both big-title cards
    ("act_chapter_card", "location_establish_card"),        # both footage-riding establishers
    ("era_stamp_overlay", "location_establish_card"),       # both place/era establishers
    ("growth_curve_chart", "statistic_bar_reveal"),         # both data charts
    ("growth_curve_chart", "money_flow_empire"),            # both data charts
    ("annotated_detail_callout", "framed_evidence_spotlight"),  # both examine an image
    ("annotated_detail_callout", "cinematic_portrait_hold"),    # both photo-led
    ("map_route_spread", "location_establish_card"),        # both map-led
    ("map_route_spread", "portrait_name_over_map"),         # both map-led
    ("proportion_ring", "statistic_bar_reveal"),            # both data charts
    ("proportion_ring", "gold_number_callout"),             # both single-figure
    # RC5.1 — four-box step template adjacency pairs removed (primitive gone).
    ("org_hierarchy_tree", "money_flow_empire"),            # both structure graphs
    ("statement_card", "kinetic_keyword"),                  # both bold typography
    ("statement_card", "pull_quote_portrait"),              # both a held line
    ("pictograph_scale", "proportion_ring"),                # both a single ratio
    ("composition_stack", "proportion_ring"),               # both parts-of-a-whole
    ("composition_stack", "statistic_bar_reveal"),          # both bar charts
    ("definition_card", "kinetic_keyword"),                 # both lead on a word
    ("vs_balance_scale", "comparison_split"),               # both A-vs-B
    ("before_after_slider", "annotated_detail_callout"),    # both lead on an image
    ("before_after_slider", "comparison_split"),            # both split the frame
    ("headline_montage", "headline_document_reveal"),       # both press / newsprint
    ("map_heat_spread", "map_route_spread"),                # both animated maps
    ("map_heat_spread", "location_establish_card"),         # both maps
    ("redacted_document", "headline_document_reveal"),      # both document pages
    ("redacted_document", "framed_evidence_spotlight"),     # both document/evidence
    ("ranked_list_countdown", "statistic_bar_reveal"),      # both bar charts
    ("ranked_list_countdown", "composition_stack"),         # both bar charts
    ("ranked_list_countdown", "pictograph_scale"),          # both count/rank
    ("sankey_flow", "money_flow_empire"),                   # both money structure
    ("sankey_flow", "composition_stack"),                   # both parts-of-a-whole
    ("sankey_flow", "proportion_ring"),                     # both a split
    ("era_band_timeline", "chronology_timeline"),           # both timelines
    ("countdown_clock", "gold_number_callout"),             # both a single big figure
    ("connection_web", "org_hierarchy_tree"),               # both node diagrams
    ("connection_web", "money_flow_empire"),                # both node graphs
    ("quote_stream", "pull_quote_portrait"),                # both quotations
    ("quote_stream", "statement_card"),                     # both a held line
    ("map_region_highlight", "location_establish_card"),    # both single-place maps
    ("map_region_highlight", "map_heat_spread"),            # both maps
    ("map_region_highlight", "map_route_spread"),           # both maps
    ("map_region_highlight", "portrait_name_over_map"),     # both maps
    ("cause_effect_chain", "org_hierarchy_tree"),           # both node diagrams
    ("cause_effect_chain", "connection_web"),               # both diagrams
    ("spectrum_meter", "gold_number_callout"),              # both a single figure/level
    ("spectrum_meter", "proportion_ring"),                  # both a single gauge value
    ("spotlight_object_hold", "framed_evidence_spotlight"),  # both spotlight reveals
    ("spotlight_object_hold", "cinematic_portrait_hold"),    # both held single-subject reveals
    ("spotlight_object_hold", "statement_card"),             # both bold text on near-black
    ("spotlight_object_hold", "kinetic_keyword"),            # both a single typographic reveal
    ("flowchart_decision", "cause_effect_chain"),            # both directional-logic diagrams
    ("flowchart_decision", "org_hierarchy_tree"),            # both branching node trees
    ("flowchart_decision", "connection_web"),                # both node diagrams
    ("world_map_arc", "map_route_spread"),                   # both animated route maps
    ("world_map_arc", "location_establish_card"),            # both maps
    ("world_map_arc", "map_heat_spread"),                    # both maps
    ("world_map_arc", "map_region_highlight"),               # both maps
    ("world_map_arc", "portrait_name_over_map"),             # both maps
    # V2.6 — Batch 1B: investigation / spy / crime / evidence family
    ("witness_testimony_card", "suspect_profile_card"),      # both person-file cards
    ("witness_testimony_card", "pull_quote_portrait"),       # both quote-led portraits
    ("witness_testimony_card", "cinematic_portrait_hold"),   # both single-person holds
    ("suspect_profile_card", "cinematic_portrait_hold"),     # both person-file/portrait
    ("suspect_profile_card", "portrait_name_over_map"),      # both identity cards
    ("investigation_location_map", "location_establish_card"),  # both location maps
    ("investigation_location_map", "map_badge_node"),        # both pinned maps
    ("investigation_location_map", "map_route_spread"),      # both maps
    ("investigation_location_map", "world_map_arc"),         # both maps
    ("investigation_location_map", "map_region_highlight"),  # both maps
    ("sightline_trajectory", "annotated_detail_callout"),    # both pointer diagrams
    ("sightline_trajectory", "route_comparison"),            # both drawn-vector diagrams
    ("route_comparison", "comparison_split"),                # both A-vs-B contrasts
    ("route_comparison", "vs_balance_scale"),                # both weigh two options
    ("route_comparison", "map_route_spread"),                # both route diagrams
    ("evidence_connection_board", "connection_web"),         # both node webs
    ("evidence_connection_board", "org_hierarchy_tree"),     # both node graphs
    ("evidence_connection_board", "money_flow_empire"),      # both hub-and-spoke graphs
    ("classified_stamp_reveal", "redacted_document"),        # both censored file pages
    ("classified_stamp_reveal", "headline_document_reveal"),  # both document reveals
    ("classified_stamp_reveal", "framed_evidence_spotlight"),  # both document/evidence
    # V2.8 — Batch 1C
    ("system_planview_flow", "cause_effect_chain"),          # both directional flows
    ("system_planview_flow", "connection_web"),              # both node systems
    ("system_planview_flow", "org_hierarchy_tree"),          # both structure diagrams
    ("system_planview_flow", "flowchart_decision"),          # both drawn diagrams
    ("packet_path_trace", "system_planview_flow"),           # both systems traces
    ("packet_path_trace", "world_map_arc"),                  # both animated paths
    ("exploit_chain", "cause_effect_chain"),                 # both staged chains
    ("measurement_callout", "annotated_detail_callout"),     # both point/annotate
    ("silhouette_scale_compare", "comparison_split"),        # both A-vs-B compares
    ("silhouette_scale_compare", "pictograph_scale"),        # both scale visuals
    ("acquisition_timeline", "money_flow_empire"),           # both empire structures
    ("acquisition_timeline", "chronology_timeline"),         # both time rails
    ("supply_chain_network", "sankey_flow"),                 # both flow diagrams
    ("footage_fact_overlay", "gold_number_callout"),         # both single facts/nums
    ("footage_object_callout", "annotated_detail_callout"),  # both object callouts
    ("footage_route_trace", "map_route_spread"),             # both route traces
    ("footage_route_trace", "velocity_route_map"),           # both route traces
    # V3.1 — the cross-section is a labelled-diagram beat; keep it off other
    # labelled-callout / diagram / scale cards so two "leader-line" graphics
    # never sit back-to-back.
    ("labeled_cross_section", "annotated_detail_callout"),
    ("labeled_cross_section", "footage_object_callout"),
    ("labeled_cross_section", "measurement_callout"),
    ("labeled_cross_section", "system_planview_flow"),
    ("labeled_cross_section", "silhouette_scale_compare"),
}

# universal QA checks every primitive's output must pass (run by qa/)
QA_CHECKS = [
    "no_black_frames", "text_readable", "face_safe_zone", "no_alpha_break",
    "no_jerky_motion", "duration_in_range", "audio_sync_ok",
]


def _entry(m):
    sp = dict(m.SPEC)
    pid = sp["id"]
    return {
        "id": pid, "family": sp.get("family"), "spec": sp,
        "render": m.render, "module": m.__name__,
        "roles": set(sp.get("roles", [])),
        "niches_ok": set(sp.get("niches_ok", [])),
        "intensity_range": tuple(sp.get("intensity_range", [1, 5])),
        "duration_range": tuple(sp.get("duration_range", [2.0, 6.0])),
        "per_video_cap": int(sp.get("per_video_cap", 3)),
        "repeat_cooldown_s": float(sp.get("repeat_cooldown_s", 30)),
        "cost": sp.get("cost", "med"),
        "layout_variants": sp.get("layout_variants", []),
        "audio_cue": sp.get("audio_cue", ""),
        "fallback": sp.get("fallback", ""),
        "required_inputs": REQUIRED_INPUTS.get(pid, set()),
    }


REGISTRY = {m.SPEC["id"]: _entry(m) for m in _MODULES}


# ── query helpers (used by director.py) ─────────────────────────────────────
def get(pid): return REGISTRY.get(pid)


def all_ids(): return list(REGISTRY)


# ── niche-name normalization (2026-06-03) ───────────────────────────────────
# The primitive `niches_ok` vocabulary is {history, geopolitics, biography,
# business, crime, tech, all}. Scripts / the portal / NIM use a RICHER niche
# vocabulary (technology, science, spy, true_crime, agriculture_history, …).
# Without normalization `eligible(niche="technology")` matched NO primitive
# (the tag is "tech") → technology rendered 0 motion graphics, and "science"
# (no tag at all) was starved. This maps each incoming niche to the SET of
# primitive tags it should satisfy. The raw niche is always included, so a
# primitive tagged with the exact niche still matches and no working niche
# regresses. Science / tech are explainer-heavy → they earn the diagram +
# data families (tech + business).
_NICHE_ALIASES = {
    "technology": {"tech"}, "tech": {"tech"},
    "science": {"tech", "business"}, "education_explainer": {"tech", "business"},
    "explainer": {"tech", "business"}, "health_longevity": {"tech", "business"},
    "health": {"tech", "business"},
    "spy": {"geopolitics", "crime"}, "spy_intel": {"geopolitics", "crime"},
    "intelligence": {"geopolitics", "crime"}, "intel": {"geopolitics", "crime"},
    "true_crime": {"crime"}, "truecrime": {"crime"}, "true-crime": {"crime"},
    "agriculture_history": {"history", "business"},
    "agriculture": {"history", "business"},
    "character_documentary": {"biography"}, "biography": {"biography"},
    "finance": {"business"}, "economics": {"business"},
    "geopolitics": {"geopolitics"}, "history": {"history"},
    "business": {"business"}, "crime": {"crime"},
}


def niche_tags(niche) -> set:
    """Primitive niche tags an incoming niche string satisfies (raw + aliases)."""
    if not niche:
        return set()
    n = str(niche).strip().lower()
    return set(_NICHE_ALIASES.get(n, set())) | {n}


# ── off-theme primitive denylist (RC4, 2026-06-05) ──────────────────────────
# A few business/finance primitives (wealth/value/empire arcs) read as factual
# data cards. On war / geopolitics / history docs they fired on ungrounded
# numbers and looked like fabricated economic claims (e.g. a wealth_arc_counter
# on an Iran-Iraq War beat). `niches_ok` already constrains by SPEC, but this is
# the single niche chokepoint, so we add a conservative, DATA-DRIVEN family
# denylist here: when the normalized niche is one of these, the listed primitive
# ids are skipped outright. This removes nothing from REGISTRY — the primitives
# stay available to the niches they genuinely belong to (business / biography).
_NICHE_FORBIDDEN = {
    "geopolitics": {"wealth_arc_counter", "money_flow_empire", "growth_curve_chart"},
    "history": {"wealth_arc_counter", "money_flow_empire"},
    "war": {"wealth_arc_counter", "money_flow_empire", "growth_curve_chart"},
}


def eligible(*, role=None, niche=None, intensity=None, have_inputs=None):
    """Primitives matching role/niche/intensity whose required inputs are all
    satisfiable from `have_inputs` (a set of available input keys)."""
    have = set(have_inputs or ())
    _ntags = niche_tags(niche)
    # the normalized niche keys this query forbids (raw niche + alias tags), so
    # e.g. "true_crime"→"crime" and a raw "war"/"history"/"geopolitics" all map
    # to their denylist entry; conservative — most niches forbid nothing.
    _forbidden = set()
    for _k in ({str(niche).strip().lower()} | _ntags) if niche else set():
        _forbidden |= _NICHE_FORBIDDEN.get(_k, set())
    out = []
    for e in REGISTRY.values():
        if role and role not in e["roles"]:
            continue
        if e["id"] in _forbidden:                  # off-theme for this niche
            continue
        if _ntags and "all" not in e["niches_ok"] and not (_ntags & e["niches_ok"]):
            continue
        if intensity is not None:
            lo, hi = e["intensity_range"]
            if not (lo <= intensity <= hi):
                continue
        if e["required_inputs"] and not e["required_inputs"].issubset(have):
            continue
        out.append(e)
    return out


def incompatible(a: str, b: str) -> bool:
    return (a, b) in INCOMPATIBLE_ADJACENT or (b, a) in INCOMPATIBLE_ADJACENT


def summary() -> dict:
    return {pid: {"family": e["family"], "roles": sorted(e["roles"]),
                  "cap": e["per_video_cap"], "cooldown": e["repeat_cooldown_s"],
                  "needs": sorted(e["required_inputs"])}
            for pid, e in REGISTRY.items()}
