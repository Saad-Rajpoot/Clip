"""Render-dispatch layer for the motion-graphics engine.

Bridges director Decisions → cached, manifest-logged premium clips that the
assembler can slot in as a scene's visual. Validates required inputs, picks
layout/palette, content-hash caches (never regenerates), gracefully falls back
when an input is missing, and records an auditable manifest entry per scene.

All six primitives expose a uniform `render(out_path, *, <inputs>, dur, fps)`,
so dispatch is a single code path. Pure-local. No paid API.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path

from . import registry

VERSION = "mg-0.2.1"          # bump to invalidate all cached motion graphics
#                              (v0.2.0 = per-primitive useful-duration windows;
#                               v0.2.1 = dark stat-card charcoal floor + foreground
#                               -only fade so a dissolve never lands on a black frame)

# V1.1 — the USEFUL on-screen duration per primitive: how long the graphic
# should OWN the scene before footage returns. The director/pipeline caps this
# at the scene length (short scenes still fill), renders the clip slightly
# longer so the closing fade parks just past the window (a clean BRIGHT cut back
# to footage), and the assembler shows footage for the rest of the scene.
USEFUL_DUR = {
    "kinetic_keyword":          3.0,   # short impact reveal only
    "gold_number_callout":      4.8,   # count-up + readable hold, then footage
    "cinematic_portrait_hold":  5.5,   # medium portrait hold
    "portrait_name_over_map":   5.5,   # medium map relationship hold
    "headline_document_reveal": 6.5,   # long enough to read the headline
    "money_flow_empire":        7.5,   # longer readable hold, never full-scene
    "chronology_timeline":      5.5,   # spine reveals nodes; era band shorter
    "pull_quote_portrait":      6.5,   # long enough to read the quote
    "comparison_split":         6.0,   # both sides reveal + read
    "framed_evidence_spotlight": 6.0,  # artifact reveal + caption read
    "statistic_bar_reveal":     6.0,   # bars grow + count-up + read
    "location_establish_card":  5.5,   # establishing push + place read
    "growth_curve_chart":       6.5,   # curve draws in + count-up + read
    "annotated_detail_callout": 6.0,   # ring + spotlight + label read
    "map_route_spread":         6.5,   # route traces + waypoints read
    "proportion_ring":          5.5,   # arc sweep + count-up + read
    # RC5.1 — bullet_list duration entry REMOVED with the primitive
    # (cheap four-box template cut from production / REGISTRY). useful_dur()
    # falls back to its 5.0s default for any legacy id, and dispatch() refuses
    # the unknown primitive outright (skip → footage), so nothing here resurrects
    # the four-box look.
    "org_hierarchy_tree":       6.0,   # root + connectors + children read
    "statement_card":           5.0,   # lines rise in + read the claim
    "pictograph_scale":         5.5,   # figures fill + count + read
    "composition_stack":        5.5,   # bar wipes + segment labels read
    "definition_card":          5.5,   # term + definition read
    "vs_balance_scale":         6.0,   # beam tips + settles + read
    "before_after_slider":      5.5,   # seam wipes + tags read
    "headline_montage":         6.0,   # clippings drop in + read latest
    "map_heat_spread":          6.5,   # blooms ignite + spread + read
    "redacted_document":        6.0,   # bars sweep + reveal line + stamp
    "ranked_list_countdown":    6.5,   # rows count down to #1 + read
    "sankey_flow":              6.0,   # ribbons flow + branch values read
    "era_band_timeline":        6.5,   # bands wipe in + era names + years
    "countdown_clock":          5.5,   # arc depletes + count falls + read
    "connection_web":           6.5,   # nodes pop + links draw + read
    "quote_stream":             6.5,   # quotes cascade in + read each
    "map_region_highlight":     6.0,   # region glows + boundary + name read
    "cause_effect_chain":       6.5,   # cards + arrows draw + outcome lands
    "spectrum_meter":           5.5,   # needle sweeps + band + readout read
    "spotlight_object_hold":    5.5,   # beam sweeps in + subject lifts + read
    "flowchart_decision":       6.5,   # question + branches draw + verdict lands
    "world_map_arc":            6.5,   # arc traces in + pins pulse + cities read
    # V2.4 — Batch 1A cinematic-map family
    "map_status_banner":        5.0,   # banner slides + year/event read, back to map
    "parchment_war_map":        6.5,   # fills settle + front draws + side labels read
    "supply_route_dashes":      6.0,   # routes draw + dots travel + pins read
    "territory_advance_arrows": 5.5,   # arrows extend + fill sweeps + label read
    "diplomatic_link":          5.5,   # connector draws + resolves + outcome read
    "logo_link_beam":           5.0,   # icons fly in + beam travels + connect + read
    "map_badge_node":           5.0,   # pin pulses + badge scales + name read
    "velocity_route_map":       6.5,   # route draws + decelerates + node labels read
    # V2.6 — Batch 1B investigation / spy / crime / evidence family
    "witness_testimony_card":   6.0,   # photo + name + quote lines reveal, then hold
    "suspect_profile_card":     6.0,   # tile + rows + classification stamp land
    "investigation_location_map": 6.5,  # pins drop + leaders draw + pop-outs read
    "sightline_trajectory":     6.0,   # rings + bearing + vector + target lock
    "route_comparison":         6.5,   # both routes draw + metric chips read
    "evidence_connection_board": 7.0,  # nodes pop + threads draw one-by-one
    "classified_stamp_reveal":  6.0,   # stamp impact + redaction wipe + declassify
    # V2.8 — Batch 1C systems / mechanism / business / hybrid
    "system_planview_flow":     6.5,   # plan builds + highlight sweeps regions + caption
    "packet_path_trace":        6.0,   # packet comet travels the path + hop pings
    "exploit_chain":            6.5,   # stages stagger in + breach pulse advances
    "measurement_callout":      4.5,   # span grows + dual-unit label reads (overlay)
    "silhouette_scale_compare": 5.5,   # silhouettes slide apart to true scale
    "acquisition_timeline":     6.5,   # targets fold in one-by-one + count-up tally
    "supply_chain_network":     6.5,   # flow lights stage-by-stage L→R
    "footage_fact_overlay":     4.0,   # screen-locked fact holds over footage (overlay)
    "footage_object_callout":   4.0,   # ring locks + leader + label (overlay)
    "footage_route_trace":      5.5,   # route draws over footage + travelling head (overlay)
    # V3.0 — biography / character family
    "portrait_legend_reveal":   5.0,   # portrait push-in + name settles + underline
    "relationship_roster":      5.5,   # one card → staggered row of N + role chips
    "wealth_arc_counter":       5.0,   # count-up + curve arc (rise/fall) + verdict colour
    "life_milestone_spine":     6.0,   # spine draws + dots light one-by-one + captions
    "verdict_duality_card":     5.5,   # poles balance + thesis types clause-by-clause
    "act_chapter_card":         4.0,   # kicker+title settle from scatter over footage
    "era_stamp_overlay":        3.2,   # year odometer + unit counter over footage (overlay)
    # V3.1 — science / engineering explainer
    "labeled_cross_section":    6.0,   # vessel settle + active-path draw + leader labels
}


def useful_dur(primitive: str) -> float:
    """Useful on-screen seconds for a primitive (before footage returns)."""
    return float(USEFUL_DUR.get(primitive, 5.0))


# place-like input keys across the map family (for map_mode inference)
_MAP_PLACE_KEYS = ("place", "region", "source", "dests", "stops", "a", "b",
                   "from_place", "to_place", "side_a", "side_b", "origin_region",
                   "target_region", "advance_corridors", "hotspots", "nodes",
                   "pins")          # investigation_location_map (real geo bed)

# non-map families that still render a real geographic bed → log map_mode too
_GEO_CARD_IDS = ("investigation_location_map",)


def _map_mode(entry, inputs):
    """Infer the geography mode + resolved/unresolved places for the manifest:
    real_current_boundaries · real_reference_map · approximate_region ·
    historical_boundary · synthetic_abstract_fallback. None for non-map cards.
    (historical_boundary is reserved for a future bundled dataset; changed-border
    scenes currently use real_reference_map — see HISTORICAL_MAP_POLICY.md.)"""
    _ent = entry or {}
    if _ent.get("family") != "maps" and _ent.get("id") not in _GEO_CARD_IDS:
        return None
    try:
        from .maps import geo
    except Exception:                                          # noqa: BLE001
        return None
    cand = []
    for k in _MAP_PLACE_KEYS:
        v = inputs.get(k)
        if isinstance(v, str) and v.strip():
            cand.append(v.strip())
        elif isinstance(v, (list, tuple)):
            for it in v:
                if isinstance(it, str) and it.strip():
                    cand.append(it.strip())
                elif isinstance(it, (list, tuple)) and it and isinstance(it[0], str):
                    cand.append(it[0].strip())
                elif isinstance(it, dict):                     # pins: {"place": ...}
                    _p = it.get("place") or it.get("name")
                    if isinstance(_p, str) and _p.strip():
                        cand.append(_p.strip())
    resolved = [c for c in dict.fromkeys(cand) if geo.resolve(c) or geo.country_entry(c)]
    unresolved = [c for c in dict.fromkeys(cand) if c not in resolved]
    if inputs.get("reference"):
        mode = "real_reference_map"
    elif resolved:
        mode = "real_current_boundaries"
    else:
        mode = "synthetic_abstract_fallback"
    return {"map_mode": mode, "resolved_places": resolved[:8],
            "unresolved_places": unresolved[:8],
            "geo_source": "natural_earth_110m" if resolved else "synthetic",
            "coordinate_confidence": "gazetteer" if resolved else "none",
            "boundary_confidence": ("reference_modern" if inputs.get("reference")
                                    else ("modern" if resolved else "none")),
            "historical_risk": bool(inputs.get("reference"))}


def _asset_sig(p) -> str:
    try:
        st = os.stat(p)
        return f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}"
    except Exception:                                          # noqa: BLE001
        return "noasset"


def cache_key(primitive: str, inputs: dict, dur: float, *, version=VERSION) -> str:
    norm = {}
    for k, v in inputs.items():
        if v is None:
            continue
        if k in ("portrait_path", "map_image", "bg_image"):
            norm[k] = _asset_sig(v)
        elif k == "branches":
            norm[k] = [list(x) for x in v]
        else:
            norm[k] = v
    blob = json.dumps({"p": primitive, "in": norm, "dur": round(float(dur), 2),
                       "v": version}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# RETIRED primitives: kept registered (so the registry count + adjacency stay
# intact) but never rendered — dispatch() skips them straight to footage. Used
# to pull a template from production without a risky full de-registration.
#   cause_effect_chain — the causal CARD->ARROW->CARD diagram looked template-y
#                        and repeated (its 2-box cousin `cause_effect` is also
#                        retired in footage.py); cut for perceived quality.
_RETIRED_PRIMITIVES = {"cause_effect_chain"}


def dispatch(decision, *, run_dir, duration: float, fps: int = 30,
             version: str = VERSION) -> dict:
    """Render (or cache-hit) the clip for one director Decision. Returns an
    auditable manifest entry dict (never raises)."""
    base = {"scene_index": getattr(decision, "scene_index", None),
            "primitive": getattr(decision, "primitive", None),
            "reason": getattr(decision, "reason", ""), "ok": False,
            "skipped": False, "fallback": False, "cache_hit": False,
            "render_s": 0.0, "path": None,
            # V3.4 — visual-variant evidence (additive; never affects rendering)
            "variant": dict(getattr(decision, "variant", {}) or {})}
    pid = base["primitive"]
    if not pid:
        base.update(skipped=True, ok=True, reason=base["reason"] or "footage")
        return base
    if pid in _RETIRED_PRIMITIVES:
        # Retired template → never render; scene keeps its footage.
        base.update(skipped=True, reason=f"retired primitive {pid} → footage")
        return base
    e = registry.get(pid)
    if e is None:
        # Primitive not in REGISTRY → refuse it gracefully. This covers any id
        # removed from production (RC5.1: `bullet_list`, the cheap
        # four-box template) that might still ride in on a legacy/cached script.
        # We never render it and never substitute another card: the scene simply
        # keeps its footage. `skipped` (not `ok`) keeps it out of `rendered`, so
        # no four-box clip can ever reach assembly. No KeyError, no crash.
        base.update(skipped=True, reason=f"unknown primitive {pid} → footage")
        return base
    inputs = dict(getattr(decision, "inputs", {}) or {})
    missing = [k for k in e["required_inputs"] if not inputs.get(k)]
    if missing:
        base.update(fallback=True, reason=f"missing inputs {missing} → "
                    f"fallback: {e['fallback']}")
        return base
    lo, hi = e["duration_range"]
    # In the current integration the motion-graphic occupies the WHOLE scene,
    # so the clip must be at least as long as the scene — otherwise the per-beat
    # slice runs past the clip end and freeze-holds a dim/faded frame for the
    # rest of the scene (a 3.2s keyword clip on a 9s scene → ~6s dark hold,
    # which read as a black span). Honour the requested scene duration (the
    # primitives stretch their hold to fill: reveal front-loads via ease-out,
    # fade stays at the tail). `lo` is still the floor; cap at a sane 30s.
    req = float(duration) if duration else (lo + hi) / 2
    dur = max(lo, min(30.0, req))
    layout = inputs.get("layout") or (e["layout_variants"][0] if e["layout_variants"] else "")
    palette = inputs.get("palette_name", "amber_gold")
    key = cache_key(pid, inputs, dur, version=version)
    mgdir = Path(run_dir) / "motion_graphics"
    mgdir.mkdir(parents=True, exist_ok=True)
    out = mgdir / f"mg_{pid}_{key}.mp4"
    base.update(key=key, layout=layout, palette=palette, duration=round(dur, 2),
                inputs={k: (_asset_sig(v) if k in ("portrait_path", "map_image", "bg_image") else v)
                        for k, v in inputs.items() if v is not None})
    _mm = _map_mode(e, inputs)            # geography honesty fields for map cards
    if _mm:
        base.update(_mm)
    if out.exists() and out.stat().st_size > 2000:
        base.update(ok=True, cache_hit=True, path=str(out))
        return base
    kw = {k: v for k, v in inputs.items() if v is not None}
    kw["dur"] = dur
    kw["fps"] = fps
    # Defensive: pass only the kwargs this primitive's render() actually declares,
    # so an extra asset key folded in for a sibling primitive (e.g. a place/name
    # hint that suits portrait_name_over_map but not cinematic_portrait_hold) can
    # never crash an otherwise-valid render. Primitives with **kwargs get everything.
    try:
        params = inspect.signature(e["render"]).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):                            # noqa: BLE001
        pass
    t0 = time.time()
    try:
        res = e["render"](str(out), **kw)
    except Exception as ex:                                    # noqa: BLE001
        base.update(reason=f"render error: {str(ex)[:120]}", fallback=True)
        return base
    base["render_s"] = round(time.time() - t0, 2)
    if res.get("ok") and out.exists() and out.stat().st_size > 2000:
        base.update(ok=True, path=str(out))
    else:
        base.update(reason=f"render failed: {res.get('err', '')[:120]}", fallback=True)
    return base


def dispatch_all(decisions, *, run_dir, durations, fps: int = 30,
                 version: str = VERSION, write_manifest: bool = True) -> dict:
    """Dispatch every decision; `durations` maps scene_index → seconds. Writes
    `motion_graphics_manifest.json` to run_dir. Returns {entries, summary}."""
    entries = []
    for d in decisions:
        si = getattr(d, "scene_index", None)
        dur = float((durations or {}).get(si, 0.0)) or 0.0
        entries.append(dispatch(d, run_dir=run_dir, duration=dur, fps=fps,
                                version=version))
    rendered = [e for e in entries if e["primitive"] and e["ok"] and not e["skipped"]]
    summary = {
        "version": version, "scenes": len(entries),
        "graphics_rendered": len(rendered),
        "cache_hits": sum(1 for e in rendered if e["cache_hit"]),
        "cache_misses": sum(1 for e in rendered if not e["cache_hit"]),
        "fallbacks": sum(1 for e in entries if e["fallback"]),
        "by_primitive": {},
        "total_render_s": round(sum(e["render_s"] for e in entries), 1),
    }
    for e in rendered:
        summary["by_primitive"][e["primitive"]] = summary["by_primitive"].get(e["primitive"], 0) + 1
    if write_manifest:
        try:
            (Path(run_dir) / "motion_graphics_manifest.json").write_text(
                json.dumps({"summary": summary, "scenes": entries}, indent=1),
                encoding="utf-8")
        except Exception:                                      # noqa: BLE001
            pass
    return {"entries": entries, "summary": summary}
