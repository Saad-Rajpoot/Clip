"""SFX director — motivated, RESTRAINED sound-design intelligence (engine-level).

The mature `sfx` engine + `assemble._gfx_sfx_seq` already emit a motivated SFX
sequence for every motion-graphics card (all 39 primitives are covered). This
director adds the spec's missing RESTRAINT + observability layer on top, without
replacing any working synthesis:

  • PER-PRIMITIVE RESTRAINT POLICY (sfx_policy.json, adversarially verified):
    default-silence cards stay silent so the line lands; every cue is capped at a
    max intensity; per-primitive cooldown + variant rotation prevent stamping.
  • UNIVERSAL COVERAGE: the policy is keyed by the 39 registry primitives, but SFX
    are generated from ~68 renderer card-kinds (`gk`). `policy_for()` resolves a
    primitive name OR a gk OR an SFX family to a policy — precise caps where names
    map, sensible FAMILY defaults everywhere else, so restraint is universal.
  • NICHE DENSITY: a per-niche restraint multiplier widens cooldowns + lowers caps
    for silence-led niches (mystery/spy) and relaxes them slightly for busier ones.
  • SFX CUE SHEET: emits `sfx_cue_sheet.json` (time, kind, family, intensity,
    variant) for QA + post-production review.

Pure logic; defensive. The only effect on a render is to *reduce* SFX intensity /
skip silence-default cues — it can never add noise or break a render.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Renderer card-kind (gk used in assemble._gfx_sfx_seq)  ->  registry primitive,
# for the cases where the verified per-primitive cap should govern the card.
_GK_TO_PRIMITIVE = {
    "statement": "statement_card", "redacted": "redacted_document",
    "conspiracy_board": "connection_web", "classified": "redacted_document",
    "case_file": "framed_evidence_spotlight", "evidence": "framed_evidence_spotlight",
    "evidence_tag": "framed_evidence_spotlight", "timeline": "chronology_timeline",
    "mini_timeline": "chronology_timeline", "era_banner": "era_band_timeline",
    "process_diagram": "process_diagram", "cause_effect": "cause_effect_chain",
    "network_graph": "connection_web", "relationship_tree": "org_hierarchy_tree",
    "family_tree": "org_hierarchy_tree", "comparison": "comparison_split",
    "stat_dashboard": "statistic_bar_reveal", "vertical_bar": "statistic_bar_reveal",
    "donut_chart": "proportion_ring", "progress_bar": "statistic_bar_reveal",
    "population_split": "composition_stack", "ratio": "vs_balance_scale",
    "heatmap": "map_heat_spread", "currency_stat": "money_flow_empire",
    "map_reveal": "map_region_highlight", "map_region": "map_region_highlight",
    "map_route": "map_route_spread", "name_reveal": "portrait_name_over_map",
    "portrait": "cinematic_portrait_hold", "mini_bio": "cinematic_portrait_hold",
    "title_card": "headline_montage", "headline_crawl": "headline_montage",
    "breaking_news": "headline_montage", "newspaper": "headline_document_reveal",
    "news_article": "headline_document_reveal", "press_release": "headline_document_reveal",
    "document": "headline_document_reveal", "quote_highlight": "quote_stream",
    "long_quote": "quote_stream", "pull_quote": "pull_quote_portrait",
    "define_the_term": "definition_card", "glossary": "definition_card",
    "countdown_clock": "countdown_clock", "spotlight": "spotlight_object_hold",
    "microscope": "spotlight_object_hold", "framed_insert": "framed_evidence_spotlight",
    "speedometer": "spectrum_meter", "surveillance": "spectrum_meter",
}

# Whoosh-family event kinds (the distracting/repetitive family that needs the most
# restraint) — mirrors assemble._WHOOSH_KINDS.
_WHOOSH_KINDS = {"reveal", "transition", "text_slam", "whoosh", "swoosh"}

# Family default policy when neither a primitive nor a gk match is found.
_FAMILY_DEFAULT = {
    "whoosh":       {"default_silence": False, "max_intensity": 0.6,  "cooldown_s": 22, "variant_count": 4},
    "impact":       {"default_silence": False, "max_intensity": 0.75, "cooldown_s": 28, "variant_count": 4},
    "foley_doc":    {"default_silence": False, "max_intensity": 0.6,  "cooldown_s": 18, "variant_count": 3},
    "foley_money":  {"default_silence": False, "max_intensity": 0.5,  "cooldown_s": 20, "variant_count": 3},
    "map":          {"default_silence": False, "max_intensity": 0.6,  "cooldown_s": 16, "variant_count": 3},
    "data":         {"default_silence": False, "max_intensity": 0.78, "cooldown_s": 14, "variant_count": 3},
    "ui":           {"default_silence": False, "max_intensity": 0.45, "cooldown_s": 12, "variant_count": 3},
    "surveillance": {"default_silence": False, "max_intensity": 0.5,  "cooldown_s": 18, "variant_count": 3},
    "keyboard":     {"default_silence": False, "max_intensity": 0.4,  "cooldown_s": 10, "variant_count": 3},
    "silence":      {"default_silence": True,  "max_intensity": 0.4,  "cooldown_s": 30, "variant_count": 2},
}

# per-niche restraint: (cooldown widen ×, max-intensity ×). Silence-led niches get
# wider cooldowns + lower caps; busier niches stay near 1.0.
_NICHE_RESTRAINT = {
    "mystery":     (1.30, 0.85),
    "spy":         (1.20, 0.90),
    "crime":       (1.12, 0.92),
    "history":     (1.10, 0.95),
    "geopolitics": (1.05, 0.97),
    "business":    (1.00, 1.00),
    "default":     (1.10, 0.95),
}

_cache: dict = {}


def _policy_table() -> dict:
    if "pol" not in _cache:
        try:
            d = json.loads((_HERE / "sfx_policy.json").read_text(encoding="utf-8"))
            _cache["pol"] = d.get("policy", {})
        except Exception:                                      # noqa: BLE001
            _cache["pol"] = {}
    return _cache["pol"]


def family_of(kind: str, gk: str = "") -> str:
    """Best-effort SFX family for an event kind (+optional card gk)."""
    k = (kind or "").lower()
    if k in _WHOOSH_KINDS:
        return "whoosh"
    if k in ("boom", "impact", "land", "text_slam", "stinger", "reveal_thud"):
        return "impact"
    if k in ("money_tick", "money"):
        return "foley_money"
    if k in ("doc_slide", "stamp", "page_flip", "doc_highlight", "evidence", "redaction"):
        return "foley_doc"
    if k in ("map_pin", "map_pulse", "map_route", "map_region", "location_lock", "radar_ping"):
        return "map"
    if k in ("keyboard", "kb_key", "terminal"):
        return "keyboard"
    if k in ("rec", "shutter", "surveillance", "blip", "beep", "target_lock"):
        return "surveillance"
    if k in ("ui", "tick", "tag_snap"):
        return "ui"
    if "tick" in k or "chart" in k or "stat" in k or "node" in k or "step" in k or "bar" in k:
        return "data"
    # fall back via the gk's mapped primitive family if available
    prim = _GK_TO_PRIMITIVE.get((gk or "").lower())
    if prim:
        p = _policy_table().get(prim)
        if p and p.get("family"):
            return p["family"]
    return "ui"


def policy_for(key: str, *, gk: str = "", family: str = "") -> dict:
    """Resolve a restraint policy for a primitive name, a renderer gk, or a family.
    Precise per-primitive cap where the name/gk maps; family default otherwise."""
    table = _policy_table()
    k = (key or "").lower()
    if k in table:
        return dict(table[k])
    prim = _GK_TO_PRIMITIVE.get((gk or k))
    if prim and prim in table:
        return dict(table[prim])
    fam = family or family_of(key, gk)
    return dict(_FAMILY_DEFAULT.get(fam, _FAMILY_DEFAULT["ui"]))


def cap_intensity(intensity: float, *, key: str = "", gk: str = "", kind: str = "",
                  niche: str = "") -> float:
    """Cap an SFX event's intensity by the (niche-adjusted) per-primitive max."""
    try:
        iv = float(intensity)
    except (TypeError, ValueError):
        return 0.0
    fam = family_of(kind, gk)
    pol = policy_for(key or gk or kind, gk=gk, family=fam)
    _, cap_mult = _NICHE_RESTRAINT.get((niche or "default").lower(),
                                       _NICHE_RESTRAINT["default"])
    cap = float(pol.get("max_intensity", 0.7)) * cap_mult
    return round(max(0.0, min(iv, cap)), 3)


def cooldown_s(*, key: str = "", gk: str = "", kind: str = "", niche: str = "") -> float:
    fam = family_of(kind, gk)
    pol = policy_for(key or gk or kind, gk=gk, family=fam)
    widen, _ = _NICHE_RESTRAINT.get((niche or "default").lower(),
                                    _NICHE_RESTRAINT["default"])
    return round(float(pol.get("cooldown_s", 16)) * widen, 1)


def should_silence(*, key: str = "", gk: str = "", kind: str = "", niche: str = "",
                   is_accent: bool = False) -> bool:
    """True if this card should stay SILENT (silence-default primitive). A rare
    deliberate `is_accent` beat is allowed through."""
    if is_accent:
        return False
    pol = policy_for(key or gk or kind, gk=gk)
    return bool(pol.get("default_silence", False))


# ── SFX cue sheet ─────────────────────────────────────────────────────────────
def build_cue_sheet(events: list, total: float, niche: str = "",
                    video_id: str = "") -> dict:
    """events: ordered list of dicts {time, kind, intensity, gk?, variant?}. Build
    `sfx_cue_sheet.json`: per-event family + the density/whoosh stats QA needs."""
    out = []
    fam_counts: dict = {}
    whoosh = 0
    for e in (events or []):
        kind = e.get("kind", "")
        gk = e.get("gk", "")
        fam = family_of(kind, gk)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        if fam == "whoosh":
            whoosh += 1
        out.append({
            "time_s": round(float(e.get("time", 0.0)), 2),
            "kind": kind, "gk": gk, "family": fam,
            "intensity": round(float(e.get("intensity", 0.0)), 3),
            "variant": e.get("variant", ""),
        })
    dur_min = max(total, 1.0) / 60.0
    families = sorted({o["family"] for o in out})
    return {
        "schema_version": 1,
        "kind": "sfx_cue_sheet",
        "video_id": video_id,
        "niche": niche,
        "total_s": round(float(total), 2),
        "total_events": len(out),
        "sfx_per_min": round(len(out) / dur_min, 2),
        "whoosh_per_min": round(whoosh / dur_min, 2),
        "family_counts": dict(sorted(fam_counts.items(), key=lambda kv: -kv[1])),
        "families_used": families,
        "events": out,
    }


def write_cue_sheet(sheet: dict, dest) -> None:
    try:
        Path(dest).write_text(json.dumps(sheet, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass


def families_used(events: list) -> list:
    return sorted({family_of(e.get("kind", ""), e.get("gk", "")) for e in (events or [])})
